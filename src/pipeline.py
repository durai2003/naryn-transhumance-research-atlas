"""Stage 3 end-to-end orchestration for cleaning, spatial processing and integration."""
from __future__ import annotations

import argparse
import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.geospatial.raster import clip_raster_to_boundary
from src.geospatial.vector import clean_geodataframe, process_vector_file, geopackage_layers
from src.integration.locations import locations_from_verified_wikidata, integrate_location_context
from src.preprocessing.tabular import clean_nasa_power_document, aggregate_climate
from src.validation.quality import validate_table, validate_geodataframe, inspect_raster, validate_dataset_register

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_transform(log: list[dict[str, Any]], step: str, status: str, details: str, **extra: Any) -> None:
    log.append({"timestamp_utc": utc_now(), "step": step, "status": status, "details": details, **extra})


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _find_acquisition_record(root: Path, asset_name: str) -> dict[str, Any]:
    """Return a successful acquisition-log row when one exists; otherwise {}."""
    log_path = root / "reports" / "acquisition_run_log.jsonl"
    if not log_path.exists():
        return {}
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("asset") == asset_name and record.get("status") == "SUCCESS":
            records.append(record)
    return records[-1] if records else {}


def _validate_geopackage(path: Path) -> dict[str, Any]:
    """Validate a candidate .gpkg before asking GDAL/pyogrio to read it.

    A filename suffix is not proof that a file is a GeoPackage. A valid
    GeoPackage is a SQLite database with the GeoPackage application id and
    required metadata tables. We also ask pyogrio to enumerate its layers.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError("candidate path is not a regular file")
    size = path.stat().st_size
    if size < 100:
        raise ValueError(f"file is implausibly small for a GeoPackage ({size} bytes)")
    with path.open("rb") as stream:
        header = stream.read(16)
    if header != b"SQLite format 3\x00":
        preview = header.hex(" ") or "<empty>"
        raise ValueError(
            "file does not have a SQLite header; it may be an HTML error page, "
            f"wrong download, or truncated archive (first bytes: {preview})"
        )

    # GeoPackage application_id is ASCII 'GPKG' (0x47504B47).
    expected_application_id = 0x47504B47
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            integrity = str(connection.execute("PRAGMA quick_check(1)").fetchone()[0])
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"SQLite database validation failed: {exc}") from exc

    missing_tables = {"gpkg_contents", "gpkg_spatial_ref_sys"} - tables
    if application_id != expected_application_id:
        raise ValueError(
            f"SQLite application_id is {application_id}, expected GeoPackage id {expected_application_id}"
        )
    if integrity.casefold() != "ok":
        raise ValueError(f"SQLite quick_check failed: {integrity}")
    if missing_tables:
        raise ValueError(f"GeoPackage metadata tables missing: {sorted(missing_tables)}")

    try:
        import pyogrio
        layers = pyogrio.list_layers(path)
    except Exception as exc:
        raise ValueError(f"GDAL/pyogrio could not enumerate GeoPackage layers: {exc}") from exc
    if len(layers) == 0:
        raise ValueError("GeoPackage contains no readable vector layers")
    return {
        "path": str(path),
        "size_bytes": size,
        "application_id": application_id,
        "sqlite_integrity": integrity,
        "layers": [str(row[0]) for row in layers],
    }


def _discover_osm_geopackages(osm_dir: Path, extracted_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Find, safely extract and validate OSM GeoPackages; ignore invalid files.

    This prevents an unrelated/truncated file with a `.gpkg` suffix from
    crashing the entire Stage 3 run. Invalid candidates are reported as
    blocked assets, while other valid candidates can still be processed.
    """
    osm_dir = Path(osm_dir)
    extracted_dir = Path(extracted_dir)
    errors: list[dict[str, str]] = []
    candidates: list[Path] = []

    if osm_dir.exists():
        # Include direct and previously extracted files, but do not descend into
        # arbitrary unrelated folders beyond this dataset's own OSM directory.
        candidates.extend(sorted(osm_dir.glob("*.gpkg")))
        if extracted_dir.exists():
            candidates.extend(sorted(extracted_dir.glob("*.gpkg")))

        for archive in sorted(osm_dir.glob("*.gpkg.zip")):
            try:
                with zipfile.ZipFile(archive) as zf:
                    bad_member = zf.testzip()
                    if bad_member:
                        raise ValueError(f"archive CRC check failed for {bad_member}")
                    members = [
                        info for info in zf.infolist()
                        if info.filename.casefold().endswith(".gpkg") and not info.is_dir()
                    ]
                    if len(members) != 1:
                        raise ValueError(f"expected exactly one .gpkg member, found {len(members)}")
                    member = members[0]
                    destination = extracted_dir / Path(member.filename).name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, destination.open("wb") as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                    candidates.append(destination)
            except Exception as exc:
                errors.append({
                    "input": str(archive),
                    "reason": f"GeoPackage archive could not be extracted: {exc}",
                })

    valid: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            _validate_geopackage(resolved)
            valid.append(resolved)
        except Exception as exc:
            errors.append({
                "input": str(candidate),
                "reason": f"Rejected invalid GeoPackage candidate: {exc}",
            })

    # Prefer the explicitly extracted archive member over a stray direct file.
    valid.sort(key=lambda item: (0 if item.parent == extracted_dir.resolve() else 1, str(item)))
    return valid, errors


def run_stage3(root: Path) -> dict[str, Any]:
    root = root.resolve()
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    reports = root / "reports"
    metadata = root / "metadata"
    transform_log: list[dict[str, Any]] = []
    quality_checks: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    warnings: list[str] = []
    output_files: list[str] = []

    # 1. Validate the existing dataset inventory/register.
    register_path = metadata / "dataset_register.csv"
    if register_path.exists():
        register_quality = validate_dataset_register(register_path)
        quality_checks.append({"check": "dataset_register", **register_quality})
        append_transform(transform_log, "register_validation", register_quality["status"],
                         "Validated required inventory fields and unique dataset identifiers")
    else:
        register_quality = {"status": "FAIL", "issues": ["dataset_register.csv is missing"]}
        quality_checks.append({"check": "dataset_register", **register_quality})

    # 2. Derive named location references strictly from the already verified QID file.
    wikidata_path = raw / "wikidata" / "verified_entity_references.json"
    locations_path = processed / "locations.csv"
    if wikidata_path.exists():
        locations = locations_from_verified_wikidata(wikidata_path)
        _save_csv(locations, locations_path)
        output_files.append(str(locations_path.relative_to(root)))
        quality_checks.append({"check": "locations_table", **validate_table(
            locations, dataset_name="locations", required_columns=["location_id", "location_name", "latitude", "longitude", "external_identifier", "source_url"],
            primary_key="location_id", latitude_column="latitude", longitude_column="longitude")})
        append_transform(transform_log, "wikidata_location_standardisation", "SUCCESS",
                         "Normalised verified coordinate-bearing Wikidata entity-page records into location keys",
                         output_rows=len(locations), source_file=str(wikidata_path.relative_to(root)))
    else:
        locations = pd.DataFrame(columns=["location_id", "location_name", "latitude", "longitude"])
        blocked.append({"input": "verified_wikidata_entity_references.json", "reason": "not present"})
        append_transform(transform_log, "wikidata_location_standardisation", "BLOCKED", "Verified location reference file absent")

    # 3. Clean actual Naryn geometry only if Stage 2 has obtained it.
    boundary_raw = raw / "boundaries" / "naryn_adm1.geojson"
    boundary_processed = processed / "vectors" / "naryn_adm1_clean.geojson"
    boundary: gpd.GeoDataFrame | None = None
    if boundary_raw.exists():
        source_boundary = gpd.read_file(boundary_raw)
        boundary, boundary_report = clean_geodataframe(source_boundary, target_crs="EPSG:4326")
        if boundary.empty:
            raise ValueError("Downloaded Naryn boundary has no valid features after cleaning")
        boundary_processed.parent.mkdir(parents=True, exist_ok=True)
        boundary.to_file(boundary_processed, driver="GeoJSON")
        output_files.append(str(boundary_processed.relative_to(root)))
        quality_checks.append({"check": "naryn_boundary", **validate_geodataframe(boundary, dataset_name="Naryn ADM1 boundary")})
        append_transform(transform_log, "boundary_cleaning", "SUCCESS", "Validated source geometry, repaired invalid features if needed, transformed to EPSG:4326",
                         **boundary_report)
    else:
        blocked.append({"input": "data/raw/boundaries/naryn_adm1.geojson", "reason": "actual geoBoundaries feature geometry is not present; spatial clipping is not run"})
        append_transform(transform_log, "boundary_cleaning", "BLOCKED", "Actual Naryn boundary absent; no boundary was invented")

    # 4. Convert NASA POWER daily JSON to a tidy long table and summaries if present.
    nasa_file = raw / "nasa_power" / "naryn_city_2015_2025_daily.json"
    daily_path = processed / "climate" / "nasa_power_daily.csv"
    monthly_path = processed / "climate" / "nasa_power_monthly.csv"
    seasonal_path = processed / "climate" / "nasa_power_seasonal.csv"
    if nasa_file.exists():
        city = locations.loc[locations["external_identifier"].eq("Q311946")] if "external_identifier" in locations else pd.DataFrame()
        if city.empty:
            raise ValueError("NASA POWER cache exists but verified Naryn city location Q311946 is unavailable")
        location = city.iloc[0]
        document = json.loads(nasa_file.read_text(encoding="utf-8"))
        acquisition = _find_acquisition_record(root, "NASA POWER daily Naryn city")
        daily = clean_nasa_power_document(
            document,
            location_id=str(location["location_id"]), location_name=str(location["location_name"]),
            latitude=float(location["latitude"]), longitude=float(location["longitude"]),
            source_url=acquisition.get("request_url", NASA_URL),
            retrieved_at=acquisition.get("timestamp_utc"),
        )
        monthly, seasonal = aggregate_climate(daily)
        _save_csv(daily, daily_path); _save_csv(monthly, monthly_path); _save_csv(seasonal, seasonal_path)
        output_files.extend(str(p.relative_to(root)) for p in [daily_path, monthly_path, seasonal_path])
        quality_checks.append({"check": "nasa_power_daily", **validate_table(
            daily, dataset_name="NASA POWER daily", required_columns=["observation_id", "location_id", "observation_date", "latitude", "longitude", "is_synthetic", "provenance"],
            primary_key="observation_id", latitude_column="latitude", longitude_column="longitude")})
        append_transform(transform_log, "nasa_power_daily_cleaning", "SUCCESS", "Parsed only source-supplied dates; mapped -999 sentinels to null; retained source parameters and units",
                         input_file=str(nasa_file.relative_to(root)), output_rows=len(daily))
        append_transform(transform_log, "climate_aggregation", "SUCCESS", "Aggregated daily T2M mean and precipitation sum by calendar month and meteorological season",
                         monthly_rows=len(monthly), seasonal_rows=len(seasonal))
    else:
        daily = pd.DataFrame(columns=["observation_id", "location_id", "observation_date", "latitude", "longitude", "t2m_c", "precipitation_mm_day", "is_synthetic", "provenance", "synthetic_disclaimer", "source_url", "retrieved_at", "processed_at"])
        monthly = pd.DataFrame(columns=["location_id", "year", "month", "t2m_mean_c", "precipitation_sum_mm", "daily_records", "t2m_valid_days", "precipitation_valid_days"])
        seasonal = pd.DataFrame(columns=["location_id", "year", "season", "t2m_mean_c", "precipitation_sum_mm", "daily_records", "t2m_valid_days", "precipitation_valid_days"])
        _save_csv(daily, daily_path); _save_csv(monthly, monthly_path); _save_csv(seasonal, seasonal_path)
        output_files.extend(str(p.relative_to(root)) for p in [daily_path, monthly_path, seasonal_path])
        blocked.append({"input": str(nasa_file.relative_to(root)), "reason": "no cached NASA POWER response; daily observations and summaries remain empty"})
        append_transform(transform_log, "nasa_power_daily_cleaning", "BLOCKED", "No NASA POWER response was downloaded; wrote schema-only empty outputs")

    # 5. Boundary-clip real raster assets only when the real boundary exists.
    raster_outputs: dict[str, Path] = {}
    raster_inspections: list[dict[str, Any]] = []
    raster_specs = [
        (raw / "glw4", processed / "rasters" / "glw4", "FAO GLW4"),
        (raw / "worldcover", processed / "rasters" / "worldcover", "ESA WorldCover"),
    ]
    for input_dir, output_dir, label in raster_specs:
        rasters = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))) if input_dir.exists() else []
        if not rasters:
            blocked.append({"input": str(input_dir.relative_to(root)), "reason": f"No local {label} primary raster; no processed raster can be generated"})
        for raster_path in rasters:
            if boundary is None:
                blocked.append({"input": str(raster_path.relative_to(root)), "reason": "actual Naryn boundary absent; raster clip deferred"})
                continue
            out_path = output_dir / f"{raster_path.stem}_naryn_clip.tif"
            clip_report = clip_raster_to_boundary(raster_path, boundary_processed, out_path)
            output_files.append(str(out_path.relative_to(root)))
            raster_outputs[f"{label}:{raster_path.stem}"] = out_path
            raster_inspections.append(inspect_raster(out_path, dataset_name=f"{label}:{raster_path.stem}"))
            append_transform(transform_log, "raster_boundary_clip", "SUCCESS", "Masked/cropped to actual Naryn boundary without resampling",
                             **clip_report)

    # 6. Clip mapped OSM context layers if a real extract and real boundary exist.
    osm_files, osm_archive_errors = _discover_osm_geopackages(raw / "osm", raw / "osm" / "extracted")
    blocked.extend(osm_archive_errors)
    if not osm_files:
        blocked.append({
            "input": "data/raw/osm/*.gpkg or *.gpkg.zip",
            "reason": "No valid GeoPackage candidate passed SQLite/GeoPackage/GDAL validation; see preceding OSM candidate diagnostics",
        })
    for osm_path in osm_files:
        if boundary is None:
            blocked.append({"input": str(osm_path.relative_to(root)), "reason": "actual Naryn boundary absent; OSM clip deferred"})
            continue
        output_gpkg = processed / "vectors" / "osm_naryn_context.gpkg"
        output_gpkg.parent.mkdir(parents=True, exist_ok=True)
        try:
            layers = geopackage_layers(osm_path)
        except Exception as exc:
            blocked.append({"input": str(osm_path.relative_to(root)), "reason": f"Layer enumeration failed after validation: {exc}"})
            continue
        successful_layers = []
        for index, layer_name in enumerate(layers):
            source = gpd.read_file(osm_path, layer=layer_name)
            clean, report = clean_geodataframe(source, target_crs="EPSG:4326")
            clipped = gpd.clip(clean, boundary)
            if not clipped.empty:
                clipped.to_file(output_gpkg, layer=layer_name[:60], driver="GPKG", mode="w" if not successful_layers else "a")
                successful_layers.append(layer_name)
            # The cleaning report already contains output_features (the number
            # remaining after cleaning). Preserve it separately so the clipped
            # feature count can be recorded without passing a duplicate keyword.
            osm_report = dict(report)
            osm_report["cleaned_features"] = osm_report.pop("output_features", len(clean))
            osm_report["output_features"] = int(len(clipped))
            append_transform(
                transform_log,
                "osm_layer_clip",
                "SUCCESS",
                "Clipped mapped OSM context only; not labelled as verified transhumance routes",
                input_layer=layer_name,
                **osm_report,
            )
        if successful_layers:
            output_files.append(str(output_gpkg.relative_to(root)))

    # 7. Integrate by shared location_id, and sample available rasters at verified points.
    sheep = next((p for k, p in raster_outputs.items() if "sheep" in k.casefold()), None)
    goat = next((p for k, p in raster_outputs.items() if "goat" in k.casefold()), None)
    worldcover = next((p for k, p in raster_outputs.items() if "worldcover" in k.casefold()), None)
    context = integrate_location_context(locations, monthly, sheep_raster=sheep, goat_raster=goat, worldcover_raster=worldcover)
    context_path = processed / "location_context.csv"
    _save_csv(context, context_path)
    output_files.append(str(context_path.relative_to(root)))
    if not context.empty:
        quality_checks.append({"check": "location_context", **validate_table(
            context, dataset_name="integrated location context", required_columns=["location_id", "location_name", "latitude", "longitude", "is_synthetic", "provenance"],
            primary_key=None, latitude_column="latitude", longitude_column="longitude")})
    append_transform(transform_log, "location_integration", "SUCCESS", "Joined climate summaries by stable Wikidata location_id; sampled only rasters that physically exist",
                     locations=len(locations), monthly_climate_rows=len(monthly), context_rows=len(context),
                     sheep_raster_available=bool(sheep), goat_raster_available=bool(goat), worldcover_available=bool(worldcover))

    # 8. Data dictionary describes actual outputs, including intentionally empty schemas.
    dictionary_rows = [
        ("location_id", "string", "Stable local key derived from external identifier; e.g. wikidata:Q311946"),
        ("location_name", "string", "Display label from the verified entity page"),
        ("latitude", "float", "WGS84 latitude from source coordinate statement"),
        ("longitude", "float", "WGS84 longitude from source coordinate statement"),
        ("observation_id", "string", "Stable key NASA-POWER:{location_id}:{date}"),
        ("observation_date", "date", "Source daily date, normalised to ISO 8601 YYYY-MM-DD"),
        ("t2m_c", "float", "NASA POWER T2M; degrees Celsius; source missing codes become null"),
        ("precipitation_mm_day", "float", "NASA POWER PRECTOTCORR; millimetres/day; source missing codes become null"),
        ("sheep_density_heads_km2", "float", "FAO sheep density sampled at location; null if source raster absent"),
        ("goat_density_heads_km2", "float", "FAO goat density sampled at location; null if source raster absent"),
        ("worldcover_class_code", "integer", "ESA WorldCover source class sampled at location; null if unavailable"),
        ("is_synthetic", "boolean", "False for externally sourced records; no research synthetic records produced in Stage 3"),
        ("provenance", "string", "Source lineage label identifying the upstream provider/method"),
    ]
    dictionary = pd.DataFrame(dictionary_rows, columns=["field", "data_type", "definition"])
    dictionary_path = metadata / "data_dictionary_stage3.csv"
    _save_csv(dictionary, dictionary_path)
    output_files.append(str(dictionary_path.relative_to(root)))

    # 9. Write machine-readable quality and transformation artefacts.
    raw_asset_presence = {
        "naryn_boundary": boundary_raw.exists(),
        "nasa_power_daily_json": nasa_file.exists(),
        "fao_glw4_rasters": bool(list((raw / "glw4").glob("*.tif")) + list((raw / "glw4").glob("*.tiff"))) if (raw / "glw4").exists() else False,
        "worldcover_rasters": bool(list((raw / "worldcover").glob("*.tif")) + list((raw / "worldcover").glob("*.tiff"))) if (raw / "worldcover").exists() else False,
        "osm_geopackage": bool(osm_files),
        "osm_geopackage_archive": bool(list((raw / "osm").glob("*.gpkg.zip"))) if (raw / "osm").exists() else False,
    }
    quality_report = {
        "generated_utc": utc_now(), "stage": "Stage 3 — data engineering pipeline",
        "raw_primary_assets_present": raw_asset_presence,
        "register_validation": register_quality,
        "checks": quality_checks,
        "raster_inspections": raster_inspections,
        "blocked_steps": blocked,
        "warnings": warnings,
        "summary": {
            "locations_from_verified_entity_pages": int(len(locations)),
            "daily_climate_records": int(len(daily)),
            "monthly_climate_records": int(len(monthly)),
            "seasonal_climate_records": int(len(seasonal)),
            "processed_rasters": int(len(raster_inspections)),
            "all_table_checks_pass": all(c.get("status") == "PASS" for c in quality_checks),
            "spatial_processing_completed": boundary is not None,
            "source_data_fabricated": False,
        },
    }
    quality_path = reports / "stage3_data_quality_report.json"
    write_json(quality_path, quality_report)
    output_files.append(str(quality_path.relative_to(root)))

    transform_path = reports / "stage3_transformation_log.csv"
    _save_csv(pd.DataFrame(transform_log), transform_path)
    output_files.append(str(transform_path.relative_to(root)))
    manifest = {
        "generated_utc": utc_now(), "stage": 3, "project_root": str(root),
        "output_files": output_files, "raw_assets": raw_asset_presence,
        "policy": "No synthetic replacements, interpolation or imputation of unavailable public source data.",
        "notes": [
            "The verified Naryn city point is a named Wikidata entity coordinate, not a weather station.",
            (
                "Spatial processing used the downloaded geoBoundaries Naryn ADM1 geometry."
                if boundary is not None else
                "Spatial clipping is blocked because the actual geoBoundaries Naryn ADM1 geometry is absent."
            ),
            (
                f"Processed {len(daily)} source NASA POWER daily observations; monthly and seasonal summaries are derived from those rows."
                if len(daily) else
                "No NASA POWER daily observations were processed; empty climate CSVs, if present, are schema-only outputs."
            ),
            (
                f"Processed {len(raster_inspections)} clipped raster output(s)."
                if raster_inspections else
                "No clipped raster outputs were produced because source rasters and/or the authoritative boundary were unavailable."
            ),
            (
                "OSM context layers were processed from the available Geofabrik GeoPackage; mapped roads and paths are not verified transhumance routes."
                if any("osm_naryn_context.gpkg" in item for item in output_files) else
                "No clipped OSM context GeoPackage was produced."
            ),
        ],
    }
    manifest_path = reports / "stage3_processing_manifest.json"
    write_json(manifest_path, manifest)
    output_files.append(str(manifest_path.relative_to(root)))

    summary = quality_report["summary"]
    execution_report = [
        "# Stage 3 execution report",
        "",
        f"Generated (UTC): {quality_report['generated_utc']}",
        "",
        "## Run summary",
        "",
        f"- Verified Wikidata locations standardised: {summary['locations_from_verified_entity_pages']}",
        f"- Daily NASA POWER observations: {summary['daily_climate_records']}",
        f"- Monthly climate summaries: {summary['monthly_climate_records']}",
        f"- Seasonal climate summaries: {summary['seasonal_climate_records']}",
        f"- Processed rasters: {summary['processed_rasters']}",
        f"- Table/register checks passing: {summary['all_table_checks_pass']}",
        f"- Actual boundary available for spatial processing: {summary['spatial_processing_completed']}",
        "- Synthetic source replacements created: no",
        "",
        "## Blocked operations",
        "",
    ]
    execution_report.extend([f"- **{item['input']}** — {item['reason']}" for item in blocked] or ["- None"])
    execution_report.extend([
        "", "## Interpretation", "",
        (
            "The pipeline ran against the files present in the workspace. "
            f"It standardised {summary['locations_from_verified_entity_pages']} verified location reference(s), "
            f"processed {summary['daily_climate_records']} NASA POWER daily observation(s), "
            f"created {summary['monthly_climate_records']} monthly and {summary['seasonal_climate_records']} seasonal climate summary row(s), "
            f"and produced {summary['processed_rasters']} clipped raster output(s). "
            f"The authoritative Naryn boundary was {'available' if boundary is not None else 'not available'}. "
            f"The table checks passed: {summary['all_table_checks_pass']}. "
            "Consult the blocked_operations list and transformation log to distinguish completed processing from unavailable inputs. "
            "No synthetic public-data replacements were created."
        ),
        "", "For field-level details, see `stage3_data_quality_report.json`, `stage3_transformation_log.csv` and `stage3_processing_manifest.json`.", "",
    ])
    execution_report_path = reports / "stage3_execution_report.md"
    execution_report_path.parent.mkdir(parents=True, exist_ok=True)
    execution_report_path.write_text("\n".join(execution_report), encoding="utf-8")
    manifest["output_files"].append(str(execution_report_path.relative_to(root)))
    write_json(manifest_path, manifest)

    return quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = run_stage3(args.root)
    summary = report["summary"]
    print("Stage 3 pipeline finished.")
    print(json.dumps(summary, indent=2))
    print(f"Quality report: {args.root.resolve() / 'reports' / 'stage3_data_quality_report.json'}")
    return 0 if report["register_validation"].get("status") == "PASS" and summary["all_table_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
