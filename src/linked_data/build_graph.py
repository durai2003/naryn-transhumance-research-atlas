"""Build a provenance-aware RDF graph from the project's actual Stage 3 outputs.

No pasture parcels, livestock samples, migration routes, interviews, or media records
are fabricated. Optional research-demo inputs are ingested only from explicit files
and synthetic records must carry all three required provenance fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCAT, DCTERMS, FOAF, GEO, OWL, PROV, RDF, RDFS, SKOS, XSD

BASE = "https://example.org/transhumance-atlas/"
TA = Namespace(BASE + "ontology#")
TAV = Namespace(BASE + "vocab/")
WGS84 = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
SCHEMA = Namespace("https://schema.org/")

LICENSE_URIS = {
    "cc by 4.0": URIRef("https://creativecommons.org/licenses/by/4.0/"),
    "creative commons attribution 4.0": URIRef("https://creativecommons.org/licenses/by/4.0/"),
    "odbl": URIRef("https://opendatacommons.org/licenses/odbl/1-0/"),
    "open database license": URIRef("https://opendatacommons.org/licenses/odbl/1-0/"),
    "cc0": URIRef("https://creativecommons.org/publicdomain/zero/1.0/"),
}


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip()).strip("-").lower()
    return text or "unnamed"


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _present(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip() != ""


def _float(value: Any) -> float | None:
    if not _present(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None and number.is_integer() else None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _add_literal(graph: Graph, subject: URIRef, predicate: URIRef, value: Any,
                 datatype: URIRef | None = None, lang: str | None = None) -> None:
    if not _present(value):
        return
    graph.add((subject, predicate, Literal(value, datatype=datatype, lang=lang)))


def _license_uri(statement: str) -> URIRef | None:
    lower = (statement or "").casefold()
    for key, uri in LICENSE_URIS.items():
        if key in lower:
            return uri
    return None


def _asset_candidates(root: Path, dataset_id: str) -> list[Path]:
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    mapping = {
        "GB-ADM1-KGZ": [raw / "boundaries" / "naryn_adm1.geojson", processed / "vectors" / "naryn_adm1_clean.geojson"],
        "KDC-PASTURE-SNOW-DROUGHT": sorted((raw / "kdc").glob("*")) if (raw / "kdc").exists() else [],
        "FAO-GLW4-2020-DENSITY": (sorted((raw / "glw4").glob("*.tif")) + sorted((processed / "rasters" / "glw4").glob("*.tif"))),
        "ESA-WORLDCOVER-2021-V200": (sorted((raw / "worldcover").glob("*.tif")) + sorted((processed / "rasters" / "worldcover").glob("*.tif"))),
        "NASA-POWER-DAILY": [raw / "nasa_power" / "naryn_city_2015_2025_daily.json", processed / "climate" / "nasa_power_daily.csv"],
        "OSM-KGZ-GEOFABRIK": (sorted((raw / "osm").glob("*.gpkg")) + sorted((raw / "osm" / "extracted").glob("*.gpkg")) + sorted((processed / "vectors").glob("osm_naryn_context.gpkg"))),
        "WIKIDATA-NARYN-ENTITIES": [raw / "wikidata" / "verified_entity_references.json"],
    }
    candidates = mapping.get(dataset_id, [])
    found: list[Path] = []
    for path in candidates:
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            continue
        # A CSV containing headers only is a schema artifact, not an acquired dataset.
        if dataset_id == "NASA-POWER-DAILY" and path.suffix.lower() == ".csv":
            try:
                if pd.read_csv(path).empty:
                    continue
            except Exception:
                continue
        found.append(path)
    return found


def _add_dataset(graph: Graph, root: Path, row: dict[str, Any], report: dict[str, Any]) -> URIRef:
    dataset_id = str(row.get("dataset_id") or "unknown")
    uri = URIRef(BASE + "dataset/" + _slug(dataset_id))
    graph.add((uri, RDF.type, TA.Dataset))
    graph.add((uri, RDF.type, DCAT.Dataset))
    graph.add((uri, DCTERMS.identifier, Literal(dataset_id)))
    _add_literal(graph, uri, DCTERMS.title, row.get("dataset_title"))
    _add_literal(graph, uri, DCTERMS.description, row.get("limitations_and_notes"))
    _add_literal(graph, uri, DCTERMS.publisher, row.get("provider"))
    _add_literal(graph, uri, DCTERMS.format, row.get("expected_format"))
    _add_literal(graph, uri, TA.licenceStatement, row.get("licence"))
    _add_literal(graph, uri, DCTERMS.coverage, row.get("temporal_spatial_coverage"))
    _add_literal(graph, uri, TA.coverageStatement, row.get("temporal_spatial_coverage"))
    _add_literal(graph, uri, TA.crsResolutionStatement, row.get("crs_resolution"))
    _add_literal(graph, uri, TA.schemaStatus, row.get("schema_or_profile_status"))
    _add_literal(graph, uri, TA.sourceRegisterStatus, row.get("acquisition_status"))
    _add_literal(graph, uri, DCTERMS.rights, row.get("attribution"))
    source_url = row.get("source_url")
    if _present(source_url):
        try:
            source_uri = URIRef(str(source_url).strip())
            graph.add((uri, TA.sourceURL, Literal(str(source_url).strip(), datatype=XSD.anyURI)))
            graph.add((uri, DCAT.landingPage, source_uri))
        except Exception:
            _add_literal(graph, uri, TA.sourceURL, source_url)
    licence = _license_uri(str(row.get("licence") or ""))
    if licence:
        graph.add((uri, DCTERMS.license, licence))
    available_paths = _asset_candidates(root, dataset_id)
    available = bool(available_paths)
    # Availability is assigned after checking whether the daily CSV contains records.
    if available:
        for index, path in enumerate(available_paths, start=1):
            graph.add((uri, TA.localAssetPath, Literal(path.relative_to(root).as_posix())))
            distribution = URIRef(str(uri) + f"/local-distribution-{index}")
            graph.add((uri, DCAT.distribution, distribution))
            graph.add((distribution, RDF.type, DCAT.Distribution))
            graph.add((distribution, DCTERMS.format, Literal(path.suffix.lstrip(".").upper() or "unknown")))
            graph.add((distribution, DCAT.byteSize, Literal(path.stat().st_size, datatype=XSD.nonNegativeInteger)))
    schema_only = False
    if dataset_id == "NASA-POWER-DAILY":
        empty_daily = root / "data" / "processed" / "climate" / "nasa_power_daily.csv"
        if empty_daily.exists() and empty_daily.stat().st_size > 0:
            try:
                schema_only = pd.read_csv(empty_daily).empty and not available
            except Exception:
                schema_only = False
    if dataset_id == "KDC-PASTURE-SNOW-DROUGHT" and available:
        only_capabilities = all(path.suffix.lower() in {".xml", ".txt"} for path in available_paths)
        status = "capabilities_only" if only_capabilities else "available"
    else:
        status = "available" if available else ("schema_only_no_records" if schema_only else "not_available_in_this_workspace")
    graph.add((uri, TA.localAvailability, Literal(status)))
    report[dataset_id] = {
        "uri": str(uri),
        "local_availability": status,
        "local_assets": [p.relative_to(root).as_posix() for p in available_paths],
        "schema_only_artifact": ("data/processed/climate/nasa_power_daily.csv" if schema_only else None),
    }
    return uri


def _add_geometry(graph: Graph, feature_uri: URIRef, geometry: Any, crs: Any = "EPSG:4326") -> URIRef:
    """Attach a valid geometry with explicit CRS84 WKT; reproject if needed."""
    try:
        from shapely import make_valid
        geom = geometry if geometry.is_valid else make_valid(geometry)
        if crs is None:
            raise ValueError("Geometry CRS is missing; refusing to assume WGS84")
        if str(crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            import geopandas as gpd
            geom_series = gpd.GeoSeries([geom], crs=crs).to_crs("EPSG:4326")
            geom = geom_series.iloc[0]
        geometry_uri = URIRef(str(feature_uri) + "/geometry")
        wkt = f'<http://www.opengis.net/def/crs/OGC/1.3/CRS84> {geom.wkt}'
        graph.add((feature_uri, GEO.hasGeometry, geometry_uri))
        graph.add((feature_uri, TA.hasGeometry, geometry_uri))
        graph.add((geometry_uri, RDF.type, GEO.Geometry))
        graph.add((geometry_uri, GEO.asWKT, Literal(wkt, datatype=GEO.wktLiteral)))
        return geometry_uri
    except Exception:
        raise


def _verified_qids(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "data" / "raw" / "wikidata" / "verified_entity_references.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {str(item.get("qid")): item for item in doc.get("entities", []) if item.get("qid") and item.get("uri", "").startswith("http://www.wikidata.org/entity/")}


def _add_locations(graph: Graph, root: Path, qids: dict[str, dict[str, Any]], report: dict[str, Any]) -> dict[str, URIRef]:
    location_uris: dict[str, URIRef] = {}
    location_csv = _read_csv(root / "data" / "processed" / "locations.csv")
    for _, row in location_csv.iterrows():
        location_id = str(row.get("location_id", ""))
        qid = str(row.get("external_identifier", ""))
        if not location_id or not _present(row.get("location_name")):
            continue
        uri = URIRef(BASE + "location/" + _slug(qid if qid in qids else location_id))
        location_uris[location_id] = uri
        graph.add((uri, RDF.type, TA.Location))
        graph.add((uri, RDF.type, GEO.Feature))
        graph.add((uri, RDFS.label, Literal(str(row["location_name"]), lang="en")))
        _add_literal(graph, uri, DCTERMS.identifier, qid if qid else location_id)
        if qid in qids:
            graph.add((uri, OWL.sameAs, URIRef(qids[qid]["uri"])))
            graph.add((uri, PROV.wasDerivedFrom, URIRef(qids[qid]["page"])))
        lat, lon = _float(row.get("latitude")), _float(row.get("longitude"))
        if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
            graph.add((uri, WGS84.lat, Literal(lat, datatype=XSD.decimal)))
            graph.add((uri, WGS84.long, Literal(lon, datatype=XSD.decimal)))
            geometry_uri = URIRef(str(uri) + "/point-geometry")
            graph.add((uri, GEO.hasGeometry, geometry_uri))
            graph.add((geometry_uri, RDF.type, GEO.Geometry))
            graph.add((geometry_uri, GEO.asWKT, Literal(
                f'<http://www.opengis.net/def/crs/OGC/1.3/CRS84> POINT({lon} {lat})',
                datatype=GEO.wktLiteral)))
        _add_literal(graph, uri, TA.provenanceLabel, row.get("provenance"))
        graph.add((uri, TA.isSynthetic, Literal(_truth(row.get("is_synthetic", False)), datatype=XSD.boolean)))
        _add_literal(graph, uri, DCTERMS.source, row.get("source_url"))
        report["locations"] += 1

    # The Region QID is verified on its entity page, but its coordinate statement
    # was not converted into a decimal point. The geometry is attached only if the
    # actual cleaned geoBoundaries feature is present.
    region_ref = qids.get("Q486375")
    region_uri = URIRef(BASE + "location/Q486375")
    if region_ref:
        graph.add((region_uri, RDF.type, TA.Location))
        graph.add((region_uri, RDF.type, GEO.Feature))
        graph.add((region_uri, RDFS.label, Literal(region_ref.get("label", "Naryn Region"), lang="en")))
        graph.add((region_uri, DCTERMS.identifier, Literal("Q486375")))
        graph.add((region_uri, OWL.sameAs, URIRef(region_ref["uri"])))
        graph.add((region_uri, PROV.wasDerivedFrom, URIRef(region_ref["page"])))
        location_uris.setdefault("naryn-region", region_uri)

    boundary_paths = [
        root / "data" / "processed" / "vectors" / "naryn_adm1_clean.geojson",
        root / "data" / "raw" / "boundaries" / "naryn_adm1.geojson",
    ]
    boundary_path = next((p for p in boundary_paths if p.exists()), None)
    if boundary_path:
        try:
            import geopandas as gpd
            boundary = gpd.read_file(boundary_path)
            if boundary.crs is None:
                raise ValueError("Naryn boundary has no declared CRS")
            name_col = next((c for c in ("shapeName", "shape_name", "name", "NAME_1") if c in boundary.columns), None)
            if name_col:
                selected = boundary[boundary[name_col].astype(str).str.casefold().isin({"naryn", "naryn region", "naryn oblast"})]
                if not selected.empty:
                    boundary = selected
            geometry = boundary.geometry.unary_union
            if geometry is not None and not geometry.is_empty:
                _add_geometry(graph, region_uri, geometry, boundary.crs)
                _add_literal(graph, region_uri, DCTERMS.source, "geoBoundaries ADM1 Naryn feature")
                graph.add((region_uri, RDF.type, TA.Location))
                graph.add((region_uri, RDF.type, GEO.Feature))
                if not list(graph.objects(region_uri, RDFS.label)):
                    graph.add((region_uri, RDFS.label, Literal("Naryn Region", lang="en")))
                location_uris.setdefault("naryn-region", region_uri)
                report["boundary_geometry_added"] = True
        except Exception as exc:
            report["warnings"].append(f"Boundary geometry was not added to RDF: {exc}")
    return location_uris


def _add_climate_csv(graph: Graph, root: Path, filename: str, dataset_uri: URIRef,
                     location_uris: dict[str, URIRef], report: dict[str, Any]) -> int:
    path = root / "data" / "processed" / "climate" / filename
    frame = _read_csv(path)
    if frame.empty:
        return 0
    count = 0
    for index, row in frame.iterrows():
        loc_id = str(row.get("location_id", ""))
        location_uri = location_uris.get(loc_id)
        if location_uri is None:
            report["warnings"].append(f"Skipped climate row {index} in {filename}: unresolved location_id {loc_id!r}")
            continue
        obs_id = row.get("observation_id")
        if not _present(obs_id):
            if filename.endswith("daily.csv"):
                obs_id = f"daily-{loc_id}-{row.get('observation_date', index)}"
            elif filename.endswith("monthly.csv"):
                obs_id = f"monthly-{loc_id}-{row.get('year')}-{row.get('month')}"
            else:
                obs_id = f"seasonal-{loc_id}-{row.get('year')}-{row.get('season')}"
        obs_uri = URIRef(BASE + "observation/climate/" + _slug(obs_id))
        graph.add((obs_uri, RDF.type, TA.ClimateObservation))
        graph.add((obs_uri, RDFS.label, Literal(str(obs_id))))
        graph.add((obs_uri, DCTERMS.identifier, Literal(str(obs_id))))
        graph.add((obs_uri, TA.atLocation, location_uri))
        graph.add((location_uri, TA.hasObservation, obs_uri))
        graph.add((obs_uri, PROV.wasDerivedFrom, dataset_uri))
        if filename.endswith("daily.csv"):
            date_value = row.get("observation_date")
            if _present(date_value):
                graph.add((obs_uri, TA.observedAt, Literal(str(date_value), datatype=XSD.date)))
            graph.add((obs_uri, TA.temporalResolution, Literal("daily")))
            _add_literal(graph, obs_uri, TA.temperatureC, _float(row.get("t2m_c")), XSD.decimal)
            _add_literal(graph, obs_uri, TA.precipitationMmDay, _float(row.get("precipitation_mm_day")), XSD.decimal)
        elif filename.endswith("monthly.csv"):
            year, month = _integer(row.get("year")), _integer(row.get("month"))
            if year and month and 1 <= month <= 12:
                graph.add((obs_uri, TA.observationPeriod, Literal(f"{year:04d}-{month:02d}")))
                graph.add((obs_uri, TA.year, Literal(year, datatype=XSD.integer)))
                graph.add((obs_uri, TA.month, Literal(month, datatype=XSD.integer)))
            graph.add((obs_uri, TA.temporalResolution, Literal("monthly")))
            _add_literal(graph, obs_uri, TA.temperatureMeanC, _float(row.get("t2m_mean_c")), XSD.decimal)
            _add_literal(graph, obs_uri, TA.precipitationSumMm, _float(row.get("precipitation_sum_mm")), XSD.decimal)
        else:
            year = _integer(row.get("year"))
            season = str(row.get("season", "")).strip().casefold()
            if year and season:
                graph.add((obs_uri, TA.observationPeriod, Literal(f"{year:04d}-{season}")))
                graph.add((obs_uri, TA.year, Literal(year, datatype=XSD.integer)))
            graph.add((obs_uri, TA.temporalResolution, Literal("seasonal")))
            graph.add((obs_uri, TA.season, Literal(season)))
            _add_literal(graph, obs_uri, TA.temperatureMeanC, _float(row.get("t2m_mean_c")), XSD.decimal)
            _add_literal(graph, obs_uri, TA.precipitationSumMm, _float(row.get("precipitation_sum_mm")), XSD.decimal)
        _add_literal(graph, obs_uri, TA.provenanceLabel, row.get("provenance") or "nasa_power_api")
        _add_literal(graph, obs_uri, TA.retrievedAt, row.get("retrieved_at"), XSD.dateTime)
        _add_literal(graph, obs_uri, PROV.generatedAtTime, row.get("processed_at"), XSD.dateTime)
        synthetic = _truth(row.get("is_synthetic", False))
        graph.add((obs_uri, TA.isSynthetic, Literal(synthetic, datatype=XSD.boolean)))
        if synthetic:
            _add_literal(graph, obs_uri, TA.syntheticDisclaimer, row.get("synthetic_disclaimer"))
        count += 1
    report["climate_observations_by_file"][filename] = count
    return count


def _add_livestock_from_context(graph: Graph, root: Path, location_uris: dict[str, URIRef],
                                dataset_uri: URIRef, report: dict[str, Any]) -> int:
    frame = _read_csv(root / "data" / "processed" / "location_context.csv")
    count = 0
    if frame.empty:
        return count
    if "location_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["location_id"], keep="first")
    for _, row in frame.iterrows():
        loc_id = str(row.get("location_id", ""))
        location_uri = location_uris.get(loc_id)
        if location_uri is None:
            continue
        row_synthetic = _truth(row.get("is_synthetic", False))
        row_provenance = str(row.get("provenance", ""))
        row_disclaimer = str(row.get("synthetic_disclaimer", "")).strip()
        if row_synthetic and (row_provenance != "synthetic_demo" or not row_disclaimer):
            report["rejected_optional_synthetic_rows"].append({"file": "location_context.csv", "id": loc_id, "reason": "synthetic raster sample requires synthetic_demo provenance and disclaimer"})
            continue
        for species, column in (("sheep", "sheep_density_heads_km2"), ("goat", "goat_density_heads_km2")):
            value = _float(row.get(column))
            if value is None:
                continue
            uri = URIRef(BASE + f"observation/livestock/{_slug(loc_id)}-{species}-glw4-2020")
            graph.add((uri, RDF.type, TA.LivestockObservation))
            graph.add((uri, RDFS.label, Literal(f"{species.title()} density at {row.get('location_name', loc_id)}")))
            graph.add((uri, TA.atLocation, location_uri))
            graph.add((location_uri, TA.hasObservation, uri))
            graph.add((uri, TA.livestockSpecies, Literal(species)))
            graph.add((uri, TA.observationValue, Literal(value, datatype=XSD.decimal)))
            graph.add((uri, TA.unit, Literal("heads/km2")))
            graph.add((uri, PROV.wasDerivedFrom, dataset_uri))
            graph.add((uri, TA.isSynthetic, Literal(row_synthetic, datatype=XSD.boolean)))
            graph.add((uri, TA.provenanceLabel, Literal("synthetic_demo" if row_synthetic else "FAO_GLW4_2020_sampled_raster")))
            if row_synthetic:
                graph.add((uri, TA.syntheticDisclaimer, Literal(row_disclaimer)))
            count += 1

        # ESA WorldCover is land-cover evidence, not an official pasture parcel.
        class_code = _integer(row.get("worldcover_class_code"))
        if class_code is not None:
            uri = URIRef(BASE + f"observation/land-cover/{_slug(loc_id)}-worldcover-2021")
            graph.add((uri, RDF.type, TA.LandCoverObservation))
            graph.add((uri, RDFS.label, Literal(f"WorldCover class sample at {row.get('location_name', loc_id)}")))
            graph.add((uri, TA.atLocation, location_uri))
            graph.add((location_uri, TA.hasObservation, uri))
            graph.add((uri, TA.worldCoverClassCode, Literal(class_code, datatype=XSD.integer)))
            graph.add((uri, PROV.wasDerivedFrom, URIRef(BASE + "dataset/esa-worldcover-2021-v200")))
            graph.add((uri, TA.isSynthetic, Literal(row_synthetic, datatype=XSD.boolean)))
            graph.add((uri, TA.provenanceLabel, Literal("synthetic_demo" if row_synthetic else "ESA_WorldCover_2021_sampled_raster")))
            if row_synthetic:
                graph.add((uri, TA.syntheticDisclaimer, Literal(row_disclaimer)))
            report.setdefault("landcover_observations", 0)
            report["landcover_observations"] += 1
    report["livestock_observations"] = count
    return count


def _add_dataset_distribution_metadata(graph: Graph, root: Path, report: dict[str, Any]) -> None:
    """Attach local derivative metadata without interpreting pixels as pasture parcels."""
    processed = root / "data" / "processed"
    derived = [
        ("NASA-POWER-DAILY-CLEAN", "NASA POWER cleaned daily climate table", processed / "climate" / "nasa_power_daily.csv", "text/csv", "NASA POWER"),
        ("NASA-POWER-MONTHLY", "NASA POWER monthly climate summary", processed / "climate" / "nasa_power_monthly.csv", "text/csv", "NASA POWER"),
        ("NASA-POWER-SEASONAL", "NASA POWER seasonal climate summary", processed / "climate" / "nasa_power_seasonal.csv", "text/csv", "NASA POWER"),
        ("NARYN-ADM1-CLEAN", "Cleaned Naryn ADM1 boundary", processed / "vectors" / "naryn_adm1_clean.geojson", "application/geo+json", "geoBoundaries"),
        ("OSM-NARYN-CONTEXT", "OpenStreetMap context clipped to Naryn", processed / "vectors" / "osm_naryn_context.gpkg", "application/geopackage+sqlite3", "OpenStreetMap via Geofabrik"),
        ("LOCATIONS-WIKIDATA", "Verified location reference table", processed / "locations.csv", "text/csv", "Wikidata"),
    ]
    for dataset_id, title, path, media_type, publisher in derived:
        if not path.exists() or path.stat().st_size == 0:
            continue
        csv_rows = None
        if path.suffix.lower() == ".csv":
            try:
                csv_rows = len(pd.read_csv(path))
            except Exception:
                csv_rows = 0
        uri = URIRef(BASE + "dataset/" + _slug(dataset_id))
        graph.add((uri, RDF.type, TA.Dataset)); graph.add((uri, RDF.type, DCAT.Dataset))
        graph.add((uri, DCTERMS.identifier, Literal(dataset_id)))
        graph.add((uri, DCTERMS.title, Literal(title, lang="en")))
        graph.add((uri, DCTERMS.publisher, Literal(publisher)))
        availability = "available" if csv_rows is None or csv_rows > 0 else "schema_only_no_records"
        graph.add((uri, TA.localAvailability, Literal(availability)))
        graph.add((uri, TA.localAssetPath, Literal(path.relative_to(root).as_posix())))
        if csv_rows is not None:
            graph.add((uri, TA.featureCount, Literal(csv_rows, datatype=XSD.integer)))
        dist = URIRef(str(uri) + "/distribution")
        graph.add((uri, DCAT.distribution, dist))
        graph.add((dist, RDF.type, DCAT.Distribution))
        graph.add((dist, DCAT.mediaType, URIRef("http://www.iana.org/assignments/media-types/" + media_type)))
        graph.add((dist, DCTERMS.format, Literal(media_type)))
        graph.add((dist, DCAT.byteSize, Literal(path.stat().st_size, datatype=XSD.nonNegativeInteger)))
        source_map = {
            "NASA-POWER-DAILY-CLEAN": "NASA-POWER-DAILY",
            "NASA-POWER-MONTHLY": "NASA-POWER-DAILY",
            "NASA-POWER-SEASONAL": "NASA-POWER-DAILY",
            "NARYN-ADM1-CLEAN": "GB-ADM1-KGZ",
            "OSM-NARYN-CONTEXT": "OSM-KGZ-GEOFABRIK",
            "LOCATIONS-WIKIDATA": "WIKIDATA-NARYN-ENTITIES",
        }
        source_id = source_map.get(dataset_id)
        if source_id:
            graph.add((uri, PROV.wasDerivedFrom, URIRef(BASE + "dataset/" + _slug(source_id))))
        report["derived_datasets_added"].append(dataset_id)

    raster_dirs = [processed / "rasters" / "worldcover", processed / "rasters" / "glw4"]
    for folder in raster_dirs:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.tif")):
            dataset_id = "DERIVED-RASTER-" + path.stem
            uri = URIRef(BASE + "dataset/" + _slug(dataset_id))
            graph.add((uri, RDF.type, TA.Dataset)); graph.add((uri, RDF.type, DCAT.Dataset))
            graph.add((uri, DCTERMS.identifier, Literal(dataset_id)))
            graph.add((uri, DCTERMS.title, Literal(path.stem.replace("_", " "), lang="en")))
            graph.add((uri, DCTERMS.publisher, Literal("Derived geospatial processing output")))
            graph.add((uri, DCTERMS.format, Literal("GeoTIFF")))
            graph.add((uri, TA.localAvailability, Literal("available")))
            graph.add((uri, TA.localAssetPath, Literal(path.relative_to(root).as_posix())))
            source_rdf = URIRef(BASE + ("dataset/esa-worldcover-2021-v200" if "worldcover" in folder.name.casefold() else "dataset/fao-glw4-2020-density"))
            graph.add((uri, PROV.wasDerivedFrom, source_rdf))
            try:
                import rasterio
                with rasterio.open(path) as raster:
                    graph.add((uri, TA.rasterResolution, Literal(str(raster.res))))
                    _add_literal(graph, uri, TA.coordinateReferenceSystem, str(raster.crs))
            except Exception as exc:
                report["warnings"].append(f"Raster metadata inspection failed for {path.name}: {exc}")
            report["derived_datasets_added"].append(dataset_id)


def _add_optional_spatial_research_records(graph: Graph, root: Path, location_uris: dict[str, URIRef],
                                            report: dict[str, Any]) -> None:
    """Ingest explicit route/pasture GeoJSON records; never infer them from OSM/WorldCover."""
    import geopandas as gpd
    specs = [
        (root / "data" / "processed" / "pastures.geojson", TA.Pasture, "pasture_id", False),
    ]
    # Only fall back to legacy Stage 4 routes.geojson if modern migration_routes.geojson is absent
    if not (root / "data" / "synthetic" / "migration_routes.geojson").exists():
        specs.insert(0, (root / "data" / "synthetic" / "routes.geojson", TA.Route, "route_id", True))
    for path, rdf_type, id_col, require_synthetic in specs:
        if not path.exists():
            continue
        frame = gpd.read_file(path)
        if frame.crs is None:
            report["warnings"].append(f"Skipped {path.name}: CRS is missing")
            continue
        added = 0
        for _, row in frame.iterrows():
            properties = row.drop(labels=[frame.geometry.name]).to_dict()
            record_id = properties.get(id_col)
            synthetic = _truth(properties.get("is_synthetic"))
            provenance = str(properties.get("provenance", ""))
            disclaimer = str(properties.get("synthetic_disclaimer", "")).strip()
            if require_synthetic and not (synthetic and provenance == "synthetic_demo" and disclaimer):
                report["rejected_optional_synthetic_rows"].append({"file": path.name, "id": str(record_id), "reason": "route demo requires is_synthetic=true, provenance=synthetic_demo and a disclaimer"})
                continue
            if rdf_type == TA.Pasture and not synthetic and not _present(properties.get("source_url")):
                report["warnings"].append(f"Skipped pasture {record_id}: non-synthetic pasture feature has no source_url")
                continue
            if synthetic and (provenance != "synthetic_demo" or not disclaimer):
                report["rejected_optional_synthetic_rows"].append({"file": path.name, "id": str(record_id), "reason": "synthetic flag/provenance/disclaimer inconsistent"})
                continue
            if not _present(record_id):
                report["warnings"].append(f"Skipped a feature in {path.name}: missing {id_col}")
                continue
            if row.geometry is None or row.geometry.is_empty:
                report["warnings"].append(f"Skipped {record_id} in {path.name}: empty geometry")
                continue
            uri = URIRef(BASE + ("route/" if rdf_type == TA.Route else "pasture/") + _slug(record_id))
            graph.add((uri, RDF.type, rdf_type))
            graph.add((uri, RDF.type, GEO.Feature))
            graph.add((uri, RDFS.label, Literal(str(properties.get("name") or properties.get("label") or record_id), lang="en")))
            graph.add((uri, DCTERMS.identifier, Literal(str(record_id))))
            graph.add((uri, TA.isSynthetic, Literal(synthetic, datatype=XSD.boolean)))
            if provenance:
                graph.add((uri, TA.provenanceLabel, Literal(provenance)))
            if synthetic:
                graph.add((uri, TA.syntheticDisclaimer, Literal(disclaimer)))
            _add_geometry(graph, uri, row.geometry, frame.crs)
            location_id = str(properties.get("location_id", "naryn-region"))
            if location_id in location_uris:
                graph.add((uri, TA.locatedIn, location_uris[location_id]))
                graph.add((uri, TA.aboutLocation, location_uris[location_id]))
            if rdf_type == TA.Route:
                for pasture_id in str(properties.get("pasture_ids", "")).split(";"):
                    if not pasture_id.strip():
                        continue
                    pasture_uri = URIRef(BASE + "pasture/" + _slug(pasture_id))
                    if (pasture_uri, RDF.type, TA.Pasture) in graph:
                        graph.add((uri, TA.connectsPasture, pasture_uri))
            added += 1
        key = "synthetic_routes" if rdf_type == TA.Route else "pasture_features"
        report[key] = added


def _add_optional_synthetic_records(graph: Graph, root: Path, location_uris: dict[str, URIRef],
                                    report: dict[str, Any]) -> None:
    """Ingest separately authored fictional interview and multimedia CSVs only."""
    specs = [
        (root / "data" / "synthetic" / "interviews.csv", "InterviewRecord", "interview_id"),
        (root / "data" / "synthetic" / "multimedia.csv", "MultimediaAsset", "asset_id"),
    ]
    for path, class_name, id_col in specs:
        frame = _read_csv(path)
        if frame.empty:
            continue
        added = 0
        for _, row in frame.iterrows():
            if not _truth(row.get("is_synthetic")) or str(row.get("provenance", "")) != "synthetic_demo" or not _present(row.get("synthetic_disclaimer")):
                report["rejected_optional_synthetic_rows"].append({"file": path.name, "id": str(row.get(id_col, "")), "reason": "synthetic flag, provenance='synthetic_demo' and disclaimer are all mandatory"})
                continue
            record_id = row.get(id_col)
            if not _present(record_id):
                report["rejected_optional_synthetic_rows"].append({"file": path.name, "reason": f"missing {id_col}"})
                continue
            uri = URIRef(BASE + "record/" + _slug(record_id))
            graph.add((uri, RDF.type, TA[class_name]))
            graph.add((uri, RDF.type, TA.ResearchRecord))
            graph.add((uri, RDFS.label, Literal(str(row.get("title") or row.get("label") or record_id), lang="en")))
            graph.add((uri, DCTERMS.identifier, Literal(str(record_id))))
            graph.add((uri, TA.isSynthetic, Literal(True, datatype=XSD.boolean)))
            graph.add((uri, TA.provenanceLabel, Literal("synthetic_demo")))
            graph.add((uri, TA.syntheticDisclaimer, Literal(str(row["synthetic_disclaimer"]))))
            _add_literal(graph, uri, DCTERMS.description, row.get("description"))
            loc_id = str(row.get("location_id", ""))
            if loc_id in location_uris:
                graph.add((uri, TA.aboutLocation, location_uris[loc_id]))
            added += 1
        report["optional_synthetic_records"][class_name] = added




def _add_simulation_entities(graph: Graph, root: Path, location_uris: dict[str, URIRef],
                             report: dict[str, Any]) -> None:
    """Ingest explicitly synthetic pastures, corridors, herds, scenarios, suitability assessments, and movement events."""
    syn_dir = root / "data" / "synthetic"
    if not syn_dir.exists():
        return
    default_disclaimer = (
        "Synthetic educational demonstration only. Generated herds, candidate grazing locations, "
        "modelled corridors, and climate scenario trajectories are fictional."
    )

    def _ensure_scenario(scen_id: str, scenario_uris: dict[str, URIRef]) -> URIRef:
        if scen_id not in scenario_uris:
            s_uri = URIRef(BASE + "scenario/" + _slug(scen_id))
            scenario_uris[scen_id] = s_uri
            graph.add((s_uri, RDF.type, TA.SimulationScenario))
            graph.add((s_uri, RDFS.label, Literal(scen_id, lang="en")))
            graph.add((s_uri, DCTERMS.identifier, Literal(scen_id)))
            graph.add((s_uri, TA.originClass, Literal("synthetic" if scen_id == "SCENARIO_BASELINE" else "simulated_scenario")))
        return scenario_uris[scen_id]

    # 1. Synthetic candidate pastures (data/synthetic/pastures.geojson)
    pastures_path = syn_dir / "pastures.geojson"
    pasture_uris: dict[str, URIRef] = {}
    if pastures_path.exists():
        try:
            doc = json.loads(pastures_path.read_text(encoding="utf-8"))
            from shapely.geometry import shape
            for feat in doc.get("features", []):
                props = feat.get("properties") or {}
                pid = props.get("pasture_id")
                if not pid or not _truth(props.get("is_synthetic")) or str(props.get("provenance")) != "synthetic_demo":
                    continue
                uri = URIRef(BASE + "pasture/" + _slug(pid))
                pasture_uris[str(pid)] = uri
                graph.add((uri, RDF.type, TA.Pasture))
                graph.add((uri, RDF.type, GEO.Feature))
                graph.add((uri, RDFS.label, Literal(str(props.get("pasture_name") or pid), lang="en")))
                graph.add((uri, DCTERMS.identifier, Literal(str(pid))))
                graph.add((uri, TA.isSynthetic, Literal(True, datatype=XSD.boolean)))
                graph.add((uri, TA.provenanceLabel, Literal("synthetic_demo")))
                graph.add((uri, TA.originClass, Literal(str(props.get("origin_class") or "synthetic"))))
                graph.add((uri, TA.syntheticDisclaimer, Literal(str(props.get("synthetic_disclaimer") or default_disclaimer))))
                graph.add((uri, TA.localAssetPath, Literal("data/synthetic/pastures.geojson")))
                _add_literal(graph, uri, WGS84.lat, _float(props.get("latitude")), XSD.decimal)
                _add_literal(graph, uri, WGS84.long, _float(props.get("longitude")), XSD.decimal)
                _add_literal(graph, uri, TA.elevationM, _float(props.get("elevation_m")), XSD.decimal)
                _add_literal(graph, uri, TA.slopeDegrees, _float(props.get("slope_degrees")), XSD.decimal)
                _add_literal(graph, uri, TA.season, props.get("seasonal_suitability"))
                _add_literal(graph, uri, TA.worldCoverClassCode, _integer(props.get("landcover_code")), XSD.integer)
                _add_literal(graph, uri, TA.randomSeed, _integer(props.get("random_seed")), XSD.integer)
                if "naryn-region" in location_uris:
                    graph.add((uri, TA.locatedIn, location_uris["naryn-region"]))
                if feat.get("geometry"):
                    _add_geometry(graph, uri, shape(feat["geometry"]), "EPSG:4326")
            report["synthetic_candidate_pastures"] = len(pasture_uris)
        except Exception as exc:
            report["warnings"].append(f"Could not ingest synthetic pastures into RDF: {exc}")

    # 2. Synthetic least-cost migration corridors (data/synthetic/migration_routes.geojson)
    corridors_path = syn_dir / "migration_routes.geojson"
    corridor_uris: dict[str, URIRef] = {}
    if corridors_path.exists():
        try:
            doc = json.loads(corridors_path.read_text(encoding="utf-8"))
            from shapely.geometry import shape
            for feat in doc.get("features", []):
                props = feat.get("properties") or {}
                rid = props.get("route_id")
                if not rid or not _truth(props.get("is_synthetic")) or str(props.get("provenance")) != "synthetic_demo":
                    continue
                uri = URIRef(BASE + "route/" + _slug(rid))
                corridor_uris[str(rid)] = uri
                graph.add((uri, RDF.type, TA.Route))
                graph.add((uri, RDF.type, GEO.Feature))
                graph.add((uri, RDFS.label, Literal(str(props.get("name") or rid), lang="en")))
                graph.add((uri, DCTERMS.identifier, Literal(str(rid))))
                graph.add((uri, TA.isSynthetic, Literal(True, datatype=XSD.boolean)))
                graph.add((uri, TA.provenanceLabel, Literal("synthetic_demo")))
                graph.add((uri, TA.originClass, Literal(str(props.get("origin_class") or "synthetic"))))
                graph.add((uri, TA.syntheticDisclaimer, Literal(str(props.get("synthetic_disclaimer") or default_disclaimer))))
                graph.add((uri, TA.localAssetPath, Literal("data/synthetic/migration_routes.geojson")))
                _add_literal(graph, uri, TA.distanceKm, _float(props.get("distance_km")), XSD.decimal)
                _add_literal(graph, uri, TA.elevationGainM, _float(props.get("elevation_gain_m")), XSD.decimal)
                _add_literal(graph, uri, TA.slopeDegrees, _float(props.get("avg_slope_deg")), XSD.decimal)
                _add_literal(graph, uri, TA.season, props.get("season"))
                _add_literal(graph, uri, TA.randomSeed, _integer(props.get("random_seed")), XSD.integer)
                if "naryn-region" in location_uris:
                    graph.add((uri, TA.locatedIn, location_uris["naryn-region"]))
                for end_col in ("origin_pasture_id", "destination_pasture_id"):
                    pid = str(props.get(end_col) or "")
                    if pid in pasture_uris:
                        graph.add((uri, TA.connectsPasture, pasture_uris[pid]))
                if feat.get("geometry"):
                    _add_geometry(graph, uri, shape(feat["geometry"]), "EPSG:4326")
            report["synthetic_corridors"] = len(corridor_uris)
        except Exception as exc:
            report["warnings"].append(f"Could not ingest migration corridors into RDF: {exc}")

    # 3. Synthetic herd groups (data/synthetic/herds.csv)
    herds_df = _read_csv(syn_dir / "herds.csv")
    herd_uris: dict[str, URIRef] = {}
    if not herds_df.empty:
        for _, row in herds_df.iterrows():
            hid = str(row.get("herd_id") or "")
            if not hid or not _truth(row.get("is_synthetic")) or str(row.get("provenance")) != "synthetic_demo":
                continue
            uri = URIRef(BASE + "herd/" + _slug(hid))
            herd_uris[hid] = uri
            graph.add((uri, RDF.type, TA.HerdGroup))
            graph.add((uri, RDFS.label, Literal(str(row.get("herd_name") or hid), lang="en")))
            graph.add((uri, DCTERMS.identifier, Literal(hid)))
            _add_literal(graph, uri, TA.livestockSpecies, row.get("species") or row.get("livestock_type"))
            _add_literal(graph, uri, TA.groupSize, _integer(row.get("group_size") if _present(row.get("group_size")) else row.get("animal_count")), XSD.integer)
            _add_literal(graph, uri, TA.randomSeed, _integer(row.get("random_seed")), XSD.integer)
            graph.add((uri, TA.isSynthetic, Literal(True, datatype=XSD.boolean)))
            graph.add((uri, TA.provenanceLabel, Literal("synthetic_demo")))
            graph.add((uri, TA.originClass, Literal(str(row.get("origin_class") or "synthetic"))))
            graph.add((uri, TA.syntheticDisclaimer, Literal(str(row.get("synthetic_disclaimer") or default_disclaimer))))
            graph.add((uri, TA.localAssetPath, Literal("data/synthetic/herds.csv")))
            # Link all direct and composite routes used by this herd
            for rcol in ("spring_route_id", "summer_route_id", "route_id"):
                raw_rid = str(row.get(rcol) or "")
                for part in raw_rid.split("+"):
                    rid_clean = part.strip()
                    if rid_clean in corridor_uris:
                        graph.add((uri, TA.usesRoute, corridor_uris[rid_clean]))
            # Link all assigned seasonal pastures
            for pcol in ("winter_pasture_id", "spring_destination_id", "summer_pasture_id", "autumn_destination_id"):
                pid_val = str(row.get(pcol) or "").strip()
                if pid_val in pasture_uris:
                    graph.add((uri, TA.usesPasture, pasture_uris[pid_val]))
        report["synthetic_herds"] = len(herd_uris)

    # 4. Climate scenarios & discrete movement events (data/synthetic/movement_events.csv)
    events_df = _read_csv(syn_dir / "movement_events.csv")
    scenario_uris: dict[str, URIRef] = {}
    if not events_df.empty:
        added_events = 0
        for _, row in events_df.iterrows():
            eid = str(row.get("event_id") or "")
            if not eid or not _truth(row.get("is_synthetic")) or str(row.get("provenance")) != "synthetic_demo":
                continue
            scen_id = str(row.get("climate_scenario") or "SCENARIO_BASELINE")
            scen_uri = _ensure_scenario(scen_id, scenario_uris)
            uri = URIRef(BASE + "event/" + _slug(eid))
            graph.add((uri, RDF.type, TA.MovementEvent))
            graph.add((uri, RDFS.label, Literal(f"{row.get('herd_id')} {row.get('event_type')} ({row.get('leg_name')})", lang="en")))
            graph.add((uri, DCTERMS.identifier, Literal(eid)))
            _add_literal(graph, uri, TA.occurredAt, row.get("timestamp"), XSD.dateTime)
            _add_literal(graph, uri, TA.eventType, row.get("event_type"))
            _add_literal(graph, uri, TA.legName, row.get("leg_name"))
            _add_literal(graph, uri, TA.elevationM, _float(row.get("elevation_m")), XSD.decimal)
            _add_literal(graph, uri, TA.randomSeed, _integer(row.get("random_seed")), XSD.integer)
            graph.add((uri, TA.underScenario, scen_uri))
            graph.add((uri, TA.isSynthetic, Literal(True, datatype=XSD.boolean)))
            graph.add((uri, TA.provenanceLabel, Literal("synthetic_demo")))
            graph.add((uri, TA.originClass, Literal(str(row.get("origin_class") or ("synthetic" if scen_id == "SCENARIO_BASELINE" else "simulated_scenario")))))
            graph.add((uri, TA.syntheticDisclaimer, Literal(default_disclaimer)))
            graph.add((uri, TA.localAssetPath, Literal("data/synthetic/movement_events.csv")))
            hid = str(row.get("herd_id") or "").strip()
            if hid in herd_uris:
                graph.add((uri, TA.involvesHerd, herd_uris[hid]))
            pid = str(row.get("pasture_id") or "").strip()
            if pid in pasture_uris:
                graph.add((uri, TA.aboutLocation, pasture_uris[pid]))
            raw_rid = str(row.get("route_id") or "")
            for part in raw_rid.split("+"):
                rid_clean = part.strip()
                if rid_clean in corridor_uris:
                    graph.add((uri, TA.usesRoute, corridor_uris[rid_clean]))
            added_events += 1
        report["synthetic_movement_events"] = added_events

    # 5. Scenario pasture suitability assessments (data/synthetic/suitability_scores.csv)
    suit_df = _read_csv(syn_dir / "suitability_scores.csv")
    if not suit_df.empty:
        added_suit = 0
        for _, row in suit_df.iterrows():
            pid = str(row.get("pasture_id") or "").strip()
            scen_id = str(row.get("scenario_id") or "").strip()
            seas = str(row.get("season") or "").strip()
            if not (pid and scen_id and seas) or not _truth(row.get("is_synthetic")) or str(row.get("provenance")) != "synthetic_demo":
                continue
            scen_uri = _ensure_scenario(scen_id, scenario_uris)
            aid = f"suit-{_slug(scen_id)}-{_slug(seas)}-{_slug(pid)}"
            uri = URIRef(BASE + "suitability/" + aid)
            graph.add((uri, RDF.type, TA.PastureSuitabilityAssessment))
            graph.add((uri, RDFS.label, Literal(f"Suitability {pid} ({seas}, {scen_id})", lang="en")))
            graph.add((uri, DCTERMS.identifier, Literal(aid)))
            _add_literal(graph, uri, TA.season, seas)
            _add_literal(graph, uri, TA.suitabilityScore, _float(row.get("suitability_score")), XSD.decimal)
            graph.add((uri, TA.underScenario, scen_uri))
            graph.add((uri, TA.isSynthetic, Literal(True, datatype=XSD.boolean)))
            graph.add((uri, TA.provenanceLabel, Literal("synthetic_demo")))
            # Normalize originClass so synthetic baseline suitability is never mislabelled 'public'
            norm_origin = "synthetic" if scen_id == "SCENARIO_BASELINE" else "simulated_scenario"
            graph.add((uri, TA.originClass, Literal(norm_origin)))
            graph.add((uri, TA.syntheticDisclaimer, Literal(default_disclaimer)))
            if pid in pasture_uris:
                graph.add((pasture_uris[pid], TA.hasSuitabilityAssessment, uri))
            added_suit += 1
        report["synthetic_suitability_assessments"] = added_suit

def _bind(graph: Graph) -> None:
    for prefix, namespace in {
        "ta": TA, "tav": TAV, "rdf": RDF, "rdfs": RDFS, "owl": OWL,
        "xsd": XSD, "dcterms": DCTERMS, "dcat": DCAT, "prov": PROV,
        "geo": GEO, "skos": SKOS, "wgs84": WGS84, "schema": SCHEMA,
    }.items():
        graph.bind(prefix, namespace)


def build_graph(root: Path) -> tuple[Graph, dict[str, Any]]:
    root = root.resolve()
    graph = Graph()
    _bind(graph)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "locations": 0,
        "boundary_geometry_added": False,
        "climate_observations_by_file": {},
        "livestock_observations": 0,
        "optional_synthetic_records": {},
        "rejected_optional_synthetic_rows": [],
        "derived_datasets_added": [],
        "warnings": [],
        "counts": {},
        "data_integrity_policy": "No pasture, livestock, climate, route, interview or media observations are invented. Synthetic records are accepted only with is_synthetic=true, provenance='synthetic_demo' and a non-empty disclaimer.",
    }

    # Include controlled vocabulary and ontology definitions in the graph itself.
    ontology_path = root / "ontology" / "transhumance.ttl"
    vocabulary_path = root / "vocab" / "transhumance-vocabulary.ttl"
    if ontology_path.exists(): graph.parse(ontology_path, format="turtle")
    else: report["warnings"].append("Ontology TTL not found; classes are still referenced by URI but definitions were not merged")
    if vocabulary_path.exists(): graph.parse(vocabulary_path, format="turtle")
    else: report["warnings"].append("SKOS vocabulary TTL not found; vocabulary definitions were not merged")

    register_path = root / "metadata" / "dataset_register.csv"
    dataset_uris: dict[str, URIRef] = {}
    if register_path.exists():
        with register_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                dataset_uris[str(row.get("dataset_id", ""))] = _add_dataset(graph, root, row, report.setdefault("source_dataset_status", {}))
    else:
        report["warnings"].append("metadata/dataset_register.csv not found")

    qids = _verified_qids(root)
    location_uris = _add_locations(graph, root, qids, report)

    # Add the public boundary geometry only when its actual processed/raw file exists.
    # It is never inferred from a point coordinate or drawn synthetically.
    boundary_uri = URIRef(BASE + "location/Q486375")
    if report["boundary_geometry_added"]:
        graph.add((boundary_uri, TA.localAssetPath, Literal("data/processed/vectors/naryn_adm1_clean.geojson")))

    nasa_uri = dataset_uris.get("NASA-POWER-DAILY", URIRef(BASE + "dataset/nasa-power-daily"))
    for filename in ("nasa_power_daily.csv", "nasa_power_monthly.csv", "nasa_power_seasonal.csv"):
        _add_climate_csv(graph, root, filename, nasa_uri, location_uris, report)

    fao_uri = dataset_uris.get("FAO-GLW4-2020-DENSITY", URIRef(BASE + "dataset/fao-glw4-2020-density"))
    _add_livestock_from_context(graph, root, location_uris, fao_uri, report)
    _add_dataset_distribution_metadata(graph, root, report)
    _add_optional_spatial_research_records(graph, root, location_uris, report)
    _add_optional_synthetic_records(graph, root, location_uris, report)
    _add_simulation_entities(graph, root, location_uris, report)

    # Record the graph-generation activity with provenance.
    activity = URIRef(BASE + "activity/linked-data-build/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    graph.add((activity, RDF.type, TA.ProcessingActivity))
    graph.add((activity, RDF.type, PROV.Activity))
    graph.add((activity, RDFS.label, Literal("RDF graph construction from the local dataset register and Stage 3 outputs")))
    graph.add((activity, PROV.startedAtTime, Literal(report["generated_at_utc"], datatype=XSD.dateTime)))
    graph.add((activity, PROV.wasAssociatedWith, URIRef(BASE + "agent/stage4-python-pipeline")))
    graph.add((URIRef(BASE + "graph/research"), PROV.wasGeneratedBy, activity))
    graph.add((URIRef(BASE + "graph/research"), RDF.type, PROV.Entity))
    graph.add((URIRef(BASE + "graph/research"), DCTERMS.title, Literal("Transhumance atlas research knowledge graph")))
    graph.add((URIRef(BASE + "graph/research"), DCTERMS.created, Literal(report["generated_at_utc"], datatype=XSD.dateTime)))

    report["counts"] = {
        "triples": len(graph),
        "locations": len(set(graph.subjects(RDF.type, TA.Location))),
        "pastures": len(set(graph.subjects(RDF.type, TA.Pasture))),
        "routes": len(set(graph.subjects(RDF.type, TA.Route))),
        "climate_observations": len(set(graph.subjects(RDF.type, TA.ClimateObservation))),
        "livestock_observations": len(set(graph.subjects(RDF.type, TA.LivestockObservation))),
        "interview_records": len(set(graph.subjects(RDF.type, TA.InterviewRecord))),
        "multimedia_assets": len(set(graph.subjects(RDF.type, TA.MultimediaAsset))),
        "datasets": len(set(graph.subjects(RDF.type, DCAT.Dataset))),
        "skos_concepts": len(set(graph.subjects(RDF.type, SKOS.Concept))),
        "herd_groups": len(set(graph.subjects(RDF.type, TA.HerdGroup))),
        "movement_events": len(set(graph.subjects(RDF.type, TA.MovementEvent))),
        "simulation_scenarios": len(set(graph.subjects(RDF.type, TA.SimulationScenario))),
        "suitability_assessments": len(set(graph.subjects(RDF.type, TA.PastureSuitabilityAssessment))),
    }
    return graph, report


def validate_graph(graph: Graph, verified_qids: set[str]) -> dict[str, Any]:
    issues: list[str] = []
    missing_dataset_metadata: list[str] = []
    for dataset in set(graph.subjects(RDF.type, DCAT.Dataset)):
        for predicate in (DCTERMS.identifier, DCTERMS.title, DCTERMS.publisher, TA.localAvailability):
            if not list(graph.objects(dataset, predicate)):
                missing_dataset_metadata.append(f"{dataset} missing {predicate}")
    if missing_dataset_metadata:
        issues.extend(missing_dataset_metadata)

    unresolved_sameas: list[str] = []
    for _, _, obj in graph.triples((None, OWL.sameAs, None)):
        qid = str(obj).rsplit("/", 1)[-1]
        if qid.startswith("Q") and qid not in verified_qids:
            unresolved_sameas.append(str(obj))
    if unresolved_sameas:
        issues.append(f"Unresolved Wikidata owl:sameAs identifiers: {unresolved_sameas}")

    synthetic_issues: list[str] = []
    for subject in set(graph.subjects(TA.isSynthetic, Literal(True, datatype=XSD.boolean))):
        provenance = {str(v) for v in graph.objects(subject, TA.provenanceLabel)}
        disclaimers = [str(v).strip() for v in graph.objects(subject, TA.syntheticDisclaimer)]
        if "synthetic_demo" not in provenance or not any(disclaimers):
            synthetic_issues.append(str(subject))
    if synthetic_issues:
        issues.append(f"Synthetic records missing required provenance/disclaimer: {synthetic_issues}")

    # A mapped OSM dataset is context only. The graph generator never types it as ta:Route.
    report = {
        "status": "PASS" if not issues else "FAIL",
        "triple_count": len(graph),
        "dataset_nodes": len(set(graph.subjects(RDF.type, DCAT.Dataset))),
        "sameAs_links": len(set(graph.subject_objects(OWL.sameAs))),
        "unresolved_sameAs": unresolved_sameas,
        "synthetic_provenance_issues": synthetic_issues,
        "missing_dataset_metadata": missing_dataset_metadata,
        "issues": issues,
    }
    return report


def write_outputs(graph: Graph, report: dict[str, Any], root: Path) -> None:
    outdir = root / "linked_data"
    outdir.mkdir(parents=True, exist_ok=True)
    ttl_path = outdir / "research_graph.ttl"
    jsonld_path = outdir / "research_graph.jsonld"
    graph.serialize(destination=ttl_path, format="turtle", encoding="utf-8")
    context = {
        "ta": str(TA), "dcterms": str(DCTERMS), "dcat": str(DCAT),
        "prov": str(PROV), "geo": str(GEO), "skos": str(SKOS),
        "wgs84": str(WGS84), "rdf": str(RDF), "rdfs": str(RDFS),
        "owl": str(OWL), "xsd": str(XSD),
    }
    jsonld_text = graph.serialize(format="json-ld", indent=2, auto_compact=True, context=context)
    jsonld_path.write_text(jsonld_text, encoding="utf-8")
    report["outputs"] = {
        "turtle": ttl_path.relative_to(root).as_posix(),
        "jsonld": jsonld_path.relative_to(root).as_posix(),
    }


def _optional_shacl_validation(graph: Graph, root: Path) -> dict[str, Any]:
    """Run pySHACL only when explicitly installed; never imply it ran otherwise."""
    shapes_path = root / "queries" / "shapes" / "linked_data_shapes.ttl"
    if not shapes_path.exists():
        return {"status": "NOT_RUN", "reason": "SHACL shapes file is missing"}
    try:
        from pyshacl import validate as shacl_validate
    except ImportError:
        return {
            "status": "NOT_RUN_OPTIONAL_DEPENDENCY_MISSING",
            "reason": "pyshacl is not installed; install requirements-stage4-shacl.txt to execute SHACL validation",
        }
    shapes = Graph().parse(shapes_path, format="turtle")
    conforms, results_graph, results_text = shacl_validate(
        data_graph=graph,
        shacl_graph=shapes,
        inference="none",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )
    return {
        "status": "PASS" if conforms else "FAIL",
        "conforms": bool(conforms),
        "results_triples": len(results_graph),
        "details": str(results_text),
    }


def build_from_root(root: Path) -> dict[str, Any]:
    graph, report = build_graph(root)
    qids = set(_verified_qids(root))
    validation = validate_graph(graph, qids)
    report["validation"] = validation
    report["shacl_validation"] = _optional_shacl_validation(graph, root)
    write_outputs(graph, report, root)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage4_linked_data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RDF/Turtle and JSON-LD linked-data graph")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = build_from_root(args.root)
    print(json.dumps(report["counts"], indent=2))
    print("Validation:", report["validation"]["status"])
    print("Turtle:", args.root.resolve() / "linked_data" / "research_graph.ttl")
    print("JSON-LD:", args.root.resolve() / "linked_data" / "research_graph.jsonld")
    return 0 if report["validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
