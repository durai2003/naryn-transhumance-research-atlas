#!/usr/bin/env python3
"""Recover only Stage 2 assets that failed or were blocked in the prior run.

This script reuses the already-downloaded geoBoundaries GeoJSON and Geofabrik ZIP.
It does not redownload NASA POWER, FAO metadata, OSM ZIP, or KDC capabilities.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_SCRIPT))
from stage2_acquire import (  # noqa: E402
    HEADERS, TIMEOUT, RETRIES, now, record,
    select_naryn_feature, extract_geopackage_from_zip, worldcover_tiles,
)


def request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    last = None
    headers = kwargs.pop("headers", HEADERS)
    for attempt in range(RETRIES):
        try:
            response = requests.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
            if attempt + 1 < RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after {RETRIES} attempts: {url}: {last}")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = request_with_retry("GET", url, stream=True)
    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    if not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {url}")


def wikidata_query(root: Path, log_path: Path) -> bool:
    query = '''SELECT ?item ?itemLabel ?coordinate WHERE {
  VALUES ?item { wd:Q486375 wd:Q311946 }
  OPTIONAL { ?item wdt:P625 ?coordinate. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}'''
    errors = []
    # The new v2 service returned 403 in the prior user run. Try v1 first;
    # fall back to v2 only if v1 fails, and retain the endpoint provenance.
    for endpoint in (
        "https://query.wikidata.org/sparql",
        "https://query-next.wikidata.org/sparql",
    ):
        try:
            response = request_with_retry(
                "POST", endpoint,
                data={"query": query},
                headers={**HEADERS, "Accept": "application/sparql-results+json"},
            )
            results = response.json()
            if not isinstance(results, dict) or "results" not in results:
                raise RuntimeError("Response is not SPARQL Results JSON")
            out = root / "data/raw/wikidata/naryn_entities_sparql.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "endpoint": endpoint,
                "query": query,
                "retrieved_utc": now(),
                "results": results,
            }, indent=2), encoding="utf-8")
            record(log_path, "Wikidata Naryn entity verification query", "SUCCESS",
                   "Saved SPARQL JSON and query provenance", endpoint=endpoint)
            return True
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    record(log_path, "Wikidata Naryn entity verification query", "FAILED",
           "Both documented endpoints failed: " + " | ".join(errors))
    return False


def inspect_fao_resources(root: Path) -> Path | None:
    source = root / "data/raw/source_metadata/fao_glw4_D-DA-1KM_mapset.json"
    if not source.exists():
        return None
    doc = json.loads(source.read_text(encoding="utf-8"))
    records = []

    def walk(value, path="$"):
        if isinstance(value, dict):
            label = next((str(value[k]) for k in ("name", "title", "label", "description") if isinstance(value.get(k), (str, int, float))), None)
            urls = {k: v for k, v in value.items() if isinstance(v, str) and ("url" in k.casefold() or v.startswith(("http://", "https://")))}
            if urls:
                records.append({"json_path": path, "label": label, "url_fields": urls, "keys": sorted(value.keys())})
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")

    walk(doc)
    out = root / "reports/fao_glw4_resource_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source_file": str(source.relative_to(root)),
        "generated_utc": now(),
        "resource_records": records,
        "manual_species_confirmation_required": True,
        "instruction": "Select verified Sheep density and Goat density raster resources only. Do not substitute animal-number layers for density layers.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run(root: Path) -> int:
    log_path = root / "reports/stage2_recovery_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Select Naryn from the already-downloaded countrywide ADM1 GeoJSON.
    source_boundary = root / "data/raw/boundaries/geoBoundaries-KGZ-ADM1.geojson"
    naryn_boundary = root / "data/raw/boundaries/naryn_adm1.geojson"
    diagnostic = root / "data/raw/source_metadata/geoboundaries_KGZ_ADM1_feature_inventory.json"
    if source_boundary.exists():
        try:
            props = select_naryn_feature(source_boundary, naryn_boundary, diagnostic)
            record(log_path, "Naryn ADM1 geometry", "SUCCESS", "Selected exact Naryn administrative unit from existing GeoJSON", properties=props, diagnostic=str(diagnostic))
        except Exception as exc:
            record(log_path, "Naryn ADM1 geometry", "FAILED", str(exc), diagnostic=str(diagnostic))
    else:
        record(log_path, "Naryn ADM1 geometry", "BLOCKED_MISSING_SOURCE_FILE", "Expected downloaded source GeoJSON was not found", expected=str(source_boundary))

    # 2. Once the real boundary exists, calculate and download only intersecting WorldCover tiles.
    if naryn_boundary.exists():
        try:
            import geopandas as gpd
            boundary = gpd.read_file(naryn_boundary)
            if boundary.crs is None:
                raise RuntimeError("Selected Naryn boundary has no CRS; refusing to guess")
            bounds = tuple(float(x) for x in boundary.to_crs("EPSG:4326").total_bounds)
            tiles = worldcover_tiles(bounds)
            record(log_path, "ESA WorldCover tile selection", "SUCCESS", "Computed tiles from actual Naryn boundary bounds", bounds=bounds, tile_ids=tiles)
            base = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            for tile in tiles:
                filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
                dest = root / "data/raw/worldcover" / filename
                if dest.exists() and dest.stat().st_size > 0:
                    record(log_path, filename, "ALREADY_PRESENT", "Existing tile retained; not downloaded again", bytes=dest.stat().st_size)
                    continue
                try:
                    download_file(base + filename, dest)
                    record(log_path, filename, "SUCCESS", "Downloaded intersecting ESA WorldCover 2021 tile", url=base + filename, bytes=dest.stat().st_size)
                except Exception as exc:
                    record(log_path, filename, "FAILED", str(exc), url=base + filename)
        except Exception as exc:
            record(log_path, "ESA WorldCover tile acquisition", "FAILED", str(exc))
    else:
        record(log_path, "ESA WorldCover 2021 tiles", "BLOCKED_NO_BOUNDARY", "No source tile selected; actual Naryn geometry is still missing")

    # 3. Extract the GeoPackage from the already-downloaded Geofabrik archive.
    archive = root / "data/raw/osm/kyrgyzstan-latest-free.gpkg.zip"
    if archive.exists():
        try:
            gpkg = extract_geopackage_from_zip(archive, root / "data/raw/osm/extracted")
            record(log_path, "Geofabrik GeoPackage extraction", "SUCCESS", "Extracted GeoPackage from existing archive", geopackage=str(gpkg), bytes=gpkg.stat().st_size)
        except Exception as exc:
            record(log_path, "Geofabrik GeoPackage extraction", "FAILED", str(exc))
    else:
        record(log_path, "Geofabrik GeoPackage extraction", "BLOCKED_MISSING_ARCHIVE", "Existing GeoPackage ZIP was not found", expected=str(archive))

    # 4. Report FAO resources but require manual species/density selection before download.
    fao_inventory = inspect_fao_resources(root)
    if fao_inventory:
        record(log_path, "FAO GLW4 resource inventory", "SUCCESS", "Extracted resource URLs and labels from local mapset metadata; sheep/goat density selection still requires review", file=str(fao_inventory))
    else:
        record(log_path, "FAO GLW4 resource inventory", "BLOCKED_MISSING_METADATA", "Local FAO mapset metadata not found")

    # 5. Retry SPARQL, using v1 first and v2 as fallback.
    wikidata_query(root, log_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raise SystemExit(run(args.root.resolve()))
