#!/usr/bin/env python3
"""Stage 2 data acquisition. Downloads only real source assets; never fabricates fallbacks."""
from __future__ import annotations
import argparse, json, math, re, sys, time, zipfile, unicodedata, shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

HEADERS = {"User-Agent": "MiniTranshumanceAtlas/0.1 (educational data integration prototype)"}
TIMEOUT = (15, 120)
RETRIES = 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(log_path: Path, asset: str, status: str, details: str, **extra: Any) -> None:
    payload = {"timestamp_utc": now(), "asset": asset, "status": status, "details": details, **extra}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"[{status}] {asset}: {details}")


def get(url: str, *, params: dict | None = None, stream: bool = False) -> requests.Response:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT, stream=stream)
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            if attempt + 1 < RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}: {last}")


def save_response(response: requests.Response, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def download(url: str, path: Path) -> None:
    r = get(url, stream=True)
    save_response(r, path)
    if path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Empty download: {url}")


def download_geoboundaries_geojson(meta: dict, destination: Path) -> None:
    """Fetch GeoJSON and resolve public Git LFS pointer, if returned."""
    url = meta.get("gjDownloadURL")
    if not url:
        raise RuntimeError("geoBoundaries metadata has no gjDownloadURL")
    r = get(url)
    body = r.content
    prefix = body[:256].decode("utf-8", errors="ignore")
    if "version https://git-lfs.github.com/spec/v1" in prefix:
        oid = re.search(r"oid sha256:([0-9a-f]{64})", prefix)
        size = re.search(r"size (\d+)", prefix)
        if not oid or not size:
            raise RuntimeError("GeoJSON URL returned an unparseable Git LFS pointer")
        batch_url = "https://github.com/wmgeolab/geoBoundaries.git/info/lfs/objects/batch"
        payload = {"operation":"download", "transfers":["basic"], "objects":[{"oid":oid.group(1), "size":int(size.group(1))}]}
        br = requests.post(batch_url, json=payload, headers={**HEADERS, "Accept":"application/vnd.git-lfs+json", "Content-Type":"application/vnd.git-lfs+json"}, timeout=TIMEOUT)
        br.raise_for_status()
        obj = br.json()["objects"][0]
        if obj.get("error"):
            raise RuntimeError(f"Git LFS object error: {obj['error']}")
        action = obj.get("actions", {}).get("download", {})
        href = action.get("href")
        if not href:
            raise RuntimeError("Git LFS batch response did not include a download URL")
        headers = action.get("header", {})
        rr = requests.get(href, headers=headers, timeout=TIMEOUT, stream=True)
        rr.raise_for_status()
        save_response(rr, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    # Validate it is parseable GeoJSON rather than an HTML error or pointer.
    doc = json.loads(destination.read_text(encoding="utf-8"))
    if doc.get("type") != "FeatureCollection" or not isinstance(doc.get("features"), list):
        raise RuntimeError("geoBoundaries file is not a GeoJSON FeatureCollection")


def normalise_admin_name(value: Any) -> str:
    """Normalise a name for exact matching; this is not fuzzy matching."""
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip()


def build_boundary_feature_inventory(doc: dict) -> list[dict]:
    """Return compact property diagnostics so an unmatched name is explainable."""
    rows = []
    for index, feature in enumerate(doc.get("features", [])):
        props = feature.get("properties") or {}
        rows.append({
            "feature_index": index,
            "property_keys": sorted(props.keys()),
            "shapeName": props.get("shapeName"),
            "shapeID": props.get("shapeID"),
            "shapeISO": props.get("shapeISO"),
            "all_properties": props,
        })
    return rows


def select_naryn_feature(source: Path, destination: Path, diagnostic_path: Path | None = None) -> dict:
    """Select Naryn only by exact administrative-name aliases; never by substring."""
    doc = json.loads(source.read_text(encoding="utf-8"))
    if doc.get("type") != "FeatureCollection" or not isinstance(doc.get("features"), list):
        raise RuntimeError("Source boundary is not a GeoJSON FeatureCollection")

    # The current KGZ ADM1 data represents the unit as 'Naryn Region'. Keep aliases
    # exact and auditable: don't accept a village whose name merely contains Naryn.
    aliases = {normalise_admin_name(x) for x in ("Naryn", "Naryn Region", "Naryn Oblast")}
    name_fields = {"shapename", "name", "name_1", "adm1_name", "admin1name", "shape_name"}
    candidates = []
    for feature in doc["features"]:
        props = feature.get("properties") or {}
        named_values = [v for k, v in props.items() if str(k).casefold() in name_fields]
        if not named_values:
            named_values = [v for v in props.values() if isinstance(v, str)]
        if any(normalise_admin_name(v) in aliases for v in named_values):
            candidates.append(feature)

    inventory = build_boundary_feature_inventory(doc)
    if diagnostic_path is not None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps({
            "source_file": str(source), "feature_count": len(doc["features"]),
            "selection_aliases": sorted(aliases), "features": inventory,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(candidates) != 1:
        summary = [{"shapeName": r.get("shapeName"), "shapeID": r.get("shapeID"), "shapeISO": r.get("shapeISO")} for r in inventory]
        raise RuntimeError(
            f"Expected exactly one exact Naryn ADM1 feature; found {len(candidates)}. "
            f"Feature summary: {summary}. Full property inventory: {diagnostic_path or 'not written'}"
        )

    feature = candidates[0]
    if not feature.get("geometry"):
        raise RuntimeError("The selected Naryn feature has no geometry; refusing to write an empty boundary")
    out = {"type": "FeatureCollection", "features": [feature],
           "name": "Naryn ADM1 selected from geoBoundaries KGZ ADM1"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return feature.get("properties") or {}


def extract_geopackage_from_zip(archive_path: Path, destination_dir: Path) -> Path:
    """Safely extract the GeoPackage member without unpacking unrelated files."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        members = [m for m in zf.infolist() if m.filename.casefold().endswith(".gpkg") and not m.is_dir()]
        if len(members) != 1:
            raise RuntimeError(f"Expected exactly one .gpkg member in {archive_path.name}; found {len(members)}")
        target = destination_dir / Path(members[0].filename).name
        with zf.open(members[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError("Extracted GeoPackage is empty")
    return target


def worldcover_tiles(bounds: tuple[float, float, float, float]) -> list[str]:
    minx, miny, maxx, maxy = bounds
    # Right/top are nudged inward so an edge exactly on a tile boundary doesn't add a non-intersecting tile.
    maxx = math.nextafter(maxx, -math.inf)
    maxy = math.nextafter(maxy, -math.inf)
    ids = []
    lat0 = math.floor(miny / 3) * 3
    lat1 = math.floor(maxy / 3) * 3
    lon0 = math.floor(minx / 3) * 3
    lon1 = math.floor(maxx / 3) * 3
    for lat in range(lat0, lat1 + 1, 3):
        for lon in range(lon0, lon1 + 1, 3):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            ids.append(ns + ew)
    return ids


def run(root: Path) -> int:
    log = root / "reports" / "acquisition_run_log.jsonl"
    meta_dir = root / "data" / "raw" / "source_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # geoBoundaries metadata + real feature collection
    try:
        url = "https://www.geoboundaries.org/api/current/gbOpen/KGZ/ADM1/"
        meta = get(url).json()
        (meta_dir / "geoboundaries_api_current_KGZ_ADM1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        record(log, "geoBoundaries ADM1 metadata", "SUCCESS", "Saved current API metadata", url=url)
        all_geo = root / "data/raw/boundaries/geoBoundaries-KGZ-ADM1.geojson"
        download_geoboundaries_geojson(meta, all_geo)
        diagnostic = meta_dir / "geoboundaries_KGZ_ADM1_feature_inventory.json"
        props = select_naryn_feature(all_geo, root / "data/raw/boundaries/naryn_adm1.geojson", diagnostic)
        record(log, "Naryn ADM1 geometry", "SUCCESS", "Selected exact Naryn Region feature from downloaded source geometry", properties=props, feature_inventory=str(diagnostic))
    except Exception as exc:
        record(log, "geoBoundaries ADM1 geometry", "FAILED", str(exc))

    # FAO catalogue metadata. The exact species download resources must be resolved from the returned official catalogue/API.
    try:
        api = "https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/GLW4-2020/mapsets/D-DA-1KM"
        r = get(api)
        path = meta_dir / "fao_glw4_D-DA-1KM_mapset.json"
        path.write_bytes(r.content)
        data = r.json()
        text_blob = json.dumps(data).casefold()
        record(log, "FAO GLW4 density mapset metadata", "SUCCESS", "Saved mapset metadata; inspect resource URLs before fetching sheep/goat layers", url=api)
        (meta_dir / "fao_glw4_species_resource_review.json").write_text(json.dumps({"source_api":api,"manual_review_required":True,"note":"Select the official Sheep density and Goats density TIF resources; do not substitute Animal Number for Animal Density.","metadata_keys":list(data) if isinstance(data,dict) else None}, indent=2), encoding="utf-8")
    except Exception as exc:
        record(log, "FAO GLW4 density mapset metadata", "FAILED", str(exc))

    # WorldCover: only select tiles after actual Naryn boundary has been retrieved.
    try:
        boundary_path = root / "data/raw/boundaries/naryn_adm1.geojson"
        if not boundary_path.exists():
            record(log, "ESA WorldCover 2021 tiles", "BLOCKED_NO_BOUNDARY", "Exact Naryn intersecting tile list cannot be selected without source boundary geometry")
        else:
            import geopandas as gpd
            gdf = gpd.read_file(boundary_path)
            if gdf.crs is None:
                raise RuntimeError("Naryn geometry has no CRS; do not guess/reassign CRS")
            gdf = gdf.to_crs("EPSG:4326")
            bounds = tuple(float(x) for x in gdf.total_bounds)
            tiles = worldcover_tiles(bounds)
            base = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            for tile in tiles:
                filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
                try:
                    download(base + filename, root / "data/raw/worldcover" / filename)
                    record(log, filename, "SUCCESS", "Downloaded ESA WorldCover 2021 Map COG", url=base+filename, bounds=bounds)
                except Exception as exc:
                    record(log, filename, "FAILED", str(exc), url=base+filename, bounds=bounds)
    except Exception as exc:
        record(log, "ESA WorldCover tile selection", "FAILED", str(exc))

    # NASA POWER daily point time series: Naryn city Q311946 coordinate statement.
    try:
        endpoint = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {"parameters":"T2M,PRECTOTCORR", "community":"AG", "longitude":76.0,
                  "latitude":41.4333333333, "start":"20150101", "end":"20251231", "format":"JSON"}
        r = get(endpoint, params=params)
        doc = r.json()
        out = root / "data/raw/nasa_power/naryn_city_2015_2025_daily.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        record(log, "NASA POWER daily Naryn city", "SUCCESS", "Saved daily T2M/PRECTOTCORR point response", request_url=r.url, file=str(out))
    except Exception as exc:
        record(log, "NASA POWER daily Naryn city", "FAILED", str(exc))

    # Geofabrik country package: source is current, but clip only after successful boundary retrieval.
    try:
        url = "https://download.geofabrik.de/asia/kyrgyzstan/kyrgyzstan-latest-free.gpkg.zip"
        dest = root / "data/raw/osm/kyrgyzstan-latest-free.gpkg.zip"
        download(url, dest)
        extracted = extract_geopackage_from_zip(dest, root / "data/raw/osm/extracted")
        record(log, "Geofabrik Kyrgyzstan GeoPackage", "SUCCESS", "Downloaded archive and extracted the GeoPackage for layer inspection; spatial clipping still requires Naryn boundary", url=url, archive_bytes=dest.stat().st_size, geopackage=str(extracted), geopackage_bytes=extracted.stat().st_size)
    except Exception as exc:
        record(log, "Geofabrik Kyrgyzstan GeoPackage", "FAILED", str(exc))

    # KDC capabilities: keep XML only if an actual capabilities document is returned.
    for label, url in [
        ("WMS", "https://kyrgyzstan.sibelius-datacube.org:5000/wms?service=WMS&request=GetCapabilities&version=1.3.0"),
        ("WCS", "https://kyrgyzstan.sibelius-datacube.org:5000/wcs?service=WCS&request=GetCapabilities&version=1.0.0")
    ]:
        try:
            r = get(url)
            if b"Capabilities" not in r.content[:200000]:
                raise RuntimeError("Response did not look like an OGC Capabilities XML document")
            dest = root / "data/raw/kdc" / f"{label.lower()}_GetCapabilities.xml"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            record(log, f"Kyrgyz Data Cube {label} GetCapabilities", "SUCCESS", "Saved OGC capabilities XML", url=url)
        except Exception as exc:
            record(log, f"Kyrgyz Data Cube {label} GetCapabilities", "FAILED", str(exc), url=url)

    # Wikidata: prefer the established v1 endpoint after the v2 endpoint returned
    # HTTP 403 in the user's run. Official migration guidance says v1 and v2
    # coexist through June 2027; record the endpoint that actually succeeds.
    query = "SELECT ?item ?itemLabel ?coordinate WHERE {\n  VALUES ?item { wd:Q486375 wd:Q311946 }\n  OPTIONAL { ?item wdt:P625 ?coordinate. }\n  SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }\n}"
    wikidata_errors = []
    for endpoint in ("https://query.wikidata.org/sparql", "https://query-next.wikidata.org/sparql"):
        try:
            r = requests.post(endpoint, data={"query": query}, headers={**HEADERS, "Accept": "application/sparql-results+json"}, timeout=TIMEOUT)
            r.raise_for_status()
            out = root / "data/raw/wikidata/naryn_entities_sparql.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"endpoint": endpoint, "query": query, "retrieved_utc": now(), "results": r.json()}, indent=2), encoding="utf-8")
            record(log, "Wikidata Naryn entity verification query", "SUCCESS", "Saved SPARQL JSON and query provenance", endpoint=endpoint)
            break
        except Exception as exc:
            wikidata_errors.append(f"{endpoint}: {exc}")
    else:
        record(log, "Wikidata Naryn entity verification query", "FAILED", "Both documented endpoints failed: " + " | ".join(wikidata_errors), endpoints=["https://query.wikidata.org/sparql", "https://query-next.wikidata.org/sparql"])


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.exit(run(args.root.resolve()))
