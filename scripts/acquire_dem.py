#!/usr/bin/env python3
"""Download and preprocess Copernicus DEM GLO-30 (with GLO-90 fallback) for Naryn Region.

Sources investigated & supported:
1. Copernicus DEM GLO-30 public AWS Open Data Cloud-Optimized GeoTIFFs (anonymous HTTPS):
   https://copernicus-dem-30m.s3.amazonaws.com/
   (1 arc-second ~30m, EPSG:4326, EGM2008 vertical datum, Copernicus DEM Licence)
2. Fallback: Copernicus DEM GLO-90 public AWS Open Data COGs:
   https://copernicus-dem-90m.s3.amazonaws.com/
   (3 arc-second ~90m, EPSG:4326, EGM2008 vertical datum)

Run from repository root:
    python scripts/acquire_dem.py --root .
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.atlas.data_access import find_boundary_path

HEADERS = {"User-Agent": "MiniTranshumanceAtlas/2.0 (educational geospatial research prototype)"}
TIMEOUT = (15, 120)
RETRIES = 3


def _tile_keys(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    minx, miny, maxx, maxy = bounds
    lon0, lon1 = math.floor(minx), math.floor(math.nextafter(maxx, -math.inf))
    lat0, lat1 = math.floor(miny), math.floor(math.nextafter(maxy, -math.inf))
    return [(lat, lon) for lat in range(lat0, lat1 + 1) for lon in range(lon0, lon1 + 1)]


def _copernicus_url(lat: int, lon: int, resolution_arcsec: int = 30) -> tuple[str, str]:
    ns = f"N{lat:02d}_00" if lat >= 0 else f"S{abs(lat):02d}_00"
    ew = f"E{lon:03d}_00" if lon >= 0 else f"W{abs(lon):03d}_00"
    if resolution_arcsec == 30:
        stem = f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM"
        bucket = "https://copernicus-dem-30m.s3.amazonaws.com"
    else:
        stem = f"Copernicus_DSM_COG_30_{ns}_{ew}_DEM"
        bucket = "https://copernicus-dem-90m.s3.amazonaws.com"
    return f"{bucket}/{stem}/{stem}.tif", f"{stem}.tif"


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            return dest.exists() and dest.stat().st_size > 0
        except Exception:
            if attempt + 1 < RETRIES:
                time.sleep(2 ** attempt)
    return False


def _derive_slope_and_hillshade(dem_path: Path, slope_path: Path, hillshade_path: Path) -> dict[str, Any]:
    import rasterio

    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata
        # Compute cell spacing in metres at mean latitude
        mean_lat = 0.5 * (src.bounds.bottom + src.bounds.top)
        dx_m = abs(src.transform.a) * 111320.0 * math.cos(math.radians(mean_lat))
        dy_m = abs(src.transform.e) * 110540.0
        filled = np.where(valid, arr, float(np.nanmedian(arr[valid])) if valid.any() else 2500.0)
        dz_dy, dz_dx = np.gradient(filled, dy_m, dx_m)
        slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
        slope_deg = np.degrees(slope_rad).astype("float32")
        slope_deg[~valid] = -9999.0

        # Multidirectional-inspired NW sun hillshade (azimuth 315 deg, altitude 38 deg)
        az = math.radians(315.0)
        alt = math.radians(38.0)
        aspect = np.arctan2(-dz_dx, dz_dy)
        hs = 255.0 * (
            math.sin(alt) * np.cos(slope_rad)
            + math.cos(alt) * np.sin(slope_rad) * np.cos(az - aspect)
        )
        hillshade = np.clip(hs, 1, 255).astype("uint8")
        hillshade[~valid] = 0

        slope_profile = src.profile.copy()
        slope_profile.update(dtype="float32", nodata=-9999.0, compress="DEFLATE")
        with rasterio.open(slope_path, "w", **slope_profile) as dst:
            dst.write(slope_deg, 1)

        hs_profile = src.profile.copy()
        hs_profile.update(dtype="uint8", nodata=0, compress="DEFLATE")
        with rasterio.open(hillshade_path, "w", **hs_profile) as dst:
            dst.write(hillshade, 1)

        return {
            "elevation_min_m": round(float(np.min(arr[valid])), 1) if valid.any() else None,
            "elevation_max_m": round(float(np.max(arr[valid])), 1) if valid.any() else None,
            "elevation_mean_m": round(float(np.mean(arr[valid])), 1) if valid.any() else None,
            "slope_mean_deg": round(float(np.mean(slope_deg[valid])), 2) if valid.any() else None,
            "slope_max_deg": round(float(np.max(slope_deg[valid])), 2) if valid.any() else None,
            "width": src.width,
            "height": src.height,
        }


def run(root: Path, *, prefer_glo90: bool = False) -> int:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.merge import merge as rio_merge

    root = root.resolve()
    boundary_path = find_boundary_path(root)
    if boundary_path is None:
        print("[BLOCKED] Authoritative Naryn ADM1 boundary not found. Run Stage 2/3 first.")
        return 1

    boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    # Add a small 0.15 deg buffer so terrain edges never clip abruptly inside the viewport
    buffered = boundary.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.15)
    bounds = tuple(float(v) for v in buffered.total_bounds)
    tiles = _tile_keys(bounds)
    print(f"Naryn DEM tile grid ({len(tiles)} 1x1 deg tiles): {tiles}")

    raw_dir = root / "data/raw/dem"
    proc_dir = root / "data/processed/rasters/dem"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    res_order = [90, 30] if prefer_glo90 else [30, 90]
    used_res = res_order[0]
    for lat, lon in tiles:
        success = False
        for res in res_order:
            url, filename = _copernicus_url(lat, lon, res)
            dest = raw_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                print(f"[ALREADY_PRESENT] {filename}")
                downloaded.append(dest)
                used_res = res
                success = True
                break
            print(f"[DOWNLOADING] {url}")
            if _download(url, dest):
                print(f"[SUCCESS] {filename} ({dest.stat().st_size:,} bytes)")
                downloaded.append(dest)
                used_res = res
                success = True
                break
        if not success:
            print(f"[WARNING] Could not retrieve Copernicus tile ({lat}, {lon})")

    if not downloaded:
        print("[FAILED] No Copernicus DEM tiles downloaded. Check network access.")
        return 1

    datasets = [rasterio.open(p) for p in downloaded]
    try:
        mosaic, out_transform = rio_merge(datasets)
        meta = datasets[0].meta.copy()
        meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=out_transform, compress="DEFLATE")
        mosaic_path = proc_dir / f"naryn_dem_glo{used_res}_mosaic_tmp.tif"
        with rasterio.open(mosaic_path, "w", **meta) as dst:
            dst.write(mosaic)
    finally:
        for ds in datasets:
            ds.close()

    clip_path = proc_dir / f"naryn_dem_glo{used_res}_clip.tif"
    with rasterio.open(mosaic_path) as src:
        shapes = [g.__geo_interface__ for g in buffered.geometry if g is not None and not g.is_empty]
        nodata = src.nodata if src.nodata is not None else -9999.0
        out_img, clip_transform = rio_mask(src, shapes, crop=True, nodata=nodata, filled=True)
        profile = src.profile.copy()
        profile.update(height=out_img.shape[1], width=out_img.shape[2], transform=clip_transform, nodata=nodata, compress="DEFLATE")
        with rasterio.open(clip_path, "w", **profile) as dst:
            dst.write(out_img)
    mosaic_path.unlink(missing_ok=True)

    slope_path = proc_dir / "naryn_dem_slope_deg.tif"
    hillshade_path = proc_dir / "naryn_dem_hillshade.tif"
    stats = _derive_slope_and_hillshade(clip_path, slope_path, hillshade_path)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "product": f"Copernicus DEM GLO-{used_res}",
        "resolution_desc": "1 arcsec (~30 m)" if used_res == 30 else "3 arcsec (~90 m)",
        "crs": "EPSG:4326",
        "vertical_datum": "EGM2008",
        "source_bucket": f"https://copernicus-dem-{used_res}m.s3.amazonaws.com/",
        "official_documentation": "https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM",
        "licence": "Copernicus DEM Licence (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018)",
        "clipped_dem_path": clip_path.relative_to(root).as_posix(),
        "slope_path": slope_path.relative_to(root).as_posix(),
        "hillshade_path": hillshade_path.relative_to(root).as_posix(),
        "tiles_used": [p.name for p in downloaded],
        "stats": stats,
    }
    (proc_dir / "naryn_dem_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--glo90", action="store_true", help="Prefer GLO-90 over GLO-30")
    args = parser.parse_args()
    raise SystemExit(run(args.root, prefer_glo90=args.glo90))
