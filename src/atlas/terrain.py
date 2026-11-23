"""Terrain DEM access, elevation/slope sampling, and route elevation profiles.

Uses downloaded Copernicus DEM GLO-30 (or GLO-90 fallback) when present under
`data/processed/rasters/dem/`. When no local DEM raster has been downloaded yet,
provides a clearly flagged analytical Tian Shan relief proxy solely so synthetic
demonstrations can compute relative mountain gradients without pretending the
values are measured Copernicus DEM samples.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

try:
    import rasterio
    from rasterio.warp import transform as rio_transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@dataclass
class TerrainStatus:
    has_local_dem: bool
    dem_path: Path | None
    slope_path: Path | None
    source_name: str
    resolution_desc: str
    crs: str
    vertical_datum: str
    licence: str
    metadata: dict[str, Any]


def find_local_dem(root: Path) -> TerrainStatus:
    root = Path(root).resolve()
    dem_dir = root / "data/processed/rasters/dem"
    raw_dem_dir = root / "data/raw/dem"
    meta_path = dem_dir / "naryn_dem_metadata.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    candidates = [
        dem_dir / "naryn_dem_glo30_clip.tif",
        dem_dir / "naryn_dem_glo90_clip.tif",
        dem_dir / "naryn_dem.tif",
    ]
    if dem_dir.exists():
        candidates.extend(sorted(dem_dir.glob("*dem*.tif")))
    if raw_dem_dir.exists():
        candidates.extend(sorted(raw_dem_dir.glob("*.tif")))

    dem_path = next((p for p in candidates if p.exists() and p.is_file() and p.stat().st_size > 0), None)
    slope_candidates = [dem_dir / "naryn_dem_slope_deg.tif"]
    slope_path = next((p for p in slope_candidates if p.exists() and p.is_file() and p.stat().st_size > 0), None)

    if dem_path is not None:
        is_glo90 = "90" in dem_path.name.lower() or "glo-90" in str(meta.get("product", "")).lower()
        return TerrainStatus(
            has_local_dem=True,
            dem_path=dem_path,
            slope_path=slope_path,
            source_name=meta.get("product", "Copernicus DEM GLO-90" if is_glo90 else "Copernicus DEM GLO-30"),
            resolution_desc=meta.get("resolution_desc", "3 arcsec (~90 m)" if is_glo90 else "1 arcsec (~30 m)"),
            crs=meta.get("crs", "EPSG:4326"),
            vertical_datum=meta.get("vertical_datum", "EGM2008"),
            licence=meta.get("licence", "Copernicus DEM Licence (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018)"),
            metadata=meta,
        )

    return TerrainStatus(
        has_local_dem=False,
        dem_path=None,
        slope_path=None,
        source_name="Analytical Tian Shan relief proxy (local Copernicus DEM not yet downloaded)",
        resolution_desc="Synthetic analytical proxy — run scripts/acquire_dem.py for real 30m DEM",
        crs="EPSG:4326",
        vertical_datum="Approximate MSL proxy (not measured DEM)",
        licence="Synthetic demonstration proxy",
        metadata=meta,
    )


def _tian_shan_proxy_elevation(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Deterministic topographic proxy reflecting Naryn's major valleys and ranges.

    Strictly a fallback when local Copernicus DEM tiles are absent. Models:
    - Naryn River valley corridor (~41.42 N, ~1950-2200 m)
    - Kochkor & Jumgal basins in the NW (~1850-2200 m)
    - Son-Kul alpine plateau (~41.84 N, 75.15 E, ~3016 m)
    - At-Bashi valley (~41.17 N, 75.80 E, ~2100 m)
    - Aksai / Kokshaal-Too high southern alpine ridges (~40.5-41.0 N, 3200-4400 m)
    - Naryn-Too & Terskey ridges (~3300-4100 m)
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    # Regional eastward & southward alpine rise across Naryn Oblast
    base = 2450.0 + 185.0 * (lon - 75.0) - 210.0 * (lat - 41.35)

    # Major alpine massifs (Gaussian ridges)
    massifs = [
        (76.35, 41.28, 0.55, 0.14, 1180.0),  # Naryn-Too ridge
        (75.50, 40.78, 0.70, 0.16, 1350.0),  # At-Bashi range
        (76.95, 40.95, 0.60, 0.22, 1320.0),  # Aksai / Kakshaal-Too highlands
        (75.15, 41.84, 0.38, 0.18, 620.0),   # Son-Kul high basin rim
        (74.45, 41.60, 0.42, 0.16, 980.0),   # Moldo-Too ridge
        (76.20, 41.92, 0.55, 0.16, 1050.0),  # Terskey western extension
    ]
    elev = base.copy()
    for mx, my, sx, sy, amp in massifs:
        elev += amp * np.exp(-0.5 * (((lon - mx) / sx) ** 2 + ((lat - my) / sy) ** 2))

    # Major river/intermontane valleys (carved troughs)
    valleys = [
        (75.95, 41.43, 0.95, 0.08, 680.0),   # Naryn River valley
        (75.75, 42.20, 0.45, 0.10, 720.0),   # Kochkor basin
        (74.45, 41.92, 0.42, 0.09, 690.0),   # Jumgal valley
        (75.80, 41.15, 0.52, 0.08, 640.0),   # At-Bashi valley
        (74.90, 41.40, 0.50, 0.09, 610.0),   # Ak-Talaa / Baetov valley
    ]
    for vx, vy, sx, sy, depth in valleys:
        elev -= depth * np.exp(-0.5 * (((lon - vx) / sx) ** 2 + ((lat - vy) / sy) ** 2))

    # Fine multi-frequency ridge modulation for realistic local relief
    elev += 140.0 * np.sin(lon * 9.5 + lat * 4.2) * np.cos(lat * 11.0 - lon * 3.1)
    elev += 75.0 * np.sin(lon * 21.0 - lat * 17.5)
    return np.clip(elev, 1620.0, 4850.0)


def sample_elevation_and_slope(
    lons: Sequence[float],
    lats: Sequence[float],
    root: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (elevation_m, slope_deg, elevation_source) for WGS84 coordinates."""
    lons_arr = np.asarray(lons, dtype=float)
    lats_arr = np.asarray(lats, dtype=float)
    status = find_local_dem(root)

    if status.has_local_dem and HAS_RASTERIO and status.dem_path is not None:
        with rasterio.open(status.dem_path) as src:
            xs, ys = lons_arr.tolist(), lats_arr.tolist()
            if str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                xs, ys = rio_transform("EPSG:4326", src.crs, xs, ys)
            samples = list(src.sample(list(zip(xs, ys)), indexes=1, masked=True))
            elev = np.array([
                float(s[0]) if not np.ma.is_masked(s[0]) and np.isfinite(s[0]) else np.nan
                for s in samples
            ], dtype=float)

        if status.slope_path is not None:
            with rasterio.open(status.slope_path) as s_src:
                s_samples = list(s_src.sample(list(zip(xs, ys)), indexes=1, masked=True))
                slope = np.array([
                    float(s[0]) if not np.ma.is_masked(s[0]) and np.isfinite(s[0]) else np.nan
                    for s in s_samples
                ], dtype=float)
        else:
            # Estimate finite-difference slope from DEM around each point (~90 m offset)
            d = 0.0009
            with rasterio.open(status.dem_path) as src:
                e_dx = np.array([float(s[0]) if not np.ma.is_masked(s[0]) else np.nan for s in src.sample(list(zip((lons_arr + d).tolist(), lats_arr.tolist())), indexes=1, masked=True)])
                e_dy = np.array([float(s[0]) if not np.ma.is_masked(s[0]) else np.nan for s in src.sample(list(zip(lons_arr.tolist(), (lats_arr + d).tolist())), indexes=1, masked=True)])
            dx_m = d * 111320.0 * np.cos(np.radians(lats_arr))
            dy_m = d * 110540.0
            grad = np.hypot((e_dx - elev) / np.maximum(dx_m, 1.0), (e_dy - elev) / dy_m)
            slope = np.degrees(np.arctan(np.nan_to_num(grad, nan=0.08)))

        # Fill any isolated nodata pixels transparently with proxy only if needed
        missing = ~np.isfinite(elev)
        if missing.any():
            elev[missing] = _tian_shan_proxy_elevation(lons_arr[missing], lats_arr[missing])
            slope[missing] = 8.0
        return elev, np.clip(np.nan_to_num(slope, nan=6.0), 0.0, 65.0), status.source_name

    # Proxy fallback
    elev = _tian_shan_proxy_elevation(lons_arr, lats_arr)
    d = 0.0025
    e_dx = _tian_shan_proxy_elevation(lons_arr + d, lats_arr)
    e_dy = _tian_shan_proxy_elevation(lons_arr, lats_arr + d)
    dx_m = d * 111320.0 * np.cos(np.radians(lats_arr))
    dy_m = d * 110540.0
    grad = np.hypot((e_dx - elev) / np.maximum(dx_m, 1.0), (e_dy - elev) / dy_m)
    slope = np.clip(np.degrees(np.arctan(grad)), 0.5, 45.0)
    return elev, slope, "synthetic_relief_proxy_pending_dem"


def compute_route_elevation_profile(
    line_wgs84: LineString,
    root: Path,
    *,
    num_samples: int = 40,
    metric_crs: str = "EPSG:32643",
) -> dict[str, Any]:
    """Sample elevation and slope evenly along a route geometry."""
    series_wgs = gpd.GeoSeries([line_wgs84], crs="EPSG:4326")
    line_m = series_wgs.to_crs(metric_crs).iloc[0]
    total_length_m = float(line_m.length)
    fracs = np.linspace(0.0, 1.0, max(num_samples, 5))
    pts_wgs = [line_wgs84.interpolate(f, normalized=True) for f in fracs]
    lons = [float(p.x) for p in pts_wgs]
    lats = [float(p.y) for p in pts_wgs]
    elevs, slopes, source = sample_elevation_and_slope(lons, lats, root)
    diffs = np.diff(elevs)
    gain = float(np.sum(diffs[diffs > 0])) if diffs.size else 0.0
    loss = float(np.sum(-diffs[diffs < 0])) if diffs.size else 0.0
    distances_km = (fracs * total_length_m / 1000.0).tolist()
    return {
        "distance_km": round(total_length_m / 1000.0, 2),
        "start_elevation_m": round(float(elevs[0]), 1),
        "end_elevation_m": round(float(elevs[-1]), 1),
        "min_elevation_m": round(float(np.min(elevs)), 1),
        "max_elevation_m": round(float(np.max(elevs)), 1),
        "elevation_gain_m": round(gain, 1),
        "elevation_loss_m": round(loss, 1),
        "avg_slope_deg": round(float(np.mean(slopes)), 2),
        "max_slope_deg": round(float(np.max(slopes)), 2),
        "elevation_source": source,
        "profile": [
            {
                "distance_km": round(d, 2),
                "lon": round(x, 5),
                "lat": round(y, 5),
                "elevation_m": round(float(z), 1),
                "slope_deg": round(float(s), 1),
            }
            for d, x, y, z, s in zip(distances_km, lons, lats, elevs, slopes)
        ],
    }


def export_local_dem_grid(
    root: Path,
    *,
    width: int = 240,
    height: int = 144,
) -> dict[str, Any] | None:
    """Export a compact base64 uint16 elevation grid from local Copernicus DEM for CesiumJS.

    Encoding: elevation_m = raw_uint16 * 0.25 - 500.0 (0.25 m vertical resolution,
    covers -500 m to 15,883 m). Value 0 marks nodata/outside-DEM so Cesium seamlessly
    falls back to global AWS Terrarium 3D elevation outside the clipped DEM footprint.
    """
    import base64

    status = find_local_dem(root)
    if not status.has_local_dem or status.dem_path is None or not HAS_RASTERIO:
        return None
    try:
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject

        with rasterio.open(status.dem_path) as src:
            if src.crs is None:
                return None
            dst_transform, dst_w, dst_h = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            from affine import Affine
            scale_x = dst_w / float(width)
            scale_y = dst_h / float(height)
            out_transform = dst_transform * Affine.scale(scale_x, scale_y)
            dest = np.full((height, width), -9999.0, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=out_transform,
                dst_crs="EPSG:4326",
                dst_nodata=-9999.0,
                resampling=Resampling.bilinear,
            )
            west = float(out_transform.c)
            north = float(out_transform.f)
            east = float(west + out_transform.a * width)
            south = float(north + out_transform.e * height)

            valid = np.isfinite(dest) & (dest > -450.0) & (dest < 9000.0)
            encoded = np.zeros((height, width), dtype="<u2")
            encoded[valid] = np.clip(np.round((dest[valid] + 500.0) / 0.25), 1, 65535).astype("<u2")
            b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            return {
                "source": status.source_name,
                "resolution": status.resolution_desc,
                "width": width,
                "height": height,
                "bounds": [round(west, 5), round(south, 5), round(east, 5), round(north, 5)],
                "offset_m": -500.0,
                "scale_m": 0.25,
                "data_b64_u16": b64,
            }
    except Exception:
        return None
