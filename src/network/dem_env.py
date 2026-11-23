"""Honest environmental samplers for Copernicus DEM, ESA WorldCover, and OSM layers.

Strict provenance guarantee:
- Elevation and slope come ONLY from a local Copernicus DEM GeoTIFF when present.
  If no local DEM exists, elevation_m and slope_deg return None with
  elevation_status='unavailable_no_local_dem'. Values are never invented.
- Land-cover classes come ONLY from local ESA WorldCover 2021 GeoTIFFs when present.
- OSM road/water/settlement geometries come ONLY from the Naryn-clipped GeoPackage.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

try:
    import rasterio
    from rasterio.warp import transform as rio_transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

WORLDCOVER_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


@dataclass
class EnvironmentInventory:
    root: Path
    dem_path: Path | None
    slope_path: Path | None
    dem_source: str
    worldcover_paths: list[Path]
    osm_gpkg_path: Path | None
    boundary_path: Path | None
    locations_path: Path | None
    climate_daily_path: Path | None


def discover_environment(root: Path) -> EnvironmentInventory:
    root = Path(root).resolve()
    dem_dir = root / "data/processed/rasters/dem"
    raw_dem_dir = root / "data/raw/dem"
    dem_candidates = [
        dem_dir / "naryn_dem_glo30_clip.tif",
        dem_dir / "naryn_dem_glo90_clip.tif",
        dem_dir / "naryn_dem.tif",
    ]
    if dem_dir.exists():
        dem_candidates.extend(sorted(dem_dir.glob("*dem*.tif")))
    if raw_dem_dir.exists():
        dem_candidates.extend(sorted(raw_dem_dir.glob("*.tif")))
    dem_path = next((p for p in dem_candidates if p.exists() and p.is_file() and p.stat().st_size > 0), None)

    slope_path = dem_dir / "naryn_dem_slope_deg.tif"
    if not (slope_path.exists() and slope_path.stat().st_size > 0):
        slope_path = None

    dem_source = "unavailable_no_local_dem"
    if dem_path is not None:
        meta_path = dem_dir / "naryn_dem_metadata.json"
        if meta_path.exists():
            try:
                dem_source = str(json.loads(meta_path.read_text(encoding="utf-8")).get("product", "Copernicus DEM"))
            except Exception:
                dem_source = "Copernicus DEM GLO-90" if "90" in dem_path.name.lower() else "Copernicus DEM GLO-30"
        else:
            dem_source = "Copernicus DEM GLO-90" if "90" in dem_path.name.lower() else "Copernicus DEM GLO-30"

    wc_dir = root / "data/processed/rasters/worldcover"
    wc_paths = sorted(p for p in wc_dir.glob("*.tif") if p.stat().st_size > 0) if wc_dir.exists() else []

    osm_path = root / "data/processed/vectors/osm_naryn_context.gpkg"
    if not (osm_path.exists() and osm_path.stat().st_size > 0):
        osm_path = None

    boundary_candidates = [
        root / "data/processed/vectors/naryn_adm1_clean.geojson",
        root / "data/raw/boundaries/naryn_adm1.geojson",
    ]
    boundary_path = next((p for p in boundary_candidates if p.exists() and p.stat().st_size > 0), None)

    loc_path = root / "data/processed/locations.csv"
    if not (loc_path.exists() and loc_path.stat().st_size > 0):
        loc_path = None

    clim_path = root / "data/processed/climate/nasa_power_daily.csv"
    if not (clim_path.exists() and clim_path.stat().st_size > 0):
        clim_path = None

    return EnvironmentInventory(
        root=root,
        dem_path=dem_path,
        slope_path=slope_path,
        dem_source=dem_source,
        worldcover_paths=wc_paths,
        osm_gpkg_path=osm_path,
        boundary_path=boundary_path,
        locations_path=loc_path,
        climate_daily_path=clim_path,
    )


def sample_dem_strict(
    lons: Sequence[float],
    lats: Sequence[float],
    env: EnvironmentInventory,
) -> tuple[list[float | None], list[float | None], str, str]:
    """Sample elevation (m) and slope (deg) strictly from local Copernicus DEM.

    Returns (elevations, slopes, elevation_status, elevation_source).
    When no local DEM or rasterio is available, returns lists of None and
    elevation_status='unavailable_no_local_dem'.
    """
    n = len(lons)
    if n == 0:
        return [], [], "unavailable_no_local_dem", "none"
    if env.dem_path is None or not HAS_RASTERIO:
        return [None] * n, [None] * n, "unavailable_no_local_dem", "unavailable_no_local_dem"

    lons_arr = np.asarray(lons, dtype=float)
    lats_arr = np.asarray(lats, dtype=float)
    with rasterio.open(env.dem_path) as src:
        xs, ys = lons_arr.tolist(), lats_arr.tolist()
        if src.crs is not None and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            xs, ys = rio_transform("EPSG:4326", src.crs, xs, ys)
        raw_samples = list(src.sample(list(zip(xs, ys)), indexes=1, masked=True))
        elev_arr = np.array([
            float(s[0]) if not np.ma.is_masked(s[0]) and np.isfinite(s[0]) and (src.nodata is None or s[0] != src.nodata) else np.nan
            for s in raw_samples
        ], dtype=float)

        if env.slope_path is not None:
            with rasterio.open(env.slope_path) as s_src:
                s_raw = list(s_src.sample(list(zip(xs, ys)), indexes=1, masked=True))
                slope_arr = np.array([
                    float(s[0]) if not np.ma.is_masked(s[0]) and np.isfinite(s[0]) and (s_src.nodata is None or s[0] != s_src.nodata) else np.nan
                    for s in s_raw
                ], dtype=float)
        else:
            d = 0.0005  # ~45-55m finite difference on DEM
            e_dx = np.array([
                float(s[0]) if not np.ma.is_masked(s[0]) and np.isfinite(s[0]) else np.nan
                for s in src.sample(list(zip((lons_arr + d).tolist(), lats_arr.tolist())), indexes=1, masked=True)
            ], dtype=float)
            e_dy = np.array([
                float(s[0]) if not np.ma.is_masked(s[0]) and np.isfinite(s[0]) else np.nan
                for s in src.sample(list(zip(lons_arr.tolist(), (lats_arr + d).tolist())), indexes=1, masked=True)
            ], dtype=float)
            dx_m = d * 111320.0 * np.cos(np.radians(lats_arr))
            dy_m = d * 110540.0
            grad = np.hypot((e_dx - elev_arr) / np.maximum(dx_m, 1.0), (e_dy - elev_arr) / dy_m)
            slope_arr = np.degrees(np.arctan(grad))

    elevs = [round(float(v), 1) if np.isfinite(v) else None for v in elev_arr]
    slopes = [round(float(np.clip(v, 0.0, 70.0)), 2) if np.isfinite(v) else None for v in slope_arr]
    return elevs, slopes, "dem_derived", env.dem_source


def sample_worldcover_strict(
    lons: Sequence[float],
    lats: Sequence[float],
    env: EnvironmentInventory,
) -> tuple[list[int | None], list[str], str]:
    """Sample ESA WorldCover 2021 land-cover class strictly from local GeoTIFFs."""
    n = len(lons)
    if n == 0:
        return [], [], "unavailable_no_local_worldcover"
    if not env.worldcover_paths or not HAS_RASTERIO:
        return [None] * n, ["unavailable_no_local_worldcover"] * n, "unavailable_no_local_worldcover"

    codes: list[int | None] = [None] * n
    labels: list[str] = ["outside_clipped_tile"] * n
    for rpath in env.worldcover_paths:
        with rasterio.open(rpath) as src:
            b = src.bounds
            for i, (x, y) in enumerate(zip(lons, lats)):
                if codes[i] is not None:
                    continue
                if b.left <= x <= b.right and b.bottom <= y <= b.top:
                    val = list(src.sample([(x, y)], indexes=1, masked=True))[0][0]
                    if not np.ma.is_masked(val) and (src.nodata is None or val != src.nodata):
                        c = int(val)
                        codes[i] = c
                        labels[i] = WORLDCOVER_LABELS.get(c, f"WorldCover class {c}")
    return codes, labels, "ESA WorldCover 2021 v200"


def compute_line_metrics(
    line_wgs84: LineString,
    env: EnvironmentInventory,
    *,
    metric_crs: str = "EPSG:32643",
    num_samples: int = 30,
) -> dict[str, Any]:
    """Calculate exact metric distance (m) and DEM elevation/slope profile for a route."""
    line_m = gpd.GeoSeries([line_wgs84], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    distance_m = round(float(line_m.length), 1)
    fracs = np.linspace(0.0, 1.0, max(num_samples, 5))
    pts = [line_wgs84.interpolate(f, normalized=True) for f in fracs]
    lons = [float(p.x) for p in pts]
    lats = [float(p.y) for p in pts]
    elevs, slopes, elev_status, elev_source = sample_dem_strict(lons, lats, env)

    valid_elevs = [e for e in elevs if e is not None]
    valid_slopes = [s for s in slopes if s is not None]
    if len(valid_elevs) >= 2:
        diffs = np.diff(np.asarray(valid_elevs, dtype=float))
        gain = round(float(np.sum(diffs[diffs > 0])), 1)
        loss = round(float(np.sum(-diffs[diffs < 0])), 1)
        start_e, end_e = valid_elevs[0], valid_elevs[-1]
        min_e, max_e = min(valid_elevs), max(valid_elevs)
    else:
        gain = loss = start_e = end_e = min_e = max_e = None

    avg_slope = round(float(np.mean(valid_slopes)), 2) if valid_slopes else None
    max_slope = round(float(np.max(valid_slopes)), 2) if valid_slopes else None

    profile = [
        {
            "distance_m": round(float(f * distance_m), 1),
            "lon": round(x, 5),
            "lat": round(y, 5),
            "elevation_m": z,
            "slope_deg": s,
        }
        for f, x, y, z, s in zip(fracs, lons, lats, elevs, slopes)
    ]

    return {
        "distance_m": distance_m,
        "start_elevation_m": start_e,
        "end_elevation_m": end_e,
        "min_elevation_m": min_e,
        "max_elevation_m": max_e,
        "elevation_gain_m": gain,
        "elevation_loss_m": loss,
        "average_slope": avg_slope,
        "maximum_slope": max_slope,
        "elevation_status": elev_status,
        "elevation_source": elev_source,
        "elevation_profile_json": json.dumps(profile),
    }


def read_geojson_safe(path: Path) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(path)
        if gdf.empty and gdf.geometry.name not in gdf.columns:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return gdf
    except Exception:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        crs = "EPSG:4326"
        if isinstance(doc.get("crs"), dict):
            name = str(doc["crs"].get("properties", {}).get("name", ""))
            if "32643" in name:
                crs = "EPSG:32643"
        feats = doc.get("features", [])
        if not feats:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
        return gpd.GeoDataFrame.from_features(feats, crs=crs)


def write_geojson_safe(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
        return
    path.write_text(gdf.to_crs("EPSG:4326").to_json(), encoding="utf-8")
