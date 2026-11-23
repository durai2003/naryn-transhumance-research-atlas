"""Generate geographically coherent synthetic candidate grazing locations inside Naryn ADM1.

Every generated record is explicitly labelled `DEMO-PASTURE-XXX` with
`is_synthetic=True` and `provenance='synthetic_demo'`. Candidate locations are
NOT officially recognised pasture parcels or grazing-rights polygons.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from src.atlas.terrain import sample_elevation_and_slope

try:
    import rasterio
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
    100: "Moss and lichen",
}

# Geographically grounded sub-regional candidate anchors inside Naryn Oblast
# (valley wintering zones, foothill spring/autumn zones, high-alpine summer zones).
# Exact coordinates are perturbed deterministically by seed and strictly validated
# to stay inside the authoritative Naryn ADM1 boundary polygon.
_ANCHOR_CATALOGUE = [
    # Wintering / lower valley candidates (6)
    ("Naryn Valley West Candidate", 75.55, 41.42, "winter", "valley_winter", 0.88),
    ("Naryn Valley Central Candidate", 76.08, 41.44, "winter", "valley_winter", 0.92),
    ("Kochkor Basin Candidate", 75.72, 42.19, "winter", "valley_winter", 0.85),
    ("Jumgal Valley Candidate", 74.48, 41.91, "winter", "valley_winter", 0.82),
    ("Ak-Talaa Valley Candidate", 74.92, 41.39, "winter", "valley_winter", 0.84),
    ("At-Bashi Valley Candidate", 75.82, 41.16, "winter", "valley_winter", 0.89),
    # Spring / Autumn foothill transition candidates (6)
    ("Kyzyl-Oi Foothill Candidate", 74.32, 41.73, "spring_autumn", "foothill_transition", 0.78),
    ("Kulanak Foothill Candidate", 75.46, 41.32, "spring_autumn", "foothill_transition", 0.80),
    ("Kara-Kujur Transition Candidate", 76.18, 41.92, "spring_autumn", "foothill_transition", 0.83),
    ("On-Archa Foothill Candidate", 76.12, 41.61, "spring_autumn", "foothill_transition", 0.81),
    ("Bash-Kaindy Transition Candidate", 75.95, 41.06, "spring_autumn", "foothill_transition", 0.79),
    ("Kazarman Approach Foothill Candidate", 74.55, 41.29, "spring_autumn", "foothill_transition", 0.76),
    # High-alpine summer jailoo candidates (6)
    ("Son-Kul North Alpine Candidate", 75.18, 41.92, "summer", "high_alpine", 0.94),
    ("Son-Kul South Alpine Candidate", 75.14, 41.72, "summer", "high_alpine", 0.91),
    ("Eki-Naryn Alpine Candidate", 76.58, 41.48, "summer", "high_alpine", 0.89),
    ("Solton-Sary High Plateau Candidate", 76.02, 41.79, "summer", "high_alpine", 0.82),
    ("Aksai High Valley Candidate", 76.62, 40.92, "summer", "high_alpine", 0.77),
    ("At-Bashi High Ridge Candidate", 75.55, 40.89, "summer", "high_alpine", 0.80),
]


def _sample_worldcover(lons: list[float], lats: list[float], root: Path) -> tuple[list[int | None], list[str]]:
    wc_dir = root / "data/processed/rasters/worldcover"
    rasters = sorted(wc_dir.glob("*.tif")) if wc_dir.exists() else []
    codes: list[int | None] = [None] * len(lons)
    labels: list[str] = ["Grassland (proxy — WorldCover tile not present)"] * len(lons)
    if not rasters or not HAS_RASTERIO:
        # Provide deterministic transparent landcover proxy label without claiming ESA raster read
        for i in range(len(lons)):
            codes[i] = 30
            labels[i] = "Grassland (demo proxy; run on Mac with WorldCover clips for native class)"
        return codes, labels

    for rpath in rasters:
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
    for i in range(len(lons)):
        if codes[i] is None:
            codes[i] = 30
            labels[i] = "Grassland (fallback outside clipped tile)"
    return codes, labels


def generate_candidate_pastures(
    boundary_gdf: gpd.GeoDataFrame,
    root: Path,
    *,
    seed: int = 42,
    count: int = 18,
    simulation_version: str = "2.0.0",
    disclaimer: str = "",
) -> gpd.GeoDataFrame:
    """Generate 15-20 synthetic candidate grazing locations strictly inside Naryn ADM1."""
    boundary_wgs = boundary_gdf.to_crs("EPSG:4326")
    region = unary_union([g if g.is_valid else g.buffer(0) for g in boundary_wgs.geometry if g is not None and not g.is_empty])
    if region.is_empty:
        raise ValueError("Naryn boundary geometry is empty")

    rng = np.random.default_rng(seed)
    anchors = _ANCHOR_CATALOGUE[: max(15, min(count, len(_ANCHOR_CATALOGUE)))]

    coords: list[tuple[float, float]] = []
    meta_rows: list[dict[str, Any]] = []

    for idx, (label, base_lon, base_lat, seasonal_role, elev_band, water_base) in enumerate(anchors, start=1):
        pt = Point(base_lon, base_lat)
        # Small deterministic jitter (up to ~3 km) while guaranteeing strict containment inside Naryn ADM1
        chosen = None
        for _ in range(40):
            jlon = float(base_lon + rng.uniform(-0.035, 0.035))
            jlat = float(base_lat + rng.uniform(-0.025, 0.025))
            cand = Point(jlon, jlat)
            if region.contains(cand):
                chosen = cand
                break
        if chosen is None:
            chosen = pt if region.contains(pt) else region.representative_point()

        coords.append((float(chosen.x), float(chosen.y)))
        meta_rows.append({
            "pasture_id": f"DEMO-PASTURE-{idx:03d}",
            "pasture_name": label,
            "seasonal_role": seasonal_role,
            "elevation_band": elev_band,
            "water_access_indicator": round(float(np.clip(water_base + rng.uniform(-0.06, 0.05), 0.35, 0.98)), 3),
        })

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    elevs, slopes, elev_source = sample_elevation_and_slope(lons, lats, root)
    wc_codes, wc_labels = _sample_worldcover(lons, lats, root)

    records = []
    for i, base in enumerate(meta_rows):
        elev_m = float(elevs[i])
        slope_deg = float(slopes[i])
        band = base["elevation_band"]
        # Ensure elevation bands remain clearly differentiated if proxy was flat locally
        if elev_source == "synthetic_relief_proxy_pending_dem":
            if band == "valley_winter":
                elev_m = float(np.clip(elev_m, 1820.0, 2320.0))
            elif band == "foothill_transition":
                elev_m = float(np.clip(elev_m, 2380.0, 2820.0))
            else:
                elev_m = float(np.clip(elev_m, 2920.0, 3540.0))

        # Vegetation indicator grounded in land-cover class (transparent demonstration index 0..1)
        wc_code = wc_codes[i]
        veg_base = {30: 0.82, 20: 0.68, 40: 0.60, 10: 0.45, 90: 0.74, 60: 0.32, 100: 0.58}.get(wc_code or 30, 0.55)
        veg_indicator = round(float(np.clip(veg_base + rng.uniform(-0.07, 0.08), 0.25, 0.95)), 3)

        records.append({
            "pasture_id": base["pasture_id"],
            "pasture_name": base["pasture_name"],
            "latitude": round(lats[i], 5),
            "longitude": round(lons[i], 5),
            "elevation_m": round(elev_m, 1),
            "slope_degrees": round(slope_deg, 2),
            "elevation_band": band,
            "landcover_code": int(wc_code) if wc_code is not None else 30,
            "landcover_class": wc_labels[i],
            "water_access_indicator": base["water_access_indicator"],
            "seasonal_suitability": base["seasonal_role"],
            "vegetation_indicator": veg_indicator,
            "elevation_source": elev_source,
            "pasture_category": "synthetic_candidate_grazing_location",
            "origin_class": "synthetic",
            "simulation_version": simulation_version,
            "random_seed": int(seed),
            "is_synthetic": True,
            "provenance": "synthetic_demo",
            "synthetic_disclaimer": disclaimer,
            "geometry": Point(lons[i], lats[i]),
        })

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
