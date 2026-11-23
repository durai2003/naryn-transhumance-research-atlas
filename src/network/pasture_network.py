"""Layer B static node generator: 40 candidate pastures, mountain passes, water points, resting points.

Geographically grounded across Naryn Oblast sub-regions (Jumgal, Kochkor, Ak-Talaa,
Naryn River Valley, On-Archa, Kara-Kujur, Son-Kul, Solton-Sary, At-Bashi, Aksai,
Eki-Naryn). Every candidate is strictly constrained inside the real Naryn ADM1
boundary and filtered against local Copernicus DEM elevation/slope and ESA
WorldCover grassland/shrubland masks whenever those rasters are present.

All synthetic candidates are explicitly labelled `is_synthetic=True` and
`provenance='synthetic_demo'`. Official pasture boundaries are never invented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from src.network.dem_env import (
    EnvironmentInventory,
    sample_dem_strict,
    sample_worldcover_strict,
)

# 40 geographically coherent sub-regional anchors inside Naryn ADM1:
# 10 winter valley zones, 10 spring foothill zones, 10 summer high-alpine jailoo zones, 10 autumn transition zones.
# Each entry: (subregion_zone, label, lon, lat, node_type)
_PASTURE_ANCHORS = [
    # 10 WINTER PASTURE CANDIDATES (low-altitude river valleys & intermontane basins)
    ("Z_JUMGAL", "Jumgal West Winter Candidate", 74.36, 41.90, "winter_pasture"),
    ("Z_JUMGAL", "Chaek Basin Winter Candidate", 74.56, 41.93, "winter_pasture"),
    ("Z_KOCHKOR", "Kochkor West Winter Candidate", 75.60, 42.22, "winter_pasture"),
    ("Z_KOCHKOR", "Kochkor East Winter Candidate", 75.86, 42.18, "winter_pasture"),
    ("Z_AKTALA", "Baetov Valley Winter Candidate", 74.94, 41.27, "winter_pasture"),
    ("Z_AKTALA", "Ak-Talaa River Winter Candidate", 75.15, 41.41, "winter_pasture"),
    ("Z_NARYN_W", "Kulanak Valley Winter Candidate", 75.56, 41.42, "winter_pasture"),
    ("Z_NARYN_C", "Naryn Valley Central Winter Candidate", 76.06, 41.44, "winter_pasture"),
    ("Z_ATBASHI", "At-Bashi West Winter Candidate", 75.70, 41.17, "winter_pasture"),
    ("Z_ATBASHI", "At-Bashi East Winter Candidate", 75.95, 41.20, "winter_pasture"),

    # 10 SPRING PASTURE CANDIDATES (sunny south-facing foothills & mid-elevation benches)
    ("Z_JUMGAL", "Kyzyl-Oi Foothill Spring Candidate", 74.26, 41.76, "spring_pasture"),
    ("Z_JUMGAL", "Min-Kush Approach Spring Candidate", 74.66, 41.74, "spring_pasture"),
    ("Z_KOCHKOR", "Sary-Bulak Foothill Spring Candidate", 75.78, 41.98, "spring_pasture"),
    ("Z_KOCHKOR", "Kara-Kujur West Spring Candidate", 76.12, 41.93, "spring_pasture"),
    ("Z_AKTALA", "Kazarman Ridge Spring Candidate", 74.52, 41.32, "spring_pasture"),
    ("Z_AKTALA", "Togolok-Moldo Foothill Spring Candidate", 75.12, 41.55, "spring_pasture"),
    ("Z_NARYN_W", "Emgekchil Approach Spring Candidate", 75.62, 41.58, "spring_pasture"),
    ("Z_NARYN_C", "On-Archa Foothill Spring Candidate", 76.08, 41.62, "spring_pasture"),
    ("Z_ATBASHI", "Kara-Suu Bench Spring Candidate", 75.56, 41.05, "spring_pasture"),
    ("Z_ATBASHI", "Bash-Kaindy Foothill Spring Candidate", 76.05, 41.07, "spring_pasture"),

    # 10 SUMMER PASTURE CANDIDATES (high-alpine jailoo plateaus & glacial valleys)
    ("Z_JUMGAL", "Suusamyr South Rim Summer Candidate", 74.16, 41.84, "summer_pasture"),
    ("Z_JUMGAL", "Son-Kul Northwest Alpine Candidate", 74.98, 41.91, "summer_pasture"),
    ("Z_KOCHKOR", "Son-Kul Northeast Alpine Candidate", 75.34, 41.90, "summer_pasture"),
    ("Z_KOCHKOR", "Solton-Sary High Plateau Summer Candidate", 76.04, 41.78, "summer_pasture"),
    ("Z_AKTALA", "Son-Kul Southwest Alpine Candidate", 74.96, 41.72, "summer_pasture"),
    ("Z_AKTALA", "Moldo-Too High Ridge Summer Candidate", 74.72, 41.56, "summer_pasture"),
    ("Z_NARYN_W", "Son-Kul Southeast Alpine Candidate", 75.38, 41.71, "summer_pasture"),
    ("Z_NARYN_C", "Eki-Naryn Upper Valley Summer Candidate", 76.58, 41.48, "summer_pasture"),
    ("Z_ATBASHI", "At-Bashi High Range Summer Candidate", 75.54, 40.89, "summer_pasture"),
    ("Z_ATBASHI", "Aksai High Basin Summer Candidate", 76.62, 40.92, "summer_pasture"),

    # 10 AUTUMN PASTURE CANDIDATES (sheltered mid-altitude descent slopes & stubble margins)
    ("Z_JUMGAL", "Jumgal South Bench Autumn Candidate", 74.42, 41.80, "autumn_pasture"),
    ("Z_JUMGAL", "Kyzyl-Jildiz Descent Autumn Candidate", 74.78, 41.84, "autumn_pasture"),
    ("Z_KOCHKOR", "Tuz-Bel Lower Slope Autumn Candidate", 75.56, 42.06, "autumn_pasture"),
    ("Z_KOCHKOR", "Kyzyl-Dobo Bench Autumn Candidate", 76.26, 41.86, "autumn_pasture"),
    ("Z_AKTALA", "Kurtka Descent Autumn Candidate", 75.04, 41.48, "autumn_pasture"),
    ("Z_AKTALA", "Terek Valley Autumn Candidate", 74.74, 41.36, "autumn_pasture"),
    ("Z_NARYN_W", "Uchkoshkon Lower Autumn Candidate", 75.44, 41.50, "autumn_pasture"),
    ("Z_NARYN_C", "Kazan-Kuigan Bench Autumn Candidate", 76.24, 41.52, "autumn_pasture"),
    ("Z_ATBASHI", "Acha-Kaindy Slope Autumn Candidate", 75.88, 41.04, "autumn_pasture"),
    ("Z_ATBASHI", "Kazybek Approach Autumn Candidate", 76.26, 41.14, "autumn_pasture"),
]

# 6 mountain pass crossing nodes along well-known Naryn pass zones (labelled synthetic_demo)
_PASS_ANCHORS = [
    ("Z_KOCHKOR", "Dolon Pass Zone Crossing", 75.74, 41.83),
    ("Z_AKTALA", "Moldo-Ashuu Pass Zone Crossing", 74.96, 41.61),
    ("Z_KOCHKOR", "Kalmak-Ashuu Pass Zone Crossing", 75.44, 41.94),
    ("Z_NARYN_W", "Terskey-Torpok Pass Zone Crossing", 75.34, 41.62),
    ("Z_ATBASHI", "Kyndy Pass Zone Crossing", 76.34, 41.06),
    ("Z_NARYN_C", "Kara-Koo Ridge Crossing", 76.26, 41.68),
]

# 12 water access point anchors used when OSM water points are absent (or supplemented)
_WATER_ANCHORS = [
    ("Z_JUMGAL", "Jumgal River Access Point", 74.48, 41.88),
    ("Z_KOCHKOR", "Chu-Kochkor Confluence Water Point", 75.74, 42.19),
    ("Z_KOCHKOR", "Son-Kul North Shore Water Point", 75.18, 41.92),
    ("Z_AKTALA", "Son-Kul South Shore Water Point", 75.15, 41.74),
    ("Z_AKTALA", "Alabuga River Water Point", 74.82, 41.35),
    ("Z_NARYN_W", "Naryn River West Water Point", 75.48, 41.42),
    ("Z_NARYN_C", "Naryn River Central Water Point", 76.02, 41.44),
    ("Z_NARYN_C", "On-Archa Stream Water Point", 76.12, 41.64),
    ("Z_NARYN_C", "Kichi-Naryn Confluence Water Point", 76.48, 41.47),
    ("Z_ATBASHI", "At-Bashi River Water Point", 75.80, 41.15),
    ("Z_ATBASHI", "Karakoyun Stream Water Point", 75.56, 41.02),
    ("Z_ATBASHI", "Aksai River Headwaters Point", 76.54, 40.94),
]


def _pick_constrained_point(
    base_lon: float,
    base_lat: float,
    region_wgs,
    rng: np.random.Generator,
    env: EnvironmentInventory,
    elev_band: tuple[float, float] | None,
    max_slope_deg: float,
    preferred_wc: set[int],
    n_candidates: int = 18,
) -> tuple[Point, float | None, float | None, int | None, str]:
    """Sample candidate points near anchor inside Naryn ADM1 and pick the best environmental match."""
    cand_pts: list[Point] = []
    for _ in range(n_candidates * 2):
        jlon = float(base_lon + rng.uniform(-0.045, 0.045))
        jlat = float(base_lat + rng.uniform(-0.032, 0.032))
        pt = Point(jlon, jlat)
        if region_wgs.contains(pt):
            cand_pts.append(pt)
        if len(cand_pts) >= n_candidates:
            break
    if not cand_pts:
        base_pt = Point(base_lon, base_lat)
        cand_pts = [base_pt if region_wgs.contains(base_pt) else region_wgs.representative_point()]

    lons = [float(p.x) for p in cand_pts]
    lats = [float(p.y) for p in cand_pts]
    elevs, slopes, _, _ = sample_dem_strict(lons, lats, env)
    wc_codes, wc_labels, _ = sample_worldcover_strict(lons, lats, env)

    best_idx = 0
    best_score = -1e9
    for i in range(len(cand_pts)):
        score = 0.0
        # Prefer gentle slopes when real DEM is present
        s = slopes[i]
        if s is not None:
            score += 2.0 if s <= max_slope_deg else -5.0 * (s - max_slope_deg)
        # Prefer target seasonal elevation band when real DEM is present
        e = elevs[i]
        if e is not None and elev_band is not None:
            lo, hi = elev_band
            if lo <= e <= hi:
                score += 4.0
            else:
                dist_band = min(abs(e - lo), abs(e - hi))
                score -= dist_band / 250.0
        # Prefer grassland (30), shrubland (20), wetland/moss when WorldCover is present
        wc = wc_codes[i]
        if wc is not None:
            if wc == 30:
                score += 4.5
            elif wc == 20:
                score += 3.2
            elif wc in preferred_wc:
                score += 1.5
            elif wc in {50, 70, 80}:  # Avoid built-up, snow/ice, permanent water
                score -= 8.0
        if score > best_score:
            best_score = score
            best_idx = i

    return cand_pts[best_idx], elevs[best_idx], slopes[best_idx], wc_codes[best_idx], wc_labels[best_idx]


def build_pastoral_nodes(
    boundary_gdf: gpd.GeoDataFrame,
    layer_a_nodes: gpd.GeoDataFrame,
    env: EnvironmentInventory,
    cfg: dict[str, Any],
    *,
    seed: int = 42,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Build Layer B static nodes (40 candidate pastures, passes, water points) + link Layer A settlements."""
    rng = np.random.default_rng(seed)
    region_wgs = unary_union(boundary_gdf.to_crs("EPSG:4326").geometry)
    constraints = cfg.get("pasture_constraints", {})
    max_slope = float(constraints.get("max_slope_deg", 26.0))
    pref_wc = set(int(c) for c in constraints.get("preferred_worldcover_codes", [30, 20, 90, 100]))
    elev_bands = constraints.get("elevation_bands_m", {})
    scale = cfg.get("network_scale", {})
    num_pastures = int(scale.get("num_pastures", 40))
    num_passes = int(scale.get("num_mountain_passes", 6))
    num_water = int(scale.get("num_water_points", 12))
    gen_ver = str(cfg.get("generator_version", "3.0.0-stage1"))
    disclaimer = str(cfg.get("provenance", {}).get("disclaimer", "Synthetic educational demonstration only."))

    has_env_rasters = (env.dem_path is not None) and bool(env.worldcover_paths)
    cand_evidence = "derived_environmental" if has_env_rasters else "synthetic_demo"
    cand_prefix = "DER-PAS" if has_env_rasters else "SYN-PAS"

    rows: list[dict[str, Any]] = []

    # 1. Generate 40 candidate pastures (10 per seasonal category, evenly sliced if custom count requested)
    per_season = max(1, num_pastures // 4)
    selected_anchors: list[tuple[str, str, float, float, str]] = []
    for ntype in ("winter_pasture", "spring_pasture", "summer_pasture", "autumn_pasture"):
        group = [a for a in _PASTURE_ANCHORS if a[4] == ntype]
        selected_anchors.extend(group[:per_season])
    # Top up if num_pastures not divisible by 4
    if len(selected_anchors) < num_pastures:
        for a in _PASTURE_ANCHORS:
            if a not in selected_anchors and len(selected_anchors) < num_pastures:
                selected_anchors.append(a)

    type_code = {"winter_pasture": "W", "spring_pasture": "SP", "summer_pasture": "SU", "autumn_pasture": "AU"}
    counters = {"winter_pasture": 0, "spring_pasture": 0, "summer_pasture": 0, "autumn_pasture": 0}

    for zone, label, blon, blat, ntype in selected_anchors:
        counters[ntype] += 1
        band_cfg = elev_bands.get(ntype)
        band = (float(band_cfg[0]), float(band_cfg[1])) if band_cfg else None
        pt, elev_m, slope_deg, wc_code, wc_label = _pick_constrained_point(
            blon, blat, region_wgs, rng, env, band, max_slope, pref_wc
        )
        nid = f"{cand_prefix}-{type_code[ntype]}{counters[ntype]:02d}"
        rows.append({
            "node_id": nid,
            "node_name": label,
            "node_type": ntype,
            "network_layer": "pastoral_mobility",
            "subregion_zone": zone,
            "latitude": round(float(pt.y), 5),
            "longitude": round(float(pt.x), 5),
            "elevation_m": elev_m,
            "slope_degrees": slope_deg,
            "elevation_status": "dem_derived" if elev_m is not None else "unavailable_no_local_dem",
            "elevation_source": env.dem_source if elev_m is not None else "unavailable_no_local_dem",
            "landcover_code": wc_code,
            "landcover_class": wc_label,
            "pasture_category": "derived_environmental_candidate" if has_env_rasters else "synthetic_demonstration_candidate",
            "seasonal_role": ntype.replace("_pasture", ""),
            "source": f"Constrained candidate generator ({env.dem_source})" if has_env_rasters else "Synthetic transhumance network generator",
            "provenance": "synthetic_demo",
            "evidence_status": cand_evidence,
            "is_synthetic": True,
            "generator_version": gen_ver,
            "random_seed": int(seed),
            "synthetic_disclaimer": disclaimer,
            "geometry": pt,
        })

    # 2. Mountain passes (geographical crossing zones inside Naryn ADM1)
    for idx, (zone, label, blon, blat) in enumerate(_PASS_ANCHORS[:num_passes], start=1):
        pt, elev_m, slope_deg, wc_code, wc_label = _pick_constrained_point(
            blon, blat, region_wgs, rng, env, (2850.0, 3950.0), 35.0, pref_wc, n_candidates=10
        )
        rows.append({
            "node_id": f"SYN-PASS-{idx:02d}",
            "node_name": label,
            "node_type": "mountain_pass",
            "network_layer": "pastoral_mobility",
            "subregion_zone": zone,
            "latitude": round(float(pt.y), 5),
            "longitude": round(float(pt.x), 5),
            "elevation_m": elev_m,
            "slope_degrees": slope_deg,
            "elevation_status": "dem_derived" if elev_m is not None else "unavailable_no_local_dem",
            "elevation_source": env.dem_source if elev_m is not None else "unavailable_no_local_dem",
            "landcover_code": wc_code,
            "landcover_class": wc_label,
            "pasture_category": "not_a_pasture",
            "seasonal_role": "transit",
            "source": "Synthetic pass crossing node (Tian Shan corridor anchor)",
            "provenance": "synthetic_demo",
            "evidence_status": "synthetic_demo",
            "is_synthetic": True,
            "generator_version": gen_ver,
            "random_seed": int(seed),
            "synthetic_disclaimer": disclaimer,
            "geometry": pt,
        })

    # 3. Water access points (reuse public Layer A OSM water points first; supplement with synthetic if none exist)
    pub_waters = layer_a_nodes.loc[layer_a_nodes["node_type"] == "water_point"] if not layer_a_nodes.empty else gpd.GeoDataFrame()
    if len(pub_waters) >= 6:
        water_mode = "public_osm_water_points_linked"
    else:
        water_mode = "synthetic_demo_water_points"
        for idx, (zone, label, blon, blat) in enumerate(_WATER_ANCHORS[:num_water], start=1):
            pt = Point(blon, blat)
            if not region_wgs.contains(pt):
                pt = region_wgs.representative_point()
            elevs, slopes, e_stat, e_src = sample_dem_strict([pt.x], [pt.y], env)
            wc_c, wc_l, _ = sample_worldcover_strict([pt.x], [pt.y], env)
            rows.append({
                "node_id": f"SYN-WAT-{idx:02d}",
                "node_name": label,
                "node_type": "water_point",
                "network_layer": "pastoral_mobility",
                "subregion_zone": zone,
                "latitude": round(float(pt.y), 5),
                "longitude": round(float(pt.x), 5),
                "elevation_m": elevs[0],
                "slope_degrees": slopes[0],
                "elevation_status": e_stat,
                "elevation_source": e_src,
                "landcover_code": wc_c[0],
                "landcover_class": wc_l[0],
                "pasture_category": "not_a_pasture",
                "seasonal_role": "water_access",
                "source": "Synthetic water access node (OSM water layer not present locally)",
                "provenance": "synthetic_demo",
                "evidence_status": "synthetic_demo",
                "is_synthetic": True,
                "generator_version": gen_ver,
                "random_seed": int(seed),
                "synthetic_disclaimer": disclaimer,
                "geometry": pt,
            })

    nodes_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    report = {
        "pastures_generated": int(sum(counters.values())),
        "pastures_by_type": counters,
        "passes_generated": min(num_passes, len(_PASS_ANCHORS)),
        "water_point_mode": water_mode,
        "dem_constrained": env.dem_path is not None,
        "worldcover_constrained": bool(env.worldcover_paths),
    }
    return nodes_gdf, report
