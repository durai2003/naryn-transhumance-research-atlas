"""Reproducible synthetic herd group generator."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


def generate_synthetic_herds(
    pastures_gdf: gpd.GeoDataFrame,
    routes_gdf: gpd.GeoDataFrame,
    sim_cfg: dict[str, Any],
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate 40+ diverse synthetic herd groups with coherent pasture assignments."""
    use_seed = int(seed if seed is not None else sim_cfg.get("random_seed", 42))
    rng = np.random.default_rng(use_seed)
    num_herds = int(sim_cfg.get("num_herds", 48))
    year = int(sim_cfg.get("simulation_year", 2024))
    sim_version = str(sim_cfg.get("simulation_version", "2.0.0"))
    disclaimer = str(sim_cfg.get("provenance", {}).get("disclaimer", "Synthetic educational demonstration only."))

    species_cfg = sim_cfg.get("species_profiles", {})
    species_names = list(species_cfg.keys()) or ["sheep", "goat", "cattle", "horse", "mixed"]
    weights = np.array([float(species_cfg[s].get("weight", 0.2)) for s in species_names], dtype=float)
    weights = weights / weights.sum()

    cal = sim_cfg.get("calendar_windows", {})
    spring_doy_range = cal.get("spring_departure_doy", [88, 122])
    summer_doy_range = cal.get("summer_departure_doy", [148, 178])
    autumn_doy_range = cal.get("autumn_departure_doy", [238, 268])
    winter_doy_range = cal.get("winter_return_doy", [292, 326])
    rest_range = cal.get("rest_days_range", [2, 8])

    winter_ids = pastures_gdf.loc[pastures_gdf["elevation_band"] == "valley_winter", "pasture_id"].tolist()
    foothill_ids = pastures_gdf.loc[pastures_gdf["elevation_band"] == "foothill_transition", "pasture_id"].tolist()
    summer_ids = pastures_gdf.loc[pastures_gdf["elevation_band"] == "high_alpine", "pasture_id"].tolist()

    # Build lookup from origin -> available corridors
    by_origin: dict[str, list[dict[str, Any]]] = {}
    for _, r in routes_gdf.iterrows():
        by_origin.setdefault(str(r["origin_pasture_id"]), []).append(r.to_dict())

    jan1 = date(year, 1, 1)
    rows: list[dict[str, Any]] = []

    for i in range(1, num_herds + 1):
        sp = str(rng.choice(species_names, p=weights))
        sp_meta = species_cfg.get(sp, {})
        size_min = int(sp_meta.get("min_size", 50))
        size_max = int(sp_meta.get("max_size", 250))
        spd_lo, spd_hi = sp_meta.get("speed_km_day_range", [7.0, 12.0])

        group_size = int(rng.integers(size_min, size_max + 1))
        speed = round(float(rng.uniform(float(spd_lo), float(spd_hi))), 2)
        rest_days = int(rng.integers(int(rest_range[0]), int(rest_range[1]) + 1))

        # Assign geographically connected winter -> spring/autumn foothill -> summer alpine chain
        winter_pid = winter_ids[(i - 1) % len(winter_ids)]
        wf_routes = [r for r in by_origin.get(winter_pid, []) if r["destination_pasture_id"] in foothill_ids]
        if wf_routes:
            chosen_wf = wf_routes[int(rng.integers(0, len(wf_routes)))]
            spring_pid = str(chosen_wf["destination_pasture_id"])
            primary_route_id = str(chosen_wf["route_id"])
        else:
            spring_pid = foothill_ids[(i - 1) % len(foothill_ids)]
            primary_route_id = str(routes_gdf.iloc[0]["route_id"])

        fs_routes = [r for r in by_origin.get(spring_pid, []) if r["destination_pasture_id"] in summer_ids]
        if fs_routes:
            chosen_fs = fs_routes[int(rng.integers(0, len(fs_routes)))]
            summer_pid = str(chosen_fs["destination_pasture_id"])
            summer_route_id = str(chosen_fs["route_id"])
        else:
            summer_pid = summer_ids[(i - 1) % len(summer_ids)]
            summer_route_id = primary_route_id

        # Staggered calendar dates (strictly chronological)
        d_spring = int(rng.integers(int(spring_doy_range[0]), int(spring_doy_range[1]) + 1))
        d_summer = max(d_spring + 28, int(rng.integers(int(summer_doy_range[0]), int(summer_doy_range[1]) + 1)))
        d_autumn = max(d_summer + 55, int(rng.integers(int(autumn_doy_range[0]), int(autumn_doy_range[1]) + 1)))
        d_winter = max(d_autumn + 26, int(rng.integers(int(winter_doy_range[0]), int(winter_doy_range[1]) + 1)))

        dep_date = (jan1 + timedelta(days=d_spring - 1)).isoformat()
        arr_date = (jan1 + timedelta(days=min(d_summer + 12, 360))).isoformat()

        rows.append({
            "herd_id": f"SYN-HERD-{i:03d}",
            "herd_name": f"Synthetic {sp.title()} Group {i:02d}",
            "species": sp,
            "livestock_type": sp,
            "group_size": group_size,
            "animal_count": group_size,
            "home_location_id": winter_pid,
            "winter_pasture_id": winter_pid,
            "spring_destination_id": spring_pid,
            "summer_pasture_id": summer_pid,
            "autumn_destination_id": spring_pid,
            "spring_departure_doy": d_spring,
            "summer_departure_doy": d_summer,
            "autumn_departure_doy": d_autumn,
            "winter_return_doy": d_winter,
            "departure_date": dep_date,
            "arrival_date": arr_date,
            "average_movement_speed_km_day": speed,
            "rest_days": rest_days,
            "route_id": summer_route_id,
            "spring_route_id": primary_route_id,
            "summer_route_id": summer_route_id,
            "simulation_scenario": "SCENARIO_BASELINE",
            "simulation_version": sim_version,
            "random_seed": use_seed,
            "origin_class": "synthetic",
            "is_synthetic": True,
            "provenance": "synthetic_demo",
            "synthetic_disclaimer": disclaimer,
        })

    return pd.DataFrame(rows)
