"""Transparent, configurable rule-based pasture suitability scoring engine.

PastureSuitability =
    w_climate    * climate_score
  + w_vegetation * vegetation_score
  + w_water      * water_access_score
  + w_season     * season_score
  - w_terrain    * terrain_penalty
  - w_route      * route_cost_penalty

Clamped to [0.0, 1.0]. All weights, thresholds, and scenario offsets come from
`config/suitability_weights.yaml`. Missing climate inputs degrade to neutral 0.50.
"""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


def score_pastures_for_scenarios(
    pastures_gdf: gpd.GeoDataFrame,
    routes_gdf: gpd.GeoDataFrame,
    scenario_climate: pd.DataFrame,
    suitability_cfg: dict[str, Any],
    *,
    reference_elev_m: float = 2044.0,
) -> pd.DataFrame:
    """Evaluate every candidate pasture by (scenario_id, season) transparently."""
    weights = suitability_cfg.get("weights", {})
    w_clim = float(weights.get("climate", 0.34))
    w_veg = float(weights.get("vegetation", 0.22))
    w_wat = float(weights.get("water_access", 0.18))
    w_seas = float(weights.get("season", 0.26))
    w_terr = float(weights.get("terrain_penalty", 0.14))
    w_route = float(weights.get("route_cost_penalty", 0.10))

    cmod = suitability_cfg.get("climate_model", {})
    lapse = float(cmod.get("lapse_rate_c_per_1000m", -6.5))
    t_opt = float(cmod.get("optimal_grazing_temp_c", 11.0))
    t_tol = float(cmod.get("temp_tolerance_c", 10.5))
    p_opt = float(cmod.get("optimal_monthly_precip_mm", 45.0))
    scenarios_meta = suitability_cfg.get("scenarios", {})

    # Average route cost into each destination pasture (normalised 0..1)
    if not routes_gdf.empty and "route_cost" in routes_gdf.columns:
        max_cost = max(float(routes_gdf["route_cost"].max()), 1.0)
        route_cost_by_dst = routes_gdf.groupby("destination_pasture_id")["route_cost"].mean().to_dict()
    else:
        max_cost = 1.0
        route_cost_by_dst = {}

    # Seasonal climate aggregates per (scenario_id, season)
    clim_agg = (
        scenario_climate.groupby(["scenario_id", "season"], as_index=False)
        .agg(
            mean_temp_c=("scenario_t2m_c", "mean"),
            mean_30d_precip_mm=("precip_30d_mm", "mean"),
            snow_risk_share=("high_alpine_snow_risk", "mean"),
        )
    )

    rows: list[dict[str, Any]] = []
    for _, c_row in clim_agg.iterrows():
        scen_id = str(c_row["scenario_id"])
        season = str(c_row["season"])
        scen_spec = scenarios_meta.get(scen_id, {})
        snow_pen_cfg = float(scen_spec.get("high_altitude_snow_penalty", 0.0))
        origin_class = str(scen_spec.get("origin_class", "simulated_scenario"))

        for _, p in pastures_gdf.iterrows():
            elev_m = float(p["elevation_m"])
            slope_deg = float(p["slope_degrees"])
            band = str(p["elevation_band"])
            veg = float(p["vegetation_indicator"])
            water = float(p["water_access_indicator"])
            pid = str(p["pasture_id"])

            # Lapse-rate adjusted pasture temperature
            if pd.isna(c_row["mean_temp_c"]):
                temp_score = 0.50
                pasture_temp = np.nan
            else:
                pasture_temp = float(c_row["mean_temp_c"]) + lapse * ((elev_m - reference_elev_m) / 1000.0)
                temp_score = float(np.exp(-0.5 * ((pasture_temp - t_opt) / t_tol) ** 2))

            # Moisture score (higher elevation retains snowmelt moisture better in dry summers)
            if pd.isna(c_row["mean_30d_precip_mm"]):
                precip_score = 0.50
            else:
                eff_precip = float(c_row["mean_30d_precip_mm"]) * (1.0 + 0.18 * max(0.0, (elev_m - 2200.0) / 1000.0))
                precip_score = float(np.clip(eff_precip / p_opt, 0.12, 1.0))

            # Heat/drought penalty in lower/mid summer zones under warmer/drier conditions
            heat_stress = 0.0
            if scen_id == "SCENARIO_WARMER_DRIER" and season == "summer" and elev_m < 3050.0:
                heat_stress = 0.22 * (1.0 - water * 0.5)

            # Early snow penalty in high alpine pastures in autumn
            early_snow_pen = 0.0
            if season == "autumn" and elev_m >= 2850.0:
                early_snow_pen = snow_pen_cfg * (0.65 + 0.35 * float(c_row["snow_risk_share"]))

            climate_score = float(np.clip(0.58 * temp_score + 0.42 * precip_score - heat_stress - early_snow_pen, 0.0, 1.0))

            # Seasonal band affinity
            band_season_matrix = {
                ("valley_winter", "winter"): 0.92,
                ("valley_winter", "spring"): 0.65,
                ("valley_winter", "summer"): 0.38,
                ("valley_winter", "autumn"): 0.72,
                ("foothill_transition", "winter"): 0.45,
                ("foothill_transition", "spring"): 0.90,
                ("foothill_transition", "summer"): 0.68,
                ("foothill_transition", "autumn"): 0.88,
                ("high_alpine", "winter"): 0.10,
                ("high_alpine", "spring"): 0.42,
                ("high_alpine", "summer"): 0.95,
                ("high_alpine", "autumn"): 0.52 - early_snow_pen * 0.5,
            }
            season_score = float(np.clip(band_season_matrix.get((band, season), 0.50), 0.0, 1.0))

            # Terrain & route cost penalties
            terrain_pen = float(np.clip((slope_deg / 35.0) * 0.75 + max(0.0, (elev_m - 3400.0) / 1200.0) * 0.25, 0.0, 1.0))
            route_pen = float(np.clip(route_cost_by_dst.get(pid, max_cost * 0.35) / max_cost, 0.0, 1.0))

            raw_score = (
                w_clim * climate_score
                + w_veg * veg
                + w_wat * water
                + w_seas * season_score
                - w_terr * terrain_pen
                - w_route * route_pen
            )
            suitability = round(float(np.clip(raw_score, 0.0, 1.0)), 4)

            rows.append({
                "scenario_id": scen_id,
                "season": season,
                "pasture_id": pid,
                "pasture_name": p["pasture_name"],
                "elevation_m": elev_m,
                "elevation_band": band,
                "estimated_pasture_temp_c": round(float(pasture_temp), 2) if np.isfinite(pasture_temp) else None,
                "climate_score": round(climate_score, 3),
                "vegetation_score": round(veg, 3),
                "water_access_score": round(water, 3),
                "season_score": round(season_score, 3),
                "terrain_penalty": round(terrain_pen, 3),
                "route_cost_penalty": round(route_pen, 3),
                "suitability_score": suitability,
                "origin_class": origin_class,
                "is_synthetic": True,
                "provenance": "synthetic_demo",
            })

    return pd.DataFrame(rows)
