"""Climate-responsive seasonal herd movement state machine and scenario switcher."""
from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


def _build_route_lookup(routes_gdf: gpd.GeoDataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    """Index corridors bidirectionally so herds can ascend and return along terrain paths."""
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, r in routes_gdf.iterrows():
        orig = str(r["origin_pasture_id"])
        dst = str(r["destination_pasture_id"])
        geom: LineString = r["geometry"]
        info = {
            "route_id": str(r["route_id"]),
            "distance_km": float(r["distance_km"]),
            "geometry": geom,
            "reversed": False,
        }
        lookup[(orig, dst)] = info
        rev_geom = LineString(list(geom.coords)[::-1])
        lookup[(dst, orig)] = {
            "route_id": str(r["route_id"]),
            "distance_km": float(r["distance_km"]),
            "geometry": rev_geom,
            "reversed": True,
        }
    return lookup


def _resolve_leg(
    orig: str,
    dst: str,
    route_lookup: dict[tuple[str, str], dict[str, Any]],
    pasture_coords: dict[str, tuple[float, float, float]],
) -> tuple[str, LineString, float]:
    """Return (route_id, LineString_WGS84, distance_km) between two pastures.

    Prefers direct least-cost corridor; otherwise composes a 2-hop corridor
    through a shared foothill hub so herds stay on terrain-following corridors.
    """
    if (orig, dst) in route_lookup:
        rec = route_lookup[(orig, dst)]
        return rec["route_id"], rec["geometry"], rec["distance_km"]

    # Try 2-hop corridor through a shared intermediate pasture
    best_chain = None
    best_dist = float("inf")
    for (a, mid), r1 in route_lookup.items():
        if a == orig and (mid, dst) in route_lookup:
            r2 = route_lookup[(mid, dst)]
            total = r1["distance_km"] + r2["distance_km"]
            if total < best_dist:
                best_dist = total
                coords = list(r1["geometry"].coords) + list(r2["geometry"].coords)[1:]
                best_chain = (f"{r1['route_id']}+{r2['route_id']}", LineString(coords), total)
    if best_chain is not None:
        return best_chain

    # Final fallback: geodesic segment between verified candidate pasture coordinates
    x1, y1, _ = pasture_coords[orig]
    x2, y2, _ = pasture_coords[dst]
    dist_km = float(math.hypot((x2 - x1) * 111.32 * math.cos(math.radians(0.5 * (y1 + y2))), (y2 - y1) * 110.54))
    return f"FALLBACK-{orig}-{dst}", LineString([(x1, y1), (x2, y2)]), max(dist_km, 1.0)


def adapt_herd_plan_for_scenario(
    herd: dict[str, Any],
    scenario_id: str,
    suitability_df: pd.DataFrame,
    pastures_gdf: gpd.GeoDataFrame,
    route_lookup: dict[tuple[str, str], dict[str, Any]],
    suitability_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Adjust summer/autumn destination and timing transparently under climate scenarios."""
    plan = {
        "winter_id": str(herd["winter_pasture_id"]),
        "spring_id": str(herd["spring_destination_id"]),
        "summer_id": str(herd["summer_pasture_id"]),
        "autumn_id": str(herd["autumn_destination_id"]),
        "d_spring": int(herd["spring_departure_doy"]),
        "d_summer": int(herd["summer_departure_doy"]),
        "d_autumn": int(herd["autumn_departure_doy"]),
        "d_winter": int(herd["winter_return_doy"]),
        "scenario_changed_route": False,
        "scenario_adaptation_note": "Baseline schedule",
    }
    if scenario_id == "SCENARIO_BASELINE":
        return plan

    scen_scores = suitability_df.loc[suitability_df["scenario_id"] == scenario_id]
    pasture_elev = pastures_gdf.set_index("pasture_id")["elevation_m"].to_dict()
    summer_candidates = pastures_gdf.loc[pastures_gdf["elevation_band"] == "high_alpine", "pasture_id"].tolist()
    foothill_candidates = pastures_gdf.loc[pastures_gdf["elevation_band"] == "foothill_transition", "pasture_id"].tolist()

    if scenario_id == "SCENARIO_WARMER_DRIER":
        # Evaluate summer suitability: if another reachable alpine pasture scores notably higher (+0.04), switch!
        s_scores = scen_scores.loc[scen_scores["season"] == "summer"].set_index("pasture_id")["suitability_score"].to_dict()
        curr_score = s_scores.get(plan["summer_id"], 0.5)
        best_alt = max(summer_candidates, key=lambda pid: (s_scores.get(pid, 0.0), pasture_elev.get(pid, 0.0)))
        if best_alt != plan["summer_id"] and s_scores.get(best_alt, 0.0) >= curr_score + 0.02:
            plan["summer_id"] = best_alt
            plan["scenario_changed_route"] = True
            plan["scenario_adaptation_note"] = (
                f"Rerouted summer grazing to higher/better-watered {best_alt} under warmer/drier summer assumptions"
            )
        # Herds also ascend slightly earlier (5-10 days) as snow melts earlier
        plan["d_summer"] = max(plan["d_spring"] + 20, plan["d_summer"] - 8)
        plan["d_autumn"] = min(plan["d_winter"] - 18, plan["d_autumn"] + 6)

    elif scenario_id == "SCENARIO_EARLY_SNOW":
        shift_days = int(suitability_cfg.get("scenarios", {}).get(scenario_id, {}).get("early_snow_shift_days", 22))
        if pasture_elev.get(plan["summer_id"], 0.0) >= 2850.0:
            plan["d_autumn"] = max(plan["d_summer"] + 32, plan["d_autumn"] - shift_days)
            plan["d_winter"] = max(plan["d_autumn"] + 18, plan["d_winter"] - int(shift_days * 0.7))
            # Pick best-scoring sheltered autumn foothill refuge
            a_scores = scen_scores.loc[scen_scores["season"] == "autumn"].set_index("pasture_id")["suitability_score"].to_dict()
            best_autumn = max(foothill_candidates, key=lambda pid: a_scores.get(pid, 0.0))
            if best_autumn != plan["autumn_id"]:
                plan["autumn_id"] = best_autumn
                plan["scenario_changed_route"] = True
            plan["scenario_adaptation_note"] = (
                f"Early alpine snow triggered {shift_days}-day earlier descent to sheltered {plan['autumn_id']}"
            )

    return plan
