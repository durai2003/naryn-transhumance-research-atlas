"""Build continuous daily time-indexed herd trajectories and discrete movement events.

Guarantees:
- Continuous spatial progression along terrain corridors (no teleportation).
- Exact alignment with actual NASA POWER dates in the simulation window.
- Explicit synthetic & scenario provenance metadata on every row.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from src.atlas.simulation.movement_engine import _build_route_lookup, _resolve_leg, adapt_herd_plan_for_scenario
from src.atlas.terrain import sample_elevation_and_slope


def build_trajectories_and_events(
    herds_df: pd.DataFrame,
    pastures_gdf: gpd.GeoDataFrame,
    routes_gdf: gpd.GeoDataFrame,
    scenario_climate: pd.DataFrame,
    suitability_df: pd.DataFrame,
    suitability_cfg: dict[str, Any],
    root: Path,
    *,
    scenarios: list[str] | None = None,
    simulation_version: str = "2.0.0",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate daily positions and discrete movement events for all herds and scenarios."""
    target_scenarios = scenarios or list(suitability_cfg.get("scenarios", {}).keys()) or ["SCENARIO_BASELINE"]
    route_lookup = _build_route_lookup(routes_gdf)

    pasture_coords = {
        str(r["pasture_id"]): (float(r["longitude"]), float(r["latitude"]), float(r["elevation_m"]))
        for _, r in pastures_gdf.iterrows()
    }
    suit_lookup = {
        (str(r["scenario_id"]), str(r["season"]), str(r["pasture_id"])): float(r["suitability_score"])
        for _, r in suitability_df.iterrows()
    }

    traj_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    event_counter = 0

    for scen_id in target_scenarios:
        scen_clim = scenario_climate.loc[scenario_climate["scenario_id"] == scen_id].sort_values("date_str").reset_index(drop=True)
        if scen_clim.empty:
            continue
        origin_class = "synthetic" if scen_id == "SCENARIO_BASELINE" else "simulated_scenario"
        dates = scen_clim["date_str"].tolist()
        doys = scen_clim["doy"].astype(int).tolist()
        years = scen_clim["year"].astype(int).tolist()
        seasons = scen_clim["season"].tolist()
        temps = scen_clim["scenario_t2m_c"].tolist()
        precips = scen_clim["scenario_precip_mm_day"].tolist()

        for _, herd_row in herds_df.iterrows():
            herd = herd_row.to_dict()
            hid = str(herd["herd_id"])
            sp = str(herd["species"])
            gsize = int(herd["group_size"])
            speed = float(herd["average_movement_speed_km_day"])
            rest_days = int(herd["rest_days"])

            plan = adapt_herd_plan_for_scenario(
                herd, scen_id, suitability_df, pastures_gdf, route_lookup, suitability_cfg
            )

            # Resolve the 4 seasonal migration legs
            leg_specs = [
                ("spring_ascent", plan["winter_id"], plan["spring_id"], plan["d_spring"]),
                ("summer_ascent", plan["spring_id"], plan["summer_id"], plan["d_summer"]),
                ("autumn_descent", plan["summer_id"], plan["autumn_id"], plan["d_autumn"]),
                ("winter_return", plan["autumn_id"], plan["winter_id"], plan["d_winter"]),
            ]

            legs = []
            prev_end_doy = 1
            for leg_name, orig, dst, dep_doy in leg_specs:
                rid, geom, dist_km = _resolve_leg(orig, dst, route_lookup, pasture_coords)
                travel_days = max(2, int(math.ceil(dist_km / max(speed, 2.0))))
                # Ensure legs never overlap chronologically
                start_doy = max(dep_doy, prev_end_doy + 2)
                end_doy = min(364, start_doy + travel_days)
                if end_doy <= start_doy:
                    end_doy = min(365, start_doy + 1)
                legs.append({
                    "leg_name": leg_name,
                    "origin": orig,
                    "destination": dst,
                    "route_id": rid,
                    "geometry": geom,
                    "distance_km": dist_km,
                    "start_doy": start_doy,
                    "end_doy": end_doy,
                })
                prev_end_doy = end_doy

                # Record departure and arrival events
                for ev_type, ev_doy, ev_pid in (("departure", start_doy, orig), ("arrival", end_doy, dst)):
                    event_counter += 1
                    idx_d = min(max(ev_doy - 1, 0), len(dates) - 1)
                    lon_e, lat_e, elev_e = pasture_coords[ev_pid]
                    event_rows.append({
                        "event_id": f"SYN-EVT-{event_counter:05d}",
                        "timestamp": f"{dates[idx_d]}T08:00:00Z",
                        "herd_id": hid,
                        "species": sp,
                        "group_size": gsize,
                        "event_type": ev_type,
                        "leg_name": leg_name,
                        "route_id": rid,
                        "pasture_id": ev_pid,
                        "latitude": round(lat_e, 5),
                        "longitude": round(lon_e, 5),
                        "elevation_m": round(elev_e, 1),
                        "climate_scenario": scen_id,
                        "scenario_changed_route": bool(plan["scenario_changed_route"]),
                        "adaptation_note": plan["scenario_adaptation_note"],
                        "origin_class": origin_class,
                        "simulation_version": simulation_version,
                        "random_seed": int(seed),
                        "is_synthetic": True,
                        "provenance": "synthetic_demo",
                    })

            # Emit daily positions for the full simulation period
            for d_idx, (d_str, doy, yr, seas, t_val, p_val) in enumerate(zip(dates, doys, years, seasons, temps, precips)):
                active_leg = None
                active_leg_idx = -1
                for li, leg in enumerate(legs):
                    if leg["start_doy"] <= doy <= leg["end_doy"]:
                        active_leg = leg
                        active_leg_idx = li
                        break

                if active_leg is not None:
                    span = max(1, active_leg["end_doy"] - active_leg["start_doy"])
                    frac = float(np.clip((doy - active_leg["start_doy"]) / span, 0.0, 1.0))
                    pt = active_leg["geometry"].interpolate(frac, normalized=True)
                    lon, lat = float(pt.x), float(pt.y)
                    o_elev = pasture_coords[active_leg["origin"]][2]
                    d_elev = pasture_coords[active_leg["destination"]][2]
                    # Smooth ridge-passing elevation interpolation (upgraded to DEM sample below)
                    elev_est = o_elev + frac * (d_elev - o_elev) + 120.0 * math.sin(math.pi * frac)
                    rid = active_leg["route_id"]
                    step_speed = round(active_leg["distance_km"] / span, 2)
                    if doy == active_leg["end_doy"]:
                        state = "arrived"
                        next_leg = legs[(active_leg_idx + 1) % len(legs)]
                        cur_pid = active_leg["destination"]
                        nxt_pid = next_leg["destination"]
                    else:
                        cur_pid = active_leg["origin"]
                        nxt_pid = active_leg["destination"]
                        if doy == active_leg["start_doy"]:
                            state = "departing"
                        elif active_leg["leg_name"] in ("autumn_descent", "winter_return"):
                            state = "returning"
                        elif active_leg["leg_name"] == "spring_ascent":
                            state = "seasonal_transition"
                        else:
                            state = "moving"
                else:
                    # Stationary at whichever pasture was most recently reached
                    if doy < legs[0]["start_doy"]:
                        stay_pid = plan["winter_id"]
                        nxt_pid = plan["spring_id"]
                        rid = legs[0]["route_id"]
                    elif doy < legs[1]["start_doy"]:
                        stay_pid = plan["spring_id"]
                        nxt_pid = plan["summer_id"]
                        rid = legs[1]["route_id"]
                    elif doy < legs[2]["start_doy"]:
                        stay_pid = plan["summer_id"]
                        nxt_pid = plan["autumn_id"]
                        rid = legs[2]["route_id"]
                    elif doy < legs[3]["start_doy"]:
                        stay_pid = plan["autumn_id"]
                        nxt_pid = plan["winter_id"]
                        rid = legs[3]["route_id"]
                    else:
                        stay_pid = plan["winter_id"]
                        nxt_pid = plan["spring_id"]
                        rid = legs[0]["route_id"]

                    lon, lat, elev_est = pasture_coords[stay_pid]
                    cur_pid = stay_pid
                    step_speed = 0.0
                    # Distinguish post-arrival resting days from ongoing grazing
                    days_since_arrival = min(
                        [doy - leg["end_doy"] for leg in legs if leg["end_doy"] < doy] or [999]
                    )
                    state = "resting" if 0 < days_since_arrival <= rest_days else "grazing"

                suit_score = suit_lookup.get((scen_id, seas, cur_pid), 0.5)

                traj_rows.append({
                    "timestamp": f"{d_str}T12:00:00Z",
                    "date": d_str,
                    "simulation_year": yr,
                    "season": seas,
                    "herd_id": hid,
                    "species": sp,
                    "group_size": gsize,
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                    "elevation_m": round(elev_est, 1),
                    "current_pasture_id": cur_pid,
                    "next_pasture_id": nxt_pid,
                    "route_id": rid,
                    "movement_state": state,
                    "movement_speed_km_day": step_speed,
                    "climate_scenario": scen_id,
                    "temperature_context": round(float(t_val), 2) if pd.notna(t_val) else None,
                    "precipitation_context": round(float(p_val), 2) if pd.notna(p_val) else None,
                    "pasture_suitability_score": round(suit_score, 3),
                    "data_source": "synthetic_movement_simulation_with_nasa_power_climate",
                    "origin_class": origin_class,
                    "simulation_version": simulation_version,
                    "random_seed": int(seed),
                    "provenance": "synthetic_demo",
                    "is_synthetic": True,
                })

    traj_df = pd.DataFrame(traj_rows)
    events_df = pd.DataFrame(event_rows)

    # Upgrade trajectory elevations in one vectorized DEM/terrain pass
    if not traj_df.empty:
        elevs, _, elev_source = sample_elevation_and_slope(
            traj_df["longitude"].to_numpy(), traj_df["latitude"].to_numpy(), root
        )
        # Only replace with DEM sample when local DEM exists; keep smooth pasture-consistent elevations for proxy
        if elev_source != "synthetic_relief_proxy_pending_dem":
            traj_df["elevation_m"] = np.round(elevs, 1)

    return traj_df, events_df
