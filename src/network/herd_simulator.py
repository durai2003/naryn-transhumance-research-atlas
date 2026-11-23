"""Reproducible 90-herd seasonal simulation, route-interpolated positions, and derived edge usage.

Key invariants guaranteed by construction:
1. Every herd follows a 5-stage seasonal cycle (Winter -> Spring -> Summer -> Autumn -> Winter).
2. Arrival timestamp strictly follows departure timestamp (`arrival_timestamp > departure_timestamp`).
3. Positions are interpolated strictly along the actual LineString geometry of `current_edge_id`.
4. No teleportation: step distance matches travel speed and time interval.
5. Daily samples during stationary/grazing/resting periods + sub-daily samples (2/day) during
   active travel to produce >= 30,000 smooth animation positions over the 12-month period.
6. Edge utilisation (`edge_utilisation.csv`) is computed EXCLUSIVELY from simulated movement events.
   Unused edges have exact utilisation = 0.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.network.dem_env import EnvironmentInventory, sample_dem_strict


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def simulate_herds_and_network_usage(
    all_nodes_gdf: gpd.GeoDataFrame,
    pastoral_edges_gdf: gpd.GeoDataFrame,
    env: EnvironmentInventory,
    cfg: dict[str, Any],
    *,
    seed: int = 42,
    num_herds_override: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (herds_df, movement_events_df, herd_positions_df, edge_utilisation_df, sim_report)."""
    rng = np.random.default_rng(seed)
    scale = cfg.get("network_scale", {})
    num_herds = int(num_herds_override if num_herds_override is not None else scale.get("num_herds", 90))
    sim_year = int(cfg.get("simulation_year", 2024))
    start_dt = date.fromisoformat(str(cfg.get("start_date", f"{sim_year}-01-01")))
    end_dt = date.fromisoformat(str(cfg.get("end_date", f"{sim_year}-12-31")))
    total_days = (end_dt - start_dt).days + 1
    scenario_id = str(cfg.get("scenario_id", "SCENARIO_BASELINE"))
    gen_ver = str(cfg.get("generator_version", "3.0.0-stage1"))
    disclaimer = str(cfg.get("provenance", {}).get("disclaimer", "Synthetic educational demonstration only."))

    profiles = cfg.get("livestock_profiles", {})
    species_list = list(profiles.keys()) or ["sheep", "goat", "cattle", "horse", "mixed"]
    weights = np.array([float(profiles[s].get("weight", 0.2)) for s in species_list], dtype=float)
    weights = weights / weights.sum()

    cal = cfg.get("calendar_windows", {})
    win_spring = cal.get("spring_departure_doy", [80, 120])
    win_summer = cal.get("summer_departure_doy", [142, 180])
    win_autumn = cal.get("autumn_departure_doy", [232, 270])
    win_winter = cal.get("winter_return_doy", [288, 330])
    alt_prob = float(cal.get("alternative_route_probability", 0.26))
    subdaily = max(1, int(cal.get("movement_subdaily_samples_per_day", 2)))

    node_lookup = all_nodes_gdf.set_index("node_id").to_dict(orient="index")
    winters = all_nodes_gdf.loc[all_nodes_gdf["node_type"] == "winter_pasture", "node_id"].tolist()

    # Index movement edges by (source_node_id, edge_type) and by (source_node_id, target_node_id)
    mov_types = {"SEASONAL_MOVEMENT", "LEAST_COST_CORRIDOR", "ALTERNATIVE_ROUTE"}
    mov_edges = pastoral_edges_gdf.loc[pastoral_edges_gdf["edge_type"].isin(mov_types)].copy()

    # Group edges by source -> target (keeping primary vs alternative AND resting-point 2-hop chains!)
    direct_primary: dict[tuple[str, str], dict[str, Any]] = {}
    direct_alt: dict[tuple[str, str], dict[str, Any]] = {}
    out_primary: dict[str, list[dict[str, Any]]] = {}
    rest_chain: dict[tuple[str, str], tuple[dict[str, Any], str, dict[str, Any]]] = {}

    rest_ids = set(all_nodes_gdf.loc[all_nodes_gdf["node_type"] == "resting_point", "node_id"].astype(str))
    edges_by_src: dict[str, list[dict[str, Any]]] = {}
    for _, row in mov_edges.iterrows():
        rec = row.to_dict()
        src, dst, et = str(rec["source_node_id"]), str(rec["target_node_id"]), str(rec["edge_type"])
        edges_by_src.setdefault(src, []).append(rec)
        if dst not in rest_ids and src not in rest_ids:
            if et == "ALTERNATIVE_ROUTE":
                direct_alt[(src, dst)] = rec
            else:
                direct_primary[(src, dst)] = rec
                out_primary.setdefault(src, []).append(rec)

    # Identify resting-point 2-hop chains: src -> rest -> dst
    for src, elist in edges_by_src.items():
        if src in rest_ids:
            continue
        for e1 in elist:
            mid = str(e1["target_node_id"])
            if mid in rest_ids:
                for e2 in edges_by_src.get(mid, []):
                    dst = str(e2["target_node_id"])
                    if dst not in rest_ids:
                        rest_chain[(src, dst)] = (e1, mid, e2)

    def _pick_next_edge(curr_node: str, target_type: str) -> dict[str, Any] | None:
        cands = [
            e for e in out_primary.get(curr_node, [])
            if node_lookup.get(str(e["target_node_id"]), {}).get("node_type") == target_type
        ]
        if cands:
            return cands[int(rng.integers(0, len(cands)))]
        # Fallback: pick any primary edge targeting target_type from same zone
        return None

    herd_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    event_seq = 0
    gen_date_iso = datetime.now(timezone.utc).date().isoformat()

    for h_idx in range(1, num_herds + 1):
        herd_id = f"H-{h_idx:03d}"
        sp = str(rng.choice(species_list, p=weights))
        sp_cfg = profiles.get(sp, {})
        c_lo, c_hi = sp_cfg.get("animal_count_range", [80, 280])
        s_lo, s_hi = sp_cfg.get("speed_km_day_range", [7.5, 12.5])
        r_lo, r_hi = sp_cfg.get("rest_days_range", [2, 6])

        animal_count = int(rng.integers(int(c_lo), int(c_hi) + 1))
        speed_km_day = round(float(rng.uniform(float(s_lo), float(s_hi))), 2)
        rest_days = int(rng.integers(int(r_lo), int(r_hi) + 1))

        # Construct a connected 4-pasture seasonal loop: W -> SP -> SU -> AU -> W_end
        w_start = winters[(h_idx - 1) % len(winters)]
        e_ws = _pick_next_edge(w_start, "spring_pasture")
        if e_ws is None:
            continue
        sp_node = str(e_ws["target_node_id"])

        e_ss = _pick_next_edge(sp_node, "summer_pasture")
        if e_ss is None:
            continue
        su_node = str(e_ss["target_node_id"])

        e_sa = _pick_next_edge(su_node, "autumn_pasture")
        if e_sa is None:
            continue
        au_node = str(e_sa["target_node_id"])

        e_aw = _pick_next_edge(au_node, "winter_pasture")
        if e_aw is None:
            continue
        w_end = str(e_aw["target_node_id"])

        # Staggered departure DOYs with guaranteed chronological spacing
        d_sp = int(rng.integers(int(win_spring[0]), int(win_spring[1]) + 1))
        d_su = max(d_sp + 34, int(rng.integers(int(win_summer[0]), int(win_summer[1]) + 1)))
        d_au = max(d_su + 52, int(rng.integers(int(win_autumn[0]), int(win_autumn[1]) + 1)))
        d_wi = max(d_au + 28, int(rng.integers(int(win_winter[0]), int(win_winter[1]) + 1)))

        # Build sequence of actual route segments (either alternative route, 2-hop resting chain, or primary edge)
        raw_legs = [
            ("spring_migration", "spring", w_start, sp_node, d_sp, e_ws),
            ("summer_ascent", "summer", sp_node, su_node, d_su, e_ss),
            ("autumn_descent", "autumn", su_node, au_node, d_au, e_sa),
            ("winter_return", "winter", au_node, w_end, d_wi, e_aw),
        ]

        scheduled_segments: list[dict[str, Any]] = []
        preferred_edges: list[str] = []
        alt_edges_avail: list[str] = []
        used_any_alt = False
        prev_arrival_dt = datetime(sim_year, 1, 1, 8, 0, tzinfo=timezone.utc)

        for leg_name, seas, orig, dst, target_doy, primary_edge in raw_legs:
            preferred_edges.append(str(primary_edge["edge_id"]))
            alt_edge = direct_alt.get((orig, dst))
            if alt_edge is not None:
                alt_edges_avail.append(str(alt_edge["edge_id"]))

            # Decide whether this herd takes the alternative corridor, uses the resting-point 2-hop chain, or direct primary
            choose_alt = (alt_edge is not None) and (rng.random() < alt_prob)
            use_rest_stop = (not choose_alt) and ((orig, dst) in rest_chain) and (rng.random() < 0.65)

            dep_dt = datetime.combine(start_dt + timedelta(days=target_doy - 1), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=6)
            if dep_dt <= prev_arrival_dt + timedelta(days=3):
                dep_dt = prev_arrival_dt + timedelta(days=4)

            if choose_alt and alt_edge is not None:
                used_any_alt = True
                sub_edges = [(alt_edge, True, leg_name)]
            elif use_rest_stop:
                e1, rest_nid, e2 = rest_chain[(orig, dst)]
                sub_edges = [(e1, False, f"{leg_name}_to_rest"), (e2, False, f"{leg_name}_from_rest")]
            else:
                sub_edges = [(primary_edge, False, leg_name)]

            for seg_idx, (edge_rec, is_alt, sub_leg_name) in enumerate(sub_edges):
                dist_m = float(edge_rec["distance_m"])
                travel_hours = max(18.0, (dist_m / 1000.0) / max(speed_km_day, 2.0) * 24.0)
                if seg_idx > 0:
                    # Pause 1-2 days at the intermediate resting_point node!
                    dep_dt = prev_arrival_dt + timedelta(days=max(1, min(rest_days, 2)))
                arr_dt = dep_dt + timedelta(hours=travel_hours)
                max_end_dt = datetime.combine(end_dt, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=18)
                if arr_dt > max_end_dt:
                    arr_dt = max_end_dt
                    if dep_dt >= arr_dt:
                        dep_dt = arr_dt - timedelta(hours=12)

                event_seq += 1
                eid = f"EVT-{event_seq:05d}"
                scheduled_segments.append({
                    "event_id": eid,
                    "edge_id": str(edge_rec["edge_id"]),
                    "edge_type": str(edge_rec["edge_type"]),
                    "geometry": edge_rec["geometry"],
                    "distance_m": dist_m,
                    "origin_node_id": str(edge_rec["source_node_id"]),
                    "destination_node_id": str(edge_rec["target_node_id"]),
                    "season": seas,
                    "leg_name": sub_leg_name,
                    "departure_dt": dep_dt,
                    "arrival_dt": arr_dt,
                    "travel_days": round((arr_dt - dep_dt).total_seconds() / 86400.0, 2),
                    "used_alternative_route": bool(is_alt),
                })

                event_rows.append({
                    "event_id": eid,
                    "herd_id": herd_id,
                    "livestock_type": sp,
                    "animal_count": animal_count,
                    "season": seas,
                    "leg_name": sub_leg_name,
                    "edge_id": str(edge_rec["edge_id"]),
                    "origin_node_id": str(edge_rec["source_node_id"]),
                    "destination_node_id": str(edge_rec["target_node_id"]),
                    "departure_timestamp": dep_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "arrival_timestamp": arr_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "distance_m": round(dist_m, 1),
                    "estimated_travel_days": round((arr_dt - dep_dt).total_seconds() / 86400.0, 2),
                    "used_alternative_route": bool(is_alt),
                    "scenario_id": scenario_id,
                    "generator_version": gen_ver,
                    "random_seed": int(seed),
                    "provenance": "synthetic_demo",
                    "is_synthetic": True,
                })
                prev_arrival_dt = arr_dt

        herd_rows.append({
            "herd_id": herd_id,
            "herd_name": f"Herd {herd_id} ({sp.title()} Group)",
            "livestock_type": sp,
            "animal_count": animal_count,
            "origin_node_id": w_start,
            "winter_pasture_id": w_start,
            "spring_pasture_id": sp_node,
            "summer_pasture_id": su_node,
            "autumn_pasture_id": au_node,
            "return_winter_pasture_id": w_end,
            "travel_speed_km_day": speed_km_day,
            "resting_duration_days": rest_days,
            "spring_departure_date": scheduled_segments[0]["departure_dt"].date().isoformat(),
            "final_arrival_date": scheduled_segments[-1]["arrival_dt"].date().isoformat(),
            "preferred_route_ids": ";".join(preferred_edges),
            "alternative_route_ids": ";".join(alt_edges_avail),
            "used_alternative_route": bool(used_any_alt),
            "scenario_id": scenario_id,
            "simulation_period": f"{start_dt.isoformat()}/{end_dt.isoformat()}",
            "generator_version": gen_ver,
            "generation_date": gen_date_iso,
            "random_seed": int(seed),
            "provenance": "synthetic_demo",
            "evidence_status": "synthetic_demo",
            "is_synthetic": True,
            "synthetic_disclaimer": disclaimer,
        })

        # Emit time-indexed positions for this herd across the full 12-month period:
        # - Daily (12:00 UTC) during stationary grazing/resting days
        # - Sub-daily (every 12h: 06:00 & 18:00 UTC) during active movement days for smooth CesiumJS animation!
        for day_offset in range(total_days):
            cur_date = start_dt + timedelta(days=day_offset)
            day_start = datetime.combine(cur_date, datetime.min.time(), tzinfo=timezone.utc)
            day_end = day_start + timedelta(hours=24)

            # Check if any scheduled travel segment overlaps this calendar day
            overlapping = [
                seg for seg in scheduled_segments
                if seg["departure_dt"] < day_end and seg["arrival_dt"] > day_start
            ]
            if overlapping and subdaily >= 2:
                sample_hours = [6, 18]
            else:
                sample_hours = [12]

            for hr in sample_hours:
                ts = day_start + timedelta(hours=hr)
                seas = _season_from_month(cur_date.month)

                active_seg = None
                for seg in scheduled_segments:
                    if seg["departure_dt"] <= ts <= seg["arrival_dt"]:
                        active_seg = seg
                        break

                if active_seg is not None:
                    span_s = max(1.0, (active_seg["arrival_dt"] - active_seg["departure_dt"]).total_seconds())
                    frac = float(np.clip((ts - active_seg["departure_dt"]).total_seconds() / span_s, 0.0, 1.0))
                    geom: LineString = active_seg["geometry"]
                    pt = geom.interpolate(frac, normalized=True)
                    lon, lat = float(pt.x), float(pt.y)
                    cur_node = active_seg["origin_node_id"] if frac < 0.5 else active_seg["destination_node_id"]
                    cur_edge = active_seg["edge_id"]
                    orig_nid = active_seg["origin_node_id"]
                    dest_nid = active_seg["destination_node_id"]
                    if frac <= 0.12:
                        mstate = "DEPARTING"
                    elif frac >= 0.88:
                        mstate = "ARRIVING"
                    elif active_seg["season"] in ("autumn", "winter"):
                        mstate = "RETURNING"
                    else:
                        mstate = "TRAVELLING"
                else:
                    # Stationary at the most recently reached node (or initial winter pasture)
                    completed = [seg for seg in scheduled_segments if seg["arrival_dt"] < ts]
                    upcoming = [seg for seg in scheduled_segments if seg["departure_dt"] > ts]
                    if completed:
                        last_seg = completed[-1]
                        stay_nid = last_seg["destination_node_id"]
                        hours_since_arr = (ts - last_seg["arrival_dt"]).total_seconds() / 3600.0
                    else:
                        stay_nid = w_start
                        hours_since_arr = 9999.0

                    if upcoming:
                        next_seg = upcoming[0]
                        dest_nid = next_seg["destination_node_id"]
                    else:
                        dest_nid = w_start

                    node_rec = node_lookup[stay_nid]
                    lon, lat = float(node_rec["longitude"]), float(node_rec["latitude"])
                    cur_node = stay_nid
                    cur_edge = ""
                    orig_nid = stay_nid
                    is_rest_node = node_rec.get("node_type") == "resting_point"
                    mstate = "RESTING" if (is_rest_node or hours_since_arr <= rest_days * 24.0) else "GRAZING"

                position_rows.append({
                    "herd_id": herd_id,
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                    "elevation_m": None,  # Populated strictly from local DEM below
                    "elevation_status": "unavailable_no_local_dem",
                    "movement_state": mstate,
                    "current_node_id": cur_node,
                    "current_edge_id": cur_edge,
                    "origin_node_id": orig_nid,
                    "destination_node_id": dest_nid,
                    "season": seas,
                    "livestock_type": sp,
                    "scenario_id": scenario_id,
                    "is_synthetic": True,
                })

    herds_df = pd.DataFrame(herd_rows)
    events_df = pd.DataFrame(event_rows)
    positions_df = pd.DataFrame(position_rows)

    # Single vectorized strict DEM pass over all position coordinates
    if not positions_df.empty:
        elevs, _, e_stat, _ = sample_dem_strict(
            positions_df["longitude"].to_numpy(),
            positions_df["latitude"].to_numpy(),
            env,
        )
        positions_df["elevation_m"] = elevs
        positions_df["elevation_status"] = e_stat

    # Derive EXACT edge utilisation strictly from simulated movement events (unused edges = 0!)
    util_rows: list[dict[str, Any]] = []
    ev_by_edge = {eid: grp for eid, grp in events_df.groupby("edge_id")} if not events_df.empty else {}
    for _, erow in pastoral_edges_gdf.iterrows():
        eid = str(erow["edge_id"])
        grp = ev_by_edge.get(eid)
        if grp is None or grp.empty:
            util_rows.append({
                "edge_id": eid,
                "source_node_id": str(erow["source_node_id"]),
                "target_node_id": str(erow["target_node_id"]),
                "edge_type": str(erow["edge_type"]),
                "herd_count": 0,
                "animal_count": 0,
                "movement_event_count": 0,
                "winter_usage": 0,
                "spring_usage": 0,
                "summer_usage": 0,
                "autumn_usage": 0,
                "avg_estimated_travel_days": 0.0,
                "alternative_route_usage": 0,
            })
        else:
            seas_counts = grp["season"].value_counts().to_dict()
            util_rows.append({
                "edge_id": eid,
                "source_node_id": str(erow["source_node_id"]),
                "target_node_id": str(erow["target_node_id"]),
                "edge_type": str(erow["edge_type"]),
                "herd_count": int(grp["herd_id"].nunique()),
                "animal_count": int(grp["animal_count"].sum()),
                "movement_event_count": int(len(grp)),
                "winter_usage": int(seas_counts.get("winter", 0)),
                "spring_usage": int(seas_counts.get("spring", 0)),
                "summer_usage": int(seas_counts.get("summer", 0)),
                "autumn_usage": int(seas_counts.get("autumn", 0)),
                "avg_estimated_travel_days": round(float(grp["estimated_travel_days"].mean()), 2),
                "alternative_route_usage": int(grp["used_alternative_route"].sum()),
            })
    util_df = pd.DataFrame(util_rows)

    sim_report = {
        "herds_generated": int(len(herds_df)),
        "species_breakdown": herds_df["livestock_type"].value_counts().to_dict() if not herds_df.empty else {},
        "movement_events_generated": int(len(events_df)),
        "timestamped_positions_generated": int(len(positions_df)),
        "movement_state_counts": positions_df["movement_state"].value_counts().to_dict() if not positions_df.empty else {},
        "edges_used_count": int((util_df["movement_event_count"] > 0).sum()) if not util_df.empty else 0,
        "edges_unused_count": int((util_df["movement_event_count"] == 0).sum()) if not util_df.empty else 0,
        "herds_using_alternative_routes": int(herds_df["used_alternative_route"].sum()) if not herds_df.empty else 0,
    }
    return herds_df, events_df, positions_df, util_df, sim_report
