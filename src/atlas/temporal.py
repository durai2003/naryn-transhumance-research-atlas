"""Shared temporal state, authoritative snapshot/corridor resolvers, and CesiumJS payload builder."""
from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from src.atlas.data_access import AtlasData
from src.atlas.terrain import export_local_dem_grid
from src.atlas.offline_basemap import build_offline_basemap

MOVING_STATES = frozenset({"departing", "moving", "seasonal_transition", "returning"})


def _gdf_to_compact_geojson(gdf: gpd.GeoDataFrame | None, props: list[str] | None = None) -> dict[str, Any]:
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    frame = gdf.to_crs("EPSG:4326").copy()
    if props is not None:
        keep = [c for c in props if c in frame.columns] + [frame.geometry.name]
        frame = frame[keep]
    return json.loads(frame.to_json())


def reconcile_trajectory_pasture_state(
    traj_df: pd.DataFrame,
    events_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Ensure current/previous pasture and next destination are logically consistent on every date.

    Resolves the known premature-flip bug in pre-generated trajectory CSVs (where
    `current_pasture_id` switched to `destination` at `frac >= 0.5` while still moving)
    strictly from the authoritative departure/arrival legs in `movement_events.csv`,
    without regenerating or modifying any dataset on disk.
    """
    if traj_df is None or traj_df.empty:
        return traj_df
    if events_df is None or events_df.empty:
        return traj_df

    out = traj_df.copy()
    dep = events_df.loc[events_df["event_type"] == "departure"].copy()
    arr = events_df.loc[events_df["event_type"] == "arrival"].copy()
    if dep.empty or arr.empty:
        return out

    dep["dep_date"] = dep["timestamp"].astype(str).str.slice(0, 10)
    arr["arr_date"] = arr["timestamp"].astype(str).str.slice(0, 10)
    legs = dep.merge(
        arr[["climate_scenario", "herd_id", "leg_name", "route_id", "pasture_id", "arr_date"]],
        on=["climate_scenario", "herd_id", "leg_name", "route_id"],
        suffixes=("_orig", "_dest"),
    ).sort_values(["climate_scenario", "herd_id", "dep_date"])

    legs_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (sc, hid), grp in legs.groupby(["climate_scenario", "herd_id"]):
        legs_map[(str(sc), str(hid))] = grp.to_dict(orient="records")

    cur_col = out["current_pasture_id"].astype(str).to_numpy(copy=True)
    nxt_col = out["next_pasture_id"].astype(str).to_numpy(copy=True)
    rid_col = out["route_id"].astype(str).to_numpy(copy=True)

    for (sc, hid), idxs in out.groupby(["climate_scenario", "herd_id"]).groups.items():
        h_legs = legs_map.get((str(sc), str(hid)))
        if not h_legs:
            continue
        idx_arr = np.asarray(idxs)
        dates_arr = out.loc[idx_arr, "date"].astype(str).to_numpy()
        states_arr = out.loc[idx_arr, "movement_state"].astype(str).to_numpy()
        n_legs = len(h_legs)
        for row_i, d_str, st in zip(idx_arr, dates_arr, states_arr):
            matched = False
            for li, leg in enumerate(h_legs):
                if leg["dep_date"] <= d_str <= leg["arr_date"]:
                    matched = True
                    next_leg = h_legs[(li + 1) % n_legs]
                    if d_str == leg["arr_date"] or st == "arrived":
                        cur_col[row_i] = str(leg["pasture_id_dest"])
                        nxt_col[row_i] = str(next_leg["pasture_id_dest"])
                    else:
                        cur_col[row_i] = str(leg["pasture_id_orig"])
                        nxt_col[row_i] = str(leg["pasture_id_dest"])
                    rid_col[row_i] = str(leg["route_id"])
                    break
            if not matched:
                upcoming = next((leg for leg in h_legs if leg["dep_date"] > d_str), None)
                if upcoming is not None:
                    cur_col[row_i] = str(upcoming["pasture_id_orig"])
                    nxt_col[row_i] = str(upcoming["pasture_id_dest"])
                    rid_col[row_i] = str(upcoming["route_id"])
                else:
                    cur_col[row_i] = str(h_legs[-1]["pasture_id_dest"])
                    nxt_col[row_i] = str(h_legs[0]["pasture_id_dest"])
                    rid_col[row_i] = str(h_legs[0]["route_id"])

    out["current_pasture_id"] = cur_col
    out["next_pasture_id"] = nxt_col
    out["route_id"] = rid_col
    return out


def _get_reconciled_trajectories(data: AtlasData) -> pd.DataFrame:
    cached = getattr(data, "_reconciled_trajectories", None)
    if cached is not None:
        return cached
    rec = reconcile_trajectory_pasture_state(data.herd_trajectories, data.movement_events)
    setattr(data, "_reconciled_trajectories", rec)
    return rec


def resolve_climate_on_date(
    data: AtlasData,
    scenario_id: str,
    selected_date: str,
) -> dict[str, Any]:
    """Return authoritative NASA POWER climate context for (scenario_id, selected_date)."""
    sc = data.scenario_climate
    if sc is None or sc.empty:
        return {"date": selected_date, "season": "unknown", "t2m_c": None, "precip_mm": None}
    sub = sc.loc[(sc["scenario_id"] == scenario_id) & (sc["date_str"] == selected_date)]
    if sub.empty:
        return {"date": selected_date, "season": "unknown", "t2m_c": None, "precip_mm": None}
    r = sub.iloc[0]
    return {
        "date": str(r["date_str"]),
        "season": str(r["season"]),
        "t2m_c": None if pd.isna(r["scenario_t2m_c"]) else round(float(r["scenario_t2m_c"]), 2),
        "precip_mm": None if pd.isna(r["scenario_precip_mm_day"]) else round(float(r["scenario_precip_mm_day"]), 2),
    }


def resolve_herd_snapshot(
    data: AtlasData,
    *,
    scenario_id: str,
    selected_date: str,
    species_filter: list[str] | None = None,
    selected_herd_id: str | None = None,
) -> dict[str, Any]:
    """Single authoritative snapshot for a date, scenario, species filter, and selected herd."""
    traj = _get_reconciled_trajectories(data)
    climate = resolve_climate_on_date(data, scenario_id, selected_date)

    if traj is None or traj.empty:
        return {
            "date": selected_date,
            "scenario_id": scenario_id,
            "climate": climate,
            "total_herds": 0,
            "moving_herds": 0,
            "stationary_herds": 0,
            "herd": None,
        }

    day_slice = traj.loc[(traj["climate_scenario"] == scenario_id) & (traj["date"] == selected_date)]
    if species_filter:
        day_slice = day_slice.loc[day_slice["species"].isin(species_filter)]
    day_slice = day_slice.sort_values("herd_id").reset_index(drop=True)

    if day_slice.empty:
        return {
            "date": selected_date,
            "scenario_id": scenario_id,
            "climate": climate,
            "total_herds": 0,
            "moving_herds": 0,
            "stationary_herds": 0,
            "herd": None,
        }

    moving_mask = day_slice["movement_state"].isin(MOVING_STATES)
    moving_count = int(moving_mask.sum())
    total_count = int(len(day_slice))

    chosen_row = None
    if selected_herd_id and selected_herd_id != "All herds":
        match = day_slice.loc[day_slice["herd_id"] == selected_herd_id]
        if not match.empty:
            chosen_row = match.iloc[0]
    if chosen_row is None:
        moving_rows = day_slice.loc[moving_mask]
        chosen_row = moving_rows.iloc[0] if not moving_rows.empty else day_slice.iloc[0]

    # Always synchronize herd climate context with scenario_climate for exact date agreement
    t_val = climate["t2m_c"] if climate["t2m_c"] is not None else (
        None if pd.isna(chosen_row["temperature_context"]) else round(float(chosen_row["temperature_context"]), 2)
    )
    p_val = climate["precip_mm"] if climate["precip_mm"] is not None else (
        None if pd.isna(chosen_row["precipitation_context"]) else round(float(chosen_row["precipitation_context"]), 2)
    )

    return {
        "date": selected_date,
        "scenario_id": scenario_id,
        "climate": climate,
        "total_herds": total_count,
        "moving_herds": moving_count,
        "stationary_herds": total_count - moving_count,
        "herd": {
            "herd_id": str(chosen_row["herd_id"]),
            "species": str(chosen_row["species"]),
            "group_size": int(chosen_row["group_size"]),
            "origin_class": str(chosen_row["origin_class"]),
            "provenance": str(chosen_row.get("provenance", "synthetic_demo")),
            "movement_state": str(chosen_row["movement_state"]),
            "is_moving": str(chosen_row["movement_state"]) in MOVING_STATES,
            "movement_speed_km_day": round(float(chosen_row["movement_speed_km_day"]), 2),
            "latitude": round(float(chosen_row["latitude"]), 5),
            "longitude": round(float(chosen_row["longitude"]), 5),
            "elevation_m": round(float(chosen_row["elevation_m"]), 1),
            "current_pasture_id": str(chosen_row["current_pasture_id"]),
            "next_pasture_id": str(chosen_row["next_pasture_id"]),
            "route_id": str(chosen_row["route_id"]),
            "suitability_score": round(float(chosen_row["pasture_suitability_score"]), 3),
            "temperature_c": t_val,
            "precipitation_mm": p_val,
        },
    }


def _orient_profile(
    profile: list[dict[str, Any]],
    *,
    reversed_dir: bool,
) -> tuple[list[dict[str, Any]], float, float, float]:
    """Return (oriented_profile, distance_km, gain_m, loss_m)."""
    if not profile:
        return [], 0.0, 0.0, 0.0
    total_km = float(profile[-1].get("distance_km", 0.0))
    if reversed_dir:
        oriented = [
            {
                **p,
                "distance_km": round(max(0.0, total_km - float(p.get("distance_km", 0.0))), 2),
            }
            for p in reversed(profile)
        ]
    else:
        oriented = [dict(p) for p in profile]
    elevs = [float(p["elevation_m"]) for p in oriented if p.get("elevation_m") is not None]
    if len(elevs) >= 2:
        diffs = np.diff(np.asarray(elevs, dtype=float))
        gain = round(float(np.sum(diffs[diffs > 0])), 1)
        loss = round(float(np.sum(-diffs[diffs < 0])), 1)
    else:
        gain, loss = 0.0, 0.0
    return oriented, round(total_km, 2), gain, loss


def resolve_corridor_profile(
    data: AtlasData,
    route_id: str | None,
    *,
    origin_pasture_id: str | None = None,
    dest_pasture_id: str | None = None,
) -> dict[str, Any] | None:
    """Resolve the exact corridor elevation profile for a herd's route_id (direct, reversed, or composite A+B).

    Returns None when no matching modelled corridor geometry exists so callers can display an
    honest message instead of showing an unrelated fallback corridor.
    """
    routes = data.migration_corridors
    if routes is None or routes.empty or not route_id:
        return None
    rid_str = str(route_id).strip()
    if not rid_str or rid_str == "nan":
        return None

    by_id = {str(r["route_id"]): r for _, r in routes.iterrows()}

    def _parse_prof(row) -> list[dict[str, Any]]:
        raw = row.get("elevation_profile_json") if hasattr(row, "get") else row["elevation_profile_json"]
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except Exception:
                return []
        return []

    # 1. Exact single corridor match
    if rid_str in by_id:
        row = by_id[rid_str]
        prof = _parse_prof(row)
        if not prof:
            return None
        c_orig = str(row.get("origin_pasture_id", ""))
        c_dest = str(row.get("destination_pasture_id", ""))
        rev = bool(origin_pasture_id and dest_pasture_id and origin_pasture_id == c_dest and dest_pasture_id == c_orig)
        oriented, dist_km, gain, loss = _orient_profile(prof, reversed_dir=rev)
        return {
            "route_id": rid_str,
            "name": str(row.get("name", rid_str)),
            "origin_pasture_id": origin_pasture_id or (c_dest if rev else c_orig),
            "destination_pasture_id": dest_pasture_id or (c_orig if rev else c_dest),
            "distance_km": dist_km,
            "elevation_gain_m": gain,
            "elevation_loss_m": loss,
            "reversed": rev,
            "profile": oriented,
        }

    # 2. Composite corridor (e.g., SYN-CORRIDOR-003+SYN-CORRIDOR-012)
    if "+" in rid_str:
        parts = [p.strip() for p in rid_str.split("+") if p.strip()]
        stitched: list[dict[str, Any]] = []
        cum_km = 0.0
        cur_node = origin_pasture_id
        for part_id in parts:
            if part_id not in by_id:
                return None
            row = by_id[part_id]
            prof = _parse_prof(row)
            if not prof:
                return None
            c_orig = str(row.get("origin_pasture_id", ""))
            c_dest = str(row.get("destination_pasture_id", ""))
            rev = bool(cur_node and cur_node == c_dest)
            oriented, seg_km, _, _ = _orient_profile(prof, reversed_dir=rev)
            for idx_p, pt in enumerate(oriented):
                if stitched and idx_p == 0:
                    continue
                stitched.append({**pt, "distance_km": round(cum_km + float(pt["distance_km"]), 2)})
            cum_km += seg_km
            cur_node = c_orig if rev else c_dest
        if len(stitched) >= 2:
            elevs = [float(p["elevation_m"]) for p in stitched if p.get("elevation_m") is not None]
            diffs = np.diff(np.asarray(elevs, dtype=float)) if len(elevs) >= 2 else np.array([])
            gain = round(float(np.sum(diffs[diffs > 0])), 1) if diffs.size else 0.0
            loss = round(float(np.sum(-diffs[diffs < 0])), 1) if diffs.size else 0.0
            return {
                "route_id": rid_str,
                "name": f"Composite corridor ({rid_str})",
                "origin_pasture_id": origin_pasture_id or "",
                "destination_pasture_id": dest_pasture_id or "",
                "distance_km": round(cum_km, 2),
                "elevation_gain_m": gain,
                "elevation_loss_m": loss,
                "reversed": False,
                "profile": stitched,
            }

    # 3. Direct endpoint match for FALLBACK routes if an existing corridor connects the same pair
    if origin_pasture_id and dest_pasture_id:
        for cid, row in by_id.items():
            c_orig = str(row.get("origin_pasture_id", ""))
            c_dest = str(row.get("destination_pasture_id", ""))
            if (origin_pasture_id == c_orig and dest_pasture_id == c_dest) or (
                origin_pasture_id == c_dest and dest_pasture_id == c_orig
            ):
                prof = _parse_prof(row)
                if prof:
                    rev = origin_pasture_id == c_dest
                    oriented, dist_km, gain, loss = _orient_profile(prof, reversed_dir=rev)
                    return {
                        "route_id": cid,
                        "name": str(row.get("name", cid)),
                        "origin_pasture_id": origin_pasture_id,
                        "destination_pasture_id": dest_pasture_id,
                        "distance_km": dist_km,
                        "elevation_gain_m": gain,
                        "elevation_loss_m": loss,
                        "reversed": rev,
                        "profile": oriented,
                    }

    return None


def build_cesium_payload(
    data: AtlasData,
    *,
    scenario_id: str,
    selected_date: str,
    species_filter: list[str],
    selected_herd_id: str | None,
    camera_mode: str = "regional_3d",
    basemap: str = "esri_satellite",
    vertical_exaggeration: float = 1.35,
    show_layers: dict[str, bool] | None = None,
    playback_speed: float = 10.0,
    cesium_ion_token: str = "",
    google_maps_api_key: str = "",
) -> dict[str, Any]:
    """Assemble a clean, synchronized JSON payload for the CesiumJS 3D globe."""
    layers = show_layers or {}

    traj = _get_reconciled_trajectories(data)
    if not traj.empty:
        traj = traj.loc[traj["climate_scenario"] == scenario_id]
        if species_filter:
            traj = traj.loc[traj["species"].isin(species_filter)]

    # Compact per-herd sampled trajectory arrays:
    # [iso_time, lon, lat, elev, state, current_pasture_id, next_pasture_id, suit, speed, route_id]
    # Keep every moving/arrived day, every transition boundary day (day before & day after moving),
    # selected_date, and periodic stationary anchors so JS lookup on ANY date returns exact state!
    herds_payload: list[dict[str, Any]] = []
    if not traj.empty:
        herd_meta = data.synthetic_herds.set_index("herd_id").to_dict(orient="index") if not data.synthetic_herds.empty else {}
        for hid, grp in traj.groupby("herd_id", sort=True):
            grp = grp.sort_values("date").reset_index(drop=True)
            is_active = grp["movement_state"].isin(MOVING_STATES | {"arrived"})
            state_changed = grp["movement_state"].ne(grp["movement_state"].shift(1))
            pasture_changed = grp["current_pasture_id"].ne(grp["current_pasture_id"].shift(1))
            keep_mask = (
                is_active
                | state_changed
                | pasture_changed
                | (grp.index % 7 == 0)
                | (grp.index == 0)
                | (grp.index == len(grp) - 1)
                | (grp["date"] == selected_date)
            )
            sub = grp.loc[keep_mask]
            samples = [
                [
                    str(r.timestamp),
                    round(float(r.longitude), 5),
                    round(float(r.latitude), 5),
                    round(float(r.elevation_m), 1),
                    str(r.movement_state),
                    str(r.current_pasture_id),
                    str(r.next_pasture_id),
                    round(float(r.pasture_suitability_score), 3),
                    round(float(r.movement_speed_km_day), 2),
                    str(r.route_id),
                ]
                for r in sub.itertuples(index=False)
            ]
            meta = herd_meta.get(str(hid), {})
            herds_payload.append({
                "herd_id": str(hid),
                "herd_name": str(meta.get("herd_name", hid)),
                "species": str(grp["species"].iloc[0]),
                "group_size": int(grp["group_size"].iloc[0]),
                "winter_pasture_id": str(meta.get("winter_pasture_id", "")),
                "summer_pasture_id": str(meta.get("summer_pasture_id", "")),
                "origin_class": str(grp["origin_class"].iloc[0]),
                "provenance": "synthetic_demo",
                "is_synthetic": True,
                "samples": samples,
            })

    # Climate timeline array for HUD & right panel readout during playback
    clim_series = []
    if not data.scenario_climate.empty:
        sc = data.scenario_climate.loc[data.scenario_climate["scenario_id"] == scenario_id].sort_values("date_str")
        clim_series = [
            {
                "date": str(r.date_str),
                "season": str(r.season),
                "t2m_c": None if pd.isna(r.scenario_t2m_c) else round(float(r.scenario_t2m_c), 2),
                "precip_mm": None if pd.isna(r.scenario_precip_mm_day) else round(float(r.scenario_precip_mm_day), 2),
                "precip_30d_mm": None if pd.isna(r.precip_30d_mm) else round(float(r.precip_30d_mm), 1),
                "snow_risk": bool(r.high_alpine_snow_risk),
            }
            for r in sc.itertuples(index=False)
        ]

    current_season = "summer"
    if clim_series:
        match = next((c for c in clim_series if c["date"] == selected_date), None)
        if match:
            current_season = match["season"]

    pastures_gdf = data.synthetic_pastures
    if pastures_gdf is not None and not pastures_gdf.empty and not data.suitability_scores.empty:
        s_map = (
            data.suitability_scores.loc[
                (data.suitability_scores["scenario_id"] == scenario_id)
                & (data.suitability_scores["season"] == current_season)
            ]
            .set_index("pasture_id")["suitability_score"]
            .to_dict()
        )
        pastures_gdf = pastures_gdf.copy()
        pastures_gdf["current_suitability"] = pastures_gdf["pasture_id"].map(s_map).fillna(0.5).round(3)

    events_payload = []
    if not data.movement_events.empty:
        ev = data.movement_events.loc[data.movement_events["climate_scenario"] == scenario_id]
        if selected_herd_id and selected_herd_id != "All herds":
            ev = ev.loc[ev["herd_id"] == selected_herd_id]
        else:
            ev = ev.head(60)
        events_payload = [
            {
                "event_id": str(r.event_id),
                "timestamp": str(r.timestamp),
                "herd_id": str(r.herd_id),
                "event_type": str(r.event_type),
                "leg_name": str(r.leg_name),
                "pasture_id": str(r.pasture_id),
                "changed": bool(r.scenario_changed_route),
            }
            for r in ev.itertuples(index=False)
        ]

    locations_payload = []
    if not data.locations.empty:
        for _, r in data.locations.dropna(subset=["latitude", "longitude"]).iterrows():
            locations_payload.append({
                "location_id": str(r.get("location_id", "")),
                "name": str(r.get("location_name", "Location")),
                "qid": str(r.get("external_identifier", "")),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "provenance": "Wikidata verified entity reference (not a weather station)",
            })

    start_iso = f"{clim_series[0]['date']}T00:00:00Z" if clim_series else "2024-01-01T00:00:00Z"
    end_iso = f"{clim_series[-1]['date']}T23:59:59Z" if clim_series else "2024-12-31T23:59:59Z"

    tstat = data.terrain_status
    terrain_meta = {
        "has_local_dem": bool(tstat.has_local_dem) if tstat else False,
        "local_dem_source": tstat.source_name if tstat else "None",
        "local_dem_resolution": tstat.resolution_desc if tstat else "N/A",
    }

    return {
        "config": {
            "cesiumIonToken": cesium_ion_token,
            "googleMapsApiKey": google_maps_api_key,
            "cameraMode": camera_mode,
            "basemap": basemap,
            "verticalExaggeration": float(vertical_exaggeration),
            "playbackSpeed": float(playback_speed),
            "scenarioId": scenario_id,
            "selectedDate": selected_date,
            "selectedHerdId": selected_herd_id if selected_herd_id != "All herds" else None,
            "startTime": start_iso,
            "endTime": end_iso,
            "currentTime": f"{selected_date}T12:00:00Z",
            "layers": layers,
            "terrainMeta": terrain_meta,
        },
        "boundary": _gdf_to_compact_geojson(data.boundary, ["shapeName", "shapeISO", "shapeID"]),
        "pastures": _gdf_to_compact_geojson(
            pastures_gdf,
            ["pasture_id", "pasture_name", "elevation_m", "slope_degrees", "elevation_band",
             "landcover_class", "water_access_indicator", "seasonal_suitability", "current_suitability",
             "elevation_source", "is_synthetic", "provenance"],
        ),
        "corridors": _gdf_to_compact_geojson(
            data.migration_corridors,
            ["route_id", "name", "origin_pasture_id", "destination_pasture_id", "season",
             "distance_km", "elevation_gain_m", "elevation_loss_m", "avg_slope_deg",
             "estimated_travel_days", "route_generation_method", "is_synthetic", "provenance"],
        ),
        "osm": {
            "roads": _gdf_to_compact_geojson(data.osm_roads, ["fclass", "name", "ref"]) if layers.get("osm_roads", True) else {"type": "FeatureCollection", "features": []},
            "waterways": _gdf_to_compact_geojson(data.osm_waterways, ["fclass", "name"]) if layers.get("osm_water", True) else {"type": "FeatureCollection", "features": []},
            "waterbodies": _gdf_to_compact_geojson(data.osm_waterbodies, ["fclass", "name"]) if layers.get("osm_water", True) else {"type": "FeatureCollection", "features": []},
            "settlements": _gdf_to_compact_geojson(data.osm_settlements, ["fclass", "name", "population"]) if layers.get("osm_settlements", True) else {"type": "FeatureCollection", "features": []},
        },
        "locations": locations_payload,
        "herds": herds_payload,
        "events": events_payload,
        "climate": clim_series,
        "localDemGrid": export_local_dem_grid(data.root),
        "offlineBasemap": build_offline_basemap(data.root),
    }
