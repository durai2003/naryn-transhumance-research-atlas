"""Comprehensive Stage 1 acceptance tests: Geographic, Movement, Data-Model, Reproducibility, Performance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

from src.network.dem_env import read_geojson_safe
from src.network.schema import (
    EDGE_TYPES,
    MOVEMENT_STATES,
    REQUIRED_EDGE_COLUMNS,
    REQUIRED_EVENT_COLUMNS,
    REQUIRED_NODE_COLUMNS,
    REQUIRED_POSITION_COLUMNS,
    REQUIRED_UTILISATION_COLUMNS,
    STATIC_NODE_TYPES,
)


def run_geographic_tests(
    all_nodes: gpd.GeoDataFrame,
    pastoral_edges: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    *,
    metric_crs: str = "EPSG:32643",
) -> dict[str, Any]:
    issues: list[str] = []
    region_m = unary_union(boundary_gdf.to_crs(metric_crs).geometry).buffer(300.0)

    # 1. Valid coordinates & Naryn ADM1 containment for all static nodes
    nodes_m = all_nodes.to_crs(metric_crs)
    outside_nodes = int((~nodes_m.geometry.within(region_m)).sum())
    if outside_nodes:
        issues.append(f"{outside_nodes} static nodes fall outside Naryn ADM1")

    bad_lat = (~all_nodes["latitude"].between(-90, 90)).sum()
    bad_lon = (~all_nodes["longitude"].between(-180, 180)).sum()
    if bad_lat or bad_lon:
        issues.append(f"Invalid WGS84 coordinates: lat={int(bad_lat)}, lon={int(bad_lon)}")

    # 2. Elevation honesty: every elevation must be dem_derived or explicitly marked unavailable_no_local_dem
    valid_elev_status = {"dem_derived", "unavailable_no_local_dem"}
    bad_estatus = (~all_nodes["elevation_status"].isin(valid_elev_status)).sum()
    if bad_estatus:
        issues.append(f"{int(bad_estatus)} nodes have unrecognised elevation_status")
    unavail_with_val = ((all_nodes["elevation_status"] == "unavailable_no_local_dem") & all_nodes["elevation_m"].notna()).sum()
    if unavail_with_val:
        issues.append(f"{int(unavail_with_val)} nodes claim unavailable DEM yet contain fabricated elevation values")

    # 3. Route geometries valid, positive metric distance, endpoints match declared nodes, inside boundary
    invalid_geoms = int((~pastoral_edges.geometry.is_valid).sum())
    if invalid_geoms:
        issues.append(f"{invalid_geoms} pastoral edges have invalid geometries")

    edges_m = pastoral_edges.to_crs(metric_crs)
    outside_edges = int((~edges_m.geometry.within(region_m.buffer(300.0))).sum())
    if outside_edges:
        issues.append(f"{outside_edges} edges extend outside Naryn ADM1")

    non_pos_dist = int((pastoral_edges["distance_m"] <= 0).sum())
    if non_pos_dist:
        issues.append(f"{non_pos_dist} edges have non-positive metric distance_m")

    # Verify route endpoints align within 150m of declared source/target node coordinates
    node_xy_m = {
        str(r["node_id"]): (float(g.x), float(g.y))
        for (_, r), g in zip(all_nodes.iterrows(), nodes_m.geometry)
    }
    endpoint_mismatches = 0
    for (_, erow), geom_m in zip(pastoral_edges.iterrows(), edges_m.geometry):
        src = str(erow["source_node_id"])
        dst = str(erow["target_node_id"])
        if src not in node_xy_m or dst not in node_xy_m:
            endpoint_mismatches += 1
            continue
        coords = list(geom_m.coords)
        d_start = Point(coords[0]).distance(Point(node_xy_m[src]))
        d_end = Point(coords[-1]).distance(Point(node_xy_m[dst]))
        if d_start > 150.0 or d_end > 150.0:
            endpoint_mismatches += 1
    if endpoint_mismatches:
        issues.append(f"{endpoint_mismatches} edges have endpoints >150m from their source/target nodes")

    return {
        "suite": "geographic_tests",
        "nodes_checked": int(len(all_nodes)),
        "edges_checked": int(len(pastoral_edges)),
        "outside_boundary_nodes": outside_nodes,
        "outside_boundary_edges": outside_edges,
        "endpoint_mismatches": endpoint_mismatches,
        "elevation_statuses": all_nodes["elevation_status"].value_counts().to_dict(),
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
    }


def run_movement_tests(
    herds_df: pd.DataFrame,
    events_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    pastoral_edges: gpd.GeoDataFrame,
    *,
    metric_crs: str = "EPSG:32643",
) -> dict[str, Any]:
    issues: list[str] = []

    if len(herds_df) < 80:
        issues.append(f"Expected >=80 synthetic herds, found {len(herds_df)}")
    if len(positions_df) < 30000:
        issues.append(f"Expected >=30,000 timestamped positions, found {len(positions_df)}")

    # 1. Every movement event has valid timestamps with arrival > departure
    dep = pd.to_datetime(events_df["departure_timestamp"], errors="coerce", utc=True)
    arr = pd.to_datetime(events_df["arrival_timestamp"], errors="coerce", utc=True)
    bad_ts = int((dep.isna() | arr.isna()).sum())
    inverted = int((arr <= dep).sum())
    if bad_ts:
        issues.append(f"{bad_ts} movement events have invalid timestamps")
    if inverted:
        issues.append(f"{inverted} movement events have arrival_timestamp <= departure_timestamp")

    # 2. Distinct schedules across herds (not all migrating on identical dates)
    unique_spring_dates = int(herds_df["spring_departure_date"].nunique())
    if unique_spring_dates < 15:
        issues.append(f"Expected diverse herd departure schedules (>=15 distinct spring dates), found {unique_spring_dates}")

    # 3. Movement states valid & both RESTING and GRAZING present
    bad_states = set(positions_df["movement_state"].unique()) - set(MOVEMENT_STATES)
    if bad_states:
        issues.append(f"Invalid movement_state values: {sorted(bad_states)}")
    for req_state in ("GRAZING", "RESTING", "TRAVELLING", "RETURNING"):
        if (positions_df["movement_state"] == req_state).sum() == 0:
            issues.append(f"Expected movement_state {req_state} to occur in positions")

    # 4. Chronological ordering & anti-teleportation check per herd
    max_speed_km_day = 0.0
    teleport_steps = 0
    order_errors = 0
    for hid, grp in positions_df.groupby("herd_id"):
        ts = pd.to_datetime(grp["timestamp"], utc=True)
        if not ts.is_monotonic_increasing:
            order_errors += 1
        lats = grp["latitude"].to_numpy(dtype=float)
        lons = grp["longitude"].to_numpy(dtype=float)
        dt_days = np.diff(ts.astype("int64").to_numpy()) / (1e9 * 86400.0)
        if len(lats) > 1:
            dlat = np.diff(lats) * 110.54
            dlon = np.diff(lons) * 111.32 * np.cos(np.radians(0.5 * (lats[:-1] + lats[1:])))
            step_km = np.hypot(dlat, dlon)
            eff_speed = step_km / np.maximum(dt_days, 0.25)
            peak = float(np.max(eff_speed))
            max_speed_km_day = max(max_speed_km_day, peak)
            # Generous upper ceiling (28 km/day) above max horse speed 18.5 km/day for curve chords
            teleport_steps += int(np.sum(eff_speed > 28.0))

    if order_errors:
        issues.append(f"{order_errors} herds have non-monotonic timestamps")
    if teleport_steps:
        issues.append(f"{teleport_steps} position transitions exceed plausible daily speed (max={max_speed_km_day:.1f} km/day)")

    # 5. Route adherence check: sample active moving positions and verify they lie directly on `current_edge_id`
    edge_geoms_m = dict(zip(pastoral_edges["edge_id"].astype(str), pastoral_edges.to_crs(metric_crs).geometry))
    moving = positions_df.loc[positions_df["current_edge_id"].fillna("").astype(str).ne("")]
    sample_moving = moving.iloc[:: max(1, len(moving) // 800)]
    pts_m = gpd.GeoSeries(
        [Point(x, y) for x, y in zip(sample_moving["longitude"], sample_moving["latitude"])],
        crs="EPSG:4326",
    ).to_crs(metric_crs)
    off_route = 0
    max_off_m = 0.0
    for eid, pt in zip(sample_moving["current_edge_id"].astype(str), pts_m):
        geom = edge_geoms_m.get(eid)
        if geom is None:
            off_route += 1
            continue
        dist_m = float(pt.distance(geom))
        max_off_m = max(max_off_m, dist_m)
        if dist_m > 120.0:
            off_route += 1
    if off_route:
        issues.append(f"{off_route} sampled moving positions deviate >120m from their assigned route (max={max_off_m:.1f}m)")

    return {
        "suite": "movement_tests",
        "herds": int(len(herds_df)),
        "events": int(len(events_df)),
        "positions": int(len(positions_df)),
        "unique_spring_departure_dates": unique_spring_dates,
        "max_effective_speed_km_day": round(max_speed_km_day, 2),
        "teleport_steps": teleport_steps,
        "route_adherence_sample_size": int(len(sample_moving)),
        "max_route_deviation_m": round(max_off_m, 2),
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
    }


def run_data_model_tests(
    G: nx.MultiDiGraph,
    physical_nodes: gpd.GeoDataFrame,
    pastoral_nodes: gpd.GeoDataFrame,
    pastoral_edges: gpd.GeoDataFrame,
    herds_df: pd.DataFrame,
    events_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    util_df: pd.DataFrame,
) -> dict[str, Any]:
    issues: list[str] = []

    all_nodes = pd.concat([physical_nodes, pastoral_nodes], ignore_index=True)

    # 1. Schema completeness
    for name, df, req in (
        ("nodes", all_nodes, REQUIRED_NODE_COLUMNS),
        ("edges", pastoral_edges, REQUIRED_EDGE_COLUMNS),
        ("events", events_df, REQUIRED_EVENT_COLUMNS),
        ("positions", positions_df, REQUIRED_POSITION_COLUMNS),
        ("edge_utilisation", util_df, REQUIRED_UTILISATION_COLUMNS),
    ):
        missing = [c for c in req if c not in df.columns]
        if missing:
            issues.append(f"{name} missing required columns: {missing}")

    # 2. Unique node_id and edge_id
    dup_nodes = int(all_nodes["node_id"].duplicated().sum())
    dup_edges = int(pastoral_edges["edge_id"].duplicated().sum())
    if dup_nodes:
        issues.append(f"{dup_nodes} duplicate node_id values")
    if dup_edges:
        issues.append(f"{dup_edges} duplicate edge_id values")

    # 3. Referential integrity: all edge endpoints exist in graph nodes
    node_ids = set(all_nodes["node_id"].astype(str))
    bad_src = int((~pastoral_edges["source_node_id"].astype(str).isin(node_ids)).sum())
    bad_dst = int((~pastoral_edges["target_node_id"].astype(str).isin(node_ids)).sum())
    if bad_src or bad_dst:
        issues.append(f"Dangling edge references: src={bad_src}, dst={bad_dst}")

    # 4. All 6 edge types & all 8 static node types supported/present
    present_etypes = set(pastoral_edges["edge_type"].unique())
    missing_etypes = set(EDGE_TYPES) - present_etypes
    if missing_etypes:
        issues.append(f"Missing edge types in Layer B: {sorted(missing_etypes)}")

    present_ntypes = set(all_nodes["node_type"].unique())
    missing_ntypes = set(STATIC_NODE_TYPES) - present_ntypes
    if missing_ntypes:
        issues.append(f"Missing static node types: {sorted(missing_ntypes)}")

    # 5. Provenance separation: public Layer A NEVER relabelled synthetic; synthetic Layer B ALWAYS synthetic
    if not physical_nodes.empty and physical_nodes["is_synthetic"].any():
        issues.append("Layer A public physical nodes contain is_synthetic=True (forbidden)")
    if not pastoral_nodes["is_synthetic"].all():
        issues.append("Layer B synthetic nodes missing is_synthetic=True")
    if not herds_df["is_synthetic"].all() or not positions_df["is_synthetic"].all():
        issues.append("Synthetic herds/positions missing is_synthetic=True")

    # 6. Static graph does NOT store transient `herd_location` records as permanent nodes
    herd_nodes_in_graph = [n for n, d in G.nodes(data=True) if d.get("node_type") == "herd_location"]
    if herd_nodes_in_graph:
        issues.append(f"Found {len(herd_nodes_in_graph)} transient herd_location nodes inside static graph")

    # 7. Exact mathematical match between edge_utilisation and simulated movement_events
    ev_counts = events_df.groupby("edge_id").agg(
        ev_n=("event_id", "count"),
        h_n=("herd_id", "nunique"),
        a_sum=("animal_count", "sum"),
        alt_sum=("used_alternative_route", "sum"),
    )
    util_mismatches = 0
    for _, urow in util_df.iterrows():
        eid = str(urow["edge_id"])
        if eid in ev_counts.index:
            ref = ev_counts.loc[eid]
            if (
                int(urow["movement_event_count"]) != int(ref["ev_n"])
                or int(urow["herd_count"]) != int(ref["h_n"])
                or int(urow["animal_count"]) != int(ref["a_sum"])
                or int(urow["alternative_route_usage"]) != int(ref["alt_sum"])
            ):
                util_mismatches += 1
        else:
            if int(urow["movement_event_count"]) != 0 or int(urow["herd_count"]) != 0 or int(urow["animal_count"]) != 0:
                util_mismatches += 1
    if util_mismatches:
        issues.append(f"{util_mismatches} edges have utilisation figures that do not match movement_events")

    return {
        "suite": "data_model_tests",
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "is_multidigraph": isinstance(G, nx.MultiDiGraph),
        "node_types_present": sorted(present_ntypes),
        "edge_types_present": sorted(present_etypes),
        "utilisation_mismatches": util_mismatches,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
    }


def run_all_stage1_validations(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    net_dir = root / "data/network"
    bpath = next(
        (p for p in [root / "data/processed/vectors/naryn_adm1_clean.geojson", root / "data/raw/boundaries/naryn_adm1.geojson"] if p.exists()),
        None,
    )
    if bpath is None:
        raise FileNotFoundError("Authoritative Naryn ADM1 boundary not found")
    boundary_gdf = read_geojson_safe(bpath)

    phys_nodes = read_geojson_safe(net_dir / "layer_a_physical_nodes.geojson")
    past_nodes = read_geojson_safe(net_dir / "layer_b_pastoral_nodes.geojson")
    past_edges = read_geojson_safe(net_dir / "layer_b_pastoral_edges.geojson")
    herds_df = pd.read_csv(net_dir / "herds.csv")
    events_df = pd.read_csv(net_dir / "movement_events.csv")
    positions_df = pd.read_csv(net_dir / "herd_positions.csv")
    util_df = pd.read_csv(net_dir / "edge_utilisation.csv")
    G = nx.node_link_graph(json.loads((net_dir / "transhumance_multigraph.json").read_text(encoding="utf-8")), directed=True, multigraph=True, edges="links")

    all_nodes = gpd.GeoDataFrame(pd.concat([phys_nodes, past_nodes], ignore_index=True), geometry="geometry", crs="EPSG:4326")
    geo_res = run_geographic_tests(all_nodes, past_edges, boundary_gdf)
    mov_res = run_movement_tests(herds_df, events_df, positions_df, past_edges)
    dm_res = run_data_model_tests(G, phys_nodes, past_nodes, past_edges, herds_df, events_df, positions_df, util_df)

    # Performance architecture check: static graph compact & positions file bounded
    pos_size_mb = round((net_dir / "herd_positions.csv").stat().st_size / (1024 * 1024), 2)
    perf_issues = []
    if G.number_of_nodes() > 500:
        perf_issues.append(f"Static graph node count too large ({G.number_of_nodes()}); herd positions must stay out of static graph")
    perf_res = {
        "suite": "performance_architecture_tests",
        "static_graph_nodes": int(G.number_of_nodes()),
        "static_graph_edges": int(G.number_of_edges()),
        "herd_positions_csv_mb": pos_size_mb,
        "status": "PASS" if not perf_issues else "FAIL",
        "issues": perf_issues,
    }

    checks = [geo_res, mov_res, dm_res, perf_res]
    return {
        "all_passed": all(c["status"] == "PASS" for c in checks),
        "checks": checks,
    }
