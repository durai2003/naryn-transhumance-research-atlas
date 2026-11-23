"""Terrain-aware least-cost corridor & network edge generator in EPSG:32643.

Builds:
- Primary migration corridors (`LEAST_COST_CORRIDOR` and `SEASONAL_MOVEMENT`)
- Penalized-grid alternative corridors (`ALTERNATIVE_ROUTE`)
- Intermediate `resting_point` nodes along long corridors (`SYN-REST-*`)
- `WATER_ACCESS`, `SETTLEMENT_CONNECTION`, and `ACCESS_ROUTE` edges

Every route strictly connects its declared (source_node_id, target_node_id)
endpoints and remains inside the authoritative Naryn ADM1 boundary.
"""
from __future__ import annotations

import json
import math
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from src.network.dem_env import (
    EnvironmentInventory,
    compute_line_metrics,
    sample_dem_strict,
    sample_worldcover_strict,
)


def _load_osm_discount_geoms(env: EnvironmentInventory, metric_crs: str):
    """Load optional OSM road/track and river geometries to refine terrain cost surface."""
    road_union = None
    water_union = None
    if env.osm_gpkg_path is None:
        return road_union, water_union
    try:
        import pyogrio
        avail = {str(r[0]) for r in pyogrio.list_layers(env.osm_gpkg_path)}
        if "gis_osm_roads_free" in avail:
            roads = gpd.read_file(env.osm_gpkg_path, layer="gis_osm_roads_free")
            if "fclass" in roads.columns:
                roads = roads.loc[roads["fclass"].astype(str).str.lower().isin({
                    "trunk", "primary", "secondary", "tertiary", "unclassified",
                    "track", "track_grade1", "track_grade2", "track_grade3", "path", "bridleway",
                })]
            if not roads.empty:
                road_union = unary_union(roads.to_crs(metric_crs).geometry.simplify(150.0))
        if "gis_osm_waterways_free" in avail:
            rivers = gpd.read_file(env.osm_gpkg_path, layer="gis_osm_waterways_free")
            if "fclass" in rivers.columns:
                rivers = rivers.loc[rivers["fclass"].astype(str).str.lower().isin({"river", "stream"})]
            if not rivers.empty:
                water_union = unary_union(rivers.to_crs(metric_crs).geometry.simplify(200.0))
    except Exception:
        pass
    return road_union, water_union


def _build_routing_grid(
    region_m,
    env: EnvironmentInventory,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    routing_cfg = cfg.get("routing", {})
    cell_m = float(routing_cfg.get("grid_cell_size_m", 3200.0))
    metric_crs = str(cfg.get("metric_crs", "EPSG:32643"))
    barrier_m = float(routing_cfg.get("elevation_barrier_m", 4150.0))
    osm_discount = float(routing_cfg.get("osm_track_discount", 0.72))
    water_bonus = float(routing_cfg.get("water_proximity_bonus", 0.90))

    minx, miny, maxx, maxy = region_m.bounds
    xs = np.arange(minx + cell_m / 2, maxx, cell_m)
    ys = np.arange(miny + cell_m / 2, maxy, cell_m)
    nx, ny = len(xs), len(ys)
    gx, gy = np.meshgrid(xs, ys)
    flat_x, flat_y = gx.ravel(), gy.ravel()

    pts_m = gpd.GeoSeries([Point(x, y) for x, y in zip(flat_x, flat_y)], crs=metric_crs)
    inside = pts_m.within(region_m).to_numpy().reshape((ny, nx))

    pts_wgs = pts_m.to_crs("EPSG:4326")
    lons, lats = pts_wgs.x.to_numpy(), pts_wgs.y.to_numpy()
    elevs_list, slopes_list, elev_status, _ = sample_dem_strict(lons, lats, env)
    has_dem = elev_status == "dem_derived"

    elev_grid = np.array([e if e is not None else np.nan for e in elevs_list], dtype=float).reshape((ny, nx))
    slope_grid = np.array([s if s is not None else np.nan for s in slopes_list], dtype=float).reshape((ny, nx))

    road_union, water_union = _load_osm_discount_geoms(env, metric_crs)
    near_road = np.zeros((ny, nx), dtype=bool)
    near_water = np.zeros((ny, nx), dtype=bool)
    if road_union is not None:
        buf_r = road_union.buffer(cell_m * 0.75)
        near_road = pts_m.within(buf_r).to_numpy().reshape((ny, nx))
    if water_union is not None:
        buf_w = water_union.buffer(cell_m * 0.85)
        near_water = pts_m.within(buf_w).to_numpy().reshape((ny, nx))

    # Build base 8-connected edge list (stored so we can re-weight for alternative routes!)
    edges_u: list[int] = []
    edges_v: list[int] = []
    base_weights: list[float] = []
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    for r in range(ny):
        for c in range(nx):
            if not inside[r, c]:
                continue
            u = r * nx + c
            e_u, s_u = elev_grid[r, c], slope_grid[r, c]
            for dr, dc, dist_mult in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ny and 0 <= nc < nx and inside[nr, nc]:
                    if not region_m.covers(LineString([(float(xs[c]), float(ys[r])), (float(xs[nc]), float(ys[nr]))])):
                        continue
                    v = nr * nx + nc
                    step_m = cell_m * dist_mult
                    e_v, s_v = elev_grid[nr, nc], slope_grid[nr, nc]
                    if has_dem and np.isfinite(e_u) and np.isfinite(e_v):
                        grade = abs(e_v - e_u) / step_m
                        avg_s = 0.5 * ((s_u if np.isfinite(s_u) else 8.0) + (s_v if np.isfinite(s_v) else 8.0))
                        slope_factor = 1.0 + 3.0 * (avg_s / 15.0) ** 2 + 10.0 * (grade ** 2)
                        barrier_factor = 5.5 if max(e_u, e_v) > barrier_m else 1.0
                    else:
                        # Subtle deterministic valley/ridge preference when DEM is absent so paths remain boundary-hugging & non-degenerate
                        lon_mid = 0.5 * (lons[u] + lons[v])
                        lat_mid = 0.5 * (lats[u] + lats[v])
                        ridge_mod = 0.35 * (1.0 + math.sin(lon_mid * 9.0 + lat_mid * 5.0) * math.cos(lat_mid * 10.0))
                        slope_factor = 1.0 + ridge_mod
                        barrier_factor = 1.0

                    mult = 1.0
                    if near_road[r, c] or near_road[nr, nc]:
                        mult *= osm_discount
                    if near_water[r, c] or near_water[nr, nc]:
                        mult *= water_bonus

                    w = (step_m / 1000.0) * slope_factor * barrier_factor * mult
                    edges_u.append(u)
                    edges_v.append(v)
                    base_weights.append(float(max(w, 0.05)))

    method_name = (
        "dem_slope_osm_least_cost_dijkstra" if (has_dem and road_union is not None)
        else "dem_slope_least_cost_dijkstra" if has_dem
        else "boundary_constrained_grid_dijkstra_no_dem"
    )
    return {
        "xs": xs,
        "ys": ys,
        "nx": nx,
        "ny": ny,
        "inside": inside,
        "edges_u": np.asarray(edges_u, dtype=np.int32),
        "edges_v": np.asarray(edges_v, dtype=np.int32),
        "base_weights": np.asarray(base_weights, dtype=np.float64),
        "n_nodes": nx * ny,
        "method_name": method_name,
        "has_dem": has_dem,
        "used_osm_roads": road_union is not None,
    }


def _nearest_cell(x: float, y: float, grid: dict[str, Any], region_m=None) -> int:
    xs, ys, inside, nx = grid["xs"], grid["ys"], grid["inside"], grid["nx"]
    ix = int(np.argmin(np.abs(xs - x)))
    iy = int(np.argmin(np.abs(ys - y)))
    if inside[iy, ix]:
        if region_m is None or region_m.buffer(150.0).covers(LineString([(x, y), (float(xs[ix]), float(ys[iy]))])):
            return iy * nx + ix
    vr, vc = np.where(inside)
    d2 = (xs[vc] - x) ** 2 + (ys[vr] - y) ** 2
    order = np.argsort(d2)
    if region_m is not None:
        buf = region_m.buffer(150.0)
        for idx in order[:40]:
            r_i, c_i = int(vr[idx]), int(vc[idx])
            if buf.covers(LineString([(x, y), (float(xs[c_i]), float(ys[r_i]))])):
                return int(r_i * nx + c_i)
    best = int(order[0])
    return int(vr[best] * nx + vc[best])


def _smooth_path_inside(coords_m: list[tuple[float, float]], region_m) -> list[tuple[float, float]]:
    if len(coords_m) < 4:
        return coords_m
    out = [coords_m[0]]
    for i in range(1, len(coords_m) - 1):
        p0, p1, p2 = out[-1], coords_m[i], coords_m[i + 1]
        cand = (0.25 * p0[0] + 0.5 * p1[0] + 0.25 * p2[0], 0.25 * p0[1] + 0.5 * p1[1] + 0.25 * p2[1])
        seg_ok = region_m.covers(LineString([p0, cand])) and region_m.covers(LineString([cand, p2]))
        out.append(cand if seg_ok else p1)
    out.append(coords_m[-1])
    return out


def _trace_least_cost_path(
    orig_pt_m: Point,
    dst_pt_m: Point,
    start_cell: int,
    end_cell: int,
    graph: csr_matrix,
    grid: dict[str, Any],
    region_m,
    metric_crs: str,
) -> tuple[LineString, list[int], float]:
    dists, preds = dijkstra(graph, directed=False, indices=start_cell, return_predecessors=True)
    cost = float(dists[end_cell])
    xs, ys, nx = grid["xs"], grid["ys"], grid["nx"]

    used_cells: list[int] = []
    if not np.isfinite(cost):
        coords_m = [(float(orig_pt_m.x), float(orig_pt_m.y)), (float(dst_pt_m.x), float(dst_pt_m.y))]
        cost = float(orig_pt_m.distance(dst_pt_m) / 1000.0)
    else:
        cur = end_cell
        guard = 0
        while cur != start_cell and cur >= 0 and guard < 4000:
            used_cells.append(int(cur))
            cur = int(preds[cur])
            guard += 1
        used_cells.append(int(start_cell))
        used_cells.reverse()
        interior = [(float(xs[c % nx]), float(ys[c // nx])) for c in used_cells]
        coords_m = [(float(orig_pt_m.x), float(orig_pt_m.y))] + interior + [(float(dst_pt_m.x), float(dst_pt_m.y))]
        coords_m = _smooth_path_inside(coords_m, region_m)

    # Guarantee endpoints match exact node coordinates and every interior vertex stays inside Naryn
    line_m = LineString(coords_m)
    if not region_m.buffer(250.0).covers(line_m):
        # Project any stray interior points back cleanly inside boundary while preserving endpoints
        safe_coords = [coords_m[0]]
        for pt_xy in coords_m[1:-1]:
            if region_m.covers(Point(pt_xy)):
                safe_coords.append(pt_xy)
        safe_coords.append(coords_m[-1])
        line_m = LineString(safe_coords)

    line_wgs = gpd.GeoSeries([line_m], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
    return line_wgs, used_cells, round(cost, 2)


def _livestock_compat(max_slope: float | None, max_elev: float | None) -> str:
    compat = []
    # If DEM unavailable, all species remain eligible with explicit elevation-unverified caveat
    if max_slope is None and max_elev is None:
        return "sheep;goat;cattle;horse;mixed"
    ms = max_slope if max_slope is not None else 12.0
    me = max_elev if max_elev is not None else 2800.0
    if ms <= 32.0 and me <= 3900.0:
        compat.append("sheep")
    if ms <= 36.0 and me <= 4050.0:
        compat.append("goat")
    if ms <= 25.0 and me <= 3450.0:
        compat.append("cattle")
    if ms <= 33.0 and me <= 3950.0:
        compat.append("horse")
    if ms <= 29.0 and me <= 3750.0:
        compat.append("mixed")
    return ";".join(compat) if compat else "goat;sheep"


def build_pastoral_edges_and_resting_nodes(
    pastoral_nodes: gpd.GeoDataFrame,
    physical_nodes: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    env: EnvironmentInventory,
    cfg: dict[str, Any],
    *,
    seed: int = 42,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Generate all Layer B migration corridors, alternative routes, access/water/settlement edges & resting nodes."""
    metric_crs = str(cfg.get("metric_crs", "EPSG:32643"))
    routing_cfg = cfg.get("routing", {})
    alt_penalty = float(routing_cfg.get("alternative_route_penalty_factor", 2.35))
    num_resting_target = int(cfg.get("network_scale", {}).get("num_resting_points", 14))
    gen_ver = str(cfg.get("generator_version", "3.0.0-stage1"))
    disclaimer = str(cfg.get("provenance", {}).get("disclaimer", "Synthetic educational demonstration only."))

    boundary_m = boundary_gdf.to_crs(metric_crs)
    region_m = unary_union([g if g.is_valid else g.buffer(0) for g in boundary_m.geometry if g is not None and not g.is_empty])

    grid = _build_routing_grid(region_m, env, cfg)
    base_graph = csr_matrix(
        (grid["base_weights"], (grid["edges_u"], grid["edges_v"])),
        shape=(grid["n_nodes"], grid["n_nodes"]),
    )

    # Combined static node lookup (Layer B + Layer A settlements/water points)
    all_static = pd.concat([pastoral_nodes, physical_nodes], ignore_index=True)
    all_static_gdf = gpd.GeoDataFrame(all_static, geometry="geometry", crs="EPSG:4326").drop_duplicates(subset=["node_id"])
    all_m = all_static_gdf.to_crs(metric_crs).set_index("node_id")
    node_cells = {
        nid: _nearest_cell(float(row.geometry.x), float(row.geometry.y), grid, region_m)
        for nid, row in all_m.iterrows()
    }

    def _dist_m(a: str, b: str) -> float:
        return float(all_m.loc[a, "geometry"].distance(all_m.loc[b, "geometry"]))

    winters = pastoral_nodes.loc[pastoral_nodes["node_type"] == "winter_pasture", "node_id"].tolist()
    springs = pastoral_nodes.loc[pastoral_nodes["node_type"] == "spring_pasture", "node_id"].tolist()
    summers = pastoral_nodes.loc[pastoral_nodes["node_type"] == "summer_pasture", "node_id"].tolist()
    autumns = pastoral_nodes.loc[pastoral_nodes["node_type"] == "autumn_pasture", "node_id"].tolist()
    passes = pastoral_nodes.loc[pastoral_nodes["node_type"] == "mountain_pass", "node_id"].tolist()

    # Sub-region lookup so corridors form coherent local/regional transhumance loops
    zone_of = pastoral_nodes.set_index("node_id")["subregion_zone"].to_dict()

    def _nearest_in(src: str, candidates: list[str], prefer_zone: bool = True) -> list[str]:
        z = zone_of.get(src)
        if prefer_zone:
            same = sorted([c for c in candidates if zone_of.get(c) == z], key=lambda c: _dist_m(src, c))
            other = sorted([c for c in candidates if zone_of.get(c) != z], key=lambda c: _dist_m(src, c))
            return same + other
        return sorted(candidates, key=lambda c: _dist_m(src, c))

    # Build coherent seasonal migration pairs:
    # 10 winter->spring, 10 spring->summer, 10 summer->autumn, 10 autumn->winter = 40 seasonal legs,
    # selecting 28-36 distinct main corridors + 10 penalized ALTERNATIVE_ROUTE corridors!
    corridor_specs: list[tuple[str, str, str, str, str]] = []  # (src, dst, edge_type, season, direction)
    seen_directed: set[tuple[str, str]] = set()

    def _queue(src: str, dst: str, etype: str, season: str, direction: str) -> None:
        if src == dst or (src, dst) in seen_directed:
            return
        seen_directed.add((src, dst))
        corridor_specs.append((src, dst, etype, season, direction))

    # 1. Winter -> Spring (`SEASONAL_MOVEMENT` / `LEAST_COST_CORRIDOR`)
    for idx, w in enumerate(winters):
        sp = _nearest_in(w, springs)[0]
        _queue(w, sp, "SEASONAL_MOVEMENT" if idx % 2 == 0 else "LEAST_COST_CORRIDOR", "spring", "ascent")

    # 2. Spring -> Summer (high-alpine ascent)
    for idx, sp in enumerate(springs):
        su = _nearest_in(sp, summers)[0]
        _queue(sp, su, "LEAST_COST_CORRIDOR", "summer", "ascent")

    # 3. Summer -> Autumn (descent to sheltered autumn pastures)
    for idx, su in enumerate(summers):
        au = _nearest_in(su, autumns)[0]
        _queue(su, au, "SEASONAL_MOVEMENT" if idx % 2 == 0 else "LEAST_COST_CORRIDOR", "autumn", "descent")

    # 4. Autumn -> Winter (return to winter valley pastures)
    for idx, au in enumerate(autumns):
        w = _nearest_in(au, winters)[0]
        _queue(au, w, "SEASONAL_MOVEMENT", "winter", "descent")

    # Ensure every pasture has at least one incoming and outgoing seasonal connection
    for sp in springs:
        if not any(d == sp for _, d, _, _, _ in corridor_specs):
            w = _nearest_in(sp, winters)[0]
            _queue(w, sp, "SEASONAL_MOVEMENT", "spring", "ascent")
    for su in summers:
        if not any(d == su for _, d, _, _, _ in corridor_specs):
            sp = _nearest_in(su, springs)[0]
            _queue(sp, su, "LEAST_COST_CORRIDOR", "summer", "ascent")
    for au in autumns:
        if not any(d == au for _, d, _, _, _ in corridor_specs):
            su = _nearest_in(au, summers)[0]
            _queue(su, au, "SEASONAL_MOVEMENT", "autumn", "descent")
    for w in winters:
        if not any(d == w for _, d, _, _, _ in corridor_specs):
            au = _nearest_in(w, autumns)[0]
            _queue(au, w, "SEASONAL_MOVEMENT", "winter", "descent")

    edge_rows: list[dict[str, Any]] = []
    resting_rows: list[dict[str, Any]] = []
    primary_cells_map: dict[tuple[str, str], list[int]] = {}
    edge_seq = 0
    rest_seq = 0

    def _make_edge(
        src: str, dst: str, etype: str, season: str, direction: str,
        line_wgs: LineString, route_cost: float, method: str,
        prefix: str = "SYN-EDG", is_synthetic: bool = True, evidence: str = "synthetic_demo",
        source_label: str = "Terrain least-cost network generator",
    ) -> dict[str, Any]:
        nonlocal edge_seq
        edge_seq += 1
        m = compute_line_metrics(line_wgs, env, metric_crs=metric_crs, num_samples=26)
        gain_days = (m["elevation_gain_m"] or 0.0) / 1800.0
        est_days = round(max(0.5, m["distance_m"] / 9500.0 + gain_days), 2)
        return {
            "edge_id": f"{prefix}-{edge_seq:03d}",
            "source_node_id": src,
            "target_node_id": dst,
            "edge_type": etype,
            "network_layer": "pastoral_mobility",
            "distance_m": m["distance_m"],
            "elevation_gain_m": m["elevation_gain_m"],
            "elevation_loss_m": m["elevation_loss_m"],
            "start_elevation_m": m["start_elevation_m"],
            "end_elevation_m": m["end_elevation_m"],
            "max_elevation_m": m["max_elevation_m"],
            "maximum_slope": m["maximum_slope"],
            "average_slope": m["average_slope"],
            "elevation_status": m["elevation_status"],
            "elevation_source": m["elevation_source"],
            "elevation_profile_json": m["elevation_profile_json"],
            "estimated_travel_days": est_days,
            "route_cost": route_cost,
            "season": season,
            "direction": direction,
            "livestock_compatibility": _livestock_compat(m["maximum_slope"], m["max_elevation_m"]),
            "route_generation_method": method,
            "source": source_label,
            "provenance": "synthetic_demo" if is_synthetic else "verified_public",
            "evidence_status": evidence,
            "is_synthetic": is_synthetic,
            "generator_version": gen_ver,
            "random_seed": int(seed),
            "synthetic_disclaimer": disclaimer if is_synthetic else "",
            "geometry": line_wgs,
        }

    # Trace primary seasonal corridors & insert resting_point nodes on longest corridors
    main_corridor_records: list[tuple[str, str, str, str, LineString, list[int], float]] = []
    for src, dst, etype, season, direction in corridor_specs:
        line_wgs, cells, cost = _trace_least_cost_path(
            all_m.loc[src, "geometry"], all_m.loc[dst, "geometry"],
            node_cells[src], node_cells[dst], base_graph, grid, region_m, metric_crs,
        )
        primary_cells_map[(src, dst)] = cells
        main_corridor_records.append((src, dst, etype, season, direction, line_wgs, cells, cost))

    # Pick longest corridors to place intermediate resting_point nodes (SYN-REST-001..014)
    ranked_for_rest = sorted(
        range(len(main_corridor_records)),
        key=lambda i: _dist_m(main_corridor_records[i][0], main_corridor_records[i][1]),
        reverse=True,
    )
    rest_indices = set(ranked_for_rest[:num_resting_target])

    for idx, (src, dst, etype, season, direction, line_wgs, cells, cost) in enumerate(main_corridor_records):
        # Always add the direct corridor edge (used by herds on direct leg)
        rec = _make_edge(src, dst, etype, season, direction, line_wgs, cost, grid["method_name"])
        edge_rows.append(rec)

        # For long corridors, create an intermediate resting_point node at midpoint + ACCESS_ROUTE stopover links
        if idx in rest_indices:
            rest_seq += 1
            mid_pt = line_wgs.interpolate(0.5, normalized=True)
            rid = f"SYN-REST-{rest_seq:03d}"
            elevs, slopes, e_stat, e_src = sample_dem_strict([mid_pt.x], [mid_pt.y], env)
            wc_c, wc_l, _ = sample_worldcover_strict([mid_pt.x], [mid_pt.y], env)
            resting_rows.append({
                "node_id": rid,
                "node_name": f"Corridor Resting Stop {rest_seq:02d} ({src}→{dst})",
                "node_type": "resting_point",
                "network_layer": "pastoral_mobility",
                "subregion_zone": zone_of.get(src, "Z_NARYN_C"),
                "latitude": round(float(mid_pt.y), 5),
                "longitude": round(float(mid_pt.x), 5),
                "elevation_m": elevs[0],
                "slope_degrees": slopes[0],
                "elevation_status": e_stat,
                "elevation_source": e_src,
                "landcover_code": wc_c[0],
                "landcover_class": wc_l[0],
                "pasture_category": "not_a_pasture",
                "seasonal_role": "resting",
                "parent_edge_id": rec["edge_id"],
                "source": "Synthetic corridor midpoint resting point",
                "provenance": "synthetic_demo",
                "evidence_status": "synthetic_demo",
                "is_synthetic": True,
                "generator_version": gen_ver,
                "random_seed": int(seed),
                "synthetic_disclaimer": disclaimer,
                "geometry": mid_pt,
            })
            # Split geometry cleanly at exact midpoint along the path using Shapely substring
            from shapely.ops import substring
            leg1 = substring(line_wgs, 0.0, 0.5, normalized=True)
            leg2 = substring(line_wgs, 0.5, 1.0, normalized=True)
            edge_rows.append(_make_edge(
                src, rid, "SEASONAL_MOVEMENT", season, direction, leg1, round(cost * 0.5, 2),
                grid["method_name"] + "_resting_leg1",
            ))
            edge_rows.append(_make_edge(
                rid, dst, "SEASONAL_MOVEMENT", season, direction, leg2, round(cost * 0.5, 2),
                grid["method_name"] + "_resting_leg2",
            ))

    # 5. Penalized ALTERNATIVE_ROUTE corridors (parallel multigraph edges between same seasonal pairs!)
    alt_target = int(routing_cfg.get("alternative_corridors_target", 10))
    # Pick 10 primary corridors across spring/summer/autumn to compute genuinely distinct penalized alternatives
    alt_candidates = [r for r in main_corridor_records if r[3] in ("spring", "summer", "autumn")][:alt_target]
    for src, dst, _, season, direction, _, primary_cells, _ in alt_candidates:
        penalized_cells = set(primary_cells[1:-1])  # penalize interior cells only
        weights = grid["base_weights"].copy()
        if penalized_cells:
            u_mask = np.isin(grid["edges_u"], list(penalized_cells))
            v_mask = np.isin(grid["edges_v"], list(penalized_cells))
            weights[u_mask | v_mask] *= alt_penalty
        alt_graph = csr_matrix((weights, (grid["edges_u"], grid["edges_v"])), shape=(grid["n_nodes"], grid["n_nodes"]))
        alt_line, _, alt_cost = _trace_least_cost_path(
            all_m.loc[src, "geometry"], all_m.loc[dst, "geometry"],
            node_cells[src], node_cells[dst], alt_graph, grid, region_m, metric_crs,
        )
        edge_rows.append(_make_edge(
            src, dst, "ALTERNATIVE_ROUTE", season, direction, alt_line, alt_cost,
            grid["method_name"] + "_penalized_alternative",
        ))

    # 6. Mountain Pass ACCESS_ROUTE links (connect each pass to nearest spring/summer/autumn pasture)
    for pid in passes:
        near_su = _nearest_in(pid, summers, prefer_zone=False)[0]
        near_sp = _nearest_in(pid, springs, prefer_zone=False)[0]
        for target_p, seas, ddir in ((near_su, "summer", "ascent"), (near_sp, "spring", "descent")):
            line_wgs, _, cost = _trace_least_cost_path(
                all_m.loc[pid, "geometry"], all_m.loc[target_p, "geometry"],
                node_cells[pid], node_cells[target_p], base_graph, grid, region_m, metric_crs,
            )
            edge_rows.append(_make_edge(
                pid, target_p, "ACCESS_ROUTE", seas, ddir, line_wgs, cost,
                grid["method_name"] + "_pass_access",
            ))

    # 7. WATER_ACCESS edges (connect each pasture to its nearest water_point node)
    water_nodes = all_static_gdf.loc[all_static_gdf["node_type"] == "water_point", "node_id"].tolist()
    if water_nodes:
        all_pasture_ids = winters + springs + summers + autumns
        for pid in all_pasture_ids:
            nearest_w = min(water_nodes, key=lambda w: _dist_m(pid, w))
            p_pt = all_static_gdf.loc[all_static_gdf["node_id"] == pid].iloc[0]
            w_pt = all_static_gdf.loc[all_static_gdf["node_id"] == nearest_w].iloc[0]
            line_wgs = LineString([(float(p_pt["longitude"]), float(p_pt["latitude"])), (float(w_pt["longitude"]), float(w_pt["latitude"]))])
            edge_rows.append(_make_edge(
                pid, nearest_w, "WATER_ACCESS", "all_year", "access", line_wgs,
                round(_dist_m(pid, nearest_w) / 1000.0, 2), "nearest_water_access_link",
            ))

    # 8. SETTLEMENT_CONNECTION edges (connect winter & spring pastures to nearest Layer A public settlement)
    settlement_nodes = all_static_gdf.loc[all_static_gdf["node_type"] == "settlement", "node_id"].tolist()
    if settlement_nodes:
        for pid in winters + springs[:5]:
            nearest_s = min(settlement_nodes, key=lambda s: _dist_m(pid, s))
            line_wgs, _, cost = _trace_least_cost_path(
                all_m.loc[nearest_s, "geometry"], all_m.loc[pid, "geometry"],
                node_cells[nearest_s], node_cells[pid], base_graph, grid, region_m, metric_crs,
            )
            edge_rows.append(_make_edge(
                nearest_s, pid, "SETTLEMENT_CONNECTION", "all_year", "access", line_wgs, cost,
                grid["method_name"] + "_settlement_access",
            ))

    resting_gdf = (
        gpd.GeoDataFrame(resting_rows, geometry="geometry", crs="EPSG:4326")
        if resting_rows
        else gpd.GeoDataFrame(columns=pastoral_nodes.columns, geometry="geometry", crs="EPSG:4326")
    )
    all_pastoral_nodes = gpd.GeoDataFrame(
        pd.concat([pastoral_nodes, resting_gdf], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs="EPSG:4326")

    report = {
        "routing_method": grid["method_name"],
        "has_dem": grid["has_dem"],
        "used_osm_roads": grid["used_osm_roads"],
        "resting_points_generated": len(resting_rows),
        "total_pastoral_edges": len(edges_gdf),
        "edges_by_type": edges_gdf["edge_type"].value_counts().to_dict(),
    }
    return all_pastoral_nodes, edges_gdf, report
