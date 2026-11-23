"""Terrain-aware least-cost migration corridor generator inside Naryn ADM1.

Uses a projected metric grid (EPSG:32643) masked strictly to the real Naryn
administrative boundary and computes least-cost paths via Dijkstra's algorithm
considering slope, elevation barriers, and distance. Never labels OpenStreetMap
roads as verified transhumance routes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from src.atlas.terrain import compute_route_elevation_profile, sample_elevation_and_slope


def _build_cost_grid(
    region_m,
    root: Path,
    cell_size_m: float = 3500.0,
    metric_crs: str = "EPSG:32643",
    elevation_barrier_m: float = 4100.0,
):
    minx, miny, maxx, maxy = region_m.bounds
    xs = np.arange(minx + cell_size_m / 2, maxx, cell_size_m)
    ys = np.arange(miny + cell_size_m / 2, maxy, cell_size_m)
    nx, ny = len(xs), len(ys)
    gx, gy = np.meshgrid(xs, ys)  # shape (ny, nx)
    flat_x = gx.ravel()
    flat_y = gy.ravel()

    # Check containment inside Naryn ADM1
    pts_m = gpd.GeoSeries([Point(x, y) for x, y in zip(flat_x, flat_y)], crs=metric_crs)
    inside = pts_m.within(region_m).to_numpy().reshape((ny, nx))

    pts_wgs = pts_m.to_crs("EPSG:4326")
    lons = pts_wgs.x.to_numpy()
    lats = pts_wgs.y.to_numpy()
    elevs_flat, slopes_flat, elev_source = sample_elevation_and_slope(lons, lats, root)
    elev_grid = elevs_flat.reshape((ny, nx))
    slope_grid = slopes_flat.reshape((ny, nx))

    # Build 8-connected sparse graph over cells inside Naryn ADM1
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]
    for r in range(ny):
        for c in range(nx):
            if not inside[r, c]:
                continue
            u = r * nx + c
            e_u = elev_grid[r, c]
            s_u = slope_grid[r, c]
            for dr, dc, dist_mult in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ny and 0 <= nc < nx and inside[nr, nc]:
                    v = nr * nx + nc
                    e_v = elev_grid[nr, nc]
                    s_v = slope_grid[nr, nc]
                    step_m = cell_size_m * dist_mult
                    grade = abs(e_v - e_u) / step_m
                    avg_slope = 0.5 * (s_u + s_v)
                    # Tobler-inspired anisotropic slope + high-alpine barrier cost
                    slope_factor = 1.0 + 3.2 * (avg_slope / 15.0) ** 2 + 12.0 * (grade ** 2)
                    barrier_factor = 6.0 if max(e_u, e_v) > elevation_barrier_m else 1.0
                    cost = (step_m / 1000.0) * slope_factor * barrier_factor
                    rows.append(u)
                    cols.append(v)
                    weights.append(float(cost))

    graph = csr_matrix((weights, (rows, cols)), shape=(nx * ny, nx * ny))
    return {
        "xs": xs,
        "ys": ys,
        "nx": nx,
        "ny": ny,
        "inside": inside,
        "elev_grid": elev_grid,
        "slope_grid": slope_grid,
        "graph": graph,
        "elev_source": elev_source,
        "cell_size_m": cell_size_m,
    }


def _nearest_valid_cell(x: float, y: float, grid: dict[str, Any]) -> int:
    xs, ys, inside = grid["xs"], grid["ys"], grid["inside"]
    nx = grid["nx"]
    ix = int(np.argmin(np.abs(xs - x)))
    iy = int(np.argmin(np.abs(ys - y)))
    if inside[iy, ix]:
        return iy * nx + ix
    # Find nearest valid inside cell
    valid_r, valid_c = np.where(inside)
    d2 = (xs[valid_c] - x) ** 2 + (ys[valid_r] - y) ** 2
    best = int(np.argmin(d2))
    return int(valid_r[best] * nx + valid_c[best])


def _smooth_inside_region(coords_m: list[tuple[float, float]], region_m) -> list[tuple[float, float]]:
    """Apply Chaikin corner smoothing while strictly keeping endpoints and vertices inside Naryn."""
    if len(coords_m) < 4:
        return coords_m
    smoothed = [coords_m[0]]
    for i in range(1, len(coords_m) - 1):
        p0, p1, p2 = coords_m[i - 1], coords_m[i], coords_m[i + 1]
        cand = (0.25 * p0[0] + 0.5 * p1[0] + 0.25 * p2[0], 0.25 * p0[1] + 0.5 * p1[1] + 0.25 * p2[1])
        if region_m.contains(Point(cand)):
            smoothed.append(cand)
        else:
            smoothed.append(p1)
    smoothed.append(coords_m[-1])
    return smoothed


def generate_migration_routes(
    pastures_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    root: Path,
    *,
    seed: int = 42,
    cell_size_m: float = 3500.0,
    metric_crs: str = "EPSG:32643",
    elevation_barrier_m: float = 4100.0,
    simulation_version: str = "2.0.0",
    disclaimer: str = "",
) -> gpd.GeoDataFrame:
    """Build 16 terrain-aware least-cost corridors connecting winter, foothill, and summer pastures."""
    boundary_m = boundary_gdf.to_crs(metric_crs)
    region_m = unary_union([g if g.is_valid else g.buffer(0) for g in boundary_m.geometry if g is not None and not g.is_empty])
    pastures_m = pastures_gdf.to_crs(metric_crs).set_index("pasture_id")

    grid = _build_cost_grid(
        region_m, root, cell_size_m=cell_size_m, metric_crs=metric_crs, elevation_barrier_m=elevation_barrier_m
    )

    # Pairs connecting lower wintering valleys -> spring/autumn foothills -> high alpine summer pastures
    # Plus direct alternative corridors for climate-scenario rerouting!
    winter_ids = pastures_gdf.loc[pastures_gdf["elevation_band"] == "valley_winter", "pasture_id"].tolist()
    foothill_ids = pastures_gdf.loc[pastures_gdf["elevation_band"] == "foothill_transition", "pasture_id"].tolist()
    summer_ids = pastures_gdf.loc[pastures_gdf["elevation_band"] == "high_alpine", "pasture_id"].tolist()

    # Deterministic geographically paired corridors (nearest 1-2 foothills per winter, nearest 2 summers per foothill)
    corridor_pairs: list[tuple[str, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _dist_m(a: str, b: str) -> float:
        return float(pastures_m.loc[a, "geometry"].distance(pastures_m.loc[b, "geometry"]))

    # 1. Winter -> Spring/Autumn foothill corridors (6)
    for w in winter_ids:
        nearest_f = sorted(foothill_ids, key=lambda f: _dist_m(w, f))[:2]
        for f in nearest_f[:1]:
            if (w, f) not in seen_pairs:
                seen_pairs.add((w, f))
                corridor_pairs.append((w, f, "spring_ascent"))

    # 2. Foothill -> Summer high-alpine corridors (8 — each foothill connects to nearest 1-2 alpine pastures for scenario switching)
    for idx_f, f in enumerate(foothill_ids):
        nearest_s = sorted(summer_ids, key=lambda s: _dist_m(f, s))
        take = 2 if idx_f < 4 else 1
        for s in nearest_s[:take]:
            if (f, s) not in seen_pairs:
                seen_pairs.add((f, s))
                corridor_pairs.append((f, s, "summer_ascent"))

    # 3. Direct Winter -> Summer / Refuge corridors (4 — enables alternative descent under early snow / warmer summer)
    for w, s in zip(winter_ids[:4], summer_ids[:4]):
        if (w, s) not in seen_pairs:
            seen_pairs.add((w, s))
            corridor_pairs.append((w, s, "direct_seasonal"))

    # Ensure every summer pasture has at least one incoming corridor
    connected_summers = {dst for _, dst, _ in corridor_pairs}
    for s in summer_ids:
        if s not in connected_summers:
            f = min(foothill_ids, key=lambda cand: _dist_m(cand, s))
            corridor_pairs.append((f, s, "summer_ascent"))

    # Compute Dijkstra shortest paths per unique origin cell
    unique_origins = sorted({orig for orig, _, _ in corridor_pairs})
    origin_cells = {
        pid: _nearest_valid_cell(float(pastures_m.loc[pid, "geometry"].x), float(pastures_m.loc[pid, "geometry"].y), grid)
        for pid in pastures_m.index
    }

    pred_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for orig in unique_origins:
        dists, preds = dijkstra(grid["graph"], directed=False, indices=origin_cells[orig], return_predecessors=True)
        pred_cache[orig] = (dists, preds)

    xs, ys, nx = grid["xs"], grid["ys"], grid["nx"]
    rows = []

    for idx, (orig, dst, season_type) in enumerate(corridor_pairs, start=1):
        dists, preds = pred_cache[orig]
        start_cell = origin_cells[orig]
        end_cell = origin_cells[dst]
        orig_pt = pastures_m.loc[orig, "geometry"]
        dst_pt = pastures_m.loc[dst, "geometry"]

        method = "terrain_least_cost_dijkstra_grid"
        cost_val = float(dists[end_cell])
        if not np.isfinite(cost_val):
            # Fallback if grid cell disconnected across narrow bottleneck
            coords_m = [(float(orig_pt.x), float(orig_pt.y)), (float(dst_pt.x), float(dst_pt.y))]
            method = "fallback_schematic_segment_disconnected_grid"
            cost_val = float(orig_pt.distance(dst_pt) / 1000.0)
        else:
            chain = []
            cur = end_cell
            guard = 0
            while cur != start_cell and cur >= 0 and guard < 5000:
                r, c = divmod(int(cur), nx)
                chain.append((float(xs[c]), float(ys[r])))
                cur = int(preds[cur])
                guard += 1
            chain.reverse()
            coords_m = [(float(orig_pt.x), float(orig_pt.y))] + chain + [(float(dst_pt.x), float(dst_pt.y))]
            coords_m = _smooth_inside_region(coords_m, region_m)

        line_m = LineString(coords_m)
        # Ensure strict clipping inside Naryn ADM1
        if not region_m.covers(line_m):
            clipped = line_m.intersection(region_m)
            if clipped.geom_type == "LineString" and not clipped.is_empty:
                line_m = clipped
            elif hasattr(clipped, "geoms"):
                parts = [g for g in clipped.geoms if g.geom_type == "LineString" and not g.is_empty]
                if parts:
                    line_m = max(parts, key=lambda g: g.length)

        line_wgs = gpd.GeoSeries([line_m], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
        prof = compute_route_elevation_profile(line_wgs, root, num_samples=32, metric_crs=metric_crs)
        est_days = max(1, int(math.ceil(prof["distance_km"] / 9.5 + prof["elevation_gain_m"] / 1800.0)))

        rows.append({
            "route_id": f"SYN-CORRIDOR-{idx:03d}",
            "name": f"{pastures_m.loc[orig, 'pasture_name']} → {pastures_m.loc[dst, 'pasture_name']}",
            "origin_pasture_id": orig,
            "destination_pasture_id": dst,
            "season": season_type,
            "distance_km": prof["distance_km"],
            "start_elevation_m": prof["start_elevation_m"],
            "end_elevation_m": prof["end_elevation_m"],
            "min_elevation_m": prof["min_elevation_m"],
            "max_elevation_m": prof["max_elevation_m"],
            "elevation_gain_m": prof["elevation_gain_m"],
            "elevation_loss_m": prof["elevation_loss_m"],
            "avg_slope_deg": prof["avg_slope_deg"],
            "estimated_travel_days": est_days,
            "route_cost": round(cost_val, 2),
            "route_generation_method": method,
            "elevation_source": prof["elevation_source"],
            "elevation_profile_json": json.dumps(prof["profile"]),
            "location_id": "naryn-region",
            "origin_class": "synthetic",
            "simulation_version": simulation_version,
            "random_seed": int(seed),
            "is_synthetic": True,
            "provenance": "synthetic_demo",
            "synthetic_disclaimer": disclaimer,
            "geometry": line_wgs,
        })

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
