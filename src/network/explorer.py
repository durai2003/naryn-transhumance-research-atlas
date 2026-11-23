"""Stage 2 Network Explorer: filtering, connected components, layouts, and synchronised payload builder."""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from src.atlas.terrain import export_local_dem_grid
from src.network.network_loader import Stage1NetworkData


NODE_CATEGORY_META: dict[str, dict[str, str]] = {
    "winter_pasture": {"label": "Winter Pasture", "color": "#38bdf8", "shape": "square"},
    "spring_pasture": {"label": "Spring Pasture", "color": "#4ade80", "shape": "triangle"},
    "summer_pasture": {"label": "Summer Pasture", "color": "#facc15", "shape": "diamond"},
    "autumn_pasture": {"label": "Autumn Pasture", "color": "#fb923c", "shape": "hexagon"},
    "settlement": {"label": "Settlement (Public)", "color": "#f8fafc", "shape": "star"},
    "water_point": {"label": "Water Point", "color": "#06b6d4", "shape": "circle"},
    "mountain_pass": {"label": "Mountain Pass", "color": "#f43f5e", "shape": "chevron"},
    "resting_point": {"label": "Resting Point", "color": "#c084fc", "shape": "dot"},
}

EDGE_CATEGORY_META: dict[str, dict[str, str]] = {
    "SEASONAL_MOVEMENT": {"label": "Seasonal Movement", "color": "#e879f9", "dash": "solid"},
    "LEAST_COST_CORRIDOR": {"label": "Least-Cost Corridor", "color": "#a855f7", "dash": "dashed"},
    "ALTERNATIVE_ROUTE": {"label": "Alternative Route", "color": "#22d3ee", "dash": "dotted"},
    "ACCESS_ROUTE": {"label": "Access / Pass / Road Link", "color": "#f59e0b", "dash": "solid"},
    "WATER_ACCESS": {"label": "Water Access", "color": "#0ea5e9", "dash": "dotted"},
    "SETTLEMENT_CONNECTION": {"label": "Settlement Connection", "color": "#94a3b8", "dash": "dashed"},
}


@dataclass
class FilteredNetwork:
    nodes_gdf: gpd.GeoDataFrame
    edges_gdf: gpd.GeoDataFrame
    graph: nx.MultiDiGraph
    components: list[dict[str, Any]]
    node_to_component: dict[str, int]
    active_filters: dict[str, Any]


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        f = float(val)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    return str(val)


def _edge_matches_livestock(
    edge_row: pd.Series,
    selected_livestock: set[str],
    all_livestock: set[str],
    edge_species_used: dict[str, set[str]],
) -> bool:
    if not selected_livestock or selected_livestock == all_livestock:
        return True
    # Physical Layer A road context links remain visible when verified_public is included
    if str(edge_row.get("network_layer", "")) == "physical_geographical":
        return True
    eid = str(edge_row.get("edge_id", ""))
    used = edge_species_used.get(eid, set())
    if used:
        return bool(used & selected_livestock)
    compat_raw = _safe_str(edge_row.get("livestock_compatibility", ""))
    compat_tokens = {tok.strip().lower() for tok in compat_raw.split(";") if tok.strip()}
    if not compat_tokens or "road_context_not_pastoral_corridor" in compat_tokens:
        return True
    return bool(compat_tokens & selected_livestock)


def filter_network(
    net: Stage1NetworkData,
    *,
    node_types: Sequence[str] | None = None,
    edge_types: Sequence[str] | None = None,
    seasons: Sequence[str] | None = None,
    livestock_types: Sequence[str] | None = None,
    climate_scenario: str = "SCENARIO_BASELINE",
    evidence_statuses: Sequence[str] | None = None,
    public_only: bool = False,
) -> FilteredNetwork:
    """Filter the Stage 1 network across all 6 Stage 2 filter dimensions and compute components."""
    nodes = net.all_nodes.copy()
    edges = net.all_edges.copy()

    if nodes.empty:
        return FilteredNetwork(
            nodes_gdf=nodes,
            edges_gdf=edges,
            graph=nx.MultiDiGraph(),
            components=[],
            node_to_component={},
            active_filters={},
        )

    # 1. Public-only mode overrides evidence status to strictly hide all synthetic items
    if public_only:
        nodes = nodes.loc[~nodes["is_synthetic"].astype(bool)]
        if not edges.empty:
            edges = edges.loc[~edges["is_synthetic"].astype(bool)]
    elif evidence_statuses is not None:
        ev_set = set(evidence_statuses)
        nodes = nodes.loc[nodes["evidence_status"].astype(str).isin(ev_set)]
        if not edges.empty:
            edges = edges.loc[edges["evidence_status"].astype(str).isin(ev_set)]

    # 2. Node type filter
    if node_types is not None:
        nt_set = set(node_types)
        nodes = nodes.loc[nodes["node_type"].astype(str).isin(nt_set)]

    visible_node_ids = set(nodes["node_id"].astype(str))

    # 3. Filter edges to endpoints in visible nodes + edge_type + season + livestock_type + scenario
    if not edges.empty:
        edges = edges.loc[
            edges["source_node_id"].astype(str).isin(visible_node_ids)
            & edges["target_node_id"].astype(str).isin(visible_node_ids)
        ]
        if edge_types is not None:
            et_set = set(edge_types)
            edges = edges.loc[edges["edge_type"].astype(str).isin(et_set)]

        if seasons is not None:
            s_set = set(seasons)
            edges = edges.loc[edges["season"].astype(str).isin(s_set)]

        all_ls = {"sheep", "goat", "cattle", "horse", "mixed"}
        sel_ls = {str(s).lower() for s in livestock_types} if livestock_types is not None else all_ls

        edge_species_used: dict[str, set[str]] = {}
        if not net.movement_events.empty:
            ev_df = net.movement_events
            if "scenario_id" in ev_df.columns and climate_scenario in set(ev_df["scenario_id"].astype(str)):
                ev_df = ev_df.loc[ev_df["scenario_id"].astype(str) == climate_scenario]
            for eid, grp in ev_df.groupby("edge_id"):
                edge_species_used[str(eid)] = set(grp["livestock_type"].astype(str).str.lower().unique())

        if sel_ls != all_ls:
            mask = edges.apply(
                lambda r: _edge_matches_livestock(r, sel_ls, all_ls, edge_species_used),
                axis=1,
            )
            edges = edges.loc[mask]

    nodes = nodes.reset_index(drop=True)
    edges = edges.reset_index(drop=True)

    # Build filtered NetworkX MultiDiGraph
    G = nx.MultiDiGraph(scenario_id=climate_scenario)
    for _, row in nodes.iterrows():
        nid = str(row["node_id"])
        G.add_node(
            nid,
            node_name=_safe_str(row.get("node_name", nid)),
            node_type=_safe_str(row.get("node_type", "")),
            network_layer=_safe_str(row.get("network_layer", "")),
            evidence_status=_safe_str(row.get("evidence_status", "")),
            is_synthetic=bool(row.get("is_synthetic", True)),
        )
    for _, row in edges.iterrows():
        eid = str(row["edge_id"])
        G.add_edge(
            str(row["source_node_id"]),
            str(row["target_node_id"]),
            key=eid,
            edge_id=eid,
            edge_type=_safe_str(row.get("edge_type", "")),
            distance_m=_safe_float(row.get("distance_m")) or 0.0,
            season=_safe_str(row.get("season", "")),
        )

    # Connected components (weakly connected components of directed multigraph, sorted largest-first)
    raw_comps = sorted(
        (sorted(list(c)) for c in nx.weakly_connected_components(G)),
        key=lambda c: (-len(c), c[0]),
    )
    components: list[dict[str, Any]] = []
    node_to_comp: dict[str, int] = {}
    for cid, member_ids in enumerate(raw_comps, start=1):
        for nid in member_ids:
            node_to_comp[nid] = cid
        sub = G.subgraph(member_ids)
        components.append({
            "component_id": cid,
            "node_count": len(member_ids),
            "edge_count": int(sub.number_of_edges()),
            "sample_nodes": member_ids[:4],
        })

    return FilteredNetwork(
        nodes_gdf=nodes,
        edges_gdf=edges,
        graph=G,
        components=components,
        node_to_component=node_to_comp,
        active_filters={
            "node_types": list(node_types) if node_types is not None else [],
            "edge_types": list(edge_types) if edge_types is not None else [],
            "seasons": list(seasons) if seasons is not None else [],
            "livestock_types": list(livestock_types) if livestock_types is not None else [],
            "climate_scenario": climate_scenario,
            "evidence_statuses": list(evidence_statuses) if evidence_statuses is not None else [],
            "public_only": bool(public_only),
        },
    )


def _normalize_positions(pos_map: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    if not pos_map:
        return {}
    xs = np.array([v[0] for v in pos_map.values()], dtype=float)
    ys = np.array([v[1] for v in pos_map.values()], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    out: dict[str, list[float]] = {}
    for nid, (x, y) in pos_map.items():
        nx_val = 0.07 + 0.86 * ((x - x_min) / x_span)
        ny_val = 0.08 + 0.84 * ((y - y_min) / y_span)
        out[nid] = [round(float(nx_val), 4), round(float(ny_val), 4)]
    return out


_EXPLORER_LOCK = threading.Lock()
_LAYOUT_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], dict[str, dict[str, list[float]]]] = {}


def compute_graph_layouts(
    filtered: FilteredNetwork,
    *,
    layout_names: Sequence[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Compute requested deterministic normalized [0, 1] x [0, 1] layouts with caching.

    When `layout_names` is None, computes all 4 layouts for full backward compatibility.
    Pass `layout_names=("geographic", "spring")` from the UI to skip expensive Kamada-Kawai.
    """
    requested = tuple(sorted(set(layout_names))) if layout_names is not None else (
        "geographic", "kamada_kawai", "layered_seasonal", "spring"
    )
    G = filtered.graph
    nodes_gdf = filtered.nodes_gdf
    if G.number_of_nodes() == 0 or nodes_gdf.empty:
        return {name: {} for name in requested}

    n_sig = tuple(sorted(nodes_gdf["node_id"].astype(str).tolist()))
    e_sig = tuple(sorted(filtered.edges_gdf["edge_id"].astype(str).tolist())) if not filtered.edges_gdf.empty else ()
    cache_key = (n_sig, e_sig, requested)
    with _EXPLORER_LOCK:
        cached = _LAYOUT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    result: dict[str, dict[str, list[float]]] = {}
    req_set = set(requested)

    # 1. Geographic XY (projected lon/lat with North at top -> smaller canvas y)
    if "geographic" in req_set:
        geo_raw: dict[str, tuple[float, float]] = {}
        for _, row in nodes_gdf.iterrows():
            nid = str(row["node_id"])
            geo_raw[nid] = (float(row["longitude"]), -float(row["latitude"]))
        result["geographic"] = _normalize_positions(geo_raw)

    # 2. Seasonal Layered layout
    if "layered_seasonal" in req_set:
        col_x = {
            "settlement": 0.08,
            "winter_pasture": 0.23,
            "spring_pasture": 0.39,
            "resting_point": 0.52,
            "mountain_pass": 0.56,
            "summer_pasture": 0.70,
            "autumn_pasture": 0.84,
            "water_point": 0.94,
        }
        layered: dict[str, list[float]] = {}
        for ntype, grp in nodes_gdf.groupby("node_type"):
            grp_s = grp.sort_values(["latitude", "longitude"], ascending=[False, True]).reset_index(drop=True)
            n = len(grp_s)
            x_base = col_x.get(str(ntype), 0.5)
            for i, row in grp_s.iterrows():
                y = 0.5 if n == 1 else (0.08 + 0.84 * (i / max(n - 1, 1)))
                x_jitter = ((i % 2) - 0.5) * 0.032 if n > 12 else 0.0
                layered[str(row["node_id"])] = [round(float(np.clip(x_base + x_jitter, 0.05, 0.96)), 4), round(float(y), 4)]
        result["layered_seasonal"] = layered

    if "spring" in req_set or "kamada_kawai" in req_set:
        G_simple = nx.Graph()
        for n in G.nodes():
            G_simple.add_node(n)
        for u, v, _ in G.edges(data=True):
            if u != v and not G_simple.has_edge(u, v):
                G_simple.add_edge(u, v, weight=1.0)

        if "spring" in req_set:
            if G_simple.number_of_nodes() == 1:
                only = next(iter(G_simple.nodes()))
                result["spring"] = {only: [0.5, 0.5]}
            else:
                k_val = 1.6 / math.sqrt(max(G_simple.number_of_nodes(), 2))
                sp_raw = nx.spring_layout(G_simple, k=k_val, iterations=45, seed=42)
                result["spring"] = _normalize_positions({str(k): (float(v[0]), float(v[1])) for k, v in sp_raw.items()})

        if "kamada_kawai" in req_set:
            kk_raw: dict[str, tuple[float, float]] = {}
            comps = sorted((list(c) for c in nx.connected_components(G_simple)), key=lambda c: (-len(c), c[0]))
            x_cursor = 0.0
            for comp in comps:
                if len(comp) == 1:
                    kk_raw[str(comp[0])] = (x_cursor, 0.0)
                    x_cursor += 0.9
                elif len(comp) == 2:
                    kk_raw[str(comp[0])] = (x_cursor, -0.35)
                    kk_raw[str(comp[1])] = (x_cursor, 0.35)
                    x_cursor += 1.0
                else:
                    sub = G_simple.subgraph(comp)
                    sub_pos = nx.kamada_kawai_layout(sub)
                    sxs = [float(p[0]) for p in sub_pos.values()]
                    width = max(max(sxs) - min(sxs), 0.6)
                    scale = max(1.0, math.sqrt(len(comp)) * 0.45)
                    for nid, xy in sub_pos.items():
                        kk_raw[str(nid)] = (x_cursor + (float(xy[0]) - min(sxs)) * scale, float(xy[1]) * scale)
                    x_cursor += width * scale + 1.1
            result["kamada_kawai"] = _normalize_positions(kk_raw)

    with _EXPLORER_LOCK:
        if len(_LAYOUT_CACHE) > 48:
            _LAYOUT_CACHE.clear()
        _LAYOUT_CACHE[cache_key] = result
    return result


def _build_node_herd_index(net: Stage1NetworkData, livestock_filter: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Map every static node_id to the synthetic herds originating, visiting, or passing through it."""
    by_node: dict[str, dict[str, dict[str, Any]]] = {}

    def _add(nid: str, hid: str, species: str, count: int, role: str) -> None:
        if not nid or nid == "nan":
            return
        if livestock_filter and species.lower() not in livestock_filter:
            return
        slot = by_node.setdefault(nid, {})
        if hid not in slot:
            slot[hid] = {"herd_id": hid, "livestock_type": species, "animal_count": count, "roles": [role]}
        elif role not in slot[hid]["roles"]:
            slot[hid]["roles"].append(role)

    if not net.herds.empty:
        role_cols = [
            ("winter_pasture_id", "winter_pasture"),
            ("spring_pasture_id", "spring_pasture"),
            ("summer_pasture_id", "summer_pasture"),
            ("autumn_pasture_id", "autumn_pasture"),
            ("return_winter_pasture_id", "winter_return"),
        ]
        for _, r in net.herds.iterrows():
            hid = _safe_str(r.get("herd_id"))
            sp = _safe_str(r.get("livestock_type"))
            cnt = _safe_int(r.get("animal_count"), 0)
            for col, role in role_cols:
                if col in r and pd.notna(r[col]):
                    _add(str(r[col]), hid, sp, cnt, role)

    if not net.movement_events.empty:
        for _, r in net.movement_events.iterrows():
            hid = _safe_str(r.get("herd_id"))
            sp = _safe_str(r.get("livestock_type"))
            cnt = _safe_int(r.get("animal_count"), 0)
            _add(_safe_str(r.get("origin_node_id")), hid, sp, cnt, "departure")
            _add(_safe_str(r.get("destination_node_id")), hid, sp, cnt, "arrival")

    return {nid: sorted(hmap.values(), key=lambda x: x["herd_id"]) for nid, hmap in by_node.items()}


def _line_coords(geom) -> list[list[float]]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [[round(float(x), 5), round(float(y), 5)] for x, y in geom.coords]
    if isinstance(geom, MultiLineString):
        pts: list[list[float]] = []
        for part in geom.geoms:
            pts.extend([[round(float(x), 5), round(float(y), 5)] for x, y in part.coords])
        return pts
    return []


def _get_net_explorer_cache(net: Stage1NetworkData) -> dict[str, Any]:
    cached = getattr(net, "_explorer_cache", None)
    if cached is not None:
        return cached
    with _EXPLORER_LOCK:
        cached = getattr(net, "_explorer_cache", None)
        if cached is not None:
            return cached
        boundary_geojson = (
            json.loads(net.boundary[["shapeName", "shapeISO", "geometry"]].to_json())
            if not net.boundary.empty and {"shapeName", "shapeISO"}.issubset(net.boundary.columns)
            else (json.loads(net.boundary.to_json()) if not net.boundary.empty else {"type": "FeatureCollection", "features": []})
        )
        cache_obj: dict[str, Any] = {
            "boundary": boundary_geojson,
            "node_herds": {},
            "payloads": {},
        }
        setattr(net, "_explorer_cache", cache_obj)
        return cache_obj


def build_network_explorer_payload(
    net: Stage1NetworkData,
    filtered: FilteredNetwork,
    *,
    basemap: str = "esri_satellite",
    vertical_exaggeration: float = 1.4,
    initial_layout: str = "layered_seasonal",
    selected_node_id: str | None = None,
    selected_edge_id: str | None = None,
    layout_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the complete JSON payload for the Synchronised Geographical + Abstract Graph Explorer."""
    net_cache = _get_net_explorer_cache(net)
    n_sig = tuple(sorted(filtered.nodes_gdf["node_id"].astype(str).tolist())) if not filtered.nodes_gdf.empty else ()
    e_sig = tuple(sorted(filtered.edges_gdf["edge_id"].astype(str).tolist())) if not filtered.edges_gdf.empty else ()
    ls_key = tuple(sorted(str(s).lower() for s in (filtered.active_filters.get("livestock_types") or [])))
    lay_key = tuple(sorted(set(layout_names))) if layout_names is not None else ("geographic", "kamada_kawai", "layered_seasonal", "spring")
    pub_only = bool(filtered.active_filters.get("public_only", False))
    base_key = (n_sig, e_sig, ls_key, lay_key, pub_only)

    with _EXPLORER_LOCK:
        cached_base = net_cache["payloads"].get(base_key)
    if cached_base is not None:
        return {
            **cached_base,
            "config": {
                "basemap": basemap,
                "verticalExaggeration": float(vertical_exaggeration),
                "initialLayout": initial_layout,
                "selectedNodeId": selected_node_id,
                "selectedEdgeId": selected_edge_id,
                "activeFilters": filtered.active_filters,
            },
        }

    G = filtered.graph
    layouts = compute_graph_layouts(filtered, layout_names=layout_names)
    ls_set = set(ls_key) if ls_key else None
    with _EXPLORER_LOCK:
        node_herds_index = net_cache["node_herds"].get(ls_key)
    if node_herds_index is None:
        node_herds_index = _build_node_herd_index(net, livestock_filter=ls_set)
        with _EXPLORER_LOCK:
            net_cache["node_herds"][ls_key] = node_herds_index

    # Pre-index node names for fast edge tooltips/inspectors
    node_names = {
        str(r["node_id"]): _safe_str(r.get("node_name", r["node_id"]))
        for _, r in filtered.nodes_gdf.iterrows()
    }

    # Build edge records & incident route lists per node
    incident_routes: dict[str, list[dict[str, Any]]] = {}
    edge_records: list[dict[str, Any]] = []
    for _, row in filtered.edges_gdf.iterrows():
        eid = str(row["edge_id"])
        src = str(row["source_node_id"])
        dst = str(row["target_node_id"])
        etype = _safe_str(row.get("edge_type", "ACCESS_ROUTE"))
        dist_m = _safe_float(row.get("distance_m")) or 0.0
        prof_raw = row.get("elevation_profile_json")
        try:
            profile = json.loads(prof_raw) if isinstance(prof_raw, str) and prof_raw.strip() else []
        except Exception:
            profile = []

        rec = {
            "edge_id": eid,
            "source_node_id": src,
            "source_node_name": node_names.get(src, src),
            "target_node_id": dst,
            "target_node_name": node_names.get(dst, dst),
            "edge_type": etype,
            "network_layer": _safe_str(row.get("network_layer", "pastoral_mobility")),
            "distance_m": round(dist_m, 1),
            "distance_km": round(dist_m / 1000.0, 2),
            "start_elevation_m": _safe_float(row.get("start_elevation_m")),
            "end_elevation_m": _safe_float(row.get("end_elevation_m")),
            "max_elevation_m": _safe_float(row.get("max_elevation_m")),
            "elevation_gain_m": _safe_float(row.get("elevation_gain_m")),
            "elevation_loss_m": _safe_float(row.get("elevation_loss_m")),
            "average_slope": _safe_float(row.get("average_slope")),
            "maximum_slope": _safe_float(row.get("maximum_slope")),
            "elevation_status": _safe_str(row.get("elevation_status", "unavailable_no_local_dem")),
            "elevation_source": _safe_str(row.get("elevation_source", "")),
            "estimated_travel_days": _safe_float(row.get("estimated_travel_days")),
            "season": _safe_str(row.get("season", "all_year")),
            "direction": _safe_str(row.get("direction", "")),
            "livestock_compatibility": _safe_str(row.get("livestock_compatibility", "")),
            "route_generation_method": _safe_str(row.get("route_generation_method", "")),
            "source": _safe_str(row.get("source", "")),
            "provenance": _safe_str(row.get("provenance", "")),
            "evidence_status": _safe_str(row.get("evidence_status", "synthetic_demo")),
            "is_synthetic": bool(row.get("is_synthetic", True)),
            "utilisation": {
                "herd_count": _safe_int(row.get("herd_count"), 0),
                "animal_count": _safe_int(row.get("animal_count"), 0),
                "movement_event_count": _safe_int(row.get("movement_event_count"), 0),
                "winter_usage": _safe_int(row.get("winter_usage"), 0),
                "spring_usage": _safe_int(row.get("spring_usage"), 0),
                "summer_usage": _safe_int(row.get("summer_usage"), 0),
                "autumn_usage": _safe_int(row.get("autumn_usage"), 0),
                "avg_estimated_travel_days": _safe_float(row.get("avg_estimated_travel_days")) or 0.0,
                "alternative_route_usage": _safe_int(row.get("alternative_route_usage"), 0),
            },
            "elevation_profile": profile,
            "coordinates": _line_coords(row.geometry),
        }
        edge_records.append(rec)

        brief_out = {
            "edge_id": eid,
            "relation": "outgoing",
            "peer_node_id": dst,
            "peer_node_name": node_names.get(dst, dst),
            "edge_type": etype,
            "season": rec["season"],
            "distance_km": rec["distance_km"],
            "herd_count": rec["utilisation"]["herd_count"],
        }
        brief_in = {
            "edge_id": eid,
            "relation": "incoming",
            "peer_node_id": src,
            "peer_node_name": node_names.get(src, src),
            "edge_type": etype,
            "season": rec["season"],
            "distance_km": rec["distance_km"],
            "herd_count": rec["utilisation"]["herd_count"],
        }
        incident_routes.setdefault(src, []).append(brief_out)
        incident_routes.setdefault(dst, []).append(brief_in)

    # Build node records with full Stage 2 Inspector attributes
    node_records: list[dict[str, Any]] = []
    for _, row in filtered.nodes_gdf.iterrows():
        nid = str(row["node_id"])
        in_neighbors = sorted(list(G.predecessors(nid))) if nid in G else []
        out_neighbors = sorted(list(G.successors(nid))) if nid in G else []
        herds_for_node = node_herds_index.get(nid, []) if not filtered.active_filters.get("public_only", False) else []
        node_records.append({
            "node_id": nid,
            "node_name": _safe_str(row.get("node_name", nid)),
            "node_type": _safe_str(row.get("node_type", "")),
            "network_layer": _safe_str(row.get("network_layer", "")),
            "subregion_zone": _safe_str(row.get("subregion_zone", "")),
            "latitude": round(float(row["latitude"]), 5),
            "longitude": round(float(row["longitude"]), 5),
            "elevation_m": _safe_float(row.get("elevation_m")),
            "slope_degrees": _safe_float(row.get("slope_degrees")),
            "elevation_status": _safe_str(row.get("elevation_status", "unavailable_no_local_dem")),
            "elevation_source": _safe_str(row.get("elevation_source", "")),
            "landcover_code": _safe_int(row.get("landcover_code"), -1) if _safe_float(row.get("landcover_code")) is not None else None,
            "landcover_class": _safe_str(row.get("landcover_class", "unavailable_no_local_worldcover")),
            "source": _safe_str(row.get("source", "")),
            "provenance": _safe_str(row.get("provenance", "")),
            "evidence_status": _safe_str(row.get("evidence_status", "synthetic_demo")),
            "is_synthetic": bool(row.get("is_synthetic", True)),
            "external_id": _safe_str(row.get("external_id", "")),
            "external_uri": _safe_str(row.get("external_uri", "")),
            "component_id": int(filtered.node_to_component.get(nid, 0)),
            "in_neighbors": in_neighbors,
            "out_neighbors": out_neighbors,
            "degree_total": len(incident_routes.get(nid, [])),
            "connected_routes": incident_routes.get(nid, []),
            "connected_herds": herds_for_node,
            "connected_herd_count": len(herds_for_node),
        })

    base_payload = {
        "stage": "stage2_network_explorer",
        "meta": {
            "nodeCategories": NODE_CATEGORY_META,
            "edgeCategories": EDGE_CATEGORY_META,
        },
        "summary": {
            "visible_nodes": len(node_records),
            "visible_edges": len(edge_records),
            "connected_components": len(filtered.components),
            "public_nodes": sum(1 for n in node_records if not n["is_synthetic"]),
            "synthetic_nodes": sum(1 for n in node_records if n["is_synthetic"]),
            "active_used_edges": sum(1 for e in edge_records if e["utilisation"]["movement_event_count"] > 0),
        },
        "components": filtered.components,
        "layouts": layouts,
        "nodes": node_records,
        "edges": edge_records,
        "boundary": net_cache["boundary"],
        "localDemGrid": export_local_dem_grid(net.root),
    }
    with _EXPLORER_LOCK:
        if len(net_cache["payloads"]) > 32:
            net_cache["payloads"].clear()
        net_cache["payloads"][base_key] = base_payload

    return {
        **base_payload,
        "config": {
            "basemap": basemap,
            "verticalExaggeration": float(vertical_exaggeration),
            "initialLayout": initial_layout,
            "selectedNodeId": selected_node_id,
            "selectedEdgeId": selected_edge_id,
            "activeFilters": filtered.active_filters,
        },
    }
