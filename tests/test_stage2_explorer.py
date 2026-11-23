"""Stage 2 Automated Tests: Filtering, Components, Layouts, Inspectors, and Same-Origin HTTP Bridge."""
from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import pytest

from src.atlas.cesium_server import AtlasAssetServer
from src.network.explorer import (
    build_network_explorer_payload,
    compute_graph_layouts,
    filter_network,
)
from src.network.network_loader import load_stage1_network

ROOT = Path(__file__).resolve().parents[1]


def _require_stage1():
    net = load_stage1_network(ROOT)
    if not net.available:
        pytest.skip("Stage 1 network artifacts not generated under data/network/")
    return net


def test_stage2_filters_and_public_only_isolation():
    net = _require_stage1()

    # Full unfiltered view
    full = filter_network(net)
    assert len(full.nodes_gdf) >= 70
    assert len(full.edges_gdf) >= 140
    assert len(full.components) >= 1

    # Public-only mode MUST hide 100% of synthetic nodes and edges
    pub = filter_network(net, public_only=True)
    assert len(pub.nodes_gdf) >= 1
    assert not pub.nodes_gdf["is_synthetic"].astype(bool).any()
    if not pub.edges_gdf.empty:
        assert not pub.edges_gdf["is_synthetic"].astype(bool).any()

    # Edge type + Season filter
    summer_lc = filter_network(
        net,
        edge_types=["LEAST_COST_CORRIDOR"],
        seasons=["summer"],
    )
    assert not summer_lc.edges_gdf.empty
    assert set(summer_lc.edges_gdf["edge_type"].unique()) == {"LEAST_COST_CORRIDOR"}
    assert set(summer_lc.edges_gdf["season"].unique()) == {"summer"}

    # Livestock filter (e.g., cattle only)
    cattle = filter_network(net, livestock_types=["cattle"])
    assert len(cattle.nodes_gdf) == len(full.nodes_gdf)
    assert len(cattle.edges_gdf) <= len(full.edges_gdf)


def test_stage2_layouts_components_and_inspector_completeness():
    net = _require_stage1()
    filtered = filter_network(net)

    # All 4 layouts must produce finite normalized coordinates for every visible node
    layouts = compute_graph_layouts(filtered)
    expected_layouts = {"layered_seasonal", "geographic", "spring", "kamada_kawai"}
    assert set(layouts.keys()) == expected_layouts
    visible_ids = set(filtered.nodes_gdf["node_id"].astype(str))
    for lname, pos_map in layouts.items():
        assert set(pos_map.keys()) == visible_ids, f"Layout {lname} missing nodes"
        for nid, xy in pos_map.items():
            assert len(xy) == 2 and all(math.isfinite(v) for v in xy)
            assert 0.0 <= xy[0] <= 1.0 and 0.0 <= xy[1] <= 1.0

    # Verify component assignment covers every node
    assert set(filtered.node_to_component.keys()) == visible_ids
    assert sum(c["node_count"] for c in filtered.components) == len(visible_ids)

    # Payload node & edge inspector fields
    payload = build_network_explorer_payload(net, filtered)
    assert payload["summary"]["visible_nodes"] == len(visible_ids)
    assert payload["summary"]["visible_edges"] == len(filtered.edges_gdf)

    required_node_fields = {
        "node_id", "node_name", "node_type", "network_layer", "latitude", "longitude",
        "elevation_m", "elevation_status", "landcover_class", "connected_routes",
        "connected_herds", "in_neighbors", "out_neighbors", "component_id",
        "source", "provenance", "evidence_status", "is_synthetic",
    }
    for n in payload["nodes"]:
        assert required_node_fields.issubset(n.keys())

    # At least the seasonal pastures must link to connected herds & routes
    pastures = [n for n in payload["nodes"] if n["node_type"].endswith("_pasture")]
    assert pastures and any(p["connected_herd_count"] > 0 for p in pastures)
    assert all(len(p["connected_routes"]) > 0 for p in pastures)

    required_edge_fields = {
        "edge_id", "source_node_id", "target_node_id", "edge_type", "distance_m",
        "distance_km", "average_slope", "maximum_slope", "elevation_profile",
        "season", "route_generation_method", "utilisation", "coordinates",
        "source", "provenance", "evidence_status", "is_synthetic",
    }
    for e in payload["edges"]:
        assert required_edge_fields.issubset(e.keys())
        assert len(e["coordinates"]) >= 2


def test_stage2_same_origin_http_server_preserves_atlas_and_serves_network_explorer():
    srv = AtlasAssetServer(ROOT)
    try:
        # Backward compatibility: existing 3D Atlas endpoints
        s_atlas = srv.register_payload(json.dumps({"config": {"selectedDate": "2024-07-02"}}))
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/health", timeout=3) as r:
            health = json.loads(r.read().decode("utf-8"))
            assert r.status == 200 and health["network_explorer"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/payload.json?state={s_atlas}", timeout=3) as r:
            assert json.loads(r.read().decode("utf-8"))["config"]["selectedDate"] == "2024-07-02"

        # New Stage 2 Network Explorer endpoints
        s_net = srv.register_network_payload(json.dumps({"stage": "stage2_network_explorer", "nodes": [], "edges": []}))
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/network/?state={s_net}", timeout=3) as r:
            html = r.read().decode("utf-8")
            assert r.status == 200 and "Stage 2" in html and "graphCanvas" in html
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/network/payload.json?state={s_net}", timeout=3) as r:
            pdata = json.loads(r.read().decode("utf-8"))
            assert pdata["stage"] == "stage2_network_explorer"
    finally:
        srv.shutdown()
