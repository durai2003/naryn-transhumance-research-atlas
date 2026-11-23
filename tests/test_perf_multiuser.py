"""Automated verification for multi-user performance optimizations & session isolation."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from src.atlas.cesium_server import AtlasAssetServer
from src.atlas.data_access import load_atlas_data
from src.atlas.temporal import build_cesium_payload
from src.network.explorer import build_network_explorer_payload, compute_graph_layouts, filter_network
from src.network.network_loader import load_stage1_network

ROOT = Path(__file__).resolve().parents[1]


def test_app_uses_cache_resource_lazy_nav_and_no_1s_polling():
    app_txt = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "@st.cache_resource(show_spinner=False)\ndef _load_data(" in app_txt
    assert "st.tabs(" not in app_txt, "app.py should use lazy navigation instead of eager st.tabs()"
    assert "run_every=1.0" not in app_txt, "Unconditional 1-second polling should be removed"
    assert "_visitor_session_id" in app_txt

    ui_s2_txt = (ROOT / "src/network/ui_stage2.py").read_text(encoding="utf-8")
    assert "@st.cache_resource(show_spinner=False)\ndef _load_stage1_cached(" in ui_s2_txt
    assert 'layout_names=("geographic", "spring")' in ui_s2_txt


def test_cesium_payload_caches_static_layers_and_herds():
    data = load_atlas_data(ROOT)
    # Cold call
    t0 = time.perf_counter()
    p1 = build_cesium_payload(
        data,
        scenario_id="SCENARIO_BASELINE",
        selected_date="2024-07-02",
        species_filter=["sheep", "goat", "cattle", "horse", "mixed"],
        selected_herd_id="All herds",
        session_id="sess_alpha",
    )
    t1 = time.perf_counter()

    # Warm call (different visitor session & date on same scenario/species)
    t2 = time.perf_counter()
    p2 = build_cesium_payload(
        data,
        scenario_id="SCENARIO_BASELINE",
        selected_date="2024-07-15",
        species_filter=["sheep", "goat", "cattle", "horse", "mixed"],
        selected_herd_id="All herds",
        session_id="sess_beta",
    )
    t3 = time.perf_counter()

    assert p1["config"]["sessionId"] == "sess_alpha"
    assert p2["config"]["sessionId"] == "sess_beta"
    # Shared cached static GeoJSONs and herds list (zero re-serialization!)
    assert p1["boundary"] is p2["boundary"]
    assert p1["corridors"] is p2["corridors"]
    assert p1["herds"] is p2["herds"]
    assert p1["pastures"] is p2["pastures"]
    assert (t3 - t2) < max((t1 - t0) * 0.25, 0.05)


def test_network_explorer_selective_layouts_and_payload_cache():
    net = load_stage1_network(ROOT)
    if not net.available:
        return
    filtered = filter_network(net)

    # Selective UI layouts skip Kamada-Kawai
    fast_layouts = compute_graph_layouts(filtered, layout_names=("geographic", "spring"))
    assert set(fast_layouts.keys()) == {"geographic", "spring"}

    # Full default still returns all 4 layouts
    all_layouts = compute_graph_layouts(filtered)
    assert set(all_layouts.keys()) == {"geographic", "kamada_kawai", "layered_seasonal", "spring"}

    # Cached explorer payload reuses nodes/edges/boundary across visitors
    p1 = build_network_explorer_payload(net, filtered, layout_names=("geographic", "spring"), basemap="esri_satellite")
    p2 = build_network_explorer_payload(net, filtered, layout_names=("geographic", "spring"), basemap="osm")
    assert p1["nodes"] is p2["nodes"]
    assert p1["edges"] is p2["edges"]
    assert p1["boundary"] is p2["boundary"]
    assert p1["config"]["basemap"] == "esri_satellite"
    assert p2["config"]["basemap"] == "osm"


def test_per_session_ui_state_isolation_on_asset_server():
    srv = AtlasAssetServer(ROOT)
    try:
        srv.set_ui_state(date="2024-01-24", herd_id="SYN-HERD-001", scenario_id="SCENARIO_BASELINE", session_id="visitor_A")
        srv.set_ui_state(date="2024-07-02", herd_id="SYN-HERD-015", scenario_id="SCENARIO_BASELINE", session_id="visitor_B")

        assert srv.get_ui_state(session_id="visitor_A")["date"] == "2024-01-24"
        assert srv.get_ui_state(session_id="visitor_A")["herd_id"] == "SYN-HERD-001"
        assert srv.get_ui_state(session_id="visitor_B")["date"] == "2024-07-02"
        assert srv.get_ui_state(session_id="visitor_B")["herd_id"] == "SYN-HERD-015"

        # HTTP POST for visitor_A must not affect visitor_B
        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/api/ui_state",
            data=json.dumps({"session_id": "visitor_A", "date": "2024-05-10", "herd_id": "SYN-HERD-005"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            res_a = json.loads(r.read().decode("utf-8"))
            assert res_a["date"] == "2024-05-10"

        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/api/ui_state?session=visitor_B", timeout=3) as r:
            res_b = json.loads(r.read().decode("utf-8"))
            assert res_b["date"] == "2024-07-02" and res_b["herd_id"] == "SYN-HERD-015"
    finally:
        srv.shutdown()
