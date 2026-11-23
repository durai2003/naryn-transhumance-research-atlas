"""Verification tests for Naryn Atlas v3.2 timeline, herd, pasture, corridor, and climate synchronisation."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from src.atlas.cesium_server import AtlasAssetServer
from src.atlas.data_access import load_atlas_data
from src.atlas.temporal import (
    MOVING_STATES,
    build_cesium_payload,
    reconcile_trajectory_pasture_state,
    resolve_corridor_profile,
    resolve_herd_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def test_jan_24_2024_stationary_winter_sync():
    data = load_atlas_data(ROOT)
    snap = resolve_herd_snapshot(
        data,
        scenario_id="SCENARIO_BASELINE",
        selected_date="2024-01-24",
        species_filter=["sheep", "goat", "cattle", "horse", "mixed"],
        selected_herd_id="SYN-HERD-015",
    )
    assert snap["date"] == "2024-01-24"
    assert snap["climate"]["season"] == "winter"
    assert snap["total_herds"] == 48
    assert snap["moving_herds"] == 0
    assert snap["stationary_herds"] == 48

    h = snap["herd"]
    assert h["herd_id"] == "SYN-HERD-015"
    assert h["movement_state"] == "grazing"
    assert h["is_moving"] is False
    assert h["current_pasture_id"] == "DEMO-PASTURE-003"
    assert h["next_pasture_id"] == "DEMO-PASTURE-009"
    assert h["current_pasture_id"] != h["next_pasture_id"]
    assert h["route_id"] == "SYN-CORRIDOR-003"
    # Climate context matches scenario_climate exactly
    assert h["temperature_c"] == snap["climate"]["t2m_c"]
    assert h["precipitation_mm"] == snap["climate"]["precip_mm"]
    assert round(h["temperature_c"], 1) == -9.8

    # Corridor elevation profile resolves to next planned route SYN-CORRIDOR-003 (not SYN-CORRIDOR-001!)
    prof = resolve_corridor_profile(
        data, h["route_id"],
        origin_pasture_id=h["current_pasture_id"],
        dest_pasture_id=h["next_pasture_id"],
    )
    assert prof is not None
    assert prof["route_id"] == "SYN-CORRIDOR-003"
    assert len(prof["profile"]) >= 5


def test_jul_2_2024_active_migration_syn_herd_015():
    data = load_atlas_data(ROOT)
    # Test both "All herds" (auto-featured active moving herd) and explicit SYN-HERD-015
    for sel in ("All herds", "SYN-HERD-015"):
        snap = resolve_herd_snapshot(
            data,
            scenario_id="SCENARIO_BASELINE",
            selected_date="2024-07-02",
            species_filter=["sheep", "goat", "cattle", "horse", "mixed"],
            selected_herd_id=sel,
        )
        assert snap["date"] == "2024-07-02"
        assert snap["climate"]["season"] == "summer"
        assert snap["moving_herds"] == 1
        assert snap["stationary_herds"] == 47
        assert snap["total_herds"] == 48

        h = snap["herd"]
        assert h["herd_id"] == "SYN-HERD-015"
        assert h["movement_state"] == "moving"
        assert h["is_moving"] is True
        # Fixed Bug 3: origin != destination while actively moving!
        assert h["current_pasture_id"] == "DEMO-PASTURE-009"
        assert h["next_pasture_id"] == "DEMO-PASTURE-015"
        assert h["current_pasture_id"] != h["next_pasture_id"]
        assert h["route_id"] == "SYN-CORRIDOR-012"
        assert h["temperature_c"] == snap["climate"]["t2m_c"]
        assert h["precipitation_mm"] == snap["climate"]["precip_mm"]
        assert round(h["temperature_c"], 1) == 8.6

        # Fixed Bug 2: elevation profile resolves to SYN-CORRIDOR-012, NOT SYN-CORRIDOR-001!
        prof = resolve_corridor_profile(
            data, h["route_id"],
            origin_pasture_id=h["current_pasture_id"],
            dest_pasture_id=h["next_pasture_id"],
        )
        assert prof is not None
        assert prof["route_id"] == "SYN-CORRIDOR-012"
        assert prof["origin_pasture_id"] == "DEMO-PASTURE-009"
        assert prof["destination_pasture_id"] == "DEMO-PASTURE-015"


def test_arrival_and_multiple_herds_different_corridors():
    data = load_atlas_data(ROOT)

    # Arrival date for SYN-HERD-015 (2024-07-03): arrived at DEMO-PASTURE-015, next planned is DEMO-PASTURE-009
    arr_snap = resolve_herd_snapshot(
        data, scenario_id="SCENARIO_BASELINE", selected_date="2024-07-03", selected_herd_id="SYN-HERD-015"
    )
    h_arr = arr_snap["herd"]
    assert h_arr["movement_state"] == "arrived"
    assert h_arr["current_pasture_id"] == "DEMO-PASTURE-015"
    assert h_arr["next_pasture_id"] == "DEMO-PASTURE-009"
    assert h_arr["current_pasture_id"] != h_arr["next_pasture_id"]

    # Herd 1 (SYN-HERD-001) during active summer migration on 2024-06-04 -> SYN-CORRIDOR-009
    s1 = resolve_herd_snapshot(
        data, scenario_id="SCENARIO_BASELINE", selected_date="2024-06-04", selected_herd_id="SYN-HERD-001"
    )
    assert s1["herd"]["is_moving"] is True
    assert s1["herd"]["current_pasture_id"] == "DEMO-PASTURE-008"
    assert s1["herd"]["next_pasture_id"] == "DEMO-PASTURE-014"
    assert s1["herd"]["route_id"] == "SYN-CORRIDOR-009"
    p1 = resolve_corridor_profile(data, s1["herd"]["route_id"], origin_pasture_id=s1["herd"]["current_pasture_id"], dest_pasture_id=s1["herd"]["next_pasture_id"])
    assert p1 is not None and p1["route_id"] == "SYN-CORRIDOR-009"

    # Herd 2 (SYN-HERD-005) during active summer migration on 2024-06-05 -> SYN-CORRIDOR-016
    s2 = resolve_herd_snapshot(
        data, scenario_id="SCENARIO_BASELINE", selected_date="2024-06-05", selected_herd_id="SYN-HERD-005"
    )
    assert s2["herd"]["is_moving"] is True
    assert s2["herd"]["current_pasture_id"] == "DEMO-PASTURE-012"
    assert s2["herd"]["next_pasture_id"] == "DEMO-PASTURE-014"
    assert s2["herd"]["route_id"] == "SYN-CORRIDOR-016"
    p2 = resolve_corridor_profile(data, s2["herd"]["route_id"], origin_pasture_id=s2["herd"]["current_pasture_id"], dest_pasture_id=s2["herd"]["next_pasture_id"])
    assert p2 is not None and p2["route_id"] == "SYN-CORRIDOR-016"
    assert p1["route_id"] != p2["route_id"]


def test_zero_current_equals_next_across_entire_reconciled_dataset_and_payload():
    data = load_atlas_data(ROOT)
    rec = reconcile_trajectory_pasture_state(data.herd_trajectories, data.movement_events)
    assert int((rec["current_pasture_id"] == rec["next_pasture_id"]).sum()) == 0

    # Verify unmatched route returns None rather than unrelated SYN-CORRIDOR-001
    assert resolve_corridor_profile(data, "NONEXISTENT-ROUTE-999") is None

    # Verify Cesium payload samples contain route_id (10 elements per sample) and reconciled pastures
    payload = build_cesium_payload(
        data,
        scenario_id="SCENARIO_BASELINE",
        selected_date="2024-07-02",
        species_filter=["sheep", "goat", "cattle", "horse", "mixed"],
        selected_herd_id="All herds",
    )
    h15 = next(h for h in payload["herds"] if h["herd_id"] == "SYN-HERD-015")
    jul2 = next(s for s in h15["samples"] if s[0].startswith("2024-07-02"))
    assert len(jul2) == 10
    assert jul2[4] == "moving"
    assert jul2[5] == "DEMO-PASTURE-009"
    assert jul2[6] == "DEMO-PASTURE-015"
    assert jul2[9] == "SYN-CORRIDOR-012"


def test_asset_server_ui_state_sync_endpoint():
    srv = AtlasAssetServer(ROOT)
    try:
        srv.set_ui_state(date="2024-07-02", herd_id="SYN-HERD-015", scenario_id="SCENARIO_BASELINE")
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/api/ui_state", timeout=3) as r:
            st0 = json.loads(r.read().decode("utf-8"))
            assert st0["date"] == "2024-07-02" and st0["herd_id"] == "SYN-HERD-015"

        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/api/ui_state",
            data=json.dumps({"date": "2024-01-24", "herd_id": "SYN-HERD-001", "moving_count": 0, "total_herds": 48}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            st1 = json.loads(r.read().decode("utf-8"))
            assert st1["date"] == "2024-01-24" and st1["herd_id"] == "SYN-HERD-001" and st1["moving_count"] == 0
    finally:
        srv.shutdown()
