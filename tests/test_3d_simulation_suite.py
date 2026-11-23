"""Automated pytest suite for the 3D Spatiotemporal Transhumance Atlas upgrade."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.atlas.data_access import find_boundary_path, load_atlas_data, read_geojson_safe
from src.atlas.temporal import build_cesium_payload
from src.validation.climate_checks import validate_climate_scenarios
from src.validation.simulation_checks import validate_herds, validate_pastures, validate_routes
from src.validation.trajectory_checks import validate_trajectories

ROOT = Path(__file__).resolve().parents[1]


def test_simulation_outputs_pass_all_checks():
    bpath = find_boundary_path(ROOT)
    if bpath is None:
        pytest.skip("Authoritative Naryn boundary not present")
    boundary = read_geojson_safe(bpath)
    syn = ROOT / "data/synthetic"
    herds = pd.read_csv(syn / "herds.csv")
    pastures = read_geojson_safe(syn / "pastures.geojson")
    routes = read_geojson_safe(syn / "migration_routes.geojson")
    traj = pd.read_csv(syn / "herd_trajectories.csv")
    clim = pd.read_csv(syn / "scenario_climate.csv")
    hist = pd.read_csv(ROOT / "data/processed/climate/nasa_power_daily.csv")

    assert validate_herds(herds)["status"] == "PASS"
    assert validate_pastures(pastures, boundary)["status"] == "PASS"
    assert validate_routes(routes, pastures, boundary)["status"] == "PASS"
    assert validate_trajectories(traj, boundary)["status"] == "PASS"
    assert validate_climate_scenarios(hist, clim)["status"] == "PASS"


def test_at_least_one_herd_changes_destination_across_scenarios():
    events = pd.read_csv(ROOT / "data/synthetic/movement_events.csv")
    changed = events.loc[events["scenario_changed_route"], "herd_id"].nunique()
    assert changed >= 1


def test_cesium_payload_builds_cleanly():
    data = load_atlas_data(ROOT)
    payload = build_cesium_payload(
        data,
        scenario_id="SCENARIO_WARMER_DRIER",
        selected_date="2024-07-15",
        species_filter=["sheep", "horse"],
        selected_herd_id="SYN-HERD-001",
    )
    assert len(payload["herds"]) > 0
    assert len(payload["climate"]) > 300
    assert payload["config"]["selectedDate"] == "2024-07-15"
    assert payload["config"]["scenarioId"] == "SCENARIO_WARMER_DRIER"


def test_cesium_frontend_template_present():
    html_path = ROOT / "frontend/cesium/index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "/*__ATLAS_PAYLOAD_JSON__*/ {}" in content
    assert "SampledPositionProperty" in content
