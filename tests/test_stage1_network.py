"""Stage 1 Automated Acceptance Tests (Geographic, Movement, Data-Model, Reproducibility, Performance)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_transhumance_network import build_stage1_network
from src.validation.network_stage1_checks import run_all_stage1_validations

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_generated_network_passes_all_acceptance_suites():
    net_dir = ROOT / "data/network"
    if not (net_dir / "network_manifest.json").exists():
        build_stage1_network(ROOT, seed=42)
    validation = run_all_stage1_validations(ROOT)
    for check in validation["checks"]:
        assert check["status"] == "PASS", f"{check['suite']} failed: {check['issues']}"
    assert validation["all_passed"] is True


def test_stage1_reproducibility_same_and_different_seeds(tmp_path: Path):
    # Stage minimal workspace pointing to authoritative Naryn boundary & config
    bpath = next(
        (p for p in [ROOT / "data/processed/vectors/naryn_adm1_clean.geojson", ROOT / "data/raw/boundaries/naryn_adm1.geojson"] if p.exists()),
        None,
    )
    if bpath is None:
        pytest.skip("Authoritative Naryn ADM1 boundary not present")

    def _make_ws(name: str) -> Path:
        ws = tmp_path / name
        (ws / "config").mkdir(parents=True)
        (ws / "data/processed/vectors").mkdir(parents=True)
        shutil.copy(ROOT / "config/network_stage1.yaml", ws / "config/network_stage1.yaml")
        shutil.copy(bpath, ws / "data/processed/vectors/naryn_adm1_clean.geojson")
        if (ROOT / "data/processed/locations.csv").exists():
            shutil.copy(ROOT / "data/processed/locations.csv", ws / "data/processed/locations.csv")
        return ws

    ws_a = _make_ws("run_seed42_a")
    ws_b = _make_ws("run_seed42_b")
    ws_c = _make_ws("run_seed99")

    build_stage1_network(ws_a, seed=42, num_herds=24)
    build_stage1_network(ws_b, seed=42, num_herds=24)
    build_stage1_network(ws_c, seed=99, num_herds=24)

    # Same seed -> exact identical herds, events, positions, utilisation
    for fn in ("herds.csv", "movement_events.csv", "herd_positions.csv", "edge_utilisation.csv"):
        df_a = pd.read_csv(ws_a / "data/network" / fn)
        df_b = pd.read_csv(ws_b / "data/network" / fn)
        # Drop generation_date if date rolls over midnight; compare deterministic fields
        cols = [c for c in df_a.columns if c != "generation_date"]
        assert df_a[cols].equals(df_b[cols]), f"Mismatch on identical seed=42 in {fn}"

    # Different seed -> distinct population and schedules
    herds_42 = pd.read_csv(ws_a / "data/network/herds.csv")
    herds_99 = pd.read_csv(ws_c / "data/network/herds.csv")
    assert not herds_42["animal_count"].equals(herds_99["animal_count"])
    assert not herds_42["spring_departure_date"].equals(herds_99["spring_departure_date"])
