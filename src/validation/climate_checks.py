"""Validation checks separating historical NASA POWER data from simulated scenarios."""
from __future__ import annotations

from typing import Any

import pandas as pd


def validate_climate_scenarios(historical_daily: pd.DataFrame, scenario_climate: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    required_scenarios = {"SCENARIO_BASELINE", "SCENARIO_WARMER_DRIER", "SCENARIO_EARLY_SNOW"}
    present = set(scenario_climate["scenario_id"].unique()) if not scenario_climate.empty else set()
    missing = required_scenarios - present
    if missing:
        issues.append(f"Missing climate scenarios: {sorted(missing)}")

    # Verify baseline equals actual NASA POWER historical observations identically
    base = scenario_climate.loc[scenario_climate["scenario_id"] == "SCENARIO_BASELINE"].copy()
    if not base.empty:
        hist = historical_daily.copy()
        hist["date_str"] = pd.to_datetime(hist["observation_date"]).dt.strftime("%Y-%m-%d")
        merged = base.merge(hist[["date_str", "t2m_c", "precipitation_mm_day"]], on="date_str", how="left")
        t_diff = (merged["scenario_t2m_c"] - merged["t2m_c"]).abs().max()
        p_diff = (merged["scenario_precip_mm_day"] - merged["precipitation_mm_day"]).abs().max()
        if pd.notna(t_diff) and t_diff > 1e-3:
            issues.append(f"Baseline scenario temperature deviates from historical NASA POWER by {t_diff}")
        if pd.notna(p_diff) and p_diff > 1e-3:
            issues.append(f"Baseline scenario precipitation deviates from historical NASA POWER by {p_diff}")
        if (base["origin_class"] != "public").any():
            issues.append("Baseline scenario must preserve origin_class='public'")

    # Verify non-baseline scenarios are explicitly marked simulated_scenario
    perturbed = scenario_climate.loc[scenario_climate["scenario_id"] != "SCENARIO_BASELINE"]
    if not perturbed.empty and not (perturbed["origin_class"] == "simulated_scenario").all():
        issues.append("Perturbed scenarios must have origin_class='simulated_scenario'")

    return {
        "check": "climate_scenarios",
        "rows": int(len(scenario_climate)),
        "scenarios_present": sorted(present),
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
    }
