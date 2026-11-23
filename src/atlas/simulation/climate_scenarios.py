"""NASA POWER historical climate loading and transparent scenario perturbation engine.

Historical records are read unmodified from `data/processed/climate/nasa_power_daily.csv`.
Hypothetical climate perturbations (Warmer & Drier Summer; Early Snow / Earlier Autumn Return)
are stored separately in `data/synthetic/scenario_climate.csv` (and `.parquet` when pyarrow
is available) with explicit `origin_class='simulated_scenario'` metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def load_historical_daily_climate(root: Path) -> pd.DataFrame:
    path = Path(root).resolve() / "data/processed/climate/nasa_power_daily.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Actual NASA POWER daily climate table not found: {path}. "
            "Run Stage 3 first; no historical climate observations are fabricated."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("NASA POWER daily climate table contains 0 rows.")
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df = df.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)
    df["year"] = df["observation_date"].dt.year.astype(int)
    df["month"] = df["observation_date"].dt.month.astype(int)
    df["doy"] = df["observation_date"].dt.dayofyear.astype(int)
    df["season"] = df["month"].map(season_from_month)
    return df


def build_scenario_climate_table(
    historical_daily: pd.DataFrame,
    suitability_cfg: dict[str, Any],
    *,
    start_date: str,
    end_date: str,
    simulation_version: str = "2.0.0",
) -> pd.DataFrame:
    """Create a separate scenario climate table covering [start_date, end_date]."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    window = historical_daily.loc[
        (historical_daily["observation_date"] >= start_ts)
        & (historical_daily["observation_date"] <= end_ts)
    ].copy()
    if window.empty:
        raise ValueError(
            f"Requested simulation window {start_date}..{end_date} has no records in NASA POWER "
            f"(available: {historical_daily['observation_date'].min().date()}.."
            f"{historical_daily['observation_date'].max().date()})."
        )

    scenarios = suitability_cfg.get("scenarios", {})
    frames: list[pd.DataFrame] = []

    for scenario_id, spec in scenarios.items():
        df = window.copy()
        t_summer = float(spec.get("temp_offset_summer_c", 0.0))
        t_shoulder = float(spec.get("temp_offset_shoulder_c", 0.0))
        t_winter = float(spec.get("temp_offset_winter_c", 0.0))
        p_summer = float(spec.get("precip_multiplier_summer", 1.0))
        p_autumn = float(spec.get("precip_multiplier_autumn", 1.0))

        # Apply clearly documented seasonal perturbations (leave NaN as NaN)
        t_delta = np.where(
            df["month"].isin([6, 7, 8]), t_summer,
            np.where(df["month"].isin([3, 4, 5, 9, 10, 11]), t_shoulder, t_winter),
        )
        # Early-snow scenario intensifies cooling from mid-August onward
        if scenario_id == "SCENARIO_EARLY_SNOW":
            t_delta = np.where(df["doy"] >= 225, t_shoulder, t_delta)

        p_mult = np.where(
            df["month"].isin([6, 7, 8]), p_summer,
            np.where(df["month"].isin([9, 10, 11]), p_autumn, 1.0),
        )
        if scenario_id == "SCENARIO_EARLY_SNOW":
            p_mult = np.where(df["doy"] >= 225, p_autumn, p_mult)

        df["scenario_id"] = scenario_id
        df["scenario_label"] = str(spec.get("label", scenario_id))
        df["origin_class"] = str(spec.get("origin_class", "simulated_scenario"))
        df["historical_t2m_c"] = df["t2m_c"]
        df["historical_precip_mm_day"] = df["precipitation_mm_day"]
        df["scenario_t2m_c"] = np.where(df["t2m_c"].notna(), np.round(df["t2m_c"] + t_delta, 3), np.nan)
        df["scenario_precip_mm_day"] = np.where(
            df["precipitation_mm_day"].notna(),
            np.round(np.maximum(0.0, df["precipitation_mm_day"] * p_mult), 3),
            np.nan,
        )
        df["temp_anomaly_c"] = np.round(t_delta, 2)
        df["precip_multiplier"] = np.round(p_mult, 2)
        # Rolling 30-day cumulative precipitation for moisture availability context
        df["precip_30d_mm"] = df["scenario_precip_mm_day"].rolling(window=30, min_periods=1).sum().round(2)
        # Snow-risk proxy at high alpine band (3100m using -6.5 C/km lapse rate from 2044m)
        high_alpine_temp = df["scenario_t2m_c"] - 6.5 * ((3100.0 - 2044.0) / 1000.0)
        df["alpine_temp_3100m_c"] = np.round(high_alpine_temp, 2)
        df["high_alpine_snow_risk"] = ((high_alpine_temp <= -1.5) & (df["precip_30d_mm"] >= 12.0)).astype(bool)
        df["spatial_resolution"] = "0.5 x 0.625 deg MERRA-2 grid (NASA POWER; not a weather station)"
        df["is_synthetic_perturbation"] = scenario_id != "SCENARIO_BASELINE"
        df["simulation_version"] = simulation_version
        df["date_str"] = df["observation_date"].dt.strftime("%Y-%m-%d")

        keep = [
            "date_str", "year", "month", "doy", "season", "location_id",
            "scenario_id", "scenario_label", "origin_class",
            "historical_t2m_c", "historical_precip_mm_day",
            "scenario_t2m_c", "scenario_precip_mm_day",
            "temp_anomaly_c", "precip_multiplier", "precip_30d_mm",
            "alpine_temp_3100m_c", "high_alpine_snow_risk",
            "spatial_resolution", "is_synthetic_perturbation", "simulation_version",
        ]
        frames.append(df[keep])

    return pd.concat(frames, ignore_index=True)
