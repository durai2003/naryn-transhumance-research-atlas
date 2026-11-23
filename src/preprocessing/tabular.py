"""Cleaning and standardisation for tabular source data.

Functions in this module never synthesize missing observations. Missing-value
sentinels are converted to nulls, not zeroes, and source provenance is retained.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

NASA_PARAMETERS = {
    "T2M": ("t2m_c", "degree_C"),
    "PRECTOTCORR": ("precipitation_mm_day", "mm/day"),
}
MISSING_SENTINELS = {"-999", "-999.0", "-9999", "-9999.0", "-999.00", "nan", "", "na", "n/a", "null", "none"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_column_name(name: Any) -> str:
    """Convert a column label to predictable snake_case."""
    value = str(name).strip()
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def normalise_text(value: Any, *, title_case: bool = False) -> Any:
    """Trim/collapse whitespace; preserve nulls and otherwise preserve case."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return pd.NA
    return text.title() if title_case else text


def parse_numeric(value: Any) -> float | None:
    """Parse numeric values, mapping documented missing sentinels to None."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    text = str(value).strip().casefold()
    if text in MISSING_SENTINELS:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def standardise_frame(
    frame: pd.DataFrame,
    *,
    text_columns: tuple[str, ...] = (),
    date_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return a standardised copy: snake_case columns, trimmed text, ISO dates."""
    out = frame.copy()
    out.columns = [normalise_column_name(c) for c in out.columns]
    for column in text_columns:
        key = normalise_column_name(column)
        if key in out.columns:
            out[key] = out[key].map(normalise_text).astype("string")
    for column in date_columns:
        key = normalise_column_name(column)
        if key in out.columns:
            out[key] = pd.to_datetime(out[key], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
            out[key] = out[key].astype("string")
    return out


def _nasa_properties(document: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        properties = document["properties"]
        parameters = properties["parameter"]
    except (KeyError, TypeError) as exc:
        raise ValueError("NASA POWER JSON missing properties.parameter mapping") from exc
    if not isinstance(parameters, Mapping):
        raise ValueError("NASA POWER properties.parameter must be an object")
    return parameters


def clean_nasa_power_document(
    document: Mapping[str, Any],
    *,
    location_id: str,
    location_name: str,
    latitude: float,
    longitude: float,
    source_url: str,
    retrieved_at: str | None = None,
    is_synthetic: bool = False,
    provenance: str = "nasa_power_api",
    synthetic_disclaimer: str | None = None,
) -> pd.DataFrame:
    """Flatten NASA POWER daily JSON into one row per date.

    The parser accepts the NASA POWER `properties.parameter` object and only
    emits dates/parameters that are present in the response. It does not pad
    gaps or impute missing values.
    """
    if not (-90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180):
        raise ValueError("Latitude/longitude outside valid WGS84 ranges")
    if is_synthetic and not (synthetic_disclaimer and synthetic_disclaimer.strip()):
        raise ValueError("Synthetic test/demo records require an explicit disclaimer")
    if is_synthetic and provenance != "synthetic_demo":
        raise ValueError("Synthetic records must use provenance='synthetic_demo'")
    if not is_synthetic and provenance == "synthetic_demo":
        raise ValueError("provenance='synthetic_demo' requires is_synthetic=True")
    parameters = _nasa_properties(document)
    available = {name: parameters[name] for name in NASA_PARAMETERS if name in parameters}
    if not available:
        raise ValueError(f"NASA POWER response contains none of the supported parameters: {list(NASA_PARAMETERS)}")

    date_sets: list[set[str]] = []
    for parameter_name, values in available.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"NASA POWER parameter {parameter_name} must map YYYYMMDD to values")
        date_sets.append(set(str(k) for k in values.keys()))
    dates = sorted(set.union(*date_sets)) if date_sets else []
    rows: list[dict[str, Any]] = []
    # Retrieval time is source provenance, not processing time. Do not invent it
    # when the acquisition log does not establish it.
    retrieved = retrieved_at
    processed_at = utc_now_iso()
    display_name = normalise_text(location_name, title_case=True)

    for date_key in dates:
        try:
            date_value = datetime.strptime(date_key, "%Y%m%d").date().isoformat()
        except ValueError:
            # Retain no unparseable date key; report the issue via missing rows.
            continue
        row: dict[str, Any] = {
            "observation_id": f"NASA-POWER:{location_id}:{date_value}",
            "dataset_id": "NASA-POWER-DAILY",
            "location_id": location_id,
            "location_name": display_name,
            "observation_date": date_value,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "is_synthetic": bool(is_synthetic),
            "provenance": provenance,
            "synthetic_disclaimer": synthetic_disclaimer or "",
            "source_url": source_url,
            "retrieved_at": retrieved,
            "processed_at": processed_at,
        }
        for source_parameter, (target_column, _unit) in NASA_PARAMETERS.items():
            source_values = available.get(source_parameter, {})
            row[target_column] = parse_numeric(source_values.get(date_key)) if date_key in source_values else None
        rows.append(row)

    columns = [
        "observation_id", "dataset_id", "location_id", "location_name", "observation_date",
        "latitude", "longitude", "t2m_c", "precipitation_mm_day", "is_synthetic",
        "provenance", "synthetic_disclaimer", "source_url", "retrieved_at", "processed_at",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_climate(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly and meteorological-season summaries from daily values."""
    expected = {"observation_date", "location_id", "t2m_c", "precipitation_mm_day"}
    missing = expected - set(observations.columns)
    if missing:
        raise ValueError(f"Climate table missing columns: {sorted(missing)}")
    if observations.empty:
        monthly_columns = ["location_id", "year", "month", "t2m_mean_c", "precipitation_sum_mm", "daily_records", "t2m_valid_days", "precipitation_valid_days"]
        seasonal_columns = ["location_id", "year", "season", "t2m_mean_c", "precipitation_sum_mm", "daily_records", "t2m_valid_days", "precipitation_valid_days"]
        return pd.DataFrame(columns=monthly_columns), pd.DataFrame(columns=seasonal_columns)

    frame = observations.copy()
    frame["_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame = frame.loc[frame["_date"].notna()].copy()
    frame["year"] = frame["_date"].dt.year.astype("Int64")
    frame["month"] = frame["_date"].dt.month.astype("Int64")
    monthly = (
        frame.groupby(["location_id", "year", "month"], dropna=False)
        .agg(t2m_mean_c=("t2m_c", "mean"),
             precipitation_sum_mm=("precipitation_mm_day", lambda values: values.sum(min_count=1)),
             daily_records=("observation_date", "count"), t2m_valid_days=("t2m_c", "count"),
             precipitation_valid_days=("precipitation_mm_day", "count"))
        .reset_index()
    )
    month_to_season = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    frame["season"] = frame["_date"].dt.month.map(month_to_season)
    # Winter is grouped by calendar year; December remains in its own calendar year.
    seasonal = (
        frame.groupby(["location_id", "year", "season"], dropna=False)
        .agg(t2m_mean_c=("t2m_c", "mean"),
             precipitation_sum_mm=("precipitation_mm_day", lambda values: values.sum(min_count=1)),
             daily_records=("observation_date", "count"), t2m_valid_days=("t2m_c", "count"),
             precipitation_valid_days=("precipitation_mm_day", "count"))
        .reset_index()
    )
    return monthly, seasonal


def read_and_clean_nasa_power_file(
    path: str,
    *,
    location_id: str,
    location_name: str,
    latitude: float,
    longitude: float,
    source_url: str,
) -> pd.DataFrame:
    """Read a cached NASA POWER JSON file and normalise it."""
    import json
    from pathlib import Path

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return clean_nasa_power_document(
        document, location_id=location_id, location_name=location_name,
        latitude=latitude, longitude=longitude, source_url=source_url,
    )
