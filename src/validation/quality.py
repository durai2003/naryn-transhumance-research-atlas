"""Reusable data-quality checks for tables, vectors, rasters and registers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
import rasterio


SYNTHETIC_PROVENANCE_TOKENS = {"synthetic_demo", "synthetic", "fictional", "test_fixture"}


def validate_table(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    required_columns: Iterable[str],
    primary_key: str | None = None,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
    synthetic_flag_column: str | None = "is_synthetic",
    provenance_column: str | None = "provenance",
    disclaimer_column: str | None = "synthetic_disclaimer",
) -> dict[str, Any]:
    """Return a machine-readable validation summary; never silently fixes data."""
    required = list(required_columns)
    missing_columns = [col for col in required if col not in frame.columns]
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_required_columns": missing_columns,
        "null_counts": {},
        "duplicate_primary_key_rows": 0,
        "invalid_coordinate_rows": 0,
        "synthetic_provenance_issues": 0,
        "status": "PASS",
        "issues": [],
    }
    if missing_columns:
        result["issues"].append(f"Missing required columns: {missing_columns}")

    for column in required:
        if column in frame.columns:
            nulls = int(frame[column].isna().sum())
            result["null_counts"][column] = nulls
            if nulls:
                result["issues"].append(f"{column}: {nulls} null values")

    if primary_key and primary_key in frame.columns:
        duplicates = frame[primary_key].duplicated(keep=False) | frame[primary_key].isna()
        result["duplicate_primary_key_rows"] = int(duplicates.sum())
        if duplicates.any():
            result["issues"].append(f"Primary key {primary_key!r} has duplicates or nulls")

    if latitude_column and longitude_column and {latitude_column, longitude_column}.issubset(frame.columns):
        lat = pd.to_numeric(frame[latitude_column], errors="coerce")
        lon = pd.to_numeric(frame[longitude_column], errors="coerce")
        bad = lat.isna() | lon.isna() | ~lat.between(-90, 90) | ~lon.between(-180, 180)
        result["invalid_coordinate_rows"] = int(bad.sum())
        if bad.any():
            result["issues"].append(f"{int(bad.sum())} rows have missing or out-of-range WGS84 coordinates")

    if synthetic_flag_column in frame.columns and provenance_column in frame.columns:
        synthetic = frame[synthetic_flag_column].map(
            lambda value: str(value).strip().casefold() in {"true", "1", "yes", "y"}
        )
        provenance = frame[provenance_column].fillna("").astype(str).str.casefold()
        claims_synthetic = provenance.str.contains(r"synthetic|fictional|test_fixture", regex=True)
        inconsistent = claims_synthetic & ~synthetic
        if disclaimer_column and disclaimer_column in frame.columns:
            disclaimer = frame[disclaimer_column].fillna("").astype(str).str.strip()
            missing_disclaimer = synthetic & disclaimer.eq("")
        else:
            missing_disclaimer = synthetic
        result["synthetic_provenance_issues"] = int((inconsistent | missing_disclaimer).sum())
        if (inconsistent | missing_disclaimer).any():
            result["issues"].append("Synthetic flag/provenance/disclaimer fields are inconsistent")

    if result["issues"]:
        result["status"] = "FAIL"
    return result


def validate_geodataframe(
    gdf: gpd.GeoDataFrame,
    *,
    dataset_name: str,
    id_column: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "features": int(len(gdf)),
        "crs": str(gdf.crs) if gdf.crs else None,
        "missing_crs": gdf.crs is None,
        "null_geometries": int(gdf.geometry.isna().sum()) if hasattr(gdf, "geometry") else None,
        "empty_geometries": int(gdf.geometry.is_empty.sum()) if hasattr(gdf, "geometry") else None,
        "invalid_geometries": int((~gdf.geometry.is_valid.fillna(False)).sum()) if hasattr(gdf, "geometry") else None,
        "duplicate_id_rows": 0,
        "status": "PASS",
        "issues": [],
    }
    if result["missing_crs"]:
        result["issues"].append("CRS missing")
    if result["null_geometries"]:
        result["issues"].append("Null geometries present")
    if result["empty_geometries"]:
        result["issues"].append("Empty geometries present")
    if result["invalid_geometries"]:
        result["issues"].append("Invalid geometries present")
    if id_column and id_column in gdf.columns:
        result["duplicate_id_rows"] = int(gdf[id_column].duplicated(keep=False).sum())
        if result["duplicate_id_rows"]:
            result["issues"].append(f"Duplicate IDs in {id_column}")
    if result["issues"]:
        result["status"] = "FAIL"
    return result


def inspect_raster(path: str | Path, *, dataset_name: str) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "dataset": dataset_name,
            "path": str(path),
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": list(src.dtypes),
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform),
            "resolution": list(src.res),
            "nodata": src.nodata,
            "bounds": list(src.bounds),
            "status": "PASS" if src.crs and src.width > 0 and src.height > 0 else "FAIL",
        }


def validate_dataset_register(register_path: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(register_path, dtype="string")
    required = ["dataset_id", "dataset_title", "provider", "source_url", "expected_format", "licence", "acquisition_status"]
    missing = [column for column in required if column not in frame.columns]
    duplicates = int(frame["dataset_id"].duplicated(keep=False).sum()) if "dataset_id" in frame.columns else 0
    nulls = {column: int(frame[column].isna().sum()) for column in required if column in frame.columns}
    issues = []
    if missing:
        issues.append(f"Missing required register columns: {missing}")
    if duplicates:
        issues.append(f"Duplicate dataset IDs: {duplicates} rows")
    for column, count in nulls.items():
        if count:
            issues.append(f"Register field {column} has {count} nulls")
    return {"rows": int(len(frame)), "required_columns": required, "missing_columns": missing,
            "duplicate_id_rows": duplicates, "null_counts": nulls,
            "status": "PASS" if not issues else "FAIL", "issues": issues}
