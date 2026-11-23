"""Automated data and geospatial validation checks for synthetic herds, pastures, and corridors."""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


def validate_herds(herds_df: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    if herds_df.empty:
        issues.append("Herds table is empty")
    if len(herds_df) < 40:
        issues.append(f"Expected >=40 synthetic herds, found {len(herds_df)}")
    if herds_df["herd_id"].duplicated().any():
        issues.append("Duplicate herd_id values found")
    if (herds_df["group_size"] <= 0).any():
        issues.append("Non-positive herd group_size detected")
    if not herds_df["is_synthetic"].all():
        issues.append("Every synthetic herd must have is_synthetic=True")
    if not (herds_df["provenance"] == "synthetic_demo").all():
        issues.append("Every synthetic herd must have provenance='synthetic_demo'")
    return {"check": "synthetic_herds", "rows": int(len(herds_df)), "status": "PASS" if not issues else "FAIL", "issues": issues}


def validate_pastures(pastures_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    issues: list[str] = []
    if pastures_gdf.crs is None or str(pastures_gdf.crs).upper() != "EPSG:4326":
        issues.append(f"Unexpected pasture CRS: {pastures_gdf.crs}")
    region = unary_union(boundary_gdf.to_crs("EPSG:4326").geometry)
    outside = (~pastures_gdf.geometry.within(region)).sum()
    if outside:
        issues.append(f"{int(outside)} candidate pastures fall outside Naryn ADM1")
    if pastures_gdf["pasture_id"].duplicated().any():
        issues.append("Duplicate pasture_id values")
    if not pastures_gdf["is_synthetic"].all():
        issues.append("Synthetic pastures missing is_synthetic=True")
    return {"check": "synthetic_pastures", "rows": int(len(pastures_gdf)), "outside_boundary": int(outside), "status": "PASS" if not issues else "FAIL", "issues": issues}


def validate_routes(routes_gdf: gpd.GeoDataFrame, pastures_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    issues: list[str] = []
    if routes_gdf.crs is None or str(routes_gdf.crs).upper() != "EPSG:4326":
        issues.append(f"Unexpected routes CRS: {routes_gdf.crs}")
    invalid_geom = (~routes_gdf.geometry.is_valid).sum()
    if invalid_geom:
        issues.append(f"{int(invalid_geom)} invalid route geometries")
    valid_pids = set(pastures_gdf["pasture_id"].astype(str))
    bad_orig = (~routes_gdf["origin_pasture_id"].astype(str).isin(valid_pids)).sum()
    bad_dst = (~routes_gdf["destination_pasture_id"].astype(str).isin(valid_pids)).sum()
    if bad_orig or bad_dst:
        issues.append(f"Unresolved route endpoints: orig={int(bad_orig)}, dst={int(bad_dst)}")
    # Allow tiny floating-point tolerance (~50m) at boundary vertices
    region_buf = unary_union(boundary_gdf.to_crs("EPSG:32643").geometry).buffer(100.0)
    routes_m = routes_gdf.to_crs("EPSG:32643")
    outside = (~routes_m.geometry.within(region_buf)).sum()
    if outside:
        issues.append(f"{int(outside)} routes extend outside Naryn ADM1")
    if (routes_gdf["distance_km"] <= 0).any():
        issues.append("Non-positive route distance detected")
    return {"check": "synthetic_routes", "rows": int(len(routes_gdf)), "outside_boundary": int(outside), "status": "PASS" if not issues else "FAIL", "issues": issues}
