"""Temporal continuity, anti-teleportation, and provenance checks for herd trajectories."""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union


def validate_trajectories(
    traj_df: pd.DataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    *,
    max_daily_jump_km: float = 35.0,
) -> dict[str, Any]:
    issues: list[str] = []
    if traj_df.empty:
        return {"check": "herd_trajectories", "rows": 0, "status": "FAIL", "issues": ["Trajectory dataset is empty"]}
    if len(traj_df) < 10000:
        issues.append(f"Expected >=10,000 trajectory rows, found {len(traj_df)}")

    # Chronological ordering and continuity (no teleportation)
    max_observed_jump = 0.0
    teleport_count = 0
    order_errors = 0

    for (scen, hid), grp in traj_df.groupby(["climate_scenario", "herd_id"]):
        ts = pd.to_datetime(grp["timestamp"], errors="coerce")
        if ts.isna().any() or not ts.is_monotonic_increasing:
            order_errors += 1
        lats = grp["latitude"].to_numpy(dtype=float)
        lons = grp["longitude"].to_numpy(dtype=float)
        if len(lats) > 1:
            dlat = np.diff(lats) * 110.54
            dlon = np.diff(lons) * 111.32 * np.cos(np.radians(0.5 * (lats[:-1] + lats[1:])))
            step_km = np.hypot(dlat, dlon)
            peak = float(np.max(step_km))
            max_observed_jump = max(max_observed_jump, peak)
            teleport_count += int(np.sum(step_km > max_daily_jump_km))

    if order_errors:
        issues.append(f"{order_errors} herd trajectories have invalid or unsorted timestamps")
    if teleport_count:
        issues.append(f"{teleport_count} daily steps exceed {max_daily_jump_km} km/day (max={max_observed_jump:.1f} km)")

    # Boundary containment check on unique sampled positions
    uniq = traj_df[["longitude", "latitude"]].drop_duplicates()
    region_buf = unary_union(boundary_gdf.to_crs("EPSG:32643").geometry).buffer(150.0)
    pts_m = gpd.GeoSeries([Point(x, y) for x, y in zip(uniq["longitude"], uniq["latitude"])], crs="EPSG:4326").to_crs("EPSG:32643")
    outside = int((~pts_m.within(region_buf)).sum())
    if outside:
        issues.append(f"{outside} unique trajectory coordinates fall outside Naryn ADM1")

    if not traj_df["is_synthetic"].all() or not (traj_df["provenance"] == "synthetic_demo").all():
        issues.append("Trajectory provenance flags are incomplete")

    return {
        "check": "herd_trajectories",
        "rows": int(len(traj_df)),
        "max_daily_step_km": round(max_observed_jump, 2),
        "teleport_violations": teleport_count,
        "outside_boundary_points": outside,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
    }
