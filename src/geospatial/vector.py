"""Vector cleaning, CRS standardisation and boundary clipping."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely import make_valid

from src.preprocessing.tabular import normalise_text

WEB_CRS = "EPSG:4326"
# Naryn is approximately within UTM zone 43N; use this projected CRS for metric
# measurements only after verifying feature coordinates and local suitability.
ANALYSIS_CRS = "EPSG:32643"


def clean_geodataframe(
    gdf: gpd.GeoDataFrame,
    *,
    id_column: str | None = None,
    target_crs: str = WEB_CRS,
    fix_invalid: bool = True,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Validate, repair and standardise a vector layer.

    Missing CRS is an error: CRS is never guessed or assigned from coordinates.
    Empty geometries are removed; invalid geometries are repaired with Shapely.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Expected a GeoDataFrame")
    if gdf.crs is None:
        raise ValueError("Input vector layer has no CRS; refusing to guess or assign one")
    report = {
        "input_features": int(len(gdf)),
        "source_crs": str(gdf.crs),
        "target_crs": target_crs,
        "empty_removed": 0,
        "invalid_before": int((~gdf.geometry.is_valid.fillna(False)).sum()),
        "repaired": 0,
        "duplicates_removed": 0,
        "out_of_range_removed": 0,
    }
    out = gdf.copy()
    # Preserve upstream field names/types but trim/collapse string attributes.
    for column in out.columns:
        if column != out.geometry.name and (pd.api.types.is_object_dtype(out[column]) or pd.api.types.is_string_dtype(out[column])):
            out[column] = out[column].map(normalise_text).astype("string")
    out = out.loc[out.geometry.notna()].copy()
    empty_mask = out.geometry.is_empty
    report["empty_removed"] = int(empty_mask.sum())
    out = out.loc[~empty_mask].copy()

    if fix_invalid and len(out):
        invalid = ~out.geometry.is_valid
        report["repaired"] = int(invalid.sum())
        if invalid.any():
            out.loc[invalid, out.geometry.name] = out.loc[invalid].geometry.map(make_valid)
    out = out.set_geometry(out.geometry.name)
    out = out.to_crs(target_crs)

    # Geographic coordinate range checks are only meaningful in EPSG:4326.
    if str(out.crs).upper() in {"EPSG:4326", "OGC:CRS84"} and len(out):
        bounds = out.geometry.bounds
        valid = (
            bounds.minx.ge(-180) & bounds.maxx.le(180) &
            bounds.miny.ge(-90) & bounds.maxy.le(90)
        )
        report["out_of_range_removed"] = int((~valid).sum())
        out = out.loc[valid].copy()

    if id_column and id_column in out.columns:
        # Keep the first source record deterministically and report duplicates.
        duplicate_mask = out[id_column].astype("string").duplicated(keep="first")
        report["duplicates_removed"] = int(duplicate_mask.sum())
        out = out.loc[~duplicate_mask].copy()

    report["output_features"] = int(len(out))
    report["output_crs"] = str(out.crs)
    report["invalid_after"] = int((~out.geometry.is_valid.fillna(False)).sum())
    return out.reset_index(drop=True), report


def clip_vector_to_boundary(
    source: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    *,
    id_column: str | None = None,
    target_crs: str = WEB_CRS,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Clean and clip a vector layer to the supplied authoritative boundary."""
    cleaned, report = clean_geodataframe(source, id_column=id_column, target_crs=target_crs)
    boundary_clean, boundary_report = clean_geodataframe(boundary, target_crs=target_crs)
    if boundary_clean.empty:
        raise ValueError("Boundary contains no usable geometries")
    clipped = gpd.clip(cleaned, boundary_clean)
    clipped = clipped.loc[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    report.update({
        "boundary_input_features": boundary_report["input_features"],
        "boundary_output_features": boundary_report["output_features"],
        "clipped_features": int(len(clipped)),
        "clip_boundary_crs": str(boundary_clean.crs),
    })
    return clipped.reset_index(drop=True), report


def process_vector_file(
    input_path: str | Path,
    output_path: str | Path,
    boundary_path: str | Path,
    *,
    layer: str | None = None,
    id_column: str | None = None,
) -> dict:
    """Read, clean, clip and write one vector layer. Requires the actual boundary."""
    input_path, output_path, boundary_path = Path(input_path), Path(output_path), Path(boundary_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Authoritative boundary not found: {boundary_path}")
    source = gpd.read_file(input_path, layer=layer)
    boundary = gpd.read_file(boundary_path)
    clipped, report = clip_vector_to_boundary(source, boundary, id_column=id_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(output_path, driver="GPKG" if output_path.suffix.lower() == ".gpkg" else None)
    report.update({"input_path": str(input_path), "output_path": str(output_path), "layer": layer})
    return report


def geopackage_layers(path: str | Path) -> list[str]:
    """List actual GeoPackage layers using GeoPandas/Pyogrio."""
    import pyogrio
    return [str(row[0]) for row in pyogrio.list_layers(path)]


def project_for_metric_analysis(gdf: gpd.GeoDataFrame, *, metric_crs: str = ANALYSIS_CRS) -> gpd.GeoDataFrame:
    """Project geometries to a locally appropriate metric CRS for area/distance.

    EPSG:32643 (WGS 84 / UTM zone 43N) is the working CRS for Naryn-area metric
    calculations. Do not use it for web exchange; return to EPSG:4326 for maps.
    """
    if gdf.crs is None:
        raise ValueError("Cannot project a vector layer without a declared CRS")
    return gdf.to_crs(metric_crs)


def add_area_km2(gdf: gpd.GeoDataFrame, *, area_column: str = "area_km2",
                 metric_crs: str = ANALYSIS_CRS) -> gpd.GeoDataFrame:
    """Return a copy with polygon area in km², measured in projected metres."""
    polygon_types = {"Polygon", "MultiPolygon"}
    if not set(gdf.geometry.geom_type.dropna().unique()).issubset(polygon_types):
        raise ValueError("Area calculation requires polygon or multipolygon geometries only")
    projected = project_for_metric_analysis(gdf, metric_crs=metric_crs)
    out = gdf.copy()
    out[area_column] = projected.geometry.area.to_numpy() / 1_000_000.0
    return out
