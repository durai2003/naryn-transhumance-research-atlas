"""Raster clipping, sampling and class summaries without resampling."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform


def clip_raster_to_boundary(
    input_path: str | Path,
    boundary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Clip raster to actual boundary; preserve CRS, dtype, pixel size and values.

    If the source has no nodata value, this function refuses to invent one. The
    caller must review the source-specific nodata policy before clipping.
    """
    input_path, boundary_path, output_path = Path(input_path), Path(boundary_path), Path(output_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Authoritative boundary not found: {boundary_path}")
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError("Boundary CRS is missing; refusing to guess")
    with rasterio.open(input_path) as src:
        if src.crs is None:
            raise ValueError("Raster CRS is missing; refusing to guess")
        if src.nodata is None:
            raise ValueError("Raster has no declared nodata value; review source-specific nodata before clipping")
        boundary_src = boundary.to_crs(src.crs)
        shapes = [geom.__geo_interface__ for geom in boundary_src.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            raise ValueError("Boundary has no usable geometry")
        out_image, out_transform = mask(src, shapes, crop=True, nodata=src.nodata, filled=True)
        profile = src.profile.copy()
        profile.update(height=out_image.shape[1], width=out_image.shape[2], transform=out_transform)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out_image)
            # Preserve dataset tags and band tags where available.
            dst.update_tags(**src.tags())
            for band in range(1, src.count + 1):
                dst.update_tags(band, **src.tags(band))
        return {
            "input_path": str(input_path), "output_path": str(output_path),
            "crs": str(src.crs), "width_before": src.width, "height_before": src.height,
            "width_after": int(out_image.shape[2]), "height_after": int(out_image.shape[1]),
            "resolution_before": list(src.res), "resolution_after": list(src.res),
            "dtype": list(src.dtypes), "nodata": src.nodata,
            "resampling": "none; source grid and pixel values retained",
        }


def sample_raster_at_points(
    raster_path: str | Path,
    points: gpd.GeoDataFrame,
    *,
    id_column: str = "location_id",
    value_column: str = "raster_value",
) -> pd.DataFrame:
    """Sample raster cells at point locations; coordinates transformed to raster CRS."""
    if points.crs is None:
        raise ValueError("Point layer CRS missing")
    if id_column not in points.columns:
        raise ValueError(f"Point layer missing key column {id_column!r}")
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Raster CRS missing")
        xs = points.geometry.x.astype(float).tolist()
        ys = points.geometry.y.astype(float).tolist()
        if points.crs != src.crs:
            xs, ys = transform(points.crs, src.crs, xs, ys)
        coords = list(zip(xs, ys))
        samples = list(src.sample(coords, indexes=1, masked=True))
        vals: list[Any] = []
        for sample in samples:
            value = sample[0]
            vals.append(None if np.ma.is_masked(value) or (src.nodata is not None and value == src.nodata) else float(value))
    return pd.DataFrame({id_column: points[id_column].astype("string").tolist(), value_column: vals})


def raster_class_percentages(
    raster_path: str | Path,
    boundary_path: str | Path,
    *,
    class_labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Calculate pixel share by class inside a boundary using source pixel grid."""
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError("Boundary CRS missing")
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Raster CRS missing")
        if src.nodata is None:
            raise ValueError("Raster nodata is not declared")
        boundary = boundary.to_crs(src.crs)
        shapes = [g.__geo_interface__ for g in boundary.geometry if g is not None and not g.is_empty]
        clipped, _ = mask(src, shapes, crop=True, nodata=src.nodata, filled=True)
        band = clipped[0]
        valid = band != src.nodata
        values, counts = np.unique(band[valid], return_counts=True)
        total = int(counts.sum())
        labels = class_labels or {}
        return pd.DataFrame([
            {"class_code": int(value), "class_label": labels.get(int(value), "unmapped_class"),
             "pixel_count": int(count), "percent_of_valid_pixels": (float(count) / total * 100 if total else None)}
            for value, count in zip(values, counts)
        ])
