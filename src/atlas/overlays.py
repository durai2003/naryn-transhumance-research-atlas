"""Raster summaries and lightweight WGS84 PNG overlays for Folium."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
from io import BytesIO

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject

from .data_access import WORLD_COVER_LABELS

WORLDCOVER_COLORS = {
    10: (0, 100, 0), 20: (255, 187, 34), 30: (255, 255, 76),
    40: (240, 150, 255), 50: (250, 0, 0), 60: (180, 180, 180),
    70: (240, 240, 240), 80: (0, 100, 200), 90: (0, 150, 160),
    95: (0, 207, 117), 100: (250, 230, 160),
}


def worldcover_rgba(values: np.ndarray, nodata: float | int | None) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    for code, color in WORLDCOVER_COLORS.items():
        mask = values == code
        rgba[mask, :3] = color
        rgba[mask, 3] = 220
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    unknown = valid & (rgba[:, :, 3] == 0)
    rgba[unknown, :3] = (120, 120, 120)
    rgba[unknown, 3] = 180
    rgba[~valid, 3] = 0
    return rgba


def continuous_rgba(values: np.ndarray, nodata: float | int | None) -> tuple[np.ndarray, tuple[float, float] | None]:
    arr = values.astype("float32", copy=False)
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    valid &= arr >= 0
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    if not valid.any():
        return rgba, None
    low, high = np.nanpercentile(arr[valid], [2, 98])
    if high <= low:
        high = low + 1.0
    scale = np.clip((arr - low) / (high - low), 0, 1)
    # Simple blue -> cyan -> yellow -> red ramp for visual context only.
    stops = np.array([[35, 70, 180], [40, 190, 190], [245, 220, 70], [180, 35, 35]], dtype=float)
    position = scale * (len(stops) - 1)
    lower = np.floor(position).astype(int).clip(0, len(stops) - 1)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    colors = stops[lower] * (1 - fraction) + stops[upper] * fraction
    rgba[:, :, :3] = np.clip(colors, 0, 255).astype(np.uint8)
    rgba[valid, 3] = 210
    rgba[~valid, 3] = 0
    return rgba, (float(low), float(high))


def raster_overlay_image(path: str | Path, *, kind: str = "continuous", max_pixels: int = 700) -> tuple[Image.Image, list[list[float]], dict[str, Any]]:
    """Reproject a raster to a small WGS84 image suitable for Folium ImageOverlay."""
    path = Path(path)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        scale = min(1.0, max_pixels / max(dst_width, dst_height))
        out_width = max(1, int(dst_width * scale))
        out_height = max(1, int(dst_height * scale))
        dst_transform = dst_transform * Affine.scale(dst_width / out_width, dst_height / out_height)
        dtype = np.dtype(src.dtypes[0])
        if kind == "worldcover" and not np.issubdtype(dtype, np.integer):
            dtype = np.dtype("uint16")
        if src.nodata is not None:
            dest_nodata = src.nodata
        elif np.issubdtype(dtype, np.integer):
            dest_nodata = int(np.iinfo(dtype).max)
        else:
            dest_nodata = -9999.0
        dest = np.full((out_height, out_width), dest_nodata, dtype=dtype)
        reproject(
            source=rasterio.band(src, 1), destination=dest,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=dest_nodata,
            resampling=Resampling.nearest if kind == "worldcover" else Resampling.bilinear,
        )
        west_south_east_north = array_bounds(out_height, out_width, dst_transform)
        west, south, east, north = map(float, west_south_east_north)
        if kind == "worldcover":
            rgba = worldcover_rgba(dest, dest_nodata)
            display_range = None
        else:
            rgba, display_range = continuous_rgba(dest, dest_nodata)
        image = Image.fromarray(rgba, mode="RGBA")
        metadata = {
            "path": str(path), "crs_source": str(src.crs), "crs_display": "EPSG:4326",
            "source_resolution": list(src.res), "source_nodata": src.nodata,
            "display_width": out_width, "display_height": out_height,
            "display_range_percentiles": display_range,
        }
        return image, [[south, west], [north, east]], metadata


def image_to_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a PNG data URI accepted by Folium/branca.

    Passing a PIL Image instance directly to some Folium/branca versions falls
    through to JSON serialization and raises ``Image is not JSON serializable``.
    A data URI is explicit, self-contained and requires no external hosting.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image")
    buffer = BytesIO()
    image.convert("RGBA").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def raster_value_profiles(paths: list[str | Path], *, max_pixels: int = 50000) -> list[dict[str, Any]]:
    """Provide descriptive raster statistics from a bounded display sample."""
    profiles = []
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs is None:
                continue
            scale = min(1.0, (max_pixels / max(src.width * src.height, 1)) ** 0.5)
            height = max(1, int(src.height * scale))
            width = max(1, int(src.width * scale))
            arr = src.read(1, out_shape=(height, width), resampling=Resampling.nearest, masked=True)
            values = np.asarray(arr.compressed(), dtype="float64")
            values = values[np.isfinite(values)]
            if src.nodata is not None:
                values = values[values != src.nodata]
            if values.size == 0:
                continue
            profiles.append({
                "file": Path(path).name,
                "sample_cells": int(values.size),
                "minimum": float(np.min(values)),
                "p05": float(np.percentile(values, 5)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "maximum": float(np.max(values)),
                "values": values,
            })
    return profiles


def worldcover_class_summary(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Count valid WorldCover pixels exactly, using native source blocks."""
    counts: dict[int, int] = {}
    for path in paths:
        with rasterio.open(path) as src:
            if src.count < 1:
                continue
            for _, window in src.block_windows(1):
                array = src.read(1, window=window, masked=True)
                raw = np.asarray(array.compressed())
                if raw.size == 0:
                    continue
                values, n = np.unique(raw, return_counts=True)
                for value, count in zip(values, n):
                    code = int(value)
                    counts[code] = counts.get(code, 0) + int(count)
    total = sum(counts.values())
    return [
        {
            "class_code": code,
            "class_label": WORLD_COVER_LABELS.get(code, "Unmapped class"),
            "pixel_count": count,
            "percent_of_valid_pixels": count / total * 100 if total else None,
        }
        for code, count in sorted(counts.items())
    ]
