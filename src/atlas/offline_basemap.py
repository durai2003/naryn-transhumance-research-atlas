"""Offline, deterministic Naryn relief basemap for embedded Cesium views."""
from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.atlas.terrain import export_local_dem_grid, find_local_dem


def _boundary_path(root: Path) -> Path | None:
    for path in (
        root / "data/processed/vectors/naryn_adm1_clean.geojson",
        root / "data/raw/boundaries/naryn_adm1.geojson",
    ):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def build_offline_basemap(root: str | Path) -> dict[str, Any] | None:
    """Build a local shaded-relief PNG and geographic rectangle for Cesium."""
    root = Path(root).resolve()
    boundary_path = _boundary_path(root)
    if boundary_path is None:
        return None
    try:
        boundary_mtime = boundary_path.stat().st_mtime_ns
        status = find_local_dem(root)
        dem_mtime = (
            status.dem_path.stat().st_mtime_ns
            if status.has_local_dem and status.dem_path is not None
            else boundary_mtime
        )
    except OSError:
        return None

    return _build_offline_basemap_cached(
        str(root), str(boundary_path), boundary_mtime, dem_mtime, 640, 480
    )


@lru_cache(maxsize=4)
def _build_offline_basemap_cached(
    root_string: str,
    boundary_path_string: str,
    boundary_mtime: int,
    dem_mtime: int,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    """Cache the base image so Streamlit reruns don't redraw it."""
    del boundary_mtime, dem_mtime
    try:
        grid = export_local_dem_grid(Path(root_string), width=width, height=height)
        if not grid:
            return None

        raw = base64.b64decode(grid["data_b64_u16"])
        encoded = np.frombuffer(raw, dtype="<u2").reshape(height, width)
        elevation = encoded.astype(np.float32) * float(grid["scale_m"]) + float(grid["offset_m"])
        valid = encoded > 0
        west, south, east, north = [float(v) for v in grid["bounds"]]
        if east <= west or north <= south:
            return None

        stops = np.asarray(
            [1600, 1850, 2100, 2400, 2700, 3000, 3350, 3700, 4100, 4500, 4850],
            dtype=np.float32,
        )
        colours = np.asarray(
            [
                [104, 132, 105], [130, 151, 112], [159, 165, 119],
                [188, 178, 132], [199, 180, 139], [185, 165, 135],
                [166, 151, 137], [158, 157, 153], [184, 184, 177],
                [218, 218, 207], [241, 241, 232],
            ],
            dtype=np.float32,
        )
        rgb = np.empty((height, width, 3), dtype=np.float32)
        for channel in range(3):
            rgb[:, :, channel] = np.interp(elevation, stops, colours[:, channel])

        mean_lat = (north + south) * 0.5
        dx_m = max((east - west) * 111320.0 * np.cos(np.radians(mean_lat)) / max(width - 1, 1), 1.0)
        dy_m = max((north - south) * 110540.0 / max(height - 1, 1), 1.0)
        dz_dy, dz_dx = np.gradient(elevation, dy_m, dx_m)
        slope = np.arctan(np.hypot(dz_dx, dz_dy))
        aspect = np.arctan2(-dz_dx, dz_dy)
        azimuth, altitude = np.radians(315.0), np.radians(45.0)
        illumination = (
            np.sin(altitude) * np.cos(slope)
            + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
        )
        rgb *= (0.72 + 0.32 * np.clip(illumination, 0.0, 1.0))[:, :, None]
        rgb[~valid] = np.asarray([190, 202, 208], dtype=np.float32)
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)

        boundary = json.loads(Path(boundary_path_string).read_text(encoding="utf-8"))
        features = boundary.get("features", []) if isinstance(boundary, dict) else []

        def to_pixel(lon: float, lat: float) -> tuple[int, int]:
            px = int(round((lon - west) / (east - west) * (width - 1)))
            py = int(round((north - lat) / (north - south) * (height - 1)))
            return px, py

        for feature in features:
            geometry = feature.get("geometry") or {}
            kind = geometry.get("type")
            coordinates = geometry.get("coordinates") or []
            rings: list[list[list[float]]] = []
            if kind == "Polygon":
                rings = coordinates[:1]
            elif kind == "MultiPolygon":
                rings = [polygon[0] for polygon in coordinates if polygon]
            for ring in rings:
                points = [to_pixel(float(p[0]), float(p[1])) for p in ring if len(p) >= 2]
                if len(points) >= 3:
                    draw.line(points, fill=(39, 121, 154), width=3, joint="curve")
                    draw.line(points, fill=(225, 242, 244), width=1, joint="curve")

        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        png_b64 = base64.b64encode(output.getvalue()).decode("ascii")
        return {
            "url": "data:image/png;base64," + png_b64,
            "bounds": [west, south, east, north],
            "width": width,
            "height": height,
            "credit": str(grid.get("source") or "Naryn analytical relief proxy"),
        }
    except Exception:
        return None
