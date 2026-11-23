"""Create clearly synthetic atlas demo records after an authoritative boundary exists.

This script creates schematic lines only to demonstrate layer styling/filtering.
It creates no field observations, participant identities, interview quotations,
or claims about actual seasonal migration behaviour.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union

from src.atlas.data_access import find_boundary_path

DISCLAIMER = (
    "Synthetic educational demonstration only. This geometry/metadata is fictional, "
    "is not a field observation, is not a verified transhumance route, and must not "
    "be interpreted as evidence of actual herder movement, pasture access, or land rights."
)
SEASONS = ["summer", "winter", "transition", "summer", "winter", "transition", "summer", "winter"]
TOPICS = [
    "seasonal mobility metadata", "pasture terminology metadata", "water availability metadata",
    "climate adaptation metadata", "livestock management metadata", "vegetation monitoring metadata",
]


def _line_parts(geometry: Any) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString" or hasattr(geometry, "geoms"):
        parts: list[LineString] = []
        for part in geometry.geoms:
            parts.extend(_line_parts(part))
        return parts
    return []


def build_synthetic_routes(boundary_path: Path, *, count: int = 8) -> gpd.GeoDataFrame:
    """Create deterministic radial schematic lines clipped to actual Naryn ADM1.

    The real boundary only constrains the demonstration geometry. It does not
    validate route locations or seasonal use. Output is returned in EPSG:4326.
    """
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError("Authoritative boundary has no CRS; refusing to guess")
    boundary = boundary.loc[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise ValueError("Authoritative boundary has no usable geometry")
    projected = boundary.to_crs("EPSG:32643")
    region = unary_union([geom if geom.is_valid else geom.buffer(0) for geom in projected.geometry])
    if region.is_empty:
        raise ValueError("Boundary union is empty")
    anchor = region.representative_point()
    minx, miny, maxx, maxy = region.bounds
    reach = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5 * 1.5
    if reach <= 0:
        raise ValueError("Boundary has zero spatial extent")

    rows = []
    # Orientations are evenly spaced across 180 degrees to avoid duplicate
    # opposite-direction transects. These are schematic lines, not routes.
    import math
    for index in range(count):
        angle = math.pi * index / count
        dx, dy = math.cos(angle), math.sin(angle)
        candidate = LineString([
            (anchor.x - dx * reach, anchor.y - dy * reach),
            (anchor.x + dx * reach, anchor.y + dy * reach),
        ])
        clipped = region.intersection(candidate)
        parts = _line_parts(clipped)
        if not parts:
            continue
        # Select the longest clipped segment for deterministic, simple output.
        line = max(parts, key=lambda item: item.length)
        if line.length < 100:
            continue
        rows.append({
            "route_id": f"SYN-ROUTE-{len(rows) + 1:03d}",
            "name": f"Synthetic schematic route {len(rows) + 1:02d}",
            "season": SEASONS[index % len(SEASONS)],
            "season_label_type": "interface_demo_filter_only",
            "location_id": "naryn-region",
            "is_synthetic": True,
            "provenance": "synthetic_demo",
            "synthetic_disclaimer": DISCLAIMER,
            "geometry": line,
        })
    if not rows:
        raise ValueError("Could not generate any schematic lines within the supplied boundary")
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32643").to_crs("EPSG:4326")
    return frame


def build_interviews(count: int = 12) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append({
            "interview_id": f"SYN-INT-{index + 1:03d}",
            "title": f"Fictional interview metadata example {index + 1:02d}",
            "description": "Metadata-only fictional record for demonstrating catalogue filters; no transcript, quotation, participant identity, or interview evidence exists.",
            "topic": TOPICS[index % len(TOPICS)],
            "location_id": "naryn-region",
            "content_available": False,
            "participant_identity_available": False,
            "is_synthetic": True,
            "provenance": "synthetic_demo",
            "synthetic_disclaimer": DISCLAIMER,
        })
    return pd.DataFrame(rows)


def build_multimedia(count: int = 3) -> pd.DataFrame:
    media_types = ["photo", "audio", "video"]
    rows = []
    for index in range(count):
        rows.append({
            "asset_id": f"SYN-MEDIA-{index + 1:03d}",
            "title": f"Fictional multimedia metadata example {index + 1:02d}",
            "asset_type": media_types[index % len(media_types)],
            "description": "Placeholder metadata only. No media file is supplied or represented as a real field asset.",
            "location_id": "naryn-region",
            "asset_available": False,
            "asset_path": "",
            "is_synthetic": True,
            "provenance": "synthetic_demo",
            "synthetic_disclaimer": DISCLAIMER,
        })
    return pd.DataFrame(rows)


def prepare_demo_data(root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    boundary_path = find_boundary_path(root)
    if boundary_path is None:
        raise FileNotFoundError(
            "Actual Naryn boundary not found. Run Stage 2 recovery and Stage 3 first; "
            "no synthetic geometry was generated."
        )
    output_dir = root / "data/synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "routes": output_dir / "routes.geojson",
        "interviews": output_dir / "interviews.csv",
        "multimedia": output_dir / "multimedia.csv",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Synthetic demo files already exist. Review them, or rerun with --overwrite. "
            + ", ".join(str(path.relative_to(root)) for path in existing)
        )
    routes = build_synthetic_routes(boundary_path, count=8)
    interviews = build_interviews(12)
    multimedia = build_multimedia(3)
    routes.to_file(outputs["routes"], driver="GeoJSON")
    interviews.to_csv(outputs["interviews"], index=False)
    multimedia.to_csv(outputs["multimedia"], index=False)
    return {
        "boundary_source": boundary_path.relative_to(root).as_posix(),
        "synthetic_disclaimer": DISCLAIMER,
        "outputs": {key: value.relative_to(root).as_posix() for key, value in outputs.items()},
        "counts": {"routes": len(routes), "interviews": len(interviews), "multimedia": len(multimedia)},
        "all_synthetic_flags_valid": bool(routes["is_synthetic"].all() and interviews["is_synthetic"].all() and multimedia["is_synthetic"].all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare_demo_data(args.root, overwrite=args.overwrite)
    print(pd.Series(report["counts"]).to_string())
    print(f"Synthetic demo files created under: {args.root.resolve() / 'data/synthetic'}")
    print("DISCLAIMER: all generated records are fictional demonstration assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
