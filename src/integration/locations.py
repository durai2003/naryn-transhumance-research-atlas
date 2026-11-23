"""Build verified location keys and integrate observations with geospatial context."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.geospatial.raster import sample_raster_at_points


def locations_from_verified_wikidata(path: str | Path) -> pd.DataFrame:
    """Create location rows only from coordinate-bearing verified entity records."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entity in document.get("entities", []):
        lat, lon = entity.get("latitude"), entity.get("longitude")
        if lat is None or lon is None:
            continue
        rows.append({
            "location_id": f"wikidata:{entity['qid']}",
            "location_name": entity["label"],
            "latitude": float(lat),
            "longitude": float(lon),
            "external_identifier": entity["qid"],
            "external_uri": entity["uri"],
            "coordinate_provenance": "verified_wikidata_entity_page_coordinate_statement",
            "coordinate_precision_note": "entity-page coordinate; not an administrative boundary or weather station",
            "is_synthetic": False,
            "provenance": "wikidata_entity_page",
            "source_url": entity["page"],
        })
    return pd.DataFrame(rows, columns=[
        "location_id", "location_name", "latitude", "longitude", "external_identifier", "external_uri",
        "coordinate_provenance", "coordinate_precision_note", "is_synthetic", "provenance", "source_url",
    ])


def locations_to_geodataframe(locations: pd.DataFrame) -> gpd.GeoDataFrame:
    required = {"location_id", "latitude", "longitude"}
    missing = required - set(locations.columns)
    if missing:
        raise ValueError(f"Location table missing columns: {sorted(missing)}")
    return gpd.GeoDataFrame(
        locations.copy(),
        geometry=[Point(float(lon), float(lat)) for lat, lon in zip(locations.latitude, locations.longitude)],
        crs="EPSG:4326",
    )


def integrate_location_context(
    locations: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    *,
    sheep_raster: str | Path | None = None,
    goat_raster: str | Path | None = None,
    worldcover_raster: str | Path | None = None,
) -> pd.DataFrame:
    """Join climate summaries by stable location ID and sample optional rasters.

    A missing raster stays null; this routine does not interpolate or substitute.
    """
    if locations.empty:
        return pd.DataFrame(columns=["location_id", "location_name"])
    out = locations.copy()
    if not climate_monthly.empty:
        climate = climate_monthly.copy()
        climate["year_month"] = (
            climate["year"].astype("string") + "-" + climate["month"].astype("string").str.zfill(2)
        )
        # Keep one row per location and month: this remains a long-format table.
        out = out.merge(climate, on="location_id", how="left", validate="one_to_many")
    else:
        out["year"] = pd.NA
        out["month"] = pd.NA
        out["t2m_mean_c"] = pd.NA
        out["precipitation_sum_mm"] = pd.NA
        out["daily_records"] = pd.NA

    points = locations_to_geodataframe(locations)
    for raster_path, column in [
        (sheep_raster, "sheep_density_heads_km2"),
        (goat_raster, "goat_density_heads_km2"),
        (worldcover_raster, "worldcover_class_code"),
    ]:
        if raster_path and Path(raster_path).exists():
            samples = sample_raster_at_points(raster_path, points, value_column=column)
            out = out.merge(samples, on="location_id", how="left", validate="many_to_one")
        else:
            out[column] = pd.NA
    return out
