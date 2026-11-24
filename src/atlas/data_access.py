"""Safe data access for the upgraded 3D Spatiotemporal Transhumance Atlas.

Preserves all Stage 1-6 loaders while adding support for:
- Local Copernicus DEM metadata and terrain status
- Reproducible synthetic herds, candidate pastures, least-cost corridors,
  time-indexed trajectories, movement events, scenario climate, and suitability
- Properly schema-aware Geofabrik OSM layers (`fclass`, `code`, `name`, `population`)
  categorised into roads/tracks, waterways/waterbodies, and settlements.

Missing inputs stay missing: this module never fabricates historical climate,
livestock density, or verified grazing parcels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import get_coordinates

from src.atlas.terrain import TerrainStatus, find_local_dem

SYNTHETIC_REQUIRED = {"is_synthetic", "provenance"}
WORLD_COVER_LABELS = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}


@dataclass
class AtlasData:
    root: Path
    locations: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_climate: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_climate: pd.DataFrame = field(default_factory=pd.DataFrame)
    seasonal_climate: pd.DataFrame = field(default_factory=pd.DataFrame)
    location_context: pd.DataFrame = field(default_factory=pd.DataFrame)
    boundary: gpd.GeoDataFrame | None = None
    verified_pastures: gpd.GeoDataFrame | None = None
    synthetic_routes: gpd.GeoDataFrame | None = None
    synthetic_interviews: pd.DataFrame = field(default_factory=pd.DataFrame)
    synthetic_multimedia: pd.DataFrame = field(default_factory=pd.DataFrame)
    # New Stage 7+ 3D Spatiotemporal Simulation datasets
    synthetic_pastures: gpd.GeoDataFrame | None = None
    migration_corridors: gpd.GeoDataFrame | None = None
    synthetic_herds: pd.DataFrame = field(default_factory=pd.DataFrame)
    herd_trajectories: pd.DataFrame = field(default_factory=pd.DataFrame)
    movement_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenario_climate: pd.DataFrame = field(default_factory=pd.DataFrame)
    suitability_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    terrain_status: TerrainStatus | None = None
    # Context layers
    osm_layers: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    osm_roads: gpd.GeoDataFrame | None = None
    osm_waterways: gpd.GeoDataFrame | None = None
    osm_waterbodies: gpd.GeoDataFrame | None = None
    osm_settlements: gpd.GeoDataFrame | None = None
    worldcover_rasters: list[Path] = field(default_factory=list)
    livestock_rasters: list[Path] = field(default_factory=list)
    kdc_rasters: list[Path] = field(default_factory=list)
    kdc_wms_layers: list[str] = field(default_factory=list)
    dataset_register: pd.DataFrame = field(default_factory=pd.DataFrame)
    graph_path: Path | None = None
    graph_report: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


_FOLIUM_COORDINATE_GEOMETRIES = {
    "Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"
}


def folium_compatible_geodataframe(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Recursively split GeometryCollections and drop null/empty/non-finite geometries."""
    if not isinstance(frame, gpd.GeoDataFrame):
        raise TypeError("Expected a GeoDataFrame")
    if frame.crs is None:
        raise ValueError("GeoJSON map layer has no CRS; refusing to guess it")

    geometry_column = frame.geometry.name
    records: list[dict[str, Any]] = []

    def components(geometry):
        if geometry is None or geometry.is_empty:
            return
        if geometry.geom_type == "GeometryCollection":
            for child in geometry.geoms:
                yield from components(child)
            return
        if geometry.geom_type not in _FOLIUM_COORDINATE_GEOMETRIES:
            return
        try:
            coordinates = get_coordinates(geometry)
        except Exception:
            return
        if coordinates.size == 0 or not np.isfinite(coordinates).all():
            return
        yield geometry

    for _, row in frame.iterrows():
        geometry = row[geometry_column]
        for component in components(geometry):
            record = row.to_dict()
            record[geometry_column] = component
            records.append(record)

    if not records:
        return frame.iloc[0:0].copy().reset_index(drop=True)
    return gpd.GeoDataFrame(records, geometry=geometry_column, crs=frame.crs).reset_index(drop=True)


def _read_tabular(csv_path: Path, parquet_path: Path | None = None) -> pd.DataFrame:
    if parquet_path is not None and parquet_path.exists() and parquet_path.stat().st_size > 0:
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
        return pd.DataFrame()


def _is_true(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _synthetic_rows_are_valid(frame: pd.DataFrame, *, require_disclaimer: bool = False) -> pd.Series:
    req = {"is_synthetic", "provenance", "synthetic_disclaimer"} if require_disclaimer else SYNTHETIC_REQUIRED
    if frame.empty or not req.issubset(frame.columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    is_syn = frame["is_synthetic"].map(_is_true)
    prov = frame["provenance"].fillna("").astype(str).eq("synthetic_demo")
    if require_disclaimer:
        disc = frame["synthetic_disclaimer"].fillna("").astype(str).str.strip().ne("")
        return is_syn & prov & disc
    return is_syn & prov



def read_geojson_safe(path: Path) -> gpd.GeoDataFrame:
    """Read a GeoJSON file via GeoPandas, falling back to json+from_features if fiona is broken."""
    try:
        return gpd.read_file(path)
    except Exception:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        crs = "EPSG:4326"
        if isinstance(doc.get("crs"), dict):
            name = doc["crs"].get("properties", {}).get("name", "")
            if "32643" in str(name):
                crs = "EPSG:32643"
        return gpd.GeoDataFrame.from_features(doc.get("features", []), crs=crs)


def write_geojson_safe(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write GeoJSON cleanly without depending on fiona."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gdf.to_crs("EPSG:4326").to_json(), encoding="utf-8")


def find_boundary_path(root: Path) -> Path | None:
    candidates = [
        root / "data/processed/vectors/naryn_adm1_clean.geojson",
        root / "data/raw/boundaries/naryn_adm1.geojson",
    ]
    return next((p for p in candidates if p.exists() and p.is_file()), None)


def load_boundary(root: Path) -> tuple[gpd.GeoDataFrame | None, str | None]:
    path = find_boundary_path(root)
    if path is None:
        return None, "Naryn administrative boundary is not available locally."
    try:
        frame = read_geojson_safe(path)
        if frame.crs is None:
            return None, f"Boundary has no declared CRS: {path.name}"
        frame = folium_compatible_geodataframe(frame)
        if frame.empty:
            return None, f"Boundary file contains no usable geometries: {path.name}"
        return frame.to_crs("EPSG:4326"), None
    except Exception as exc:
        return None, f"Could not read Naryn boundary ({path.name}): {exc}"


def load_verified_pastures(root: Path) -> tuple[gpd.GeoDataFrame | None, str | None]:
    path = root / "data/processed/pastures.geojson"
    if not path.exists():
        return None, "No independently sourced pasture polygons are available. WorldCover classes are not pasture parcels."
    try:
        frame = read_geojson_safe(path)
        if frame.crs is None or "source_url" not in frame.columns:
            return None, "Pasture features lack CRS or source_url provenance."
        if "is_synthetic" in frame.columns:
            frame = frame.loc[~frame["is_synthetic"].map(_is_true)].copy()
        frame = frame.loc[frame["source_url"].fillna("").astype(str).str.strip().ne("")].copy()
        frame = folium_compatible_geodataframe(frame)
        return (frame.to_crs("EPSG:4326"), None) if not frame.empty else (None, "No verified pasture features passed checks.")
    except Exception as exc:
        return None, f"Could not load sourced pasture features: {exc}"


def _load_synthetic_geojson(path: Path, label: str, *, require_disclaimer: bool = True) -> tuple[gpd.GeoDataFrame | None, str | None]:
    if not path.exists():
        return None, f"{label} ({path.name}) not found."
    try:
        frame = read_geojson_safe(path)
        if frame.crs is None:
            return None, f"{label} has no declared CRS."
        mask = _synthetic_rows_are_valid(frame, require_disclaimer=require_disclaimer)
        frame = folium_compatible_geodataframe(frame.loc[mask].copy())
        if frame.empty:
            return None, f"No features in {path.name} passed synthetic provenance validation."
        return frame.to_crs("EPSG:4326"), None
    except Exception as exc:
        return None, f"Could not read {label}: {exc}"


def load_synthetic_routes(root: Path) -> tuple[gpd.GeoDataFrame | None, str | None]:
    return _load_synthetic_geojson(root / "data/synthetic/routes.geojson", "Synthetic schematic routes", require_disclaimer=True)


def load_synthetic_metadata(root: Path, filename: str, id_column: str) -> tuple[pd.DataFrame, str | None]:
    path = root / "data/synthetic" / filename
    frame = _read_tabular(path)
    if frame.empty:
        return frame, f"{filename} is not available."
    if id_column not in frame.columns:
        return pd.DataFrame(), f"{filename} is missing {id_column}."
    frame = frame.loc[_synthetic_rows_are_valid(frame, require_disclaimer=True)].copy()
    return frame, None if not frame.empty else f"No rows in {filename} passed synthetic provenance validation."


def _load_kdc_wms_layers(root: Path) -> list[str]:
    import xml.etree.ElementTree as ET
    kdc_dir = root / "data/raw/kdc"
    candidates = sorted(kdc_dir.glob("*wms*GetCapabilities*.xml")) if kdc_dir.exists() else []
    if not candidates:
        return []
    try:
        tree = ET.parse(candidates[0])
        result = []
        for element in tree.iter():
            if element.tag.rsplit("}", 1)[-1] == "Layer":
                name = next((c.text for c in list(element) if c.tag.rsplit("}", 1)[-1] == "Name" and c.text), None)
                if name:
                    result.append(name.strip())
        return sorted(set(result))
    except (ET.ParseError, OSError):
        return []


def _load_osm_structured(root: Path) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, gpd.GeoDataFrame | None], list[str]]:
    """Load Naryn-clipped Geofabrik OSM GeoPackage preserving real `fclass` attributes."""
    path = root / "data/processed/vectors/osm_naryn_context.gpkg"
    structured: dict[str, gpd.GeoDataFrame | None] = {
        "roads": None, "waterways": None, "waterbodies": None, "settlements": None
    }
    if not path.exists():
        return {}, structured, ["Clipped OSM GeoPackage not found; countrywide raw extract is never loaded into the browser."]

    warnings: list[str] = []
    layers_dict: dict[str, gpd.GeoDataFrame] = {}
    try:
        import pyogrio
        available = {str(row[0]) for row in pyogrio.list_layers(path)}
    except Exception as exc:
        return {}, structured, [f"Could not list OSM GeoPackage layers: {exc}"]

    # Priority transport/hydrology/settlement layers for the 3D globe (compact, simplified for smooth WebGL rendering)
    priority_specs = [
        ("gis_osm_roads_free", "roads", 4000, ["osm_id", "fclass", "name", "ref"], 0.0004),
        ("gis_osm_waterways_free", "waterways", 2500, ["osm_id", "fclass", "name", "width"], 0.0005),
        ("gis_osm_water_a_free", "waterbodies", 600, ["osm_id", "fclass", "name"], 0.0005),
        ("gis_osm_places_free", "settlements", 300, ["osm_id", "fclass", "name", "population"], None),
    ]
    for layer_name, key, limit, keep_cols, tol in priority_specs:
        if layer_name not in available:
            continue
        try:
            gdf = gpd.read_file(path, layer=layer_name)
            if gdf.empty or gdf.crs is None:
                continue
            gdf = gdf.to_crs("EPSG:4326")
            # Prioritise major roads/tracks and named waterways so truncation keeps the most informative features
            if key == "roads" and "fclass" in gdf.columns:
                road_priority = {
                    "trunk": 1, "primary": 1, "secondary": 2, "tertiary": 3,
                    "unclassified": 4, "residential": 5, "track": 5,
                    "track_grade1": 5, "track_grade2": 5, "track_grade3": 6,
                    "path": 6, "bridleway": 6, "footway": 7,
                }
                gdf["_prio"] = gdf["fclass"].astype(str).str.lower().map(road_priority).fillna(9)
                gdf = gdf.sort_values("_prio").head(limit).drop(columns=["_prio"])
            elif key == "waterways" and "fclass" in gdf.columns:
                w_prio = {"river": 1, "canal": 2, "stream": 3}
                gdf["_prio"] = gdf["fclass"].astype(str).str.lower().map(w_prio).fillna(5)
                gdf = gdf.sort_values("_prio").head(limit).drop(columns=["_prio"])
            elif key == "waterbodies":
                # Keep largest water polygons (Son-Kul, Chatyr-Kul, reservoirs)
                gdf["_area"] = gdf.geometry.area
                gdf = gdf.sort_values("_area", ascending=False).head(limit).drop(columns=["_area"])
            else:
                gdf = gdf.head(limit)

            cols = [c for c in keep_cols if c in gdf.columns] + [gdf.geometry.name]
            gdf = gdf[cols].copy()
            if tol is not None:
                gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
            gdf = folium_compatible_geodataframe(gdf)
            if not gdf.empty:
                structured[key] = gdf
                layers_dict[layer_name] = gdf
        except Exception as exc:
            warnings.append(f"OSM layer {layer_name} skipped: {exc}")

    return layers_dict, structured, warnings


def _raster_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})


def load_graph_report(root: Path) -> tuple[Path | None, dict[str, Any]]:
    report_path = root / "reports/stage4_linked_data_report.json"
    graph_candidates = [root / "linked_data/research_graph.ttl", root / "linked_data/research_graph.jsonld"]
    graph_path = next((p for p in graph_candidates if p.exists() and p.stat().st_size > 0), None)
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            report = {}
    return graph_path, report


def load_dataset_register(root: Path) -> pd.DataFrame:
    register = _read_tabular(root / "metadata/dataset_register.csv")
    if register.empty:
        return register
    local_path_map = {
        "GB-ADM1-KGZ": [root / "data/raw/boundaries/naryn_adm1.geojson", root / "data/processed/vectors/naryn_adm1_clean.geojson"],
        "KDC-PASTURE-SNOW-DROUGHT": (list((root / "data/raw/kdc").glob("*")) + _raster_files(root / "data/processed/rasters/kdc")) if (root / "data/raw/kdc").exists() else _raster_files(root / "data/processed/rasters/kdc"),
        "FAO-GLW4-2020-DENSITY": _raster_files(root / "data/processed/rasters/glw4") or _raster_files(root / "data/raw/glw4"),
        "ESA-WORLDCOVER-2021-V200": _raster_files(root / "data/processed/rasters/worldcover") or _raster_files(root / "data/raw/worldcover"),
        "NASA-POWER-DAILY": [root / "data/raw/nasa_power/naryn_city_2015_2025_daily.json", root / "data/processed/climate/nasa_power_daily.csv"],
        "OSM-KGZ-GEOFABRIK": [root / "data/processed/vectors/osm_naryn_context.gpkg", root / "data/raw/osm/extracted/kyrgyzstan.gpkg"],
        "WIKIDATA-NARYN-ENTITIES": [root / "data/raw/wikidata/verified_entity_references.json"],
    }
    statuses, local_assets = [], []
    for dataset_id in register.get("dataset_id", pd.Series(dtype=str)).fillna("").astype(str):
        candidates = local_path_map.get(dataset_id, [])
        existing = [p for p in candidates if p.exists() and p.is_file() and p.stat().st_size > 0]
        if dataset_id == "NASA-POWER-DAILY":
            existing = [p for p in existing if p.suffix.lower() != ".csv" or not _read_tabular(p).empty]
        statuses.append("available" if existing else "not available locally")
        local_assets.append("; ".join(p.relative_to(root).as_posix() for p in existing))
    register = register.copy()
    register["local_status"] = statuses
    register["local_assets"] = local_assets
    return register


def load_atlas_data(root: str | Path, *, max_osm_features_per_layer: int = 2500) -> AtlasData:
    root = Path(root).resolve()
    data = AtlasData(root=root)
    data.locations = _read_tabular(root / "data/processed/locations.csv")
    data.daily_climate = _read_tabular(root / "data/processed/climate/nasa_power_daily.csv")
    data.monthly_climate = _read_tabular(root / "data/processed/climate/nasa_power_monthly.csv")
    data.seasonal_climate = _read_tabular(root / "data/processed/climate/nasa_power_seasonal.csv")
    data.location_context = _read_tabular(root / "data/processed/location_context.csv")
    data.terrain_status = find_local_dem(root)

    data.boundary, w = load_boundary(root)
    if w: data.warnings.append(w)
    data.verified_pastures, w = load_verified_pastures(root)
    if w: data.warnings.append(w)
    data.synthetic_routes, w = load_synthetic_routes(root)
    if w: data.warnings.append(w)
    data.synthetic_interviews, w = load_synthetic_metadata(root, "interviews.csv", "interview_id")
    if w: data.warnings.append(w)
    data.synthetic_multimedia, w = load_synthetic_metadata(root, "multimedia.csv", "asset_id")
    if w: data.warnings.append(w)

    # New Stage 7+ simulation datasets
    data.synthetic_pastures, w = _load_synthetic_geojson(root / "data/synthetic/pastures.geojson", "Synthetic candidate pastures")
    if w: data.warnings.append(w)
    data.migration_corridors, w = _load_synthetic_geojson(root / "data/synthetic/migration_routes.geojson", "Synthetic migration corridors")
    if w: data.warnings.append(w)
    data.synthetic_herds = _read_tabular(root / "data/synthetic/herds.csv")
    data.herd_trajectories = _read_tabular(root / "data/synthetic/herd_trajectories.csv", root / "data/synthetic/herd_trajectories.parquet")
    data.movement_events = _read_tabular(root / "data/synthetic/movement_events.csv")
    data.scenario_climate = _read_tabular(root / "data/synthetic/scenario_climate.csv", root / "data/synthetic/scenario_climate.parquet")
    data.suitability_scores = _read_tabular(root / "data/synthetic/suitability_scores.csv")

    data.osm_layers, structured, osm_warnings = _load_osm_structured(root)
    data.osm_roads = structured["roads"]
    data.osm_waterways = structured["waterways"]
    data.osm_waterbodies = structured["waterbodies"]
    data.osm_settlements = structured["settlements"]
    data.warnings.extend(osm_warnings)

    data.worldcover_rasters = _raster_files(root / "data/processed/rasters/worldcover")
    data.livestock_rasters = _raster_files(root / "data/processed/rasters/glw4")
    data.kdc_rasters = _raster_files(root / "data/processed/rasters/kdc")
    data.kdc_wms_layers = _load_kdc_wms_layers(root)
    data.dataset_register = load_dataset_register(root)
    data.graph_path, data.graph_report = load_graph_report(root)
    return data


def map_center(data: AtlasData) -> tuple[float, float]:
    if data.boundary is not None and not data.boundary.empty:
        try:
            proj = data.boundary.to_crs("EPSG:32643").geometry
            union = proj.union_all() if hasattr(proj, "union_all") else proj.unary_union
            rep = union.representative_point()
            pt = gpd.GeoSeries([rep], crs="EPSG:32643").to_crs("EPSG:4326").iloc[0]
            return float(pt.y), float(pt.x)
        except Exception:
            pass
    if not data.locations.empty and {"latitude", "longitude"}.issubset(data.locations.columns):
        valid = data.locations.dropna(subset=["latitude", "longitude"])
        if not valid.empty:
            return float(valid.iloc[0]["latitude"]), float(valid.iloc[0]["longitude"])
    return 41.4333333, 76.0


def available_climate_locations(data: AtlasData) -> list[str]:
    if data.daily_climate.empty or "location_id" not in data.daily_climate.columns:
        return data.locations["location_id"].dropna().astype(str).unique().tolist() if "location_id" in data.locations.columns else []
    return data.daily_climate["location_id"].dropna().astype(str).unique().tolist()
