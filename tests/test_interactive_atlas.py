from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from scripts.prepare_atlas_demo_data import prepare_demo_data, build_synthetic_routes
from src.atlas.data_access import (
    load_atlas_data,
    load_synthetic_routes,
    load_synthetic_metadata,
    map_center,
)
from src.atlas.overlays import worldcover_rgba, worldcover_class_summary


def _write_boundary(root: Path) -> Path:
    path = root / "data/raw/boundaries/naryn_adm1.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    boundary = gpd.GeoDataFrame(
        {"shapeName": ["Naryn Region"]},
        geometry=[Polygon([(75.0, 41.0), (77.0, 41.0), (77.0, 42.0), (75.0, 42.0)])],
        crs="EPSG:4326",
    )
    boundary.to_file(path, driver="GeoJSON")
    return path


def test_synthetic_demo_generation_requires_real_boundary(tmp_path: Path):
    try:
        prepare_demo_data(tmp_path)
    except FileNotFoundError as exc:
        assert "Actual Naryn boundary not found" in str(exc)
    else:
        raise AssertionError("Expected no synthetic route geometry without the real boundary")


def test_synthetic_demo_generation_is_separated_and_fully_flagged(tmp_path: Path):
    _write_boundary(tmp_path)
    report = prepare_demo_data(tmp_path)
    assert report["counts"] == {"routes": 8, "interviews": 12, "multimedia": 3}
    routes = gpd.read_file(tmp_path / "data/synthetic/routes.geojson")
    assert routes.crs.to_epsg() == 4326
    assert routes["is_synthetic"].astype(bool).all()
    assert routes["provenance"].eq("synthetic_demo").all()
    assert routes["synthetic_disclaimer"].str.len().gt(0).all()
    assert routes.geometry.geom_type.eq("LineString").all()
    interviews = pd.read_csv(tmp_path / "data/synthetic/interviews.csv")
    assert len(interviews) == 12
    assert not interviews["content_available"].any()
    assert interviews["is_synthetic"].all()
    media = pd.read_csv(tmp_path / "data/synthetic/multimedia.csv")
    assert len(media) == 3
    assert not media["asset_available"].any()
    assert media["provenance"].eq("synthetic_demo").all()


def test_synthetic_route_loader_rejects_missing_provenance(tmp_path: Path):
    route_path = tmp_path / "data/synthetic/routes.geojson"
    route_path.parent.mkdir(parents=True)
    route = gpd.GeoDataFrame(
        {"route_id": ["BAD-1"], "is_synthetic": [False], "provenance": ["osm"], "synthetic_disclaimer": [""]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )
    route.to_file(route_path, driver="GeoJSON")
    loaded, warning = load_synthetic_routes(tmp_path)
    assert loaded is None
    assert "No routes passed synthetic provenance validation" in warning


def test_map_center_prefers_boundary_over_fallback(tmp_path: Path):
    _write_boundary(tmp_path)
    data = load_atlas_data(tmp_path)
    latitude, longitude = map_center(data)
    assert 41.0 < latitude < 42.0
    assert 75.0 < longitude < 77.0


def test_map_center_uses_verified_naryn_point_without_boundary(tmp_path: Path):
    data_dir = tmp_path / "data/processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame([{
        "location_id": "wikidata:Q311946", "location_name": "Naryn",
        "latitude": 41.4333333, "longitude": 76.0,
    }]).to_csv(data_dir / "locations.csv", index=False)
    data = load_atlas_data(tmp_path)
    assert map_center(data) == (41.4333333, 76.0)


def test_worldcover_rgba_maps_classes_and_transparency():
    values = np.array([[30, 80, 0, 255]], dtype="uint8")
    rgba = worldcover_rgba(values, nodata=0)
    assert tuple(rgba[0, 0, :3]) == (255, 255, 76)
    assert rgba[0, 0, 3] == 220
    assert rgba[0, 1, 3] == 220
    assert rgba[0, 2, 3] == 0
    assert rgba.shape == (1, 4, 4)


def test_worldcover_class_summary_counts_native_pixels(tmp_path: Path):
    path = tmp_path / "worldcover.tif"
    values = np.array([[30, 30, 80], [70, 80, 80]], dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=3, count=1,
        dtype="uint8", crs="EPSG:4326", transform=from_origin(75, 42, 0.01, 0.01), nodata=0,
    ) as dst:
        dst.write(values, 1)
    summary = worldcover_class_summary([path])
    indexed = {row["class_code"]: row for row in summary}
    assert indexed[30]["pixel_count"] == 2
    assert indexed[80]["pixel_count"] == 3
    assert indexed[70]["class_label"] == "Snow and ice"
    assert sum(row["percent_of_valid_pixels"] for row in summary) == 100


def test_loader_does_not_invent_missing_climate_or_synthetic_records(tmp_path: Path):
    data = load_atlas_data(tmp_path)
    assert data.daily_climate.empty
    assert data.monthly_climate.empty
    assert data.synthetic_routes is None
    assert data.synthetic_interviews.empty
    assert data.synthetic_multimedia.empty
    assert any("Synthetic route demonstrations have not been generated" in item for item in data.warnings)


def test_osm_loader_never_displays_countrywide_raw_extract(tmp_path: Path):
    from src.atlas.data_access import load_atlas_data
    raw_osm = tmp_path / "data/raw/osm"
    raw_osm.mkdir(parents=True)
    # A raw/countrywide candidate is deliberately present; only the processed
    # Naryn-clipped GeoPackage is eligible for map display.
    (raw_osm / "countrywide.gpkg").write_bytes(b"not-opened-by-atlas")
    data = load_atlas_data(tmp_path)
    assert data.osm_layers == {}
    assert any("countrywide raw extract is not shown" in warning for warning in data.warnings)


def test_fictional_interview_metadata_requires_all_provenance_fields(tmp_path: Path):
    path = tmp_path / "data/synthetic/interviews.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{
        "interview_id": "SYN-INT-BAD", "title": "Not a valid synthetic row",
        "is_synthetic": True, "provenance": "synthetic_demo", "synthetic_disclaimer": "",
    }]).to_csv(path, index=False)
    frame, warning = load_synthetic_metadata(tmp_path, "interviews.csv", "interview_id")
    assert frame.empty
    assert "No rows" in warning


def test_raster_overlay_prepares_small_wgs84_image(tmp_path: Path):
    from src.atlas.overlays import raster_overlay_image
    path = tmp_path / "worldcover_overlay.tif"
    pixels = np.array([[30, 30, 80], [70, 80, 80]], dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=3, count=1,
        dtype="uint8", crs="EPSG:4326", transform=from_origin(75.0, 42.0, 0.01, 0.01), nodata=0,
    ) as dst:
        dst.write(pixels, 1)
    image, bounds, metadata = raster_overlay_image(path, kind="worldcover", max_pixels=100)
    assert image.mode == "RGBA"
    assert image.size[0] > 0 and image.size[1] > 0
    assert bounds[0][0] < bounds[1][0]
    assert bounds[0][1] < bounds[1][1]
    assert metadata["crs_display"] == "EPSG:4326"


def test_folium_geometry_sanitiser_splits_geometry_collections_and_drops_nulls():
    import folium
    import json
    from shapely.geometry import GeometryCollection, LineString, Point
    from src.atlas.data_access import folium_compatible_geodataframe

    frame = gpd.GeoDataFrame(
        {"feature_id": ["mixed", "missing"]},
        geometry=[
            GeometryCollection([
                Point(76.0, 41.4),
                LineString([(76.0, 41.4), (76.1, 41.5)]),
            ]),
            None,
        ],
        crs="EPSG:4326",
    )
    safe = folium_compatible_geodataframe(frame)

    assert len(safe) == 2
    assert set(safe.geometry.geom_type) == {"Point", "LineString"}
    assert safe["feature_id"].eq("mixed").all()

    map_object = folium.Map(location=[41.4, 76.0], zoom_start=8)
    folium.GeoJson(data=json.loads(safe.to_json())).add_to(map_object)
    bounds = map_object.get_bounds()
    assert bounds[0][0] <= 41.4 <= bounds[1][0]
    assert bounds[0][1] <= 76.0 <= bounds[1][1]


def test_raster_overlay_pil_image_is_converted_to_folium_serializable_data_uri(tmp_path: Path):
    import folium
    from branca.utilities import image_to_url
    from src.atlas.overlays import raster_overlay_image, image_to_data_uri

    path = tmp_path / "worldcover_serialization.tif"
    pixels = np.array([[30, 80], [70, 80]], dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1,
        dtype="uint8", crs="EPSG:4326",
        transform=from_origin(75.0, 42.0, 0.01, 0.01), nodata=0,
    ) as dst:
        dst.write(pixels, 1)

    image, bounds, _ = raster_overlay_image(path, kind="worldcover")
    uri = image_to_data_uri(image)
    assert uri.startswith("data:image/png;base64,")
    assert image_to_url(uri) == uri

    map_object = folium.Map(location=[41.99, 75.01], zoom_start=8)
    folium.raster_layers.ImageOverlay(image=uri, bounds=bounds).add_to(map_object)
    rendered = map_object.get_root().render()
    assert "data:image/png;base64," in rendered
