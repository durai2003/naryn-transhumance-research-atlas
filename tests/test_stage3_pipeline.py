"""Stage 3 tests use isolated, explicitly illustrative fixtures only."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, Point

from src.preprocessing.tabular import clean_nasa_power_document, aggregate_climate, standardise_frame
from src.geospatial.vector import clean_geodataframe, clip_vector_to_boundary
from src.geospatial.raster import clip_raster_to_boundary, sample_raster_at_points
from src.validation.quality import validate_table
from src.integration.locations import locations_from_verified_wikidata
from src.pipeline import run_stage3


def test_nasa_parser_normalises_and_preserves_missing_sentinel_as_null():
    # Test-only example, not an actual NASA observation.
    fixture = {
        "properties": {"parameter": {
            "T2M": {"20200101": " 4.2 ", "20200102": "-999"},
            "PRECTOTCORR": {"20200101": "-999", "20200102": "1.25"},
        }}
    }
    result = clean_nasa_power_document(
        fixture, location_id="wikidata:Q311946", location_name=" NARYN   CITY ",
        latitude=41.4333333333, longitude=76.0, source_url="https://example.invalid/test-fixture",
        retrieved_at="2026-11-21T00:00:00+00:00",
        is_synthetic=True,
        provenance="synthetic_demo",
        synthetic_disclaimer="Test fixture only; not observed weather data.",
    )
    assert result.loc[0, "location_name"] == "Naryn City"
    assert result.loc[0, "t2m_c"] == 4.2
    assert pd.isna(result.loc[0, "precipitation_mm_day"])
    assert pd.isna(result.loc[1, "t2m_c"])
    assert result.loc[1, "precipitation_mm_day"] == 1.25
    assert result["is_synthetic"].all()
    assert result["provenance"].eq("synthetic_demo").all()
    assert result["synthetic_disclaimer"].str.len().gt(0).all()


def test_synthetic_nasa_fixture_requires_explicit_disclaimer():
    fixture = {"properties": {"parameter": {"T2M": {"20200101": "1"}}}}
    with pytest.raises(ValueError, match="explicit disclaimer"):
        clean_nasa_power_document(
            fixture, location_id="fixture", location_name="fixture", latitude=41, longitude=76,
            source_url="https://example.invalid", is_synthetic=True, provenance="synthetic_demo",
        )
    with pytest.raises(ValueError, match="requires is_synthetic=True"):
        clean_nasa_power_document(
            fixture, location_id="fixture", location_name="fixture", latitude=41, longitude=76,
            source_url="https://example.invalid", is_synthetic=False, provenance="synthetic_demo",
        )


def test_climate_monthly_and_seasonal_aggregation():
    frame = pd.DataFrame({
        "location_id": ["loc-1", "loc-1", "loc-1"],
        "observation_date": ["2020-01-01", "2020-01-02", "2020-07-01"],
        "t2m_c": [1.0, 3.0, 10.0],
        "precipitation_mm_day": [2.0, 1.0, 4.0],
    })
    monthly, seasonal = aggregate_climate(frame)
    assert len(monthly) == 2
    jan = monthly.loc[monthly.month.eq(1)].iloc[0]
    assert jan.t2m_mean_c == 2.0
    assert jan.precipitation_sum_mm == 3.0
    assert set(seasonal.season) == {"winter", "summer"}


def test_climate_aggregation_keeps_all_missing_precipitation_null():
    frame = pd.DataFrame({
        "location_id": ["loc-1", "loc-1"],
        "observation_date": ["2020-01-01", "2020-01-02"],
        "t2m_c": [1.0, 2.0],
        "precipitation_mm_day": [None, None],
    })
    monthly, _ = aggregate_climate(frame)
    assert pd.isna(monthly.loc[0, "precipitation_sum_mm"])
    assert monthly.loc[0, "precipitation_valid_days"] == 0


def test_standardise_frame_column_names_text_and_date():
    frame = pd.DataFrame({" Location Name ": ["  NARYN   CITY "], "Obs Date": ["2020-01-02"]})
    cleaned = standardise_frame(frame, text_columns=("location_name",), date_columns=("obs_date",))
    assert list(cleaned.columns) == ["location_name", "obs_date"]
    assert cleaned.iloc[0].location_name == "NARYN CITY"
    assert cleaned.iloc[0].obs_date == "2020-01-02"


def test_vector_cleaning_repairs_invalid_and_drops_duplicate_id():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame({"feature_id": ["a", "a"], "label": ["first", "duplicate"]},
                           geometry=[bowtie, Point(0.5, 0.5)], crs="EPSG:4326")
    cleaned, report = clean_geodataframe(gdf, id_column="feature_id")
    assert len(cleaned) == 1
    assert cleaned.geometry.iloc[0].is_valid
    assert report["repaired"] == 1
    assert report["duplicates_removed"] == 1


def test_vector_without_crs_is_rejected():
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(76, 41)])
    with pytest.raises(ValueError, match="no CRS"):
        clean_geodataframe(gdf)


def test_vector_clip_uses_the_supplied_boundary():
    source = gpd.GeoDataFrame({"id": ["crossing"]},
                              geometry=[Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])], crs="EPSG:4326")
    boundary = gpd.GeoDataFrame({"name": ["test boundary"]},
                                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
    clipped, report = clip_vector_to_boundary(source, boundary, id_column="id")
    assert len(clipped) == 1
    assert clipped.total_bounds == pytest.approx([0, 0, 1, 1])
    assert report["clipped_features"] == 1


def test_raster_clip_preserves_source_grid_resolution(tmp_path: Path):
    raster_path = tmp_path / "test_source.tif"
    boundary_path = tmp_path / "test_boundary.geojson"
    output_path = tmp_path / "test_clip.tif"
    data = np.arange(100, dtype="float32").reshape((10, 10))
    with rasterio.open(raster_path, "w", driver="GTiff", height=10, width=10, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(0, 10, 1, 1), nodata=-9999) as dst:
        dst.write(data, 1)
    boundary = gpd.GeoDataFrame({"id": ["test"]},
                                geometry=[Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])], crs="EPSG:4326")
    boundary.to_file(boundary_path, driver="GeoJSON")
    clip_raster_to_boundary(raster_path, boundary_path, output_path)
    with rasterio.open(raster_path) as src, rasterio.open(output_path) as out:
        assert out.res == src.res
        assert out.crs == src.crs
        assert out.nodata == src.nodata
        assert out.width == 6 and out.height == 6


def test_raster_sample_transforms_point_to_raster_crs(tmp_path: Path):
    raster_path = tmp_path / "point_sample.tif"
    arr = np.full((4, 4), 7, dtype="float32")
    with rasterio.open(raster_path, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(74, 43, 1, 1), nodata=-9999) as dst:
        dst.write(arr, 1)
    points = gpd.GeoDataFrame({"location_id": ["test-loc"]}, geometry=[Point(75.5, 41.5)], crs="EPSG:4326")
    result = sample_raster_at_points(raster_path, points)
    assert result.loc[0, "raster_value"] == 7.0


def test_validator_flags_bad_coordinates_and_synthetic_provenance():
    frame = pd.DataFrame({
        "record_id": ["x"], "latitude": [200], "longitude": [76],
        "is_synthetic": [False], "provenance": ["synthetic_demo"],
    })
    result = validate_table(frame, dataset_name="test", required_columns=["record_id"], primary_key="record_id",
                            latitude_column="latitude", longitude_column="longitude")
    assert result["status"] == "FAIL"
    assert result["invalid_coordinate_rows"] == 1
    assert result["synthetic_provenance_issues"] == 1


def test_verified_wikidata_location_extraction_uses_only_coordinate_bearing_entities(tmp_path: Path):
    source = tmp_path / "verified.json"
    source.write_text(json.dumps({"entities": [
        {"label": "Naryn Region", "qid": "Q486375", "uri": "http://www.wikidata.org/entity/Q486375", "page": "https://www.wikidata.org/wiki/Q486375"},
        {"label": "Naryn", "qid": "Q311946", "uri": "http://www.wikidata.org/entity/Q311946", "page": "https://www.wikidata.org/wiki/Q311946", "latitude": 41.4333333, "longitude": 76.0},
    ]}), encoding="utf-8")
    locations = locations_from_verified_wikidata(source)
    assert len(locations) == 1
    assert locations.iloc[0].external_identifier == "Q311946"
    assert bool(locations.iloc[0].is_synthetic) is False


def test_pipeline_no_downloaded_datasets_is_explicit_and_never_fabricates(tmp_path: Path):
    # Minimal independent workspace assembled from existing verified metadata only.
    (tmp_path / "metadata").mkdir(parents=True)
    (tmp_path / "data/raw/wikidata").mkdir(parents=True)
    (tmp_path / "data/raw/nasa_power").mkdir(parents=True)
    (tmp_path / "data/raw/boundaries").mkdir(parents=True)
    (tmp_path / "data/raw/glw4").mkdir(parents=True)
    (tmp_path / "data/raw/worldcover").mkdir(parents=True)
    (tmp_path / "data/raw/osm").mkdir(parents=True)
    register = Path(__file__).resolve().parents[1] / "metadata/dataset_register.csv"
    (tmp_path / "metadata/dataset_register.csv").write_bytes(register.read_bytes())
    wikidata = Path(__file__).resolve().parents[1] / "data/raw/wikidata/verified_entity_references.json"
    (tmp_path / "data/raw/wikidata/verified_entity_references.json").write_bytes(wikidata.read_bytes())
    report = run_stage3(tmp_path)
    assert report["summary"]["locations_from_verified_entity_pages"] == 1
    assert report["summary"]["daily_climate_records"] == 0
    assert report["summary"]["source_data_fabricated"] is False
    assert report["summary"]["spatial_processing_completed"] is False
    assert report["blocked_steps"]
    location_table = pd.read_csv(tmp_path / "data/processed/locations.csv")
    assert location_table.iloc[0].external_identifier == "Q311946"
    assert not location_table["is_synthetic"].any()


def test_projected_analysis_crs_is_used_for_metric_geometry():
    from src.geospatial.vector import project_for_metric_analysis, add_area_km2, ANALYSIS_CRS
    polygon = gpd.GeoDataFrame({"name": ["test polygon"]},
                               geometry=[Polygon([(76, 41), (76.01, 41), (76.01, 41.01), (76, 41.01)])],
                               crs="EPSG:4326")
    projected = project_for_metric_analysis(polygon)
    assert projected.crs.to_string() == ANALYSIS_CRS
    assert projected.geometry.iloc[0].area > 0
    measured = add_area_km2(polygon)
    assert measured.loc[0, "area_km2"] > 0


def test_geofabrik_geopackage_zip_is_extracted_to_fixed_safe_path(tmp_path: Path):
    import zipfile
    import geopandas as gpd
    from shapely.geometry import Point
    from src.pipeline import _discover_osm_geopackages
    from src.geospatial.vector import geopackage_layers

    osm_dir = tmp_path / "osm"
    osm_dir.mkdir()

    # Create a genuine GeoPackage.
    source_gpkg = tmp_path / "source.gpkg"
    fixture = gpd.GeoDataFrame(
        {"name": ["test_location"]},
        geometry=[Point(76.0, 41.43)],
        crs="EPSG:4326",
    )
    fixture.to_file(
        source_gpkg,
        driver="GPKG",
        layer="fixture_points",
        index=False,
    )

    # Package it using a nested archive path.
    archive = osm_dir / "kyrgyzstan-latest-free.gpkg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source_gpkg, arcname="nested/path/kyrgyzstan.gpkg")
        zf.writestr("README.txt", "archive readme")

    paths, errors = _discover_osm_geopackages(
        osm_dir, osm_dir / "extracted"
    )

    assert not errors, errors
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].name == "kyrgyzstan.gpkg"
    assert paths[0].parent == osm_dir / "extracted"
    assert "fixture_points" in geopackage_layers(paths[0])
