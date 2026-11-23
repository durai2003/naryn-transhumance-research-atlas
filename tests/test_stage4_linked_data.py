from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from rdflib import Graph, Namespace, RDF, OWL, URIRef
from rdflib.namespace import DCAT, DCTERMS, SKOS

from src.linked_data.build_graph import BASE, TA, build_from_root, build_graph, validate_graph
from scripts.run_linked_data import QUERY_FILES


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "metadata").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/raw/wikidata").mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    (tmp_path / "ontology").mkdir(parents=True, exist_ok=True)
    (tmp_path / "vocab").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ontology/transhumance.ttl").write_bytes((project_root / "ontology/transhumance.ttl").read_bytes())
    (tmp_path / "vocab/transhumance-vocabulary.ttl").write_bytes((project_root / "vocab/transhumance-vocabulary.ttl").read_bytes())
    (tmp_path / "data/processed/climate").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(exist_ok=True)
    (tmp_path / "data/synthetic").mkdir(parents=True)
    with (tmp_path / "metadata/dataset_register.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset_id", "dataset_title", "provider", "source_url", "expected_format",
            "licence", "limitations_and_notes", "acquisition_status",
        ])
        writer.writeheader()
        writer.writerow({
            "dataset_id": "NASA-POWER-DAILY", "dataset_title": "NASA POWER daily climate",
            "provider": "NASA POWER", "source_url": "https://power.larc.nasa.gov/docs/",
            "expected_format": "JSON/CSV", "licence": "NASA POWER attribution required",
            "limitations_and_notes": "Model grid point estimate; not a weather station.",
            "acquisition_status": "API_RESPONSE_RETRIEVED",
        })
    entities = {"entities": [{
        "label": "Naryn", "qid": "Q311946", "uri": "http://www.wikidata.org/entity/Q311946",
        "page": "https://www.wikidata.org/wiki/Q311946", "latitude": 41.4333333, "longitude": 76.0,
    }]}
    (tmp_path / "data/raw/wikidata/verified_entity_references.json").write_text(json.dumps(entities), encoding="utf-8")
    pd.DataFrame([{
        "location_id": "wikidata:Q311946", "location_name": "Naryn", "latitude": 41.4333333,
        "longitude": 76.0, "external_identifier": "Q311946", "external_uri": "http://www.wikidata.org/entity/Q311946",
        "source_url": "https://www.wikidata.org/wiki/Q311946", "is_synthetic": False,
        "provenance": "wikidata_entity_page",
    }]).to_csv(tmp_path / "data/processed/locations.csv", index=False)
    pd.DataFrame([{
        "observation_id": "NASA-POWER:wikidata:Q311946:2020-01-01", "location_id": "wikidata:Q311946",
        "observation_date": "2020-01-01", "t2m_c": 2.5, "precipitation_mm_day": None,
        "is_synthetic": True, "provenance": "synthetic_demo",
        "synthetic_disclaimer": "Unit-test fixture only; not real weather data.",
    }]).to_csv(tmp_path / "data/processed/climate/nasa_power_daily.csv", index=False)
    pd.DataFrame([{
        "location_id": "wikidata:Q311946", "year": 2020, "month": 1,
        "t2m_mean_c": 2.5, "precipitation_sum_mm": 4.0,
        "is_synthetic": True, "provenance": "synthetic_demo",
        "synthetic_disclaimer": "Unit-test fixture only; not real climate data.",
    }]).to_csv(tmp_path / "data/processed/climate/nasa_power_monthly.csv", index=False)
    pd.DataFrame([{
        "location_id": "wikidata:Q311946", "year": 2020, "season": "winter",
        "t2m_mean_c": 1.5, "precipitation_sum_mm": 12.0,
        "is_synthetic": True, "provenance": "synthetic_demo",
        "synthetic_disclaimer": "Unit-test fixture only; not real climate data.",
    }]).to_csv(tmp_path / "data/processed/climate/nasa_power_seasonal.csv", index=False)
    pd.DataFrame([{
        "interview_id": "demo-interview-001", "title": "Fictional interview metadata fixture",
        "location_id": "wikidata:Q311946", "is_synthetic": True,
        "provenance": "synthetic_demo", "synthetic_disclaimer": "Unit-test fixture; no real interview took place.",
    }]).to_csv(tmp_path / "data/synthetic/interviews.csv", index=False)
    return tmp_path


def test_graph_build_has_exact_verified_wikidata_link_and_dataset(tmp_path: Path):
    root = _workspace(tmp_path)
    graph, report = build_graph(root)
    city = URIRef(BASE + "location/q311946")
    assert (city, OWL.sameAs, URIRef("http://www.wikidata.org/entity/Q311946")) in graph
    assert report["counts"]["locations"] == 1
    assert report["counts"]["climate_observations"] == 3
    assert report["counts"]["interview_records"] == 1


def test_all_five_sparql_queries_execute_against_fixture_graph(tmp_path: Path):
    root = _workspace(tmp_path)
    build_from_root(root)
    graph = Graph().parse(root / "linked_data/research_graph.ttl", format="turtle")
    for query_file in QUERY_FILES:
        query = (Path(__file__).resolve().parents[1] / "queries" / query_file).read_text(encoding="utf-8")
        result = graph.query(query)
        assert result is not None
    climate = graph.query((Path(__file__).resolve().parents[1] / "queries/03_climate_observations_for_location.rq").read_text())
    assert len(list(climate)) == 3
    datasets = graph.query((Path(__file__).resolve().parents[1] / "queries/05_datasets_and_metadata.rq").read_text())
    assert len(list(datasets)) >= 1


def test_jsonld_export_round_trips_and_skos_concepts_present(tmp_path: Path):
    root = _workspace(tmp_path)
    report = build_from_root(root)
    ttl_graph = Graph().parse(root / "linked_data/research_graph.ttl", format="turtle")
    jsonld_graph = Graph().parse(root / "linked_data/research_graph.jsonld", format="json-ld")
    from rdflib.compare import isomorphic
    assert isomorphic(ttl_graph, jsonld_graph)
    assert report["counts"]["skos_concepts"] == 7
    assert len(set(ttl_graph.subjects(RDF.type, SKOS.Concept))) == 7


def test_worldcover_sample_is_landcover_not_a_pasture(tmp_path: Path):
    root = _workspace(tmp_path)
    pd.DataFrame([{
        "location_id": "wikidata:Q311946", "location_name": "Naryn",
        "worldcover_class_code": 30, "sheep_density_heads_km2": None,
        "goat_density_heads_km2": None, "is_synthetic": True,
        "provenance": "synthetic_demo",
        "synthetic_disclaimer": "Unit-test location-context fixture; not a source raster sample.",
    }]).to_csv(root / "data/processed/location_context.csv", index=False)
    graph, report = build_graph(root)
    assert report["counts"]["pastures"] == 0
    assert report["counts"]["routes"] == 0
    assert report.get("landcover_observations") == 1
    assert len(list(graph.subjects(RDF.type, TA.LandCoverObservation))) == 1
    assert not list(graph.subjects(RDF.type, TA.Pasture))
    assert not list(graph.subjects(RDF.type, TA.Route))


def test_synthetic_research_records_require_three_provenance_fields(tmp_path: Path):
    root = _workspace(tmp_path)
    pd.DataFrame([{
        "interview_id": "bad-demo", "title": "Must be rejected", "location_id": "wikidata:Q311946",
        "is_synthetic": True, "provenance": "synthetic_demo", "synthetic_disclaimer": "",
    }]).to_csv(root / "data/synthetic/interviews.csv", index=False)
    graph, report = build_graph(root)
    assert report["counts"]["interview_records"] == 0
    assert len(report["rejected_optional_synthetic_rows"]) == 1


def test_graph_validation_reports_missing_dataset_metadata(tmp_path: Path):
    root = _workspace(tmp_path)
    graph, _ = build_graph(root)
    incomplete = URIRef(BASE + "dataset/incomplete")
    graph.add((incomplete, RDF.type, DCAT.Dataset))
    result = validate_graph(graph, {"Q311946"})
    assert result["status"] == "FAIL"
    assert any("incomplete" in item for item in result["missing_dataset_metadata"])