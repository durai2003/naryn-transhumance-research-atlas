"""Automated RDF integration tests for candidate pastures, corridors, herds, movement events, and suitability."""
from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF
from rdflib.namespace import GEO, XSD

from scripts.run_linked_data import run_queries
from src.linked_data.build_graph import build_from_root

ROOT = Path(__file__).resolve().parents[1]
TA = Namespace("https://example.org/transhumance-atlas/ontology#")


def test_rdf_graph_contains_pastures_corridors_herds_events_and_suitability():
    report = build_from_root(ROOT)
    assert report["validation"]["status"] == "PASS", report["validation"]["issues"]

    shacl = report.get("shacl_validation", {})
    if shacl.get("status") not in ("NOT_RUN", "NOT_RUN_OPTIONAL_DEPENDENCY_MISSING"):
        assert shacl["status"] == "PASS", shacl.get("details", "")

    counts = report["counts"]
    assert counts["pastures"] == 18
    assert counts["routes"] == 21
    assert counts["herd_groups"] == 48
    assert counts["movement_events"] == 1152
    assert counts["simulation_scenarios"] == 3
    assert counts["suitability_assessments"] == 216

    graph = Graph().parse(ROOT / "linked_data" / "research_graph.ttl", format="turtle")

    # Every candidate pasture has GeoSPARQL geometry, elevation, and synthetic provenance
    pastures = list(graph.subjects(RDF.type, TA.Pasture))
    assert len(pastures) == 18
    for p in pastures:
        assert list(graph.objects(p, GEO.hasGeometry)), f"{p} missing geo:hasGeometry"
        assert list(graph.objects(p, TA.elevationM)), f"{p} missing ta:elevationM"
        assert (p, TA.isSynthetic, Literal(True, datatype=XSD.boolean)) in graph
        assert (p, TA.provenanceLabel, Literal("synthetic_demo")) in graph
        assert list(graph.objects(p, TA.hasSuitabilityAssessment)), f"{p} missing suitability assessments"

    # Every corridor connects 2 pastures
    routes = list(graph.subjects(RDF.type, TA.Route))
    assert len(routes) == 21
    for r in routes:
        connected = list(graph.objects(r, TA.connectsPasture))
        assert len(connected) == 2, f"{r} expected 2 connected pastures, got {len(connected)}"

    # Every herd links to species, groupSize, corridors, and seasonal pastures
    herds = list(graph.subjects(RDF.type, TA.HerdGroup))
    assert len(herds) == 48
    for h in herds:
        assert list(graph.objects(h, TA.livestockSpecies))
        assert list(graph.objects(h, TA.groupSize))
        assert list(graph.objects(h, TA.usesRoute)), f"{h} missing ta:usesRoute"
        assert list(graph.objects(h, TA.usesPasture)), f"{h} missing ta:usesPasture"

    # Every movement event links to timestamp, herd, pasture, and scenario
    events = list(graph.subjects(RDF.type, TA.MovementEvent))
    assert len(events) == 1152
    for e in events:
        assert list(graph.objects(e, TA.occurredAt))
        assert list(graph.objects(e, TA.involvesHerd))
        assert list(graph.objects(e, TA.aboutLocation))
        assert list(graph.objects(e, TA.underScenario))


def test_sparql_queries_return_pastures_routes_herds_events_and_suitability():
    bundle = run_queries(ROOT)
    assert bundle["graph_validation"]["status"] == "PASS"
    q = bundle["queries"]
    assert q["01_pasture_locations.rq"]["row_count"] == 18
    assert q["02_routes_for_pasture.rq"]["row_count"] == 42
    assert q["06_synthetic_herds_and_routes.rq"]["row_count"] >= 48
    assert q["07_scenario_suitability_and_events.rq"]["row_count"] == 100  # LIMIT 100 of 1,152 events
    if "08_pasture_suitability_by_scenario.rq" in q:
        assert q["08_pasture_suitability_by_scenario.rq"]["row_count"] == 100  # LIMIT 100 of 216 assessments
