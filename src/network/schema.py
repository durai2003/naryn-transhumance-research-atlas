"""Canonical schema definitions for the 3-layer Transhumance Network (Stage 1).

Three logically separate layers linked by stable identifiers:
- Layer A (`physical_geographical`): Genuine verifiable OSM/Wikidata features ONLY.
- Layer B (`pastoral_mobility`): Seasonal pastures, passes, resting points, water access,
  modelled corridors, synthetic herds, movement events, and time-indexed herd positions.
- Layer C (`linked_knowledge`): Semantic URIs and PROV-O / DCAT / GeoSPARQL linkage manifest.
"""
from __future__ import annotations

NETWORK_LAYERS = ("physical_geographical", "pastoral_mobility", "linked_knowledge")

# Static geographical node categories stored in the NetworkX MultiDiGraph.
# Note: `herd_location` is a temporal observation record stored in `herd_positions`,
# NOT a permanent static geographical graph node.
STATIC_NODE_TYPES = (
    "winter_pasture",
    "spring_pasture",
    "summer_pasture",
    "autumn_pasture",
    "settlement",
    "water_point",
    "mountain_pass",
    "resting_point",
)

ALL_NODE_TYPES = STATIC_NODE_TYPES + ("herd_location",)

PASTURE_NODE_TYPES = (
    "winter_pasture",
    "spring_pasture",
    "summer_pasture",
    "autumn_pasture",
)

EDGE_TYPES = (
    "SEASONAL_MOVEMENT",
    "ACCESS_ROUTE",
    "LEAST_COST_CORRIDOR",
    "WATER_ACCESS",
    "SETTLEMENT_CONNECTION",
    "ALTERNATIVE_ROUTE",
)

MOVEMENT_STATES = (
    "GRAZING",
    "TRAVELLING",
    "RESTING",
    "DEPARTING",
    "ARRIVING",
    "RETURNING",
)

EVIDENCE_STATUSES = (
    "verified_public",
    "derived_environmental",
    "synthetic_demo",
)

REQUIRED_NODE_COLUMNS = (
    "node_id",
    "node_name",
    "node_type",
    "network_layer",
    "latitude",
    "longitude",
    "elevation_m",
    "elevation_status",
    "elevation_source",
    "source",
    "provenance",
    "evidence_status",
    "is_synthetic",
)

REQUIRED_EDGE_COLUMNS = (
    "edge_id",
    "source_node_id",
    "target_node_id",
    "edge_type",
    "network_layer",
    "distance_m",
    "elevation_gain_m",
    "maximum_slope",
    "average_slope",
    "season",
    "direction",
    "livestock_compatibility",
    "route_generation_method",
    "source",
    "provenance",
    "evidence_status",
    "is_synthetic",
)

REQUIRED_POSITION_COLUMNS = (
    "herd_id",
    "timestamp",
    "latitude",
    "longitude",
    "elevation_m",
    "elevation_status",
    "movement_state",
    "current_node_id",
    "current_edge_id",
    "origin_node_id",
    "destination_node_id",
    "season",
    "livestock_type",
    "scenario_id",
    "is_synthetic",
)

REQUIRED_EVENT_COLUMNS = (
    "event_id",
    "herd_id",
    "livestock_type",
    "animal_count",
    "season",
    "leg_name",
    "edge_id",
    "origin_node_id",
    "destination_node_id",
    "departure_timestamp",
    "arrival_timestamp",
    "distance_m",
    "estimated_travel_days",
    "used_alternative_route",
    "scenario_id",
    "provenance",
    "is_synthetic",
)

REQUIRED_UTILISATION_COLUMNS = (
    "edge_id",
    "source_node_id",
    "target_node_id",
    "edge_type",
    "herd_count",
    "animal_count",
    "movement_event_count",
    "winter_usage",
    "spring_usage",
    "summer_usage",
    "autumn_usage",
    "avg_estimated_travel_days",
    "alternative_route_usage",
)
