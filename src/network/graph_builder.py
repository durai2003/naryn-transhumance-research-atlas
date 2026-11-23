"""Assemble the NetworkX MultiDiGraph across Layers A, B, and C and export structured Stage 1 datasets."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd

from src.network.dem_env import write_geojson_safe

PROTOTYPE_URI_BASE = "https://example.org/transhumance-atlas/"


def build_networkx_multidigraph(
    physical_nodes: gpd.GeoDataFrame,
    physical_edges: gpd.GeoDataFrame,
    pastoral_nodes: gpd.GeoDataFrame,
    pastoral_edges: gpd.GeoDataFrame,
    edge_utilisation: pd.DataFrame,
    cfg: dict[str, Any],
) -> nx.MultiDiGraph:
    """Build a directed multigraph with Layer A & Layer B clearly tagged on every node/edge."""
    G = nx.MultiDiGraph(
        name="Naryn Transhumance Spatial-Temporal-Semantic Network (Stage 1)",
        generator_version=str(cfg.get("generator_version", "3.0.0-stage1")),
        random_seed=int(cfg.get("random_seed", 42)),
        scenario_id=str(cfg.get("scenario_id", "SCENARIO_BASELINE")),
        study_area=str(cfg.get("study_area", "Naryn Region, Kyrgyzstan")),
        metric_crs=str(cfg.get("metric_crs", "EPSG:32643")),
        web_crs=str(cfg.get("web_crs", "EPSG:4326")),
    )

    def _clean_val(v: Any) -> Any:
        if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
            return None
        if isinstance(v, (int, float, bool, str)):
            return v
        return str(v)

    for df in (physical_nodes, pastoral_nodes):
        if df.empty:
            continue
        for _, row in df.iterrows():
            nid = str(row["node_id"])
            attrs = {k: _clean_val(v) for k, v in row.items() if k != "geometry"}
            attrs["wkt"] = row.geometry.wkt if row.geometry is not None else None
            G.add_node(nid, **attrs)

    util_by_edge = edge_utilisation.set_index("edge_id").to_dict(orient="index") if not edge_utilisation.empty else {}

    for df in (physical_edges, pastoral_edges):
        if df.empty:
            continue
        for _, row in df.iterrows():
            eid = str(row["edge_id"])
            src = str(row["source_node_id"])
            dst = str(row["target_node_id"])
            attrs = {k: _clean_val(v) for k, v in row.items() if k not in ("geometry", "elevation_profile_json")}
            # Merge exact derived utilisation into edge attributes (0 for physical/unused edges)
            u = util_by_edge.get(eid)
            if u:
                attrs.update({k: _clean_val(v) for k, v in u.items() if k not in ("edge_id", "source_node_id", "target_node_id", "edge_type")})
            else:
                attrs.update({
                    "herd_count": 0, "animal_count": 0, "movement_event_count": 0,
                    "winter_usage": 0, "spring_usage": 0, "summer_usage": 0, "autumn_usage": 0,
                    "avg_estimated_travel_days": 0.0, "alternative_route_usage": 0,
                })
            G.add_edge(src, dst, key=eid, **attrs)

    return G


def build_layer_c_semantic_links(
    physical_nodes: gpd.GeoDataFrame,
    pastoral_nodes: gpd.GeoDataFrame,
    pastoral_edges: gpd.GeoDataFrame,
    herds_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build Layer C stable-identifier semantic linkage manifest bridging GIS layers to Linked Open Data."""
    entity_links: list[dict[str, Any]] = []

    if not physical_nodes.empty:
        for _, r in physical_nodes.iterrows():
            entity_links.append({
                "stable_id": str(r["node_id"]),
                "network_layer": "physical_geographical",
                "rdf_type": "ta:Location",
                "prototype_uri": f"{PROTOTYPE_URI_BASE}node/{r['node_id']}",
                "external_uri": r.get("external_uri"),
                "evidence_status": str(r.get("evidence_status", "verified_public")),
                "is_synthetic": False,
                "prov_wasDerivedFrom": str(r.get("provenance", "")),
            })

    if not pastoral_nodes.empty:
        for _, r in pastoral_nodes.iterrows():
            ntype = str(r["node_type"])
            rdf_cls = "ta:Pasture" if ntype.endswith("_pasture") else "ta:Location"
            entity_links.append({
                "stable_id": str(r["node_id"]),
                "network_layer": "pastoral_mobility",
                "rdf_type": rdf_cls,
                "prototype_uri": f"{PROTOTYPE_URI_BASE}node/{r['node_id']}",
                "external_uri": None,
                "evidence_status": str(r.get("evidence_status", "synthetic_demo")),
                "is_synthetic": True,
                "prov_wasDerivedFrom": str(r.get("provenance", "synthetic_demo")),
            })

    if not herds_df.empty:
        for _, r in herds_df.iterrows():
            entity_links.append({
                "stable_id": str(r["herd_id"]),
                "network_layer": "pastoral_mobility",
                "rdf_type": "ta:HerdGroup",
                "prototype_uri": f"{PROTOTYPE_URI_BASE}herd/{r['herd_id']}",
                "external_uri": None,
                "evidence_status": "synthetic_demo",
                "is_synthetic": True,
                "prov_wasDerivedFrom": "synthetic_demo",
            })

    return {
        "layer": "linked_knowledge",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": str(cfg.get("generator_version", "3.0.0-stage1")),
        "random_seed": int(cfg.get("random_seed", 42)),
        "scenario_id": str(cfg.get("scenario_id", "SCENARIO_BASELINE")),
        "note": "Prototype example.org URIs are local stage identifiers, not registered persistent identifiers. Connects Layer A (physical) and Layer B (pastoral) to existing Stage 4 ontology/transhumance.ttl and linked_data/research_graph.ttl.",
        "study_area_sameAs": "http://www.wikidata.org/entity/Q486375",
        "reference_city_sameAs": "http://www.wikidata.org/entity/Q311946",
        "linked_entity_count": len(entity_links),
        "pastoral_edge_count": int(len(pastoral_edges)),
        "entities": entity_links,
    }


def _save_tabular(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> list[str]:
    saved = []
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    saved.append(csv_path.name)
    try:
        df.to_parquet(parquet_path, index=False)
        saved.append(parquet_path.name)
    except Exception:
        pass
    return saved


def export_stage1_outputs(
    root: Path,
    *,
    G: nx.MultiDiGraph,
    physical_nodes: gpd.GeoDataFrame,
    physical_edges: gpd.GeoDataFrame,
    pastoral_nodes: gpd.GeoDataFrame,
    pastoral_edges: gpd.GeoDataFrame,
    herds_df: pd.DataFrame,
    events_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    util_df: pd.DataFrame,
    layer_c_links: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(root).resolve() / "data/network"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Layer A & Layer B geospatial exports
    write_geojson_safe(physical_nodes, out_dir / "layer_a_physical_nodes.geojson")
    write_geojson_safe(physical_edges, out_dir / "layer_a_physical_edges.geojson")
    write_geojson_safe(pastoral_nodes, out_dir / "layer_b_pastoral_nodes.geojson")

    # Merge utilisation stats into Layer B edges GeoJSON for direct Stage 2/3 map+graph consumption
    edges_with_util = pastoral_edges.merge(
        util_df[["edge_id", "herd_count", "animal_count", "movement_event_count",
                 "winter_usage", "spring_usage", "summer_usage", "autumn_usage",
                 "avg_estimated_travel_days", "alternative_route_usage"]],
        on="edge_id", how="left",
    )
    write_geojson_safe(edges_with_util, out_dir / "layer_b_pastoral_edges.geojson")

    # Tabular & Parquet exports
    herds_files = _save_tabular(herds_df, out_dir / "herds.csv", out_dir / "herds.parquet")
    events_files = _save_tabular(events_df, out_dir / "movement_events.csv", out_dir / "movement_events.parquet")
    pos_files = _save_tabular(positions_df, out_dir / "herd_positions.csv", out_dir / "herd_positions.parquet")
    util_files = _save_tabular(util_df, out_dir / "edge_utilisation.csv", out_dir / "edge_utilisation.parquet")

    # NetworkX MultiDiGraph exports (node-link JSON + GraphML with clean string/primitive attributes)
    nl_data = nx.node_link_data(G, edges="links")
    (out_dir / "transhumance_multigraph.json").write_text(
        json.dumps(nl_data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    # GraphML requires non-None scalar attributes
    G_gml = nx.MultiDiGraph()
    for k, v in G.graph.items():
        if v is not None:
            G_gml.graph[k] = str(v) if not isinstance(v, (int, float, bool, str)) else v
    for nid, attrs in G.nodes(data=True):
        G_gml.add_node(nid, **{k: ("" if v is None else v) for k, v in attrs.items()})
    for u, v, key, attrs in G.edges(keys=True, data=True):
        G_gml.add_edge(u, v, key=key, **{k: ("" if val is None else val) for k, val in attrs.items()})
    nx.write_graphml(G_gml, out_dir / "transhumance_multigraph.graphml")

    # Layer C semantic link manifest & build manifest
    (out_dir / "layer_c_semantic_links.json").write_text(
        json.dumps(layer_c_links, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest["exported_files"] = {
        "layer_a": ["layer_a_physical_nodes.geojson", "layer_a_physical_edges.geojson"],
        "layer_b": ["layer_b_pastoral_nodes.geojson", "layer_b_pastoral_edges.geojson"],
        "layer_c": ["layer_c_semantic_links.json"],
        "multigraph": ["transhumance_multigraph.json", "transhumance_multigraph.graphml"],
        "herds": herds_files,
        "movement_events": events_files,
        "herd_positions": pos_files,
        "edge_utilisation": util_files,
    }
    (out_dir / "network_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
