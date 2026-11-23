"""Cached loader for Stage 1 spatial, temporal, and semantic transhumance network artifacts.

Intentionally keeps historical timestamped positions (`herd_positions.csv`) out of the
static explorer memory footprint so Stage 2 node/edge/graph interactions stay fast.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd

from src.network.dem_env import read_geojson_safe


@dataclass
class Stage1NetworkData:
    root: Path
    available: bool
    manifest: dict[str, Any] = field(default_factory=dict)
    boundary: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    physical_nodes: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    physical_edges: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    pastoral_nodes: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    pastoral_edges: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    all_nodes: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    all_edges: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    herds: pd.DataFrame = field(default_factory=pd.DataFrame)
    movement_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    edge_utilisation: pd.DataFrame = field(default_factory=pd.DataFrame)
    semantic_links: dict[str, Any] = field(default_factory=dict)
    multigraph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)


def _read_table_prefer_parquet(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    if parquet_path.exists() and parquet_path.stat().st_size > 0:
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def load_stage1_network(root: Path) -> Stage1NetworkData:
    """Load Stage 1 network layers and tables without loading 30k+ historical positions."""
    root = Path(root).resolve()
    net_dir = root / "data/network"
    manifest_path = net_dir / "network_manifest.json"
    if not manifest_path.exists():
        return Stage1NetworkData(root=root, available=False)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    bpath = next(
        (
            p
            for p in (
                root / "data/processed/vectors/naryn_adm1_clean.geojson",
                root / "data/raw/boundaries/naryn_adm1.geojson",
            )
            if p.exists() and p.stat().st_size > 0
        ),
        None,
    )
    boundary = read_geojson_safe(bpath).to_crs("EPSG:4326") if bpath else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def _load_geo(name: str) -> gpd.GeoDataFrame:
        p = net_dir / name
        if not p.exists() or p.stat().st_size == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        gdf = read_geojson_safe(p)
        if gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return gdf.to_crs("EPSG:4326")

    phys_nodes = _load_geo("layer_a_physical_nodes.geojson")
    phys_edges = _load_geo("layer_a_physical_edges.geojson")
    past_nodes = _load_geo("layer_b_pastoral_nodes.geojson")
    past_edges = _load_geo("layer_b_pastoral_edges.geojson")

    herds = _read_table_prefer_parquet(net_dir / "herds.csv", net_dir / "herds.parquet")
    events = _read_table_prefer_parquet(net_dir / "movement_events.csv", net_dir / "movement_events.parquet")
    util = _read_table_prefer_parquet(net_dir / "edge_utilisation.csv", net_dir / "edge_utilisation.parquet")

    # Combine Layer A and Layer B while strictly preserving `network_layer`, `evidence_status`, `is_synthetic`
    node_frames = [df for df in (phys_nodes, past_nodes) if not df.empty]
    all_nodes = (
        gpd.GeoDataFrame(pd.concat(node_frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
        .drop_duplicates(subset=["node_id"])
        .reset_index(drop=True)
        if node_frames
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )

    edge_frames = [df for df in (phys_edges, past_edges) if not df.empty]
    all_edges = (
        gpd.GeoDataFrame(pd.concat(edge_frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
        .drop_duplicates(subset=["edge_id"])
        .reset_index(drop=True)
        if edge_frames
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )

    # Ensure utilisation columns exist uniformly on Layer A & Layer B edges (0 for unused/physical edges)
    util_defaults = {
        "herd_count": 0,
        "animal_count": 0,
        "movement_event_count": 0,
        "winter_usage": 0,
        "spring_usage": 0,
        "summer_usage": 0,
        "autumn_usage": 0,
        "avg_estimated_travel_days": 0.0,
        "alternative_route_usage": 0,
    }
    if not all_edges.empty:
        for col, default_val in util_defaults.items():
            if col not in all_edges.columns:
                all_edges[col] = default_val
            else:
                all_edges[col] = all_edges[col].fillna(default_val)

    sem_path = net_dir / "layer_c_semantic_links.json"
    semantic_links = json.loads(sem_path.read_text(encoding="utf-8")) if sem_path.exists() else {}

    mg_path = net_dir / "transhumance_multigraph.json"
    if mg_path.exists():
        mg_json = json.loads(mg_path.read_text(encoding="utf-8"))
        edges_key = "links" if "links" in mg_json else "edges"
        G = nx.node_link_graph(mg_json, directed=True, multigraph=True, edges=edges_key)
    else:
        G = nx.MultiDiGraph()

    return Stage1NetworkData(
        root=root,
        available=True,
        manifest=manifest,
        boundary=boundary,
        physical_nodes=phys_nodes,
        physical_edges=phys_edges,
        pastoral_nodes=past_nodes,
        pastoral_edges=past_edges,
        all_nodes=all_nodes,
        all_edges=all_edges,
        herds=herds,
        movement_events=events,
        edge_utilisation=util,
        semantic_links=semantic_links,
        multigraph=G,
    )
