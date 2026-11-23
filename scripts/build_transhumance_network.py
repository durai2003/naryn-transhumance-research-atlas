#!/usr/bin/env python3
"""Stage 1 CLI: Build & validate the 3-layer spatial, temporal, and semantic transhumance network.

Usage:
    python3 scripts/build_transhumance_network.py --root . --seed 42
    python3 scripts/build_transhumance_network.py --root . --inspect
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.atlas.simulation.config_loader import load_yaml_config
from src.network.dem_env import discover_environment, read_geojson_safe
from src.network.graph_builder import (
    build_layer_c_semantic_links,
    build_networkx_multidigraph,
    export_stage1_outputs,
)
from src.network.herd_simulator import simulate_herds_and_network_usage
from src.network.pasture_network import build_pastoral_nodes
from src.network.physical_layer import build_physical_layer
from src.network.routing import build_pastoral_edges_and_resting_nodes
from src.validation.network_stage1_checks import run_all_stage1_validations


def build_stage1_network(
    root: Path,
    *,
    seed: int | None = None,
    num_herds: int | None = None,
    num_pastures: int | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    cfg = load_yaml_config(root / "config/network_stage1.yaml")
    if seed is not None:
        cfg["random_seed"] = int(seed)
    if num_herds is not None:
        cfg.setdefault("network_scale", {})["num_herds"] = int(num_herds)
    if num_pastures is not None:
        cfg.setdefault("network_scale", {})["num_pastures"] = int(num_pastures)

    use_seed = int(cfg.get("random_seed", 42))
    env = discover_environment(root)
    if env.boundary_path is None:
        raise FileNotFoundError("Authoritative Naryn ADM1 boundary not found under data/processed/vectors/ or data/raw/boundaries/.")
    boundary_gdf = read_geojson_safe(env.boundary_path)
    if boundary_gdf.crs is None:
        raise ValueError("Naryn ADM1 boundary has no declared CRS.")

    # Layer A: Physical geographical network (verified public features only)
    physical_nodes, physical_edges, layer_a_report = build_physical_layer(boundary_gdf, env, cfg)

    # Layer B (Part 1): Static pastoral nodes (40 candidate pastures, passes, water points)
    pastoral_nodes_base, pasture_report = build_pastoral_nodes(boundary_gdf, physical_nodes, env, cfg, seed=use_seed)

    # Layer B (Part 2): Metric least-cost corridors, alternative routes, access/water/settlement edges + resting nodes
    pastoral_nodes, pastoral_edges, routing_report = build_pastoral_edges_and_resting_nodes(
        pastoral_nodes_base, physical_nodes, boundary_gdf, env, cfg, seed=use_seed
    )

    # Layer B (Part 3): Diverse herd simulation, movement events, route-interpolated positions, derived edge usage
    all_static_nodes = gpd.GeoDataFrame(
        pd.concat([physical_nodes, pastoral_nodes], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    herds_df, events_df, positions_df, util_df, sim_report = simulate_herds_and_network_usage(
        all_static_nodes, pastoral_edges, env, cfg, seed=use_seed, num_herds_override=num_herds
    )

    # MultiDiGraph + Layer C semantic links
    G = build_networkx_multidigraph(physical_nodes, physical_edges, pastoral_nodes, pastoral_edges, util_df, cfg)
    layer_c_links = build_layer_c_semantic_links(physical_nodes, pastoral_nodes, pastoral_edges, herds_df, cfg)

    manifest: dict[str, Any] = {
        "stage": "Stage 1 — Spatial, Temporal and Semantic Transhumance Network",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": str(cfg.get("generator_version", "3.0.0-stage1")),
        "random_seed": use_seed,
        "scenario_id": str(cfg.get("scenario_id", "SCENARIO_BASELINE")),
        "simulation_period": [str(cfg.get("start_date")), str(cfg.get("end_date"))],
        "environment_inputs": {
            "boundary_path": env.boundary_path.relative_to(root).as_posix() if env.boundary_path else None,
            "dem_available": env.dem_path is not None,
            "dem_source": env.dem_source,
            "worldcover_tiles_available": len(env.worldcover_paths),
            "osm_gpkg_available": env.osm_gpkg_path is not None,
            "climate_daily_available": env.climate_daily_path is not None,
        },
        "layer_a_physical": layer_a_report,
        "layer_b_pastures": pasture_report,
        "layer_b_routing": routing_report,
        "layer_b_simulation": sim_report,
        "layer_c_semantic_entities": layer_c_links["linked_entity_count"],
        "multigraph_summary": {
            "graph_class": "networkx.MultiDiGraph",
            "total_nodes": int(G.number_of_nodes()),
            "total_edges": int(G.number_of_edges()),
        },
    }

    manifest = export_stage1_outputs(
        root,
        G=G,
        physical_nodes=physical_nodes,
        physical_edges=physical_edges,
        pastoral_nodes=pastoral_nodes,
        pastoral_edges=pastoral_edges,
        herds_df=herds_df,
        events_df=events_df,
        positions_df=positions_df,
        util_df=util_df,
        layer_c_links=layer_c_links,
        manifest=manifest,
    )

    validation = run_all_stage1_validations(root)
    report_dir = root / "reports/network_stage1_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    full_report = {"manifest": manifest, "validation": validation}
    (report_dir / "stage1_validation_report.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return full_report


def inspect_existing(root: Path) -> int:
    mpath = Path(root).resolve() / "data/network/network_manifest.json"
    vpath = Path(root).resolve() / "reports/network_stage1_validation/stage1_validation_report.json"
    if not mpath.exists():
        print("No Stage 1 network generated yet. Run without --inspect first.")
        return 1
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    print("=== STAGE 1 NETWORK MANIFEST ===")
    print(json.dumps(manifest, indent=2))
    if vpath.exists():
        val = json.loads(vpath.read_text(encoding="utf-8")).get("validation", {})
        print("\n=== STAGE 1 VALIDATION ===")
        print(json.dumps(val, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--herds", type=int, default=None)
    parser.add_argument("--pastures", type=int, default=None)
    parser.add_argument("--inspect", action="store_true", help="Inspect existing generated Stage 1 network without regenerating")
    args = parser.parse_args()
    if args.inspect:
        return inspect_existing(args.root)
    report = build_stage1_network(args.root, seed=args.seed, num_herds=args.herds, num_pastures=args.pastures)
    print(json.dumps(report, indent=2))
    return 0 if report["validation"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
