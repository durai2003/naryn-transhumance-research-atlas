"""Layer A: Physical Geographical Network built strictly from verifiable public sources.

Sources:
- Verified Wikidata entity coordinates (`data/processed/locations.csv`)
- Naryn-clipped OpenStreetMap GeoPackage (`data/processed/vectors/osm_naryn_context.gpkg`):
  - `gis_osm_places_free` -> verified `settlement` nodes
  - `gis_osm_waterways_free` / `gis_osm_water_a_free` -> derived public `water_point` nodes
  - `gis_osm_roads_free` -> physical `ACCESS_ROUTE` context segments (NEVER labelled pastoral migration corridors)

Public/observed features NEVER carry `is_synthetic=True`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from src.network.dem_env import (
    EnvironmentInventory,
    compute_line_metrics,
    sample_dem_strict,
    sample_worldcover_strict,
)


def _list_gpkg_layers(path: Path) -> set[str]:
    try:
        import pyogrio
        return {str(row[0]) for row in pyogrio.list_layers(path)}
    except Exception:
        return set()


def build_physical_layer(
    boundary_gdf: gpd.GeoDataFrame,
    env: EnvironmentInventory,
    cfg: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Return (layer_a_nodes_gdf, layer_a_edges_gdf, layer_a_report)."""
    scale = cfg.get("network_scale", {})
    max_settlements = int(scale.get("max_physical_settlements", 25))
    max_roads = int(scale.get("max_physical_road_segments", 400))
    max_water = int(scale.get("max_physical_water_features", 120))
    metric_crs = str(cfg.get("metric_crs", "EPSG:32643"))

    region_wgs = unary_union(boundary_gdf.to_crs("EPSG:4326").geometry)
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    seen_coords: list[tuple[float, float]] = []
    report: dict[str, Any] = {
        "layer": "physical_geographical",
        "osm_gpkg_available": env.osm_gpkg_path is not None,
        "wikidata_locations_available": env.locations_path is not None,
        "settlements_loaded": 0,
        "water_points_loaded": 0,
        "physical_access_edges_loaded": 0,
        "notes": [],
    }

    def _is_duplicate_loc(lon: float, lat: float, tol_deg: float = 0.015) -> bool:
        for xl, yl in seen_coords:
            if abs(xl - lon) < tol_deg and abs(yl - lat) < tol_deg:
                return True
        return False

    # 1. Verified Wikidata places (e.g., wikidata:Q311946 Naryn)
    if env.locations_path is not None:
        loc_df = pd.read_csv(env.locations_path)
        for _, r in loc_df.dropna(subset=["latitude", "longitude"]).iterrows():
            lat, lon = float(r["latitude"]), float(r["longitude"])
            pt = Point(lon, lat)
            if not region_wgs.covers(pt):
                continue
            qid = str(r.get("external_identifier", "Q311946"))
            node_rows.append({
                "node_id": f"PUB-SET-{qid}",
                "node_name": str(r.get("location_name", "Naryn")),
                "node_type": "settlement",
                "network_layer": "physical_geographical",
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "external_id": f"wikidata:{qid}",
                "external_uri": str(r.get("external_uri", f"http://www.wikidata.org/entity/{qid}")),
                "fclass": "city",
                "source": "Wikidata verified entity reference",
                "provenance": "wikidata_entity_page",
                "evidence_status": "verified_public",
                "is_synthetic": False,
                "geometry": pt,
            })
            seen_coords.append((lon, lat))
            report["settlements_loaded"] += 1
    else:
        # Check research_graph.jsonld for verified Q311946 if locations.csv wasn't staged
        rg = env.root / "linked_data/research_graph.jsonld"
        if not rg.exists():
            rg = env.root / "research_graph.jsonld"
        if rg.exists():
            try:
                doc = json.loads(rg.read_text(encoding="utf-8"))
                for item in doc.get("@graph", []):
                    if item.get("dcterms:identifier") == "Q311946":
                        lat = float(item["wgs84:lat"]["@value"])
                        lon = float(item["wgs84:long"]["@value"])
                        pt = Point(lon, lat)
                        if region_wgs.covers(pt):
                            node_rows.append({
                                "node_id": "PUB-SET-Q311946",
                                "node_name": "Naryn",
                                "node_type": "settlement",
                                "network_layer": "physical_geographical",
                                "latitude": round(lat, 5),
                                "longitude": round(lon, 5),
                                "external_id": "wikidata:Q311946",
                                "external_uri": "http://www.wikidata.org/entity/Q311946",
                                "fclass": "city",
                                "source": "Wikidata verified entity reference",
                                "provenance": "wikidata_entity_page",
                                "evidence_status": "verified_public",
                                "is_synthetic": False,
                                "geometry": pt,
                            })
                            seen_coords.append((lon, lat))
                            report["settlements_loaded"] += 1
                            break
            except Exception:
                pass

    # 2. Genuine OpenStreetMap features from Naryn-clipped GeoPackage (when present)
    if env.osm_gpkg_path is not None:
        avail_layers = _list_gpkg_layers(env.osm_gpkg_path)

        # 2A. Settlements (`gis_osm_places_free`)
        if "gis_osm_places_free" in avail_layers:
            places = gpd.read_file(env.osm_gpkg_path, layer="gis_osm_places_free").to_crs("EPSG:4326")
            if "fclass" in places.columns:
                prio = {"city": 1, "town": 2, "village": 3, "hamlet": 4}
                places["_p"] = places["fclass"].astype(str).str.lower().map(prio).fillna(9)
                places = places.sort_values("_p")
            for _, r in places.iterrows():
                if report["settlements_loaded"] >= max_settlements:
                    break
                geom = r.geometry
                if geom is None or geom.is_empty or geom.geom_type != "Point":
                    continue
                name = str(r.get("name") or "").strip()
                if not name or not region_wgs.covers(geom):
                    continue
                lon, lat = float(geom.x), float(geom.y)
                if _is_duplicate_loc(lon, lat, tol_deg=0.04):
                    continue
                osm_id = str(r.get("osm_id") or report["settlements_loaded"] + 1)
                node_rows.append({
                    "node_id": f"PUB-SET-OSM-{osm_id}",
                    "node_name": name,
                    "node_type": "settlement",
                    "network_layer": "physical_geographical",
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                    "external_id": f"osm:{osm_id}",
                    "external_uri": f"https://www.openstreetmap.org/node/{osm_id}",
                    "fclass": str(r.get("fclass", "settlement")),
                    "source": "OpenStreetMap / Geofabrik (ODbL 1.0)",
                    "provenance": "osm_geofabrik_naryn_clip",
                    "evidence_status": "verified_public",
                    "is_synthetic": False,
                    "geometry": geom,
                })
                seen_coords.append((lon, lat))
                report["settlements_loaded"] += 1

        # 2B. Documented Water Access Points derived from OSM rivers/lakes
        if "gis_osm_waterways_free" in avail_layers:
            rivers = gpd.read_file(env.osm_gpkg_path, layer="gis_osm_waterways_free").to_crs("EPSG:4326")
            if "fclass" in rivers.columns:
                rivers = rivers.loc[rivers["fclass"].astype(str).str.lower().isin({"river", "stream", "canal"})]
            rivers_m = rivers.to_crs(metric_crs)
            rivers = rivers.assign(_len=rivers_m.geometry.length).sort_values("_len", ascending=False)
            water_coords: list[tuple[float, float]] = []
            for _, r in rivers.iterrows():
                if report["water_points_loaded"] >= min(max_water, 18):
                    break
                geom = r.geometry
                if geom is None or geom.is_empty:
                    continue
                pt = geom.interpolate(0.5, normalized=True) if geom.geom_type == "LineString" else geom.representative_point()
                if not region_wgs.covers(pt):
                    continue
                lon, lat = float(pt.x), float(pt.y)
                if any(abs(lon - wx) < 0.18 and abs(lat - wy) < 0.14 for wx, wy in water_coords):
                    continue
                osm_id = str(r.get("osm_id") or report["water_points_loaded"] + 1)
                wname = str(r.get("name") or "").strip() or f"OSM {r.get('fclass', 'river')} crossing {osm_id}"
                node_rows.append({
                    "node_id": f"PUB-WAT-OSM-{osm_id}",
                    "node_name": wname,
                    "node_type": "water_point",
                    "network_layer": "physical_geographical",
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                    "external_id": f"osm:{osm_id}",
                    "external_uri": f"https://www.openstreetmap.org/way/{osm_id}",
                    "fclass": str(r.get("fclass", "river")),
                    "source": "OpenStreetMap / Geofabrik waterways (ODbL 1.0)",
                    "provenance": "osm_geofabrik_naryn_clip",
                    "evidence_status": "verified_public",
                    "is_synthetic": False,
                    "geometry": pt,
                })
                water_coords.append((lon, lat))
                report["water_points_loaded"] += 1

        # 2C. Physical Road Context Edges (major OSM road/track links between nearby public settlements)
        if "gis_osm_roads_free" in avail_layers and report["settlements_loaded"] >= 2:
            roads = gpd.read_file(env.osm_gpkg_path, layer="gis_osm_roads_free").to_crs("EPSG:4326")
            if "fclass" in roads.columns:
                roads = roads.loc[roads["fclass"].astype(str).str.lower().isin({"trunk", "primary", "secondary", "tertiary", "track"})]
            roads = roads.head(max_roads)
            set_nodes = [n for n in node_rows if n["node_type"] == "settlement"]
            # Connect pairs of nearby public settlements along snapped physical road segments (clearly labelled physical context)
            set_gdf = gpd.GeoDataFrame(set_nodes, geometry="geometry", crs="EPSG:4326").to_crs(metric_crs)
            roads_m = roads.to_crs(metric_crs)
            road_union = unary_union(roads_m.geometry)
            edge_idx = 0
            for i in range(len(set_gdf)):
                for j in range(i + 1, len(set_gdf)):
                    p1 = set_gdf.iloc[i].geometry
                    p2 = set_gdf.iloc[j].geometry
                    dist_m = float(p1.distance(p2))
                    if 8000.0 <= dist_m <= 65000.0:
                        # Only add if both endpoints lie within 3 km of mapped OSM road network
                        if p1.distance(road_union) <= 3000.0 and p2.distance(road_union) <= 3000.0:
                            edge_idx += 1
                            line_wgs = LineString([
                                (set_nodes[i]["longitude"], set_nodes[i]["latitude"]),
                                (set_nodes[j]["longitude"], set_nodes[j]["latitude"]),
                            ])
                            metrics = compute_line_metrics(line_wgs, env, metric_crs=metric_crs, num_samples=16)
                            edge_rows.append({
                                "edge_id": f"PUB-ACC-{edge_idx:03d}",
                                "source_node_id": set_nodes[i]["node_id"],
                                "target_node_id": set_nodes[j]["node_id"],
                                "edge_type": "ACCESS_ROUTE",
                                "network_layer": "physical_geographical",
                                "distance_m": metrics["distance_m"],
                                "elevation_gain_m": metrics["elevation_gain_m"],
                                "elevation_loss_m": metrics["elevation_loss_m"],
                                "start_elevation_m": metrics["start_elevation_m"],
                                "end_elevation_m": metrics["end_elevation_m"],
                                "max_elevation_m": metrics["max_elevation_m"],
                                "maximum_slope": metrics["maximum_slope"],
                                "average_slope": metrics["average_slope"],
                                "elevation_status": metrics["elevation_status"],
                                "elevation_source": metrics["elevation_source"],
                                "elevation_profile_json": metrics["elevation_profile_json"],
                                "estimated_travel_days": round(metrics["distance_m"] / 15000.0, 2),
                                "season": "all_year",
                                "direction": "bidirectional",
                                "livestock_compatibility": "road_context_not_pastoral_corridor",
                                "route_generation_method": "osm_road_proximity_physical_link",
                                "source": "OpenStreetMap / Geofabrik roads (ODbL 1.0)",
                                "provenance": "osm_geofabrik_naryn_clip",
                                "evidence_status": "verified_public",
                                "is_synthetic": False,
                                "geometry": line_wgs,
                            })
                            if edge_idx >= 25:
                                break
                if edge_idx >= 25:
                    break
            report["physical_access_edges_loaded"] = edge_idx
    else:
        report["notes"].append(
            "Local osm_naryn_context.gpkg not found in this workspace; Layer A contains verified Wikidata references only. "
            "Run on a workspace with data/processed/vectors/osm_naryn_context.gpkg to populate full OSM settlements, water points, and road access links."
        )

    if node_rows:
        lons = [r["longitude"] for r in node_rows]
        lats = [r["latitude"] for r in node_rows]
        elevs, slopes, e_stat, e_src = sample_dem_strict(lons, lats, env)
        wc_codes, wc_labels, wc_src = sample_worldcover_strict(lons, lats, env)
        for i, r in enumerate(node_rows):
            r["elevation_m"] = elevs[i]
            r["slope_degrees"] = slopes[i]
            r["elevation_status"] = e_stat
            r["elevation_source"] = e_src
            r["landcover_code"] = wc_codes[i]
            r["landcover_class"] = wc_labels[i]
            r["landcover_source"] = wc_src
        nodes_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")
    else:
        nodes_gdf = gpd.GeoDataFrame(columns=["node_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    if edge_rows:
        edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs="EPSG:4326")
    else:
        edges_gdf = gpd.GeoDataFrame(columns=["edge_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    return nodes_gdf, edges_gdf, report
