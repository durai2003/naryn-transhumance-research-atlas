# Stage 1 — Spatial, Temporal and Semantic Transhumance Network Architecture

## 1. Three-Layer Network Model (`networkx.MultiDiGraph`)

Stage 1 introduces a directed multigraph architecture (`src/network/`) that preserves all existing 3D CesiumJS, Copernicus DEM, NASA POWER, WorldCover, OSM, and RDF components while adding a formal geospatial network model ready for Stages 2–5:

- **Layer A — Physical Geographical Network (`layer_a_physical_nodes.geojson`, `layer_a_physical_edges.geojson`)**:
  Contains ONLY verifiable public features from `data/processed/locations.csv` (Wikidata `Q311946`) and `data/processed/vectors/osm_naryn_context.gpkg` (settlements, rivers/water points, and physical road access context). Every Layer A entity has `is_synthetic=false`, `evidence_status='verified_public'`. OSM roads are never labelled as historical transhumance corridors.
- **Layer B — Pastoral Mobility Network (`layer_b_pastoral_nodes.geojson`, `layer_b_pastoral_edges.geojson`, `herds.csv`, `movement_events.csv`, `herd_positions.csv`/`.parquet`, `edge_utilisation.csv`)**:
  Represents the seasonal grazing system across 8 static node categories (`winter_pasture`, `spring_pasture`, `summer_pasture`, `autumn_pasture`, `settlement`, `water_point`, `mountain_pass`, `resting_point`) and 6 edge categories (`SEASONAL_MOVEMENT`, `ACCESS_ROUTE`, `LEAST_COST_CORRIDOR`, `WATER_ACCESS`, `SETTLEMENT_CONNECTION`, `ALTERNATIVE_ROUTE`).
  - **Temporal modelling distinction**: Transient `herd_location` observations are stored separately in `herd_positions.csv` (and `.parquet` when `pyarrow` is installed), linked via `herd_id`, `current_node_id`, and `current_edge_id` — never injected as thousands of permanent nodes in the static graph.
  - **Derived edge utilisation**: `edge_utilisation.csv` and the edge attributes in `transhumance_multigraph.json`/`.graphml` are aggregated strictly from `movement_events.csv`. Any unused edge has exact utilisation `0`.
- **Layer C — Linked Knowledge Graph Bridge (`layer_c_semantic_links.json`)**:
  Connects Layer A and Layer B stable IDs (`PUB-SET-*`, `DER-PAS-*`/`SYN-PAS-*`, `SYN-REST-*`, `H-*`) to prototype URIs, Wikidata `owl:sameAs` references (`Q486375`, `Q311946`), and PROV-O provenance metadata, ready for Stage 2/5 RDF and Graph Explorer integration.

## 2. Elevation & Environmental Mask Honesty

- When local Copernicus DEM (`data/processed/rasters/dem/naryn_dem_glo30_clip.tif`), ESA WorldCover (`data/processed/rasters/worldcover/*.tif`), and OSM (`data/processed/vectors/osm_naryn_context.gpkg`) are present on disk, candidate pasture selection and least-cost routing (`EPSG:32643`) actively apply DEM slope/altitude masks, WorldCover grassland/shrubland preferences, and OSM road/river proximity costs, recording `elevation_status='dem_derived'`.
- If a local DEM is absent in a minimal test environment, `elevation_m`, `elevation_gain_m`, `average_slope`, and `maximum_slope` are explicitly stored as `null` with `elevation_status='unavailable_no_local_dem'` rather than inventing fake elevations.

## 3. Reproducibility & Regeneration

```bash
python3 scripts/build_transhumance_network.py --root . --seed 42
python3 scripts/build_transhumance_network.py --root . --inspect
python3 -m pytest -q tests/test_stage1_network.py
```
