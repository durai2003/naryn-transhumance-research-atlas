# 3D Spatiotemporal Atlas Architecture, Methodology & Limitations

## 1. Summary of Upgrade (2D Folium → 3D CesiumJS)

The previous Stage 6 application rendered Naryn Region on a flat 2D Folium map (`st_folium`) with 8 straight radial schematic routes, a static season dropdown, and unstyled OSM layers. This upgrade preserves all Stage 1–5 data pipelines, metadata catalogues, and Linked Open Data assets while introducing:

1. **Genuine 3D Globe & Mountain Terrain (`frontend/cesium/index.html`)**: Full-Earth CesiumJS viewer with global imagery (Esri World Imagery + reference labels, Esri World Topo, OpenTopoMap, OSM), tiered global 3D terrain (Cesium World Terrain with vertex normals when `CESIUM_ION_ACCESS_TOKEN` is set; automatic token-free Esri `WorldElevation3D` fallback; explicit warning banner if offline — never a silent ellipsoid fallback), adjustable relief exaggeration (`1.0×–3.0×`), oblique 3D mountain camera presets, and click-to-inspect elevation.
2. **Local Copernicus DEM Preprocessing (`scripts/acquire_dem.py`, `src/atlas/terrain.py`)**: Downloads 1×1° Copernicus DEM GLO-30 tiles (30 m, `EPSG:4326`, EGM2008) from AWS Open Data with automatic GLO-90 (90 m) fallback, mosaics and clips to Naryn ADM1 (+0.15° buffer), and derives slope and hillshade rasters.
3. **Photogrammetry Honesty**: Google Photorealistic 3D Tiles is exposed only as an optional toggle because continuous photogrammetric 3D mesh coverage is generally unavailable across rural Naryn mountain valleys. DEM-driven 3D terrain draped with high-resolution satellite imagery is the scientifically accurate primary mode.
4. **Reproducible Synthetic Transhumance Simulation (`scripts/generate_herd_simulation.py`)**:
   - **18 candidate grazing locations** (`DEMO-PASTURE-001`..`018`) strictly contained within real Naryn ADM1 across valley wintering, foothill transition, and high-alpine summer bands.
   - **21 terrain-aware least-cost corridors** (`SYN-CORRIDOR-001`..`021`) computed via Dijkstra shortest path on a projected `EPSG:32643` slope/elevation-barrier cost grid, with full route elevation profiles.
   - **48 synthetic herd groups** (`SYN-HERD-001`..`048`) across 5 species compositions (`sheep`, `goat`, `cattle`, `horse`, `mixed`) with staggered departure/return calendars, speeds, and rest days.
   - **52,704 continuous daily trajectory records** (17,568 per scenario × 3 scenarios, `max_daily_step_km = 17.68`, 0 teleports) and **1,152 discrete movement events**.
5. **Climate-Responsive Rule-Based Suitability & Scenario Engine (`config/suitability_weights.yaml`)**:
   - `SCENARIO_BASELINE`: Exact unmodified NASA POWER daily `T2M` and `PRECTOTCORR` (`0.5° × 0.625°` MERRA-2 grid at Naryn city `Q311946`).
   - `SCENARIO_WARMER_DRIER`: Configurable `+2.8 °C` summer warming and `45%` summer precipitation reduction.
   - `SCENARIO_EARLY_SNOW`: Configurable `-3.8 °C` late-summer/autumn cooling and `+60%` autumn precipitation, shifting alpine descent earlier by 22 days and rerouting herds to sheltered foothill refuges.
6. **Extended Linked Open Data & Fixed SHACL Validation**:
   - Fixed the 10 Stage 4 SHACL failures in `queries/shapes/linked_data_shapes.ttl` (`dcterms:title` now accepts both `xsd:string` and `rdf:langString`).
   - Extended `ontology/transhumance.ttl`, `src/linked_data/build_graph.py`, and `queries/06–07_*.rq` to represent `ta:HerdGroup`, `ta:MovementEvent`, `ta:SimulationScenario`, and synthetic candidate pastures/corridors with strict provenance separation.

## 2. Explicit Origin Classification & Limitations

| Origin Class | Examples | Interpretation Rule |
|---|---|---|
| `public` | geoBoundaries Naryn ADM1, NASA POWER daily, ESA WorldCover 2021, OSM roads/rivers/settlements, Wikidata QIDs | Real external data; OSM roads are context only (not migration routes); WorldCover grassland is not grazing tenure; NASA POWER is a coarse grid (not a weather station). |
| `derived` | Copernicus DEM slope/hillshade, route elevation profiles, monthly/seasonal climate summaries | Deterministically computed from public datasets. |
| `synthetic` | `DEMO-PASTURE-*`, `SYN-CORRIDOR-*`, `SYN-HERD-*`, `herd_trajectories.csv`, fictional interviews/media | Reproducible demonstration data (`is_synthetic=true`, `provenance=synthetic_demo`, `seed=42`); NOT field observations or GPS tracks. |
| `simulated_scenario` | Warmer/Drier and Early Snow climate perturbations & rerouted trajectories | Illustrative rule-based model scenarios; NOT forecasts of real herder behaviour. |
