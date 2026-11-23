# Mini Transhumance Research Data Atlas (3D Spatiotemporal Edition)

An interactive 3D geospatial research-data prototype for exploring mountain transhumance systems in **Naryn Region, Kyrgyzstan**. This upgrade extends the existing Stages 1–6 repository (data acquisition, Stage 3 geospatial/climate pipeline, Stage 4 RDF/JSON-LD/SPARQL knowledge graph, and Streamlit interface) into a full **CesiumJS 3D globe** with animated synthetic herds, terrain-aware least-cost corridors, and climate-responsive pasture suitability scenarios.

> **Research Integrity Notice:** Public source datasets (`public`), derived products (`derived`), synthetic demonstration herds/pastures/corridors (`synthetic`), and hypothetical climate perturbations (`simulated_scenario`) are stored and labelled separately. Synthetic records are not real GPS tracks, field interviews, or official pasture-rights parcels.

## Quick Start (macOS / Linux)

Run from your existing `transhumance-atlas/` repository root after extracting this upgrade archive:

```bash
# 1. Create a backup branch & activate your existing Python virtual environment
git checkout -b feat/3d-cesium-spatiotemporal-atlas
source .venv/bin/activate

# 2. Install upgraded dependencies
python -m pip install -r requirements.txt

# 3. Configure optional tokens (token-free Esri 3D terrain + imagery works out of the box)
cp .env.example .env

# 4. Download & preprocess Copernicus DEM GLO-30 (30m, automatic GLO-90 fallback)
python scripts/acquire_dem.py --root .

# 5. Regenerate the reproducible synthetic simulation using your local DEM & WorldCover
python scripts/generate_herd_simulation.py --root . --seed 42

# 6. Rebuild RDF/JSON-LD graph & execute all 7 SPARQL queries + fixed SHACL validation
python scripts/run_linked_data.py --root .

# 7. Run the automated test suite
pytest -q

# 8. Launch the 3D Spatiotemporal Atlas
python -m streamlit run app.py
```

## Key Upgraded Components

- **3D Globe & Terrain:** `frontend/cesium/index.html`, `src/atlas/cesium_bridge.py`, `src/atlas/terrain.py`, `scripts/acquire_dem.py`
- **Synthetic Simulation & Scenario Engine:** `config/simulation.yaml`, `config/suitability_weights.yaml`, `src/atlas/simulation/*`, `scripts/generate_herd_simulation.py`
- **Synchronized Temporal Atlas:** `app.py`, `src/atlas/temporal.py`, `src/atlas/data_access.py`
- **Extended Linked Open Data & Fixed SHACL:** `ontology/transhumance.ttl`, `queries/shapes/linked_data_shapes.ttl`, `src/linked_data/build_graph.py`, `queries/06_synthetic_herds_and_routes.rq`, `queries/07_scenario_suitability_and_events.rq`
- **Validation & Tests:** `src/validation/simulation_checks.py`, `src/validation/trajectory_checks.py`, `src/validation/climate_checks.py`, `tests/test_3d_simulation_suite.py`

See `docs/stage7_3d_atlas_architecture.md` for full methodology, data provenance, and limitations.
