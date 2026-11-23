# Mini Transhumance Research Data Atlas (3D Spatiotemporal Edition)

An interactive 3D geospatial research-data prototype for exploring mountain transhumance systems in **Naryn Region, Kyrgyzstan**. The application combines public geographic and climate datasets, reproducible synthetic livestock-movement demonstrations, a spatial-temporal-semantic network explorer, and a provenance-aware RDF/SPARQL knowledge graph.

> **Research integrity notice:** Public source datasets, derived products, synthetic demonstration herds/pastures/corridors, and hypothetical climate perturbations are kept distinct. Synthetic records are not real GPS tracks, field interviews, observed migration routes, or official pasture-rights parcels. OpenStreetMap roads and tracks are geographic context only.

## Main application areas

- **3D Spatiotemporal Atlas:** CesiumJS globe, terrain and imagery layers, date timeline, synthetic herd movement, climate scenarios, and pasture suitability comparisons.
- **Network Analysis:** A layered spatial and pastoral mobility network with geographic and abstract graph views, filters, and feature inspection.
- **Linked Open Data & SPARQL:** RDF/Turtle and JSON-LD knowledge graph, domain ontology, SHACL shapes, and eight packaged SPARQL 1.1 queries.

See `docs/stage7_3d_atlas_architecture.md` for the system design, data provenance, and limitations.

## Run locally

Run these commands from the repository root. The repository already contains processed demonstration inputs and generated simulation/network outputs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Optional: acquire and process a local Copernicus DEM (requires network access and additional disk space):

```bash
python scripts/acquire_dem.py --root .
```

When a local Copernicus DEM is unavailable, the application explicitly labels its analytical terrain fallback as a synthetic relief proxy. Do not describe that fallback as measured elevation data.

To rebuild the RDF graph and run its packaged SPARQL validation pipeline:

```bash
python scripts/run_linked_data.py --root .
```

Run the automated tests with:

```bash
pytest -q
```

## Hosted deployment (Render)

The `deploy/render-container` branch contains a Docker-based deployment configuration for Render. A small Nginx reverse proxy exposes both the Streamlit interface and the CesiumJS asset server through the same public HTTPS origin. This is necessary because the local development setup uses a second HTTP server for Cesium assets, network payloads, map tiles, and timeline synchronisation.

To deploy:

1. Connect your GitHub account to Render and grant access to this private repository.
2. In the Render dashboard, choose **New → Blueprint**.
3. Select this repository and the `deploy/render-container` branch.
4. Review the service plan and environment variables from `render.yaml`, then apply the Blueprint.
5. Wait for the Docker build and health check to complete. The application health endpoint is `/healthz`; the Cesium asset service is routed through `/atlas/`.

The Blueprint defaults to Render's free web-service plan for an initial test. Free services have limited resources, may spin down after inactivity, and use an ephemeral filesystem. Cached Cesium/imagery/terrain assets are therefore treated as disposable cache data. If the service exceeds available memory during startup or interaction, inspect the Render logs and consider a larger instance or reducing the data loaded per session.

Optional API credentials such as `CESIUM_ION_ACCESS_TOKEN` or `GOOGLE_MAPS_API_KEY` should be added through Render's environment-variable settings only when needed. Do not commit secrets to Git.

## Main code locations

- `app.py` — Streamlit entry point.
- `frontend/cesium/index.html` — 3D atlas interface.
- `frontend/network/explorer.html` — network explorer interface.
- `src/atlas/` — atlas data access, Cesium bridge/server, terrain, and temporal state.
- `src/network/` — network loading, graph modelling, filtering, and explorer payloads.
- `src/linked_data/` — RDF graph construction and validation.
- `ontology/`, `queries/`, `vocab/` — semantic model, SHACL shapes, SPARQL queries, and controlled vocabulary.
- `data/processed/` — processed public/derived geospatial and climate inputs.
- `data/synthetic/` and `data/network/` — clearly labelled synthetic demonstration outputs and network artifacts.
- `tests/` — automated validation tests.
