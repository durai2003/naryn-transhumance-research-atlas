# Stage 6 implementation report

## Implemented

- `app.py`: Streamlit application with Folium map, climate charts, WorldCover class summary, catalogue, synthetic research metadata, RDF graph explorer and provenance panel.
- `src/atlas/data_access.py`: safe loaders for Stage 3 outputs, verified pastures, processed OSM only, graph and provenance-validated synthetic data.
- `src/atlas/overlays.py`: WGS84 raster image overlays, categorical WorldCover palette, continuous raster display palette, exact native-pixel WorldCover class summary, and sampled KDC raster profiles.
- `scripts/prepare_atlas_demo_data.py`: creates eight clearly schematic synthetic route lines constrained by the actual Naryn ADM1 geometry, twelve metadata-only fictional interview rows and three fictional multimedia metadata rows.
- `requirements-stage6.txt` and `requirements.txt`: frontend dependencies.
- `docs/stage6_atlas.md`: data path and layer policy, setup, interpretation, attribution and interview notes.

## Provenance separation

Public data uses source-specific layer names. Derived products (monthly/seasonal climate and class-share summaries) are identified as derived. Synthetic routes are separate GeoJSON features with `is_synthetic=true`, `provenance=synthetic_demo`, a non-empty disclaimer, and dashed purple styling. Fictional interview/media metadata are separate CSVs and are metadata-only. No real transcript, participant, quote, photo, video or audio is created.

The app uses only `data/processed/vectors/osm_naryn_context.gpkg` for OSM map rendering. It never presents the countrywide raw Geofabrik extract as a Naryn clip. WorldCover is described as land cover, not pasture parcels. OSM lines are mapped context, not migration routes. FAO density is not animal movement evidence.

## Automated checks

The project test suite in this workspace: **40 passed** (includes the existing Stage 1–4 tests plus Stage 6 tests). `app.py` and new Python modules pass Python syntax compilation.

Test coverage includes: synthetic geometry generation requires an actual boundary; generated routes are explicitly flagged; fictional metadata carries provenance; invalid synthetic records are excluded; map centre prefers real boundary; fallback to verified Naryn city point; WorldCover RGBA/nodata and exact class counts; missing inputs stay empty; and countrywide raw OSM is not displayed.

## Runtime verification limitation

Streamlit and streamlit-folium were not installed in this development container: the package index could not be reached because DNS resolution failed. Therefore the Streamlit UI itself was not launched here. The app was syntax-compiled, and its core data loaders, geometry generation and raster helpers were unit-tested. Install `requirements-stage6.txt` on the user's Mac using the same Python 3.13 interpreter used by Stages 1–5, then launch with `python -m streamlit run app.py`.

The development container does not contain the user's full Stage 3 raw/processed files. The Mac run reported 4,018 daily climate records, 132 monthly summaries, 44 seasonal summaries and four processed rasters. The atlas reads those files dynamically from the documented project paths; it does not copy or replace those source assets in this patch.

## Outstanding data dependencies

- FAO GLW4 sheep/goat density rasters were not confirmed downloaded in the latest user-reported Stage 3 run; the map does not display livestock density without actual processed rasters.
- No independently sourced pasture polygons are assumed; the pasture layer appears only if `data/processed/pastures.geojson` has source URLs and non-synthetic records.
- Kyrgyz Data Cube capability XML may be listed, but the app does not automatically render or redistribute WMS/WCS products before terms and layer semantics are human-verified.
- No real interviews or actual seasonal migration routes are included.
