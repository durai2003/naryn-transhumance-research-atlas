# Stage 3 execution report

Generated (UTC): 2026-11-21T12:08:07.934241+00:00

## Run summary

- Verified Wikidata locations standardised: 1
- Daily NASA POWER observations: 4018
- Monthly climate summaries: 132
- Seasonal climate summaries: 44
- Processed rasters: 4
- Table/register checks passing: True
- Actual boundary available for spatial processing: True
- Synthetic source replacements created: no

## Blocked operations

- **data/raw/glw4** — No local FAO GLW4 primary raster; no processed raster can be generated
- **/Users/duraisrs/Desktop/transhumance_atlas/transhumance-atlas/data/raw/osm/kyrgyzstan-latest-free.gpkg** — Rejected invalid GeoPackage candidate: candidate path is not a regular file

## Interpretation

The pipeline ran against the files present in the workspace. It standardised 1 verified location reference(s), processed 4018 NASA POWER daily observation(s), created 132 monthly and 44 seasonal climate summary row(s), and produced 4 clipped raster output(s). The authoritative Naryn boundary was available. The table checks passed: True. Consult the blocked_operations list and transformation log to distinguish completed processing from unavailable inputs. No synthetic public-data replacements were created.

For field-level details, see `stage3_data_quality_report.json`, `stage3_transformation_log.csv` and `stage3_processing_manifest.json`.
