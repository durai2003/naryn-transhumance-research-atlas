# Stage 1–2 source audit and acquisition report

**Project:** Mini Transhumance Research Data Atlas  
**Study area:** Naryn Region, Kyrgyzstan  
**Audit date:** 21 November 2026  
**Scope:** Stage 1 (source verification) and Stage 2 (dataset acquisition and schema inspection)

## Executive result

Provider documentation and catalogue metadata were reviewed for all seven specified sources. Several useful specifications and licences could be confirmed. **No primary raster or vector dataset bytes were successfully downloaded into this workspace.** The shell environment failed external DNS resolution (`curl: Could not resolve host`) for all tested providers; the browser/research retrieval layer also rejected or timed out several binary/API endpoints. I therefore have not reported a dataset as downloaded, have not claimed to inspect local raster/vector schemas, and have not created synthetic substitute records.

Stage 1 is complete to the extent of publicly retrievable documentation and metadata. Stage 2 is **partially completed**: source-described schemas and two Wikidata entity pages were inspected, but the required geospatial and time-series datasets remain unacquired. The scripts in `scripts/` are prepared for a network-enabled rerun.

## Acquisition status

| Source | Verification result | Format/schema evidence | Licence / terms | Local data status |
|---|---|---|---|---|
| geoBoundaries KGZ ADM1 | Current API metadata retrieved; 7 ADM1 units, represented year 2017. Current metadata says source: OpenStreetMap/Wambacher. | GeoJSON/TopoJSON/ZIP URLs advertised. Raw GeoJSON URL returned a Git LFS pointer, not feature content; Naryn feature fields and geometry were not inspected. | **Specific API response reports ODbL 1.0.** This differs from the generic geoBoundaries guidance that gbOpen is normally CC-BY 4.0; use the specific record pending provider confirmation. | Not downloaded. |
| Kyrgyz Data Cube | CAIAG documentation is accessible and lists pasture, snow, drought and temperature products. | WMS/WCS GetCapabilities requests could not be retrieved; documented product labels are not verified WMS layer names or WCS coverage IDs. | No specific licence found in the reviewed documentation; redistribution terms must be confirmed with CAIAG. | No capabilities document or product data downloaded. |
| FAO GLW4 2020 | FAO catalogue metadata identifies global sheep and goat density products for reference year 2020. | Catalogue profile: GeoTIFF/TIF, EPSG:4326, approx. 0.008333333° (~1 km at equator), no-data -9999, values head/km², Float64, nominal global dimensions 21600 × 43200. These are catalogue-described properties, not locally inspected raster metadata. | CC BY 4.0. | No sheep or goat raster downloaded. FAO catalogue opened as 403 in the web retrieval layer; file resources were not reached. |
| ESA WorldCover 2021 v200 | Public AWS access and ESA download routes documented. | 10 m COG, EPSG:4326, 3° × 3° tiles; Map and InputQuality layers. Map has 11 discrete class values. Full class schema appears below. | CC BY 4.0. | No tile downloaded. Exact region tile list is deferred until the actual Naryn geometry is available. |
| NASA POWER Daily | Official daily API documentation and parameters confirmed. | JSON/CSV/NetCDF/ASCII; `T2M` (2 m air temperature, °C) and `PRECTOTCORR` (corrected precipitation, mm/day). Meteorological source resolution 0.5° × 0.625°. | Retain POWER/source citations; the API documentation reviewed here does not by itself establish a standalone CC licence for this parameter response. | No daily response downloaded; selected point request could not be executed. |
| Geofabrik / OpenStreetMap | Current Kyrgyzstan extract listing verified. Advertised GeoPackage ZIP ~164 MB; PBF ~73.7 MB at time checked. | GeoPackage layer and field schema not inspected. Countrywide extract must be clipped to Naryn after geometry becomes available. | ODbL 1.0; public map attribution `© OpenStreetMap contributors`. | GPKG fetch timed out; PBF was rejected as binary by browser retrieval; shell download failed. |
| Wikidata | Entity pages inspected: Naryn Region `Q486375`; Naryn city `Q311946`; `Q27204290` redirects to `Q311946`. | Identifier labels/descriptions and city coordinate statement were checked on entity pages. A SPARQL result table/schema was not obtained. | CC0 for structured data. | Entity reference JSON recorded; SPARQL query not run. |

## Detailed schema notes

### 1. geoBoundaries

The current `gbOpen/KGZ/ADM1` metadata response reports boundary ID `KGZ-ADM1-38789647`, country-level API response for seven ADM1 units, represented year 2017, source `OpenStreetMap, Wambacher`, source data update date 19 January 2023 and build date 12 December 2023. The API response advertises GeoJSON and ZIP download URLs. The raw GeoJSON URL yielded a Git LFS pointer (object size 805,339 bytes); its actual feature content was not delivered. Consequently, **the Naryn feature selection, property names, geometry type, CRS and bounds have not been inspected**. Do not proceed by drawing or approximating the boundary. The acquisition script first resolves the advertised file, detects LFS pointers, and then selects only an exact Naryn feature from the returned attributes.

**Licence note:** The specific current API record states `Open Data Commons Open Database License 1.0`. Although general geoBoundaries documentation describes gbOpen as generally CC-BY 4.0 compatible, the specific release record has been retained in the register as ODbL pending manual clarification. Treat the more restrictive/source-specific record as controlling until verified.

### 2. Kyrgyz Data Cube

CAIAG documentation describes products including NDVI, Pasture Anomaly, Pasture Biomass (kg/ha), Pasture Trend, Snow Percentage, NDSI, NDDI, NDWI, VHI, VCI and Land Surface Temperature. Snow Percentage is described as the share of days with snow cover over a ten-day period. WMS and WCS service URLs are published. The service GetCapabilities documents were not obtained, so the service endpoint health, CRS, layer/coverage identifiers, dimensions, temporal coverage and access terms remain unverified. Do not treat product names in the descriptive page as service layer names.

### 3. FAO GLW4 2020

The specified catalogue item is the 1 km GLW4 2020 density dataset. The documented raster profile is EPSG:4326, Float64, 0.008333333 degrees, no-data `-9999`, global dimensions 21600 rows × 43200 columns, units `head/km²`. It is a modelled spatial distribution estimated using Random Forest downscaling of harmonised subnational census distributions; it is not a count of animals at a specific farm and contains no migration paths. Sheep and goat layers are to be downloaded separately, clipped only after the Naryn boundary is available, and kept at source resolution. The FAO catalogue page exposes per-species download resources but the direct raster links could not be retrieved in this run.

### 4. ESA WorldCover 2021

The 2021 product is version `v200`; its raster is a 10 m COG in EPSG:4326, delivered in 3° × 3° tiles. The `Map` band contains these class codes:

| Code | Class |
|---:|---|
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare / sparse vegetation |
| 70 | Snow and ice |
| 80 | Permanent water bodies |
| 90 | Herbaceous wetland |
| 95 | Mangroves |
| 100 | Moss and lichen |

The separate `InputQuality` layer is 3-band; it is not the land-cover classification. Class 30/20/80/70 may be useful as landscape context, but these classes do not encode official pasture tenure, seasonal use rights or verified pasture boundaries. ESA gives the required map attribution: `© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium`.

### 5. NASA POWER

The daily point API accepts a parameter list, community, longitude, latitude, start/end dates and output format. The planned first point is Naryn city (Wikidata Q311946; 41.4333333 N, 76.0 E), requesting `T2M` and `PRECTOTCORR` for 2015-01-01 through 2025-12-31 in JSON. POWER documents meteorological grid resolution of 0.5° × 0.625°; the point request samples the source grid rather than creating station-resolution observations. The API warns against repeatedly requesting the same relative grid location. Data should be cached and not over-requested. The selected request's actual data availability and JSON response schema could not be checked here.

### 6. Geofabrik / OpenStreetMap

Geofabrik listed a Kyrgyzstan-wide GeoPackage archive (about 164 MB) and PBF (about 73.7 MB). The public extract excludes OSM contributor usernames, user IDs and changeset IDs. The download is countrywide and should be clipped to Naryn only after the ADM1 boundary is downloaded. Required public attribution is `© OpenStreetMap contributors`; database reuse is governed by ODbL. Mapped roads and paths are contextual OSM features, not confirmed transhumance routes.

### 7. Wikidata

Two entity pages were inspected: Naryn Region `Q486375` and Naryn city `Q311946`. `Q27204290` resolves to `Q311946`. The city page gives a coordinate statement of 41°26′0″N, 76°0′0″E (41.4333333, 76.0 decimal degrees); it is suitable as a named demonstration point for the planned POWER call. The region item also has a coordinate statement, but that is not an administrative boundary geometry. The v2 SPARQL endpoint documented for the ongoing migration is `https://query-next.wikidata.org/sparql`; the endpoint query was not executed in this run. Do not mark the reference identifiers as SPARQL-verified until that query is run and retained with provenance. Wikidata structured data is CC0.

## Download and execution blockers

1. Shell requests to `geoboundaries.org`, `kyrgyzstan.sibelius-datacube.org`, `data.apps.fao.org`, `download.geofabrik.de` and `power.larc.nasa.gov` failed before HTTP with DNS resolution errors.
2. The geoBoundaries raw GitHub URL delivered an LFS pointer, not the geometry object. The separate container download attempt failed.
3. The FAO page returned 403 in the research retrieval layer; the exact sheep/goat resource download URLs were not exposed to this run.
4. The ESA page and public bucket instructions were accessible, but without the boundary the exact complete Naryn tile set was not selected, and external downloads were blocked.
5. The POWER documentation was accessible, but the actual JSON API request was not.
6. Geofabrik listing was accessible; the GeoPackage request timed out and PBF was not handled as binary by the research retrieval layer.
7. Wikidata pages are accessible; the SPARQL query endpoint was not.

These are **execution-environment access failures and unverified endpoint statuses**, not proof that all providers are down. No fabricated data, fake service responses, random climate values, guessed polygon or synthetic replacement was created.

## What is ready for the next run

- `metadata/dataset_register.csv` and `.json`: one row per source with provider, expected format/schema, licence, attribution, access state and limitations.
- `data/raw/source_metadata/source_audit_evidence.json`: structured source metadata/evidence (not primary dataset files).
- `data/raw/wikidata/verified_entity_references.json`: entity-page verified QIDs and coordinate evidence (not SPARQL results).
- `reports/download_log.csv`: each download/API attempt and outcome.
- `scripts/stage2_acquire.py`: network-enabled acquisition with retries, timeouts, provenance and explicit failure logging.
- `scripts/stage2_inspect.py`: local raster/vector/schema inventory once files exist.

## Stage 2 completion gate

Do not start spatial integration or clip rasters until the following files exist and pass inspection:

1. `data/raw/boundaries/naryn_adm1.geojson` (selected from the actual downloaded geoBoundaries feature collection; CRS and geometry valid).
2. Separate FAO sheep and goat density TIFs with CRS, shape, transform, dtype, units and nodata captured.
3. All ESA WorldCover 2021 Map COG tile(s) intersecting the actual Naryn geometry.
4. NASA POWER JSON for the named representative point, with response metadata and parameter units retained.
5. Geofabrik GeoPackage, with layers, geometry types and field names inventoried before filtering/clipping.
6. Kyrgyz Data Cube WMS/WCS GetCapabilities XML, if service access is available; otherwise keep it explicitly unavailable and document the fallback decision.
7. Wikidata SPARQL JSON result and query text with endpoint and retrieval timestamp.

No local primary dataset has been registered as downloaded yet. The register intentionally distinguishes source-described schema from actual file inspection.
