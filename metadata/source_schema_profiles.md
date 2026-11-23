# Source schema profiles — Stage 1 / Stage 2

This file separates **provider-described schema** from **local-file inspection**. No primary raster or vector dataset was downloaded in the audit runtime; therefore the values below are catalogue/documentation profiles, not claims about bytes present in `data/raw/`.

| Source | Provider-described structure | What still requires local inspection |
|---|---|---|
| geoBoundaries KGZ ADM1 | GeoJSON feature collection / TopoJSON / ZIP advertised by current API. Seven ADM1 records reported in API metadata. | Feature property names, selected Naryn feature, geometry type, validity, CRS, bounds, vertex count. |
| Kyrgyz Data Cube | OGC WMS 1.3.0 and WCS 1.0.0 endpoints are published. Product concepts documented: NDVI, pasture anomaly/biomass/trend, snow percentage/NDSI, drought NDDI, NDWI, VHI, VCI and LST. | `GetCapabilities` XML, named layers/coverages, formats, CRS, dimensions, temporal axis, access constraints and licences. |
| FAO GLW4 2020 | GeoTIFF/TIF; 21600 rows x 43200 columns; EPSG:4326; cell size 0.008333333°; Float64; nodata -9999; values in head/km². Separate sheep and goat density assets are the target. | Each downloaded TIF's width/height, band count, block layout, dtype, nodata, transform, bounds, compression, overviews, actual unit tags and checksums. |
| ESA WorldCover 2021 v200 | COG; EPSG:4326; 10 m nominal pixel size; 3° x 3° tile layout. `Map` class band plus separate 3-band `InputQuality`. Class values: 10,20,30,40,50,60,70,80,90,95,100. | Exact Naryn-intersecting tile list; each tile's actual transform, dimensions, dtype, nodata, class frequencies, coverage and checksum. |
| NASA POWER daily | Point endpoint supports JSON/CSV/NetCDF/ASCII. Planned parameters: `T2M` °C and `PRECTOTCORR` mm/day; daily records. | Response JSON structure and metadata, actual coverage/completeness, missing-value codes, grid cell / source metadata, API query timestamp and units as encoded in response. |
| Geofabrik OSM | Country extract advertised as GeoPackage ZIP or PBF. Current public extract omits contributor usernames, user IDs and changeset IDs. | GeoPackage layers, field names, geometry types, layer CRS, counts, bounds, tag distributions, extract timestamp, plus Naryn clip results. |
| Wikidata | SPARQL JSON is intended output. Page-verified items: Naryn Region `Q486375`, Naryn city `Q311946`; `Q27204290` redirects to `Q311946`. | Retained SPARQL query and JSON result, property bindings, references/qualifiers, retrieval timestamp, endpoint version. |

## Required profile checks before acceptance

- **Raster:** verify CRS, dimensions, pixel size, transform, bounds, nodata, dtype, band count, units and data range using Rasterio; compute checksum; confirm tile intersects actual Naryn boundary.
- **Vector:** verify CRS is defined (do not guess one), geometry types, geometry validity, field names/types, feature count, coordinate bounds and exact Naryn feature match using GeoPandas/OGR.
- **Time series:** verify parameter keys, metadata headers, units, dates, coordinate/grid description, missing-value codes and duplicate dates; cache exact response and request URL.
- **Linked data:** retain query text, endpoint, HTTP status, response format, query timestamp and exact QIDs. An entity page alone is not a SPARQL execution record.
