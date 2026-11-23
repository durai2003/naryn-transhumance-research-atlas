# Stage 6 map-rendering hotfix

## Symptoms fixed

1. ESA WorldCover and raster overlays emitted `Object of type Image is not JSON serializable`.
2. `streamlit_folium` failed while computing bounds with `KeyError: 'coordinates'`.

## Root causes

- A Pillow `Image` instance was passed directly to `folium.raster_layers.ImageOverlay`. The installed Branca version did not treat that Pillow object as an array or file-like object and attempted JSON serialization.
- At least one GeoJSON geometry added to the map is a `GeometryCollection` (or otherwise lacks normal coordinate arrays). Branca's `get_bounds` walker expects normal GeoJSON geometries with a `coordinates` member.

## Fixes

- `src/atlas/overlays.py` adds `image_to_data_uri`, converting the RGBA Pillow image to an embedded PNG data URI.
- `app.py` passes the data URI to all WorldCover, FAO and Kyrgyz Data Cube `ImageOverlay` elements.
- `src/atlas/data_access.py` adds `folium_compatible_geodataframe`. It recursively separates `GeometryCollection` members into standard Point/LineString/Polygon geometries, and excludes null, empty, unsupported and non-finite geometries.
- Boundary, verified pasture, synthetic route and clipped OSM layers pass through this geometry sanitizer before GeoJSON serialization.
- OSM rendering continues to use only the processed Naryn-clipped GeoPackage, not the countrywide raw extract.

## Verification

- Raster PNG data URI encoding and Folium map rendering: tested with a temporary raster fixture.
- GeometryCollection splitting and map bounds computation: tested with a temporary geometry fixture.
- Full development-workspace suite: 42 passed.
- Python syntax compilation: passed.

The complete Streamlit application must still be re-run in the user's local environment against their actual downloaded Naryn assets; this development workspace does not contain those assets.
