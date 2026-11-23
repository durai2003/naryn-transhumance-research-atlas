# Stage 4 — Linked Open Data implementation

## Purpose

Stage 4 converts the project's catalogue and processed Stage 3 outputs into a provenance-aware RDF knowledge graph. It does not infer missing research entities. In particular, the graph builder does not turn WorldCover grassland into a pasture polygon, OSM paths into migration routes, or livestock-density values into animal movements.

## Files

- `ontology/transhumance.ttl` — application ontology: classes, properties, domains/ranges, and modelling notes.
- `vocab/transhumance-vocabulary.ttl` — seven English SKOS concepts requested for the prototype.
- `src/linked_data/build_graph.py` — reads the local dataset register, verified Wikidata entity file and Stage 3 outputs; builds RDF and JSON-LD.
- `linked_data/research_graph.ttl` — RDF graph in Turtle.
- `linked_data/research_graph.jsonld` — same graph in JSON-LD with a compact namespace context.
- `queries/*.rq` — five separate executable SPARQL 1.1 SELECT queries.
- `queries/example_queries.rq` — index of the five executable query files.
- `queries/shapes/linked_data_shapes.ttl` — optional SHACL shapes for catalogue and synthetic-record checks.
- `scripts/run_linked_data.py` — rebuilds the graph, runs all five queries and writes a machine-readable result report.
- `reports/stage4_linked_data_report.json` — graph counts, provenance/availability status and graph quality checks.
- `reports/stage4_sparql_query_results.json` — query row counts and result samples.

## Namespace design

The application namespace is `https://example.org/transhumance-atlas/`. This is a systematic placeholder namespace for the portfolio prototype. It is **not registered, owned or operated as a persistent identifier service**. The project must not describe these URIs as production persistent identifiers. For production, replace the namespace with an institution-controlled domain or a registered identifier system, then publish resolution and versioning policies.

Resource patterns include:

- `/location/{verified-QID-or-local-key}`
- `/observation/climate/{stable-observation-id}`
- `/observation/livestock/{location-species-product}`
- `/route/{route-id}`
- `/record/{interview-or-media-id}`
- `/dataset/{dataset-id}`

Only exact, page-verified Wikidata entity matches in `data/raw/wikidata/verified_entity_references.json` receive `owl:sameAs`. In the current source evidence, Naryn Region is `Q486375` and Naryn city is `Q311946`. The city coordinates are from the Wikidata entity coordinate statement; they are not a meteorological station. The Naryn Region coordinate string is not converted to a decimal coordinate. The `owl:sameAs` links are based on exact entity page verification, not on a claim that the application URIs have been registered by Wikidata.

## Vocabulary and ontology

The ontology uses RDF/RDFS/OWL for the application schema; Dublin Core Terms and DCAT for catalogue metadata; PROV-O for derivation/build provenance; GeoSPARQL for features, geometries and WKT; SKOS for controlled terms; and WGS84 latitude/longitude predicates for the verified Naryn city point.

The SKOS vocabulary contains exactly these seven concepts: Summer pasture, Winter pasture, Seasonal migration route, Drought risk, Water availability, Vegetation condition, and Livestock observation. Labels and definitions are English only. No Kyrgyz or Russian translations have been invented. Alternative labels are English wording variants only where useful.

## Graph construction behaviour

The graph builder reads these files when present:

- `metadata/dataset_register.csv`
- `data/raw/wikidata/verified_entity_references.json`
- `data/processed/locations.csv`
- `data/processed/climate/nasa_power_daily.csv`
- `data/processed/climate/nasa_power_monthly.csv`
- `data/processed/climate/nasa_power_seasonal.csv`
- `data/processed/location_context.csv`
- `data/processed/vectors/naryn_adm1_clean.geojson` (or the raw Naryn GeoJSON as fallback)
- processed WorldCover/GLW4 rasters and processed OSM context GeoPackage for dataset/distribution metadata
- optional `data/synthetic/routes.geojson`, `data/synthetic/interviews.csv`, and `data/synthetic/multimedia.csv`

Optional demo input fields:

- `routes.geojson`: `route_id`, `name` (or `label`), `location_id`, optional `pasture_ids` (semicolon-separated IDs that must resolve to explicit pasture resources), `is_synthetic=true`, `provenance=synthetic_demo`, `synthetic_disclaimer`, and a geometry with a declared CRS.
- `interviews.csv`: `interview_id`, `title`, optional `description`, `location_id`, `is_synthetic=true`, `provenance=synthetic_demo`, and `synthetic_disclaimer`. This is metadata only; no interview transcript or quotation is created.
- `multimedia.csv`: `asset_id`, `title`, optional `description`, `location_id`, `is_synthetic=true`, `provenance=synthetic_demo`, and `synthetic_disclaimer`.

The project does not generate these records automatically. They are ingested only if separately prepared and clearly marked.

Rows are keyed by the stable `location_id`, not by fuzzy place-name matching. Climate observations preserve original date, parameter units, provenance and missing-value semantics. Monthly and seasonal summaries are represented as aggregated climate observations with their temporal resolution and aggregation period.

The builder links `location_context.csv` livestock fields only when they contain a non-null numeric raster sample. It links WorldCover class samples as `LandCoverObservation`; it never types them as `Pasture`. The data register is represented as DCAT dataset resources. Computed local availability is kept separate from the original source-register status so an old register status cannot override what is actually present in the current workspace.

Optional synthetic records are ignored unless `is_synthetic=true`, `provenance=synthetic_demo`, and a non-empty `synthetic_disclaimer` are all present. OSM layers remain a dataset/distribution; the pipeline does not convert any OSM road or path into a transhumance `Route`.

## The triple model

RDF stores each statement as a subject, predicate and object. For example, the Naryn city resource has statements equivalent to:

```turtle
<https://example.org/transhumance-atlas/location/q311946>
    a ta:Location, geo:Feature ;
    rdfs:label "Naryn"@en ;
    wgs84:lat "41.4333333"^^xsd:decimal ;
    wgs84:long "76.0"^^xsd:decimal ;
    owl:sameAs <http://www.wikidata.org/entity/Q311946> .
```

A climate observation is linked to the location and to its source dataset:

```turtle
<https://example.org/transhumance-atlas/observation/climate/nasa-power-wikidata-q311946-2020-01-01>
    a ta:ClimateObservation ;
    ta:atLocation <https://example.org/transhumance-atlas/location/q311946> ;
    ta:observedAt "2020-01-01"^^xsd:date ;
    ta:temperatureC "..."^^xsd:decimal ;
    prov:wasDerivedFrom <https://example.org/transhumance-atlas/dataset/nasa-power-daily> .
```

The ellipsis above is explanatory notation, not a generated temperature value. Actual numeric triples are emitted only from numeric values in the processed CSV.

## SPARQL query catalogue

1. `01_pasture_locations.rq` — finds resources explicitly typed `ta:Pasture`.
2. `02_routes_for_pasture.rq` — finds route-to-pasture relationships.
3. `03_climate_observations_for_location.rq` — retrieves daily and aggregated climate records linked to Wikidata Naryn city `Q311946`.
4. `04_research_records_for_location.rq` — retrieves explicitly linked interviews, media, routes or pasture records for Naryn city.
5. `05_datasets_and_metadata.rq` — returns dataset IDs, titles, publishers, licences, local availability and source links.

An empty result is a valid result: it means the current graph contains no matching records. It does not prove that such pastures, routes, interviews or assets do not exist outside the current acquired data.

## Run

From the repository root:

```bash
python -m pip install -r requirements-stage4.txt
python -m src.linked_data.build_graph --root .
python scripts/run_linked_data.py --root .
pytest -q
```

The second command rebuilds the graph and executes all five queries. It writes both output serialisations and the reports listed above. It reads the actual local files, so run it on the Mac project folder after Stage 3 has produced the latest outputs. The source data themselves are not copied into the package created in this environment.

## Validation and limitations

The built-in RDF checks cover required DCAT fields, exact Wikidata links and synthetic-record provenance/disclaimer consistency. The SHACL file is supplied for a SHACL engine such as `pyshacl`; `pyshacl` is an optional install and is not claimed to have run unless the report says so. The automated tests also parse the JSON-LD export back into RDF and execute all five SPARQL queries.

The `example.org` URIs are placeholders, not persistent identifiers. The graph currently cannot contain verified pasture parcels, real transhumance routes, real interviews or multimedia metadata unless corresponding source records become available. FAO sheep/goat observations appear only if the Stage 3 location context contains actual non-null sampled raster values. A successful RDF build does not imply full FAIR compliance or that every requested data class is populated.
