# Stage 4 — Explain the triples and queries

## 1. What RDF does

RDF expresses data as three-part statements:

`subject — predicate — object`

For the atlas:

- Subject: the Naryn city resource.
- Predicate: `rdfs:label`.
- Object: the text `Naryn`.

Another statement links that same subject to its verified Wikidata page using `owl:sameAs`. A third statement gives its coordinate. The subject URI is reused across statements, so separate facts become a connected resource rather than unrelated CSV cells.

## 2. Why use URIs and namespaces?

A URI gives each resource a stable identity within the prototype. Namespaces shorten repeated URI prefixes. `ta:` is the project vocabulary, while `dcterms:`, `dcat:`, `prov:`, `geo:`, `skos:` and `wgs84:` reuse established vocabularies. The `example.org` URI pattern is a placeholder until an institution provides a persistent domain.

## 3. Key relationships

- `ta:atLocation`: climate or livestock observation → location.
- `ta:hasObservation`: location → observation.
- `prov:wasDerivedFrom`: derived record or dataset → upstream dataset.
- `geo:hasGeometry` / `geo:asWKT`: spatial feature → geometry → CRS-qualified WKT.
- `owl:sameAs`: local location URI → exactly verified Wikidata entity.
- `dcat:distribution`: dataset → local representation/distribution.
- `skos:inScheme`: controlled concept → transhumance concept scheme.

## 4. How to explain one climate record

A daily climate observation is a resource typed as `ta:ClimateObservation`. It links to the Naryn city location with `ta:atLocation`, stores its actual source date with `ta:observedAt`, stores `T2M` as degrees Celsius and `PRECTOTCORR` as millimetres/day, and links back to the NASA POWER dataset with PROV-O. A missing source sentinel is not converted to zero. If the CSV cell is empty, the value triple is omitted.

Monthly and seasonal rows are also represented as `ta:ClimateObservation`, but `ta:temporalResolution` states `monthly` or `seasonal`; the predicates `ta:temperatureMeanC` and `ta:precipitationSumMm` make the aggregation semantics clear.

## 5. Why SKOS is separate from the ontology

The ontology defines the data model (what a Pasture, Route or ClimateObservation is). SKOS defines controlled terms used to classify or describe domain concepts (for example, Summer pasture and Drought risk). This separates schema from terminology and permits controlled labels and definitions to evolve independently.

## 6. How to explain the five SPARQL queries

- Query 1 finds explicit `ta:Pasture` instances. It must not return grassland pixels simply because WorldCover class 30 is grassland.
- Query 2 follows `ta:connectsPasture` from a `ta:Route` to a `ta:Pasture`. OSM roads are not inserted into this relation.
- Query 3 follows `ta:atLocation` from climate observations to the exact Naryn city URI linked to Wikidata Q311946.
- Query 4 follows `ta:aboutLocation` or `ta:locatedIn` for location-related research records, including synthetic entries only when their provenance disclaimer is complete.
- Query 5 reads the DCAT catalogue and shows the distinction between a source URL and whether an asset is physically present in the local workspace.

## 7. Empty query output is not failure

The current acquired data do not include verified pasture parcels or real seasonal migration routes. Queries 1 and 2 may therefore return zero rows. The correct engineering behaviour is to report zero matching resources, not fabricate a pasture or turn an OSM path into a seasonal route. The same principle applies to interview and multimedia metadata.

## 8. Interview explanation in one minute

"I represented the atlas as a knowledge graph using RDFLib. I defined application classes and relations in an OWL/RDFS ontology, and created a seven-term SKOS vocabulary. DCAT and Dublin Core describe source and derived datasets, PROV-O records derivation, and GeoSPARQL represents geometries with declared CRS. I link Naryn city to Wikidata only using the verified Q311946 identifier. Climate observations are linked by a stable location key and retain daily/monthly/seasonal resolution and units. Five SPARQL queries retrieve pasture, route, climate, research and dataset information. Empty results are preserved when the evidence is not available; mapped roads and land-cover pixels are not mislabelled as migration routes or pasture parcels. The example.org namespace is still a placeholder and must be replaced before production publication."
