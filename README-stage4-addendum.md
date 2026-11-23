## Stage 4 — Linked Open Data

Install Stage 4 dependencies and run from the repository root:

```bash
python3 -m pip install -r requirements-stage4.txt
python3 scripts/run_linked_data.py --root .
python3 -m pytest -q
```

The runner builds `linked_data/research_graph.ttl` and `linked_data/research_graph.jsonld`, executes the five SPARQL queries under `queries/`, and updates the Stage 4 reports. Run it from the existing Mac project root after Stage 3. It reads local files and does not fabricate missing pasture, route, livestock, climate, interview or multimedia observations.

Optional SHACL engine:

```bash
python3 -m pip install -r requirements-stage4-shacl.txt
python3 scripts/run_linked_data.py --root .
```

Read `docs/stage4_linked_data.md` and `docs/stage4_triples_walkthrough.md` for the data model, triple interpretation, query explanations and limitations.

The `https://example.org/transhumance-atlas/` namespace is a placeholder, not a registered persistent identifier service. Replace it with institution-controlled persistent URIs before production publication.
