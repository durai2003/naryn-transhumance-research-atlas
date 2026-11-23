# Mini Transhumance Research Data Atlas — Stages 1–2

This workspace contains the source audit, dataset register, download log, and reproducible acquisition/inspection scripts for Stage 1 and Stage 2 only. It is an independent educational prototype. It is not the official Leiden research project and does not contain field observations or genuine migration routes.

## Current state

See `reports/stage1_stage2_source_audit.md` for the evidence-based result. Source documentation and selected metadata are checked, but actual source files could not be downloaded in this execution environment because outbound DNS/network access failed. No synthetic replacements were generated.

## Layout

```text
transhumance-atlas/
├── data/
│   ├── raw/
│   │   ├── boundaries/       # geoBoundaries files go here after retrieval
│   │   ├── glw4/             # FAO sheep/goat density rasters
│   │   ├── worldcover/       # ESA WorldCover 2021 tiles
│   │   ├── nasa_power/       # cached NASA POWER JSON responses
│   │   ├── osm/              # Geofabrik GeoPackage/PBF extract
│   │   ├── kdc/              # WMS/WCS capabilities XML and any allowed exports
│   │   ├── wikidata/         # verified QID references and SPARQL results
│   │   └── source_metadata/  # provenance evidence; not primary data
│   └── processed/            # intentionally empty until real sources are acquired
├── metadata/
│   ├── dataset_register.csv
│   └── dataset_register.json
├── reports/
│   ├── stage1_stage2_source_audit.md
│   └── download_log.csv
├── scripts/
│   ├── stage2_acquire.py
│   └── stage2_inspect.py
└── tests/
    └── test_stage12_register.py
```

## Run in a network-enabled environment

Python 3.11+ is recommended. Install the packages required for downloading and inspecting the sources:

```bash
python -m venv .venv
source .venv/bin/activate             # macOS/Linux
python -m pip install -r requirements-stage12.txt
```

Run acquisition:

```bash
python scripts/stage2_acquire.py --root .
```

Run the inspection pass:

```bash
python scripts/stage2_inspect.py --root .
```

The acquisition script writes to `data/raw/`, appends outcomes to `reports/acquisition_run_log.jsonl`, and does not invent replacement records. It uses explicit timeouts and retry/backoff. A failed or restricted source stays absent and is recorded as failed/restricted. The script only downloads the documented WorldCover Map tiles once the actual Naryn boundary file exists. It queries the Wikidata v2 SPARQL endpoint and stores the exact query text and JSON response. KDC GetCapabilities documents are saved only when the service returns them.

## Important licensing / interpretation rules

- Use the specific geoBoundaries API record's ODbL licence in the interim; reconcile the apparent difference from general gbOpen guidance before redistribution.
- Keep FAO sheep and goat density separate, preserve the original 0.008333333° grid and head/km² unit, and cite CC BY 4.0.
- ESA WorldCover classes are land-cover classes, not official pasture parcels, grazing rights or seasonal pasture-use assignments.
- OSM paths/roads are mapped transport context, not confirmed transhumance routes. Attribute OSM and comply with ODbL.
- NASA POWER is grid-based climate data, not a local weather-station observation. Retain source parameters, units, coordinates, grid context and cache timestamp.
- KDC licence and data-service capabilities require explicit verification before redistributing any retrieved products.
- Wikidata entity pages are not SPARQL result provenance. Retain the query and response before claiming SPARQL verification.

## Do not proceed to spatial processing yet

The Naryn polygon, FAO rasters, WorldCover tiles and OSM extract are not present. Stage 2 is incomplete until the data are acquired and `stage2_inspect.py` can report their local schemas. This README deliberately avoids claiming that the scripts have been run against unavailable datasets.
