import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_register_has_all_seven_source_groups():
    with (ROOT/'metadata/dataset_register.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7
    ids = {r['dataset_id'] for r in rows}
    assert {'GB-ADM1-KGZ','KDC-PASTURE-SNOW-DROUGHT','FAO-GLW4-2020-DENSITY',
            'ESA-WORLDCOVER-2021-V200','NASA-POWER-DAILY','OSM-KGZ-GEOFABRIK','WIKIDATA-NARYN-ENTITIES'} == ids


def test_no_missing_assets_mislabeled_as_downloaded():
    with (ROOT/'metadata/dataset_register.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row['acquisition_status'] not in {'SUCCESS_ENTITY_PAGES'}:
            assert 'NOT_DOWNLOADED' in row['acquisition_status'] or 'FAILED' in row['acquisition_status'] or 'VERIFIED' in row['acquisition_status'] or 'DOCUMENTATION_ONLY' in row['acquisition_status'] or 'LISTING_VERIFIED' in row['acquisition_status'] or 'CATALOGUE_SCHEMA' in row['acquisition_status'] or 'SPECIFICATION' in row['acquisition_status']


def test_wikidata_identifiers_are_page_sourced_not_sparql_claim():
    data = json.loads((ROOT/'data/raw/wikidata/verified_entity_references.json').read_text(encoding='utf-8'))
    assert {e['qid'] for e in data['entities']} == {'Q486375','Q311946'}
    assert 'could not be queried' in data['retrieval_method']
