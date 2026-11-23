"""Build the knowledge graph and execute the five packaged SPARQL examples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `python scripts/run_linked_data.py` work from the repository root on macOS/Linux.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdflib import Graph
from src.linked_data.build_graph import build_from_root

QUERY_FILES = [
    "01_pasture_locations.rq",
    "02_routes_for_pasture.rq",
    "03_climate_observations_for_location.rq",
    "04_research_records_for_location.rq",
    "05_datasets_and_metadata.rq",
]


def _value(value):
    return str(value) if value is not None else None


def run_queries(root: Path) -> dict:
    root = root.resolve()
    report = build_from_root(root)
    graph = Graph()
    graph.parse(root / "linked_data" / "research_graph.ttl", format="turtle")
    outputs = {}
    for filename in QUERY_FILES:
        path = root / "queries" / filename
        query = path.read_text(encoding="utf-8")
        result = graph.query(query)
        rows = []
        for row in result:
            rows.append({str(var): _value(row[var]) for var in result.vars})
        outputs[filename] = {"row_count": len(rows), "rows": rows[:100]}
    bundle = {
        "graph_validation": report["validation"],
        "graph_counts": report["counts"],
        "queries": outputs,
        "note": "Zero rows indicate no matching verified records in the currently built graph; they are not evidence that such features do not exist in the real world.",
    }
    out = root / "reports" / "stage4_sparql_query_results.json"
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"graph_counts": bundle["graph_counts"], "query_rows": {k: v["row_count"] for k, v in outputs.items()}, "validation": report["validation"]["status"], "query_report": str(out)}, indent=2))
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    bundle = run_queries(args.root)
    return 0 if bundle["graph_validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
