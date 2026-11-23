#!/usr/bin/env python3
"""Inspect only files that genuinely exist; write machine-readable schema inventory."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path


def main(root: Path) -> int:
    inventory = {"generated_utc":datetime.now(timezone.utc).isoformat(), "files":[], "warnings":[]}
    raw = root / "data/raw"
    for path in sorted(raw.rglob('*')):
        if not path.is_file() or path.name.endswith('.gitkeep') or 'source_metadata' in path.parts:
            continue
        item = {"path":str(path.relative_to(root)), "bytes":path.stat().st_size, "inspection":"generic file metadata only"}
        suffix = path.suffix.casefold()
        try:
            if suffix in {'.tif','.tiff'}:
                import rasterio
                with rasterio.open(path) as ds:
                    item.update({"driver":ds.driver,"width":ds.width,"height":ds.height,"count":ds.count,
                                 "dtypes":list(ds.dtypes),"crs":str(ds.crs),"transform":list(ds.transform),
                                 "bounds":list(ds.bounds),"nodata":ds.nodata,"resolution":list(ds.res),
                                 "inspection":"rasterio local-file inspection"})
            elif suffix in {'.gpkg','.geojson','.shp'}:
                import geopandas as gpd
                if suffix == '.gpkg':
                    import pyogrio
                    layers = pyogrio.list_layers(path)
                    item["layers"] = []
                    for row in layers:
                        layer_name = str(row[0])
                        try:
                            gdf = gpd.read_file(path, layer=layer_name, rows=1)
                            item["layers"].append({"name":layer_name,"geometry_type":str(gdf.geom_type.iloc[0]) if len(gdf) else None,
                                                    "crs":str(gdf.crs),"columns":list(gdf.columns)})
                        except Exception as exc:
                            item["layers"].append({"name":layer_name,"inspection_error":str(exc)})
                else:
                    gdf = gpd.read_file(path)
                    item.update({"feature_count":len(gdf),"crs":str(gdf.crs),"geometry_types":sorted(gdf.geom_type.dropna().unique().tolist()),
                                 "columns":list(gdf.columns),"bounds":list(gdf.total_bounds),"inspection":"GeoPandas local-file inspection"})
            elif suffix == '.json':
                data = json.loads(path.read_text(encoding='utf-8'))
                item["top_level_type"] = type(data).__name__
                if isinstance(data, dict):
                    item["top_level_keys"] = list(data.keys())
                item["inspection"] = "JSON structure inspected"
            elif suffix == '.xml':
                import xml.etree.ElementTree as ET
                root_xml = ET.parse(path).getroot()
                item.update({"root_tag":root_xml.tag,"inspection":"XML well-formedness inspected"})
            elif suffix == '.zip':
                import zipfile
                with zipfile.ZipFile(path) as z:
                    item.update({"archive_members":z.namelist()[:100],"archive_member_count":len(z.namelist()),"inspection":"archive directory inspected"})
            elif suffix in {'.pbf'}:
                item["inspection"] = "PBF not decoded by this generic inspector; use osmium/pyosmium or GDAL tooling"
        except Exception as exc:
            item["inspection_error"] = str(exc)
            inventory["warnings"].append({"path":item["path"],"error":str(exc)})
        inventory["files"].append(item)
    out = root / "reports" / "local_schema_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Inspected {len(inventory['files'])} local files; inventory: {out}")
    if not inventory['files']:
        print("No primary dataset files present. Source metadata is intentionally excluded from the local dataset inventory.")
    return 0

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); sys.exit(main(a.root.resolve()))
