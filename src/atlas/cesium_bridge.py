"""Streamlit <-> Same-Origin CesiumJS 3D globe & Stage 2 Network Explorer bridge."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from src.atlas.cesium_server import get_asset_server


def render_cesium_globe(root: Path, payload: dict[str, Any], *, height: int = 720) -> None:
    """Register the 3D Atlas payload on the local same-origin asset server and mount the iframe."""
    root = Path(root).resolve()
    template_path = root / "frontend/cesium/index.html"
    if not template_path.exists():
        raise FileNotFoundError(f"CesiumJS frontend template missing: {template_path}")

    server = get_asset_server(root)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    state_id = server.register_payload(payload_json)
    url = server.iframe_url(state_id)
    components.iframe(url, height=height, scrolling=False)


def render_network_explorer(root: Path, payload: dict[str, Any], *, height: int = 780) -> None:
    """Register the Stage 2 Synchronised Network Explorer payload and mount its same-origin iframe."""
    root = Path(root).resolve()
    template_path = root / "frontend/network/explorer.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Network Explorer frontend template missing: {template_path}")

    server = get_asset_server(root)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    state_id = server.register_network_payload(payload_json)
    url = server.network_iframe_url(state_id)
    components.iframe(url, height=height, scrolling=False)
