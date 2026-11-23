from __future__ import annotations

import time
from pathlib import Path

import requests

from src.atlas.cesium_server import AtlasAssetServer


def _wait_until_ready(server: AtlasAssetServer) -> None:
    for _ in range(50):
        try:
            response = requests.get(f"http://127.0.0.1:{server.port}/health", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.02)
    raise AssertionError("Asset server did not become ready")


def test_hosted_asset_server_uses_same_origin_prefixed_iframe_urls(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ATLAS_ASSET_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("ATLAS_ASSET_PORT", "0")
    monkeypatch.setenv("ATLAS_URL_PREFIX", "/atlas")

    server = AtlasAssetServer(tmp_path)
    try:
        _wait_until_ready(server)
        assert server.iframe_url("state123") == "/atlas/?state=state123"
        assert server.network_iframe_url("network123") == "/atlas/network/?state=network123"
    finally:
        server.shutdown()


def test_local_development_keeps_loopback_iframe_urls(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ATLAS_ASSET_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("ATLAS_ASSET_PORT", "0")
    monkeypatch.delenv("ATLAS_URL_PREFIX", raising=False)

    server = AtlasAssetServer(tmp_path)
    try:
        _wait_until_ready(server)
        assert server.iframe_url("state123") == f"http://127.0.0.1:{server.port}/?state=state123"
        assert server.network_iframe_url("network123") == (
            f"http://127.0.0.1:{server.port}/network/?state=network123"
        )
    finally:
        server.shutdown()


def test_atlas_payloads_are_content_addressable_and_cacheable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ATLAS_ASSET_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("ATLAS_ASSET_PORT", "0")
    monkeypatch.setenv("ATLAS_URL_PREFIX", "/atlas")

    server = AtlasAssetServer(tmp_path)
    try:
        _wait_until_ready(server)
        atlas_state = server.register_payload('{"kind":"atlas","value":1}')
        network_state = server.register_network_payload('{"kind":"network","value":2}')

        atlas_response = requests.get(
            f"http://127.0.0.1:{server.port}/payload.json?state={atlas_state}",
            timeout=2,
        )
        network_response = requests.get(
            f"http://127.0.0.1:{server.port}/network/payload.json?state={network_state}",
            timeout=2,
        )

        assert atlas_response.json() == {"kind": "atlas", "value": 1}
        assert network_response.json() == {"kind": "network", "value": 2}
        assert atlas_response.headers["Cache-Control"] == "public, max-age=604800"
        assert network_response.headers["Cache-Control"] == "public, max-age=604800"
    finally:
        server.shutdown()
