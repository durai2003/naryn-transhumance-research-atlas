from __future__ import annotations

from pathlib import Path

from src.atlas.cesium_server import AtlasAssetServer


def test_hosted_asset_server_uses_same_origin_prefixed_iframe_urls(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ATLAS_ASSET_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("ATLAS_ASSET_PORT", "0")
    monkeypatch.setenv("ATLAS_URL_PREFIX", "/atlas")

    server = AtlasAssetServer(tmp_path)
    try:
        assert server.port > 0
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
        assert server.iframe_url("state123") == f"http://127.0.0.1:{server.port}/?state=state123"
        assert server.network_iframe_url("network123") == (
            f"http://127.0.0.1:{server.port}/network/?state=network123"
        )
    finally:
        server.shutdown()
