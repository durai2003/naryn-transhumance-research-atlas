"""Same-origin local HTTP server for the CesiumJS 3D globe and Stage 2 Network Explorer inside Streamlit.

Serves:
- `/` and `/payload.json` -> Existing 3D Spatiotemporal Atlas (`frontend/cesium/index.html`)
- `/network/` and `/network/payload.json` -> Stage 2 Synchronised Geographical + Abstract Graph Explorer (`frontend/network/explorer.html`)
- `/cesium/*` -> Locally cached CesiumJS 1.123.0 runtime & Web Workers (100% same-origin)
- `/proxy/terrarium/*` -> Cached AWS Terrarium 3D elevation tiles
- `/proxy/imagery/*` -> Cached Esri / OpenTopoMap / OSM global imagery tiles
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

CESIUM_VERSION = "1.123.0"
CESIUM_CDN_BASE = f"https://cdn.jsdelivr.net/npm/cesium@{CESIUM_VERSION}/Build/Cesium/"
TERRARIUM_BASE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/"

IMAGERY_TEMPLATES = {
    "esri_satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "esri_hillshade": "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
    "esri_labels": "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    "esri_topo": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    "opentopo": "https://tile.opentopomap.org/{z}/{x}/{y}.png",
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}

EXTRA_MIME = {
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".wasm": "application/wasm",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".html": "text/html; charset=utf-8",
}


class AtlasAssetServer:
    """Thread-safe local HTTP server serving CesiumJS, Atlas, and Stage 2 Network Explorer same-origin."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.url_prefix = os.environ.get("ATLAS_URL_PREFIX", "").rstrip("/")
        self.template_path = self.root / "frontend/cesium/index.html"
        self.network_template_path = self.root / "frontend/network/explorer.html"
        self.cache_root = self.root / ".cache"
        self.cesium_cache = self.cache_root / f"cesium-{CESIUM_VERSION}"
        self.terrarium_cache = self.cache_root / "terrarium"
        self.imagery_cache = self.cache_root / "imagery"
        for d in (self.cesium_cache, self.terrarium_cache, self.imagery_cache):
            d.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._payloads: dict[str, bytes] = {"empty": b"{}"}
        self._latest_state_id: str = "empty"
        self._network_payloads: dict[str, bytes] = {"empty": b"{}"}
        self._latest_network_state_id: str = "empty"
        self._ui_state: dict[str, Any] = {
            "date": None,
            "herd_id": None,
            "scenario_id": None,
            "moving_count": None,
            "total_herds": None,
            "source": "init",
        }

        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update({
            "User-Agent": "MiniTranshumanceAtlas/3.2 (educational geospatial research prototype)"
        })

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send(self, body: bytes, content_type: str, status: int = 200, cache_sec: int = 0) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                if cache_sec > 0:
                    self.send_header("Cache-Control", f"public, max-age={cache_sec}")
                else:
                    self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_OPTIONS(self) -> None:
                self._send(b"", "text/plain", 204)

            def do_POST(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != "/api/ui_state":
                    self._send(b"Not found", "text/plain", 404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(min(length, 65536))
                    data = json.loads(raw.decode("utf-8")) if raw else {}
                    if isinstance(data, dict):
                        server_ref.update_ui_state(data)
                    body = json.dumps(server_ref.get_ui_state()).encode("utf-8")
                    self._send(body, "application/json; charset=utf-8", 200, cache_sec=0)
                except Exception as exc:
                    self._send(f"Bad payload: {exc}".encode("utf-8"), "text/plain", 400)

            def do_GET(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                path = parsed.path
                query = urllib.parse.parse_qs(parsed.query)

                if path in ("/", "/index.html"):
                    if not server_ref.template_path.exists():
                        self._send(b"Missing frontend/cesium/index.html", "text/plain", 404)
                        return
                    self._send(server_ref.template_path.read_bytes(), "text/html; charset=utf-8", 200, cache_sec=0)
                    return

                if path == "/payload.json":
                    state_id = (query.get("state") or [server_ref._latest_state_id])[0]
                    self._send(server_ref.get_payload(state_id), "application/json; charset=utf-8", 200, cache_sec=0)
                    return

                if path in ("/network", "/network/", "/network/index.html"):
                    if not server_ref.network_template_path.exists():
                        self._send(b"Missing frontend/network/explorer.html", "text/plain", 404)
                        return
                    self._send(server_ref.network_template_path.read_bytes(), "text/html; charset=utf-8", 200, cache_sec=0)
                    return

                if path == "/network/payload.json":
                    state_id = (query.get("state") or [server_ref._latest_network_state_id])[0]
                    self._send(server_ref.get_network_payload(state_id), "application/json; charset=utf-8", 200, cache_sec=0)
                    return

                if path == "/health":
                    self._send(b'{"status":"ok","network_explorer":true,"ui_sync":true}', "application/json", 200)
                    return

                if path == "/api/ui_state":
                    body = json.dumps(server_ref.get_ui_state()).encode("utf-8")
                    self._send(body, "application/json; charset=utf-8", 200, cache_sec=0)
                    return

                if path.startswith("/cesium/"):
                    rel = urllib.parse.unquote(path[len("/cesium/"):]).lstrip("/")
                    self._serve_cached_upstream(
                        rel=rel,
                        cache_dir=server_ref.cesium_cache,
                        upstream_url=CESIUM_CDN_BASE + rel,
                        cache_sec=604800,
                    )
                    return

                if path.startswith("/proxy/terrarium/"):
                    parts = path[len("/proxy/terrarium/"):].split("/")
                    if len(parts) == 3 and parts[2].endswith(".png"):
                        z, x, y_png = parts
                        if z.isdigit() and x.isdigit() and y_png[:-4].isdigit():
                            rel = f"{z}/{x}/{y_png}"
                            self._serve_cached_upstream(
                                rel=rel,
                                cache_dir=server_ref.terrarium_cache,
                                upstream_url=TERRARIUM_BASE + rel,
                                cache_sec=604800,
                                force_mime="image/png",
                            )
                            return
                    self._send(b"Invalid terrarium tile", "text/plain", 400)
                    return

                if path.startswith("/proxy/imagery/"):
                    parts = path[len("/proxy/imagery/"):].split("/")
                    if len(parts) == 4:
                        style, z, y, x_raw = parts
                        x = x_raw.split(".")[0]
                        tmpl = IMAGERY_TEMPLATES.get(style)
                        if tmpl and z.isdigit() and y.isdigit() and x.isdigit():
                            rel = f"{style}/{z}/{y}/{x}.img"
                            upstream = tmpl.format(z=z, y=y, x=x)
                            self._serve_cached_upstream(
                                rel=rel,
                                cache_dir=server_ref.imagery_cache,
                                upstream_url=upstream,
                                cache_sec=604800,
                            )
                            return
                    self._send(b"Invalid imagery tile", "text/plain", 400)
                    return

                self._send(b"Not found", "text/plain", 404)

            def _serve_cached_upstream(
                self,
                *,
                rel: str,
                cache_dir: Path,
                upstream_url: str,
                cache_sec: int,
                force_mime: str | None = None,
            ) -> None:
                if ".." in rel or rel.startswith("/"):
                    self._send(b"Forbidden", "text/plain", 403)
                    return
                target = (cache_dir / rel).resolve()
                if not str(target).startswith(str(cache_dir.resolve())):
                    self._send(b"Forbidden", "text/plain", 403)
                    return

                mime = force_mime or EXTRA_MIME.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"

                if target.exists() and target.is_file() and target.stat().st_size > 0:
                    data = target.read_bytes()
                    if target.suffix == ".img":
                        mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
                    self._send(data, mime, 200, cache_sec=cache_sec)
                    return

                try:
                    resp = server_ref._session.get(upstream_url, timeout=(8, 25))
                    if resp.status_code != 200:
                        self._send(b"Upstream status " + str(resp.status_code).encode(), "text/plain", resp.status_code)
                        return
                    data = resp.content
                    resp_mime = force_mime or EXTRA_MIME.get(target.suffix.lower()) or resp.headers.get("Content-Type") or mime
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_suffix(target.suffix + f".{threading.get_ident()}.tmp")
                    tmp.write_bytes(data)
                    tmp.replace(target)
                    self._send(data, resp_mime, 200, cache_sec=cache_sec)
                except Exception as exc:
                    self._send(f"Upstream fetch failed: {exc}".encode("utf-8"), "text/plain", HTTPStatus.BAD_GATEWAY)

        bind_host = os.environ.get("ATLAS_ASSET_BIND_HOST", "127.0.0.1")
        bind_port = int(os.environ.get("ATLAS_ASSET_PORT", "0"))
        self._httpd = ThreadingHTTPServer((bind_host, bind_port), Handler)
        self._httpd.daemon_threads = True
        self.port: int = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="AtlasAssetServer", daemon=True)
        self._thread.start()

    def register_payload(self, payload_json: str) -> str:
        raw = payload_json.encode("utf-8")
        state_id = hashlib.md5(raw).hexdigest()[:12]
        with self._lock:
            if len(self._payloads) > 12:
                self._payloads.clear()
            self._payloads[state_id] = raw
            self._latest_state_id = state_id
        return state_id

    def get_payload(self, state_id: str) -> bytes:
        with self._lock:
            return self._payloads.get(state_id) or self._payloads.get(self._latest_state_id, b"{}")

    def iframe_url(self, state_id: str) -> str:
        if self.url_prefix:
            return f"{self.url_prefix}/?state={state_id}"
        return f"http://127.0.0.1:{self.port}/?state={state_id}"

    def register_network_payload(self, payload_json: str) -> str:
        raw = payload_json.encode("utf-8")
        state_id = hashlib.md5(raw).hexdigest()[:12]
        with self._lock:
            if len(self._network_payloads) > 12:
                self._network_payloads.clear()
            self._network_payloads[state_id] = raw
            self._latest_network_state_id = state_id
        return state_id

    def get_network_payload(self, state_id: str) -> bytes:
        with self._lock:
            return self._network_payloads.get(state_id) or self._network_payloads.get(self._latest_network_state_id, b"{}")

    def network_iframe_url(self, state_id: str) -> str:
        if self.url_prefix:
            return f"{self.url_prefix}/network/?state={state_id}"
        return f"http://127.0.0.1:{self.port}/network/?state={state_id}"

    def set_ui_state(
        self,
        *,
        date: str | None = None,
        herd_id: str | None = None,
        scenario_id: str | None = None,
        source: str = "streamlit",
    ) -> None:
        with self._lock:
            self._ui_state = {
                "date": date,
                "herd_id": herd_id,
                "scenario_id": scenario_id,
                "moving_count": None,
                "total_herds": None,
                "source": source,
            }

    def update_ui_state(self, patch: dict[str, Any]) -> None:
        with self._lock:
            for k in ("date", "herd_id", "scenario_id", "moving_count", "total_herds"):
                if k in patch:
                    self._ui_state[k] = patch[k]
            self._ui_state["source"] = str(patch.get("source", "cesium"))

    def get_ui_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._ui_state)

    def shutdown(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass


_SERVER_LOCK = threading.Lock()
_GLOBAL_SERVERS: dict[str, AtlasAssetServer] = {}


def get_asset_server(root: Path) -> AtlasAssetServer:
    key = str(Path(root).resolve())
    with _SERVER_LOCK:
        srv = _GLOBAL_SERVERS.get(key)
        if srv is None:
            srv = AtlasAssetServer(Path(key))
            _GLOBAL_SERVERS[key] = srv
        return srv
