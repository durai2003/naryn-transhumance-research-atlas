"""Bridge between Streamlit backend state and inline CesiumJS / Network frontends."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.atlas.cesium_server import get_asset_server

CESIUM_CDN = "https://cdn.jsdelivr.net/npm/cesium@1.123.0/Build/Cesium/"


def use_local_asset_server() -> bool:
    return os.getenv("ATLAS_USE_LOCAL_SERVER", "").strip().lower() in {"1", "true", "yes"}


def build_inline_html(template_html: str, payload: dict[str, Any], *, global_var: str) -> str:
    html = template_html.replace("/cesium/Widgets/widgets.css", CESIUM_CDN + "Widgets/widgets.css")
    html = html.replace("/cesium/Cesium.js", CESIUM_CDN + "Cesium.js")
    html = html.replace('"/cesium/"', f'"{CESIUM_CDN}"')
    html = html.replace("'/cesium/'", f"'{CESIUM_CDN}'")

    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    shim = (
        "<script>\n"
        "window.__ATLAS_INLINE_MODE__ = true;\n"
        f"window.CESIUM_BASE_URL = {json.dumps(CESIUM_CDN)};\n"
        "const _W = window.Worker;\n"
        "if (_W) {\n"
        "  window.Worker = function(u, o) {\n"
        "    const s = String(u);\n"
        "    if (/^https?:/i.test(s) && !s.startsWith(location.origin)) {\n"
        "      const d = s.slice(0, s.lastIndexOf('/') + 1);\n"
        "      const c = 'self.CESIUM_BASE_URL=' + JSON.stringify(" + json.dumps(CESIUM_CDN) + ") + ';'\n"
        "        + 'var _b=' + JSON.stringify(d) + ';'\n"
        "        + 'var _i=self.importScripts.bind(self);'\n"
        "        + 'self.importScripts=function(){return _i.apply(self,[].map.call(arguments,function(x){return /^(https?:|blob:|data:)/i.test(x)?x:new URL(x,_b).href}))};'\n"
        "        + 'importScripts(' + JSON.stringify(s) + ');';\n"
        "      return new _W(URL.createObjectURL(new Blob([c], {type: 'application/javascript'})), o);\n"
        "    }\n"
        "    return new _W(u, o);\n"
        "  };\n"
        "  window.Worker.prototype = _W.prototype;\n"
        "}\n"
        f"window.{global_var} = {raw};\n"
        "</script>"
    )
    return html.replace("<head>", "<head>\n" + shim, 1) if "<head>" in html else shim + html


def _find_template(root: Path, primary_rel: str, fallback_rel: str) -> Path | None:
    for rel in (primary_rel, fallback_rel):
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def render_cesium_globe(root: Path, payload: dict[str, Any], *, height: int = 690) -> None:
    root_path = Path(root).resolve()
    tp = _find_template(root_path, "frontend/cesium/index.html", "frontend/frontend/cesium/index.html")
    if tp is None:
        st.error("Missing frontend/cesium/index.html in repository.")
        return
    if use_local_asset_server():
        srv = get_asset_server(root_path)
        sid = srv.register_payload(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        components.iframe(srv.iframe_url(sid), height=height, scrolling=False)
        return
    components.html(build_inline_html(tp.read_text(encoding="utf-8"), payload, global_var="__ATLAS_PAYLOAD__"), height=height, scrolling=False)


def render_network_explorer(root: Path, payload: dict[str, Any], *, height: int = 760) -> None:
    root_path = Path(root).resolve()
    tp = _find_template(root_path, "frontend/network/explorer.html", "frontend/frontend/network/explorer.html")
    if tp is None:
        st.error("Missing frontend/network/explorer.html in repository.")
        return
    if use_local_asset_server():
        srv = get_asset_server(root_path)
        sid = srv.register_network_payload(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        components.iframe(srv.network_iframe_url(sid), height=height, scrolling=False)
        return
    components.html(build_inline_html(tp.read_text(encoding="utf-8"), payload, global_var="__NETWORK_PAYLOAD__"), height=height, scrolling=False)
