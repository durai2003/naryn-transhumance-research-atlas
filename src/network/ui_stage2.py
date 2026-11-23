"""Network Analysis tab: synchronised geographical map and node-edge graph view."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.atlas.cesium_bridge import render_network_explorer
from src.network.explorer import (
    EDGE_CATEGORY_META,
    build_network_explorer_payload,
    filter_network,
)
from src.network.network_loader import Stage1NetworkData, load_stage1_network


@st.cache_resource(show_spinner=False)
def _load_stage1_cached(root_string: str) -> Stage1NetworkData:
    return load_stage1_network(Path(root_string))


def render_network_analysis_tab(
    root: Path,
    *,
    basemap: str = "esri_satellite",
    vertical_exaggeration: float = 1.4,
    default_scenario: str = "SCENARIO_BASELINE",
) -> None:
    """Render a clean, practical Network Analysis view (map + graph + compact table)."""
    net = _load_stage1_cached(str(root))
    if not net.available:
        st.warning(
            "Network data not found under `data/network/`. "
            "Run `python3 scripts/build_transhumance_network.py --root . --seed 42`."
        )
        return

    st.subheader("Transhumance Network Explorer")

    all_node_types = sorted(net.all_nodes["node_type"].dropna().astype(str).unique().tolist())
    all_edge_types = sorted(net.all_edges["edge_type"].dropna().astype(str).unique().tolist()) if not net.all_edges.empty else list(EDGE_CATEGORY_META.keys())
    all_seasons = sorted(net.all_edges["season"].dropna().astype(str).unique().tolist()) if not net.all_edges.empty else ["winter", "spring", "summer", "autumn", "all_year"]

    # Single compact filter row — practical and readable
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.1, 0.9])
    with c1:
        sel_node_types = st.multiselect("Node Types", options=all_node_types, default=all_node_types, key="s2_node_types")
    with c2:
        sel_edge_types = st.multiselect("Edge Types", options=all_edge_types, default=all_edge_types, key="s2_edge_types")
    with c3:
        sel_seasons = st.multiselect("Seasons", options=all_seasons, default=all_seasons, key="s2_seasons")
    with c4:
        public_only = st.checkbox("Public data only", value=False, key="s2_public_only")
        initial_layout = st.selectbox(
            "Graph Layout",
            options=["geographic", "spring"],
            format_func=lambda k: "Geographic" if k == "geographic" else "Force Layout",
            key="s2_layout",
        )

    filtered = filter_network(
        net,
        node_types=sel_node_types,
        edge_types=sel_edge_types,
        seasons=sel_seasons,
        climate_scenario=default_scenario,
        public_only=public_only,
    )

    st.caption(
        f"Showing **{len(filtered.nodes_gdf)}** nodes and **{len(filtered.edges_gdf)}** edges "
        f"across **{len(filtered.components)}** connected component(s). Click any node or route to inspect."
    )

    payload = build_network_explorer_payload(
        net,
        filtered,
        basemap=basemap,
        vertical_exaggeration=vertical_exaggeration,
        initial_layout=initial_layout,
        layout_names=("geographic", "spring"),
    )
    render_network_explorer(root, payload, height=740)

    with st.expander("View filtered nodes & edges tables", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            if not filtered.nodes_gdf.empty:
                cols = [c for c in ["node_id", "node_name", "node_type", "elevation_m", "landcover_class", "is_synthetic"] if c in filtered.nodes_gdf.columns]
                st.dataframe(filtered.nodes_gdf[cols], use_container_width=True, hide_index=True)
        with t2:
            if not filtered.edges_gdf.empty:
                cols = [c for c in ["edge_id", "source_node_id", "target_node_id", "edge_type", "distance_m", "season", "herd_count"] if c in filtered.edges_gdf.columns]
                st.dataframe(filtered.edges_gdf[cols], use_container_width=True, hide_index=True)
