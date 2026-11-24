"""Stage 2 Streamlit UI module: Network Analysis tab with Synchronised Geographical & Abstract Graph Explorer."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.atlas.cesium_bridge import render_network_explorer
from src.network.explorer import (
    EDGE_CATEGORY_META,
    NODE_CATEGORY_META,
    build_network_explorer_payload,
    filter_network,
)
from src.network.network_loader import Stage1NetworkData, load_stage1_network


@st.cache_resource(show_spinner=False)
def _load_stage1_cached(root_string: str) -> Stage1NetworkData:
    """Share the read-only geospatial network object instead of deserializing it per rerun."""
    return load_stage1_network(Path(root_string))


@st.cache_resource(show_spinner=False)
def _cached_network_payload(
    cache_key: str,
    _net: Stage1NetworkData,
    _filtered,
    _config: tuple,
):
    """Keep the expensive layouts and geometry serialisation for each filter state."""
    del cache_key
    basemap, vertical_exaggeration, initial_layout, selected_node_id, selected_edge_id = _config
    return build_network_explorer_payload(
        _net,
        _filtered,
        basemap=basemap,
        vertical_exaggeration=vertical_exaggeration,
        initial_layout=initial_layout,
        selected_node_id=selected_node_id,
        selected_edge_id=selected_edge_id,
    )


def render_network_analysis_tab(
    root: Path,
    *,
    basemap: str = "esri_satellite",
    vertical_exaggeration: float = 1.4,
    default_scenario: str = "SCENARIO_BASELINE",
) -> None:
    """Render the Stage 2 Network Analysis tab (preserving Stage 3–5 extension slots)."""
    net = _load_stage1_cached(str(root))
    if not net.available:
        st.warning(
            "Stage 1 network artifacts not found under `data/network/`. "
            "Run `python3 scripts/build_transhumance_network.py --root . --seed 42` first."
        )
        return

    st.markdown(
        "### Stage 2 · Interactive Nodes, Edges & Synchronised Graph Explorer"
    )
    st.caption(
        "Click any node or edge in either the **Geographical View (CesiumJS 3D)** or the **Abstract Graph View** "
        "to synchronise selection, highlight routes and 1-hop neighbours, and inspect full provenance and DEM elevation profiles."
    )

    # --- FILTER BAR (All 6 Stage 2 filter dimensions + Public-Only toggle) ---
    all_node_types = sorted(net.all_nodes["node_type"].dropna().astype(str).unique().tolist())
    all_edge_types = sorted(net.all_edges["edge_type"].dropna().astype(str).unique().tolist()) if not net.all_edges.empty else list(EDGE_CATEGORY_META.keys())
    all_seasons = sorted(net.all_edges["season"].dropna().astype(str).unique().tolist()) if not net.all_edges.empty else ["winter", "spring", "summer", "autumn", "all_year"]
    all_livestock = sorted(net.herds["livestock_type"].dropna().astype(str).unique().tolist()) if not net.herds.empty else ["sheep", "goat", "cattle", "horse", "mixed"]
    all_evidence = sorted(net.all_nodes["evidence_status"].dropna().astype(str).unique().tolist())
    scenario_options = sorted(set([default_scenario, "SCENARIO_BASELINE", "SCENARIO_WARMER_DRIER", "SCENARIO_EARLY_SNOW"]))

    f_row1 = st.columns([1.35, 1.35, 1.1, 1.1])
    with f_row1[0]:
        sel_node_types = st.multiselect(
            "Node Type",
            options=all_node_types,
            default=all_node_types,
            key="s2_node_types",
        )
    with f_row1[1]:
        sel_edge_types = st.multiselect(
            "Edge Type",
            options=all_edge_types,
            default=all_edge_types,
            key="s2_edge_types",
        )
    with f_row1[2]:
        sel_seasons = st.multiselect(
            "Season",
            options=all_seasons,
            default=all_seasons,
            key="s2_seasons",
        )
    with f_row1[3]:
        sel_livestock = st.multiselect(
            "Livestock Type",
            options=all_livestock,
            default=all_livestock,
            key="s2_livestock",
        )

    f_row2 = st.columns([1.2, 1.35, 1.15, 1.2])
    with f_row2[0]:
        sel_scenario = st.selectbox(
            "Climate Scenario",
            options=scenario_options,
            index=scenario_options.index(default_scenario) if default_scenario in scenario_options else 0,
            key="s2_scenario",
        )
    with f_row2[1]:
        sel_evidence = st.multiselect(
            "Evidence Status",
            options=all_evidence,
            default=all_evidence,
            key="s2_evidence",
        )
    with f_row2[2]:
        initial_layout = st.selectbox(
            "Default Graph Layout",
            options=["layered_seasonal", "geographic", "spring", "kamada_kawai"],
            format_func=lambda k: {
                "layered_seasonal": "Seasonal Layered",
                "geographic": "Geographic XY",
                "spring": "Force / Spring",
                "kamada_kawai": "Kamada-Kawai",
            }.get(k, k),
            key="s2_layout",
        )
    with f_row2[3]:
        public_only = st.toggle(
            "Public sourced data only (hide synthetic)",
            value=False,
            key="s2_public_only",
            help="Hides all synthetic demonstration nodes, corridors, and herds to inspect only verified public OSM/Wikidata geography.",
        )

    if sel_scenario != "SCENARIO_BASELINE":
        st.info(
            f"Viewing baseline Stage 1 network under `{sel_scenario}` filter tag. "
            "Climate-driven pasture unavailability and resilience recalculations are scheduled for **Stage 4**."
        )

    filtered = filter_network(
        net,
        node_types=sel_node_types,
        edge_types=sel_edge_types,
        seasons=sel_seasons,
        livestock_types=sel_livestock,
        climate_scenario=sel_scenario,
        evidence_statuses=sel_evidence,
        public_only=public_only,
    )

    # Searchable quick-jump selectors synced with BOTH Geographical & Abstract Graph views
    q_cols = st.columns([1.4, 1.6, 1.0, 1.0, 1.0])
    node_opts = ["(Click on map or graph)"] + [
        f"{r['node_id']} · {r['node_name']} [{r['node_type']}]"
        for _, r in filtered.nodes_gdf.iterrows()
    ]
    edge_opts = ["(Click on map or graph)"] + [
        f"{r['edge_id']} · {r['source_node_id']} → {r['target_node_id']} [{r['edge_type']}]"
        for _, r in filtered.edges_gdf.iterrows()
    ]
    with q_cols[0]:
        jump_node = st.selectbox("Jump to Node (syncs both views)", node_opts, index=0, key="s2_jump_node")
    with q_cols[1]:
        jump_edge = st.selectbox("Jump to Edge / Route (syncs both views)", edge_opts, index=0, key="s2_jump_edge")

    selected_node_id = jump_node.split(" · ")[0] if jump_node != "(Click on map or graph)" else None
    selected_edge_id = jump_edge.split(" · ")[0] if jump_edge != "(Click on map or graph)" else None
    if selected_edge_id is not None:
        selected_node_id = None  # Edge jump takes priority if both chosen

    # Summary KPIs
    pub_nodes = int((~filtered.nodes_gdf["is_synthetic"].astype(bool)).sum()) if not filtered.nodes_gdf.empty else 0
    syn_nodes = int(filtered.nodes_gdf["is_synthetic"].astype(bool).sum()) if not filtered.nodes_gdf.empty else 0
    used_edges = int((filtered.edges_gdf["movement_event_count"] > 0).sum()) if not filtered.edges_gdf.empty else 0
    q_cols[2].metric("Visible Nodes", f"{len(filtered.nodes_gdf)}", f"{pub_nodes} pub / {syn_nodes} syn")
    q_cols[3].metric("Visible Edges", f"{len(filtered.edges_gdf)}", f"{used_edges} active")
    q_cols[4].metric("Connected Components", f"{len(filtered.components)}")

    # Cache payload construction by the visible graph/filter state.
    # This avoids repeating NetworkX layouts and GeoJSON serialisation on unrelated reruns.
    payload_key_data = {
        "root": str(root.resolve()),
        "node_ids": filtered.nodes_gdf["node_id"].astype(str).tolist(),
        "edge_ids": filtered.edges_gdf["edge_id"].astype(str).tolist(),
        "filters": filtered.active_filters,
        "basemap": basemap,
        "vertical_exaggeration": vertical_exaggeration,
        "initial_layout": initial_layout,
        "selected_node_id": selected_node_id,
        "selected_edge_id": selected_edge_id,
    }
    payload_cache_key = hashlib.sha256(
        json.dumps(payload_key_data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    payload = _cached_network_payload(
        payload_cache_key,
        net,
        filtered,
        (
            basemap,
            vertical_exaggeration,
            initial_layout,
            selected_node_id,
            selected_edge_id,
        ),
    )
    render_network_explorer(root, payload, height=780)

    # --- STREAMLIT DETAILED INSPECTOR & COMPONENT SUMMARY BELOW ---
    st.divider()
    tab_insp, tab_comp, tab_tables = st.tabs([
        "🔎 Selected Node / Edge Deep Inspector",
        "🧩 Connected Components Breakdown",
        "📋 Filtered Nodes & Edges Tables",
    ])

    with tab_insp:
        if selected_edge_id:
            edge_match = next((e for e in payload["edges"] if e["edge_id"] == selected_edge_id), None)
            if edge_match:
                c_left, c_right = st.columns([1.2, 1.5])
                with c_left:
                    st.markdown(f"#### Edge `{edge_match['edge_id']}` · `{edge_match['edge_type']}`")
                    st.markdown(
                        f"- **Origin:** `{edge_match['source_node_id']}` ({edge_match['source_node_name']})\n"
                        f"- **Destination:** `{edge_match['target_node_id']}` ({edge_match['target_node_name']})\n"
                        f"- **Distance:** `{edge_match['distance_km']} km` (`{edge_match['distance_m']:.0f} m`)\n"
                        f"- **Slope (avg / max):** `{edge_match['average_slope']}° / {edge_match['maximum_slope']}°`\n"
                        f"- **Elevation Gain / Loss:** `+{edge_match['elevation_gain_m']} m / -{edge_match['elevation_loss_m']} m`\n"
                        f"- **Season & Direction:** `{edge_match['season']}` · `{edge_match['direction']}`\n"
                        f"- **Route Method:** `{edge_match['route_generation_method']}`\n"
                        f"- **Utilisation:** `{edge_match['utilisation']['herd_count']}` herds · "
                        f"`{edge_match['utilisation']['animal_count']:,}` animals · "
                        f"`{edge_match['utilisation']['movement_event_count']}` events\n"
                        f"- **Provenance:** `{edge_match['provenance']}` (`{edge_match['evidence_status']}` · `is_synthetic={edge_match['is_synthetic']}`)"
                    )
                with c_right:
                    prof = edge_match.get("elevation_profile") or []
                    pdf = pd.DataFrame(prof)
                    if not pdf.empty and "elevation_m" in pdf.columns and pdf["elevation_m"].notna().any():
                        pdf["distance_km"] = pdf["distance_m"] / 1000.0
                        fig = px.area(
                            pdf, x="distance_km", y="elevation_m",
                            title=f"DEM Elevation Profile · {edge_match['edge_id']} ({edge_match['elevation_source']})",
                            labels={"distance_km": "Distance along route (km)", "elevation_m": "Elevation (m)"},
                        )
                        fig.update_traces(line_color="#facc15", fillcolor="rgba(250, 204, 21, 0.25)")
                        fig.update_layout(height=300, margin=dict(l=30, r=20, t=40, b=30))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No local DEM elevation samples attached to this edge.")
        elif selected_node_id:
            node_match = next((n for n in payload["nodes"] if n["node_id"] == selected_node_id), None)
            if node_match:
                c_left, c_right = st.columns([1.15, 1.55])
                with c_left:
                    st.markdown(f"#### Node `{node_match['node_id']}` · {node_match['node_name']}")
                    st.markdown(
                        f"- **Category:** `{node_match['node_type']}` (`{node_match['network_layer']}`)\n"
                        f"- **Coordinates:** `{node_match['latitude']:.5f}°N, {node_match['longitude']:.5f}°E`\n"
                        f"- **Elevation / Slope:** `{node_match['elevation_m']} m` · `{node_match['slope_degrees']}°` (`{node_match['elevation_source']}`)\n"
                        f"- **Land Cover:** `{node_match['landcover_class']}` (code `{node_match['landcover_code']}`)\n"
                        f"- **Component:** `#{node_match['component_id']}`\n"
                        f"- **1-Hop Neighbours:** `{len(node_match['in_neighbors'])}` incoming / `{len(node_match['out_neighbors'])}` outgoing\n"
                        f"- **Source & Provenance:** `{node_match['source']}` · `{node_match['provenance']}` (`{node_match['evidence_status']}`)"
                    )
                with c_right:
                    st.markdown(f"**Connected Routes ({len(node_match['connected_routes'])})**")
                    if node_match["connected_routes"]:
                        st.dataframe(pd.DataFrame(node_match["connected_routes"]), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No connected routes visible under current filters.")
                    st.markdown(f"**Connected Herds ({node_match['connected_herd_count']})**")
                    if node_match["connected_herds"]:
                        st.dataframe(pd.DataFrame(node_match["connected_herds"]), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No simulated herds linked to this node.")
        else:
            st.caption(
                "Tip: Click any node or edge directly inside the interactive dual-pane explorer above to see live details "
                "immediately in the embedded Inspector, or pick a node/edge from the **Jump to Node / Edge** dropdowns above "
                "for full tabular and Plotly inspection here."
            )

    with tab_comp:
        if filtered.components:
            comp_df = pd.DataFrame(filtered.components)
            comp_df["sample_nodes"] = comp_df["sample_nodes"].apply(lambda x: ", ".join(x))
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("No nodes match the current filter combination.")

    with tab_tables:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Filtered Nodes ({len(filtered.nodes_gdf)})**")
            if not filtered.nodes_gdf.empty:
                cols = [c for c in ["node_id", "node_name", "node_type", "network_layer", "elevation_m", "landcover_class", "evidence_status", "is_synthetic"] if c in filtered.nodes_gdf.columns]
                st.dataframe(filtered.nodes_gdf[cols], use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**Filtered Edges ({len(filtered.edges_gdf)})**")
            if not filtered.edges_gdf.empty:
                cols = [c for c in ["edge_id", "source_node_id", "target_node_id", "edge_type", "distance_m", "elevation_gain_m", "average_slope", "season", "herd_count", "evidence_status"] if c in filtered.edges_gdf.columns]
                st.dataframe(filtered.edges_gdf[cols], use_container_width=True, hide_index=True)
