"""Mini Transhumance Research Data Atlas — 3D Spatiotemporal CesiumJS + Streamlit Application.

Launch from repository root:
    python3 -m streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdflib import Graph, Namespace, RDF

from src.atlas.cesium_bridge import render_cesium_globe
from src.atlas.cesium_server import get_asset_server
from src.atlas.data_access import AtlasData, load_atlas_data
from src.atlas.temporal import (
    build_cesium_payload,
    resolve_corridor_profile,
    resolve_herd_snapshot,
)
from src.network.ui_stage2 import render_network_analysis_tab

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
TA = Namespace("https://example.org/transhumance-atlas/ontology#")

st.set_page_config(
    page_title="3D Transhumance Research Data Atlas — Naryn, Kyrgyzstan",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep page header and tabs cleanly below Streamlit's top toolbar (never submerged)
st.markdown(
    "<style>.block-container{padding-top:3.5rem;padding-bottom:1.4rem;}</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_data(root_string: str) -> AtlasData:
    return load_atlas_data(Path(root_string))


def _render_climate_cursor_chart(data: AtlasData, scenario_id: str, authoritative_date: str) -> None:
    sc = data.scenario_climate
    if sc.empty:
        st.info("Scenario climate table not generated yet. Run `python3 scripts/generate_herd_simulation.py --root . --seed 42`.")
        return
    base = sc.loc[sc["scenario_id"] == "SCENARIO_BASELINE"].sort_values("date_str")
    curr = sc.loc[sc["scenario_id"] == scenario_id].sort_values("date_str")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=curr["date_str"], y=curr["scenario_precip_mm_day"], name="Precip (mm/day)",
        marker_color="rgba(56, 189, 248, 0.45)", yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=base["date_str"], y=base["scenario_t2m_c"], name="Historical NASA POWER T2M (°C)",
        line=dict(color="#94a3b8", width=1.6, dash="dot"),
    ))
    if scenario_id != "SCENARIO_BASELINE":
        fig.add_trace(go.Scatter(
            x=curr["date_str"], y=curr["scenario_t2m_c"], name=f"{scenario_id} T2M (°C)",
            line=dict(color="#f97316", width=2.2),
        ))
    fig.add_vline(x=authoritative_date, line_width=2, line_dash="dash", line_color="#eab308")
    fig.update_layout(
        title=f"Synchronized Daily Climate ({authoritative_date} · NASA POWER 0.5°×0.625° Grid)",
        height=310, margin=dict(l=30, r=30, t=42, b=30),
        yaxis=dict(title="Temperature (°C)"),
        yaxis2=dict(title="Precip (mm/day)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.22),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_elevation_profile(data: AtlasData, snapshot: dict) -> None:
    herd = snapshot.get("herd")
    if not herd:
        st.info("No herd selected or visible under the current filters.")
        return

    route_id = herd.get("route_id")
    origin_pid = herd.get("current_pasture_id")
    dest_pid = herd.get("next_pasture_id")
    resolved = resolve_corridor_profile(
        data,
        route_id,
        origin_pasture_id=origin_pid,
        dest_pasture_id=dest_pid,
    )
    if resolved is None or not resolved.get("profile"):
        st.info(
            f"No modelled corridor elevation profile is available for **{herd['herd_id']}** "
            f"(route `{route_id}`: `{origin_pid}` → `{dest_pid}`) on `{snapshot['date']}`."
        )
        return

    pdf = pd.DataFrame(resolved["profile"])
    role_tag = "Active Corridor" if herd.get("is_moving") else "Next Planned Corridor"
    title = (
        f"Terrain Elevation Profile · {resolved['route_id']} "
        f"({herd['herd_id']} · {role_tag}: {resolved['origin_pasture_id']} → {resolved['destination_pasture_id']} · "
        f"{resolved['distance_km']} km, +{resolved['elevation_gain_m']}m)"
    )
    fig = px.area(
        pdf, x="distance_km", y="elevation_m",
        title=title,
        labels={"distance_km": "Distance (km)", "elevation_m": "Elevation (m)"},
    )
    fig.update_traces(line_color="#a855f7", fillcolor="rgba(168, 85, 247, 0.25)")
    fig.update_layout(height=310, margin=dict(l=30, r=20, t=42, b=30))
    st.plotly_chart(fig, use_container_width=True)


def _render_scenario_comparison(data: AtlasData, scenario_id: str, selected_season: str) -> None:
    if data.suitability_scores.empty:
        st.info("Suitability scores not available yet.")
        return
    df = data.suitability_scores
    base = df.loc[(df["scenario_id"] == "SCENARIO_BASELINE") & (df["season"] == selected_season)][
        ["pasture_id", "pasture_name", "elevation_m", "suitability_score"]
    ].rename(columns={"suitability_score": "baseline_score"})
    comp = df.loc[(df["scenario_id"] == scenario_id) & (df["season"] == selected_season)][
        ["pasture_id", "suitability_score"]
    ].rename(columns={"suitability_score": "scenario_score"})
    merged = base.merge(comp, on="pasture_id", how="inner")
    merged["delta"] = (merged["scenario_score"] - merged["baseline_score"]).round(3)
    merged = merged.sort_values("scenario_score", ascending=False)

    fig = px.bar(
        merged, x="pasture_id", y=["baseline_score", "scenario_score"], barmode="group",
        title=f"Pasture Suitability Comparison ({selected_season.title()}) · Baseline vs {scenario_id}",
        labels={"value": "Suitability Score (0–1)", "pasture_id": "Synthetic Candidate Pasture"},
    )
    fig.update_layout(height=310, margin=dict(l=30, r=20, t=42, b=30), legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, use_container_width=True)

    if not data.movement_events.empty:
        rerouted = data.movement_events.loc[
            (data.movement_events["climate_scenario"] == scenario_id)
            & (data.movement_events["scenario_changed_route"])
        ]["herd_id"].nunique()
        st.caption(
            f"Illustrative climate-responsive simulation: **{rerouted}** synthetic herd group(s) changed destination "
            f"or corridor under `{scenario_id}` relative to `SCENARIO_BASELINE`."
        )


@st.cache_resource(show_spinner=False)
def _load_rdf_graph_cached(graph_path_str: str, mtime: float) -> Graph:
    del mtime
    gp = Path(graph_path_str)
    g = Graph()
    g.parse(gp, format="turtle" if gp.suffix == ".ttl" else "json-ld")
    return g


def _rdf_graph_needs_rebuild(root: Path, graph_path: Path | None) -> bool:
    syn_sources = [
        root / "data/synthetic/pastures.geojson",
        root / "data/synthetic/migration_routes.geojson",
        root / "data/synthetic/herds.csv",
        root / "data/synthetic/movement_events.csv",
        root / "data/synthetic/suitability_scores.csv",
    ]
    existing_syn = [p for p in syn_sources if p.exists() and p.stat().st_size > 0]
    if not existing_syn:
        return False
    if graph_path is None or not graph_path.exists():
        return True
    graph_mtime = graph_path.stat().st_mtime
    if any(p.stat().st_mtime > graph_mtime for p in existing_syn):
        return True
    return False


def _render_linked_data_explorer(data: AtlasData) -> None:
    from src.linked_data.build_graph import build_from_root

    ttl_path = data.root / "linked_data" / "research_graph.ttl"
    graph_path = ttl_path if ttl_path.exists() else data.graph_path

    c_hdr, c_btn = st.columns([4.2, 1.0])
    with c_btn:
        manual_rebuild = st.button("Rebuild RDF Graph", key="btn_rebuild_rdf")

    # Auto-rebuild if synthetic simulation files exist and graph is missing or older, or user clicked Rebuild
    if manual_rebuild or _rdf_graph_needs_rebuild(data.root, graph_path):
        with st.spinner("Building RDF Knowledge Graph (pastures, corridors, herds, events & suitability)…"):
            build_from_root(data.root)
            _load_rdf_graph_cached.clear()
        graph_path = ttl_path if ttl_path.exists() else data.graph_path

    if graph_path is None or not graph_path.exists():
        st.info("RDF graph not built yet. Click **Rebuild RDF Graph** or run `python3 scripts/run_linked_data.py --root .`.")
        return

    graph = _load_rdf_graph_cached(str(graph_path), graph_path.stat().st_mtime)

    # Also auto-rebuild once if an older pre-simulation TTL is still loaded (0 pastures/herds/events while synthetic files exist)
    has_syn = (data.root / "data/synthetic/pastures.geojson").exists() and (data.root / "data/synthetic/herds.csv").exists()
    pasture_cnt = sum(1 for _ in graph.subjects(RDF.type, TA.Pasture))
    herd_cnt = sum(1 for _ in graph.subjects(RDF.type, TA.HerdGroup))
    event_cnt = sum(1 for _ in graph.subjects(RDF.type, TA.MovementEvent))
    if has_syn and (pasture_cnt == 0 or herd_cnt == 0 or event_cnt == 0) and not st.session_state.get("_rdf_auto_rebuilt"):
        st.session_state["_rdf_auto_rebuilt"] = True
        with st.spinner("Updating stale RDF graph with candidate pastures, herds & movement events…"):
            build_from_root(data.root)
            _load_rdf_graph_cached.clear()
        graph = _load_rdf_graph_cached(str(ttl_path), ttl_path.stat().st_mtime)
        graph_path = ttl_path

    classes = {
        "Locations": TA.Location,
        "Climate Obs.": TA.ClimateObservation,
        "Candidate Pastures": TA.Pasture,
        "Corridors/Routes": TA.Route,
        "Herd Groups": TA.HerdGroup,
        "Movement Events": TA.MovementEvent,
        "Suitability Assessments": TA.PastureSuitabilityAssessment,
        "Datasets": TA.Dataset,
    }
    cols = st.columns(4)
    for i, (label, cls) in enumerate(classes.items()):
        cols[i % 4].metric(label, sum(1 for _ in graph.subjects(RDF.type, cls)))
    st.caption(f"Graph: `{graph_path.relative_to(data.root)}` · **{len(graph):,}** triples")

    q_dir = data.root / "queries"
    rq_files = sorted(q_dir.glob("0*.rq")) if q_dir.exists() else []
    if rq_files:
        chosen = st.selectbox("Execute packaged SPARQL 1.1 query", [p.name for p in rq_files])
        q_text = (q_dir / chosen).read_text(encoding="utf-8")
        st.code(q_text, language="sparql")
        if st.button("Run SPARQL query"):
            res = graph.query(q_text)
            rows = [{str(v): (str(r[v]) if r[v] is not None else None) for v in res.vars} for r in res]
            st.caption(f"Returned {len(rows)} rows (showing up to 100).")
            st.dataframe(pd.DataFrame(rows[:100]), use_container_width=True, hide_index=True)


def _render_synced_charts_body(
    data: AtlasData,
    scenario_id: str,
    sidebar_date: str,
    species_filter: list[str],
    sidebar_herd_id: str,
) -> None:
    """Render climate cursor and corridor elevation profile locked to the authoritative timeline state."""
    server = get_asset_server(ROOT)
    ui_state = server.get_ui_state()
    auth_date = ui_state.get("date") or sidebar_date
    # Honour explicit herd selection from sidebar or clicked herd from CesiumJS
    if sidebar_herd_id != "All herds":
        auth_herd = sidebar_herd_id
    else:
        auth_herd = ui_state.get("herd_id") or "All herds"

    snapshot = resolve_herd_snapshot(
        data,
        scenario_id=scenario_id,
        selected_date=auth_date,
        species_filter=species_filter,
        selected_herd_id=auth_herd,
    )
    c1, c2 = st.columns(2)
    with c1:
        _render_climate_cursor_chart(data, scenario_id, snapshot["date"])
    with c2:
        _render_elevation_profile(data, snapshot)


if hasattr(st, "fragment"):
    @st.fragment(run_every=1.0)
    def _synced_charts_panel(
        data: AtlasData,
        scenario_id: str,
        sidebar_date: str,
        species_filter: list[str],
        sidebar_herd_id: str,
    ) -> None:
        _render_synced_charts_body(data, scenario_id, sidebar_date, species_filter, sidebar_herd_id)
else:
    def _synced_charts_panel(
        data: AtlasData,
        scenario_id: str,
        sidebar_date: str,
        species_filter: list[str],
        sidebar_herd_id: str,
    ) -> None:
        _render_synced_charts_body(data, scenario_id, sidebar_date, species_filter, sidebar_herd_id)


def main() -> None:
    data = _load_data(str(ROOT))

    with st.sidebar:
        st.header("3D Atlas Controls")
        scenario_options = (
            sorted(data.scenario_climate["scenario_id"].unique().tolist())
            if not data.scenario_climate.empty
            else ["SCENARIO_BASELINE", "SCENARIO_WARMER_DRIER", "SCENARIO_EARLY_SNOW"]
        )
        scenario_id = st.selectbox("Climate Scenario", scenario_options, index=0)

        available_dates = (
            sorted(data.scenario_climate.loc[data.scenario_climate["scenario_id"] == scenario_id, "date_str"].unique().tolist())
            if not data.scenario_climate.empty
            else ["2024-07-02"]
        )
        default_idx = min(len(available_dates) // 2, len(available_dates) - 1)
        selected_date = st.select_slider("Synchronized Simulation Date", options=available_dates, value=available_dates[default_idx])

        selected_season = st.selectbox("Suitability Season View", ["summer", "autumn", "spring", "winter"], index=0)
        all_species = sorted(data.synthetic_herds["species"].unique().tolist()) if not data.synthetic_herds.empty else ["sheep", "goat", "cattle", "horse", "mixed"]
        species_filter = st.multiselect("Species Filter", all_species, default=all_species)

        herd_ids = ["All herds"] + (data.synthetic_herds["herd_id"].tolist() if not data.synthetic_herds.empty else [])
        selected_herd_id = st.selectbox("Focus / Track Herd", herd_ids)

        st.divider()
        st.subheader("3D Globe & Layers")
        camera_mode = st.selectbox("Initial View Mode", ["regional_3d", "global", "central_asia", "son_kul", "at_bashi"], index=0)
        basemap = st.selectbox("Global Basemap", ["esri_satellite", "esri_topo", "opentopo", "osm"], index=0)
        vertical_exaggeration = st.slider("Terrain Relief Exaggeration", 1.0, 3.0, 1.35, 0.05)
        playback_speed = st.slider("Default Playback Speed (days/s)", 1, 30, 6)

        show_layers = {
            "boundary": st.checkbox("PUBLIC · Naryn ADM1 Boundary", value=True),
            "osm_roads": st.checkbox("PUBLIC · OSM Roads & Tracks (context)", value=True),
            "osm_water": st.checkbox("PUBLIC · OSM Rivers & Lakes", value=True),
            "osm_settlements": st.checkbox("PUBLIC · OSM Settlements", value=True),
            "pastures": st.checkbox("SYNTHETIC · Candidate Pastures", value=True),
            "corridors": st.checkbox("SYNTHETIC · Least-Cost Corridors", value=True),
            "trails": st.checkbox("SYNTHETIC · Animated Herd Trails", value=True),
            "google_3d_tiles": st.checkbox("Optional · Google Photorealistic 3D Tiles", value=False),
        }

    # Push sidebar changes into the shared asset server state so sidebar & CesiumJS stay 100% synced
    sidebar_sig = (scenario_id, selected_date, selected_herd_id, tuple(species_filter))
    if st.session_state.get("_last_sidebar_sig") != sidebar_sig:
        st.session_state["_last_sidebar_sig"] = sidebar_sig
        get_asset_server(ROOT).set_ui_state(
            date=selected_date,
            herd_id=None if selected_herd_id == "All herds" else selected_herd_id,
            scenario_id=scenario_id,
            source="streamlit",
        )

    st.markdown(
        f"#### Mini Transhumance Research Data Atlas (3D) · Naryn Region, Kyrgyzstan · Scenario: `{scenario_id}`"
    )

    atlas_tab, network_tab, rdf_tab = st.tabs([
        "🌍 3D Spatiotemporal Atlas",
        "🕸️ Network Analysis",
        "🔗 Linked Open Data & SPARQL",
    ])

    with atlas_tab:
        payload = build_cesium_payload(
            data,
            scenario_id=scenario_id,
            selected_date=selected_date,
            species_filter=species_filter,
            selected_herd_id=selected_herd_id,
            camera_mode=camera_mode,
            basemap=basemap,
            vertical_exaggeration=vertical_exaggeration,
            show_layers=show_layers,
            playback_speed=playback_speed,
            cesium_ion_token=os.getenv("CESIUM_ION_ACCESS_TOKEN", ""),
            google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", ""),
        )
        # Unified synchronized 3D globe + live right-hand Selected Herd & State panel
        render_cesium_globe(ROOT, payload, height=690)

        # Synchronized bottom climate cursor + exact corridor elevation profile (polls authoritative Cesium timeline state)
        _synced_charts_panel(data, scenario_id, selected_date, species_filter, selected_herd_id)

        _render_scenario_comparison(data, scenario_id, selected_season)

    with network_tab:
        render_network_analysis_tab(
            ROOT,
            basemap=basemap,
            vertical_exaggeration=vertical_exaggeration,
            default_scenario=scenario_id,
        )

    with rdf_tab:
        st.subheader("RDF Knowledge Graph & Executable SPARQL 1.1")
        _render_linked_data_explorer(data)


if __name__ == "__main__":
    main()
