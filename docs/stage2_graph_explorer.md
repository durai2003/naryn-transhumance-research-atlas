# Stage 2 — Interactive Nodes, Edges & Synchronised Graph Explorer

## Overview

Stage 2 adds the **`🕸️ Network Analysis`** tab to `app.py` while preserving all existing 3D Spatiotemporal Atlas, Dataset Catalogue, Linked Open Data/SPARQL, and Provenance tabs.

## Synchronised Two-View Architecture

Served same-origin via `AtlasAssetServer` (`/network/` and `/network/payload.json`):

1. **Geographical View (Left Pane — CesiumJS 3D Globe)**:
   - Starts straight top-down (`-90°`) with one-click `📐 3D Tilt` toggle and `World` / `Naryn` / `Son-Kul` / `At-Bashi` presets.
   - Displays static Layer A (public settlements, OSM water points, physical road links) and Layer B (pastures, passes, resting points, corridors, water/settlement access) using distinct marker shapes and line styles.
   - Highlights selected routes in bold gold with origin/destination badges and highlights all incident routes when a node is selected.
2. **Abstract Graph View (Right Pane — Interactive High-DPI Canvas Multigraph)**:
   - Zero external CDN dependencies; supports pan, zoom, hover tooltips, curved parallel multigraph edges with directional arrowheads, connected-component filtering/highlighting, and 1-hop neighbour focus.
   - Provides 4 deterministic layouts computed via NetworkX (`Seasonal Layered`, `Geographic XY`, `Force / Spring`, `Kamada-Kawai`).
3. **Bidirectional Synchronisation & Feature Inspector**:
   - Clicking a node or edge on either view immediately highlights and synchronises the other view and updates the Inspector:
     - **Node Inspector**: ID, name, category, layer, coordinates, Copernicus DEM elevation & slope, ESA WorldCover land cover, connected component, clickable 1-hop neighbours, clickable connected routes, connected herds, source & provenance.
     - **Edge Inspector**: clickable origin/destination nodes, metric distance, inline Copernicus DEM elevation profile SVG, avg/max slope, elevation gain/loss, seasonal activity counts, route generation method, livestock compatibility, derived utilisation, source & provenance.
   - Multi-dimensional filters cover `node type`, `edge type`, `season`, `livestock type`, `climate scenario`, `evidence status`, plus a one-click **"Public sourced data only (hide synthetic)"** toggle.

## Ready for Stage 3–5

`src/network/network_loader.py` and `src/network/explorer.py` expose the filtered `nx.MultiDiGraph` cleanly so Stage 3 (centrality, shortest paths, temporal graph playback) and Stage 4 (climate resilience recalculations) attach directly inside the `🕸️ Network Analysis` tab without redesign.
