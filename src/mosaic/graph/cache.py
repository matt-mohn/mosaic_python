"""Caching utilities for preprocessed graph data."""

import logging
import math
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx

from mosaic.io.shapefile import shapefile_fingerprint
from mosaic.paths import cache_dir as _default_cache_dir

log = logging.getLogger("mosaic")


def get_cache_path(
    shapefile_path: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """Cache file path keyed by shapefile stem.

    Filename is only the key; content changes are caught by the fingerprint
    stored inside the pickle, not by the filename.
    """
    cdir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    return cdir / f"{Path(shapefile_path).stem}.pkl"


def save_cached_graph(
    graph: nx.Graph,
    cache_path: str | Path,
    shapefile_path: str | Path,
) -> None:
    """Pickle the adjacency graph alongside a fingerprint of its source.

    Best-effort: a cache write failure must never fail the caller.
    """
    cache_path = Path(cache_path)

    payload = {
        "cache_version": 3,
        "fingerprint": shapefile_fingerprint(shapefile_path),
        "nodes": list(graph.nodes()),
        "edges": list(graph.edges()),
        "shared_lengths": [d.get("shared_length") for _, _, d in graph.edges(data=True)],
        # Virtual bridge edges (added by bridge_components) carry an attribute
        # that list(graph.edges()) drops, so persist them separately and re-tag
        # on load.
        "virtual_edges": [
            (u, v) for u, v, d in graph.edges(data=True) if d.get("virtual")
        ],
        "populations": {n: graph.nodes[n].get("population", 0) for n in graph.nodes()},
    }

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
    except Exception as exc:
        log.warning(f"Could not write graph cache to {cache_path}: {exc}.")


def load_cached_graph(
    cache_path: str | Path,
    shapefile_path: str | Path,
    gdf: gpd.GeoDataFrame | None = None,
    *,
    populations=None,
    county_ids=None,
) -> nx.Graph | None:
    """Load the cached graph iff its fingerprint matches the live shapefile.

    Returns None on: missing cache file, unreadable cache, or fingerprint
    mismatch. Caller rebuilds. With gdf supplied, retain geometric adjacency
    and reconstruct virtual bridges using the currently selected data arrays.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        log.warning(f"Graph cache unreadable at {cache_path}: {exc}. Rebuilding.")
        return None

    # Real intersection lengths belong to the fingerprinted geometry;
    # virtual bridges are rebuilt from the selected population/county data.
    if not isinstance(payload, dict) or payload.get("cache_version") != 3:
        log.info(
            f"Graph cache for {Path(shapefile_path).name} has an unsupported format. Rebuilding."
        )
        return None

    live_fp = shapefile_fingerprint(shapefile_path)
    if not live_fp or payload.get("fingerprint") != live_fp:
        log.info(
            f"Graph cache stale for {Path(shapefile_path).name} "
            f"(fingerprint mismatch). Rebuilding."
        )
        return None

    try:
        G = nx.Graph()
        geometries = gdf.geometry.to_numpy() if gdf is not None else None
        for node in payload["nodes"]:
            attrs = {"population": payload["populations"].get(node, 0)}
            if geometries is not None:
                attrs["geometry"] = geometries[node]
            G.add_node(node, **attrs)
        G.add_edges_from(payload["edges"])
        lengths = payload["shared_lengths"]
        if len(lengths) != len(payload["edges"]):
            raise ValueError("edge lengths do not match cached edges")
        for (u, v), length in zip(payload["edges"], lengths):
            if length is None or not math.isfinite(length) or length < 0:
                raise ValueError("invalid cached intersection length")
            G[u][v]["shared_length"] = float(length)
        for u, v in payload["virtual_edges"]:
            if G.has_edge(u, v):
                G[u][v]["virtual"] = True
        if gdf is not None:
            from mosaic.graph.adjacency import apply_graph_inputs
            apply_graph_inputs(G, gdf, populations=populations, county_ids=county_ids)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, nx.NetworkXError) as exc:
        log.warning("Graph cache invalid at %s: %s. Rebuilding.", cache_path, exc)
        return None
    return G
