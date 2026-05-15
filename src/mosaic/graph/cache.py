"""Caching utilities for preprocessed graph data."""

import logging
import pickle
from pathlib import Path

import networkx as nx
import geopandas as gpd

from mosaic.io.shapefile import shapefile_fingerprint

log = logging.getLogger("mosaic")


def get_cache_path(shapefile_path: str | Path, cache_dir: str | Path = "cache") -> Path:
    """Cache file path keyed by shapefile stem.

    Filename is only the key; content changes are caught by the fingerprint
    stored inside the pickle, not by the filename.
    """
    return Path(cache_dir) / f"{Path(shapefile_path).stem}.pkl"


def save_cached_graph(
    graph: nx.Graph,
    cache_path: str | Path,
    shapefile_path: str | Path,
) -> None:
    """Pickle the adjacency graph alongside a fingerprint of its source."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "fingerprint": shapefile_fingerprint(shapefile_path),
        "nodes": list(graph.nodes()),
        "edges": list(graph.edges()),
        "populations": {n: graph.nodes[n].get("population", 0) for n in graph.nodes()},
    }

    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


def load_cached_graph(
    cache_path: str | Path,
    shapefile_path: str | Path,
    gdf: gpd.GeoDataFrame | None = None,
) -> nx.Graph | None:
    """Load the cached graph iff its fingerprint matches the live shapefile.

    Returns None on: missing cache file, unreadable cache, or fingerprint
    mismatch. Caller rebuilds.
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

    live_fp = shapefile_fingerprint(shapefile_path)
    if not live_fp or payload.get("fingerprint") != live_fp:
        log.info(
            f"Graph cache stale for {Path(shapefile_path).name} "
            f"(fingerprint mismatch). Rebuilding."
        )
        return None

    G = nx.Graph()
    for node in payload["nodes"]:
        attrs = {"population": payload["populations"].get(node, 0)}
        if gdf is not None and node < len(gdf):
            attrs["geometry"] = gdf.iloc[node].geometry
        G.add_node(node, **attrs)
    G.add_edges_from(payload["edges"])
    return G
