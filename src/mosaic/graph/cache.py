"""Caching utilities for preprocessed graph data."""

import hashlib
import pickle
from pathlib import Path

import networkx as nx
import geopandas as gpd


def get_cache_path(shapefile_path: str | Path, cache_dir: str | Path = "cache") -> Path:
    """
    Generate a cache file path based on shapefile path.

    Args:
        shapefile_path: Path to the original shapefile
        cache_dir: Directory to store cache files

    Returns:
        Path to the cache file
    """
    shapefile_path = Path(shapefile_path)
    cache_dir = Path(cache_dir)

    # Use shapefile name as base
    name = shapefile_path.stem
    cache_path = cache_dir / f"{name}.pkl"

    return cache_path


def save_cached_graph(graph: nx.Graph, cache_path: str | Path) -> None:
    """
    Save a preprocessed graph to cache.

    Args:
        graph: The adjacency graph to cache
        cache_path: Path to save the cache file
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove geometry objects before pickling (they're large and we can reload)
    # Store only the essential structure
    graph_data = {
        "nodes": list(graph.nodes()),
        "edges": list(graph.edges()),
        "populations": {n: graph.nodes[n].get("population", 0) for n in graph.nodes()},
    }

    with open(cache_path, "wb") as f:
        pickle.dump(graph_data, f)


def load_cached_graph(cache_path: str | Path, gdf: gpd.GeoDataFrame | None = None) -> nx.Graph | None:
    """
    Load a cached graph if it exists.

    Args:
        cache_path: Path to the cache file
        gdf: Optional GeoDataFrame to restore geometry attributes

    Returns:
        The loaded graph, or None if cache doesn't exist
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return None

    with open(cache_path, "rb") as f:
        graph_data = pickle.load(f)

    G = nx.Graph()

    # Restore nodes
    for node in graph_data["nodes"]:
        attrs = {"population": graph_data["populations"].get(node, 0)}
        if gdf is not None and node < len(gdf):
            attrs["geometry"] = gdf.iloc[node].geometry
        G.add_node(node, **attrs)

    # Restore edges
    G.add_edges_from(graph_data["edges"])

    return G
