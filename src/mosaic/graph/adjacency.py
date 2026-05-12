"""Build adjacency graph from shapefile geometries."""

import geopandas as gpd
import igraph as ig
import networkx as nx
from shapely import STRtree
from shapely.geometry import Polygon, MultiPolygon


def nx_to_igraph(nxg: nx.Graph) -> ig.Graph:
    """Convert a NetworkX graph to an igraph Graph with original node IDs
    preserved as the `name` vertex attribute (sorted index order)."""
    node_list = sorted(nxg.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nxg.edges()]
    g = ig.Graph(n=len(node_list), edges=edges, directed=False)
    g.vs["name"] = node_list
    return g


def build_adjacency_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build a precinct adjacency graph from geometries.

    Two precincts are adjacent if their geometries touch or intersect
    (excluding point-only contacts).

    Args:
        gdf: GeoDataFrame with geometry column. Index should be 0..N-1.

    Returns:
        networkx.Graph where:
        - Nodes are precinct indices (0 to N-1)
        - Node attributes: 'population', 'geometry'
        - Edges connect adjacent precincts
    """
    G = nx.Graph()

    # Add nodes with attributes
    for idx, row in gdf.iterrows():
        G.add_node(
            idx,
            population=row.get("population", 0),
            geometry=row.geometry,
        )

    # Build spatial index for efficient adjacency detection
    geometries = gdf.geometry.tolist()
    tree = STRtree(geometries)

    # Find adjacencies
    for i, geom in enumerate(geometries):
        # Query tree for potential neighbors
        candidate_indices = tree.query(geom)

        for j in candidate_indices:
            if j <= i:
                continue  # Avoid duplicates and self-loops

            other_geom = geometries[j]

            # Check if they actually touch (not just bounding box overlap)
            if geom.touches(other_geom) or geom.intersects(other_geom):
                # Exclude point-only contacts (corners touching)
                intersection = geom.intersection(other_geom)
                if not intersection.is_empty:
                    # Accept if intersection has length (shared edge) or area (overlap)
                    if hasattr(intersection, "length") and intersection.length > 0:
                        G.add_edge(i, j)
                    elif hasattr(intersection, "area") and intersection.area > 0:
                        G.add_edge(i, j)

    return G
