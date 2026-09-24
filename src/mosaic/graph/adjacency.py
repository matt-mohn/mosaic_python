"""Build adjacency graph from shapefile geometries."""

import logging
import warnings

import geopandas as gpd
import igraph as ig
import networkx as nx
import numpy as np
from shapely import STRtree

log = logging.getLogger("mosaic")


def nx_to_igraph(nxg: nx.Graph) -> ig.Graph:
    """Convert a NetworkX graph to an igraph Graph with original node IDs
    preserved as the `name` vertex attribute (sorted index order).

    The per-edge ``virtual`` flag (set by bridge_components) is carried onto
    ``g.es["virtual"]`` in edge-id order so GraphContext can mask virtual edges
    out of the cut-edge count. Endpoints and flags are collected in the same
    NetworkX traversal order, which igraph preserves in get_edgelist()."""
    node_list = sorted(nxg.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = []
    virtual = []
    for u, v, data in nxg.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        virtual.append(bool(data.get("virtual", False)))
    g = ig.Graph(n=len(node_list), edges=edges, directed=False)
    g.vs["name"] = node_list
    if edges:
        g.es["virtual"] = virtual
    return g


def build_adjacency_graph(gdf: gpd.GeoDataFrame, *, populations=None,
                          county_ids=None) -> nx.Graph:
    """
    Build a precinct adjacency graph from geometries.

    A real edge joins two precincts whose geometries touch or intersect
    through positive length or area. Point-only contacts are excluded.
    Disconnected components may later receive non-geographic virtual edges.

    Args:
        gdf: GeoDataFrame with geometry column. Nodes follow row position.
        populations: Selected population values in row order. If omitted,
            use the literal 'population' column, or zeros if absent.
        county_ids: Selected county IDs in row order. None disables county
            preference when choosing island bridges; no column is auto-detected.

    Returns:
        networkx.Graph where:
        - Nodes are precinct indices (0 to N-1)
        - Node attributes: 'population', 'geometry'
        - Real edges retain intersection lengths for compactness preparation
        - Virtual edges connect otherwise disconnected components
    """
    G = nx.Graph()

    # Build spatial index for efficient adjacency detection
    geometries = gdf.geometry.tolist()
    G.add_nodes_from((i, {"geometry": geom}) for i, geom in enumerate(geometries))
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
                    # Preserve intersection length, including the perimeter of
                    # polygon overlaps, for the compactness geometry stage.
                    length = float(intersection.length)
                    if length > 0 or intersection.area > 0:
                        G.add_edge(i, j, shared_length=length)

    apply_graph_inputs(G, gdf, populations=populations, county_ids=county_ids)
    return G


def apply_graph_inputs(G, gdf, *, populations=None, county_ids=None):
    """Apply selected data and rebuild virtual bridges, retaining real edges.

    Cached geometry adjacency is independent of column selection. Population
    chooses the mainland, and county IDs influence bridges, so both must be
    applied again when loading a cached graph.
    """
    if set(G.nodes) != set(range(len(gdf))):
        raise ValueError("Graph nodes must match precinct row positions")
    if populations is None:
        populations = gdf["population"].to_numpy() if "population" in gdf else np.zeros(len(gdf))
    populations = np.asarray(populations)
    if (populations.shape != (len(gdf),) or not np.all(np.isfinite(populations))
            or np.any(populations < 0)):
        raise ValueError("Population must contain one finite, nonnegative value per precinct")
    if county_ids is not None:
        county_ids = np.asarray(county_ids)
        if county_ids.shape != (len(gdf),):
            raise ValueError("County IDs must contain one value per precinct")
    nx.set_node_attributes(G, dict(enumerate(populations)), "population")
    G.remove_edges_from([(u, v) for u, v, attrs in G.edges(data=True) if attrs.get("virtual")])
    added = bridge_components(G, gdf, county_ids=county_ids)
    if added:
        log.info(
            f"Bridged {len(added)} disconnected component(s) with virtual edges"
        )

    return added


def _component_population(G: nx.Graph, comp) -> float:
    """Total population of a component (0 when the attribute is missing)."""
    return sum(float(G.nodes[n].get("population", 0) or 0) for n in comp)


def _best_bridge(comp, s_arr, s_cents, geoms, cents, county_ids, k):
    """Best (tier, distance, o, s) virtual edge from a component to the set S.

    tier 0 = the S endpoint shares the island node's county, tier 1 = any.
    Tier 0 checks every same-county endpoint in S. Tier 1 computes exact
    polygon distance only for up to `k` endpoints prefiltered by centroid
    distance. Distance ties use endpoint node IDs. Returns None when S is empty.
    """
    if len(s_arr) == 0:
        return None
    best_t0 = None  # (dist, o, s)
    best_t1 = None
    for o in comp:
        oc = cents[o]
        og = geoms[o]
        d2 = (s_cents[:, 0] - oc[0]) ** 2 + (s_cents[:, 1] - oc[1]) ** 2
        # Exact polygon distance is the cost within a candidate set prefiltered
        # to at most k endpoints by centroid distance.
        if len(s_arr) > k:
            near = np.argpartition(d2, k)[:k]
        else:
            near = np.arange(len(s_arr))
        for j in near.tolist():
            s = int(s_arr[j])
            dist = og.distance(geoms[s])
            cand = (dist, o, s)
            if best_t1 is None or cand < best_t1:
                best_t1 = cand
        if county_ids is not None:
            same = s_arr[county_ids[s_arr] == county_ids[o]]
            for s in same.tolist():
                s = int(s)
                dist = og.distance(geoms[s])
                cand = (dist, o, s)
                if best_t0 is None or cand < best_t0:
                    best_t0 = cand
    if best_t0 is not None:
        return (0, best_t0[0], best_t0[1], best_t0[2])
    if best_t1 is not None:
        return (1, best_t1[0], best_t1[1], best_t1[2])
    return None


def bridge_components(
    G: nx.Graph,
    gdf: gpd.GeoDataFrame,
    county_ids: np.ndarray | None = None,
    k_prefilter: int = 64,
) -> list[tuple[int, int]]:
    """Add virtual edges between disconnected graph components when possible.

    A degree-0 precinct (lone island) and an N-precinct island are the same
    case: each is a connected component that isn't the mainland. The mainland
    is the component with the largest population; every other component is
    linked into the growing connected set by one virtual edge. An N-precinct
    component needs one bridge because its members are already connected.

    Bridge selection (Prim-style growth over components):
      * tier 0 — an endpoint in the same county as the island, if one exists;
      * tier 1 — otherwise choose from up to `k_prefilter` endpoints nearest
        by centroid distance;
      * minimum exact polygon-to-polygon distance among the tier's candidates;
      * deterministic id tie-break.

    Bridge selection uses no RNG. The shapefile on disk is not modified. Added
    edges carry ``virtual=True``.

    Returns the list of (u, v) virtual edges added (empty if already connected).
    """
    components = [sorted(c) for c in nx.connected_components(G)]
    if len(components) <= 1:
        return []

    # Mainland = largest population; tie-break on smallest node id.
    components.sort(key=lambda c: (-_component_population(G, c), c[0]))

    geoms = gdf.geometry.values
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*geographic CRS.*", category=UserWarning
        )
        cents = np.column_stack([
            gdf.geometry.centroid.x.values,
            gdf.geometry.centroid.y.values,
        ]).astype(np.float64)

    n = len(gdf)
    in_S = np.zeros(n, dtype=bool)
    s_nodes = list(components[0])
    for node in s_nodes:
        in_S[node] = True

    remaining = components[1:]
    added: list[tuple[int, int]] = []

    # Geometry stays fixed as S grows. Retain each component's best candidate
    # and compare it only with candidates from the component just added to S.
    # This avoids repeating exact polygon-distance calculations against nodes
    # already considered. The centroid prefilter runs per added component.
    best_cache: list = [None] * len(remaining)
    grown = s_nodes  # first round: every node of the mainland is new to S

    while remaining:
        g_arr = np.fromiter(grown, dtype=np.int64, count=len(grown))
        g_cents = cents[g_arr]
        best = None  # (sort_key, position_in_remaining, (o, s))
        for pos, comp in enumerate(remaining):
            cand = _best_bridge(
                comp, g_arr, g_cents, geoms, cents, county_ids, k_prefilter
            )
            prev = best_cache[pos]
            if cand is not None and (prev is None or cand < prev):
                best_cache[pos] = cand
            cand = best_cache[pos]
            if cand is None:
                continue
            tier, dist, o, s = cand
            key = (tier, dist, min(o, s), max(o, s))
            if best is None or key < best[0]:
                best = (key, pos, (o, s))
        if best is None:
            # No finite bridge (degenerate geometry); stop rather than loop.
            log.warning("bridge_components: no valid bridge found; "
                        f"{len(remaining)} component(s) left unconnected")
            break
        pos = best[1]
        o, s = best[2]
        G.add_edge(o, s, virtual=True, shared_length=0.0)
        added.append((o, s))
        for node in remaining[pos]:
            in_S[node] = True
            s_nodes.append(node)
        grown = remaining.pop(pos)
        best_cache.pop(pos)

    return added
