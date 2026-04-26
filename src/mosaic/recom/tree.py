"""Spanning tree operations for ReCom algorithm using igraph."""

import numpy as np
import igraph as ig
from typing import Optional, Tuple


def random_spanning_tree_ig(graph: ig.Graph) -> ig.Graph:
    """Generate a random spanning tree (random weights + MST). Used by partition."""
    n_edges = graph.ecount()
    if n_edges == 0:
        raise ValueError("Cannot create spanning tree of graph with no edges")
    weights = np.random.random(n_edges)
    mst_edges = graph.spanning_tree(weights=weights, return_tree=False)
    return graph.subgraph_edges(mst_edges)


def find_balanced_cut_ig(
    graph: ig.Graph,
    populations: np.ndarray,
    target_pop: float,
    tolerance: float,
    max_attempts: int = 100,
    one_sided: bool = False,
    county_array: Optional[np.ndarray] = None,
    county_bias: float = 1.0,
) -> list | None:
    """Find a balanced bipartition of the graph using random spanning trees."""
    if "name" in graph.vs.attributes():
        node_ids = np.array(graph.vs["name"])
    else:
        node_ids = np.arange(graph.vcount())

    sub_pops = populations[node_ids]
    total_pop = sub_pops.sum()
    min_pop = target_pop * (1 - tolerance)
    max_pop = target_pop * (1 + tolerance)

    n = graph.vcount()
    m = graph.ecount()
    subtree_pops = np.empty(n, dtype=np.float64)

    # Fetch the subgraph edge list once — reused every attempt to build the
    # MST adjacency list without creating a new igraph Graph object per attempt.
    edge_list = graph.get_edgelist()  # list of (u, v) indexed by edge id

    # Precompute cross-county mask once; reused in every spanning-tree attempt.
    cross_county_mask = None
    if county_array is not None and county_bias != 1.0 and m > 0:
        eu_arr = np.array([e[0] for e in edge_list], dtype=np.int32)
        ev_arr = np.array([e[1] for e in edge_list], dtype=np.int32)
        cross_county_mask = (county_array[node_ids[eu_arr]] !=
                             county_array[node_ids[ev_arr]])

    for _ in range(max_attempts):
        weights = np.random.random(m)
        if cross_county_mask is not None:
            weights[cross_county_mask] *= county_bias
        mst_edge_ids = graph.spanning_tree(weights=weights, return_tree=False)

        # Build MST adjacency list from edge indices directly.
        # Avoids graph.subgraph_edges() and all per-node tree.neighbors() calls.
        tree_adj: list[list[int]] = [[] for _ in range(n)]
        for eid in mst_edge_ids:
            eu, ev = edge_list[eid]
            tree_adj[eu].append(ev)
            tree_adj[ev].append(eu)

        # Prefer non-leaf roots so there are valid cuts to explore.
        candidates = [i for i, adj in enumerate(tree_adj) if len(adj) > 1]
        if not candidates:
            continue
        root = candidates[np.random.randint(len(candidates))]

        # BFS using the prebuilt adjacency list — no igraph calls inside loop.
        parent = np.full(n, -1, dtype=np.int32)
        children: list[list[int]] = [[] for _ in range(n)]
        visited = [False] * n
        visited[root] = True
        bfs_queue = [root]
        head = 0
        while head < len(bfs_queue):
            v = bfs_queue[head]; head += 1
            for nb in tree_adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = v
                    children[v].append(nb)
                    bfs_queue.append(nb)

        # Subtree populations: process reverse BFS order (≡ post-order on trees).
        subtree_pops[:] = sub_pops
        for v in reversed(bfs_queue):
            p = parent[v]
            if p >= 0:
                subtree_pops[p] += subtree_pops[v]

        # Vectorised validity check across all non-root nodes.
        other_pops = total_pop - subtree_pops
        if one_sided:
            tree_valid = (subtree_pops >= min_pop) & (subtree_pops <= max_pop)
            other_valid = (other_pops >= min_pop) & (other_pops <= max_pop)
            valid = tree_valid | other_valid
        else:
            valid = ((subtree_pops >= min_pop) & (subtree_pops <= max_pop) &
                     (other_pops >= min_pop) & (other_pops <= max_pop))
        valid[root] = False

        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) > 0:
            v = valid_indices[0]
            subtree_nodes = _get_subtree_nodes_fast(v, children, n)
            if one_sided and not (min_pop <= subtree_pops[v] <= max_pop):
                mask = np.ones(n, dtype=np.bool_)
                mask[subtree_nodes] = False
                return node_ids[mask].tolist()
            return node_ids[subtree_nodes].tolist()

    return None


def _get_subtree_nodes_fast(root: int, children: list, n: int) -> np.ndarray:
    """Get subtree nodes using pre-allocated array."""
    result = np.zeros(n, dtype=np.int32)
    result[0] = root
    count = 1
    head = 0
    while head < count:
        v = result[head]
        head += 1
        for child in children[v]:
            result[count] = child
            count += 1
    return result[:count]


def _get_subtree_nodes(root: int, children: list) -> list:
    """Get all nodes in subtree rooted at given node."""
    result = []
    stack = [root]
    while stack:
        v = stack.pop()
        result.append(v)
        stack.extend(children[v])
    return result


# ── Legacy NetworkX implementations (kept for compatibility) ──────────────────

import networkx as nx
from collections import deque


def random_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """NetworkX version — slower, kept for compatibility."""
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        raise ValueError("Cannot create spanning tree of graph with no edges")
    weights = np.random.random(n_edges)
    for i, (u, v) in enumerate(graph.edges()):
        graph[u][v]["weight"] = weights[i]
    return nx.minimum_spanning_tree(graph)


def get_tree_structure(tree: nx.Graph, root: int) -> Tuple[dict, dict]:
    """Compute parent and children for a rooted tree via BFS."""
    parent = {root: None}
    children = {node: [] for node in tree.nodes()}
    queue = deque([root])
    visited = {root}
    while queue:
        node = queue.popleft()
        for neighbor in tree.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                queue.append(neighbor)
    return parent, children


def calc_subtree_populations(children: dict, root: int, populations: np.ndarray) -> dict:
    """Calculate subtree populations using post-order traversal."""
    subtree_pops = {}
    stack = [root]
    post_order = []
    while stack:
        node = stack.pop()
        post_order.append(node)
        for child in children[node]:
            stack.append(child)
    for node in reversed(post_order):
        pop = populations[node]
        for child in children[node]:
            pop += subtree_pops[child]
        subtree_pops[node] = pop
    return subtree_pops


def get_subtree_nodes(node: int, children: dict) -> list:
    """Get all nodes in subtree."""
    result = []
    queue = deque([node])
    while queue:
        current = queue.popleft()
        result.append(current)
        queue.extend(children[current])
    return result


def find_balanced_cut(
    graph: nx.Graph,
    populations: np.ndarray,
    target_pop: float,
    tolerance: float,
    max_attempts: int = 1000,
    one_sided: bool = False,
) -> list | None:
    """NetworkX version — kept for compatibility."""
    total_pop = sum(populations[n] for n in graph.nodes())
    min_pop = target_pop * (1 - tolerance)
    max_pop = target_pop * (1 + tolerance)

    for attempt in range(max_attempts):
        tree = random_spanning_tree(graph)
        candidates = [n for n, d in tree.degree() if d > 1]
        if not candidates:
            continue
        root = candidates[np.random.randint(len(candidates))]
        parent, children = get_tree_structure(tree, root)
        subtree_pops = calc_subtree_populations(children, root, populations)

        for node in tree.nodes():
            if node == root or parent.get(node) is None:
                continue
            tree_pop = subtree_pops[node]
            other_pop = total_pop - tree_pop
            if one_sided:
                tree_valid = min_pop <= tree_pop <= max_pop
                other_valid = min_pop <= other_pop <= max_pop
                if tree_valid or other_valid:
                    if tree_valid:
                        return get_subtree_nodes(node, children)
                    else:
                        subtree = set(get_subtree_nodes(node, children))
                        return [n for n in graph.nodes() if n not in subtree]
            else:
                if min_pop <= tree_pop <= max_pop and min_pop <= other_pop <= max_pop:
                    return get_subtree_nodes(node, children)

    return None
