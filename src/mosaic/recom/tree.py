"""Spanning tree operations for ReCom algorithm using igraph."""

import time
import numpy as np
import igraph as ig
from typing import Optional, Tuple

# ── Optional Numba JIT acceleration ──────────────────────────────────────────
# Speeds up the inner BFS + subtree-population loop by ~1.7-2x on the full
# find_balanced_cut_ig call. Falls back to pure Python if Numba is unavailable.

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _nb_build_csr(eu_m, ev_m, n_nodes, indptr, indices, degree, cursor):
        """Build CSR adjacency for the MST from its edge endpoint arrays."""
        for i in range(n_nodes):
            degree[i] = 0
        for i in range(len(eu_m)):
            degree[eu_m[i]] += 1
            degree[ev_m[i]] += 1
        indptr[0] = 0
        for i in range(n_nodes):
            indptr[i + 1] = indptr[i] + degree[i]
        for i in range(n_nodes):
            cursor[i] = indptr[i]
        for i in range(len(eu_m)):
            u = eu_m[i]; v = ev_m[i]
            indices[cursor[u]] = v; cursor[u] += 1
            indices[cursor[v]] = u; cursor[v] += 1

    @_njit(cache=True)
    def _nb_bfs_subtree(indptr, indices, n_nodes, sub_pops,
                        root, parent, bfs_q, stpops, visited):
        """BFS the MST (CSR) from root; accumulate subtree populations."""
        for i in range(n_nodes):
            visited[i] = False
            parent[i]  = -1
        visited[root] = True
        bfs_q[0] = root
        head = np.int32(0)
        tail = np.int32(1)
        while head < tail:
            v = bfs_q[head]; head += 1
            for j in range(indptr[v], indptr[v + 1]):
                nb = indices[j]
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb]  = v
                    bfs_q[tail] = nb; tail += 1
        for i in range(n_nodes):
            stpops[i] = sub_pops[i]
        for i in range(tail - 1, -1, -1):
            v = bfs_q[i]
            p = parent[v]
            if p >= 0:
                stpops[p] += stpops[v]
        return tail

    @_njit(cache=True)
    def _nb_collect_subtree(v_root, bfs_q, tail, parent, in_sub, result):
        """Collect nodes in the subtree of v_root using BFS order + parent array."""
        for i in range(len(in_sub)):
            in_sub[i] = False
        in_sub[v_root] = True
        count = np.int32(0)
        for i in range(tail):
            node = bfs_q[i]
            par  = parent[node]
            if node == v_root or (par >= 0 and in_sub[par]):
                in_sub[node] = True
                result[count] = node
                count += 1
        return count

    _NUMBA_OK = True

except ImportError:
    _NUMBA_OK = False


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
    timeout: float | None = None,
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

    start_time = time.perf_counter() if timeout else None

    edge_list = graph.get_edgelist()  # list of (u, v) indexed by edge id

    eu_arr = np.array([e[0] for e in edge_list], dtype=np.int32)
    ev_arr = np.array([e[1] for e in edge_list], dtype=np.int32)

    if _NUMBA_OK:
        # ── Allocate per-subgraph buffers (reused across all attempts) ────────
        _nb_ptr  = np.zeros(n + 1,           dtype=np.int32)
        _nb_idx  = np.zeros(max(2*(n-1), 1), dtype=np.int32)
        _nb_deg  = np.zeros(n,               dtype=np.int32)
        _nb_cur  = np.zeros(n,               dtype=np.int32)
        _nb_bfsq = np.zeros(n,               dtype=np.int32)
        _nb_par  = np.zeros(n,               dtype=np.int32)
        _nb_vis  = np.zeros(n,               dtype=np.bool_)
        _nb_stp  = np.zeros(n,               dtype=np.float64)
        _nb_insub= np.zeros(n,               dtype=np.bool_)
        _nb_res  = np.zeros(n,               dtype=np.int32)

    cross_county_mask = None
    if county_array is not None and county_bias != 1.0 and m > 0:
        _nc_local = county_array[node_ids]
        cross_county_mask = (_nc_local[eu_arr] != _nc_local[ev_arr])

    if not _NUMBA_OK:
        subtree_pops = np.empty(n, dtype=np.float64)

    for _ in range(max_attempts):
        if start_time and (time.perf_counter() - start_time) > timeout:
            return None

        if _NUMBA_OK:
            # ── igraph MST + Numba BFS ────────────────────────────────────────
            weights = np.random.random(m)
            if cross_county_mask is not None:
                weights[cross_county_mask] *= county_bias
            mst_ids = np.array(
                graph.spanning_tree(weights=weights, return_tree=False),
                dtype=np.int32,
            )
            eu_m = eu_arr[mst_ids]
            ev_m = ev_arr[mst_ids]

            _nb_build_csr(eu_m, ev_m, n, _nb_ptr, _nb_idx, _nb_deg, _nb_cur)

            candidates = np.flatnonzero(_nb_deg > 1)
            if len(candidates) == 0:
                continue
            root = int(candidates[np.random.randint(len(candidates))])

            tail = _nb_bfs_subtree(_nb_ptr, _nb_idx, n, sub_pops,
                                   np.int32(root), _nb_par, _nb_bfsq,
                                   _nb_stp, _nb_vis)

            other_pops = total_pop - _nb_stp
            if one_sided:
                valid = ((_nb_stp >= min_pop) & (_nb_stp <= max_pop) |
                         (other_pops >= min_pop) & (other_pops <= max_pop))
            else:
                valid = ((_nb_stp >= min_pop) & (_nb_stp <= max_pop) &
                         (other_pops >= min_pop) & (other_pops <= max_pop))
            valid[root] = False

            valid_indices = np.flatnonzero(valid)
            if len(valid_indices) > 0:
                v = int(valid_indices[0])
                cnt = _nb_collect_subtree(np.int32(v), _nb_bfsq, tail,
                                          _nb_par, _nb_insub, _nb_res)
                subtree_nodes = _nb_res[:cnt]
                if one_sided and not (min_pop <= _nb_stp[v] <= max_pop):
                    mask = np.ones(n, dtype=np.bool_)
                    mask[subtree_nodes] = False
                    return node_ids[mask].tolist()
                return node_ids[subtree_nodes].tolist()

        else:
            # ── Pure-Python fallback: igraph spanning_tree + list BFS ─────────
            weights = np.random.random(m)
            if cross_county_mask is not None:
                weights[cross_county_mask] *= county_bias
            mst_edge_ids = graph.spanning_tree(weights=weights, return_tree=False)

            tree_adj: list[list[int]] = [[] for _ in range(n)]
            for eid in mst_edge_ids:
                eu, ev = edge_list[eid]
                tree_adj[eu].append(ev)
                tree_adj[ev].append(eu)

            candidates = [i for i, adj in enumerate(tree_adj) if len(adj) > 1]
            if not candidates:
                continue
            root = candidates[np.random.randint(len(candidates))]

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

            subtree_pops[:] = sub_pops
            for v in reversed(bfs_queue):
                p = parent[v]
                if p >= 0:
                    subtree_pops[p] += subtree_pops[v]

            other_pops = total_pop - subtree_pops
            if one_sided:
                tree_valid  = (subtree_pops >= min_pop) & (subtree_pops <= max_pop)
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
    timeout: float | None = None,
) -> list | None:
    """NetworkX version — kept for compatibility."""
    total_pop = sum(populations[n] for n in graph.nodes())
    min_pop = target_pop * (1 - tolerance)
    max_pop = target_pop * (1 + tolerance)
    start_time = time.perf_counter() if timeout else None

    for attempt in range(max_attempts):
        if start_time and (time.perf_counter() - start_time) > timeout:
            return None
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
