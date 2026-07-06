"""Spanning tree operations for ReCom algorithm using igraph."""

import time
from dataclasses import dataclass
import numpy as np
import igraph as ig
from typing import Optional


@dataclass
class FbcScratch:
    """Persistent scratch buffers for find_balanced_cut_fast.

    Sized once per GraphContext to the parent graph's node and edge counts;
    reused across every ReCom step so the hot loop allocates nothing. Three
    of the four buffers are write-then-overwrite; only `in_merged` carries
    persistent state, and it is reset (in-place at positions touched by the
    last call) before each new call.
    """
    in_merged: np.ndarray  # bool[n_total] — True for nodes in current merged region
    local_idx: np.ndarray  # int32[n_total] — global -> local, valid where in_merged
    loc_eu: np.ndarray     # int32[n_edges_parent] — merged-region edges (local idx)
    loc_ev: np.ndarray     # int32[n_edges_parent]

    @classmethod
    def allocate(cls, n_nodes: int, n_edges: int) -> "FbcScratch":
        return cls(
            in_merged=np.zeros(n_nodes, dtype=np.bool_),
            local_idx=np.empty(n_nodes, dtype=np.int32),
            loc_eu=np.empty(max(n_edges, 1), dtype=np.int32),
            loc_ev=np.empty(max(n_edges, 1), dtype=np.int32),
        )

# ── Optional Numba JIT acceleration ──────────────────────────────────────────
# Powers the hot-path kernels (cut edges, MST, BFS, contiguity). Falls back to
# pure Python / igraph when Numba is unavailable.

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _nb_cut_edges(eu, ev, assignment, out):
        """Emit indices of edges whose endpoints differ in district, ascending.

        Bit-identical to np.where(assignment[eu] != assignment[ev])[0] (both
        scan edges in ascending index order), but a single branchy pass with no
        7529-wide fancy-gather / boolean temporaries. `out` must hold n_edges.
        """
        cnt = 0
        for i in range(eu.shape[0]):
            if assignment[eu[i]] != assignment[ev[i]]:
                out[cnt] = i
                cnt += 1
        return cnt

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

    @_njit(cache=True)
    def _nb_extract_local_edges(parent_eu, parent_ev, in_merged, local_idx,
                                out_eu, out_ev):
        """Scan parent edge arrays; emit edges with both endpoints in the merged
        region, re-indexed into local (0..n_merged-1) space. Single pass, O(m)."""
        m = parent_eu.shape[0]
        cnt = np.int32(0)
        for i in range(m):
            u = parent_eu[i]
            v = parent_ev[i]
            if in_merged[u] and in_merged[v]:
                out_eu[cnt] = local_idx[u]
                out_ev[cnt] = local_idx[v]
                cnt += 1
        return cnt

    @_njit(cache=True)
    def _nb_kruskal_mst(eu, ev, sorted_idx, n_nodes,
                        uf_parent, uf_rank, mst_eu, mst_ev):
        """Kruskal MST with union-by-rank + path compression.

        sorted_idx is a permutation of edge indices -- ascending-weight (county
        bias) or uniform-random (no bias) -- yielding a random spanning tree.
        Returns the MST edge count (== n_nodes-1 iff connected; merged regions
        are connected by construction). A disconnected input returns a partial
        forest and the caller's BFS drops the attempt.
        """
        for i in range(n_nodes):
            uf_parent[i] = i
            uf_rank[i] = 0
        m = sorted_idx.shape[0]
        target = n_nodes - 1
        cnt = np.int32(0)
        for k in range(m):
            e = sorted_idx[k]
            u = eu[e]
            v = ev[e]
            # find(u) with path halving
            ru = u
            while uf_parent[ru] != ru:
                uf_parent[ru] = uf_parent[uf_parent[ru]]
                ru = uf_parent[ru]
            rv = v
            while uf_parent[rv] != rv:
                uf_parent[rv] = uf_parent[uf_parent[rv]]
                rv = uf_parent[rv]
            if ru == rv:
                continue
            if uf_rank[ru] < uf_rank[rv]:
                uf_parent[ru] = rv
            elif uf_rank[ru] > uf_rank[rv]:
                uf_parent[rv] = ru
            else:
                uf_parent[rv] = ru
                uf_rank[ru] += 1
            mst_eu[cnt] = u
            mst_ev[cnt] = v
            cnt += 1
            if cnt == target:
                break
        return cnt

    @_njit(cache=True)
    def _nb_collect_candidates(degree, n_nodes, out):
        """Internal (degree > 1) MST nodes in ascending order -- the per-attempt
        root pool. out[:count] holds the ids; ascending order keeps the
        subsequent randint pick deterministic."""
        cnt = np.int32(0)
        for i in range(n_nodes):
            if degree[i] > 1:
                out[cnt] = i
                cnt += 1
        return cnt

    @_njit(cache=True)
    def _nb_first_valid_cut(stpops, total_pop, min_pop, max_pop,
                            one_sided, root, n_nodes):
        """Lowest node index i (i != root) whose cut edge yields a balanced
        bipartition, or -1 if none. Ascending scan returns the lowest index, so
        the pick is deterministic."""
        for i in range(n_nodes):
            if i == root:
                continue
            s = stpops[i]
            o = total_pop - s
            s_ok = (s >= min_pop) and (s <= max_pop)
            o_ok = (o >= min_pop) and (o <= max_pop)
            ok = (s_ok or o_ok) if one_sided else (s_ok and o_ok)
            if ok:
                return i
        return -1

    @_njit(cache=True)
    def _nb_district_connected_without(edge_u, edge_v, assignment, district, exclude):
        """True if precincts assigned to `district`, minus `exclude`, form one
        connected component under the edges internal to that set.

        Replaces igraph subgraph().is_connected() in the flip contiguity check:
        a union-find over a single edge scan, no graph object or list build.
        Scans all edges (incl. virtual bridges), matching the igraph path which
        ran on the full graph.
        """
        n = assignment.shape[0]
        parent = np.empty(n, dtype=np.int32)
        for i in range(n):
            parent[i] = i

        for e in range(edge_u.shape[0]):
            u = edge_u[e]
            v = edge_v[e]
            if u == exclude or v == exclude:
                continue
            if assignment[u] != district or assignment[v] != district:
                continue
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            if u != v:
                parent[u] = v

        root = -1
        for i in range(n):
            if i == exclude or assignment[i] != district:
                continue
            r = i
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            if root == -1:
                root = r
            elif r != root:
                return False
        return True

    _NUMBA_OK = True

except ImportError:
    _NUMBA_OK = False


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
    out_state: dict | None = None,
) -> list | None:
    """Find a balanced bipartition of the graph using random spanning trees.

    If `out_state` is provided (an empty dict), it is populated on success with
    the data needed to attempt a follow-up cut in the residual tree without
    building a fresh MST — see `try_residual_balanced_cut`. The state captures
    the successful attempt's MST + BFS structure in SUBGRAPH index space.
    """
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
            mst_ids = np.asarray(
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
                carved_is_complement = (
                    one_sided and not (min_pop <= _nb_stp[v] <= max_pop)
                )
                if carved_is_complement:
                    mask = np.ones(n, dtype=np.bool_)
                    mask[subtree_nodes] = False
                    carved_sub_idx = np.flatnonzero(mask).astype(np.int32)
                    result = node_ids[mask].tolist()
                else:
                    carved_sub_idx = subtree_nodes.astype(np.int32, copy=True)
                    result = node_ids[subtree_nodes].tolist()
                if out_state is not None:
                    # _nb_par/_nb_bfsq/_nb_stp are stack-local to this call; the
                    # dict reference keeps them alive after we return. A second
                    # find_balanced_cut_ig invocation allocates its own fresh
                    # buffers, so there's no aliasing risk.
                    out_state.update(
                        carved_sub_idx=carved_sub_idx,
                        v_cut=v,
                        carved_is_complement=carved_is_complement,
                        parent=_nb_par,
                        bfs_q=_nb_bfsq,
                        tail=int(tail),
                        subtree_pops=_nb_stp,
                        root=int(root),
                        n=n,
                        node_ids=node_ids,
                        total_pop=float(total_pop),
                    )
                return result

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
                v = int(valid_indices[0])
                subtree_nodes = _get_subtree_nodes_fast(v, children, n)
                carved_is_complement = (
                    one_sided and not (min_pop <= subtree_pops[v] <= max_pop)
                )
                if carved_is_complement:
                    mask = np.ones(n, dtype=np.bool_)
                    mask[subtree_nodes] = False
                    carved_sub_idx = np.flatnonzero(mask).astype(np.int32)
                    result = node_ids[mask].tolist()
                else:
                    carved_sub_idx = subtree_nodes.astype(np.int32, copy=True)
                    result = node_ids[subtree_nodes].tolist()
                if out_state is not None:
                    # parent/subtree_pops are local to this call; the dict keeps
                    # them alive. bfs_queue needs an np conversion regardless.
                    out_state.update(
                        carved_sub_idx=carved_sub_idx,
                        v_cut=v,
                        carved_is_complement=carved_is_complement,
                        parent=parent,
                        bfs_q=np.asarray(bfs_queue, dtype=np.int32),
                        tail=len(bfs_queue),
                        subtree_pops=subtree_pops,
                        root=int(root),
                        n=n,
                        node_ids=node_ids,
                        total_pop=float(total_pop),
                    )
                return result

    return None


def find_balanced_cut_fast(
    parent_edge_u: np.ndarray,
    parent_edge_v: np.ndarray,
    scratch: FbcScratch,
    merged_nodes: np.ndarray,
    populations: np.ndarray,
    target_pop: float,
    tolerance: float,
    max_attempts: int = 100,
    one_sided: bool = False,
    county_array: Optional[np.ndarray] = None,
    county_bias: float = 1.0,
    timeout: float | None = None,
    out_state: dict | None = None,
) -> list | None:
    """Numba fast path for balanced bipartition. Equivalent to find_balanced_cut_ig
    but bypasses igraph entirely on the hot loop:

      * No igraph.subgraph() — local edge list is extracted from the parent's
        edge_u/edge_v arrays via a single O(m) Numba scan.
      * No igraph.spanning_tree() — random-weight MST via Numba Kruskal directly
        on the local edges.

    `merged_nodes` is a 1-D int32 array of GLOBAL precinct IDs for the region
    being split. Caller is responsible for ensuring those nodes induce a
    connected subgraph (true by construction in recom_step_ig and _n3, where
    A∪B and A∪B∪C are joined through the picked cut edge).

    On success, returns the carved district's nodes as a list of GLOBAL IDs —
    same shape as find_balanced_cut_ig — and (if `out_state` is supplied)
    populates the same dict schema, so try_residual_balanced_cut works unchanged.

    Numba is required. Caller should check `_NUMBA_OK` and fall back to
    find_balanced_cut_ig + igraph subgraph if Numba is unavailable.
    """
    if not _NUMBA_OK:
        raise RuntimeError(
            "find_balanced_cut_fast requires Numba; check _NUMBA_OK and "
            "dispatch to find_balanced_cut_ig as a fallback"
        )

    n = int(merged_nodes.shape[0])
    if n == 0:
        return None

    in_merged = scratch.in_merged
    local_idx = scratch.local_idx
    loc_eu = scratch.loc_eu
    loc_ev = scratch.loc_ev

    start_time = time.perf_counter() if timeout else None

    # Sort ascending to match ig.Graph.subgraph()'s vertex reordering, so this
    # path and find_balanced_cut_ig share one local-index space. Without it the
    # deterministic first-pick acquires an A-side bias (concat puts A nodes in
    # the low index range) that strands districts.
    merged_nodes = np.sort(merged_nodes).astype(np.int32, copy=False)

    # Mark region + build global->local map. Only n_merged entries touched; the
    # rest stay False from the finally-reset, so extract excludes outside nodes.
    in_merged[merged_nodes] = True
    local_idx[merged_nodes] = np.arange(n, dtype=np.int32)

    sub_pops = populations[merged_nodes].astype(np.float64, copy=True)
    total_pop = float(sub_pops.sum())
    min_pop = target_pop * (1.0 - tolerance)
    max_pop = target_pop * (1.0 + tolerance)

    try:
        m = int(_nb_extract_local_edges(
            parent_edge_u, parent_edge_v, in_merged, local_idx,
            loc_eu, loc_ev,
        ))
        # Views into the scratch — no copy. Safe to slice; we don't write
        # past `m` and we re-extract on the next call.
        local_eu = loc_eu[:m]
        local_ev = loc_ev[:m]

        if m == 0:
            return None

        cross_county_mask = None
        if county_array is not None and county_bias != 1.0:
            cty_local = county_array[merged_nodes]
            cross_county_mask = (cty_local[local_eu] != cty_local[local_ev])

        # Per-call BFS/CSR/collect buffers. Stack-local, not pooled: out_state
        # captures _nb_par/_nb_bfsq/_nb_stp on success, so pooling would collide
        # with the n=3 residual path.
        _nb_ptr   = np.zeros(n + 1,            dtype=np.int32)
        _nb_idx   = np.zeros(max(2 * (n - 1), 1), dtype=np.int32)
        _nb_deg   = np.zeros(n,                dtype=np.int32)
        _nb_cur   = np.zeros(n,                dtype=np.int32)
        _nb_bfsq  = np.zeros(n,                dtype=np.int32)
        _nb_par   = np.zeros(n,                dtype=np.int32)
        _nb_vis   = np.zeros(n,                dtype=np.bool_)
        _nb_stp   = np.zeros(n,                dtype=np.float64)
        _nb_insub = np.zeros(n,                dtype=np.bool_)
        _nb_res   = np.zeros(n,                dtype=np.int32)
        _nb_cand  = np.empty(n,                dtype=np.int32)
        _uf_par   = np.empty(n,                dtype=np.int32)
        _uf_rank  = np.empty(n,                dtype=np.int32)
        _mst_eu   = np.empty(max(n - 1, 1),    dtype=np.int32)
        _mst_ev   = np.empty(max(n - 1, 1),    dtype=np.int32)

        for _ in range(max_attempts):
            if start_time and (time.perf_counter() - start_time) > timeout:
                return None

            if cross_county_mask is None:
                # No county bias: ascending-weight Kruskal over i.i.d. uniform
                # weights is Kruskal over a uniform random edge permutation.
                # permutation(m) gives that same distribution without the float
                # draw + O(m log m) argsort.
                sorted_idx = np.random.permutation(m).astype(np.int32)
            else:
                # County bias tilts edge order toward within-county cuts, so we
                # need real weights to scale before sorting.
                weights = np.random.random(m)
                weights[cross_county_mask] *= county_bias
                sorted_idx = np.argsort(weights).astype(np.int32)

            k = int(_nb_kruskal_mst(
                local_eu, local_ev, sorted_idx, n,
                _uf_par, _uf_rank, _mst_eu, _mst_ev,
            ))

            # Isolated vertices (e.g. island precincts) break the connected-
            # region invariant, so Kruskal returns a partial forest. Mirror
            # find_balanced_cut_ig: build CSR on whatever edges it returned and
            # let the BFS cover the component containing `root`.
            _nb_build_csr(_mst_eu[:k], _mst_ev[:k], n,
                          _nb_ptr, _nb_idx, _nb_deg, _nb_cur)

            n_cand = int(_nb_collect_candidates(_nb_deg, n, _nb_cand))
            if n_cand == 0:
                continue
            root = int(_nb_cand[np.random.randint(n_cand)])

            tail = _nb_bfs_subtree(_nb_ptr, _nb_idx, n, sub_pops,
                                   np.int32(root), _nb_par, _nb_bfsq,
                                   _nb_stp, _nb_vis)

            # Lowest valid cut node (see _nb_first_valid_cut).
            v = int(_nb_first_valid_cut(
                _nb_stp, total_pop, min_pop, max_pop, one_sided,
                np.int32(root), n,
            ))

            if v >= 0:
                cnt = _nb_collect_subtree(np.int32(v), _nb_bfsq, tail,
                                          _nb_par, _nb_insub, _nb_res)
                subtree_nodes = _nb_res[:cnt]
                carved_is_complement = (
                    one_sided and not (min_pop <= _nb_stp[v] <= max_pop)
                )
                if carved_is_complement:
                    mask = np.ones(n, dtype=np.bool_)
                    mask[subtree_nodes] = False
                    carved_sub_idx = np.flatnonzero(mask).astype(np.int32)
                    result = merged_nodes[mask].tolist()
                else:
                    carved_sub_idx = subtree_nodes.astype(np.int32, copy=True)
                    result = merged_nodes[subtree_nodes].tolist()
                if out_state is not None:
                    out_state.update(
                        carved_sub_idx=carved_sub_idx,
                        v_cut=v,
                        carved_is_complement=carved_is_complement,
                        parent=_nb_par,
                        bfs_q=_nb_bfsq,
                        tail=int(tail),
                        subtree_pops=_nb_stp,
                        root=int(root),
                        n=n,
                        node_ids=merged_nodes,
                        total_pop=total_pop,
                    )
                return result

        return None

    finally:
        # Reset only the n_merged positions we touched (not n_total), so reset
        # cost stays O(region size) and doesn't grow with the parent graph.
        in_merged[merged_nodes] = False


def try_residual_balanced_cut(
    state: dict,
    target_pop: float,
    tolerance: float,
) -> list | None:
    """Try to find a balanced bipartition in the residual of state's MST.

    "Residual" = the parent MST with `state["carved_sub_idx"]` removed. Always a
    valid spanning tree of the residual region (removing any subtree from a tree
    leaves a tree on what's left). One shot — no retry, no new MST built.

    Returns a list of original node IDs forming the carved district on success,
    or None if no balanced cut exists in this residual. On None, caller should
    fall back to building a fresh MST on the 2-region subgraph.

    Algorithm:
      Case A — carved = subtree(v_cut): residual rooted at original root.
        For node x in residual: subtree(x) pop in residual =
          subtree_pops[x]                     if x is NOT an ancestor of v_cut
          subtree_pops[x] - subtree_pops[v_cut]   if x IS an ancestor of v_cut

      Case B — carved = complement of subtree(v_cut): residual = subtree(v_cut).
        For node x in residual (x != v_cut): subtree(x) pop in residual =
          subtree_pops[x]   (the carved is outside this branch, no adjustment)
    """
    v_cut: int = state["v_cut"]
    parent: np.ndarray = state["parent"]
    subtree_pops: np.ndarray = state["subtree_pops"]
    bfs_q: np.ndarray = state["bfs_q"]
    tail: int = state["tail"]
    root: int = state["root"]
    n: int = state["n"]
    carved_sub_idx: np.ndarray = state["carved_sub_idx"]
    carved_is_complement: bool = state["carved_is_complement"]
    node_ids: np.ndarray = state["node_ids"]
    total_pop: float = state["total_pop"]

    if carved_is_complement:
        carved_pop = total_pop - float(subtree_pops[v_cut])
    else:
        carved_pop = float(subtree_pops[v_cut])
    total_residual = total_pop - carved_pop

    min_pop = target_pop * (1.0 - tolerance)
    max_pop = target_pop * (1.0 + tolerance)

    in_carved = np.zeros(n, dtype=np.bool_)
    in_carved[carved_sub_idx] = True

    if carved_is_complement:
        side_a = subtree_pops
        side_b = total_residual - subtree_pops
        valid = ((side_a >= min_pop) & (side_a <= max_pop) &
                 (side_b >= min_pop) & (side_b <= max_pop))
        valid[in_carved] = False
        valid[v_cut] = False  # v_cut is the residual's root — no parent edge to cut
    else:
        # Mark ancestors of v_cut (path from v_cut to original root).
        is_ancestor = np.zeros(n, dtype=np.bool_)
        u = int(parent[v_cut])
        while u >= 0:
            is_ancestor[u] = True
            u = int(parent[u])

        adjusted = subtree_pops.copy()
        adjusted[is_ancestor] -= subtree_pops[v_cut]
        other = total_residual - adjusted

        valid = ((adjusted >= min_pop) & (adjusted <= max_pop) &
                 (other >= min_pop) & (other <= max_pop))
        valid[in_carved] = False
        valid[root] = False  # original root has no parent edge to cut

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return None

    x = int(valid_indices[0])

    # Collect subtree(x) in T (not residual) via the existing kernel, then mask
    # carved nodes. Case A (x ancestor of v_cut): subtree(x) in T strictly
    # contains the carved subtree, post-filter removes it. Case B and
    # non-ancestor x in Case A: post-filter is a no-op (subtree disjoint from
    # carved). Either way, result matches the explicit-skip Python loop.
    if _NUMBA_OK:
        in_sub = np.zeros(n, dtype=np.bool_)
        result_buf = np.zeros(n, dtype=np.int32)
        cnt = _nb_collect_subtree(np.int32(x), bfs_q, np.int32(tail),
                                  parent, in_sub, result_buf)
        subtree_in_T = result_buf[:cnt]
    else:
        in_sub = np.zeros(n, dtype=np.bool_)
        in_sub[x] = True
        collected = [x]
        for i in range(tail):
            node = int(bfs_q[i])
            par = int(parent[node])
            if node != x and par >= 0 and in_sub[par]:
                in_sub[node] = True
                collected.append(node)
        subtree_in_T = np.asarray(collected, dtype=np.int32)

    subtree_in_residual = subtree_in_T[~in_carved[subtree_in_T]]
    if len(subtree_in_residual) == 0:
        return None
    return node_ids[subtree_in_residual].tolist()


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
