"""Algorithm execution thread for the GUI."""

import logging
import time
from collections import deque
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np

logging.getLogger("mosaic").addHandler(logging.NullHandler())
log = logging.getLogger("mosaic")

import igraph as ig

from mosaic.gui.state import SharedState, AlgorithmStatus
from mosaic.io.inspect import inspect_shapefile, ShapefileConfig, ShapefileInspection
from mosaic.graph import build_adjacency_graph, load_cached_graph, save_cached_graph, get_cache_path
from mosaic.recom import create_initial_partition
from mosaic.recom.recombination import recom_step_ig, GraphContext
from mosaic.recom.annealing import init_annealing, accept_proposal, cool_temperature
from mosaic.scoring import score_plan
from mosaic.scoring.precompute import PPData, precompute_pp_data

# Rolling window size for acceptance rate chart
_ACCEPTANCE_WINDOW = 200


def _build_score_breakdown(ps, cfg) -> dict:
    """Return {metric_name: pct_of_total} for non-zero contributors, sorted by name."""
    total = ps.total
    if total <= 0:
        return {}
    result = {}
    def _add(name: str, contrib: float) -> None:
        if contrib > 0:
            result[name] = contrib / total * 100.0
    _add("Cut Edges",      cfg.weight_cut_edges * ps.cut_edges)
    _add("County Splits",  cfg.weight_county_splits * ps.county_splits)
    _add("Compactness (PP)", cfg.weight_polsby_popper * ps.polsby_popper)
    _add("Pop. Deviation", cfg.weight_pop_deviation * ps.pop_deviation)
    if cfg.weight_mean_median:
        _add("Mean-Median",    cfg.weight_mean_median *
             ((ps.mean_median - cfg.target_mean_median) * 100) ** 2)
    if cfg.weight_efficiency_gap:
        _add("Efficiency Gap", cfg.weight_efficiency_gap *
             ((ps.efficiency_gap - cfg.target_efficiency_gap) * 100) ** 2)
    if cfg.weight_dem_seats:
        _add("Dem Seats",      cfg.weight_dem_seats *
             (ps.dem_seats - cfg.target_dem_seats) ** 2 * 100)
    _add("Competitiveness", cfg.weight_competitiveness * ps.competitiveness * 100)
    if cfg.weight_majority_chance_dem:
        _add("D Majority", cfg.weight_majority_chance_dem *
             (1.0 - ps.majority_chance_dem) ** 1.5 * 100)
    if cfg.weight_majority_chance_rep:
        _add("R Majority", cfg.weight_majority_chance_rep *
             (1.0 - ps.majority_chance_rep) ** 1.5 * 100)
    if cfg.weight_hinge:
        _add("Hinge", cfg.weight_hinge * (1.0 - ps.hinge_chance) ** 1.5 * 100)
    return result


def nx_to_igraph(nxg: nx.Graph) -> ig.Graph:
    node_list = sorted(nxg.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nxg.edges()]
    g = ig.Graph(n=len(node_list), edges=edges, directed=False)
    g.vs["name"] = node_list
    return g


class AlgorithmRunner:
    """Runs ReCom + annealing on a background thread; writes progress to SharedState."""

    def __init__(self, state: SharedState):
        self.state = state
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.graph: Optional[nx.Graph] = None
        self.graph_ig: Optional[ig.Graph] = None
        self.graph_ctx: Optional[GraphContext] = None
        self.populations: Optional[np.ndarray] = None
        self.county_array: Optional[np.ndarray] = None
        self.county_pops: Optional[np.ndarray] = None
        self.pp_data: Optional[PPData] = None
        self.election_arrays: list[tuple[np.ndarray, np.ndarray]] = []
        self.id_col_name: str = "precinct_id"

        # Holds the ShapefileInspection until the user confirms in the dialog
        self._pending_inspection: Optional[ShapefileInspection] = None

    # ── Phase 1: Inspect ─────────────────────────────────────────────────────

    def start_inspection(self, path: str) -> None:
        """
        Read the shapefile and collect column statistics (fast — no graph build).
        Signals state.shp_inspect_ready when done so the GUI can show the dialog.
        """
        self.state.update(
            status=AlgorithmStatus.LOADING,
            status_message="Reading shapefile...",
            shapefile_path=path,
        )
        insp = inspect_shapefile(path)
        self._pending_inspection = insp

        if insp.load_error:
            self.state.update(
                status=AlgorithmStatus.ERROR,
                error_message=f"Could not read shapefile: {insp.load_error}",
            )
        else:
            log.info(f"Inspected {path}: {insp.n_precincts} precincts, "
                     f"{len(insp.columns)} columns")
            self.state.update(
                status=AlgorithmStatus.IDLE,
                status_message=f"Inspected — {insp.n_precincts:,} precincts found",
                shp_inspect_ready=True,
            )

    # ── Phase 2: Complete load ────────────────────────────────────────────────

    def complete_load(self, inspection: ShapefileInspection, config: ShapefileConfig) -> bool:
        """
        Build the adjacency graph using the user-confirmed column assignments.
        Uses the GDF already loaded during inspection — no second file read.
        """
        self.state.update(
            status=AlgorithmStatus.LOADING,
            status_message="Loading data...",
        )
        try:
            gdf = inspection.gdf
            self.gdf = gdf
            self.id_col_name = config.id_col

            # Population
            pop_series = gdf[config.pop_col]
            if pop_series.isna().any():
                n_null = int(pop_series.isna().sum())
                log.warning(f"Population column '{config.pop_col}' has {n_null} null value(s) — converted to 0")
                pop_series = pop_series.fillna(0)
            self.populations = pop_series.values.astype(np.int64)

            # County array
            if config.county_col and config.county_col in gdf.columns:
                vals = gdf[config.county_col].values
                _, ids = np.unique(vals, return_inverse=True)
                self.county_array = ids.astype(np.int32)
                n_counties = int(ids.max()) + 1
                self.county_pops = np.bincount(
                    self.county_array,
                    weights=self.populations.astype(np.float64),
                    minlength=n_counties,
                )
                log.info(f"County col '{config.county_col}': {n_counties} unique counties")
            else:
                self.county_array = None
                self.county_pops = None

            # Election arrays
            self.election_arrays = []
            for dem_col, gop_col in config.elections:
                if dem_col in gdf.columns and gop_col in gdf.columns:
                    dem_s = gdf[dem_col]
                    gop_s = gdf[gop_col]
                    for col_name, s in ((dem_col, dem_s), (gop_col, gop_s)):
                        if s.isna().any():
                            n_null = int(s.isna().sum())
                            log.warning(f"Election column '{col_name}' has {n_null} null value(s) — converted to 0")
                    dem_s = dem_s.fillna(0)
                    gop_s = gop_s.fillna(0)
                    dem = dem_s.values.astype(np.int64)
                    gop = gop_s.values.astype(np.int64)
                    self.election_arrays.append((dem, gop))
                    log.info(f"Election: {dem_col}/{gop_col} — "
                             f"D:{dem.sum():,}  R:{gop.sum():,}")

            n = len(gdf)
            log.info(f"Loaded {n} precincts, total pop: {self.populations.sum():,}")
            self.state.update(
                num_precincts=n,
                total_population=int(self.populations.sum()),
                status_message=f"Loaded {n:,} precincts",
            )

            # Load or build adjacency graph
            cache_path = get_cache_path(inspection.path)
            self.graph = load_cached_graph(cache_path, gdf)

            if self.graph is not None:
                log.info(f"Cached graph: {self.graph.number_of_nodes()} nodes, "
                         f"{self.graph.number_of_edges()} edges")
                self.state.update(
                    status_message=f"Loaded cached graph "
                                   f"({self.graph.number_of_edges():,} edges)",
                )
            else:
                log.info("Building adjacency graph from scratch...")
                self.state.update(
                    status=AlgorithmStatus.BUILDING_GRAPH,
                    status_message="Building adjacency graph...",
                )
                self.graph = build_adjacency_graph(gdf)
                save_cached_graph(self.graph, cache_path)
                log.info(f"Built graph: {self.graph.number_of_nodes()} nodes, "
                         f"{self.graph.number_of_edges()} edges")
                self.state.update(
                    status_message=f"Built graph ({self.graph.number_of_edges():,} edges)",
                )

            self.graph_ig = nx_to_igraph(self.graph)
            self.graph_ctx = GraphContext(self.graph_ig)

            # Precompute geometry for scoring
            self.state.update(status_message="Precomputing geometry data...")
            self.pp_data = precompute_pp_data(gdf, self.graph)

            n_edges = self.graph.number_of_edges()
            log.info(f"Load complete: {n} precincts, {n_edges} edges")
            self.state.update(
                status=AlgorithmStatus.IDLE,
                status_message=f"Ready ({n_edges:,} edges)",
                gdf_ready=True,
            )
            return True

        except Exception as e:
            log.error(f"complete_load error: {e}", exc_info=True)
            self.state.update(status=AlgorithmStatus.ERROR, error_message=str(e))
            return False

    # ── Algorithm ─────────────────────────────────────────────────────────────

    def run_algorithm(self):
        """Main ReCom + annealing loop."""
        if self.graph is None or self.populations is None or self.graph_ctx is None:
            self.state.update(status=AlgorithmStatus.ERROR, error_message="No shapefile loaded")
            return

        (num_districts, tolerance, max_iterations, seed,
         score_config, annealing_config,
         county_bias_enabled, county_bias) = self.state.get(
            "num_districts", "pop_tolerance", "max_iterations", "seed",
            "score_config", "annealing_config",
            "county_bias_enabled", "county_bias",
        )

        ideal_pop = self.populations.sum() / num_districts
        ctx = self.graph_ctx
        effective_bias = county_bias if county_bias_enabled else 1.0

        _skw = dict(
            county_ids=self.county_array,
            populations=self.populations,
            ideal_pop=ideal_pop,
            tolerance=tolerance,
            pp_data=self.pp_data,
            n_districts=num_districts,
            dem_votes=self.election_arrays[0][0] if self.election_arrays else None,
            gop_votes=self.election_arrays[0][1] if self.election_arrays else None,
        )

        worse_window: deque[int] = deque(maxlen=_ACCEPTANCE_WINDOW)

        try:
            # ── Initial partition ────────────────────────────────────────────
            log.info(
                f"Partitioning: {num_districts} districts, "
                f"{tolerance*100:.1f}% tolerance, {max_iterations} iters"
            )
            self.state.update(
                status=AlgorithmStatus.PARTITIONING,
                status_message="Creating initial partition...",
            )
            assignment = create_initial_partition(
                self.graph, self.populations, num_districts, tolerance,
                seed=seed,
                on_progress=lambda c, t: self.state.update(
                    status_message=f"Creating district {c}/{t}..."
                ),
            )
            log.info("Initial partition complete")
            self.state.update(initial_assignment=assignment.copy())

            # ── Initial score ────────────────────────────────────────────────
            cut_edge_indices = np.where(
                assignment[ctx.edge_u] != assignment[ctx.edge_v]
            )[0].astype(np.int32)
            current_ps = score_plan(cut_edge_indices, score_config,
                                    assignment=assignment, **_skw)

            with self.state._lock:
                self.state.score_history.append(current_ps.total)
                self.state.county_splits_score_history.append(current_ps.county_splits)
                self.state.county_excess_splits_history.append(current_ps.county_excess_splits)
                self.state.county_clean_districts_history.append(current_ps.county_clean_districts)
                self.state.mm_history.append(current_ps.mean_median)
                self.state.eg_history.append(current_ps.efficiency_gap)
                self.state.dem_seats_history.append(current_ps.dem_seats)
                self.state.competitive_count_history.append(0)
                self.state.pp_history.append(1.0 - current_ps.polsby_popper / 100.0)
                self.state.cut_edges_history.append(current_ps.cut_edges)
                self.state.majority_dem_history.append(current_ps.majority_chance_dem)
                self.state.majority_rep_history.append(current_ps.majority_chance_rep)
                self.state.hinge_history.append(current_ps.hinge_chance)

            self.state.update(
                status=AlgorithmStatus.RUNNING,
                status_message="Running ReCom...",
                current_assignment=assignment.copy(),
                current_score=current_ps.total,
                current_cut_edges=current_ps.cut_edges,
                best_assignment=assignment.copy(),
                best_score=current_ps.total,
                best_iteration=0,
                best_cut_edges=current_ps.cut_edges,
                best_temperature=0.0,
                best_score_history_len=1,
                best_temp_history_len=0,
                best_acc_history_len=0,
                score_breakdown=_build_score_breakdown(current_ps, score_config),
            )

            # ── Annealing init ───────────────────────────────────────────────
            if annealing_config.enabled:
                ann = init_annealing(current_ps.total, annealing_config, max_iterations)
                log.info(
                    f"Annealing ({annealing_config.cooling_mode}): "
                    f"initial_temp={ann.initial_temp:.3f}, "
                    f"cooling_rate={ann.cooling_rate:.6f}"
                )
                self.state.update(current_temperature=ann.temperature)
                with self.state._lock:
                    self.state.temperature_history.append(ann.temperature)
            else:
                ann = None
                log.info("Annealing disabled")

            # ── Main loop ────────────────────────────────────────────────────
            self.state.update(start_time=time.time())
            log.info(f"Starting ReCom loop: {max_iterations} iterations")

            (map_render_interval,) = self.state.get("map_render_interval")
            _last_map_time = 0.0

            for iteration in range(1, max_iterations + 1):
                if self.state.check_should_stop():
                    log.info("Stopped by user")
                    self.state.update(
                        status=AlgorithmStatus.IDLE,
                        status_message="Stopped",
                        end_time=time.time(),
                        map_needs_update=True,
                    )
                    return

                if self.state.check_should_pause():
                    _pause_start = time.time()
                    self.state.update(
                        status=AlgorithmStatus.PAUSED,
                        status_message="Paused",
                        pause_time=_pause_start,
                    )
                    while self.state.check_should_pause():
                        if self.state.check_should_stop():
                            self.state.update(
                                status=AlgorithmStatus.IDLE,
                                status_message="Stopped",
                                end_time=time.time(),
                                pause_time=0.0,
                            )
                            return
                        time.sleep(0.05)
                    # Shift start_time forward by pause duration so IPS stays accurate
                    (current_start,) = self.state.get("start_time")
                    self.state.update(
                        start_time=current_start + (time.time() - _pause_start),
                        pause_time=0.0,
                        status=AlgorithmStatus.RUNNING,
                        status_message="Running ReCom...",
                    )

                # ── ReCom step ───────────────────────────────────────────────
                new_assignment, valid, new_cut_indices = recom_step_ig(
                    ctx, assignment, self.populations, ideal_pop, tolerance,
                    cut_edge_indices,
                    county_array=self.county_array, county_bias=effective_bias,
                )

                if not valid:
                    self.state.update(current_iteration=iteration)
                    if ann is not None:
                        cool_temperature(ann)
                    continue

                # ── Score proposal ───────────────────────────────────────────
                proposed_ps = score_plan(new_cut_indices, score_config,
                                         assignment=new_assignment, **_skw)

                # ── Metropolis acceptance ────────────────────────────────────
                if ann is not None:
                    is_worse = proposed_ps.total > current_ps.total
                    accepted = accept_proposal(current_ps.total, proposed_ps.total, ann)
                    cool_temperature(ann)
                    if is_worse:
                        worse_window.append(1 if accepted else 0)
                else:
                    accepted = True
                    is_worse = False

                if accepted:
                    assignment = new_assignment
                    cut_edge_indices = new_cut_indices
                    current_ps = proposed_ps
                    self.state.update(
                        current_iteration=iteration,
                        current_assignment=assignment.copy(),
                        current_score=current_ps.total,
                        current_cut_edges=current_ps.cut_edges,
                        successful_steps=self.state.successful_steps + 1,
                        score_breakdown=_build_score_breakdown(current_ps, score_config),
                    )
                else:
                    self.state.update(current_iteration=iteration)

                # ── History bookkeeping (before best check so lengths are accurate)
                # Competitive district count (|share - 0.5| < 0.05) — computed outside lock
                comp_count = 0
                if self.election_arrays:
                    _dem, _gop = self.election_arrays[0]
                    _dem_d = np.bincount(assignment, weights=_dem.astype(np.float64), minlength=num_districts)
                    _gop_d = np.bincount(assignment, weights=_gop.astype(np.float64), minlength=num_districts)
                    _tot_d = _dem_d + _gop_d
                    _shares = np.where(_tot_d > 0, _dem_d / _tot_d, 0.5)
                    comp_count = int((np.abs(_shares - 0.5) < 0.05).sum())

                n_score = n_temp = n_acc = 0
                with self.state._lock:
                    self.state.score_history.append(current_ps.total)
                    self.state.county_splits_score_history.append(current_ps.county_splits)
                    self.state.county_excess_splits_history.append(current_ps.county_excess_splits)
                    self.state.county_clean_districts_history.append(current_ps.county_clean_districts)
                    self.state.mm_history.append(current_ps.mean_median)
                    self.state.eg_history.append(current_ps.efficiency_gap)
                    self.state.dem_seats_history.append(current_ps.dem_seats)
                    self.state.competitive_count_history.append(comp_count)
                    self.state.pp_history.append(1.0 - current_ps.polsby_popper / 100.0)
                    self.state.cut_edges_history.append(current_ps.cut_edges)
                    self.state.majority_dem_history.append(current_ps.majority_chance_dem)
                    self.state.majority_rep_history.append(current_ps.majority_chance_rep)
                    self.state.hinge_history.append(current_ps.hinge_chance)
                    n_score = len(self.state.score_history)
                    if ann is not None:
                        self.state.temperature_history.append(ann.temperature)
                        n_temp = len(self.state.temperature_history)
                        self.state.current_temperature = ann.temperature
                        self.state.accepted_worse = ann.accepted_worse
                        self.state.rejected_worse = ann.rejected_worse
                        if len(worse_window) >= 10:
                            rate = sum(worse_window) / len(worse_window)
                            self.state.acceptance_rate_history.append(
                                [float(iteration), rate]
                            )
                            n_acc = len(self.state.acceptance_rate_history)

                # ── Best plan tracking
                if accepted and current_ps.total < self.state.best_score:
                    self.state.update(
                        best_assignment=assignment.copy(),
                        best_score=current_ps.total,
                        best_iteration=iteration,
                        best_cut_edges=current_ps.cut_edges,
                        best_temperature=ann.temperature if ann else 0.0,
                        best_score_history_len=n_score,
                        best_temp_history_len=n_temp,
                        best_acc_history_len=n_acc,
                    )

                _now = time.time()
                if _now - _last_map_time >= map_render_interval:
                    self.state.update(map_needs_update=True)
                    _last_map_time = _now

                if iteration % 100 == 0:
                    elapsed = time.time() - self.state.start_time
                    ips = iteration / elapsed if elapsed > 0 else 0
                    temp_str = f", temp={ann.temperature:.3f}" if ann else ""
                    log.info(
                        f"Iter {iteration}: score={current_ps.total:.1f} "
                        f"(cuts={current_ps.cut_edges}), {ips:.1f} IPS{temp_str}"
                    )

            log.info(f"Completed {max_iterations} iterations")
            self.state.update(
                status=AlgorithmStatus.COMPLETED,
                status_message=f"Completed {max_iterations} iterations",
                end_time=time.time(),
                map_needs_update=True,
            )

        except Exception as e:
            log.error(f"Algorithm error: {e}", exc_info=True)
            self.state.update(status=AlgorithmStatus.ERROR, error_message=str(e))
