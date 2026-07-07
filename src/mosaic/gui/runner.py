"""Algorithm execution thread for the GUI."""

import logging
import random as _random
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
from mosaic.graph import (
    build_adjacency_graph, load_cached_graph, save_cached_graph,
    get_cache_path, nx_to_igraph,
)
from mosaic.recom import create_initial_partition
from mosaic.recom.recombination import recom_step_ig, recom_step_ig_n3, GraphContext
from mosaic.recom.flip import flip_step_ig, flip_rate_curve

# Defaults match engine.py / MosaicPy's GUI; not exposed as sliders today.
_FLIP_STEEPNESS = 9.5
from mosaic.recom.annealing import (
    init_annealing, accept_proposal, cool_temperature, relaunch_temperature,
)
from mosaic.scoring import score_plan
from mosaic.scoring.precompute import (
    PPData, precompute_pp_data, precompute_county_data,
)
from mosaic.scoring.reock import ReockData, precompute_reock_data
from mosaic.scoring.alignment import AlignmentData
from mosaic.scoring.cache import (
    get_pp_cache_path, load_cached_pp_data, save_cached_pp_data,
)

# Rolling window size for acceptance rate chart
_ACCEPTANCE_WINDOW = 200

# Tolerance Ratchet: over [_RATCHET_RAMP_START, 1.0] of the run the active
# population tolerance drifts linearly from the user's map-wide tolerance down
# to _RATCHET_FLOOR. The floor is clamped to the start so a freakishly low
# preset never gets loosened (effective_floor = min(floor, start)).
_RATCHET_FLOOR = 0.0025        # 0.25%
_RATCHET_RAMP_START = 0.33     # ramp begins at the 33% mark


def ratchet_ceiling(progress: float, start_tol: float, floor: float) -> float:
    """Scheduled population-tolerance ceiling at a given run progress [0, 1].

    Flat at start_tol until _RATCHET_RAMP_START, then linear down to floor at
    progress 1.0. Callers clamp floor to start_tol (min) so the schedule is
    monotonically non-increasing and never loosens a tight preset.
    """
    if progress <= _RATCHET_RAMP_START:
        return start_tol
    frac = (progress - _RATCHET_RAMP_START) / (1.0 - _RATCHET_RAMP_START)
    return start_tol + (floor - start_tol) * min(frac, 1.0)


def ratchet_step(
    active_tol: float,
    progress: float,
    start_tol: float,
    floor: float,
    map_max_dev: float,
) -> float:
    """Next hard tolerance for one eligible ratchet point (all fractions).

    Drops active_tol down to hug the map's achieved deviation, floored by the
    scheduled soft ceiling: target = max(soft_ceiling, map_max_dev). Bounded by
    min(active_tol, target) so it only ever tightens. Because the soft ceiling
    floors the result, the hard value can sit above it (e.g. soft 1.4%, map
    1.7% -> 1.7%) instead of snapping to the schedule and stranding the map.
    """
    soft = ratchet_ceiling(progress, start_tol, floor)
    return min(active_tol, max(soft, map_max_dev))


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
    _add("Excess Splits",    cfg.weight_county_excess  * ps.county_excess_score)
    _add("Single-County Districts", cfg.weight_county_unified * ps.county_unified_score)
    _add("County Congruence", cfg.weight_holistic_splitting * ps.holistic_splitting)
    _add("Polsby-Popper", cfg.weight_polsby_popper * ps.polsby_popper)
    _add("Reock", cfg.weight_reock * ps.reock)
    _add("Compactness", cfg.weight_holistic_compactness * ps.holistic_compactness)
    _add("Population Deviation", cfg.weight_pop_deviation * ps.pop_deviation)
    _add("Alignment", cfg.weight_alignment * ps.alignment)
    # MM / EG breakdown re-uses score_plan's exact math by recomputing the
    # per-mode penalty from the stored raw values.
    def _mm_eg_pen(raw: float, mode: str, bound: float) -> float:
        if mode == "favor_dem":
            d = max(0.0, min(1.0, (raw + bound) / (2.0 * bound)))
        elif mode == "favor_rep":
            d = max(0.0, min(1.0, (bound - raw) / (2.0 * bound)))
        else:
            d = min(1.0, abs(raw) / bound)
        return (d * d if cfg.partisan_quadratic_penalty else d) * 100.0
    if cfg.weight_mean_median:
        _add("Mean-Median",    cfg.weight_mean_median *
             _mm_eg_pen(ps.mean_median, cfg.mm_mode, cfg.mm_bound))
    if cfg.weight_efficiency_gap:
        _add("Efficiency Gap", cfg.weight_efficiency_gap *
             _mm_eg_pen(ps.efficiency_gap, cfg.eg_mode, cfg.eg_bound))
    if cfg.weight_dem_seats:
        _add("Dem Seats", cfg.weight_dem_seats * ps.dem_seats_penalty)
    _add("Proportionality",
         cfg.weight_holistic_proportionality * ps.holistic_proportionality)
    _add("Competitiveness",
         cfg.weight_holistic_competitiveness * ps.holistic_competitiveness)
    if cfg.weight_majority_chance_dem:
        _add("D Majority", cfg.weight_majority_chance_dem *
             (1.0 - ps.majority_chance_dem) ** 1.5 * 100)
    if cfg.weight_majority_chance_rep:
        _add("R Majority", cfg.weight_majority_chance_rep *
             (1.0 - ps.majority_chance_rep) ** 1.5 * 100)
    if cfg.weight_hinge:
        _add("Hinge", cfg.weight_hinge * (1.0 - ps.hinge_chance) ** 1.5 * 100)
    return result


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
        self.reock_data: Optional[ReockData] = None
        self.alignment_data: Optional[AlignmentData] = None
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
        try:
            insp = inspect_shapefile(path)
        except Exception as exc:
            # Without this catch the thread dies silently and the GUI sits on
            # "Reading shapefile..." forever.
            import sys as _sys
            from mosaic.crash import write_crash_log
            crash_path = write_crash_log(
                exc, context={"phase": "start_inspection", "shapefile": path}
            )
            log.error(f"start_inspection error: {exc}", exc_info=True)
            print(
                f"\n[mosaic] Could not read shapefile {path}\n"
                f"        {type(exc).__name__}: {exc}\n"
                f"        Log: {crash_path}\n",
                file=_sys.stderr,
            )
            self.state.update(
                status=AlgorithmStatus.ERROR,
                error_message=(
                    f"Could not read shapefile.\n"
                    f"{type(exc).__name__}: {exc}\n"
                    f"Log: {crash_path}"
                ),
                status_message=f"Read failed: {type(exc).__name__} — see crash log",
            )
            return
        self._pending_inspection = insp

        if insp.load_error:
            self.state.update(
                status=AlgorithmStatus.ERROR,
                error_message=f"Could not read shapefile: {insp.load_error}",
                status_message="Read failed — see error panel",
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
            self.graph = load_cached_graph(cache_path, inspection.path, gdf)

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
                save_cached_graph(self.graph, cache_path, inspection.path)
                log.info(f"Built graph: {self.graph.number_of_nodes()} nodes, "
                         f"{self.graph.number_of_edges()} edges")
                self.state.update(
                    status_message=f"Built graph ({self.graph.number_of_edges():,} edges)",
                )

            from mosaic.io.validate import check_connectivity
            conn_issues = check_connectivity(self.graph)
            if conn_issues:
                msg = conn_issues[0]
                log.error(msg)
                self.state.update(
                    status=AlgorithmStatus.ERROR,
                    error_message=msg,
                    status_message=(msg[:80] + "...") if len(msg) > 80 else msg,
                )
                return False

            self.graph_ig = nx_to_igraph(self.graph)
            self.graph_ctx = GraphContext(self.graph_ig)

            # Geometry data for Polsby-Popper: try cache first, recompute on miss.
            pp_cache_path = get_pp_cache_path(inspection.path)
            self.pp_data = load_cached_pp_data(
                pp_cache_path,
                inspection.path,
                n_precincts=n,
                n_edges=self.graph.number_of_edges(),
            )
            if self.pp_data is not None:
                log.info(f"Loaded cached PP geometry from {pp_cache_path}")
                self.state.update(status_message="Loaded cached geometry data")
            else:
                self.state.update(status_message="Precomputing geometry data...")
                self.pp_data = precompute_pp_data(gdf, self.graph)
                if self.pp_data is not None:
                    save_cached_pp_data(self.pp_data, pp_cache_path, inspection.path)
                    log.info(f"Saved PP geometry cache to {pp_cache_path}")

            # Reock directional-extrema precompute (not cached — fast enough to recompute)
            self.reock_data = precompute_reock_data(gdf)

            n_edges = self.graph.number_of_edges()
            log.info(f"Load complete: {n} precincts, {n_edges} edges")
            self.state.update(
                status=AlgorithmStatus.IDLE,
                status_message=f"Ready ({n_edges:,} edges)",
                gdf_ready=True,
            )
            return True

        except Exception as e:
            import sys as _sys
            from mosaic.crash import write_crash_log
            shp = str(getattr(inspection, "path", "?"))
            crash_path = write_crash_log(
                e,
                context={"phase": "complete_load", "shapefile": shp},
            )
            log.error(f"complete_load error: {e}", exc_info=True)
            print(
                f"\n[mosaic] Failed to load {shp}\n"
                f"        {type(e).__name__}: {e}\n"
                f"        Log: {crash_path}\n",
                file=_sys.stderr,
            )
            self.state.update(
                status=AlgorithmStatus.ERROR,
                error_message=(
                    f"{type(e).__name__}: {e}\n"
                    f"Log: {crash_path}"
                ),
                status_message=f"Load failed: {type(e).__name__} — see crash log",
            )
            return False

    # ── Algorithm ─────────────────────────────────────────────────────────────

    def run_algorithm(self):
        """Main ReCom + annealing loop."""
        if self.graph is None or self.populations is None or self.graph_ctx is None:
            self.state.update(status=AlgorithmStatus.ERROR, error_message="No shapefile loaded")
            return

        (num_districts, tolerance, max_iterations, seed,
         score_config, annealing_config,
         county_bias_enabled, county_bias,
         n3_probability, n3_max_attempts_per_stage,
         flip_enabled, flip_midpoint,
         tolerance_ratchet_mode,
         hot_start_assignment) = self.state.get(
            "num_districts", "pop_tolerance", "max_iterations", "seed",
            "score_config", "annealing_config",
            "county_bias_enabled", "county_bias",
            "n3_probability", "n3_max_attempts_per_stage",
            "flip_enabled", "flip_midpoint",
            "tolerance_ratchet_mode",
            "hot_start_assignment",
        )

        # If user supplied a seed, seed BOTH RNGs the algorithm touches:
        # np.random (tree weights, recom edge picks) and Python's random
        # (annealing Metropolis accept). seed=None (GUI value 0) means
        # "fresh random run" -- leave RNG state alone so reruns differ.
        if seed is not None:
            np.random.seed(seed)
            _random.seed(seed)
            log.info(f"Run seed: {seed}")
        else:
            log.info("Run seed: random (no seed set)")

        ideal_pop = self.populations.sum() / num_districts
        ctx = self.graph_ctx
        effective_bias = county_bias if county_bias_enabled else 1.0
        # Freeze the n=3 mix at run start. When n3_enabled is False, the dispatch
        # in the hot loop short-circuits BEFORE np.random.random() is even called,
        # so a pure n=2 run pays only a single bool check per iteration.
        n3_enabled = n3_probability > 0.0
        if n3_enabled:
            log.info(
                f"n=3 ReCom mix enabled: p={n3_probability:.1%}, "
                f"max_attempts_per_stage={n3_max_attempts_per_stage}"
            )
        # Flip dispatch short-circuits when disabled so the legacy ReCom-only
        # RNG sequence stays byte-identical at the cost of a single bool check.
        if flip_enabled:
            log.info(
                "Polish Flips enabled: two-piece logistic ramp "
                f"(5%->50% at p={flip_midpoint:.2f}->85%, steepness={_FLIP_STEEPNESS})"
            )

        # Precompute county constants once per run (county_pops / allowances /
        # max_clean / pops_f). The per-iteration county scorers reuse these
        # instead of rebuilding them every step. Scoring always uses the fixed
        # map-wide tolerance (not the ratchet's active_tolerance), so this stays
        # valid for the whole run.
        county_data = (
            precompute_county_data(
                self.county_array, self.populations, ideal_pop, tolerance,
            )
            if self.county_array is not None
            else None
        )

        _skw = dict(
            county_ids=self.county_array,
            county_data=county_data,
            populations=self.populations,
            ideal_pop=ideal_pop,
            tolerance=tolerance,
            pp_data=self.pp_data,
            reock_data=self.reock_data,
            alignment_data=self.alignment_data,
            n_districts=num_districts,
            dem_votes=self.election_arrays[0][0] if self.election_arrays else None,
            gop_votes=self.election_arrays[0][1] if self.election_arrays else None,
            real_edge_mask=ctx.real_edge_mask,
        )

        # ── Tolerance Ratchet setup ──────────────────────────────────────────
        # active_tolerance starts at the user's map-wide tolerance and only ever
        # tightens. effective_floor is clamped to the start so a preset already
        # at/below the floor stays flat (the ratchet becomes a no-op, never
        # loosens). When on, force pop-deviation components so pop_dev_max is
        # available regardless of the deviation score's weight.
        ratchet_mode = tolerance_ratchet_mode or "off"
        ratchet_on = ratchet_mode in ("standard", "strict")
        ratchet_strict = ratchet_mode == "strict"
        active_tolerance = tolerance
        ratchet_floor = min(_RATCHET_FLOOR, tolerance)
        if ratchet_on:
            _skw["force_pop_components"] = True
            log.info(
                f"Tolerance Ratchet ({ratchet_mode}): {tolerance*100:.2f}% -> "
                f"{ratchet_floor*100:.2f}% over run progress "
                f"[{_RATCHET_RAMP_START:.0%}, 100%]"
            )
        self.state.update(active_pop_tolerance=active_tolerance)

        def _apply_ratchet() -> None:
            """Ratchet the hard tolerance down to hug the current map's max
            deviation, with the scheduled soft ceiling as the floor it can reach.

            target = max(soft_ceiling, map_max_dev): never below what the map
            achieves (no stranding) and never below the schedule (no
            over-tightening ahead of plan). Only ever decreases active_tolerance.
            """
            nonlocal active_tolerance
            # current_ps.pop_dev_max is a percentage; convert to a fraction.
            target = ratchet_step(
                active_tolerance, iteration / max_iterations,
                tolerance, ratchet_floor, current_ps.pop_dev_max / 100.0,
            )
            if target < active_tolerance:
                active_tolerance = target
                self.state.update(active_pop_tolerance=active_tolerance)
                log.info(
                    f"Tolerance Ratchet: tightened to {target*100:.3f}% at iter "
                    f"{iteration} (max dev {current_ps.pop_dev_max:.3f}%)"
                )

        worse_window: deque[int] = deque(maxlen=_ACCEPTANCE_WINDOW)

        try:
            # ── Initial partition ────────────────────────────────────────────
            log.info(
                f"Partitioning: {num_districts} districts, "
                f"{tolerance*100:.1f}% tolerance, {max_iterations} iters"
            )
            if hot_start_assignment is not None:
                self.state.update(
                    status=AlgorithmStatus.PARTITIONING,
                    status_message="Using hot start assignment...",
                )
                assignment = hot_start_assignment.astype(np.int32).copy()
                log.info(
                    f"Hot start: skipping bipartition, using preloaded assignment "
                    f"({int(assignment.max()) + 1} districts, {len(assignment)} precincts)"
                )
            else:
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
                    should_cancel=lambda: (
                        self.state.check_should_stop() or self.state.check_should_pause()
                    ),
                )
                if assignment is None:
                    self.state.request_resume()
                    self.state.update(status=AlgorithmStatus.IDLE, status_message="")
                    return
                log.info("Initial partition complete")
            self.state.update(initial_assignment=assignment.copy())

            # ── Initial score ────────────────────────────────────────────────
            cut_edge_indices = ctx.compute_cut_edges(assignment)
            current_ps = score_plan(cut_edge_indices, score_config,
                                    assignment=assignment, **_skw)

            with self.state._lock:
                self.state.score_history.append(current_ps.total)
                self.state.county_splits_score_history.append(
                    current_ps.county_excess_score + current_ps.county_unified_score)
                self.state.county_excess_splits_history.append(current_ps.county_excess_splits)
                self.state.county_unified_districts_history.append(current_ps.county_unified_districts)
                self.state.mm_history.append(current_ps.mean_median)
                self.state.eg_history.append(current_ps.efficiency_gap)
                self.state.dem_seats_history.append(current_ps.dem_seats)
                self.state.pp_history.append(1.0 - current_ps.polsby_popper / 100.0)
                self.state.reock_history.append(1.0 - current_ps.reock / 100.0)
                self.state.holistic_compactness_history.append(current_ps.holistic_compactness)
                self.state.holistic_splitting_history.append(current_ps.holistic_splitting)
                self.state.holistic_proportionality_history.append(current_ps.holistic_proportionality)
                self.state.holistic_competitiveness_history.append(current_ps.holistic_competitiveness)
                self.state.pop_deviation_history.append(current_ps.pop_deviation)
                self.state.pop_dev_max_history.append(current_ps.pop_dev_max)
                self.state.pop_dev_mean_history.append(current_ps.pop_dev_mean)
                self.state.alignment_mean_ret_history.append(current_ps.alignment_mean_ret)
                self.state.alignment_min_ret_history.append(current_ps.alignment_min_ret)
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
            _launch_watch_fired = False

            for iteration in range(1, max_iterations + 1):
                # Launch Watch: re-anchor temperature after the first N iters
                # using the post-warmup current score rather than the initial.
                if (
                    ann is not None
                    and annealing_config.launch_watch
                    and not _launch_watch_fired
                    and iteration > annealing_config.launch_watch_iter
                ):
                    remaining = max_iterations - iteration + 1
                    relaunch_temperature(
                        ann, current_ps.total, annealing_config, remaining,
                    )
                    _launch_watch_fired = True
                    log.info(
                        f"Launch Watch (iter {iteration}): re-anchored temp "
                        f"to {ann.initial_temp:.3f}, "
                        f"cooling_rate={ann.cooling_rate:.6f}"
                    )
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

                # ── Proposal dispatch ────────────────────────────────────────
                # Order: flip wins over n=3, which wins over n=2. Each branch
                # short-circuits when its enable flag is False so a pure
                # n=2 ReCom run pays only a single bool check per iteration.
                if flip_enabled:
                    _progress = iteration / max_iterations if max_iterations > 0 else 0.0
                    _flip_p = flip_rate_curve(_progress, flip_midpoint, _FLIP_STEEPNESS)
                else:
                    _flip_p = 0.0
                if flip_enabled and np.random.random() < _flip_p:
                    new_assignment, valid, new_cut_indices = flip_step_ig(
                        ctx, assignment, self.populations, ideal_pop, active_tolerance,
                        cut_edge_indices,
                        county_array=self.county_array,
                        county_bias=effective_bias,
                    )
                elif n3_enabled and np.random.random() < n3_probability:
                    new_assignment, valid, new_cut_indices = recom_step_ig_n3(
                        ctx, assignment, self.populations, ideal_pop, active_tolerance,
                        cut_edge_indices,
                        county_array=self.county_array, county_bias=effective_bias,
                        max_attempts_per_stage=n3_max_attempts_per_stage,
                    )
                else:
                    new_assignment, valid, new_cut_indices = recom_step_ig(
                        ctx, assignment, self.populations, ideal_pop, active_tolerance,
                        cut_edge_indices,
                        county_array=self.county_array, county_bias=effective_bias,
                    )

                if not valid:
                    self.state.update(current_iteration=iteration,
                                      current_flip_rate=_flip_p)
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
                        current_flip_rate=_flip_p,
                        current_assignment=assignment.copy(),
                        current_score=current_ps.total,
                        current_cut_edges=current_ps.cut_edges,
                        successful_steps=self.state.successful_steps + 1,
                        score_breakdown=_build_score_breakdown(current_ps, score_config),
                    )
                else:
                    self.state.update(current_iteration=iteration,
                                      current_flip_rate=_flip_p)

                n_score = n_temp = n_acc = 0
                with self.state._lock:
                    self.state.score_history.append(current_ps.total)
                    self.state.county_splits_score_history.append(
                        current_ps.county_excess_score + current_ps.county_unified_score)
                    self.state.county_excess_splits_history.append(current_ps.county_excess_splits)
                    self.state.county_unified_districts_history.append(current_ps.county_unified_districts)
                    self.state.mm_history.append(current_ps.mean_median)
                    self.state.eg_history.append(current_ps.efficiency_gap)
                    self.state.dem_seats_history.append(current_ps.dem_seats)
                    self.state.pp_history.append(1.0 - current_ps.polsby_popper / 100.0)
                    self.state.reock_history.append(1.0 - current_ps.reock / 100.0)
                    self.state.holistic_compactness_history.append(current_ps.holistic_compactness)
                    self.state.holistic_splitting_history.append(current_ps.holistic_splitting)
                    self.state.holistic_proportionality_history.append(current_ps.holistic_proportionality)
                    self.state.holistic_competitiveness_history.append(current_ps.holistic_competitiveness)
                    self.state.pop_deviation_history.append(current_ps.pop_deviation)
                    self.state.pop_dev_max_history.append(current_ps.pop_dev_max)
                    self.state.pop_dev_mean_history.append(current_ps.pop_dev_mean)
                    self.state.alignment_mean_ret_history.append(current_ps.alignment_mean_ret)
                    self.state.alignment_min_ret_history.append(current_ps.alignment_min_ret)
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
                    # Standard ratchet: only new-best plans are eligible.
                    if ratchet_on and not ratchet_strict:
                        _apply_ratchet()

                # Strict ratchet: every iteration is eligible.
                if ratchet_strict:
                    _apply_ratchet()

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
            from mosaic.crash import write_crash_log
            crash_path = write_crash_log(
                e,
                context={
                    "phase": "run_algorithm",
                    "iteration": locals().get("iteration", "?"),
                    "max_iterations": locals().get("max_iterations", "?"),
                },
            )
            log.error(f"Algorithm error: {e}", exc_info=True)
            self.state.update(
                status=AlgorithmStatus.ERROR,
                error_message=f"{type(e).__name__}: {e}  (log: {crash_path})",
            )
