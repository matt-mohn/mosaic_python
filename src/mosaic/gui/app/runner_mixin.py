"""Run/pause/reset/revert control and district renumbering."""
from ._common import (
    _DIR_TO_MODE,
    AlgorithmStatus,
    AnnealingConfig,
    ScoreConfig,
    dpg,
    np,
    threading,
    warnings,
)


class RunnerMixin:
    """Run/pause/reset/revert control and district renumbering."""

    def _on_revert_to_best(self):
        """Rewind display to the best-scoring iteration seen so far."""
        with self.state._lock:
            if self.state.best_assignment is None:
                return
            best_assign = self.state.best_assignment.copy()
            best_score  = self.state.best_score
            best_iter   = self.state.best_iteration
            best_cuts   = self.state.best_cut_edges
            best_temp   = self.state.best_temperature
            n_score     = self.state.best_score_history_len
            n_temp      = self.state.best_temp_history_len
            n_acc       = self.state.best_acc_history_len
            n_dist      = self.state.num_districts
            max_iter    = self.state.max_iterations
            _init       = (self.state.initial_assignment.copy()
                           if self.state.initial_assignment is not None else None)
            self.state.score_history           = self.state.score_history[:n_score]
            self.state.temperature_history     = self.state.temperature_history[:n_temp]
            self.state.acceptance_rate_history = self.state.acceptance_rate_history[:n_acc]
            self.state.county_splits_score_history    = self.state.county_splits_score_history[:n_score]
            self.state.county_excess_splits_history   = self.state.county_excess_splits_history[:n_score]
            self.state.county_unified_districts_history = self.state.county_unified_districts_history[:n_score]
            self.state.mm_history               = self.state.mm_history[:n_score]
            self.state.eg_history               = self.state.eg_history[:n_score]
            self.state.partisan_bias_history    = self.state.partisan_bias_history[:n_score]
            self.state.partisan_gini_history    = self.state.partisan_gini_history[:n_score]
            self.state.dem_seats_history        = self.state.dem_seats_history[:n_score]
            self.state.pp_history               = self.state.pp_history[:n_score]
            self.state.reock_history            = self.state.reock_history[:n_score]
            self.state.holistic_compactness_history = self.state.holistic_compactness_history[:n_score]
            self.state.holistic_splitting_history = self.state.holistic_splitting_history[:n_score]
            self.state.holistic_proportionality_history = self.state.holistic_proportionality_history[:n_score]
            self.state.holistic_competitiveness_history = self.state.holistic_competitiveness_history[:n_score]
            self.state.pop_deviation_history = self.state.pop_deviation_history[:n_score]
            self.state.pop_dev_max_history   = self.state.pop_dev_max_history[:n_score]
            self.state.pop_dev_mean_history  = self.state.pop_dev_mean_history[:n_score]
            self.state.cut_edges_history        = self.state.cut_edges_history[:n_score]
            # Lockstep with score_history; trim here too so the buffers below
            # don't re-grow from the post-best tail on the next frame.
            self.state.alignment_mean_ret_history = self.state.alignment_mean_ret_history[:n_score]
            self.state.alignment_min_ret_history  = self.state.alignment_min_ret_history[:n_score]
            self.state.majority_dem_history     = self.state.majority_dem_history[:n_score]
            self.state.majority_rep_history     = self.state.majority_rep_history[:n_score]
            self.state.hinge_history            = self.state.hinge_history[:n_score]

        # Tell the worker to stop, then wait for it to exit before we touch
        # state -- a worker parked in its pause-wait loop can otherwise wake
        # up and resume normal iteration mid-revert ("paused then unpaused
        # on revert").
        self.state.request_stop()
        if self.algorithm_thread is not None and self.algorithm_thread.is_alive():
            self.algorithm_thread.join(timeout=0.5)

        self.state.update(
            status=AlgorithmStatus.IDLE,
            status_message=f"Reverted to best (iteration {best_iter:,})",
            current_assignment=best_assign,
            current_score=best_score,
            current_cut_edges=best_cuts,
            current_iteration=best_iter,
            current_temperature=best_temp,
        )
        if max_iter > 0:
            dpg.set_value(self._progress, best_iter / max_iter)

        # Trim local history buffers to match the reverted-to state
        for buf in (
            self._buf_score, self._buf_cs_score, self._buf_cs_excess,
            self._buf_cs_clean, self._buf_mm, self._buf_eg,
            self._buf_seats, self._buf_pp, self._buf_reock,
            self._buf_hc, self._buf_hsplit, self._buf_hprop, self._buf_hcmp,
            self._buf_popdev,
            self._buf_popdev_max, self._buf_popdev_mean, self._buf_cuts,
            self._buf_align_mean, self._buf_align_min,
            self._buf_maj_dem, self._buf_maj_rep, self._buf_hinge,
        ):
            buf.trim_to(best_iter, n_score)
        self._buf_temp.trim_to(best_iter, n_temp)
        self._buf_acc.trim_to(best_iter, n_acc)

        if self.map_view is not None and self.map_view._loaded:
            self.map_view.render_assignment(best_assign, n_dist, _init)

    def _on_run(self):
        if self.runner is None or self.runner.graph is None:
            self.state.update(status=AlgorithmStatus.ERROR,
                              error_message="Please load a shapefile first")
            return

        # A prior run's thread (paused runs keep theirs alive, as Relight does)
        # must be stopped first, or it races the new run over shared state.
        if self.algorithm_thread is not None and self.algorithm_thread.is_alive():
            self.state.request_stop()
            self.algorithm_thread.join(timeout=2.0)

        # Relight reseeds each run from the live map, via the same seed slot the
        # runner already reads for Hot Start (the two are mutually exclusive). No
        # map yet (e.g. right after Reset) -> clear the slot -> random seed.
        if self._relight_active:
            with self.state._lock:
                cur = (self.state.current_assignment.copy()
                       if self.state.current_assignment is not None else None)
            self.state.update(hot_start_assignment=cur)

        # Seed sanity: validated when the seed was made, but the Districts
        # slider or the Population Tolerance could have moved since (Relight
        # especially reseeds from a map built under the old settings). Re-check
        # both against the current values.
        (hot_start,) = self.state.get("hot_start_assignment")
        if hot_start is not None:
            what = "Relight map" if self._relight_active else "Hot start"
            fix  = ("Clear Relight" if self._relight_active
                    else "Clear the hot start")
            n_dist_slider = dpg.get_value(self._num_districts)
            n_dist_seed = int(hot_start.max()) + 1
            if n_dist_seed != n_dist_slider:
                self._show_hot_start_error(
                    f"{what} has {n_dist_seed} districts but the Districts "
                    f"slider is set to {n_dist_slider}. {fix} or adjust "
                    f"the slider."
                )
                return
            pops = self.runner.populations
            if pops is not None and n_dist_slider > 0:
                pop_f = pops.astype(np.float64)
                ideal = pop_f.sum() / n_dist_slider
                if ideal > 0:
                    dist_pop = np.bincount(
                        hot_start.astype(np.int64), weights=pop_f,
                        minlength=n_dist_slider)
                    max_dev = float(np.abs((dist_pop - ideal) / ideal).max())
                    tol = dpg.get_value(self._tolerance) / 100.0
                    if max_dev > tol:
                        self._show_hot_start_error(
                            f"{what} has a district at {max_dev * 100:.2f}% "
                            f"population deviation, over the {tol * 100:.2f}% "
                            f"Population Tolerance. Loosen the tolerance "
                            f"or {fix}."
                        )
                        return

        cs_on      = dpg.get_value(self._cs_enabled)
        mm_on      = dpg.get_value(self._mm_enabled)
        eg_on      = dpg.get_value(self._eg_enabled)
        pb_on      = dpg.get_value(self._pb_enabled)
        pg_on      = dpg.get_value(self._pg_enabled)
        seats_on   = dpg.get_value(self._seats_enabled)
        maj_on     = dpg.get_value(self._majority_enabled)
        robust_eg = dpg.get_value(self._eg_mode) == "Robust (recommended)"

        w_cut  = (dpg.get_value(self._w_cut_edges)
                  if dpg.get_value(self._cut_enabled) else 0.0)
        w_cs_excess  = dpg.get_value(self._w_county_excess)  if cs_on else 0.0
        w_cs_unified = dpg.get_value(self._w_county_unified) if cs_on else 0.0
        w_pp   = (dpg.get_value(self._w_polsby_popper)
                  if dpg.get_value(self._pp_enabled) else 0.0)
        w_reock = (dpg.get_value(self._w_reock)
                   if dpg.get_value(self._reock_enabled) else 0.0)
        w_hc   = (dpg.get_value(self._w_holistic_compactness)
                  if dpg.get_value(self._hc_enabled) else 0.0)
        w_hsplit = (dpg.get_value(self._w_holistic_splitting)
                    if dpg.get_value(self._hsplit_enabled) else 0.0)
        w_hprop = (dpg.get_value(self._w_holistic_proportionality)
                   if dpg.get_value(self._hprop_enabled) else 0.0)
        w_hcmp  = (dpg.get_value(self._w_holistic_competitiveness)
                   if dpg.get_value(self._hcmp_enabled) else 0.0)
        w_pd   = (dpg.get_value(self._w_pop_deviation)
                  if dpg.get_value(self._popdev_enabled) else 0.0)
        # Alignment only counts when enabled AND a reference plan is loaded.
        _align_on = (dpg.get_value(self._alignment_enabled)
                     and self.runner is not None
                     and self.runner.alignment_data is not None)
        w_align = dpg.get_value(self._w_alignment) if _align_on else 0.0
        _align_focus = {
            "All residents": "none", "Republican": "rep", "Democratic": "dem",
        }[dpg.get_value(self._alignment_focus)]
        # Safe harbor cannot exceed population tolerance
        _tol    = dpg.get_value(self._tolerance) / 100.0
        _harbor = min(dpg.get_value(self._pop_dev_harbor) / 100.0, _tol)

        n_dist_run = dpg.get_value(self._num_districts)

        score_cfg = ScoreConfig(
            weight_cut_edges=w_cut,
            weight_county_excess=w_cs_excess,
            weight_county_unified=w_cs_unified,
            weight_holistic_splitting=w_hsplit,
            holistic_splitting_unclipped=dpg.get_value(self._hsplit_unclipped),
            weight_polsby_popper=w_pp,
            weight_reock=w_reock,
            weight_holistic_compactness=w_hc,
            weight_pop_deviation=w_pd,
            pop_deviation_safe_harbor=_harbor,
            weight_alignment=w_align,
            alignment_party_focus=_align_focus,
            alignment_restrict_to_party=dpg.get_value(self._alignment_restrict),
            alignment_win_threshold=dpg.get_value(self._alignment_win_threshold),
            weight_mean_median=dpg.get_value(self._w_mean_median) if mm_on else 0.0,
            mm_mode=_DIR_TO_MODE[dpg.get_value(self._mm_dir)],
            mm_bound=dpg.get_value(self._mm_bound),
            weight_efficiency_gap=dpg.get_value(self._w_efficiency_gap) if eg_on else 0.0,
            eg_mode=_DIR_TO_MODE[dpg.get_value(self._eg_dir)],
            eg_bound=dpg.get_value(self._eg_bound),
            partisan_quadratic_penalty=dpg.get_value(self._partisan_quadratic_penalty),
            use_robust_eg=robust_eg,
            weight_partisan_bias=dpg.get_value(self._w_partisan_bias) if pb_on else 0.0,
            pbias_mode=_DIR_TO_MODE[dpg.get_value(self._pb_dir)],
            pbias_bound=dpg.get_value(self._pbias_bound),
            weight_partisan_gini=dpg.get_value(self._w_partisan_gini) if pg_on else 0.0,
            weight_dem_seats=dpg.get_value(self._w_dem_seats) if seats_on else 0.0,
            dem_seats_favor_dem=(dpg.get_value(self._dem_seats_dir) == "D"),
            weight_holistic_proportionality=w_hprop,
            weight_holistic_competitiveness=w_hcmp,
            competitiveness_unclipped=dpg.get_value(self._comp_unclipped),
            weight_majority_chance_dem=(dpg.get_value(self._w_majority)
                                        if maj_on and dpg.get_value(self._majority_dem_chk)
                                        else 0.0),
            weight_majority_chance_rep=(dpg.get_value(self._w_majority)
                                        if maj_on and dpg.get_value(self._majority_rep_chk)
                                        else 0.0),
            election_win_prob_at_55=dpg.get_value(self._win_prob),
            election_swing_sigma=dpg.get_value(self._swing_sigma),
            weight_hinge=(dpg.get_value(self._w_hinge)
                          if dpg.get_value(self._hinge_enabled) else 0.0),
            hinge_threshold=max(1, min(dpg.get_value(self._hinge_threshold), n_dist_run)),
            hinge_dem=dpg.get_value(self._hinge_dem_chk),
        )

        guided = dpg.get_value(self._cool_mode) == "Guided (recommended)"
        ann_cfg = AnnealingConfig(
            enabled=dpg.get_value(self._ann_enabled),
            initial_temp_factor=dpg.get_value(self._temp_factor),
            cooling_mode="GUIDED" if guided else "STATIC",
            guide_fraction=dpg.get_value(self._guide_frac),
            target_temp=dpg.get_value(self._target_temp),
            cooling_rate=dpg.get_value(self._cooling_rate),
            launch_watch=dpg.get_value(self._launch_watch_enabled),
            launch_watch_iter=dpg.get_value(self._launch_watch_iter),
        )
        seed = dpg.get_value(self._seed) or None

        cb_en  = dpg.get_value(self._county_bias_enabled)
        cb_val = dpg.get_value(self._county_bias)

        self.state.update(
            num_districts=dpg.get_value(self._num_districts),
            pop_tolerance=_tol,
            tolerance_ratchet_mode={
                "Off": "off", "Standard": "standard", "Strict": "strict",
            }[dpg.get_value(self._tolerance_ratchet_mode)],
            max_iterations=dpg.get_value(self._iterations),
            seed=seed,
            score_config=score_cfg,
            annealing_config=ann_cfg,
            map_render_interval=dpg.get_value(self._map_interval),
            county_bias_enabled=cb_en,
            county_bias=cb_val,
            n3_probability=dpg.get_value(self._n3_pct) / 100.0,
            flip_enabled=dpg.get_value(self._flip_enabled),
            flip_midpoint=dpg.get_value(self._flip_midpoint) / 100.0,
        )
        self._clear_all_series()
        self.state.reset_run()

        self.algorithm_thread = threading.Thread(
            target=self.runner.run_algorithm, daemon=True, name="algo",
        )
        self.algorithm_thread.start()

    def _on_pause(self):
        if self.state.status == AlgorithmStatus.PAUSED:
            self.state.request_resume()
        else:
            self.state.request_pause()

    def _on_reset(self):
        self.state.request_stop()
        with self.state._lock:
            self.state.score_history = []
            self.state.temperature_history = []
            self.state.acceptance_rate_history = []
            self.state.county_splits_score_history = []
            self.state.county_excess_splits_history = []
            self.state.county_unified_districts_history = []
            self.state.mm_history = []
            self.state.eg_history = []
            self.state.partisan_bias_history = []
            self.state.partisan_gini_history = []
            self.state.dem_seats_history = []
            self.state.pp_history = []
            self.state.reock_history = []
            self.state.holistic_compactness_history = []
            self.state.holistic_splitting_history = []
            self.state.holistic_proportionality_history = []
            self.state.holistic_competitiveness_history = []
            self.state.pop_deviation_history = []
            self.state.pop_dev_max_history = []
            self.state.pop_dev_mean_history = []
            self.state.cut_edges_history = []
            self.state.majority_dem_history = []
            self.state.majority_rep_history = []
            self.state.hinge_history = []
        self.state.update(
            status=AlgorithmStatus.IDLE,
            status_message="",
            current_score=float("inf"),
            best_score=float("inf"),
            current_cut_edges=0,
            current_iteration=0,
            successful_steps=0,
            current_flip_rate=0.0,
            current_temperature=0.0,
            accepted_worse=0,
            rejected_worse=0,
            best_assignment=None,
            current_assignment=None,
            district_label_map=None,
            score_breakdown={},
        )
        # Reset wipes the current map, so a Relight armed against it no longer
        # applies -- turn it off (restoring the pre-Relight annealing settings).
        self._clear_relight()
        # Reset status bar and score readouts
        dpg.set_value(self._status_txt, "Status: Idle")
        dpg.set_value(self._iter_txt,   "Iteration: 0 / 0")
        dpg.set_value(self._time_txt,   "Time: 0:00")
        dpg.set_value(self._ips_txt,    "Iter/sec: 0.0")
        dpg.set_value(self._eta_txt,    "Est. left: --")
        dpg.set_value(self._progress,   0.0)
        dpg.set_value(self._score_txt,  "Score: --")
        dpg.set_value(self._best_txt,   "Best:  --   (iter. --)")
        dpg.set_value(self._temp_txt,   "Temperature: --")
        dpg.set_value(self._acc_txt,    "Entropy: --")
        dpg.set_value(self._succ_txt,   "Accepted steps: --")
        dpg.set_value(self._flip_txt,   "Flip rate: 0.0%")
        self._clear_all_series()
        if self.map_view is not None and self.map_view._loaded:
            self.map_view.draw_blank()

    def _stable_labeled_assignment(self, assignment) -> "np.ndarray | None":
        """`assignment` relabeled to match the live map / District Info labels.

        ReCom internally renumbers districts over the course of a run; the GUI
        uses ``stable_color_mapping`` against the initial partition to keep the
        same physical region under the same number across iterations. CSV
        exports must apply the same permutation, otherwise users see one label
        in Mosaic and a different one in their spreadsheet.
        """
        if assignment is None:
            return None
        initial = self.state.initial_assignment
        n_dist = self.state.num_districts or int(assignment.max()) + 1
        if initial is not None and len(initial) == len(assignment):
            from mosaic.gui.map_view import stable_color_mapping
            return stable_color_mapping(assignment, initial, n_dist)
        return assignment

    def _export_labeled_assignment(self, assignment) -> "np.ndarray | None":
        """`assignment` with any geographic renumber applied, 0-indexed.

        Exports add 1 to produce the district column, so we return the
        geographic number minus 1. With no renumber active this is just the
        stable-labeled assignment.
        """
        stable = self._stable_labeled_assignment(assignment)
        if stable is None:
            return None
        lm = self.state.district_label_map
        n_dist = self.state.num_districts or int(stable.max()) + 1
        if lm is not None and len(lm) == n_dist:
            return (lm[stable] - 1).astype(stable.dtype)
        return stable

    def _renumber_render(self, label_map) -> None:
        """Store label_map in state and re-render the static map with it."""
        self.state.update(district_label_map=label_map)
        with self.state._lock:
            assignment = (self.state.current_assignment.copy()
                          if self.state.current_assignment is not None else None)
            initial = (self.state.initial_assignment.copy()
                       if self.state.initial_assignment is not None else None)
            n_dist = self.state.num_districts
        if self.map_view is not None and assignment is not None and n_dist > 0:
            self.map_view.district_label_map = label_map
            self.map_view.fast_labels = False   # static map: precise labels
            self.map_view.render_assignment(assignment, n_dist, initial)

    def _renumber_centroids_xy(self):
        """Per-precinct centroid (x, y) in the gdf CRS, cached by gdf identity."""
        gdf = self.runner.gdf
        gid = id(gdf)
        cached = self._renumber_centroids
        if cached is not None and cached[0] == gid:
            return cached[1], cached[2]
        # Centroids are only used for relative geographic ordering of districts,
        # so the geographic-CRS "results likely incorrect" warning is moot here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Results from 'centroid' are likely incorrect.*",
            )
            cents = gdf.geometry.centroid
            x = np.asarray(cents.x, dtype=np.float64)
            y = np.asarray(cents.y, dtype=np.float64)
        self._renumber_centroids = (gid, x, y)
        return x, y

    def _apply_renumber(self, rule: "str | None") -> None:
        """Apply (or clear, if rule is None) a geographic renumber to the
        current static map. Numbers only -- colors are untouched. No-op while a
        run is in progress. Does not write to the status bar.
        """
        if rule is None:
            self._renumber_render(None)
            return
        if self.runner is None or self.runner.gdf is None \
                or self.runner.populations is None:
            return
        with self.state._lock:
            status = self.state.status
            assignment = (self.state.current_assignment.copy()
                          if self.state.current_assignment is not None else None)
            initial = (self.state.initial_assignment.copy()
                       if self.state.initial_assignment is not None else None)
            n_dist = self.state.num_districts
        if status in (AlgorithmStatus.RUNNING, AlgorithmStatus.PARTITIONING):
            return
        if assignment is None or n_dist <= 0:
            return

        # Rank by stable color index so numbers line up with colors/panels.
        from mosaic.gui.map_view import stable_color_mapping
        if initial is not None and len(initial) == len(assignment):
            stable = stable_color_mapping(assignment, initial, n_dist)
        else:
            stable = assignment

        if rule == "infer":
            label_map = self._infer_label_map(stable, n_dist)
            if label_map is None:        # no alignment plan loaded -> no-op
                return
        elif rule == "proximity":
            from mosaic.renumber import proximity_label_map
            x, y = self._renumber_centroids_xy()
            pops = np.asarray(self.runner.populations, dtype=np.float64)
            label_map = proximity_label_map(stable, x, y, pops, n_dist)
        else:
            from mosaic.renumber import geographic_label_map
            x, y = self._renumber_centroids_xy()
            pops = np.asarray(self.runner.populations, dtype=np.float64)
            label_map = geographic_label_map(stable, x, y, pops, n_dist, rule=rule)
        self._renumber_render(label_map)

    def _infer_label_map(self, stable: np.ndarray, n_dist: int):
        """Label map adopting the loaded reference plan's numbers, or None if no
        alignment plan is loaded. District counts need not match: the matching
        is rectangular and unmatched proposed districts get fresh numbers.

        The overlap is measured in whatever the alignment is set to align on:
        residents (population), D votes, or R votes -- mirroring the alignment
        score's party-focus setting.
        """
        ad = getattr(self.runner, "alignment_data", None)
        if ad is None or ad.alt_labels is None:
            return None
        from mosaic.renumber import infer_label_map_from_reference

        focus = self.state.score_config.alignment_party_focus
        dem = gop = None
        if self.runner.election_arrays:
            dem, gop = self.runner.election_arrays[0]
        if focus == "rep" and gop is not None:
            weights = np.asarray(gop, dtype=np.float64)
        elif focus == "dem" and dem is not None:
            weights = np.asarray(dem, dtype=np.float64)
        else:
            weights = np.asarray(self.runner.populations, dtype=np.float64)

        return infer_label_map_from_reference(
            stable, ad.alt_assignment, ad.alt_labels, weights,
            n_dist, ad.n_alt_districts,
        )

    # ── Renumber control sync (Advanced menu check + options radio) ──────────────

    _RENUMBER_RULE_TO_LABEL = {
        "nw_se":      "Northwest to Southeast",
        "n_s":        "North to South",
        "proximity":  "By proximity",
        "infer":      "Infer from alignment",
    }
    _RENUMBER_LABEL_TO_RULE = {v: k for k, v in _RENUMBER_RULE_TO_LABEL.items()}

    def _alignment_loaded(self) -> bool:
        return (self.runner is not None
                and getattr(self.runner, "alignment_data", None) is not None)

    def _renumber_radio_items(self) -> list:
        items = ["By proximity", "Northwest to Southeast", "North to South", "None"]
        if self._alignment_loaded():
            items.append("Infer from alignment")
        return items

    def _renumber_rule_label(self) -> str:
        """Radio label reflecting current renumber state."""
        if not self._renumber_enabled:
            return "None"
        return self._RENUMBER_RULE_TO_LABEL.get(self._renumber_rule, "By proximity")

    def _sync_renumber_widgets(self) -> None:
        dpg.set_value(self._renumber_after_run, self._renumber_enabled)
        if dpg.does_item_exist("renumber_rule_radio"):
            dpg.set_value("renumber_rule_radio", self._renumber_rule_label())

    def _maybe_live_renumber(self) -> None:
        """Reflect the current setting on the displayed static map immediately."""
        self._apply_renumber(self._renumber_rule if self._renumber_enabled else None)

    def _on_renumber_after_run_toggle(self, sender, app_data) -> None:
        self._renumber_enabled = bool(app_data)
        if self._renumber_enabled and self._renumber_rule not in ("nw_se", "n_s", "proximity"):
            self._renumber_rule = "proximity"
        self._sync_renumber_widgets()
        self._maybe_live_renumber()

    def _on_renumber_rule_change(self, sender, app_data) -> None:
        if app_data == "None":
            self._renumber_enabled = False
        else:
            self._renumber_enabled = True
            self._renumber_rule = self._RENUMBER_LABEL_TO_RULE.get(app_data, "proximity")
        self._sync_renumber_widgets()
        self._maybe_live_renumber()

    def _on_open_renumber_options(self) -> None:
        # "Infer from alignment" only appears when a reference plan is loaded;
        # if it was selected and the plan is gone, fall back to the diagonal.
        if self._renumber_rule == "infer" and not self._alignment_loaded():
            self._renumber_rule = "proximity"
        # Rebuilt each open (radio items depend on current state); the helper
        # deletes any prior instance. _sync_renumber_widgets() guards on the tag
        # existing, so deleting it on close is safe.
        with self._dialog(
            "Renumber Options", "renumber_options_window", (240, 170),
            secondary=("Close",
                       lambda: dpg.delete_item("renumber_options_window")),
        ):
            dpg.add_text("Renumber districts:")
            dpg.add_separator()
            dpg.add_radio_button(
                items=self._renumber_radio_items(),
                tag="renumber_rule_radio",
                default_value=self._renumber_rule_label(),
                callback=self._on_renumber_rule_change,
            )
        dpg.focus_item("renumber_options_window")
