"""Per-frame UI refresh: live plots, tables, and status labels."""
from ._common import (
    _CONTRIB_BAR_METRICS,
    _PLOT_LIMIT,
    _RENDER_TARGET,
    AlgorithmStatus,
    Path,
    _fmt_dur,
    _SeriesBuffer,
    dpg,
    np,
    threading,
    time,
)


class UpdatesMixin:
    """Per-frame UI refresh: live plots, tables, and status labels."""

    def _update_district_info_table(self) -> None:
        """Recompute the district info table from the live assignment.

        Called once per frame while the panel is visible.  Rows are rebuilt
        only when the district count changes; otherwise we just rewrite
        cell values via set_value so column widths stay stable.
        """
        if not dpg.is_item_shown("panel_district_info"):
            return
        if self.runner is None or self.runner.populations is None:
            dpg.configure_item("district_info_empty_lbl", show=True)
            return

        with self.state._lock:
            assignment = (self.state.current_assignment.copy()
                          if self.state.current_assignment is not None else None)
            initial = (self.state.initial_assignment.copy()
                       if self.state.initial_assignment is not None else None)
            n_dist = self.state.num_districts
            current_iter = self.state.current_iteration
            status = self.state.status
            label_map = self.state.district_label_map

        # Throttle: while running, only refresh every N iterations (per the
        # slider).  When idle/paused, always show fresh data.  An update is
        # also forced when the district count changes or the panel was just
        # re-opened (_dist_info_last_iter reset to -1).
        is_running = status in (AlgorithmStatus.RUNNING,
                                AlgorithmStatus.PARTITIONING)
        interval = max(1, int(dpg.get_value(self._dist_info_update_every)))
        force = (self._dist_info_last_iter < 0
                 or self._dist_table_built_rows != n_dist)
        if is_running and not force:
            if current_iter - self._dist_info_last_iter < interval:
                return

        if assignment is None or n_dist <= 0:
            dpg.configure_item("district_info_empty_lbl", show=True)
            return
        dpg.configure_item("district_info_empty_lbl", show=False)

        populations = self.runner.populations
        ideal_pop = float(populations.sum()) / n_dist if n_dist > 0 else 1.0

        # Per-district populations and deviation
        pop_d = np.bincount(assignment, weights=populations.astype(np.float64),
                            minlength=n_dist)
        pop_dev_pct = (pop_d - ideal_pop) / ideal_pop * 100.0

        # Polsby-Popper per district (mirror logic from io/export.py)
        pp_data = self.runner.pp_data
        if pp_data is not None and len(pp_data.areas) == len(assignment):
            _FOUR_PI = 4.0 * np.pi
            dist_area = np.bincount(assignment, weights=pp_data.areas,
                                    minlength=n_dist).astype(np.float64)
            dist_perim = np.bincount(assignment, weights=pp_data.ext_perimeters,
                                     minlength=n_dist).astype(np.float64)
            eu, ev, elen = pp_data.edge_u, pp_data.edge_v, pp_data.edge_len
            if len(eu) > 0:
                eu_d = assignment[eu]
                ev_d = assignment[ev]
                is_cut = eu_d != ev_d
                if is_cut.any():
                    cut_len = elen[is_cut]
                    np.add.at(dist_perim, eu_d[is_cut], cut_len)
                    np.add.at(dist_perim, ev_d[is_cut], cut_len)
            safe_perim = np.where(dist_perim > 0.0, dist_perim, 1.0)
            pp_per_dist = np.clip(_FOUR_PI * dist_area / (safe_perim ** 2),
                                  0.0, 1.0)
        else:
            pp_per_dist = None

        # Vote shares
        dem_pct = rep_pct = None
        if (self.runner.election_arrays
                and len(self.runner.election_arrays[0][0]) == len(assignment)):
            dem, gop = self.runner.election_arrays[0]
            dem_d = np.bincount(assignment, weights=dem.astype(np.float64),
                                minlength=n_dist)
            gop_d = np.bincount(assignment, weights=gop.astype(np.float64),
                                minlength=n_dist)
            total_d = dem_d + gop_d
            with np.errstate(invalid="ignore"):
                dem_pct = np.where(total_d > 0, dem_d / total_d * 100.0, np.nan)
                rep_pct = np.where(total_d > 0, gop_d / total_d * 100.0, np.nan)

        # Stable district -> displayed label (matches map labelling). A
        # geographic renumber maps the stable color index through label_map.
        use_lm = label_map is not None and len(label_map) == n_dist
        d_to_label = {}
        if initial is not None and len(initial) == len(assignment):
            from mosaic.gui.map_view import stable_color_mapping
            stable_colors = stable_color_mapping(assignment, initial, n_dist)
            for d in range(n_dist):
                mask = assignment == d
                if mask.any():
                    si = int(stable_colors[mask][0])
                    d_to_label[d] = int(label_map[si]) if use_lm else si + 1
        else:
            for d in range(n_dist):
                d_to_label[d] = int(label_map[d]) if use_lm else d + 1

        # Display order: rows sorted by displayed label ascending. Labels are
        # 1..k for the default/geographic numbering but can be non-contiguous
        # under "Infer from alignment" (reference numbers + fresh ids), so we
        # order by the label value rather than assuming label == row + 1.
        ordered_ds = [d for d, _ in sorted(d_to_label.items(),
                                           key=lambda kv: kv[1])]

        # Rebuild rows only when district count changes
        if self._dist_table_built_rows != n_dist:
            dpg.delete_item("district_info_table", children_only=True, slot=1)
            for r in range(n_dist):
                with dpg.table_row(parent="district_info_table"):
                    for _, _, key in self._DIST_COLS:
                        dpg.add_text("", tag=f"di_r{r}_{key}")
            self._dist_table_built_rows = n_dist

        for r in range(n_dist):
            d = ordered_ds[r] if r < len(ordered_ds) else r
            lab = d_to_label.get(d, d + 1)
            dpg.set_value(f"di_r{r}_dist", str(lab))
            dpg.set_value(f"di_r{r}_pop", f"{int(pop_d[d]):,}")
            dpg.set_value(f"di_r{r}_pdev", f"{pop_dev_pct[d]:+.2f}%")
            dpg.set_value(
                f"di_r{r}_pp",
                f"{pp_per_dist[d] * 100:.1f}" if pp_per_dist is not None else "—",
            )
            if dem_pct is not None and not np.isnan(dem_pct[d]):
                dp = float(dem_pct[d])
                rp = float(rep_pct[d])
                dpg.set_value(f"di_r{r}_demp", f"{dp:.1f}%")
                dpg.set_value(f"di_r{r}_repp", f"{rp:.1f}%")
                m = dp - rp
                dpg.set_value(f"di_r{r}_margin", f"{m:+.1f}")
            else:
                dpg.set_value(f"di_r{r}_demp", "—")
                dpg.set_value(f"di_r{r}_repp", "—")
                dpg.set_value(f"di_r{r}_margin", "—")

        self._dist_info_last_iter = current_iter

    def _update_ui(self):
        snap = self.state.get_all()
        status  = snap["status"]
        is_running      = status == AlgorithmStatus.RUNNING
        is_paused       = status == AlgorithmStatus.PAUSED
        is_partitioning = status == AlgorithmStatus.PARTITIONING
        is_busy    = status in (
            AlgorithmStatus.RUNNING,
            AlgorithmStatus.PARTITIONING,
            AlgorithmStatus.PAUSED,
        )

        # ── Auto-renumber on run completion (Advanced checkbox) ──────────────────
        # Fire once on the running -> COMPLETED transition.
        if (status == AlgorithmStatus.COMPLETED
                and getattr(self, "_last_seen_status", None)
                    != AlgorithmStatus.COMPLETED
                and self._renumber_enabled):
            self._apply_renumber(self._renumber_rule)
        self._last_seen_status = status

        # ── Inspection complete → show shapefile dialog (or skip for recent open) ─
        if snap["shp_inspect_ready"]:
            self.state.update(shp_inspect_ready=False)
            if self.runner and self.runner._pending_inspection is not None:
                if self._pending_recent_config is not None:
                    _cfg = self._pending_recent_config
                    self._pending_recent_config = None
                    self._on_shp_confirm(self.runner._pending_inspection, _cfg)
                else:
                    self._shp_dialog.populate(self.runner._pending_inspection)

        # ── Load complete → update shapefile info label ────────────────────────
        if snap["gdf_ready"]:
            self.state.update(gdf_ready=False)
            self._update_shp_info_label()
            # Mark this gdf as fully populated (all overlay arrays present) so
            # the map bg-load below may load it. complete_load sets gdf +
            # populations + county + elections + pp before pulsing gdf_ready,
            # so latching here guarantees the map never captures partial data.
            self._map_data_gdf_id = (id(self.runner.gdf)
                                     if self.runner and self.runner.gdf is not None
                                     else 0)

        # ── Map: trigger background load when new shapefile is ready ──────────
        loaded_path = self.state.shapefile_path
        loaded_gdf_id = (id(self.runner.gdf)
                         if self.runner and self.runner.gdf is not None
                         else 0)
        # Gate on the fully-loaded latch, not a status snapshot: on Open Recent
        # complete_load is kicked off in this same frame, so a stale IDLE
        # status (left by inspection) could otherwise let the map load a
        # runner that has only gdf+populations set — county/elections/pp still
        # None — and the same-identity gdf never re-triggers a reload.
        fully_loaded = (loaded_gdf_id != 0
                        and loaded_gdf_id == self._map_data_gdf_id)
        if (loaded_path
                and (loaded_path != self._map_loaded_path
                     or loaded_gdf_id != self._map_loaded_gdf_id)
                and not self._map_loading
                and fully_loaded
                and self.runner
                and self.runner.gdf is not None):
            self._map_loading = True
            self._map_loaded_path = loaded_path
            self._map_loaded_gdf_id = loaded_gdf_id
            gdf_ref = self.runner.gdf
            county_array_ref = self.runner.county_array
            dem_ref = (self.runner.election_arrays[0][0]
                       if self.runner.election_arrays else None)
            gop_ref = (self.runner.election_arrays[0][1]
                       if self.runner.election_arrays else None)
            pp_data_ref  = self.runner.pp_data
            reock_data_ref = self.runner.reock_data
            pop_ref      = self.runner.populations
            mv = self.map_view
            def _bg_load():
                try:
                    mv.load(gdf_ref, county_array=county_array_ref,
                            dem_votes=dem_ref, gop_votes=gop_ref,
                            pp_data=pp_data_ref, reock_data=reock_data_ref,
                            populations=pop_ref)
                    self._map_ready = True
                except Exception as exc:
                    import sys as _sys

                    from mosaic.crash import write_crash_log
                    crash_path = write_crash_log(
                        exc, context={"phase": "map_view_load"}
                    )
                    print(
                        f"\n[mosaic] Map rendering failed.\n"
                        f"        {type(exc).__name__}: {exc}\n"
                        f"        Log: {crash_path}\n",
                        file=_sys.stderr,
                    )
                    self.state.update(
                        error_message=(
                            f"Map render failed: {type(exc).__name__}: {exc}\n"
                            f"Log: {crash_path}"
                        ),
                        status_message="Map render failed — see crash log",
                    )
                finally:
                    self._map_loading = False
            threading.Thread(target=_bg_load, daemon=True).start()

        if self._map_ready:
            self._map_ready = False
            if self.map_view is not None:
                self.map_view.draw_blank()

        # ── Map: colour update triggered by runner ────────────────────────────
        with self.state._lock:
            map_needs = self.state.map_needs_update
            if map_needs:
                self.state.map_needs_update = False
                _assignment = (
                    self.state.current_assignment.copy()
                    if self.state.current_assignment is not None else None
                )
                _initial = (
                    self.state.initial_assignment.copy()
                    if self.state.initial_assignment is not None else None
                )
                _n_dist = self.state.num_districts
                _label_map = self.state.district_label_map
            else:
                _assignment = _initial = None
                _n_dist = 0
                _label_map = self.state.district_label_map

        # Label-mode tracking: cheap centroid while running, precise
        # pole-of-inaccessibility when idle.  On the transition running ->
        # idle, force one re-render so the precise labels actually appear
        # (otherwise the last frame's approximate labels persist).
        labels_should_be_fast = status in (
            AlgorithmStatus.RUNNING, AlgorithmStatus.PARTITIONING,
        )
        if (self._labels_were_fast and not labels_should_be_fast
                and self.map_view is not None
                and self.map_view.show_labels):
            map_needs = True
        self._labels_were_fast = labels_should_be_fast

        if self.map_view is not None:
            self.map_view.district_label_map = _label_map
        if (map_needs
                and _assignment is not None
                and self.map_view is not None):
            self.map_view.fast_labels = labels_should_be_fast
            self.map_view.render_assignment(_assignment, _n_dist, _initial)

        # ── Status / iteration ────────────────────────────────────────────────
        msg = snap["status_message"]
        if status == AlgorithmStatus.ERROR:
            err = self.state.error_message or msg
            dpg.set_value(
                self._status_txt,
                "ERROR — " + err if err else "ERROR",
            )
            self.theme.retoken(self._status_txt, "error")
        else:
            dpg.set_value(
                self._status_txt,
                "Status: " + status.value + (" -- " + msg if msg else ""),
            )
            self.theme.retoken(self._status_txt, "body")
        cur    = snap["current_iteration"]
        max_it = snap["max_iterations"]
        dpg.set_value(self._iter_txt, f"Iteration: {cur:,} / {max_it:,}")
        if max_it > 0:
            dpg.set_value(self._progress, cur / max_it)

        # Timer / IPS
        st = self.state.start_time
        et = self.state.end_time
        pt = self.state.pause_time
        if st > 0 and cur > 0:
            if is_running:
                elapsed = time.time() - st
            elif is_paused and pt > 0:
                elapsed = pt - st          # frozen at the moment pause began
            else:
                elapsed = et - st if et > 0 else time.time() - st
            ips = cur / elapsed if elapsed > 0 else 0
            dpg.set_value(self._time_txt, f"Time: {_fmt_dur(elapsed)}")
            dpg.set_value(self._ips_txt, f"Iter/sec: {ips:.1f}")
            if is_running and ips > 0 and cur < max_it:
                dpg.set_value(self._eta_txt,
                              f"Est. left: {_fmt_dur((max_it - cur) / ips)}")
            else:
                dpg.set_value(self._eta_txt, "Est. left: --")

        # Score
        score   = snap["current_score"]
        best    = snap["best_score"]
        best_it = snap["best_iteration"]
        if score < float("inf"):
            dpg.set_value(self._score_txt, f"Score: {score:.2f}")
        if best < float("inf"):
            dpg.set_value(self._best_txt,
                          f"Best:  {best:.2f}   (iter. {best_it:,})")

        # Temperature & acceptance
        temp = snap["current_temperature"]
        dpg.set_value(
            self._temp_txt,
            f"Temperature: {temp:.4f}" if temp > 0
            else "Temperature: -- (annealing off)",
        )
        acc = snap["accepted_worse"]
        rej = snap["rejected_worse"]
        total_worse = acc + rej
        if total_worse > 0:
            rate = 100.0 * acc / total_worse
            dpg.set_value(self._acc_txt,
                          f"Entropy: {rate:.1f}%  ({acc:,} / {total_worse:,})")
        else:
            dpg.set_value(self._acc_txt, "Entropy: --")

        dpg.set_value(self._succ_txt,
                      f"Accepted steps: {snap['successful_steps']:,}")
        dpg.set_value(self._flip_txt,
                      f"Flip rate: {snap['current_flip_rate'] * 100.0:.1f}%")

        # ── Plots & side panels ──────────────────────────────────────────────
        self._update_plots_and_panels()

        # Keep district-count-dependent sliders bounded
        n_dist_val = dpg.get_value(self._num_districts)
        dpg.configure_item(self._hinge_threshold,  max_value=n_dist_val)

        # ── Button states ─────────────────────────────────────────────────────
        # Scores menu stays usable during a run (changes apply to the next run;
        # the running config is frozen at start). Hiding a score via that menu
        # force-disables it (see _set_score_row_vis), so a hidden score can never
        # keep silently affecting annealing.
        self._sync_seed_controls()
        dpg.configure_item(self._run_btn,    enabled=not is_busy)
        dpg.configure_item(self._pause_btn,  enabled=is_running or is_paused or is_partitioning)
        dpg.configure_item(self._pause_btn,
                           label="Resume" if is_paused else "Pause")
        dpg.configure_item(self._reset_btn,  enabled=True)
        has_result = self.state.best_assignment is not None
        # Allow saves while paused; block only during active running/partitioning.
        can_save = has_result and not is_running and not is_partitioning
        dpg.configure_item(self._export_btn,  enabled=can_save)
        dpg.configure_item(self._metrics_btn, enabled=can_save)
        if self._file_save_asgn_item:
            dpg.configure_item(self._file_save_asgn_item,  enabled=can_save)
            dpg.configure_item(self._file_save_metrics_item, enabled=can_save)
        can_revert = has_result and status in (
            AlgorithmStatus.IDLE, AlgorithmStatus.PAUSED,
            AlgorithmStatus.COMPLETED, AlgorithmStatus.ERROR,
        )
        if can_revert:
            # Nothing to revert to when the plan on the map already IS the best
            # one -- e.g. just after Revert to Best, or the final iteration was
            # itself the best. Grey the button out in that case.
            with self.state._lock:
                cur  = self.state.current_assignment
                best = self.state.best_assignment
            if cur is not None and best is not None and np.array_equal(cur, best):
                can_revert = False
        dpg.configure_item(self._revert_btn, enabled=can_revert)

        # ── Button nudge / anti-nudge themes ─────────────────────────────────
        graph_ready = self.runner is not None and self.runner.graph is not None
        dpg.bind_item_theme(
            self._run_btn,
            self.theme.nudge_theme if (not is_busy and graph_ready and not has_result)
            else self.theme.antinudge_theme,
        )
        dpg.bind_item_theme(
            self._pause_btn,
            self.theme.nudge_theme if (is_running or is_paused)
            else self.theme.antinudge_theme,
        )
        # Once a map exists, Reset is a live "start over" action -- give it the
        # neutral light-grey look (theme 0 = the app's default button style, with
        # bright body text) so it reads as available, apart from the dark
        # inactive buttons. Before any result there's nothing to reset: dark grey.
        dpg.bind_item_theme(
            self._reset_btn,
            0 if has_result else self.theme.antinudge_theme,
        )
        dpg.bind_item_theme(
            self._export_btn,
            self.theme.nudge_theme if (has_result and not is_busy)
            else self.theme.antinudge_theme,
        )
        dpg.bind_item_theme(
            self._metrics_btn,
            self.theme.nudge_theme if (has_result and not is_busy)
            else self.theme.antinudge_theme,
        )
        # Blue while it would do something; otherwise anti-nudge, which now
        # renders as the dark disabled grey in its disabled state.
        dpg.bind_item_theme(
            self._revert_btn,
            self.theme.nudge_theme if can_revert else self.theme.antinudge_theme,
        )

    def _update_plots_and_panels(self) -> None:
        """Per-frame plot redraws and side panel refreshes."""
        # A read cursor past the end of a reset history yields an empty delta
        # forever, freezing the charts; rebuild so they recover on their own.
        with self.state._lock:
            _desynced = self._buf_score.read > len(self.state.score_history)
        if _desynced:
            self._clear_all_series()
        # ── Plots ─────────────────────────────────────────────────────────────
        # One lock acquisition, copying only the delta since the last call.
        with self.state._lock:
            _sd   = list(self.state.score_history[self._buf_score.read:])
            _ad   = list(self.state.acceptance_rate_history[self._buf_acc.read:])
            _td   = list(self.state.temperature_history[self._buf_temp.read:])
            _csd  = list(self.state.county_splits_score_history[self._buf_cs_score.read:])
            _ced  = list(self.state.county_excess_splits_history[self._buf_cs_excess.read:])
            _cld  = list(self.state.county_unified_districts_history[self._buf_cs_clean.read:])
            _md   = list(self.state.mm_history[self._buf_mm.read:])
            _ed   = list(self.state.eg_history[self._buf_eg.read:])
            _pbd  = list(self.state.partisan_bias_history[self._buf_pb.read:])
            _pgd  = list(self.state.partisan_gini_history[self._buf_pg.read:])
            _sed  = list(self.state.dem_seats_history[self._buf_seats.read:])
            _pd   = list(self.state.pp_history[self._buf_pp.read:])
            _rd   = list(self.state.reock_history[self._buf_reock.read:])
            _hcd  = list(self.state.holistic_compactness_history[self._buf_hc.read:])
            _hsd  = list(self.state.holistic_splitting_history[self._buf_hsplit.read:])
            _hpd  = list(self.state.holistic_proportionality_history[self._buf_hprop.read:])
            _hmd  = list(self.state.holistic_competitiveness_history[self._buf_hcmp.read:])
            _pvd  = list(self.state.pop_deviation_history[self._buf_popdev.read:])
            _pvd_max  = list(self.state.pop_dev_max_history[self._buf_popdev_max.read:])
            _pvd_mean = list(self.state.pop_dev_mean_history[self._buf_popdev_mean.read:])
            _almd = list(self.state.alignment_mean_ret_history[self._buf_align_mean.read:])
            _alnd = list(self.state.alignment_min_ret_history[self._buf_align_min.read:])
            _cutd = list(self.state.cut_edges_history[self._buf_cuts.read:])
            _mjd  = list(self.state.majority_dem_history[self._buf_maj_dem.read:])
            _mjr  = list(self.state.majority_rep_history[self._buf_maj_rep.read:])
            _hgd  = list(self.state.hinge_history[self._buf_hinge.read:])
            _ivd  = list(self.state.inversion_history[self._buf_inversion.read:])

        self._buf_score.add(_sd)
        self._buf_acc.add_pairs(_ad, scale=100.0)
        self._buf_temp.add(_td)
        self._buf_cs_score.add(_csd)
        self._buf_cs_excess.add(_ced)
        self._buf_cs_clean.add(_cld)
        self._buf_mm.add(_md)
        self._buf_eg.add(_ed)
        self._buf_pb.add(_pbd)
        self._buf_pg.add(_pgd)
        self._buf_seats.add(_sed)
        self._buf_pp.add([v * 100.0 for v in _pd])
        self._buf_reock.add([v * 100.0 for v in _rd])
        # Holistic charts: convert penalty (0=best) to rating (100=best) for display.
        # Splitting is the exception -- its penalty is uncapped (unclipped mode),
        # so it is charted as the raw penalty (0 = best) to stay honest above 100.
        self._buf_hc.add([100.0 - v for v in _hcd])
        self._buf_hsplit.add(_hsd)
        self._buf_hprop.add([100.0 - v for v in _hpd])
        self._buf_hcmp.add([100.0 - v for v in _hmd])
        self._buf_popdev.add(_pvd)
        self._buf_popdev_max.add(_pvd_max)
        self._buf_popdev_mean.add(_pvd_mean)
        self._buf_align_mean.add(_almd)
        self._buf_align_min.add(_alnd)
        self._buf_cuts.add(_cutd)
        self._buf_maj_dem.add([v * 100.0 for v in _mjd])
        self._buf_maj_rep.add([v * 100.0 for v in _mjr])
        self._buf_hinge.add([v * 100.0 for v in _hgd])
        self._buf_inversion.add([v * 100.0 for v in _ivd])

        limit = dpg.get_value(self._limit_plots) if self._limit_plots else False

        def _render(buf: _SeriesBuffer, series_tag: str, x_tag: str, y_tag: str,
                    fit_y: bool = True) -> None:
            if not buf.ys:
                return
            xs, ys = buf.plot_data(limit)
            # Render-time uniform subsample.  Buffer is untouched; we just
            # don't ship more than ~_RENDER_TARGET points to DPG per frame.
            # Three rules:
            #   1) Only subsample when "Limit plots" is ON.  When OFF, the
            #      user has explicitly asked for the full series and accepts
            #      the perf cost.
            #   2) Pin the stride to ABSOLUTE buffer position (not the
            #      window-relative start), so the set of sampled iterations
            #      is stable across frames as the window slides.  Without
            #      this, the strided sample rotates through `step` phases
            #      and the line flickers vertically each frame.
            #   3) Always include the indices of the windowed min, max, and
            #      last point so fit_axis_data has stable bounds and the
            #      leading edge tracks current data.
            n = len(xs)
            if limit and n > _RENDER_TARGET:
                step = n // _RENDER_TARGET
                # Where this window starts in the full buffer.  Used to
                # offset the stride so absolute positions divisible by
                # `step` remain sampled even as the window slides.
                window_start = max(0, len(buf.ys) - _PLOT_LIMIT)
                offset = (step - window_start % step) % step
                y_min_idx = ys.index(min(ys))
                y_max_idx = ys.index(max(ys))
                idx = sorted(set(range(offset, n, step))
                             | {y_min_idx, y_max_idx, n - 1})
                xs = [xs[i] for i in idx]
                ys = [ys[i] for i in idx]
            dpg.set_value(series_tag, [xs, ys])
            dpg.fit_axis_data(x_tag)
            # Skip y-fit for axes locked to a fixed range (e.g. probability
            # axes [0, 100]) — otherwise fit_axis_data overrides the lock
            # and the line vanishes when data sits on the boundary.
            if fit_y:
                dpg.fit_axis_data(y_tag)

        _render(self._buf_score, "score_series", "score_x", "score_y")
        _render(self._buf_acc,   "acc_series",   "acc_x",   "acc_y")

        if dpg.is_item_shown("panel_temperature"):
            _render(self._buf_temp, "panel_temp_series", "panel_temp_x", "panel_temp_y")

        cs_on = dpg.get_value(self._cs_enabled)
        if dpg.is_item_shown("panel_county_splits"):
            dpg.configure_item("cs_charts_grp",   show=cs_on)
            dpg.configure_item("cs_inactive_lbl", show=not cs_on)
            if cs_on:
                _render(self._buf_cs_excess, "cs_excess_series", "cs_excess_x", "cs_excess_y")
                _render(self._buf_cs_clean,  "cs_clean_series",  "cs_clean_x",  "cs_clean_y")
                if self._buf_cs_excess.ys:
                    _, w = self._buf_cs_excess.plot_data(limit)
                    hi = max(1, int(max(w)))
                    dpg.set_axis_limits("cs_excess_y", 0, hi + 1)
            max_clean = None
            if (self.runner is not None and self.runner.county_pops is not None
                    and self.runner.populations is not None):
                n_dist = dpg.get_value(self._num_districts)
                tol = dpg.get_value(self._tolerance) / 100.0
                ideal_pop = float(self.runner.populations.sum()) / n_dist if n_dist > 0 else 1.0
                min_dp = max(ideal_pop * (1.0 - tol), 1.0)
                max_clean = int(np.floor(self.runner.county_pops / min_dp).sum())
                dpg.set_value("cs_max_clean_note",
                              f"Maximum feasible single-county districts: {max_clean:,}")
            if cs_on and self._buf_cs_clean.ys:
                xs, w = self._buf_cs_clean.plot_data(limit)
                hi = int(max(w))
                if max_clean is not None:
                    hi = max(hi, max_clean)
                dpg.set_axis_limits("cs_clean_y", 0, hi + 1)
                # Horizontal reference line spanning the current x-range.
                if max_clean is not None and xs:
                    dpg.set_value("cs_clean_max",
                                  [[xs[0], xs[-1]], [max_clean, max_clean]])

        pp_on    = dpg.get_value(self._pp_enabled)

        if dpg.is_item_shown("panel_mm"):
            dpg.configure_item("mm_plot_grp",     show=self._has_elections)
            dpg.configure_item("mm_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_mm, "mm_series", "mm_x", "mm_y")
        if dpg.is_item_shown("panel_eg"):
            dpg.configure_item("eg_plot_grp",     show=self._has_elections)
            dpg.configure_item("eg_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_eg, "eg_series", "eg_x", "eg_y")
        if dpg.is_item_shown("panel_pb"):
            dpg.configure_item("pb_plot_grp",     show=self._has_elections)
            dpg.configure_item("pb_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_pb, "pb_series", "pb_x", "pb_y")
        if dpg.is_item_shown("panel_pg"):
            dpg.configure_item("pg_plot_grp",     show=self._has_elections)
            dpg.configure_item("pg_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_pg, "pg_series", "pg_x", "pg_y", fit_y=False)
        if dpg.is_item_shown("panel_dem_seats"):
            dpg.configure_item("seats_plot_grp",     show=self._has_elections)
            dpg.configure_item("seats_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_seats, "seats_series", "seats_x", "seats_y")
        if dpg.is_item_shown("panel_majority"):
            dpg.configure_item("majority_plot_grp",     show=self._has_elections)
            dpg.configure_item("majority_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_maj_dem, "maj_dem_series", "maj_x", "maj_y", fit_y=False)
                _render(self._buf_maj_rep, "maj_rep_series", "maj_x", "maj_y", fit_y=False)
        if dpg.is_item_shown("panel_hinge"):
            # Hinge needs election data AND an established threshold (the Hinge
            # score enabled); otherwise it charts against a stale default.
            hinge_active = self._has_elections and dpg.get_value(self._hinge_enabled)
            dpg.configure_item("hinge_plot_grp",     show=hinge_active)
            dpg.configure_item("hinge_inactive_lbl", show=not hinge_active)
            if not hinge_active:
                dpg.set_value(
                    "hinge_inactive_lbl",
                    "Load election data to use this panel."
                    if not self._has_elections
                    else "Enable the Hinge score to set a threshold for this panel.")
            if hinge_active:
                _render(self._buf_hinge, "hinge_series", "hinge_x", "hinge_y", fit_y=False)
        if dpg.is_item_shown("panel_pp"):
            dpg.configure_item("pp_plot_grp",     show=pp_on)
            dpg.configure_item("pp_inactive_lbl", show=not pp_on)
            if pp_on:
                _render(self._buf_pp, "pp_series", "pp_x", "pp_y")
        if dpg.is_item_shown("panel_reock"):
            reock_on = dpg.get_value(self._reock_enabled)
            dpg.configure_item("reock_plot_grp",     show=reock_on)
            dpg.configure_item("reock_inactive_lbl", show=not reock_on)
            if reock_on:
                _render(self._buf_reock, "reock_series", "reock_x", "reock_y")
        if dpg.is_item_shown("panel_hc"):
            hc_on = dpg.get_value(self._hc_enabled)
            dpg.configure_item("hc_plot_grp",     show=hc_on)
            dpg.configure_item("hc_inactive_lbl", show=not hc_on)
            if hc_on:
                _render(self._buf_hc, "hc_series", "hc_x", "hc_y")
        if dpg.is_item_shown("panel_hsplit"):
            hs_on = dpg.get_value(self._hsplit_enabled)
            dpg.configure_item("hsplit_plot_grp",     show=hs_on)
            dpg.configure_item("hsplit_inactive_lbl", show=not hs_on)
            if hs_on:
                # Floor the y-axis at 0 (penalty >= 0); auto-fit only the top.
                _render(self._buf_hsplit, "hsplit_series", "hsplit_x", "hsplit_y",
                        fit_y=False)
                if self._buf_hsplit.ys:
                    hi = max(self._buf_hsplit.ys)
                    dpg.set_axis_limits("hsplit_y", 0.0, hi * 1.05 + 1e-6)
        if dpg.is_item_shown("panel_hprop"):
            dpg.configure_item("hprop_plot_grp",     show=self._has_elections)
            dpg.configure_item("hprop_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_hprop, "hprop_series", "hprop_x", "hprop_y")
                _render(self._buf_inversion, "hprop_inversion_series", "hprop_x", "hprop_y")
        if dpg.is_item_shown("panel_hcmp"):
            dpg.configure_item("hcmp_plot_grp",     show=self._has_elections)
            dpg.configure_item("hcmp_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_hcmp, "hcmp_series", "hcmp_x", "hcmp_y")
        if dpg.is_item_shown("panel_popdev"):
            # The ratchet records deviation each iteration (force_pop_components),
            # so the chart has real data even when the score weight is 0.
            popdev_on = (dpg.get_value(self._popdev_enabled)
                         or dpg.get_value(self._tolerance_ratchet_mode) != "Off")
            dpg.configure_item("popdev_plot_grp",     show=popdev_on)
            dpg.configure_item("popdev_inactive_lbl", show=not popdev_on)
            if popdev_on:
                _render(self._buf_popdev_max,  "popdev_max_series",  "popdev_x", "popdev_y")
                _render(self._buf_popdev_mean, "popdev_mean_series", "popdev_x", "popdev_y")
                # Fit x-axis to data; set y-axis with floor at 0 and reasonable headroom.
                dpg.fit_axis_data("popdev_x")
                y_max = max(self._buf_popdev_max.ys) if self._buf_popdev_max.ys else 1.0
                dpg.set_axis_limits("popdev_y", 0.0, max(y_max * 1.15, 2.0))
        if dpg.is_item_shown("panel_alignment"):
            align_on = (dpg.get_value(self._alignment_enabled)
                        and self.runner is not None
                        and self.runner.alignment_data is not None)
            dpg.configure_item("alignment_plot_grp",     show=align_on)
            dpg.configure_item("alignment_inactive_lbl", show=not align_on)
            if align_on:
                _render(self._buf_align_mean, "alignment_mean_series",
                        "alignment_x", "alignment_y")
                _render(self._buf_align_min, "alignment_min_series",
                        "alignment_x", "alignment_y")
                dpg.set_axis_limits("alignment_y", 0.0, 100.0)
        if dpg.is_item_shown("panel_cut_edges"):
            _render(self._buf_cuts, "cuts_series", "cuts_x", "cuts_y")

        # Score Contributors panel — live bar chart, only active metrics shown
        if dpg.is_item_shown("panel_score_contrib"):
            with self.state._lock:
                bd = dict(self.state.score_breakdown)
            active = [(i, name, short)
                      for i, (name, short, _) in enumerate(_CONTRIB_BAR_METRICS)
                      if bd.get(name, 0.0) > 0.0]
            if active:
                dpg.set_axis_ticks(
                    "contrib_x",
                    tuple((short, float(pos + 1))
                          for pos, (_, _, short) in enumerate(active)),
                )
                dpg.set_axis_limits("contrib_x", 0.5, len(active) + 0.5)
            active_idx = {i for i, _, _ in active}
            for pos, (orig_i, name, _) in enumerate(active):
                dpg.set_value(self._contrib_bar_series[orig_i],
                              [[float(pos + 1)], [bd[name]]])
                dpg.configure_item(self._contrib_bar_series[orig_i], show=True)
            for i in range(len(_CONTRIB_BAR_METRICS)):
                if i not in active_idx:
                    dpg.configure_item(self._contrib_bar_series[i], show=False)

        # Partisanship panel — live bar chart from current assignment
        if dpg.is_item_shown("panel_partisanship"):
            from mosaic.gui.map_view import _PARTISAN_BREAKS
            with self.state._lock:
                _pa = (self.state.current_assignment.copy()
                       if self.state.current_assignment is not None else None)
                _pnd = self.state.num_districts
            if (_pa is not None and self.runner is not None
                    and self.runner.election_arrays
                    and len(self.runner.election_arrays[0][0]) == len(_pa)):
                _dem, _gop = self.runner.election_arrays[0]
                _dem_d = np.bincount(_pa, weights=_dem.astype(np.float64), minlength=_pnd)
                _gop_d = np.bincount(_pa, weights=_gop.astype(np.float64), minlength=_pnd)
                _tot_d = _dem_d + _gop_d
                _shares = np.where(_tot_d > 0, _dem_d / _tot_d, 0.5)
                sorted_shares = np.sort(_shares)
                ranks = np.arange(1, len(sorted_shares) + 1, dtype=float)
                bucket_idx = np.searchsorted(_PARTISAN_BREAKS, sorted_shares, side="right") - 1
                bucket_idx = np.clip(bucket_idx, 0, 11)
                for bi in range(12):
                    mask = bucket_idx == bi
                    dpg.set_value(
                        self._partisan_bar_series[bi],
                        [ranks[mask].tolist(), (sorted_shares[mask] * 100.0).tolist()],
                    )
                n = len(sorted_shares)
                dpg.set_value("partisan_ref", [[0.5, n + 0.5], [50.0, 50.0]])
                median_x = (n + 1) / 2.0
                dpg.set_value("partisan_median", [[median_x, median_x], [0.0, 100.0]])
                dpg.fit_axis_data("partisan_x")
                dpg.set_axis_limits("partisan_y", 0.0, 100.0)

        # Win Chance panel — same layout, Y = P(D wins) via unified Gaussian model
        if dpg.is_item_shown("panel_win_chance"):
            import math

            from mosaic.gui.map_view import _PARTISAN_BREAKS
            from mosaic.scoring.partisan import k_to_sigma, p_win_gaussian
            with self.state._lock:
                _wa = (self.state.current_assignment.copy()
                       if self.state.current_assignment is not None else None)
                _wnd = self.state.num_districts
            if (_wa is not None and self.runner is not None
                    and self.runner.election_arrays
                    and len(self.runner.election_arrays[0][0]) == len(_wa)):
                _dem, _gop = self.runner.election_arrays[0]
                _dem_d = np.bincount(_wa, weights=_dem.astype(np.float64), minlength=_wnd)
                _gop_d = np.bincount(_wa, weights=_gop.astype(np.float64), minlength=_wnd)
                _tot_d = _dem_d + _gop_d
                _shares = np.where(_tot_d > 0, _dem_d / _tot_d, 0.5)
                _sigma_d    = k_to_sigma(dpg.get_value(self._win_prob))
                _sigma_comb = math.sqrt(dpg.get_value(self._swing_sigma) ** 2 + _sigma_d ** 2)
                _p_win = p_win_gaussian(_shares, _sigma_comb)
                sorted_p = np.sort(_p_win)
                wranks = np.arange(1, len(sorted_p) + 1, dtype=float)
                wbucket = np.searchsorted(_PARTISAN_BREAKS, sorted_p, side="right") - 1
                wbucket = np.clip(wbucket, 0, 11)
                for bi in range(12):
                    mask = wbucket == bi
                    dpg.set_value(
                        self._win_chance_bar_series[bi],
                        [wranks[mask].tolist(), (sorted_p[mask] * 100.0).tolist()],
                    )
                wn = len(sorted_p)
                dpg.set_value("win_chance_ref", [[0.5, wn + 0.5], [50.0, 50.0]])
                wmedian_x = (wn + 1) / 2.0
                dpg.set_value("win_chance_median", [[wmedian_x, wmedian_x], [0.0, 100.0]])
                dpg.fit_axis_data("win_chance_x")
                dpg.set_axis_limits("win_chance_y", 0.0, 100.0)

        # District Info table — per-district metrics for the current plan
        self._update_district_info_table()

        # Phase plot (metric-vs-metric trajectory); no-op unless its panel is open
        self._update_phase_plot()

        # ── Keyboard shortcuts (Ctrl+N/O/S/W) ─────────────────────────────────
        _ctrl = (dpg.is_key_down(dpg.mvKey_LControl)
                 or dpg.is_key_down(dpg.mvKey_RControl))
        if _ctrl:
            if dpg.is_key_pressed(dpg.mvKey_N):
                self._on_new()
            elif dpg.is_key_pressed(dpg.mvKey_O):
                self._on_import_shapefile()
            elif dpg.is_key_pressed(dpg.mvKey_S):
                # Gate the Save shortcut like the Save button: a result exists
                # and no run is active (see _update_ui's can_save).
                _status = self.state.get_all()["status"]
                if (self.state.best_assignment is not None
                        and _status != AlgorithmStatus.RUNNING
                        and _status != AlgorithmStatus.PARTITIONING):
                    self._on_file_save_assignments()
            elif dpg.is_key_pressed(dpg.mvKey_W):
                self._on_close()

    def _clear_all_series(self) -> None:
        """Clear local history buffers and blank all DPG plot series."""
        for buf in (
            self._buf_score, self._buf_acc, self._buf_temp,
            self._buf_cs_score, self._buf_cs_excess, self._buf_cs_clean,
            self._buf_mm, self._buf_eg, self._buf_seats,
            self._buf_pb, self._buf_pg,
            self._buf_pp, self._buf_reock, self._buf_hc,
            self._buf_hsplit, self._buf_hprop, self._buf_hcmp,
            self._buf_popdev,
            self._buf_popdev_max, self._buf_popdev_mean, self._buf_cuts,
            self._buf_align_mean, self._buf_align_min,
            self._buf_maj_dem, self._buf_maj_rep, self._buf_hinge,
            self._buf_inversion,
        ):
            buf.clear()
        empty = [[], []]
        for tag in (
            "score_series", "acc_series", "panel_temp_series",
            "cs_excess_series", "cs_clean_series",
            "mm_series", "eg_series", "seats_series",
            "pb_series", "pg_series",
            "pp_series", "reock_series",
            "popdev_max_series", "popdev_mean_series", "cuts_series",
            "alignment_mean_series", "alignment_min_series",
            "maj_dem_series", "maj_rep_series", "hinge_series",
        ):
            dpg.set_value(tag, empty)
        for ax in (
            "score_x", "score_y", "acc_x", "acc_y",
            "panel_temp_x", "panel_temp_y",
            "cs_excess_x", "cs_excess_y",
            "cs_clean_x", "cs_clean_y",
            "mm_x", "mm_y", "eg_x", "eg_y",
            "pb_x", "pb_y", "pg_x", "pg_y",
            "seats_x", "seats_y",
            "pp_x", "pp_y", "popdev_x", "popdev_y", "cuts_x", "cuts_y",
            "alignment_x", "alignment_y",
            "maj_x", "maj_y", "hinge_x", "hinge_y",
        ):
            dpg.set_axis_limits_auto(ax)
        # Re-pin axes whose ranges should always be locked (probability bands).
        # Without this, set_axis_limits_auto above releases them and the next
        # render auto-fits to whatever the data happens to be, causing the
        # axis to "stick" at a small range like [0, 0.1].
        dpg.set_axis_limits("pp_y",     0.0, 100.0)
        dpg.set_axis_limits("maj_y",    0.0, 100.0)
        dpg.set_axis_limits("hinge_y",  0.0, 100.0)
        dpg.set_axis_limits("popdev_y", 0.0, 5.0)
        dpg.set_axis_limits("alignment_y", 0.0, 100.0)
        dpg.set_axis_limits("pg_y", 0.0, 100.0)
        for s in self._partisan_bar_series:
            dpg.set_value(s, empty)
        for s in self._win_chance_bar_series:
            dpg.set_value(s, empty)
        for i, s in enumerate(self._contrib_bar_series):
            dpg.set_value(s, [[float(i + 1)], [0.0]])

    # ── Shapefile info label ──────────────────────────────────────────────────

    def _update_shp_info_label(self) -> None:
        """Refresh the shapefile status line after a successful load."""
        if self.runner is None:
            return
        # Prefer inspection path; fall back to state for the display label.
        insp = self.runner._pending_inspection
        path = (getattr(insp, "path", None) or self.state.shapefile_path or "")
        stem = Path(path).stem if path else "shapefile"
        dpg.set_value(self._shp_info, f"Loaded: {stem}")
        self.theme.retoken(self._shp_info, "success_pale")

        # Read actual loaded state from runner — avoids races on _loaded_config.
        has_county = self.runner.county_array is not None
        self.theme.retoken(self._cs_lbl,
                           "secondary" if has_county else "disabled_deep")
        dpg.configure_item(self._cs_enabled, enabled=has_county)
        if not has_county:
            dpg.set_value(self._cs_enabled, False)
            dpg.configure_item("cs_controls", show=False)
        dpg.configure_item(self._county_overlay, enabled=has_county)
        dpg.configure_item(self._splits_view, enabled=has_county)
        if not has_county and dpg.get_value(self._county_overlay):
            dpg.set_value(self._county_overlay, False)
        if not has_county and dpg.get_value(self._splits_view):
            dpg.set_value(self._splits_view, False)
            if self.map_view:
                self.map_view.splits_view = False
        # Enable/disable county splits panel menu item; close if now unavailable
        dpg.configure_item(self._panel_cs_item, enabled=has_county)
        if not has_county and dpg.is_item_shown("panel_county_splits"):
            dpg.set_value(self._panel_cs_item, False)
            dpg.configure_item("panel_county_splits", show=False)

        # Partisan overlay map toggle
        has_elections = bool(self.runner.election_arrays)
        self._has_elections = has_elections
        dpg.configure_item(self._partisan_overlay, enabled=has_elections)
        dpg.configure_item(self._district_partisan, enabled=has_elections)
        if not has_elections:
            if dpg.get_value(self._partisan_overlay):
                dpg.set_value(self._partisan_overlay, False)
                if self.map_view:
                    self.map_view.partisan_overlay = False
            if dpg.get_value(self._district_partisan):
                dpg.set_value(self._district_partisan, False)
                if self.map_view:
                    self.map_view.district_partisan_overlay = False
        elif self._restore_partisan_on_load:
            self._restore_partisan_on_load = False
            # Elections loaded from Recent; controls are already enabled above.
            # User enables overlays manually to avoid unintended toggle.

        # Compactness and Pop. Deviation map views. PP alone is enough to shade;
        # the overlay blends in Reock when reock_data is present and falls back
        # to PP-only when it isn't (don't gate the overlay on Reock).
        has_compact = self.runner is not None and self.runner.pp_data is not None
        has_pops = self.runner is not None and self.runner.populations is not None
        dpg.configure_item(self._compactness_view, enabled=has_compact)
        dpg.configure_item(self._pop_dev_view,     enabled=has_pops)
        if not has_compact and dpg.get_value(self._compactness_view):
            dpg.set_value(self._compactness_view, False)
            if self.map_view:
                self.map_view.compactness_view = False
        if not has_pops and dpg.get_value(self._pop_dev_view):
            dpg.set_value(self._pop_dev_view, False)
            if self.map_view:
                self.map_view.pop_dev_view = False

        # Enable/disable partisan metric controls based on election data
        _pt_token = "secondary" if has_elections else "disabled_deep"
        for chk, lbl in [
            (self._mm_enabled,       self._mm_lbl),
            (self._eg_enabled,       self._eg_lbl),
            (self._pb_enabled,       self._pb_lbl),
            (self._pg_enabled,       self._pg_lbl),
            (self._hprop_enabled,    self._hprop_lbl),
            (self._hcmp_enabled,     self._hcmp_lbl),
            (self._seats_enabled,    self._seats_lbl),
            (self._majority_enabled, self._majority_lbl),
            (self._hinge_enabled,    self._hinge_lbl),
        ]:
            dpg.configure_item(chk, enabled=has_elections)
            self.theme.retoken(lbl, _pt_token)
        if not has_elections:
            for chk, ctrl_tag in [
                (self._mm_enabled,       "mm_controls"),
                (self._eg_enabled,       "eg_controls"),
                (self._pb_enabled,       "pb_controls"),
                (self._pg_enabled,       "pg_controls"),
                (self._hprop_enabled,    "hprop_controls"),
                (self._hcmp_enabled,     "hcmp_controls"),
                (self._seats_enabled,    "seats_controls"),
                (self._majority_enabled, "majority_controls"),
                (self._hinge_enabled,    "hinge_controls"),
            ]:
                dpg.set_value(chk, False)
                dpg.configure_item(ctrl_tag, show=False)
        # Enable/disable partisan panel menu items; close any open ones when unavailable
        for item_tag, panel_tag in [
            (self._panel_partisan_item,   "panel_partisanship"),
            (self._panel_win_chance_item, "panel_win_chance"),
            (self._panel_mm_item,         "panel_mm"),
            (self._panel_eg_item,         "panel_eg"),
            (self._panel_pb_item,         "panel_pb"),
            (self._panel_pg_item,         "panel_pg"),
            (self._panel_hprop_item,      "panel_hprop"),
            (self._panel_seats_item,      "panel_dem_seats"),
            (self._panel_hcmp_item,       "panel_hcmp"),
            (self._panel_majority_item,   "panel_majority"),
            (self._panel_hinge_item,      "panel_hinge"),
        ]:
            dpg.configure_item(item_tag, enabled=has_elections)
            if not has_elections and dpg.is_item_shown(panel_tag):
                dpg.set_value(item_tag, False)
                dpg.configure_item(panel_tag, show=False)

    # ── Popup toggle callbacks ────────────────────────────────────────────────
