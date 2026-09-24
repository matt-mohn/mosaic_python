"""Ensemble tool: repeat full runs until a count or duration is reached."""
from ._common import (
    _DIALOG_GAP,
    _DIALOG_PAD,
    _DIALOG_RM,
    AlgorithmStatus,
    dpg,
    output_dir,
    threading,
    time,
)

_ENS_W = 460
_ENS_TAG = "popup_ensemble"
_STEP_W = 24          # the "-" / "+" buttons beside the progress bar
# Light mode only: the palette's near-white default button disappears against
# the dialog, so the ensemble window's buttons step down to a mid gray.
_ENS_BTN = {"bg": (206, 211, 220), "hover": (190, 197, 210), "active": (174, 183, 200),
            "text": (22, 26, 35),
            "off_bg": (228, 231, 236), "off_text": (150, 156, 168)}


class EnsembleMixin:
    """Ensemble tool: repeat full runs until a count or duration is reached."""

    def _sync_ensemble_controls(self) -> None:
        """Ensemble needs an open map, an idle chain, and Relight off (Relight
        chains each run onto the last). Called every frame."""
        has_map = self.runner is not None and self.runner.gdf is not None
        busy = self.state.status in (AlgorithmStatus.RUNNING,
                                     AlgorithmStatus.PARTITIONING,
                                     AlgorithmStatus.PAUSED)
        busy = busy or self._session_workers_busy() or self._session_action_pending()
        dpg.configure_item(self._ensemble_item,
                           enabled=has_map and not busy and not self._relight_active)
        # The window may already be open when Relight gets armed; Relight
        # would reseed every run from the on-screen map, so block Start too.
        if dpg.does_item_exist(_ENS_TAG) and not self._ensemble_active:
            dpg.configure_item(self._ens_start_btn,
                               enabled=has_map and not busy and not self._relight_active)

    def _on_open_ensemble(self) -> None:
        if self._queue_session_action(self._on_open_ensemble):
            return
        if dpg.does_item_exist(_ENS_TAG):
            dpg.configure_item(_ENS_TAG, show=True)
            return
        wrap = _ENS_W - 2 * _DIALOG_PAD
        with self._dialog("Ensemble", _ENS_TAG, (_ENS_W, 360), modal=False):
            dpg.add_text(
                "Repeat full runs with the current map and settings. "
                "Save each completed map and its metrics.",
                wrap=wrap)
            # Dividers bracket the end-condition section.
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                self._ens_end_lbl = self.theme.text("End after:", "body")
                self._ens_mode = dpg.add_radio_button(
                    ["Count", "Duration", "Unlimited"], default_value="Count",
                    horizontal=True,
                    callback=self._on_ensemble_mode)
            with dpg.group(tag="ens_count_row", horizontal=True):
                self._ens_count = dpg.add_input_int(
                    default_value=10, min_value=1, min_clamped=True, width=120)
                self._ens_runs_lbl = self.theme.text("runs", "body")
            with dpg.group(tag="ens_duration_row", horizontal=True, show=False):
                self._ens_minutes = dpg.add_input_float(
                    default_value=5.0, min_value=0.1, min_clamped=True,
                    step=1.0, format="%.1f", width=120)
                self._ens_min_lbl = self.theme.text("minutes", "body")
            with dpg.group(tag="ens_unlimited_row", show=False):
                self._ens_unl_lbl = self.theme.text("Runs until Force Stop.", "muted")
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                self._ens_start_btn = dpg.add_button(
                    label="Start", width=90, callback=self._on_ensemble_start)
                dpg.bind_item_theme(self._ens_start_btn, self.theme.nudge_theme)
                self._ens_stop_btn = dpg.add_button(
                    label="Force Stop", width=100, enabled=False,
                    callback=self._on_ensemble_stop)
                self._ens_reset_btn = dpg.add_button(
                    label="Reset", width=70, enabled=False,
                    callback=self._on_ensemble_reset)
                self._ens_open_btn = dpg.add_button(
                    label="Open Folder", width=110, enabled=False,
                    callback=lambda: self._open_in_os(self._ens_writer.folder))
            dpg.add_spacer(height=6)
            # "-" / "+" beside the bar shrink or extend the run in progress.
            with dpg.group(horizontal=True):
                self._ens_bar = dpg.add_progress_bar(
                    default_value=0.0,
                    width=_ENS_W - 2 * _DIALOG_PAD - 2 * _STEP_W - 16)
                self._ens_shrink_btn = dpg.add_button(
                    label="-", width=_STEP_W, enabled=False,
                    callback=self._on_ensemble_shrink)
                self._ens_extend_btn = dpg.add_button(
                    label="+", width=_STEP_W, enabled=False,
                    callback=self._on_ensemble_extend)
            self._ens_status = dpg.add_text("Not started.", wrap=wrap)
            self._ens_eta = self.theme.text("", "muted", wrap=wrap)
            self._ens_folder_txt = self.theme.text("", "muted", wrap=wrap)
            # Targeting status: a header line, then one fixed-column row per
            # setting (built at Start, when the tuned settings are known).
            with dpg.group(tag="ens_target_box", show=False):
                self._ens_target_txt = self.theme.text("", "muted", wrap=wrap)
                dpg.add_table(tag="ens_target_table", header_row=False,
                              policy=dpg.mvTable_SizingFixedFit,
                              borders_innerV=False, borders_outerV=False,
                              borders_innerH=False, borders_outerH=False,
                              pad_outerX=False)
                with dpg.tooltip(self._ens_target_txt, tag="ens_target_help"):
                    dpg.add_text(
                        "Estimated score change and bootstrap interval for each "
                        "setting. Raising/falling describes the setting's latest "
                        "adjustment. Beta estimates do not guarantee improvement.",
                        wrap=300)
            # Results views: one button per pop-out, enabled once runs exist.
            dpg.add_spacer(height=6)
            dpg.add_separator()
            dpg.add_spacer(height=4)
            self.theme.text("Views", "heading")
            with dpg.group(horizontal=True):
                self._ens_hist_btn = dpg.add_button(
                    label="Histograms", width=100, enabled=False,
                    callback=self._on_open_ens_hist)
                self._ens_scatter_btn = dpg.add_button(
                    label="Scatterplot", width=100, enabled=False,
                    callback=self._on_open_ens_scatter)
                self._ens_roster_btn = dpg.add_button(
                    label="Roster", width=80, enabled=False,
                    callback=self._on_open_ens_roster)
                self._ens_map_btn = dpg.add_button(
                    label="Map", width=60, enabled=False,
                    callback=self._on_open_ens_map)
            # Footer: Advanced on the left, Close on the right.
            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                adv_w = self._dialog_btn_w("Advanced")
                close_w = self._dialog_btn_w("Close")
                dpg.add_button(label="Advanced", width=adv_w,
                               callback=self._on_open_ens_advanced)
                dpg.add_spacer(width=max(0, _ENS_W - 2 * _DIALOG_PAD - _DIALOG_RM
                                         - adv_w - close_w - 2 * _DIALOG_GAP))
                close = dpg.add_button(label="Close", width=close_w,
                                       callback=self._on_close_ensemble)
                dpg.bind_item_theme(close, self.theme.antinudge_theme)

        self._sync_ensemble_theme()

    def _on_open_ens_advanced(self) -> None:
        if self._queue_session_action(self._on_open_ens_advanced):
            return
        tag = "popup_ens_advanced"
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=True)
            dpg.focus_item(tag)
            return
        w = 400
        with self._dialog("Ensemble Advanced", tag, (w, 260), modal=False,
                          secondary=("Close",
                                     lambda: dpg.configure_item(tag, show=False))):
            self._ens_keep_best_chk = dpg.add_checkbox(
                label="Keep best map", default_value=self._ens_keep_best,
                callback=lambda s, v: setattr(self, "_ens_keep_best", v))
            self.theme.text(
                "Keep each run's lowest-scoring map instead of the map it "
                "ends on. Applies from the next Start.", "muted",
                wrap=w - 2 * _DIALOG_PAD)
            dpg.add_spacer(height=8)
            self._ens_targeting_chk = dpg.add_checkbox(
                label="Targeting (Beta)", default_value=self._ens_targeting,
                callback=self._on_targeting_toggle)
            from mosaic.targeting import FIRST_FIT, MAX_ITERATIONS, WARMUP_RUNS
            self.theme.text(
                f"Experiment with run settings to seek lower scores. "
                f"Tuning starts after {FIRST_FIT} runs; the first {WARMUP_RUNS} "
                f"are marked warm-up. No improvement is guaranteed. "
                f"Maximum {MAX_ITERATIONS:,} iterations per run.",
                "muted", wrap=w - 2 * _DIALOG_PAD)
            # Only meaningful with Targeting, so only shown with it.
            with dpg.group(tag="ens_tail_group", show=self._ens_targeting):
                dpg.add_spacer(height=4)
                self._ens_tail_chk = dpg.add_checkbox(
                    label="Favor lower-tail scores", default_value=self._ens_target_tail,
                    indent=20, callback=self._on_target_tail_toggle)
                self.theme.text(
                    "Target the 20th percentile of scores using all recent runs. "
                    "Needs more runs and more fitting time.",
                    "muted", wrap=w - 2 * _DIALOG_PAD - 20, indent=20)
        self._sync_ens_advanced()

    def _on_target_tail_toggle(self, sender, value) -> None:
        """Enable Keep best map with lower-tail targeting; users can uncheck it."""
        self._ens_target_tail = value
        if value:
            self._ens_keep_best = True
            dpg.set_value(self._ens_keep_best_chk, True)

    def _on_targeting_toggle(self, sender, value) -> None:
        self._ens_targeting = value
        dpg.configure_item("ens_tail_group", show=value)
        self._sync_ens_advanced()

    def _sync_ens_advanced(self) -> None:
        """Advanced options are fixed for the ensemble in progress."""
        if dpg.does_item_exist("popup_ens_advanced"):
            for chk in (self._ens_keep_best_chk, self._ens_targeting_chk):
                dpg.configure_item(chk, enabled=not self._ensemble_active)
            dpg.configure_item(self._ens_tail_chk, enabled=not self._ensemble_active)

    def _lock_end_condition(self, locked: bool) -> None:
        """Gray the end-condition controls while a run is in progress; they
        can't change mid-run (the -/+ buttons adjust it instead)."""
        self._sync_ens_advanced()
        for item in (self._ens_mode, self._ens_count, self._ens_minutes):
            dpg.configure_item(item, enabled=not locked)
            dpg.bind_item_theme(item, self._ens_locked_theme() if locked else 0)
        for lbl, rest in ((self._ens_end_lbl, "body"), (self._ens_runs_lbl, "body"),
                          (self._ens_min_lbl, "body"), (self._ens_unl_lbl, "muted")):
            self.theme.retoken(lbl, "disabled_deep" if locked else rest)

    def _ens_locked_theme(self):
        """Disabled-state look for radio and number inputs, rebuilt per palette."""
        mode = self.theme.palette.name
        cache = self.__dict__.setdefault("_ens_locked_themes", {})
        if mode not in cache:
            text = self.theme.color("disabled_deep")
            with dpg.theme() as t:
                for comp in (dpg.mvRadioButton, dpg.mvInputInt, dpg.mvInputFloat):
                    with dpg.theme_component(comp, enabled_state=False):
                        dpg.add_theme_color(dpg.mvThemeCol_Text, text)
                        dpg.add_theme_style(dpg.mvStyleVar_Alpha, 0.55)
            cache[mode] = t
        return cache[mode]

    def _sync_ensemble_theme(self) -> None:
        """Bind the gray-button theme in light mode; dark mode keeps the
        palette's buttons. Called on open and on theme change."""
        if not dpg.does_item_exist(_ENS_TAG):
            return
        if not hasattr(self, "_ens_btn_theme"):
            c = _ENS_BTN
            with dpg.theme() as self._ens_btn_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, c["bg"])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, c["hover"])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, c["active"])
                    dpg.add_theme_color(dpg.mvThemeCol_Text, c["text"])
                with dpg.theme_component(dpg.mvButton, enabled_state=False):
                    for col in (dpg.mvThemeCol_Button, dpg.mvThemeCol_ButtonHovered,
                                dpg.mvThemeCol_ButtonActive):
                        dpg.add_theme_color(col, c["off_bg"])
                    dpg.add_theme_color(dpg.mvThemeCol_Text, c["off_text"])
        light = self.theme.palette.name == "light"
        dpg.bind_item_theme(_ENS_TAG, self._ens_btn_theme if light else 0)
        if self._ensemble_active:
            self._lock_end_condition(True)   # re-gray in the new palette

    def _on_ensemble_mode(self) -> None:
        mode = dpg.get_value(self._ens_mode)
        dpg.configure_item("ens_count_row", show=mode == "Count")
        dpg.configure_item("ens_duration_row", show=mode == "Duration")
        dpg.configure_item("ens_unlimited_row", show=mode == "Unlimited")

    def _on_close_ensemble(self) -> None:
        if not self._ensemble_active:
            dpg.configure_item(_ENS_TAG, show=False)

    def _ensemble_settings(self, spec) -> dict:
        """What the ensemble folder records about how it was made."""
        import dataclasses
        s = self.state
        (hot_name,) = s.get("hot_start_filename")
        return {
            "shapefile": s.shapefile_path,
            "ensemble": {
                "end": spec.mode,
                "count": spec.count if spec.mode == "count" else None,
                "duration_minutes": (spec.duration_s / 60.0
                                     if spec.mode == "duration" else None),
                "base_seed": spec.base_seed,
                "targeting": spec.targeting,
                "map_kept": "best" if spec.keep_best else "final",
                # Hinge sits outside presets; record what hinge_chance measures.
                "hinge_threshold": (s.score_config.hinge_threshold
                                    if s.score_config.weight_hinge else None),
                "hinge_party": ((("D" if s.score_config.hinge_dem else "R"))
                                if s.score_config.weight_hinge else None),
            },
            "run": {
                "num_districts": s.num_districts,
                "tolerance": s.pop_tolerance,
                "tolerance_ratchet": s.tolerance_ratchet_mode,
                "iterations": s.max_iterations,
                "n3_probability": s.n3_probability,
                "flip_enabled": s.flip_enabled,
                "flip_midpoint": s.flip_midpoint,
                "tree_mode": s.tree_mode,
                "hot_start": hot_name or None,
            },
            "annealing": dataclasses.asdict(s.annealing_config),
            "preset": {"mosaic_preset": 1, **self._preset_from_widgets()},
        }

    def _on_ensemble_start(self) -> None:
        if self._queue_session_action(self._on_ensemble_start):
            return
        from datetime import datetime
        from uuid import uuid4

        from mosaic.ensemble import (
            EnsembleProgress,
            EnsembleSpec,
            EnsembleWriter,
            data_available,
            run_ensemble,
            summary_columns,
        )
        if self._ensemble_active or self.runner is None or self.runner.gdf is None:
            return
        if self._relight_active:
            dpg.set_value(self._ens_status, "Clear Relight to start an ensemble.")
            return
        if self._wait_for_workers(self._on_ensemble_start):
            return
        if not self._capture_run_settings():
            return
        (seed,) = self.state.get("seed")
        spec = EnsembleSpec(
            mode=dpg.get_value(self._ens_mode).lower(),   # count / duration / unlimited
            count=max(1, dpg.get_value(self._ens_count)),
            duration_s=max(0.1, dpg.get_value(self._ens_minutes)) * 60.0,
            base_seed=seed,
            keep_best=self._ens_keep_best,
            targeting=self._ens_targeting,
        )
        progress = EnsembleProgress()
        if spec.targeting:
            from mosaic.targeting import Targeter
            s = self.state
            progress.targeter = Targeter.from_settings(
                s.max_iterations, s.n3_probability, s.flip_enabled, s.flip_midpoint,
                s.annealing_config.enabled, s.annealing_config.initial_temp_factor,
                guided=s.annealing_config.cooling_mode == "GUIDED",
                guide_fraction=s.annealing_config.guide_fraction,
                tail=self._ens_target_tail, seed=seed)
        tuned = (tuple(k.name for k in progress.targeter.knobs)
                 if progress.targeter else ())
        precinct_ids = list(range(len(self.runner.gdf)))
        id_col = "precinct_id"
        col = self._loaded_config.id_col if self._loaded_config else ""
        if col and col in self.runner.gdf.columns:
            precinct_ids, id_col = self.runner.gdf[col].tolist(), col
        folder = output_dir() / (
            f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}")
        try:
            writer = EnsembleWriter(folder, precinct_ids, id_col,
                                    summary_columns(data_available(
                                        self.runner, self.state.score_config), tuned))
            # Snapshot now: targeting rewrites run settings in state per run.
            self._ens_settings0 = self._ensemble_settings(spec)
            writer.write_settings(self._ens_settings0)
        except Exception as e:
            dpg.set_value(self._ens_status, f"Could not create {folder}: {e}")
            return

        self._ens_writer = writer
        self._ens_source_gdf = self.runner.gdf
        self._ens_spec = spec
        self._ens_progress = progress
        dpg.configure_item("ens_target_box", show=spec.targeting)
        dpg.set_value(self._ens_target_txt, "")
        self._build_target_table()
        self._ensemble_active = True
        self._clear_all_series()
        self.state.update(status_message="Ensemble running")
        self.algorithm_thread = threading.Thread(
            target=run_ensemble, name="ensemble", daemon=False,
            args=(self.runner, self.state, spec, writer, self._ens_progress))
        self.algorithm_thread.start()

        self._freeze_main(True)
        dpg.configure_item(self._ens_hist_btn, enabled=True)
        dpg.configure_item(self._ens_scatter_btn, enabled=True)
        dpg.configure_item(self._ens_roster_btn, enabled=True)
        dpg.configure_item(self._ens_map_btn, enabled=True)
        dpg.configure_item(self._ens_start_btn, enabled=False)
        self._lock_end_condition(True)
        dpg.configure_item(self._ens_stop_btn, enabled=True)
        # Nothing to extend or shrink without an end condition.
        dpg.configure_item(self._ens_extend_btn, enabled=spec.mode != "unlimited")
        dpg.configure_item(self._ens_reset_btn, enabled=False)
        dpg.configure_item(self._ens_open_btn, enabled=True)
        dpg.set_value(self._ens_bar, 0.0)
        dpg.set_value(self._ens_folder_txt, f"Writing to {folder}")

    _TARGET_COLS = ("name", "effect", "interval", "word", "history")
    _WORD_TOKEN = {"raising": "accent_green", "falling": "error",
                   "unclear": "muted", "--": "muted"}

    def _build_target_table(self) -> None:
        """One row per tuned setting, cells kept for per-frame updates."""
        dpg.delete_item("ens_target_table", children_only=True)
        self._ens_target_cells = []
        t = self._ens_progress.targeter
        if t is None:
            return
        for _ in self._TARGET_COLS:
            dpg.add_table_column(parent="ens_target_table")
        for _ in t.knobs:
            with dpg.table_row(parent="ens_target_table"):
                self._ens_target_cells.append({
                    c: self.theme.text("", "body" if c == "name" else "muted")
                    for c in self._TARGET_COLS})

    def _show_targeting(self, run: int) -> None:
        t = self._ens_progress.targeter
        if t is None:
            return
        with self._ens_progress.lock:
            head, rows = (self._ens_progress.target_header,
                          self._ens_progress.target_rows)
        dpg.set_value(self._ens_target_txt, head)
        for cells, r in zip(self._ens_target_cells, rows):
            dpg.set_value(cells["name"], f"{r['label']} {r['value']}")
            for c in ("effect", "interval", "word", "history"):
                dpg.set_value(cells[c], r[c])
            self.theme.retoken(cells["word"], self._WORD_TOKEN.get(r["word"], "muted"))

    def _on_ensemble_stop(self) -> None:
        with self._ens_progress.lock:
            self._ens_progress.stop_requested = True
        self.state.request_stop()
        dpg.configure_item(self._ens_stop_btn, enabled=False)
        dpg.configure_item(self._ens_extend_btn, enabled=False)
        dpg.configure_item(self._ens_shrink_btn, enabled=False)
        dpg.set_value(self._ens_status, "Stopping; completed runs are kept...")
        dpg.set_value(self._ens_eta, "")

    def _on_ensemble_extend(self) -> None:
        """Grow the current target count or duration by 20% (compounds). The
        loop re-reads the spec before each run, so this takes effect at once."""
        with self._ens_progress.lock:
            if self._ens_spec.mode == "count":
                self._ens_spec.count += max(1, round(self._ens_spec.count * 0.2))
            else:
                self._ens_spec.duration_s *= 1.2

    def _on_ensemble_shrink(self) -> None:
        """Cut the unfinished part by 20% (compounds): runs not yet complete in
        count mode, time left in duration mode. Never below the run in
        progress, which always finishes and is kept."""
        p = self._ens_progress
        with p.lock:
            if self._ens_spec.mode == "count":
                self._ens_spec.count = shrink_count(
                    self._ens_spec.count, p.runs_done, p.current_run)
            else:
                self._ens_spec.duration_s = shrink_duration(
                    self._ens_spec.duration_s, time.time() - p.started)

    def _on_ensemble_reset(self) -> None:
        """Clear the finished ensemble's readout. Files and views are untouched."""
        dpg.set_value(self._ens_bar, 0.0)
        dpg.set_value(self._ens_status, "Not started.")
        dpg.set_value(self._ens_eta, "")
        dpg.set_value(self._ens_folder_txt, "")
        dpg.set_value(self._ens_target_txt, "")
        dpg.configure_item("ens_target_box", show=False)
        dpg.configure_item(self._ens_reset_btn, enabled=False)
        dpg.configure_item(self._ens_open_btn, enabled=False)

    def _update_ensemble_ui(self) -> None:
        """Per-frame refresh while an ensemble runs; replaces _update_ui."""
        p = self._ens_progress
        with p.lock:
            done, cur, started = p.runs_done, p.current_run, p.started
            finished, error, stopping = p.finished, p.error, p.stop_requested
        with p.lock:
            count, duration, done_elapsed = (self._ens_spec.count,
                                             self._ens_spec.duration_s,
                                             p.done_elapsed)
        mode = self._ens_spec.mode
        elapsed = time.time() - started if started else 0.0
        (it, max_it) = self.state.get("current_iteration", "max_iterations")
        run_frac = it / max_it if max_it else 0.0
        in_run = cur > done
        if mode == "count":
            frac = (done + (run_frac if in_run else 0.0)) / count
            line = f"Run {min(cur, count)} of {count}"
        elif mode == "duration":
            frac = elapsed / duration
            line = f"Run {cur}  --  {_clock(elapsed)} of {_clock(duration)}"
        else:
            frac = unlimited_progress(elapsed)
            line = f"Run {cur}  --  {_clock(elapsed)}"
        if not finished:
            dpg.set_value(self._ens_bar, min(frac, 1.0))
            if not stopping:
                can = (shrink_count(count, done, cur) < count if mode == "count"
                       else duration - elapsed > 1.0 if mode == "duration"
                       else False)
                dpg.configure_item(self._ens_shrink_btn, enabled=can)
            if not stopping:
                dpg.set_value(self._ens_status,
                              f"{line}  --  {done} complete  --  "
                              f"iteration {it:,} / {max_it:,}")
                dpg.set_value(self._ens_eta, _estimate(
                    mode, count, duration, done, in_run, elapsed,
                    done_elapsed, run_frac) if mode != "unlimited"
                    else "Runs until Force Stop.")
            self._show_targeting(max(cur, 1))
            return

        self._ensemble_active = False
        self._freeze_main(False)
        dpg.configure_item(self._ens_start_btn, enabled=True)
        self._lock_end_condition(False)
        dpg.configure_item(self._ens_stop_btn, enabled=False)
        dpg.configure_item(self._ens_extend_btn, enabled=False)
        dpg.configure_item(self._ens_shrink_btn, enabled=False)
        dpg.configure_item(self._ens_reset_btn, enabled=True)
        dpg.set_value(self._ens_eta, "")
        # Unlimited ends by Force Stop; that's its finish, not an interruption.
        complete = not error and (not stopping or mode == "unlimited")
        dpg.set_value(self._ens_bar, 1.0 if complete else min(frac, 1.0))
        if error:
            msg = f"Stopped by an error after {done} runs: {error}"
        elif stopping and mode == "unlimited":
            msg = f"Stopped: {done} runs in {_clock(elapsed)}."
        elif stopping:
            msg = f"Force stopped: {done} runs kept ({_clock(elapsed)})."
        else:
            msg = f"Finished: {done} runs in {_clock(elapsed)}."
        dpg.set_value(self._ens_status, msg)
        self._show_targeting(done + 1)
        # The main window's charts and map describe no single run; start clean.
        self._on_reset()
        self.state.update(status_message=f"Ensemble: {done} runs saved")


def unlimited_progress(elapsed_s: float) -> float:
    """Progress-bar illusion for Unlimited: rises fast then crawls, never
    full. 0.95 * t / (t + 60 s): ~5% at 3 s, 25% at 21 s, 80% at 5.3 min,
    85% at 8.5 min, 90% at 18 min, and still creeping after that."""
    t = max(0.0, elapsed_s)
    return 0.95 * t / (t + 60.0)


def shrink_count(count: int, done: int, current: int) -> int:
    """Target after one shrink: 20% of the unfinished runs (at least 1) off,
    floored at the run in progress (it always completes)."""
    cut = max(1, round((count - done) * 0.2))
    return max(current, 1, count - cut)


def shrink_duration(duration_s: float, elapsed_s: float) -> float:
    """Duration after one shrink: time left cut by 20%. The run in progress
    still finishes past it, as at any deadline."""
    return elapsed_s + 0.8 * max(0.0, duration_s - elapsed_s)


def _estimate(mode, count, duration, done, in_run, elapsed, done_elapsed,
              run_frac) -> str:
    """Est. time left (count mode) or est. final run count (duration mode),
    from the mean completed-run time, or the current run's pace before any."""
    into = elapsed - done_elapsed          # time spent on the run in progress
    if done:
        per_run = done_elapsed / done
    elif run_frac > 0.02:
        per_run = into / run_frac
    else:
        return "Estimating..."
    if mode == "count":
        left = max(0.0, (count - done) * per_run - (into if in_run else 0.0))
        return f"Est. time left: {_clock(left)}"
    # Runs keep starting until the deadline; the current one finishes past it.
    cur_end = done_elapsed + per_run if in_run else elapsed
    more = max(0, -int(-(duration - cur_end) // per_run)) if cur_end < duration else 0
    return f"Est. final count: {done + int(in_run) + more} runs"


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
