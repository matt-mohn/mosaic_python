"""File/session menu: recent files, new/close, updates, output dir."""
from ._common import (
    _DOWNLOAD_URL,
    _RECENT_FILE,
    _RECENT_MAX,
    _RECENT_PRESETS_FILE,
    _SETTINGS_DIR,
    _UPDATE_CHECK_URL,
    AlgorithmRunner,
    Path,
    ShapefileConfig,
    __version__,
    dpg,
    log,
    np,
    output_dir,
    webbrowser,
)


class MenuMixin:
    """File/session menu: recent files, new/close, updates, output dir."""

    @staticmethod
    def _version_tuple(v):
        """Parse 'X.Y.Z' to a comparable tuple; () if unparseable."""
        try:
            return tuple(int(x) for x in v.split("."))
        except (ValueError, AttributeError):
            return ()

    def _on_open_output_dir(self):
        """Open output/ (saved maps + assignment/metric CSVs) in the OS file browser."""
        import os
        import subprocess
        import sys
        out = output_dir()
        out.mkdir(parents=True, exist_ok=True)  # ensure the folder exists before opening
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(out)], check=False)
            elif os.name == "nt":
                os.startfile(str(out))  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(out)], check=False)
            self.state.update(status_message=f"Opened {out}")
        except Exception as e:
            self.state.update(status_message=f"Could not open output folder: {e}")

    def _open_in_os(self, path) -> None:
        """Open a saved file (or folder) in the OS default handler. Best-effort."""
        import os
        import subprocess
        import sys
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception:
            pass

    def _on_check_updates(self):
        """Manual update check (File menu). Compares the local version to the
        public repo's pyproject version. Synchronous — it's user-initiated — with
        a short timeout so it can't hang the UI for long."""
        import re as _re
        import urllib.request
        latest = None
        try:
            with urllib.request.urlopen(_UPDATE_CHECK_URL, timeout=5) as resp:
                text = resp.read().decode("utf-8")
            m = _re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, _re.M)
            latest = m.group(1) if m else None
        except Exception:
            latest = None
        self._show_update_result(latest)

    def _show_update_result(self, latest) -> None:
        cur = __version__
        update_available = False
        if latest is None:
            msg = ("Couldn't reach GitHub to check for updates. "
                   "Check your connection and try again.")
        elif self._version_tuple(latest) > self._version_tuple(cur):
            update_available = True
            msg = (f"A newer version is available: v{latest} (you have v{cur}).\n\n"
                   "Click Download to open the install page, then follow the "
                   "download-and-run steps for your computer.")
        else:
            msg = f"You're up to date (v{cur})."
        if update_available:
            buttons = [
                ("Download", lambda: webbrowser.open(_DOWNLOAD_URL), "primary"),
                ("Close", lambda: dpg.delete_item("popup_update")),
            ]
        else:
            buttons = [("Close", lambda: dpg.delete_item("popup_update"))]
        with self._dialog("Check for updates", "popup_update", (520, 220),
                          buttons=buttons):
            dpg.add_text(msg, wrap=500)

    def _load_recent_shapefiles(self) -> None:
        """Read recent-shapefile list from disk."""
        import json
        if not _RECENT_FILE.exists():
            return
        try:
            data = json.loads(_RECENT_FILE.read_text(encoding="utf-8"))
            # Drop malformed entries here, not at render time: the menu is
            # built during setup(), where a bad entry crashes startup.
            self._recent_shapefiles = [
                e for e in data
                if isinstance(e, dict) and isinstance(e.get("path"), str)
            ] if isinstance(data, list) else []
        except Exception:
            self._recent_shapefiles = []

    def _save_recent_shapefiles(self) -> None:
        """Persist recent-shapefile list to disk."""
        import json
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            _RECENT_FILE.write_text(
                json.dumps(self._recent_shapefiles[:_RECENT_MAX], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _push_recent_shapefile(self, path: str,
                               config: "ShapefileConfig") -> None:
        """Prepend path+config to the recent list, cap it, refresh menu."""
        entry = {
            "path": path,
            "config": {
                "pop_col":    config.pop_col,
                "id_col":     config.id_col,
                "county_col": config.county_col,
                "elections":  config.elections,
                "demographics": config.demographics,
            },
        }
        self._recent_shapefiles = [
            e for e in self._recent_shapefiles if e.get("path") != path
        ]
        self._recent_shapefiles.insert(0, entry)
        self._recent_shapefiles = self._recent_shapefiles[:_RECENT_MAX]
        self._save_recent_shapefiles()
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        """Rebuild the Open Recent submenu from _recent_shapefiles."""
        if not dpg.does_item_exist("file_recent_menu"):
            return
        dpg.delete_item("file_recent_menu", children_only=True)
        if not self._recent_shapefiles:
            dpg.add_menu_item(
                label="(no recent files)", enabled=False,
                parent="file_recent_menu",
            )
            return
        for entry in self._recent_shapefiles:
            dpg.add_menu_item(
                label=Path(entry["path"]).name,
                callback=self._on_open_recent,
                user_data=entry,
                parent="file_recent_menu",
            )

    def _on_open_recent(self, sender, app_data, user_data) -> None:
        """Open a recently used shapefile, skipping the column picker."""
        if self._queue_session_action(self._on_open_recent, sender, app_data, user_data):
            return
        if not user_data:
            return
        entry = user_data
        path = entry.get("path", "")
        if not path or not Path(path).exists():
            self.state.update(
                status_message=f"Recent file not found: {path}")
            return
        cfg_d = entry.get("config", {})
        # Normalize stored election pairs to ShapefileConfig's tuple format.
        raw_elections = cfg_d.get("elections", [])
        elections = [tuple(e) for e in raw_elections if len(e) == 2]
        demographics = cfg_d.get("demographics") or None
        config = ShapefileConfig(
            pop_col=cfg_d.get("pop_col", ""),
            id_col=cfg_d.get("id_col", ""),
            county_col=cfg_d.get("county_col"),
            elections=elections,
            demographics=demographics,
        )
        self._request_open_shapefile(path, config)

    def _plan_unsaved(self) -> bool:
        """True if a run result exists and differs from the last saved assignment
        (Save Assignments writes current_assignment, so compare against that)."""
        with self.state._lock:
            best = self.state.best_assignment
            cur = (self.state.current_assignment.copy()
                   if self.state.current_assignment is not None else None)
        if best is None:
            return False
        if cur is None or self._saved_plan is None:
            return True
        return not np.array_equal(cur, self._saved_plan)

    def _on_close(self) -> None:
        """File > Close / Ctrl+W: warn only if the plan has unsaved changes."""
        if self._plan_unsaved():
            dpg.configure_item("popup_close_confirm", show=True)
        else:
            dpg.stop_dearpygui()

    def _on_new(self) -> None:
        """File > New: warn only if the plan has unsaved changes."""
        if self._queue_session_action(self._on_new):
            return
        if self._ensemble_active or self._session_action_pending():
            return
        if self._plan_unsaved():
            dpg.configure_item("popup_new_confirm", show=True)
        else:
            self._do_new()

    def _do_new(self) -> None:
        """Clear displayed results and replace the runner with an unloaded one."""
        if self._queue_session_action(self._do_new):
            return
        if self._ensemble_active:
            return
        dpg.configure_item("popup_new_confirm", show=False)
        if self._wait_for_workers(self._do_new):
            return
        self._on_reset()
        self._clear_file_state()
        self._saved_plan = None
        self._reset_map_navigation()
        self.runner = AlgorithmRunner(self.state)
        self._sync_tolerance_people()
        self._loaded_config = None
        self._has_elections = False
        self._map_loaded_path = ""
        self._map_loaded_gdf_id = 0
        self._map_data_gdf_id = 0
        self._map_loading = False
        self._map_ready = False
        self.state.update(shapefile_path="", status_message="")
        self._update_hot_start_display(None)
        self._set_preset_info([])
        dpg.set_value(self._shp_info, "Load a shapefile to begin.")
        self.theme.retoken(self._shp_info, "muted")
        if self.map_view is not None:
            self.map_view.wipe()
            self._update_map_preview()

    # ── Score presets (Configuration menu) ────────────────────────────────────

    # (section, enabled checkbox, toggle callback, Scores-menu item, row tag,
    #  label). Hinge is absent on purpose; applying a preset turns it off.
    _PRESET_SCORES = (
        ("cut_edges", "_cut_enabled", "_on_cut_toggle",
         "_svis_cuts", "score_row_cuts", "Cut Edges"),
        ("holistic_compactness", "_hc_enabled", "_on_hc_toggle",
         "_svis_hc", "score_row_hc", "Compactness"),
        ("polsby_popper", "_pp_enabled", "_on_pp_toggle",
         "_svis_pp", "score_row_pp", "Polsby-Popper"),
        ("reock", "_reock_enabled", "_on_reock_toggle",
         "_svis_reock", "score_row_reock", "Reock"),
        ("holistic_splitting", "_hsplit_enabled", "_on_hsplit_toggle",
         "_svis_hsplit", "score_row_hsplit", "County Congruence"),
        # After County Congruence, whose toggle turns the bias on.
        ("county_bias", "_county_bias_enabled", "_on_county_bias_toggle",
         "_svis_countybias", "score_row_countybias", "County-Edge Bias"),
        ("county_splits", "_cs_enabled", "_on_cs_toggle",
         "_svis_cs", "score_row_cs", "Classic Splitting"),
        ("pop_deviation", "_popdev_enabled", "_on_popdev_score_toggle",
         "_svis_popdev", "score_row_popdev", "Population Deviation"),
        ("alignment", "_alignment_enabled", "_on_alignment_toggle",
         "_svis_alignment", "score_row_alignment", "Alignment"),
        ("mean_median", "_mm_enabled", "_on_mm_toggle",
         "_svis_mm", "score_row_mm", "Mean-Median"),
        ("efficiency_gap", "_eg_enabled", "_on_eg_toggle",
         "_svis_eg", "score_row_eg", "Efficiency Gap"),
        ("partisan_bias", "_pb_enabled", "_on_pb_toggle",
         "_svis_pb", "score_row_pb", "Partisan Bias"),
        ("partisan_gini", "_pg_enabled", "_on_pg_toggle",
         "_svis_pg", "score_row_pg", "Partisan Gini"),
        ("holistic_proportionality", "_hprop_enabled", "_on_hprop_toggle",
         "_svis_hprop", "score_row_hprop", "Proportionality"),
        ("holistic_competitiveness", "_hcmp_enabled", "_on_hcmp_toggle",
         "_svis_hcmp", "score_row_hcmp", "Competitiveness"),
        ("dem_seats", "_seats_enabled", "_on_seats_toggle",
         "_svis_seats", "score_row_seats", "Expected Dem Seats"),
        ("majority_chance", "_majority_enabled", "_on_majority_toggle",
         "_svis_majority", "score_row_majority", "Chance of Majority"),
        ("representation", "_representation_enabled", "_on_representation_toggle",
         "_svis_representation", "score_row_representation",
         "Electoral Opportunity"),
        ("minority_cohesion", "_minority_cohesion_enabled",
         "_on_minority_cohesion_toggle", "_svis_minority_cohesion",
         "score_row_minority_cohesion", "Neighborhood Severance"),
        ("community_congruence", "_community_congruence_enabled",
         "_on_community_congruence_toggle", "_svis_community_congruence",
         "score_row_community_congruence", "Community Dispersion"),
    )

    # (section, key, widget attr) for values stored as-is on a widget.
    _PRESET_VALUES = (
        ("cut_edges", "weight", "_w_cut_edges"),
        ("county_splits", "weight_excess", "_w_county_excess"),
        ("county_splits", "weight_unified", "_w_county_unified"),
        ("holistic_splitting", "weight", "_w_holistic_splitting"),
        ("holistic_splitting", "unclipped", "_hsplit_unclipped"),
        ("county_bias", "multiplier", "_county_bias"),
        ("holistic_compactness", "weight", "_w_holistic_compactness"),
        ("holistic_compactness", "unclipped", "_hcompact_unclipped"),
        ("polsby_popper", "weight", "_w_polsby_popper"),
        ("reock", "weight", "_w_reock"),
        ("pop_deviation", "weight", "_w_pop_deviation"),
        ("alignment", "weight", "_w_alignment"),
        ("alignment", "win_threshold", "_alignment_win_threshold"),
        ("mean_median", "weight", "_w_mean_median"),
        ("mean_median", "bound", "_mm_bound"),
        ("efficiency_gap", "weight", "_w_efficiency_gap"),
        ("efficiency_gap", "bound", "_eg_bound"),
        ("partisan_bias", "weight", "_w_partisan_bias"),
        ("partisan_bias", "bound", "_pbias_bound"),
        ("partisan_gini", "weight", "_w_partisan_gini"),
        ("dem_seats", "weight", "_w_dem_seats"),
        ("holistic_proportionality", "weight", "_w_holistic_proportionality"),
        ("holistic_proportionality", "unclipped", "_prop_unclipped"),
        ("holistic_competitiveness", "weight", "_w_holistic_competitiveness"),
        ("holistic_competitiveness", "unclipped", "_comp_unclipped"),
        ("majority_chance", "weight", "_w_majority"),
        ("representation", "weight", "_w_representation"),
        ("representation", "unclipped", "_representation_unclipped"),
        ("minority_cohesion", "weight", "_w_minority_cohesion"),
        ("community_congruence", "weight", "_w_community_congruence"),
        ("partisan", "win_prob_at_55", "_win_prob"),
        ("partisan", "swing_sigma", "_swing_sigma"),
        ("partisan", "quadratic_penalty", "_partisan_quadratic_penalty"),
        ("opportunity", "midpoint", "_opportunity_midpoint"),
        ("opportunity", "steepness", "_opportunity_steepness"),
        ("opportunity", "solid", "_opportunity_solid"),
        ("opportunity", "smart_targets", "_opportunity_smart_targets"),
    )

    # Data a score needs; a preset applied without it is reported by data, not score.
    _PRESET_NEEDS = {
        **{s: "election" for s in (
            "mean_median", "efficiency_gap", "partisan_bias", "partisan_gini",
            "holistic_proportionality", "holistic_competitiveness", "dem_seats",
            "majority_chance")},
        **{s: "demographic" for s in (
            "representation", "minority_cohesion", "community_congruence")},
        **{s: "county" for s in ("county_splits", "holistic_splitting", "county_bias")},
        "alignment": "reference plan",
    }

    _PRESET_DIR_ATTRS = (("mean_median", "_mm_dir"), ("efficiency_gap", "_eg_dir"),
                         ("partisan_bias", "_pb_dir"))
    _FOCUS_LABELS = {"none": "All residents", "rep": "Republican",
                     "dem": "Democratic"}

    def _preset_from_widgets(self) -> dict:
        from mosaic.presets import default_preset
        p = default_preset()
        for sec, chk, *_ in self._PRESET_SCORES:
            p[sec]["enabled"] = bool(dpg.get_value(getattr(self, chk)))
        for sec, key, attr in self._PRESET_VALUES:
            v = dpg.get_value(getattr(self, attr))
            d = p[sec][key]
            p[sec][key] = (bool(v) if isinstance(d, bool)
                           else float(v) if isinstance(d, float) else int(v))
        mode = {"Fair": "fair", "D": "favor_dem", "R": "favor_rep"}
        for sec, attr in self._PRESET_DIR_ATTRS:
            p[sec]["mode"] = mode[dpg.get_value(getattr(self, attr))]
        p["efficiency_gap"]["robust"] = (
            dpg.get_value(self._eg_mode) == "Robust (recommended)")
        p["dem_seats"]["favor"] = (
            "dem" if dpg.get_value(self._dem_seats_dir) == "D" else "rep")
        p["majority_chance"]["party"] = (
            "rep" if dpg.get_value(self._majority_rep_chk) else "dem")
        focus = {v: k for k, v in self._FOCUS_LABELS.items()}
        p["alignment"]["party_focus"] = focus[dpg.get_value(self._alignment_focus)]
        p["alignment"]["restrict_to_party"] = bool(
            dpg.get_value(self._alignment_restrict))
        p["pop_deviation"]["safe_harbor"] = dpg.get_value(self._pop_dev_harbor) / 100.0
        return p

    @staticmethod
    def _preset_set(item, value) -> None:
        """set_value, clamped to a slider's range and cast to its type."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cfg = dpg.get_item_configuration(item)
            lo, hi = cfg.get("min_value"), cfg.get("max_value")
            if lo is not None and hi is not None:
                value = min(max(value, lo), hi)
            if dpg.get_item_type(item).endswith("SliderInt"):
                value = int(round(value))
        dpg.set_value(item, value)

    def _apply_preset(self, p: dict) -> tuple[list[str], list[str]]:
        """Apply preset values to the widgets; return (missing data, other notes)."""
        from mosaic.presets import validate_preset
        p = validate_preset(p)
        for sec, key, attr in self._PRESET_VALUES:
            self._preset_set(getattr(self, attr), p[sec][key])
        label = {"fair": "Fair", "favor_dem": "D", "favor_rep": "R"}
        for sec, attr in self._PRESET_DIR_ATTRS:
            dpg.set_value(getattr(self, attr), label[p[sec]["mode"]])
        dpg.set_value(self._eg_mode, "Robust (recommended)"
                      if p["efficiency_gap"]["robust"] else "Static")
        dpg.set_value(self._dem_seats_dir,
                      "D" if p["dem_seats"]["favor"] == "dem" else "R")
        dpg.set_value(self._majority_dem_chk, p["majority_chance"]["party"] == "dem")
        dpg.set_value(self._majority_rep_chk, p["majority_chance"]["party"] == "rep")
        dpg.set_value(self._alignment_focus,
                      self._FOCUS_LABELS[p["alignment"]["party_focus"]])
        self._on_alignment_focus()
        dpg.set_value(self._alignment_restrict,
                      p["alignment"]["restrict_to_party"]
                      and p["alignment"]["party_focus"] != "none")
        self._preset_set(self._pop_dev_harbor,
                         p["pop_deviation"]["safe_harbor"] * 100.0)

        if dpg.get_value(self._hinge_enabled):
            dpg.set_value(self._hinge_enabled, False)
            self._on_hinge_toggle()

        r = self.runner
        have = {
            "election": bool(getattr(r, "election_arrays", None)),
            "demographic": getattr(r, "vap_data", None) is not None,
            "county": getattr(r, "county_array", None) is not None,
            "reference plan": getattr(r, "alignment_data", None) is not None,
        }
        missing: list[str] = []
        for sec, chk, _cb, _svis, _row, lbl in self._PRESET_SCORES:
            want = p[sec]["enabled"]
            need = self._PRESET_NEEDS.get(sec)
            gap = None
            if want and need and not have[need]:
                gap = need if need == "reference plan" else f"{need} data"
            elif want and not self._preset_chk_usable(chk):
                gap = f"{lbl} (not applicable here)"   # data present, nothing to measure
            if gap is not None:
                if gap not in missing:
                    missing.append(gap)
                if not self._preset_chk_usable(chk):
                    want = False
            self._preset_set_enabled(sec, want)
        order = ["election data", "demographic data", "county data", "reference plan"]
        missing.sort(key=lambda g: order.index(g) if g in order else len(order))

        tol = dpg.get_value(self._tolerance)
        if p["pop_deviation"]["safe_harbor"] * 100.0 > tol:
            other = [f"safe harbor capped at {tol:.2f}%"]
        else:
            other = []
        return missing, other

    def _on_clear_scores(self) -> None:
        """Reset every scoring control to its startup value, with every score
        off (Compactness included). Row visibility is left alone."""
        from mosaic.presets import default_preset
        p = default_preset()
        p["holistic_compactness"]["enabled"] = False
        self._apply_preset(p)
        # Hinge sits outside presets; restore its startup values here.
        self._preset_set(self._w_hinge, 1)
        self._preset_set(self._hinge_threshold, 8)
        dpg.set_value(self._hinge_dem_chk, True)
        dpg.set_value(self._hinge_rep_chk, False)
        self._set_preset_info([])
        self.state.update(status_message="Cleared all scores.")

    def _preset_chk_usable(self, chk: str) -> bool:
        return bool(dpg.get_item_configuration(getattr(self, chk)).get("enabled", True))

    def _preset_set_enabled(self, sec: str, on: bool) -> None:
        """Set one score's checkbox; fire its toggle only on a change, since
        some toggles have side effects (County Congruence turns on county bias)."""
        for s, chk, cb, svis, row, _ in self._PRESET_SCORES:
            if s != sec:
                continue
            item = getattr(self, chk)
            if on:
                # A hidden row would force the score back off; reveal it.
                dpg.set_value(getattr(self, svis), True)
                dpg.configure_item(row, show=True)
            if bool(dpg.get_value(item)) != on:
                dpg.set_value(item, on)
                getattr(self, cb)()
            return

    def _set_preset_info(self, missing: list[str], other: list[str] = ()) -> None:
        """Warning line under the shapefile label; hidden when nothing to say."""
        parts = (["missing " + ", ".join(missing)] if missing else []) + list(other)
        if parts:
            dpg.set_value(self._preset_info,
                          "Preset partly applied: " + "; ".join(parts))
        dpg.configure_item(self._preset_info, show=bool(parts))

    def _pick_preset_path(self, save: bool, on_path) -> None:
        """Select a preset path with the platform picker; ignore Cancel."""
        import sys

        from mosaic.paths import presets_dir
        pdir = presets_dir()
        pdir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            from mosaic.gui.file_dialog import windows_file_dialog
            try:
                path = windows_file_dialog(
                    save=save, title="Save Preset" if save else "Apply Preset",
                    file_filter="Mosaic presets (*.toml)|*.toml|All files (*.*)|*.*",
                    initial_dir=pdir, extension="toml",
                    default_name="preset.toml" if save else "",
                )
            except Exception:
                log.exception("Preset picker failed")
                self.state.update(status_message="Could not open preset picker. See log.")
                return
            if path:
                on_path(path)
            return
        if sys.platform == "darwin":
            if dpg.does_item_exist("__preset_file_dialog"):
                dpg.delete_item("__preset_file_dialog")

            def _cb(_s, d):
                path = d.get("file_path_name", "") if isinstance(d, dict) else ""
                if path:
                    on_path(path)
            with dpg.file_dialog(
                directory_selector=False, show=True, modal=True, callback=_cb,
                cancel_callback=lambda *_: None, default_path=str(pdir),
                default_filename="preset" if save else "",
                width=700, height=450, tag="__preset_file_dialog",
                label="Save Preset" if save else "Apply Preset",
            ):
                dpg.add_file_extension(".toml", color=(120, 220, 120, 255))
            return
        import tkinter as tk
        from tkinter import filedialog
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            types = [("Mosaic presets", "*.toml"), ("All files", "*.*")]
            if save:
                path = filedialog.asksaveasfilename(
                    title="Save Preset", defaultextension=".toml",
                    filetypes=types, initialdir=str(pdir))
            else:
                path = filedialog.askopenfilename(
                    title="Apply Preset", filetypes=types, initialdir=str(pdir))
            root.destroy()
        except Exception:
            log.exception("Preset: file dialog failed")
            self.state.update(status_message="Preset: file dialog could not open.")
            return
        if path:
            on_path(path)

    def _on_save_preset(self) -> None:
        self._pick_preset_path(True, self._save_preset_to)

    def _save_preset_to(self, path: str) -> None:
        from mosaic.presets import write_preset
        p = Path(path)
        if p.suffix.lower() != ".toml":
            p = p.with_suffix(".toml")
        try:
            write_preset(p, self._preset_from_widgets())
        except Exception as e:
            self.state.update(status_message=f"Could not save preset: {e}")
            return
        self._push_recent_preset(str(p))
        self.state.update(status_message=f"Saved preset '{p.stem}'.")

    def _sync_preset_controls(self) -> None:
        """Apply needs an open map, since gating reads its data. Like any score
        edit, it is allowed mid-run and takes effect on the next run. Called
        every frame."""
        has_map = self.runner is not None and self.runner.gdf is not None
        dpg.configure_item(self._preset_apply_item, enabled=has_map)
        dpg.configure_item("cfg_preset_recent_menu", enabled=has_map)

    def _preset_blocked(self) -> bool:
        """Backstop for the menu gating in _sync_preset_controls."""
        if self.runner is None or self.runner.gdf is None:
            self.state.update(status_message="Open a map before applying a preset.")
            return True
        return False

    def _on_apply_preset(self) -> None:
        if not self._preset_blocked():
            self._pick_preset_path(False, self._apply_preset_from)

    def _apply_preset_from(self, path: str) -> None:
        from mosaic.presets import read_preset
        if self._preset_blocked():
            return
        p = Path(path)
        try:
            preset, warnings = read_preset(p)
        except FileNotFoundError:
            self._refresh_recent_presets_menu()
            self.state.update(status_message=f"Preset not found: {p}")
            return
        except Exception as e:
            self.state.update(status_message=f"Could not read preset '{p.name}': {e}")
            return
        for w in warnings:
            log.warning(f"Preset {p.name}: {w}")
        try:
            missing, other = self._apply_preset(preset)
        except ValueError as e:
            self.state.update(status_message=f"Could not apply preset '{p.name}': {e}")
            return
        if warnings:
            other.append(f"{len(warnings)} preset warnings (see log)")
        self._push_recent_preset(str(p))
        self._set_preset_info(missing, other)
        self.state.update(status_message=f"Applied preset '{p.stem}'.")

    def _load_recent_presets(self) -> None:
        import json
        self._recent_presets = []
        if not _RECENT_PRESETS_FILE.exists():
            return
        try:
            data = json.loads(_RECENT_PRESETS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._recent_presets = [x for x in data if isinstance(x, str)]
        except Exception:
            pass

    def _save_recent_presets(self) -> None:
        import json
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            _RECENT_PRESETS_FILE.write_text(
                json.dumps(self._recent_presets[:_RECENT_MAX], indent=2),
                encoding="utf-8")
        except Exception:
            pass

    def _push_recent_preset(self, path: str) -> None:
        self._recent_presets = [path] + [
            x for x in self._recent_presets if x != path][:_RECENT_MAX - 1]
        self._save_recent_presets()
        self._refresh_recent_presets_menu()

    def _refresh_recent_presets_menu(self) -> None:
        """Rebuild Apply Recent Preset; files that are gone are dropped."""
        if not dpg.does_item_exist("cfg_preset_recent_menu"):
            return
        dpg.delete_item("cfg_preset_recent_menu", children_only=True)
        live = [x for x in self._recent_presets if Path(x).exists()]
        if live != self._recent_presets:
            self._recent_presets = live
            self._save_recent_presets()
        if not live:
            dpg.add_menu_item(label="(no recent presets)", enabled=False,
                              parent="cfg_preset_recent_menu")
            return
        for x in live:
            dpg.add_menu_item(label=Path(x).stem, parent="cfg_preset_recent_menu",
                              callback=lambda s, a, u: self._apply_preset_from(u),
                              user_data=x)
