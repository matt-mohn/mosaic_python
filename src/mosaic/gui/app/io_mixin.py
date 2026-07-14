"""Shapefile / hot-start / alignment loading, pickers, and seeding."""
from ._common import (
    AlgorithmRunner,
    AlgorithmStatus,
    Path,
    ShapefileConfig,
    ShapefileInspection,
    dpg,
    log,
    threading,
)


class IOMixin:
    """Shapefile / hot-start / alignment loading, pickers, and seeding."""

    def _on_import_shapefile(self):
        # Platform split: Windows uses a PowerShell OpenFileDialog (tkinter's
        # native picker throws "Catastrophic failure" against DPG's Win32 loop).
        # mac + Linux use DPG's own dialog: tkinter fights DPG for the Cocoa run
        # loop on macOS, and PowerShell doesn't exist on either.
        import os
        if os.name == "nt":
            self._pick_shapefile_tk()
        else:
            self._pick_shapefile_dpg()

    def _pick_shapefile_tk(self):
        # Use PowerShell OpenFileDialog on Windows to avoid Win32 message-loop
        # conflicts between tkinter and DPG that cause repeated dialog opens.
        import subprocess

        from mosaic.paths import mosaic_data_dir, shapefiles_dir
        shp_dir = shapefiles_dir()
        init_dir = str(shp_dir if shp_dir.is_dir() else mosaic_data_dir())
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.OpenFileDialog; "
            "$d.Title = 'Select Shapefile'; "
            "$d.Filter = 'Shapefiles (*.shp)|*.shp|All files (*.*)|*.*'; "
            f"$d.InitialDirectory = '{init_dir}'; "
            "$null = $d.ShowDialog(); "
            "Write-Output $d.FileName"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True, timeout=120,
            )
            path = r.stdout.strip()
        except Exception:
            path = ""
        if path:
            self._on_shapefile_selected(None, {"file_path_name": path})

    def _pick_shapefile_dpg(self):
        if dpg.does_item_exist("__shp_file_dialog"):
            dpg.delete_item("__shp_file_dialog")
        from mosaic.paths import mosaic_data_dir, shapefiles_dir
        shp_dir = shapefiles_dir()
        default_path = str(shp_dir if shp_dir.is_dir() else mosaic_data_dir())
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            modal=True,
            callback=self._on_shapefile_selected,
            cancel_callback=lambda *_: None,
            default_path=default_path,
            width=700,
            height=450,
            tag="__shp_file_dialog",
            label="Select Shapefile",
        ):
            dpg.add_file_extension(".shp", color=(120, 220, 120, 255))
            dpg.add_file_extension(".*")

    # ── Hot start (Advanced menu) ────────────────────────────────────────────────

    def _on_load_hot_start(self):
        log.info("Hot start: menu clicked")
        if self.runner is None or self.runner.gdf is None or self.runner.graph is None:
            self._show_hot_start_error(
                "Load a shapefile before loading a hot start.",
            )
            return
        import sys
        if sys.platform == "darwin":
            self._pick_hot_start_dpg()
        else:
            self._pick_hot_start_tk()

    def _pick_hot_start_tk(self):
        import tkinter as tk
        from tkinter import filedialog

        from mosaic.paths import mosaic_data_dir
        initialdir = str(mosaic_data_dir())
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Select Hot Start CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=initialdir,
            )
            root.destroy()
        except Exception:
            log.exception("Hot start: tk file dialog failed")
            self._show_hot_start_error(
                "File dialog could not be opened. See log for details."
            )
            return
        log.info(f"Hot start: tk file dialog returned path={path!r}")
        if not path:
            return
        self._show_hot_start_column_picker(path)

    def _pick_hot_start_dpg(self):
        if dpg.does_item_exist("__hot_start_file_dialog"):
            dpg.delete_item("__hot_start_file_dialog")
        from mosaic.paths import mosaic_data_dir
        default_path = str(mosaic_data_dir())
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            modal=True,
            callback=lambda s, d: self._show_hot_start_column_picker(
                d.get("file_path_name", "") if isinstance(d, dict) else ""
            ),
            cancel_callback=lambda *_: None,
            default_path=default_path,
            width=700,
            height=450,
            tag="__hot_start_file_dialog",
            label="Select Hot Start CSV",
        ):
            dpg.add_file_extension(".csv", color=(120, 220, 120, 255))
            dpg.add_file_extension(".*")

    def _show_hot_start_column_picker(self, path: str) -> None:
        log.info(f"Hot start: column picker invoked with path={path!r}")
        if not path:
            return
        from mosaic.io.hot_start import HotStartError, read_csv_columns
        try:
            columns = read_csv_columns(path)
        except HotStartError as e:
            self._show_hot_start_error(str(e))
            return
        except Exception as e:
            log.exception("Hot start: read_csv_columns raised unexpectedly")
            self._show_hot_start_error(f"Could not read CSV header: {e}")
            return
        if not columns:
            self._show_hot_start_error("CSV appears to be empty.")
            return

        gdf_id_col = self.runner.id_col_name if self.runner else ""
        default_id = gdf_id_col if gdf_id_col in columns else columns[0]
        default_district = next(
            (c for c in columns if c.lower() == "district"),
            columns[-1] if len(columns) > 1 else columns[0],
        )

        id_combo = district_combo = None
        with self._dialog(
            "Hot Start", "popup_hot_start_picker", (520, 240),
            primary=("Load", lambda: self._apply_hot_start(
                path, dpg.get_value(id_combo), dpg.get_value(district_combo))),
            secondary=("Cancel",
                       lambda: dpg.delete_item("popup_hot_start_picker")),
        ):
            dpg.add_text(f"File: {Path(path).name}", wrap=500)
            dpg.add_text(
                f"Shapefile ID column: {gdf_id_col} "
                f"(CSV values will be matched against these)",
                wrap=500,
            )
            dpg.add_separator()
            id_combo = dpg.add_combo(
                items=columns, default_value=default_id,
                label="Precinct ID column (in CSV)", width=240,
            )
            district_combo = dpg.add_combo(
                items=columns, default_value=default_district,
                label="District column (in CSV)", width=240,
            )

    def _apply_hot_start(
        self, path: str, csv_id_col: str, csv_district_col: str,
    ) -> None:
        log.info(
            f"Hot start: apply path={path!r} id_col={csv_id_col!r} "
            f"district_col={csv_district_col!r}"
        )
        if not path:
            return
        if dpg.does_item_exist("popup_hot_start_picker"):
            dpg.delete_item("popup_hot_start_picker")
        from mosaic.io.hot_start import HotStartError, load_hot_start
        try:
            assignment, info = load_hot_start(
                path,
                gdf=self.runner.gdf,
                gdf_id_col=self.runner.id_col_name,
                csv_id_col=csv_id_col,
                csv_district_col=csv_district_col,
                populations=self.runner.populations,
                graph=self.runner.graph,
                num_districts=dpg.get_value(self._num_districts),
                tolerance=dpg.get_value(self._tolerance) / 100.0,
            )
        except HotStartError as e:
            log.info(f"Hot start: validation failed -- {e}")
            self._show_hot_start_error(str(e))
            return
        except Exception as e:
            log.exception("Unexpected hot start error")
            self._show_hot_start_error(f"Unexpected error: {e}")
            return

        log.info(
            f"Hot start: state.update with assignment shape={assignment.shape} "
            f"file={info['filename']}"
        )
        self.state.update(
            hot_start_assignment=assignment,
            hot_start_filename=info["filename"],
        )
        try:
            self._update_hot_start_display(info)
        except Exception:
            log.exception("Hot start: display update failed (state is loaded)")
            self._show_hot_start_error(
                "Hot start loaded, but the status text could not be updated. "
                "See log for details."
            )

    def _on_clear_hot_start(self):
        self.state.update(
            hot_start_assignment=None,
            hot_start_filename="",
        )
        self._update_hot_start_display(None)

    # ── Relight ──────────────────────────────────────────────────────────────
    # Relight is a "continue refining" mode: each Start reseeds from the current
    # on-screen map (so runs chain, unlike Hot Start which pins a CSV) and a
    # polish preset is applied to the annealing controls. The pre-arm control
    # values are snapshotted so Clear restores them. Mutually exclusive with Hot
    # Start; cleared (with restore) on Reset or when the shapefile or district
    # count changes, since each invalidates the on-screen map it reseeds from.
    _RELIGHT_PRESET = {
        "ann": True, "cool": "Guided (recommended)", "temp": 0.01,
        "guide": 0.80, "watch": False, "n3": 50, "flip_en": True, "flip_mid": 75,
    }

    def _on_relight_toggle(self) -> None:
        if dpg.get_value(self._relight_item):
            self._arm_relight()
        else:
            self._clear_relight()

    def _on_num_districts_change(self, *_args) -> None:
        """Changing the district count invalidates the on-screen map, so a
        Relight armed against it no longer applies -- turn it off (restoring the
        pre-Relight annealing settings). No-op when Relight isn't armed."""
        if self._relight_active:
            self._clear_relight()

    def _relight_snapshot(self) -> dict:
        return {
            "ann":      dpg.get_value(self._ann_enabled),
            "cool":     dpg.get_value(self._cool_mode),
            "temp":     dpg.get_value(self._temp_factor),
            "guide":    dpg.get_value(self._guide_frac),
            "watch":    dpg.get_value(self._launch_watch_enabled),
            "n3":       dpg.get_value(self._n3_pct),
            "flip_en":  dpg.get_value(self._flip_enabled),
            "flip_mid": dpg.get_value(self._flip_midpoint),
        }

    def _relight_write(self, s: dict) -> None:
        """Push a snapshot dict into the annealing controls, then re-run the
        toggle callbacks so the sub-control groups show/hide correctly."""
        dpg.set_value(self._ann_enabled, s["ann"])
        dpg.set_value(self._cool_mode, s["cool"])
        dpg.set_value(self._temp_factor, s["temp"])
        dpg.set_value(self._guide_frac, s["guide"])
        dpg.set_value(self._launch_watch_enabled, s["watch"])
        dpg.set_value(self._n3_pct, s["n3"])
        dpg.set_value(self._flip_enabled, s["flip_en"])
        dpg.set_value(self._flip_midpoint, s["flip_mid"])
        self._on_ann_toggle()
        self._on_cool_mode()
        self._on_launch_watch_toggle()

    def _arm_relight(self) -> None:
        # The menu item is gated, but double-check: needs a map on screen and no
        # Hot Start loaded.
        with self.state._lock:
            has_map = self.state.current_assignment is not None
        real_hot = self.state.get("hot_start_filename")[0] != ""
        if not has_map or real_hot:
            dpg.set_value(self._relight_item, False)   # refuse; snap back
            return
        self._relight_saved = self._relight_snapshot()
        self._relight_write(self._RELIGHT_PRESET)
        # Relight chains short runs; show the full trajectory, not just last 10k.
        dpg.set_value(self._limit_plots, False)
        self._relight_active = True
        dpg.set_value(self._relight_item, True)
        self._update_relight_display()

    def _clear_relight(self) -> None:
        if not self._relight_active and self._relight_saved is None:
            return
        if self._relight_saved is not None:
            self._relight_write(self._relight_saved)
            self._relight_saved = None
        self._relight_active = False
        # Drop the reseed the run path injected, so a later non-Relight run
        # doesn't inherit a stale seed. Only touch the slot when no real Hot Start
        # owns it (the two are mutually exclusive, but guard anyway).
        if self.state.get("hot_start_filename")[0] == "":
            self.state.update(hot_start_assignment=None)
        dpg.set_value(self._relight_item, False)
        self._update_relight_display()

    def _update_relight_display(self) -> None:
        if self._relight_active:
            dpg.set_value(self._relight_info,
                          "RELIGHT: reseeding from the current map each run "
                          "with ultra-low temperature")
            dpg.configure_item(self._relight_info, show=True)
        else:
            dpg.configure_item(self._relight_info, show=False)
            dpg.set_value(self._relight_info, "")

    def _sync_seed_controls(self) -> None:
        """Enable-state for the Relight / Hot Start menu items: mutually
        exclusive, and Relight needs a map with a paused/ended run. Runs every
        frame, so it avoids taking the state lock (a None-check and an enum read)."""
        has_map = self.state.current_assignment is not None
        status = self.state.status
        real_hot = self.state.get("hot_start_filename")[0] != ""
        idle = status in (AlgorithmStatus.IDLE, AlgorithmStatus.PAUSED,
                          AlgorithmStatus.COMPLETED, AlgorithmStatus.ERROR)
        can_arm = has_map and idle and not real_hot
        dpg.configure_item(self._relight_item,
                           enabled=self._relight_active or can_arm)
        dpg.configure_item(self._relight_clear_item, enabled=self._relight_active)
        # Hot Start is locked out while Relight is armed.
        dpg.configure_item(self._hot_start_load_item,
                           enabled=not self._relight_active)

    # ── Alignment reference plan ─────────────────────────────────────────
    # Mirrors the hot-start CSV flow, but loads a *reference* plan for the
    # Alignment score. Deliberately permissive: a reference plan may have a
    # different district count and need not be contiguous/pop-balanced.

    def _on_load_alignment(self):
        if self.runner is None or self.runner.gdf is None:
            self._show_alignment_error(
                "Load a shapefile before loading a reference plan.")
            return
        import sys
        if sys.platform == "darwin":
            self._pick_alignment_dpg()
        else:
            self._pick_alignment_tk()

    def _pick_alignment_tk(self):
        import tkinter as tk
        from tkinter import filedialog

        from mosaic.paths import mosaic_data_dir
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Select Reference Plan CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=str(mosaic_data_dir()),
            )
            root.destroy()
        except Exception:
            log.exception("Alignment: tk file dialog failed")
            self._show_alignment_error(
                "File dialog could not be opened. See log for details.")
            return
        if not path:
            return
        self._show_alignment_column_picker(path)

    def _pick_alignment_dpg(self):
        if dpg.does_item_exist("__alignment_file_dialog"):
            dpg.delete_item("__alignment_file_dialog")
        from mosaic.paths import mosaic_data_dir
        with dpg.file_dialog(
            directory_selector=False, show=True, modal=True,
            callback=lambda s, d: self._show_alignment_column_picker(
                d.get("file_path_name", "") if isinstance(d, dict) else ""),
            cancel_callback=lambda *_: None,
            default_path=str(mosaic_data_dir()),
            width=700, height=450, tag="__alignment_file_dialog",
            label="Select Reference Plan CSV",
        ):
            dpg.add_file_extension(".csv", color=(120, 220, 120, 255))
            dpg.add_file_extension(".*")

    def _show_alignment_column_picker(self, path: str) -> None:
        if not path:
            return
        from mosaic.io.hot_start import HotStartError, read_csv_columns
        try:
            columns = read_csv_columns(path)
        except (HotStartError, Exception) as e:
            self._show_alignment_error(f"Could not read CSV header: {e}")
            return
        if not columns:
            self._show_alignment_error("CSV appears to be empty.")
            return

        gdf_id_col = self.runner.id_col_name if self.runner else ""
        default_id = gdf_id_col if gdf_id_col in columns else columns[0]
        default_district = next(
            (c for c in columns if c.lower() == "district"),
            columns[-1] if len(columns) > 1 else columns[0],
        )
        id_combo = district_combo = None
        with self._dialog(
            "Reference Plan", "popup_alignment_picker", (520, 240),
            primary=("Load", lambda: self._apply_alignment(
                path, dpg.get_value(id_combo), dpg.get_value(district_combo))),
            secondary=("Cancel",
                       lambda: dpg.delete_item("popup_alignment_picker")),
        ):
            dpg.add_text(f"File: {Path(path).name}", wrap=500)
            dpg.add_text(
                f"Shapefile ID column: {gdf_id_col} "
                f"(CSV values will be matched against these)", wrap=500,
            )
            dpg.add_separator()
            id_combo = dpg.add_combo(
                items=columns, default_value=default_id,
                label="Precinct ID column (in CSV)", width=240,
            )
            district_combo = dpg.add_combo(
                items=columns, default_value=default_district,
                label="District column (in CSV)", width=240,
            )

    def _apply_alignment(self, path, csv_id_col, csv_district_col) -> None:
        if not path:
            return
        if dpg.does_item_exist("popup_alignment_picker"):
            dpg.delete_item("popup_alignment_picker")
        from mosaic.scoring.alignment import (
            AlignmentError,
            precompute_alignment_data,
        )
        # Pass election votes (if loaded) so the reference's per-district
        # partisan shares are cached for the "only districts that party wins"
        # restriction. Votes are in gdf row order, matching the reference.
        _dem = _gop = None
        if self.runner.election_arrays:
            _dem, _gop = self.runner.election_arrays[0]
        try:
            data = precompute_alignment_data(
                path, gdf=self.runner.gdf,
                gdf_id_col=self.runner.id_col_name,
                csv_id_col=csv_id_col, csv_district_col=csv_district_col,
                dem_votes=_dem, gop_votes=_gop,
            )
        except AlignmentError as e:
            self._show_alignment_error(str(e))
            return
        except Exception as e:
            log.exception("Unexpected alignment load error")
            self._show_alignment_error(f"Unexpected error: {e}")
            return

        self.runner.alignment_data = data
        n_run = dpg.get_value(self._num_districts)
        note = f", run {n_run}" if data.n_alt_districts != n_run else ""
        dpg.set_value(
            self._alignment_info,
            f"{data.filename} ({data.n_alt_districts} dist{note})",
        )
        self.theme.retoken(self._alignment_info, "accent_green")
        log.info(
            f"Alignment reference set: {data.filename}, "
            f"{data.n_alt_districts} districts"
        )
        if not dpg.get_value(self._alignment_enabled):
            dpg.set_value(self._alignment_enabled, True)
            self._on_alignment_toggle()

    def _on_clear_alignment(self):
        if self.runner is not None:
            self.runner.alignment_data = None
        dpg.set_value(self._alignment_info, "No plan loaded")
        self.theme.retoken(self._alignment_info, "error")

    def _show_alignment_error(self, message: str) -> None:
        # Non-modal (like the hot-start error): it can be raised in the same frame
        # a modal picker was deleted, and a new modal would hide behind DPG's
        # defunct overlay.
        with self._dialog(
            "Alignment Error", "popup_alignment_error", (620, 320),
            modal=False,
            secondary=("Close",
                       lambda: dpg.delete_item("popup_alignment_error")),
        ):
            dpg.add_text(message, wrap=600)

    def _update_hot_start_display(self, info) -> None:
        if info is None:
            dpg.configure_item(self._hot_start_info, show=False)
            dpg.set_value(self._hot_start_info, "")
            dpg.configure_item(self._hot_start_clear_item, enabled=False)
        else:
            msg = (
                f"HOT START: {info['filename']} -- "
                f"{info['n_districts']} districts, "
                f"max dev {info['max_dev_pct']:.2f}%"
            )
            dpg.set_value(self._hot_start_info, msg)
            dpg.configure_item(self._hot_start_info, show=True)
            dpg.configure_item(self._hot_start_clear_item, enabled=True)

    def _show_hot_start_error(self, message: str) -> None:
        log.info(f"Hot start error popup: {message[:200]}")
        # Non-modal on purpose: created in the same frame that just deleted the
        # modal column-picker popup, and a new modal window would hide behind
        # DPG's defunct modal overlay.
        with self._dialog(
            "Hot Start Error", "popup_hot_start_error", (620, 320),
            modal=False,
            secondary=("Close",
                       lambda: dpg.delete_item("popup_hot_start_error")),
        ):
            dpg.add_text(message, wrap=600)
        try:
            dpg.focus_item("popup_hot_start_error")
        except Exception:
            pass

    def _on_shapefile_selected(self, sender, app_data):
        path = app_data.get("file_path_name", "") if isinstance(app_data, dict) else ""
        if not path:
            return
        self.runner = AlgorithmRunner(self.state)
        self._loaded_config = None
        self._has_elections = False
        # Immediately disable election-dependent controls so there's no stale
        # overlay state visible while the new file loads.
        dpg.configure_item(self._partisan_overlay, enabled=False)
        dpg.set_value(self._partisan_overlay, False)
        dpg.configure_item(self._district_partisan, enabled=False)
        dpg.set_value(self._district_partisan, False)
        if self.map_view:
            self.map_view.partisan_overlay = False
            self.map_view.district_partisan_overlay = False
        # Hot start (and Relight) were tied to the previous shapefile's map.
        self.state.update(
            current_assignment=None,
            best_assignment=None,
            initial_assignment=None,
            district_label_map=None,
            hot_start_assignment=None,
            hot_start_filename="",
        )
        self._update_hot_start_display(None)
        self._clear_relight()   # turn off + restore the pre-Relight settings
        dpg.set_value(self._shp_info, "Reading shapefile...")
        self.theme.retoken(self._shp_info, "muted")
        threading.Thread(
            target=self.runner.start_inspection, args=(path,), daemon=True,
        ).start()

    def _on_shp_confirm(self, inspection: ShapefileInspection,
                        config: ShapefileConfig) -> None:
        """Called by ShapefileDialog when the user clicks Confirm and Load."""
        self._loaded_config = config
        self._push_recent_shapefile(inspection.path, config)
        # Flush all history and series before the new load so charts start
        # fresh and the previous file's data doesn't bleed through.
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
            self.state.inversion_history = []
        self._clear_all_series()
        dpg.set_value(self._shp_info, "Building graph...")
        self.theme.retoken(self._shp_info, "muted")
        threading.Thread(
            target=self.runner.complete_load,
            args=(inspection, config),
            daemon=True,
        ).start()

    def _on_shp_cancel(self) -> None:
        """Called by ShapefileDialog when the user clicks Cancel."""
        self.state.update(
            status=AlgorithmStatus.IDLE,
            status_message="Load cancelled",
        )
        dpg.set_value(self._shp_info, "No shapefile loaded")
        self.theme.retoken(self._shp_info, "muted")
