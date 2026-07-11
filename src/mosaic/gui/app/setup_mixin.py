"""Builds the main two-column window layout (one large builder)."""
from ._common import (
    _APP_ICON,
    _ASSETS_DIR,
    _CAMERA_ICON_SIZE,
    _HALF_PLOT_H,
    _HALF_PLOT_W,
    _LEFT_W,
    _MAP_DH,
    _MAP_DW,
    _MAP_H,
    _SCORE_COL_W,
    _TOP_H,
    _VP_H,
    _VP_W,
    MapView,
    Path,
    ShapefileDialog,
    _build_camera_icon,
    _build_more_icon,
    dpg,
    np,
)


class SetupMixin:
    """Builds the main two-column window layout (one large builder)."""

    def setup(self):
        # On Windows the taskbar groups by host process (python.exe) unless an
        # explicit AppUserModelID is set — without this, the taskbar shows the
        # Python logo even when the window's icon is correct.
        import sys as _sys
        if _sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "MattMohn.Mosaic"
                )
            except Exception:
                pass

        dpg.create_context()
        _icon_path = str(_APP_ICON) if _APP_ICON.exists() else ""
        dpg.create_viewport(
            title="Mosaic",
            width=_VP_W, height=_VP_H,
            small_icon=_icon_path,
            large_icon=_icon_path,
        )

        # ── Texture registry ──────────────────────────────────────────────────
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=_MAP_DW, height=_MAP_DH,
                default_value=np.zeros(_MAP_DW * _MAP_DH * 4, dtype=np.float32),
                format=dpg.mvFormat_Float_rgba,
                tag="map_texture",
            )
            dpg.add_raw_texture(
                width=_CAMERA_ICON_SIZE, height=_CAMERA_ICON_SIZE,
                default_value=_build_camera_icon((220, 220, 220)),
                format=dpg.mvFormat_Float_rgba,
                tag="camera_icon_texture",
            )
            dpg.add_raw_texture(
                width=_CAMERA_ICON_SIZE, height=_CAMERA_ICON_SIZE,
                default_value=_build_more_icon((220, 220, 220)),
                format=dpg.mvFormat_Float_rgba,
                tag="more_icon_texture",
            )

        # ── Build palette themes and bind initial ─────────────────────────────
        self.theme.build()
        self.theme.apply(self.theme.palette.name)
        self._sync_map_bg_to_theme()
        self._sync_camera_icon_to_theme()

        # Smaller font for compact toolbars (map overlay checkbox row).
        self._small_font = 0
        _inter_regular = _ASSETS_DIR / "fonts" / "inter" / "Inter-Regular.ttf"
        if _inter_regular.exists():
            with dpg.font_registry():
                self._small_font = dpg.add_font(str(_inter_regular), 13)

        self._shp_dialog = ShapefileDialog(
            confirm_cb=self._on_shp_confirm,
            cancel_cb=self._on_shp_cancel,
            theme=self.theme,
        )
        self._shp_dialog.build(_VP_W, _VP_H)

        self._load_recent_shapefiles()

        self._build_population_popup()
        self._build_seed_popup()
        self._build_new_confirm_popup()
        self._build_close_confirm_popup()
        self._build_advanced_save_popup()
        self._build_opt_popup()
        self._build_partisan_popup()
        self._build_alignment_settings_popup()
        self._build_help_popup()
        self._build_temperature_panel()
        self._build_score_contrib_panel()
        self._build_ref_line_themes()
        self._build_county_splits_panel()
        self._build_partisanship_panel()
        self._build_win_chance_panel()
        self._build_mm_panel()
        self._build_eg_panel()
        self._build_pb_panel()
        self._build_pg_panel()
        self._build_dem_seats_panel()
        self._build_pp_panel()
        self._build_reock_panel()
        self._build_alignment_panel()
        self._build_hc_panel()
        self._build_hsplit_panel()
        self._build_hprop_panel()
        self._build_hcmp_panel()
        self._build_popdev_panel()
        self._build_cut_edges_panel()
        self._build_majority_panel()
        self._build_hinge_panel()
        self._build_district_info_panel()
        self._build_phase_panel()

        # ── Main window ───────────────────────────────────────────────────────
        with dpg.window(tag="main_window", no_scrollbar=True):


            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(
                        label="New", shortcut="Ctrl+N",
                        callback=self._on_new,
                    )
                    dpg.add_menu_item(
                        label="Open...", shortcut="Ctrl+O",
                        callback=self._on_import_shapefile,
                    )
                    with dpg.menu(label="Open Recent",
                                  tag="file_recent_menu"):
                        if self._recent_shapefiles:
                            for _re in self._recent_shapefiles:
                                dpg.add_menu_item(
                                    label=Path(_re["path"]).name,
                                    callback=self._on_open_recent,
                                    user_data=_re,
                                )
                        else:
                            dpg.add_menu_item(
                                label="(no recent files)", enabled=False,
                            )
                    dpg.add_separator()
                    self._file_save_asgn_item = dpg.add_menu_item(
                        label="Save Assignments", shortcut="Ctrl+S",
                        enabled=False,
                        callback=self._on_file_save_assignments,
                    )
                    self._file_save_metrics_item = dpg.add_menu_item(
                        label="Save District Info",
                        enabled=False,
                        callback=self._on_file_save_metrics,
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Open output directory...",
                        callback=self._on_open_output_dir,
                    )
                    dpg.add_menu_item(
                        label="Check for updates...",
                        callback=self._on_check_updates,
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Close", shortcut="Ctrl+W",
                        callback=self._on_close,
                    )

                with dpg.menu(label="Configuration"):
                    dpg.add_menu_item(
                        label="Population...",
                        callback=lambda: dpg.configure_item(
                            "popup_population", show=True),
                    )
                    dpg.add_menu_item(
                        label="Seed...",
                        callback=lambda: dpg.configure_item(
                            "popup_seed", show=True),
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Annealing Settings...",
                        callback=lambda: dpg.configure_item(
                            "popup_opt", show=True),
                    )
                    dpg.add_menu_item(
                        label="Partisanship Settings...",
                        callback=lambda: dpg.configure_item(
                            "popup_partisan", show=True),
                    )
                with dpg.menu(label="Appearance"):
                    dpg.add_text("Theme")
                    self._theme_radio = dpg.add_radio_button(
                        items=["Light", "Dark"],
                        default_value="Light",
                        horizontal=True,
                        callback=self._on_theme_change,
                    )
                    dpg.add_separator()
                    self._limit_plots = dpg.add_checkbox(
                        label="Limit plots to last 10,000 iterations",
                        default_value=True,
                    )
                    dpg.add_separator()
                    dpg.add_text("Map Render Interval:")
                    self._map_interval = dpg.add_slider_float(
                        label="sec",
                        default_value=0.75, min_value=0.25, max_value=30.0,
                        format="%.2f s", width=160,
                        callback=lambda s, d: self.state.update(
                            map_render_interval=d),
                    )
                with dpg.menu(label="Scores", tag="menu_scores"):
                    # Structural
                    self._svis_cuts = dpg.add_menu_item(
                        label="Cut Edges", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_cuts", dpg.get_value(self._svis_cuts),
                            self._cut_enabled, self._on_cut_toggle),
                    )
                    self._svis_hsplit = dpg.add_menu_item(
                        label="County Congruence", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_hsplit", dpg.get_value(self._svis_hsplit),
                            self._hsplit_enabled, self._on_hsplit_toggle),
                    )
                    self._svis_popdev = dpg.add_menu_item(
                        label="Population Deviation", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_popdev", dpg.get_value(self._svis_popdev),
                            self._popdev_enabled, self._on_popdev_score_toggle),
                    )
                    self._svis_alignment = dpg.add_menu_item(
                        label="Alignment", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_alignment", dpg.get_value(self._svis_alignment),
                            self._alignment_enabled, self._on_alignment_toggle),
                    )
                    dpg.add_separator()
                    # Compactness
                    self._svis_hc = dpg.add_menu_item(
                        label="Compactness", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_hc", dpg.get_value(self._svis_hc),
                            self._hc_enabled, self._on_hc_toggle),
                    )
                    self._svis_pp = dpg.add_menu_item(
                        label="Polsby-Popper", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_pp", dpg.get_value(self._svis_pp),
                            self._pp_enabled, self._on_pp_toggle),
                    )
                    self._svis_reock = dpg.add_menu_item(
                        label="Reock", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_reock", dpg.get_value(self._svis_reock),
                            self._reock_enabled, self._on_reock_toggle),
                    )
                    dpg.add_separator()
                    # Partisan
                    self._svis_mm = dpg.add_menu_item(
                        label="Mean-Median", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_mm", dpg.get_value(self._svis_mm),
                            self._mm_enabled, self._on_mm_toggle),
                    )
                    self._svis_eg = dpg.add_menu_item(
                        label="Efficiency Gap", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_eg", dpg.get_value(self._svis_eg),
                            self._eg_enabled, self._on_eg_toggle),
                    )
                    self._svis_pb = dpg.add_menu_item(
                        label="Partisan Bias", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_pb", dpg.get_value(self._svis_pb),
                            self._pb_enabled, self._on_pb_toggle),
                    )
                    self._svis_pg = dpg.add_menu_item(
                        label="Partisan Gini", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_pg", dpg.get_value(self._svis_pg),
                            self._pg_enabled, self._on_pg_toggle),
                    )
                    self._svis_hprop = dpg.add_menu_item(
                        label="Proportionality", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_hprop", dpg.get_value(self._svis_hprop),
                            self._hprop_enabled, self._on_hprop_toggle),
                    )
                    self._svis_hcmp = dpg.add_menu_item(
                        label="Competitiveness", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_hcmp", dpg.get_value(self._svis_hcmp),
                            self._hcmp_enabled, self._on_hcmp_toggle),
                    )
                    self._svis_seats = dpg.add_menu_item(
                        label="Expected Dem Seats", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_seats", dpg.get_value(self._svis_seats),
                            self._seats_enabled, self._on_seats_toggle),
                    )
                    self._svis_majority = dpg.add_menu_item(
                        label="Chance of Majority", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_majority", dpg.get_value(self._svis_majority),
                            self._majority_enabled, self._on_majority_toggle),
                    )
                    self._svis_hinge = dpg.add_menu_item(
                        label="Supermajority/Hinge", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_hinge", dpg.get_value(self._svis_hinge),
                            self._hinge_enabled, self._on_hinge_toggle),
                    )
                    dpg.add_separator()
                    # Classic county splits (optional)
                    self._svis_cs = dpg.add_menu_item(
                        label="Classic Splitting", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_cs", dpg.get_value(self._svis_cs),
                            self._cs_enabled, self._on_cs_toggle),
                    )

                with dpg.menu(label="Views"):
                    # Structural
                    self._panel_cuts_item = dpg.add_menu_item(
                        label="Cut Edges", check=True, default_value=False,
                        callback=self._on_panel_cuts_toggle,
                    )
                    self._panel_hsplit_item = dpg.add_menu_item(
                        label="County Congruence", check=True, default_value=False,
                        callback=self._on_panel_hsplit_toggle,
                    )
                    self._panel_popdev_item = dpg.add_menu_item(
                        label="Population Deviation", check=True, default_value=False,
                        callback=self._on_panel_popdev_toggle,
                    )
                    self._panel_alignment_item = dpg.add_menu_item(
                        label="Alignment", check=True, default_value=False,
                        callback=self._on_panel_alignment_toggle,
                    )
                    dpg.add_separator()
                    # Compactness
                    self._panel_hc_item = dpg.add_menu_item(
                        label="Compactness", check=True, default_value=False,
                        callback=self._on_panel_hc_toggle,
                    )
                    self._panel_pp_item = dpg.add_menu_item(
                        label="Polsby-Popper", check=True, default_value=False,
                        callback=self._on_panel_pp_toggle,
                    )
                    self._panel_reock_item = dpg.add_menu_item(
                        label="Reock", check=True, default_value=False,
                        callback=self._on_panel_reock_toggle,
                    )
                    dpg.add_separator()
                    # Partisan
                    self._panel_mm_item = dpg.add_menu_item(
                        label="Mean-Median", check=True, default_value=False,
                        callback=self._on_panel_mm_toggle,
                    )
                    self._panel_eg_item = dpg.add_menu_item(
                        label="Efficiency Gap", check=True, default_value=False,
                        callback=self._on_panel_eg_toggle,
                    )
                    self._panel_pb_item = dpg.add_menu_item(
                        label="Partisan Bias", check=True, default_value=False,
                        callback=self._on_panel_pb_toggle,
                    )
                    self._panel_pg_item = dpg.add_menu_item(
                        label="Partisan Gini", check=True, default_value=False,
                        callback=self._on_panel_pg_toggle,
                    )
                    self._panel_hprop_item = dpg.add_menu_item(
                        label="Proportionality", check=True, default_value=False,
                        callback=self._on_panel_hprop_toggle,
                    )
                    self._panel_hcmp_item = dpg.add_menu_item(
                        label="Competitiveness", check=True, default_value=False,
                        callback=self._on_panel_hcmp_toggle,
                    )
                    self._panel_seats_item = dpg.add_menu_item(
                        label="Expected Dem Seats", check=True, default_value=False,
                        callback=self._on_panel_seats_toggle,
                    )
                    self._panel_majority_item = dpg.add_menu_item(
                        label="Chance of Majority", check=True, default_value=False,
                        callback=self._on_panel_majority_toggle,
                    )
                    self._panel_hinge_item = dpg.add_menu_item(
                        label="Supermajority/Hinge", check=True, default_value=False,
                        callback=self._on_panel_hinge_toggle,
                    )
                    dpg.add_separator()
                    # Classic county splits (optional)
                    self._panel_cs_item = dpg.add_menu_item(
                        label="Classic Splitting", check=True, default_value=False,
                        callback=self._on_panel_cs_toggle,
                    )
                    dpg.add_separator()
                    # Views-only (no corresponding score)
                    self._panel_temp_item = dpg.add_menu_item(
                        label="Temperature", check=True, default_value=False,
                        callback=self._on_panel_temp_toggle,
                    )
                    self._panel_partisan_item = dpg.add_menu_item(
                        label="Partisanship", check=True, default_value=False,
                        callback=self._on_panel_partisan_toggle,
                    )
                    self._panel_win_chance_item = dpg.add_menu_item(
                        label="Win Chance", check=True, default_value=False,
                        callback=self._on_panel_win_chance_toggle,
                    )
                    self._panel_contrib_item = dpg.add_menu_item(
                        label="Score Contributors", check=True, default_value=False,
                        callback=self._on_panel_contrib_toggle,
                    )
                    self._panel_district_item = dpg.add_menu_item(
                        label="District Info", check=True, default_value=False,
                        callback=self._on_panel_district_toggle,
                    )
                    self._panel_phase_item = dpg.add_menu_item(
                        label="Comet Plot", check=True, default_value=False,
                        callback=self._on_panel_phase_toggle,
                    )

                with dpg.menu(label="Help"):
                    dpg.add_menu_item(
                        label="Open Help...",
                        callback=lambda: dpg.configure_item("popup_help", show=True),
                    )

                with dpg.menu(label="Advanced", tag="menu_debug"):
                    self._hot_start_load_item = dpg.add_menu_item(
                        label="Load Hot Start...",
                        callback=self._on_load_hot_start,
                    )
                    self._tooltip(
                        self._hot_start_load_item,
                        "Start from an existing plan instead of a random seed: "
                        "a CSV with a precinct id and a 1-indexed district "
                        "column, the format Save Assignments exports. District "
                        "count must match Districts.",
                    )
                    self._hot_start_clear_item = dpg.add_menu_item(
                        label="Clear Hot Start",
                        enabled=False,
                        callback=self._on_clear_hot_start,
                    )
                    self._tooltip(
                        self._hot_start_clear_item,
                        "Discard the loaded hot start and go back to a random "
                        "seed on the next run.",
                    )

                    dpg.add_separator()
                    self._relight_item = dpg.add_menu_item(
                        label="Relight",
                        check=True, default_value=False, enabled=False,
                        callback=self._on_relight_toggle,
                    )
                    self._tooltip(
                        self._relight_item,
                        "Continue refining from the current map: each Start "
                        "reseeds from what's on screen (so runs chain), and a "
                        "polish preset is applied (low initial heat, long guide, "
                        "heavy n=3 mix, late flips; Launch Watch off). Needs a "
                        "paused or finished run, and no Hot Start loaded.",
                    )
                    self._relight_clear_item = dpg.add_menu_item(
                        label="Clear Relight",
                        enabled=False,
                        callback=lambda: self._clear_relight(),
                    )
                    self._tooltip(
                        self._relight_clear_item,
                        "Turn Relight off and restore the annealing settings to "
                        "what they were before Relight was armed.",
                    )

                    dpg.add_separator()
                    self._renumber_after_run = dpg.add_menu_item(
                        label="Renumber districts after run",
                        check=True, default_value=True,
                        callback=self._on_renumber_after_run_toggle,
                    )
                    self._tooltip(
                        self._renumber_after_run,
                        "When a run finishes, renumber districts by geography "
                        "(numbers only -- colors stay put). Choose the sweep in "
                        "Renumber options. On by default.",
                    )
                    self._renumber_options_item = dpg.add_menu_item(
                        label="Renumber options...",
                        callback=self._on_open_renumber_options,
                    )
                    self._tooltip(
                        self._renumber_options_item,
                        "Pick how districts are renumbered: None (off), "
                        "Northwest to Southeast, or North to South.",
                    )

            with dpg.child_window(height=_TOP_H, border=False,
                                  no_scrollbar=True):
                with dpg.group(horizontal=True):

                    # ── Left column ───────────────────────────────────────────
                    with dpg.child_window(width=_LEFT_W, border=False):

                        self.theme.text("Load Shapefile", "heading")
                        dpg.add_separator()
                        dpg.add_button(
                            label="Import Shapefile from File",
                            callback=self._on_import_shapefile,
                            width=_LEFT_W - 20,
                        )
                        self._shp_info = self.theme.text(
                            "No shapefile loaded", "muted",
                        )
                        self._hot_start_info = self.theme.text(
                            "", "warning",
                        )
                        dpg.configure_item(self._hot_start_info, show=False)
                        self._relight_info = self.theme.text(
                            "", "warning",
                        )
                        dpg.configure_item(self._relight_info, show=False)

                        dpg.add_spacer(height=6)
                        self.theme.text("Run Parameters", "heading")
                        dpg.add_separator()
                        _inp_w = (_LEFT_W - 24) // 2
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                self.theme.text("Districts", "subheading")
                                self._num_districts = dpg.add_input_int(
                                    label="##dist",
                                    default_value=5, min_value=2, max_value=500,
                                    width=_inp_w, step=0,
                                    callback=self._on_num_districts_change,
                                )
                                self._tooltip(
                                    self._num_districts,
                                    "Number of districts to draw (default 5, "
                                    "range 2 to 500). Match the chamber size "
                                    "you're mapping; a hot start with a "
                                    "different count is rejected.",
                                )
                            with dpg.group():
                                self.theme.text("Iterations", "subheading")
                                self._iterations = dpg.add_input_int(
                                    label="##iter",
                                    default_value=5000, min_value=1,
                                    max_value=1_000_000,
                                    width=_inp_w, step=0,
                                )
                                self._tooltip(
                                    self._iterations,
                                    "How many annealing steps to run (default "
                                    "5000, max 1,000,000). The cooling "
                                    "schedule is sized against this number, so "
                                    "more iterations means slower, more "
                                    "thorough cooling.",
                                )

                        dpg.add_spacer(height=6)
                        self.theme.text("Controls", "heading")
                        dpg.add_separator()
                        _btn_w = (_LEFT_W - 30) // 2
                        with dpg.group(horizontal=True):
                            self._run_btn = dpg.add_button(
                                label="Start", callback=self._on_run,
                                width=_btn_w,
                            )
                            self._pause_btn = dpg.add_button(
                                label="Pause", callback=self._on_pause,
                                width=_btn_w, enabled=False,
                            )
                        with dpg.group(horizontal=True):
                            self._reset_btn = dpg.add_button(
                                label="Reset", callback=self._on_reset,
                                width=_btn_w,
                            )
                            self._revert_btn = dpg.add_button(
                                label="Revert to Best",
                                callback=self._on_revert_to_best,
                                width=_btn_w, enabled=False,
                            )
                        with dpg.group(horizontal=True):
                            self._export_btn = dpg.add_button(
                                label="Quick Save",
                                callback=self._on_export,
                                width=_btn_w, enabled=False,
                            )
                            self._metrics_btn = dpg.add_button(
                                label="Save Metrics",
                                callback=self._on_export_metrics,
                                width=_btn_w, enabled=False,
                            )

                        dpg.add_spacer(height=6)
                        self.theme.text("Status", "heading")
                        dpg.add_separator()
                        self._status_txt = dpg.add_text("Status: Idle")
                        self._iter_txt   = dpg.add_text("Iteration: 0 / 0")
                        # Timer readout as a fixed 3-column table. width=-1 fills
                        # the panel and SizingStretchSame splits it into equal
                        # columns by WIDTH, not content -- so a value gaining a
                        # digit clips inside its own column instead of shoving the
                        # neighbours (flicker-free, no mono font). Thin gray
                        # dividers via borders_innerV (theme table_border_light).
                        with dpg.theme() as _timer_theme:
                            with dpg.theme_component(dpg.mvTable):
                                dpg.add_theme_style(
                                    dpg.mvStyleVar_CellPadding, 9, 3,
                                    category=dpg.mvThemeCat_Core)
                        with dpg.table(
                            header_row=False, borders_innerH=False,
                            borders_outerH=False, borders_innerV=True,
                            borders_outerV=False, width=-1,
                            policy=dpg.mvTable_SizingStretchSame,
                        ) as _timer_table:
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            with dpg.table_row():
                                self._time_txt = dpg.add_text("Time: 0:00")
                                self._ips_txt  = dpg.add_text("Iter/sec: 0.0")
                                self._eta_txt  = dpg.add_text("Est. left: --")
                        dpg.bind_item_theme(_timer_table, _timer_theme)
                        self._progress   = dpg.add_progress_bar(
                            default_value=0.0, width=_LEFT_W - 30,
                        )
                        dpg.add_spacer(height=4)
                        # Two equal columns: score/best/temperature on the left,
                        # entropy/accepted/flip on the right. width=-1 + StretchSame
                        # fixes the divider at the halfway mark regardless of text.
                        # The bound theme widens CellPadding so the thin gray inner
                        # border (borders_innerV, theme table_border_light) has
                        # breathing room before the right-hand column.
                        with dpg.theme() as _stats_theme:
                            with dpg.theme_component(dpg.mvTable):
                                dpg.add_theme_style(
                                    dpg.mvStyleVar_CellPadding, 12, 3,
                                    category=dpg.mvThemeCat_Core)
                        with dpg.table(
                            header_row=False, borders_innerH=False,
                            borders_outerH=False, borders_innerV=True,
                            borders_outerV=False, width=-1,
                            policy=dpg.mvTable_SizingStretchSame,
                        ) as _stats_table:
                            dpg.add_table_column()
                            dpg.add_table_column()
                            with dpg.table_row():
                                self._score_txt = dpg.add_text("Score: --")
                                self._acc_txt   = dpg.add_text("Entropy: --")
                            with dpg.table_row():
                                self._best_txt  = dpg.add_text(
                                    "Best:  --   (iter. --)")
                                self._succ_txt  = dpg.add_text(
                                    "Accepted steps: --")
                            with dpg.table_row():
                                self._temp_txt  = dpg.add_text("Temperature: --")
                                self._flip_txt  = dpg.add_text("Flip rate: 0.0%")
                        dpg.bind_item_theme(_stats_table, _stats_theme)

                    # ── Right column (map + plots) ────────────────────────────
                    with dpg.child_window(width=-1, border=False,
                                          no_scrollbar=True):

                        self.theme.text("District Map", "heading")
                        with dpg.child_window(
                            height=_MAP_H, width=-1, border=True,
                            tag="map_container",
                        ):
                            dpg.add_image("map_texture")
                        with dpg.group(horizontal=True) as overlay_row:
                            self._county_overlay = dpg.add_checkbox(
                                label="County",
                                default_value=False,
                                callback=self._on_county_overlay_toggle,
                            )
                            self._tooltip(
                                self._county_overlay,
                                "Draw county boundaries over the map.",
                            )
                            self._splits_view = dpg.add_checkbox(
                                label="Splits",
                                default_value=False,
                                enabled=False,
                                callback=self._on_splits_view_toggle,
                            )
                            self._tooltip(
                                self._splits_view,
                                "Highlight counties split across districts. "
                                "Needs a county column to be loaded.",
                            )
                            self._partisan_overlay = dpg.add_checkbox(
                                label="Pct. Results",
                                default_value=False,
                                enabled=False,
                                callback=self._on_partisan_overlay_toggle,
                            )
                            self._tooltip(
                                self._partisan_overlay,
                                "Shade each precinct by its vote margin. "
                                "Stays disabled until election data is "
                                "loaded.",
                            )
                            self._district_partisan = dpg.add_checkbox(
                                label="Dist. Results",
                                default_value=False,
                                enabled=False,
                                callback=self._on_district_partisan_toggle,
                            )
                            self._tooltip(
                                self._district_partisan,
                                "Shade each district by its aggregate vote "
                                "margin. Stays disabled until election data "
                                "is loaded.",
                            )
                            self._compactness_view = dpg.add_checkbox(
                                label="Compactness",
                                default_value=False,
                                enabled=False,
                                callback=self._on_compactness_toggle,
                            )
                            self._tooltip(
                                self._compactness_view,
                                "Shade each district by its compactness "
                                "(Polsby-Popper + Reock blend).",
                            )
                            self._pop_dev_view = dpg.add_checkbox(
                                label="Pop. Dev.",
                                default_value=False,
                                enabled=False,
                                callback=self._on_pop_dev_toggle,
                            )
                            self._precinct_overlay = dpg.add_checkbox(
                                label="Precincts",
                                default_value=False,
                                callback=self._on_precinct_overlay_toggle,
                            )
                            self._show_labels = dpg.add_checkbox(
                                label="Labels",
                                default_value=False,
                                callback=self._on_labels_toggle,
                            )
                            # Adjusted by _align_photo_icons() on viewport
                            # resize so the photo buttons sit at the right edge.
                            self._overlay_fill = dpg.add_spacer(width=20)
                            self._cam_btn = cam_btn = dpg.add_image_button(
                                "camera_icon_texture",
                                callback=self._on_save_map,
                                width=20, height=20,
                            )
                            with dpg.tooltip(cam_btn):
                                dpg.add_text("Quick PNG")
                            self._more_btn = more_btn = dpg.add_image_button(
                                "more_icon_texture",
                                callback=self._on_advanced_save_open,
                                width=20, height=20,
                            )
                            with dpg.tooltip(more_btn):
                                dpg.add_text("Photo Menu")
                            self._save_spinner = dpg.add_loading_indicator(
                                style=0, radius=2.0, show=False,
                                color=self.theme.color("body"),
                                secondary_color=self.theme.color("muted"),
                            )
                        if self._small_font:
                            dpg.bind_item_font(overlay_row, self._small_font)

                        dpg.add_spacer(height=4)

                        with dpg.group(horizontal=True):
                            with dpg.group():
                                self._hint(self.theme.text("Score", "heading"), "score")
                                with dpg.plot(height=_HALF_PLOT_H,
                                              width=_HALF_PLOT_W, no_menus=True):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis,
                                                      label="Iteration",
                                                      tag="score_x")
                                    with dpg.plot_axis(dpg.mvYAxis,
                                                       label="Score",
                                                       tag="score_y"):
                                        dpg.add_line_series(
                                            [], [], label="Score",
                                            tag="score_series")
                            with dpg.group():
                                self._hint(self.theme.text("Entropy", "heading"), "entropy")
                                with dpg.plot(height=_HALF_PLOT_H, width=-1, no_menus=True):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis,
                                                      label="Iteration",
                                                      tag="acc_x")
                                    with dpg.plot_axis(dpg.mvYAxis,
                                                       label="Rate (%)",
                                                       tag="acc_y"):
                                        dpg.add_line_series(
                                            [], [], label="Entropy",
                                            tag="acc_series")

            # ── Score panel (bottom) ──────────────────────────────────────────
            with dpg.child_window(height=-1, border=True):
                with dpg.group(horizontal=True):
                    self.theme.text("Full list of scores available in toolbar", "muted")
                    dpg.add_spacer(width=12)
                    dpg.add_button(
                        label="Score Contributors >>",
                        callback=lambda: self._show_panel(
                            "panel_score_contrib", self._panel_contrib_item),
                    )
                dpg.add_separator()
                with dpg.group(horizontal=True):

                    # Col 1: structural metrics
                    with dpg.child_window(width=_SCORE_COL_W, height=-1,
                                          border=False):
                        with dpg.group(tag="score_row_cuts", show=False):
                            with dpg.group(horizontal=True):
                                self._cut_enabled = dpg.add_checkbox(
                                    default_value=False,
                                    callback=self._on_cut_toggle,
                                )
                                self._cut_lbl = self.theme.text(
                                    "Cut Edges", "accent_green",
                                )
                                self._hint(self._cut_lbl, "cut_edges")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_cut_edges", self._panel_cuts_item))
                            with dpg.group(tag="cut_edge_controls", show=False):
                                self._w_cut_edges = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_hsplit", show=True):
                            with dpg.group(horizontal=True):
                                self._hsplit_enabled = dpg.add_checkbox(
                                    default_value=False,
                                    callback=self._on_hsplit_toggle,
                                )
                                self._hsplit_lbl = self.theme.text(
                                    "County Congruence", "disabled",
                                )
                                self._hint(self._hsplit_lbl, "holistic_splitting")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel(
                                        "panel_hsplit", self._panel_hsplit_item))
                            with dpg.group(tag="hsplit_controls", show=False):
                                self._w_holistic_splitting = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                                dpg.add_spacer(height=3)
                                self._county_bias_enabled = dpg.add_checkbox(
                                    label="County-Edge Bias",
                                    default_value=False,
                                    callback=self._on_county_bias_toggle,
                                )
                                self._hint(self._county_bias_enabled, "county_edge_bias")
                                with dpg.group(tag="county_bias_controls", show=False):
                                    self._county_bias = dpg.add_slider_int(
                                        label="Multiplier",
                                        default_value=5, min_value=1, max_value=20,
                                        width=_SCORE_COL_W - 120,
                                    )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_cs", show=False):
                            with dpg.group(horizontal=True):
                                self._cs_enabled = dpg.add_checkbox(
                                    default_value=False,
                                    callback=self._on_cs_toggle,
                                )
                                self._cs_lbl = self.theme.text(
                                    "Classic Splitting", "disabled",
                                )
                                self._hint(self._cs_lbl, "county_splits")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_county_splits", self._panel_cs_item))
                            with dpg.group(tag="cs_controls", show=False):
                                self._w_county_excess = dpg.add_slider_int(
                                    label="Excess weight",
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 175,
                                )
                                self._w_county_unified = dpg.add_slider_int(
                                    label="Single-county weight",
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 175,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_popdev", show=True):
                            with dpg.group(horizontal=True):
                                self._popdev_enabled = dpg.add_checkbox(
                                    default_value=False,
                                    callback=self._on_popdev_score_toggle,
                                )
                                self._popdev_lbl = self.theme.text(
                                    "Population Deviation", "secondary",
                                )
                                self._hint(self._popdev_lbl, "pop_deviation")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_popdev", self._panel_popdev_item))
                                dpg.add_button(label="...", width=24,
                                    callback=lambda: dpg.configure_item(
                                        "popup_population",
                                        show=not dpg.is_item_shown("popup_population")))
                            with dpg.group(tag="popdev_controls", show=False):
                                self._w_pop_deviation = dpg.add_slider_float(
                                    label="Weight",
                                    default_value=1.0, min_value=0.0, max_value=100.0,
                                    format="%.1f", width=_SCORE_COL_W - 100,
                                )

                        with dpg.group(tag="score_row_alignment", show=False):
                            with dpg.group(horizontal=True):
                                self._alignment_enabled = dpg.add_checkbox(
                                    default_value=False,
                                    callback=self._on_alignment_toggle,
                                )
                                self._alignment_lbl = self.theme.text(
                                    "Alignment", "secondary",
                                )
                                self._hint(self._alignment_lbl, "alignment")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel(
                                        "panel_alignment", self._panel_alignment_item))
                                dpg.add_button(label="...", width=24,
                                    callback=lambda: dpg.configure_item(
                                        "popup_alignment_settings",
                                        show=not dpg.is_item_shown("popup_alignment_settings")))
                            with dpg.group(tag="alignment_controls", show=False):
                                # Status line: red until a reference plan is loaded.
                                # Load / Clear and all partisan options live in the
                                # "..." settings popup.
                                self._alignment_info = self.theme.text(
                                    "No plan loaded", "error",
                                )
                                self._w_alignment = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                            dpg.add_spacer(height=4)

                    # Col 2: shape + partisan bias
                    with dpg.child_window(width=_SCORE_COL_W, height=-1,
                                          border=False):
                        with dpg.group(tag="score_row_pp", show=False):
                            with dpg.group(horizontal=True):
                                self._pp_enabled = dpg.add_checkbox(
                                    default_value=False,
                                    callback=self._on_pp_toggle,
                                )
                                self._pp_lbl = self.theme.text(
                                    "Polsby-Popper", "secondary",
                                )
                                self._hint(self._pp_lbl, "compactness")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_pp", self._panel_pp_item))
                            with dpg.group(tag="pp_controls", show=False):
                                self._w_polsby_popper = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_reock", show=False):
                            with dpg.group(horizontal=True):
                                self._reock_enabled = dpg.add_checkbox(
                                    default_value=False,
                                    callback=self._on_reock_toggle,
                                )
                                self._reock_lbl = self.theme.text(
                                    "Reock", "secondary",
                                )
                                self._hint(self._reock_lbl, "reock")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_reock", self._panel_reock_item))
                            with dpg.group(tag="reock_controls", show=False):
                                self._w_reock = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_hc", show=True):
                            with dpg.group(horizontal=True):
                                self._hc_enabled = dpg.add_checkbox(
                                    default_value=True,
                                    callback=self._on_hc_toggle,
                                )
                                self._hc_lbl = self.theme.text(
                                    "Compactness", "accent_green",
                                )
                                self._hint(self._hc_lbl, "holistic_compactness")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_hc", self._panel_hc_item))
                            with dpg.group(tag="hc_controls", show=True):
                                self._w_holistic_compactness = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=50, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_mm", show=False):
                            with dpg.group(horizontal=True):
                                self._mm_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_mm_toggle,
                                )
                                self._mm_lbl = self.theme.text(
                                    "Mean-Median Difference",
                                    "disabled_deep",
                                )
                                self._hint(self._mm_lbl, "mean_median")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_mm", self._panel_mm_item))
                                dpg.add_button(label="...", width=24,
                                    callback=lambda: dpg.configure_item(
                                        "popup_partisan",
                                        show=not dpg.is_item_shown("popup_partisan")))
                            with dpg.group(tag="mm_controls", show=False):
                                self._w_mean_median = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                                self._mm_dir = dpg.add_radio_button(
                                    items=["Fair", "D", "R"],
                                    default_value="Fair", horizontal=True,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_eg", show=True):
                            with dpg.group(horizontal=True):
                                self._eg_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_eg_toggle,
                                )
                                self._eg_lbl = self.theme.text(
                                    "Efficiency Gap",
                                    "disabled_deep",
                                )
                                self._hint(self._eg_lbl, "efficiency_gap")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_eg", self._panel_eg_item))
                                dpg.add_button(label="...", width=24,
                                    callback=lambda: dpg.configure_item(
                                        "popup_partisan",
                                        show=not dpg.is_item_shown("popup_partisan")))
                            with dpg.group(tag="eg_controls", show=False):
                                self._w_efficiency_gap = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                                self._eg_dir = dpg.add_radio_button(
                                    items=["Fair", "D", "R"],
                                    default_value="Fair", horizontal=True,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_pb", show=False):
                            with dpg.group(horizontal=True):
                                self._pb_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_pb_toggle,
                                )
                                self._pb_lbl = self.theme.text(
                                    "Partisan Bias",
                                    "disabled_deep",
                                )
                                self._hint(self._pb_lbl, "partisan_bias")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_pb", self._panel_pb_item))
                                dpg.add_button(label="...", width=24,
                                    callback=lambda: dpg.configure_item(
                                        "popup_partisan",
                                        show=not dpg.is_item_shown("popup_partisan")))
                            with dpg.group(tag="pb_controls", show=False):
                                self._w_partisan_bias = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                                self._pb_dir = dpg.add_radio_button(
                                    items=["Fair", "D", "R"],
                                    default_value="Fair", horizontal=True,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_pg", show=False):
                            with dpg.group(horizontal=True):
                                self._pg_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_pg_toggle,
                                )
                                self._pg_lbl = self.theme.text(
                                    "Partisan Gini",
                                    "disabled_deep",
                                )
                                self._hint(self._pg_lbl, "partisan_gini")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_pg", self._panel_pg_item))
                            with dpg.group(tag="pg_controls", show=False):
                                self._w_partisan_gini = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_hprop", show=False):
                            with dpg.group(horizontal=True):
                                self._hprop_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_hprop_toggle,
                                )
                                self._hprop_lbl = self.theme.text(
                                    "Proportionality",
                                    "disabled_deep",
                                )
                                self._hint(self._hprop_lbl, "holistic_proportionality")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel(
                                        "panel_hprop", self._panel_hprop_item))
                            with dpg.group(tag="hprop_controls", show=False):
                                self._w_holistic_proportionality = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                            dpg.add_spacer(height=4)

                    # Col 3: outcome metrics
                    with dpg.child_window(width=-1, height=-1, border=False):
                        with dpg.group(tag="score_row_hcmp", show=True):
                            with dpg.group(horizontal=True):
                                self._hcmp_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_hcmp_toggle,
                                )
                                self._hcmp_lbl = self.theme.text(
                                    "Competitiveness",
                                    "disabled_deep",
                                )
                                self._hint(self._hcmp_lbl, "holistic_competitiveness")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel(
                                        "panel_hcmp", self._panel_hcmp_item))
                            with dpg.group(tag="hcmp_controls", show=False):
                                self._w_holistic_competitiveness = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_seats", show=False):
                            with dpg.group(horizontal=True):
                                self._seats_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_seats_toggle,
                                )
                                self._seats_lbl = self.theme.text(
                                    "Expected Dem Seats",
                                    "disabled_deep",
                                )
                                self._hint(self._seats_lbl, "dem_seats")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_dem_seats", self._panel_seats_item))
                            with dpg.group(tag="seats_controls", show=False):
                                self._w_dem_seats = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                                self._dem_seats_dir = dpg.add_radio_button(
                                    items=["D", "R"],
                                    default_value="D", horizontal=True,
                                )
                            dpg.add_spacer(height=4)

                        with dpg.group(tag="score_row_majority", show=False):
                            with dpg.group(horizontal=True):
                                self._majority_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_majority_toggle,
                                )
                                self._majority_lbl = self.theme.text(
                                    "Chance of Majority",
                                    "disabled_deep",
                                )
                                self._hint(self._majority_lbl, "majority_chance")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_majority", self._panel_majority_item))
                            with dpg.group(tag="majority_controls", show=False):
                                self._w_majority = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                                with dpg.group(horizontal=True):
                                    self._majority_dem_chk = dpg.add_checkbox(
                                        label="D", default_value=True,
                                        callback=self._on_majority_dem_chk,
                                    )
                                    dpg.add_spacer(width=12)
                                    self._majority_rep_chk = dpg.add_checkbox(
                                        label="R", default_value=False,
                                        callback=self._on_majority_rep_chk,
                                    )

                        with dpg.group(tag="score_row_hinge", show=False):
                            with dpg.group(horizontal=True):
                                self._hinge_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_hinge_toggle,
                                )
                                self._hinge_lbl = self.theme.text(
                                    "Supermajority/Hinge",
                                    "disabled_deep",
                                )
                                self._hint(self._hinge_lbl, "hinge")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_hinge", self._panel_hinge_item))
                            with dpg.group(tag="hinge_controls", show=False):
                                self._w_hinge = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                                self._hinge_threshold = dpg.add_slider_int(
                                    label="Threshold",
                                    default_value=8, min_value=1, max_value=14,
                                    width=_SCORE_COL_W - 100,
                                )
                                with dpg.group(horizontal=True):
                                    self._hinge_dem_chk = dpg.add_checkbox(
                                        label="D", default_value=True,
                                        callback=self._on_hinge_dem_chk,
                                    )
                                    dpg.add_spacer(width=12)
                                    self._hinge_rep_chk = dpg.add_checkbox(
                                        label="R", default_value=False,
                                        callback=self._on_hinge_rep_chk,
                                    )

        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()

        self.map_view = MapView("map_texture", _MAP_DW, _MAP_DH)
        # The earlier _sync_map_bg_to_theme() at __init__ time ran before
        # MapView existed, so MapView._bg_color is still the module default.
        # Re-sync now that the view is constructed, otherwise the first
        # shapefile load builds its LUT against a stale dark background.
        self._sync_map_bg_to_theme()

    # ── Popup builders ────────────────────────────────────────────────────────

    # ── Shared dialog scaffolding ─────────────────────────────────────────────
