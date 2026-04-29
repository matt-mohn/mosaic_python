"""Main Dear PyGui application -- two-column layout with live district map."""

import threading
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg
import numpy as np

from mosaic.gui.state import SharedState, AlgorithmStatus
from mosaic.gui.runner import AlgorithmRunner
from mosaic.gui.map_view import MapView
from mosaic.gui.shp_dialog import ShapefileDialog
from mosaic.io.inspect import ShapefileConfig, ShapefileInspection
from mosaic.scoring.score import ScoreConfig
from mosaic.recom.annealing import AnnealingConfig

_PLOT_LIMIT   = 10_000   # max points rendered when limit-plots is on
_COMPACT_AT   = 20_000   # compact local buffer when it exceeds this
_COMPACT_KEEP = 10_000   # keep last N at full resolution after compaction
_COMPACT_THIN = 5        # keep 1-in-N for the older portion


class _SeriesBuffer:
    """Incremental, self-compacting local history for one plot series.

    The GUI copies only the *delta* since the last frame under the lock
    (~IPS/fps items, not the full history), keeping lock hold time constant
    regardless of total run length.  When the buffer exceeds _COMPACT_AT
    entries, everything older than _COMPACT_KEEP is thinned to 1-in-_COMPACT_THIN.
    """

    __slots__ = ("xs", "ys", "read")

    def __init__(self):
        self.xs: list = []   # iteration indices (survive thinning intact)
        self.ys: list = []   # values
        self.read: int = 0   # items consumed from SharedState list so far

    def add(self, new_ys: list, *, scale: float = 1.0) -> None:
        """Append plain-value delta (already copied outside the lock)."""
        if not new_ys:
            return
        base = self.read
        n = len(new_ys)
        self.xs.extend(range(base, base + n))
        self.ys.extend(new_ys if scale == 1.0 else (v * scale for v in new_ys))
        self.read += n
        self._maybe_compact()

    def add_pairs(self, new_pairs: list, *, scale: float = 1.0) -> None:
        """Append [iter, value] pairs (acceptance_rate_history format)."""
        if not new_pairs:
            return
        self.read += len(new_pairs)
        self.xs.extend(int(p[0]) for p in new_pairs)
        self.ys.extend(p[1] * scale for p in new_pairs)
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        if len(self.ys) <= _COMPACT_AT:
            return
        cut = len(self.ys) - _COMPACT_KEEP
        self.xs = self.xs[:cut:_COMPACT_THIN] + self.xs[cut:]
        self.ys = self.ys[:cut:_COMPACT_THIN] + self.ys[cut:]

    def plot_data(self, limit: bool) -> tuple:
        if limit and len(self.ys) > _PLOT_LIMIT:
            return self.xs[-_PLOT_LIMIT:], self.ys[-_PLOT_LIMIT:]
        return self.xs, self.ys

    def trim_to(self, max_iter: int, new_read: int) -> None:
        """Drop everything past max_iter; update read pointer (Revert to Best)."""
        import bisect
        idx = bisect.bisect_right(self.xs, max_iter)
        del self.xs[idx:]
        del self.ys[idx:]
        self.read = new_read

    def clear(self) -> None:
        self.xs.clear()
        self.ys.clear()
        self.read = 0


# Score Contributor panel — alphabetical metric rows
_CONTRIB_METRICS = [
    ("Competitiveness", "contrib_competitiveness"),
    ("County Splits",   "contrib_county_splits"),
    ("Cut Edges",       "contrib_cut_edges"),
    ("Dem Seats",       "contrib_dem_seats"),
    ("Efficiency Gap",  "contrib_efficiency_gap"),
    ("Mean-Median",     "contrib_mean_median"),
    ("Polsby-Popper",   "contrib_polsby_popper"),
]

# ── Layout constants ──────────────────────────────────────────────────────────
_VP_W         = 1340
_VP_H         = 880
_LEFT_W       = 440    # fixed left-column width
_TOP_H        = 640    # fixed top section height (left + right columns)
_SCORE_H      = 250    # pinned bottom score panel height
_SCORE_COL_W  = (_VP_W - 40) // 3   # ~433px per score column
_MAP_H        = 390    # map panel height (child_window)
_MAP_DW       = _VP_W - _LEFT_W - 32   # texture pixel width  (~868)
_MAP_DH       = _MAP_H - 22            # texture pixel height (~368)
_PLOT_H       = 155    # single-plot height (score row)
_HALF_PLOT_H  = 150    # height of the two side-by-side plots
_HALF_PLOT_W  = (_MAP_DW - 10) // 2   # width of each half-plot (~429)


class MosaicApp:
    """Main application -- Dear PyGui interface coordinating the algorithm thread."""

    def __init__(self):
        self.state = SharedState()
        self.runner: Optional[AlgorithmRunner] = None
        self.algorithm_thread: Optional[threading.Thread] = None
        self.map_view: Optional[MapView] = None
        self._shp_dialog: Optional[ShapefileDialog] = None

        # Stored after the user confirms the shapefile dialog
        self._loaded_config: Optional[ShapefileConfig] = None

        # Map background-load tracking (app-local, no SharedState)
        self._map_loading: bool = False
        self._map_ready: bool = False
        self._map_loaded_path: str = ""

        # Plot appearance toggle (app-local)
        self._limit_plots: int | str = ""   # DPG checkbox tag, set during setup

        # Score Contributor panel tracking
        self._last_contrib_iter: int = -1

        # Track what data the current shapefile has
        self._has_elections: bool = False
        self._has_county: bool = False

        # Local history buffers — incremental delta copy, self-compacting
        self._buf_score     = _SeriesBuffer()
        self._buf_acc       = _SeriesBuffer()
        self._buf_temp      = _SeriesBuffer()
        self._buf_cs_score  = _SeriesBuffer()
        self._buf_cs_excess = _SeriesBuffer()
        self._buf_cs_clean  = _SeriesBuffer()
        self._buf_mm        = _SeriesBuffer()
        self._buf_eg        = _SeriesBuffer()
        self._buf_seats     = _SeriesBuffer()
        self._buf_comp      = _SeriesBuffer()
        self._buf_pp        = _SeriesBuffer()
        self._buf_cuts      = _SeriesBuffer()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(
            title="MosaicPy Demo",
            width=_VP_W, height=_VP_H,
        )

        # ── Texture registry ──────────────────────────────────────────────────
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=_MAP_DW, height=_MAP_DH,
                default_value=np.zeros(_MAP_DW * _MAP_DH * 4, dtype=np.float32),
                format=dpg.mvFormat_Float_rgba,
                tag="map_texture",
            )

        # ── Popup windows (built outside main window context) ─────────────────
        # Nudge theme: lighter grey + white text — recommended primary action
        with dpg.theme() as self._nudge_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,       (95,  100, 115, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (130, 138, 158, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (165, 172, 198, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          (255, 255, 255, 255))

        # Anti-nudge theme: dark grey + greyed text — available but not primary
        with dpg.theme() as self._antinudge_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,       (38,  38,  42,  255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (52,  52,  58,  255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (65,  65,  72,  255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          (115, 115, 125, 255))

        self._shp_dialog = ShapefileDialog(
            confirm_cb=self._on_shp_confirm,
            cancel_cb=self._on_shp_cancel,
        )
        self._shp_dialog.build(_VP_W, _VP_H)

        self._build_population_popup()
        self._build_seed_popup()
        self._build_opt_popup()
        self._build_partisan_popup()
        self._build_help_popup()
        self._build_temperature_panel()
        self._build_score_contrib_panel()
        self._build_county_splits_panel()
        self._build_partisanship_panel()
        self._build_mm_panel()
        self._build_eg_panel()
        self._build_dem_seats_panel()
        self._build_comp_panel()
        self._build_pp_panel()
        self._build_cut_edges_panel()

        # ── Main window ───────────────────────────────────────────────────────
        with dpg.window(tag="main_window", no_scrollbar=True):

            with dpg.menu_bar():
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
                    self._limit_plots = dpg.add_checkbox(
                        label="Limit plots to last 10,000 iterations",
                        default_value=True,
                    )
                with dpg.menu(label="Panels"):
                    self._panel_temp_item = dpg.add_menu_item(
                        label="Temperature",
                        check=True,
                        default_value=False,
                        callback=self._on_panel_temp_toggle,
                    )
                    self._panel_cs_item = dpg.add_menu_item(
                        label="County Splits", check=True, default_value=False,
                        callback=self._on_panel_cs_toggle,
                    )
                    self._panel_partisan_item = dpg.add_menu_item(
                        label="Partisanship", check=True, default_value=False,
                        callback=self._on_panel_partisan_toggle,
                    )
                    self._panel_mm_item = dpg.add_menu_item(
                        label="Mean-Median", check=True, default_value=False,
                        callback=self._on_panel_mm_toggle,
                    )
                    self._panel_eg_item = dpg.add_menu_item(
                        label="Efficiency Gap", check=True, default_value=False,
                        callback=self._on_panel_eg_toggle,
                    )
                    self._panel_seats_item = dpg.add_menu_item(
                        label="Expected Dem Seats", check=True, default_value=False,
                        callback=self._on_panel_seats_toggle,
                    )
                    self._panel_comp_item = dpg.add_menu_item(
                        label="Competitiveness", check=True, default_value=False,
                        callback=self._on_panel_comp_toggle,
                    )
                    self._panel_pp_item = dpg.add_menu_item(
                        label="Polsby-Popper", check=True, default_value=False,
                        callback=self._on_panel_pp_toggle,
                    )
                    self._panel_cuts_item = dpg.add_menu_item(
                        label="Cut Edges", check=True, default_value=False,
                        callback=self._on_panel_cuts_toggle,
                    )
                    dpg.add_separator()
                    self._panel_contrib_item = dpg.add_menu_item(
                        label="Score Contributors", check=True, default_value=False,
                        callback=self._on_panel_contrib_toggle,
                    )

                with dpg.menu(label="Help"):
                    dpg.add_menu_item(
                        label="Open Help...",
                        callback=lambda: dpg.configure_item("popup_help", show=True),
                    )

                with dpg.menu(label="Map"):
                    dpg.add_text("Render interval:")
                    self._map_interval = dpg.add_slider_float(
                        label="sec",
                        default_value=0.75, min_value=0.25, max_value=30.0,
                        format="%.2f s", width=160,
                        callback=lambda s, d: self.state.update(
                            map_render_interval=d),
                    )

            with dpg.child_window(height=_TOP_H, border=False,
                                  no_scrollbar=True):
                with dpg.group(horizontal=True):

                    # ── Left column ───────────────────────────────────────────
                    with dpg.child_window(width=_LEFT_W, border=False):

                        dpg.add_text("MosaicPy Demo", color=(255, 200, 60))
                        dpg.add_separator()
                        dpg.add_spacer(height=6)

                        dpg.add_text("Load Shapefile", color=(200, 200, 100))
                        dpg.add_separator()
                        dpg.add_button(
                            label="Import Shapefile from File",
                            callback=self._on_import_shapefile,
                            width=_LEFT_W - 20,
                        )
                        self._shp_info = dpg.add_text(
                            "No shapefile loaded", color=(150, 150, 150),
                        )

                        dpg.add_spacer(height=8)
                        dpg.add_text("Run Parameters", color=(200, 200, 100))
                        dpg.add_separator()
                        _inp_w = (_LEFT_W - 24) // 2
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text("Districts",
                                             color=(170, 170, 170))
                                self._num_districts = dpg.add_input_int(
                                    label="##dist",
                                    default_value=5, min_value=2, max_value=500,
                                    width=_inp_w, step=0,
                                )
                            with dpg.group():
                                dpg.add_text("Iterations",
                                             color=(170, 170, 170))
                                self._iterations = dpg.add_input_int(
                                    label="##iter",
                                    default_value=2500, min_value=1,
                                    max_value=1_000_000,
                                    width=_inp_w, step=0,
                                )

                        dpg.add_spacer(height=8)
                        dpg.add_text("Controls", color=(200, 200, 100))
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
                                label="Save Assignments",
                                callback=self._on_export,
                                width=_btn_w, enabled=False,
                            )
                            self._metrics_btn = dpg.add_button(
                                label="Save Metrics",
                                callback=self._on_export_metrics,
                                width=_btn_w, enabled=False,
                            )

                        dpg.add_spacer(height=8)
                        dpg.add_text("Status", color=(200, 200, 100))
                        dpg.add_separator()
                        self._status_txt = dpg.add_text("Status: Idle")
                        self._iter_txt   = dpg.add_text("Iteration: 0 / 0")
                        self._timer_txt  = dpg.add_text(
                            "Time: 0:00  |  0 iter/sec")
                        self._progress   = dpg.add_progress_bar(
                            default_value=0.0, width=_LEFT_W - 30,
                        )
                        dpg.add_spacer(height=4)
                        self._score_txt = dpg.add_text("Score: --")
                        self._best_txt  = dpg.add_text(
                            "Best:  --   (iteration --)")
                        self._temp_txt  = dpg.add_text("Temperature: --")
                        self._acc_txt   = dpg.add_text("Entropy: --")
                        self._succ_txt  = dpg.add_text("Accepted steps: --")

                    # ── Right column (map + plots) ────────────────────────────
                    with dpg.child_window(width=-1, border=False):

                        dpg.add_text("District Map", color=(200, 200, 100))
                        with dpg.child_window(
                            height=_MAP_H, width=-1, border=True,
                            tag="map_container",
                        ):
                            dpg.add_image("map_texture")
                        with dpg.group(horizontal=True):
                            self._county_overlay = dpg.add_checkbox(
                                label="County",
                                default_value=False,
                                callback=self._on_county_overlay_toggle,
                            )
                            self._splits_view = dpg.add_checkbox(
                                label="Splits",
                                default_value=False,
                                enabled=False,
                                callback=self._on_splits_view_toggle,
                            )
                            self._partisan_overlay = dpg.add_checkbox(
                                label="Precinct Results",
                                default_value=False,
                                enabled=False,
                                callback=self._on_partisan_overlay_toggle,
                            )
                            self._district_partisan = dpg.add_checkbox(
                                label="District Results",
                                default_value=False,
                                enabled=False,
                                callback=self._on_district_partisan_toggle,
                            )
                            self._compactness_view = dpg.add_checkbox(
                                label="Compactness",
                                default_value=False,
                                enabled=False,
                                callback=self._on_compactness_toggle,
                            )
                            self._pop_dev_view = dpg.add_checkbox(
                                label="Pop. Deviation",
                                default_value=False,
                                enabled=False,
                                callback=self._on_pop_dev_toggle,
                            )

                        dpg.add_spacer(height=4)

                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text("Score History",
                                             color=(200, 200, 100))
                                with dpg.plot(height=_HALF_PLOT_H,
                                              width=_HALF_PLOT_W):
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
                                dpg.add_text("Entropy",
                                             color=(200, 200, 100))
                                with dpg.plot(height=_HALF_PLOT_H, width=-1):
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

                    # Col 1: structural metrics
                    with dpg.child_window(width=_SCORE_COL_W, height=-1,
                                          border=False):
                        # Cut Edges
                        with dpg.group(horizontal=True):
                            self._cut_enabled = dpg.add_checkbox(
                                default_value=True,
                                callback=self._on_cut_toggle,
                            )
                            self._cut_lbl = dpg.add_text(
                                "Cut Edges", color=(90, 220, 90),
                            )
                        with dpg.group(tag="cut_edge_controls"):
                            self._w_cut_edges = dpg.add_slider_int(
                                label="Weight",
                                default_value=1, min_value=0, max_value=100,
                                width=_SCORE_COL_W - 100,
                            )

                        dpg.add_spacer(height=4)

                        # County Splits + Bias
                        with dpg.group(horizontal=True):
                            self._cs_enabled = dpg.add_checkbox(
                                default_value=False,
                                callback=self._on_cs_toggle,
                            )
                            self._cs_lbl = dpg.add_text(
                                "County Splits and Bias", color=(110, 110, 110),
                            )
                        with dpg.group(tag="cs_controls", show=False):
                            self._w_county_splits = dpg.add_slider_int(
                                label="Weight",
                                default_value=1, min_value=0, max_value=100,
                                width=_SCORE_COL_W - 100,
                            )
                            dpg.add_spacer(height=3)
                            self._county_bias_enabled = dpg.add_checkbox(
                                label="County-Edge Bias",
                                default_value=False,
                                callback=self._on_county_bias_toggle,
                            )
                            with dpg.group(tag="county_bias_controls", show=False):
                                self._county_bias = dpg.add_slider_int(
                                    label="Multiplier",
                                    default_value=5, min_value=1, max_value=20,
                                    width=_SCORE_COL_W - 120,
                                )
                                dpg.add_text(
                                    "  cross-county edges less likely cut",
                                    color=(150, 150, 150),
                                )

                    # Col 2: shape + partisan bias
                    with dpg.child_window(width=_SCORE_COL_W, height=-1,
                                          border=False):
                        # Polsby-Popper
                        with dpg.group(horizontal=True):
                            self._pp_enabled = dpg.add_checkbox(
                                default_value=False,
                                callback=self._on_pp_toggle,
                            )
                            self._pp_lbl = dpg.add_text(
                                "Polsby-Popper", color=(180, 180, 180),
                            )
                        with dpg.group(tag="pp_controls", show=False):
                            self._w_polsby_popper = dpg.add_slider_int(
                                label="Weight",
                                default_value=1, min_value=0, max_value=100,
                                width=_SCORE_COL_W - 100,
                            )

                        dpg.add_spacer(height=4)

                        # Mean-Median Difference
                        with dpg.group(horizontal=True):
                            self._mm_enabled = dpg.add_checkbox(
                                default_value=False, enabled=False,
                                callback=self._on_mm_toggle,
                            )
                            self._mm_lbl = dpg.add_text(
                                "Mean-Median Difference",
                                color=(90, 90, 90),
                            )
                        with dpg.group(tag="mm_controls", show=False):
                            self._w_mean_median = dpg.add_slider_int(
                                label="Weight",
                                default_value=1, min_value=0, max_value=100,
                                width=_SCORE_COL_W - 100,
                            )
                            self._target_mean_median = dpg.add_slider_float(
                                label="Target MM",
                                default_value=0.0, min_value=-0.15, max_value=0.15,
                                format="%.3f", width=_SCORE_COL_W - 100,
                            )
                            dpg.add_text(
                                "  - = D advantage  |  + = R advantage",
                                color=(110, 110, 110),
                            )

                        dpg.add_spacer(height=4)

                        # Efficiency Gap
                        with dpg.group(horizontal=True):
                            self._eg_enabled = dpg.add_checkbox(
                                default_value=False, enabled=False,
                                callback=self._on_eg_toggle,
                            )
                            self._eg_lbl = dpg.add_text(
                                "Efficiency Gap",
                                color=(90, 90, 90),
                            )
                        with dpg.group(tag="eg_controls", show=False):
                            self._w_efficiency_gap = dpg.add_slider_int(
                                label="Weight",
                                default_value=1, min_value=0, max_value=100,
                                width=_SCORE_COL_W - 100,
                            )
                            self._target_efficiency_gap = dpg.add_slider_float(
                                label="Target EG",
                                default_value=0.0, min_value=-0.35, max_value=0.35,
                                format="%.3f", width=_SCORE_COL_W - 100,
                            )
                            dpg.add_text(
                                "  - = D bias  |  + = R bias",
                                color=(110, 110, 110),
                            )

                    # Col 3: outcome metrics
                    with dpg.child_window(width=-1, height=-1, border=False):
                        # Competitiveness
                        with dpg.group(horizontal=True):
                            self._comp_enabled = dpg.add_checkbox(
                                default_value=False, enabled=False,
                                callback=self._on_comp_toggle,
                            )
                            self._comp_lbl = dpg.add_text(
                                "Competitiveness",
                                color=(90, 90, 90),
                            )
                        with dpg.group(tag="comp_controls", show=False):
                            self._w_competitiveness = dpg.add_slider_int(
                                label="Weight",
                                default_value=1, min_value=0, max_value=100,
                                width=_SCORE_COL_W - 100,
                            )

                        dpg.add_spacer(height=4)

                        # Expected Dem Seats
                        with dpg.group(horizontal=True):
                            self._seats_enabled = dpg.add_checkbox(
                                default_value=False, enabled=False,
                                callback=self._on_seats_toggle,
                            )
                            self._seats_lbl = dpg.add_text(
                                "Expected Dem Seats",
                                color=(90, 90, 90),
                            )
                        with dpg.group(tag="seats_controls", show=False):
                            self._w_dem_seats = dpg.add_slider_int(
                                label="Weight",
                                default_value=1, min_value=0, max_value=100,
                                width=_SCORE_COL_W - 100,
                            )
                            self._target_dem_seats = dpg.add_slider_int(
                                label="Target Seats",
                                default_value=7, min_value=1, max_value=14,
                                width=_SCORE_COL_W - 100,
                            )

        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()

        self.map_view = MapView("map_texture", _MAP_DW, _MAP_DH)

    # ── Popup builders ────────────────────────────────────────────────────────

    def _build_population_popup(self):
        with dpg.window(
            label="Run Config -- Population",
            tag="popup_population", show=False,
            modal=True, no_close=True,
            width=420, height=150,
            pos=[(_VP_W - 420) // 2, (_VP_H - 150) // 2],
        ):
            self._tolerance = dpg.add_slider_float(
                label="Population Tolerance",
                default_value=0.05, min_value=0.001, max_value=0.5,
                format="%.3f", width=260,
            )
            dpg.add_text("  (e.g. 0.05 = 5% max deviation)",
                         color=(150, 150, 150))
            dpg.add_spacer(height=8)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item("popup_population",
                                                    show=False),
                width=80,
            )

    def _build_seed_popup(self):
        with dpg.window(
            label="Run Config -- Seed",
            tag="popup_seed", show=False,
            modal=True, no_close=True,
            width=340, height=120,
            pos=[(_VP_W - 340) // 2, (_VP_H - 120) // 2],
        ):
            self._seed = dpg.add_input_int(
                label="Random Seed  (0 = random)",
                default_value=0, min_value=0, width=120,
            )
            dpg.add_spacer(height=8)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item("popup_seed", show=False),
                width=80,
            )

    def _build_opt_popup(self):
        with dpg.window(
            label="Optimization -- Annealing Settings",
            tag="popup_opt", show=False,
            modal=True, no_close=True,
            width=460, height=340,
            pos=[(_VP_W - 460) // 2, (_VP_H - 340) // 2],
        ):
            self._ann_enabled = dpg.add_checkbox(
                label="Enable simulated annealing", default_value=True,
                callback=self._on_ann_toggle,
            )
            dpg.add_separator()

            with dpg.group(tag="ann_body"):
                self._temp_factor = dpg.add_slider_float(
                    label="Initial Temp Factor",
                    default_value=0.2, min_value=0.01, max_value=2.0,
                    format="%.3f", width=260,
                )
                dpg.add_text("  initial_temp = factor x initial_score",
                             color=(150, 150, 150))
                dpg.add_spacer(height=6)

                dpg.add_text("Cooling mode:")
                self._cool_mode = dpg.add_radio_button(
                    items=["Guided (recommended)", "Static"],
                    default_value="Guided (recommended)",
                    callback=self._on_cool_mode,
                    horizontal=True,
                )
                dpg.add_spacer(height=4)

                with dpg.group(tag="guided_controls"):
                    self._guide_frac = dpg.add_slider_float(
                        label="Guide Point",
                        default_value=0.9, min_value=0.5, max_value=1.0,
                        format="%.2f", width=200,
                    )
                    dpg.add_text("  fraction of iterations to cool over",
                                 color=(150, 150, 150))
                    self._target_temp = dpg.add_input_float(
                        label="Target Temp",
                        default_value=1.0, min_value=0.001,
                        format="%.3f", width=120,
                    )
                    dpg.add_text(
                        "  temperature at the guide point (absolute)",
                        color=(150, 150, 150),
                    )

                with dpg.group(tag="static_controls", show=False):
                    self._cooling_rate = dpg.add_slider_float(
                        label="Cooling Rate / iteration",
                        default_value=0.9995, min_value=0.990,
                        max_value=0.99999, format="%.5f", width=260,
                    )

            dpg.add_spacer(height=8)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item("popup_opt", show=False),
                width=80,
            )

    def _build_partisan_popup(self):
        with dpg.window(
            label="Optimization -- Partisanship Settings",
            tag="popup_partisan", show=False,
            modal=True, no_close=True,
            width=460, height=220,
            pos=[(_VP_W - 460) // 2, (_VP_H - 220) // 2],
        ):
            dpg.add_text(
                "Applied when partisan metrics are enabled.",
                color=(150, 150, 150),
            )
            dpg.add_separator()
            dpg.add_spacer(height=6)

            self._win_prob = dpg.add_slider_float(
                label="Win Prob at 55% vote share",
                default_value=0.9, min_value=0.51, max_value=0.999,
                format="%.3f", width=220,
            )
            dpg.add_text(
                "  P(D wins district | D has 55% of two-party vote)",
                color=(150, 150, 150),
            )
            dpg.add_spacer(height=8)

            dpg.add_text("Efficiency Gap mode:")
            self._eg_mode = dpg.add_radio_button(
                items=["Robust (recommended)", "Static"],
                default_value="Robust (recommended)",
                horizontal=True,
            )
            dpg.add_text(
                "  Robust efficiency gap weights efficiency gap by multiple swings from the baseline election provided",
                color=(150, 150, 150),
            )
            dpg.add_spacer(height=8)

            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item("popup_partisan", show=False),
                width=80,
            )

    def _build_temperature_panel(self):
        with dpg.window(
            label="Temperature",
            tag="panel_temperature",
            show=False,
            width=520, height=280,
            pos=[_LEFT_W + 60, 80],
            on_close=self._on_temp_panel_close,
        ):
            with dpg.plot(height=-1, width=-1, tag="panel_temp_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(
                    dpg.mvXAxis, label="Iteration", tag="panel_temp_x",
                )
                with dpg.plot_axis(
                    dpg.mvYAxis, label="Temp", tag="panel_temp_y",
                ):
                    dpg.add_line_series([], [], label="Temp",
                                        tag="panel_temp_series")

    def _build_county_splits_panel(self):
        with dpg.window(
            label="County Splits", tag="panel_county_splits",
            show=False, width=540, height=420,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_cs_item, False),
        ):
            with dpg.group(tag="cs_charts_grp"):
                dpg.add_text("County Splits Score", color=(200, 200, 100))
                with dpg.plot(height=140, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_score_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Score", tag="cs_score_y"):
                        dpg.add_line_series([], [], label="CS Score", tag="cs_score_series")
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("Excess Splits", color=(200, 200, 100))
                        with dpg.plot(height=150, width=248):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_excess_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="cs_excess_y"):
                                dpg.add_line_series([], [], label="Excess", tag="cs_excess_series")
                    with dpg.group():
                        dpg.add_text("Single-County Districts", color=(200, 200, 100))
                        with dpg.plot(height=150, width=-1):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_clean_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="cs_clean_y"):
                                dpg.add_line_series([], [], label="Single-County", tag="cs_clean_series")
                dpg.add_text(
                    "", tag="cs_max_clean_note",
                    color=(140, 170, 140),
                )
            dpg.add_text(
                "Apply a score to use this panel.",
                tag="cs_inactive_lbl", show=False, color=(150, 150, 150),
            )

    def _build_partisanship_panel(self):
        # Create 12 themes for partisan color gradient (using map_view palette)
        from mosaic.gui.map_view import _PARTISAN_BREAKS, _PARTISAN_RGBA
        self._partisan_bar_themes = []
        for rgba in _PARTISAN_RGBA:
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvBarSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Fill,
                        (int(rgba[0]), int(rgba[1]), int(rgba[2]), 220),
                        category=dpg.mvThemeCat_Plots,
                    )
            self._partisan_bar_themes.append(t)

        # Theme for reference lines: thin white semi-transparent
        with dpg.theme() as self._partisan_ref_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (255, 255, 255, 140),
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 1.0,
                    category=dpg.mvThemeCat_Plots,
                )

        with dpg.window(
            label="Partisanship", tag="panel_partisanship",
            show=False, width=540, height=320,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_partisan_item, False),
        ):
            with dpg.plot(height=-1, width=-1, tag="partisan_plot"):
                dpg.add_plot_axis(
                    dpg.mvXAxis, tag="partisan_x",
                    label="Districts (least to most Democratic)",
                    no_tick_labels=True,
                )
                with dpg.plot_axis(dpg.mvYAxis, label="D Vote Share", tag="partisan_y"):
                    self._partisan_bar_series = []
                    for i in range(12):
                        s = dpg.add_bar_series([], [], weight=0.85, show=True)
                        dpg.bind_item_theme(s, self._partisan_bar_themes[i])
                        self._partisan_bar_series.append(s)
                    # Reference lines drawn after bars so they render on top;
                    # ##-prefix suppresses legend entries.
                    _ref = dpg.add_line_series(
                        [0, 200], [0.5, 0.5], label="##ref50", tag="partisan_ref",
                    )
                    dpg.bind_item_theme(_ref, self._partisan_ref_theme)
                    _med = dpg.add_line_series(
                        [], [], label="##refmed", tag="partisan_median",
                    )
                    dpg.bind_item_theme(_med, self._partisan_ref_theme)
        dpg.set_axis_limits("partisan_y", 0.0, 1.0)

    def _build_mm_panel(self):
        with dpg.window(
            label="Mean-Median Difference", tag="panel_mm",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_mm_item, False),
        ):
            with dpg.group(tag="mm_plot_grp"):
                with dpg.plot(height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="mm_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Mean-Median", tag="mm_y"):
                        dpg.add_line_series([], [], label="MM", tag="mm_series")
            dpg.add_text(
                "Load election data to use this panel.",
                tag="mm_inactive_lbl", show=False, color=(150, 150, 150),
            )

    def _build_eg_panel(self):
        with dpg.window(
            label="Efficiency Gap", tag="panel_eg",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_eg_item, False),
        ):
            with dpg.group(tag="eg_plot_grp"):
                with dpg.plot(height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="eg_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Efficiency Gap", tag="eg_y"):
                        dpg.add_line_series([], [], label="EG", tag="eg_series")
            dpg.add_text(
                "Load election data to use this panel.",
                tag="eg_inactive_lbl", show=False, color=(150, 150, 150),
            )

    def _build_dem_seats_panel(self):
        with dpg.window(
            label="Expected Dem Seats", tag="panel_dem_seats",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_seats_item, False),
        ):
            with dpg.group(tag="seats_plot_grp"):
                with dpg.plot(height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="seats_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Expected D Seats", tag="seats_y"):
                        dpg.add_line_series([], [], label="D Seats", tag="seats_series")
            dpg.add_text(
                "Load election data to use this panel.",
                tag="seats_inactive_lbl", show=False, color=(150, 150, 150),
            )

    def _build_comp_panel(self):
        with dpg.window(
            label="Competitiveness", tag="panel_comp",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_comp_item, False),
        ):
            with dpg.group(tag="comp_plot_grp"):
                with dpg.plot(height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="comp_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Competitive Districts", tag="comp_y"):
                        dpg.add_line_series([], [], label="Competitive", tag="comp_series")
            dpg.add_text(
                "Load election data to use this panel.",
                tag="comp_inactive_lbl", show=False, color=(150, 150, 150),
            )

    def _build_pp_panel(self):
        with dpg.window(
            label="Polsby-Popper", tag="panel_pp",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_pp_item, False),
        ):
            with dpg.group(tag="pp_plot_grp"):
                with dpg.plot(height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="pp_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="PP Penalty (lower=compact)", tag="pp_y"):
                        dpg.add_line_series([], [], label="PP", tag="pp_series")
            dpg.add_text(
                "Apply a score to use this panel.",
                tag="pp_inactive_lbl", show=False, color=(150, 150, 150),
            )

    def _build_cut_edges_panel(self):
        with dpg.window(
            label="Cut Edges", tag="panel_cut_edges",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_cuts_item, False),
        ):
            with dpg.plot(height=-1, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cuts_x")
                with dpg.plot_axis(dpg.mvYAxis, label="Cut Edges", tag="cuts_y"):
                    dpg.add_line_series([], [], label="Cuts", tag="cuts_series")

    def _build_score_contrib_panel(self):
        with dpg.window(
            label="Score Contributors", tag="panel_score_contrib",
            show=False, width=340, height=240,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_contrib_item, False),
        ):
            dpg.add_text("% of weighted score each metric contributes.",
                         color=(150, 150, 150))
            dpg.add_text("Updated every 500 iterations.",
                         color=(120, 120, 120))
            dpg.add_separator()
            dpg.add_spacer(height=4)
            for _name, _tag in _CONTRIB_METRICS:
                dpg.add_text("", tag=_tag, show=False)
            self._contrib_status = dpg.add_text(
                "Start a run to see contributors.", color=(130, 130, 130),
            )

    def _build_help_popup(self):
        with dpg.window(
            label="Help", tag="popup_help",
            show=False, width=580, height=680,
            pos=[(_VP_W - 580) // 2, (_VP_H - 680) // 2],
        ):
            with dpg.child_window(height=-40, border=False):
                dpg.add_text(
                    "This is a MosaicPy demo from Matt Mohn (@mattmxhn)",
                    color=(255, 200, 60),
                )
                dpg.add_spacer(height=6)

                dpg.add_text("Overview", color=(200, 200, 100))
                dpg.add_separator()
                dpg.add_text(
                    "Mosaic uses simulated annealing and recombination to produce "
                    "a mass of different redistricting plans that can meet certain "
                    "criteria. Simulated annealing is an algorithmic technique that "
                    "'cools' over time. At the start, Mosaic runs wildly across the "
                    "map, making tons of edits - not all of which are improvements. "
                    "As the temperature falls, Mosaic becomes more picky - a process "
                    "that creates a more optimal map. The other part of Mosaic "
                    "(recombination) is how Mosaic makes changes. Mosaic combines "
                    "random pairs of touching districts hundreds of times a second, "
                    "then divides them in a new way. This 'recombination' approach "
                    "means that the entire map can be replaced or reconfigured in "
                    "only a few increments of a second.",
                    wrap=540,
                )
                dpg.add_spacer(height=8)

                dpg.add_text("Shapefiles", color=(200, 200, 100))
                dpg.add_separator()
                dpg.add_text(
                    "Mosaic needs a shapefile (.shp) with precinct or voting district "
                    "polygons. Your shapefile should include a population column, and "
                    "optionally a county column (for county splits scoring) and "
                    "Democratic/Republican vote columns (for partisan metrics). You "
                    "can download precinct shapefiles from the U.S. Census TIGER/Line "
                    "repository and join election data from Dave's Redistricting App. "
                    "Mosaic runs faster on generalized (simplified) shapefiles.",
                    wrap=540,
                )
                dpg.add_spacer(height=8)

                dpg.add_text("Configuration", color=(200, 200, 100))
                dpg.add_separator()
                dpg.add_text(
                    "Annealing settings are in Configuration > Annealing Settings. "
                    "Population deviation tolerance is in Configuration > Population. "
                    "Partisan metric settings are in Configuration > Partisanship Settings.",
                    wrap=540,
                )
                dpg.add_spacer(height=8)

                dpg.add_text("Scores", color=(200, 200, 100))
                dpg.add_separator()
                dpg.add_text(
                    "Scores are set on the left side before annealing begins. To "
                    "follow a specific score, check its panel in the Panels menu.",
                    wrap=540,
                )
                dpg.add_text(
                    "  - Cut Edges: Fewer cut edges = more compact districts\n"
                    "  - County Splits: Penalizes unnecessary county splits\n"
                    "  - Polsby-Popper: Geometric compactness (circle = 1.0)\n"
                    "  - Mean-Median: Partisan asymmetry (0 = balanced)\n"
                    "  - Efficiency Gap: Wasted votes (0 = no advantage)\n"
                    "  - Competitiveness: Lower = more competitive districts\n"
                    "  - Expected Dem Seats: Probabilistic seat count",
                    wrap=540, color=(180, 180, 180),
                )
                dpg.add_spacer(height=8)

                dpg.add_text("Basic Usage", color=(200, 200, 100))
                dpg.add_separator()
                dpg.add_text(
                    "1. Click 'Import Shapefile from File' to select and load a shapefile\n"
                    "2. Map your columns in the picker (population, county, votes)\n"
                    "3. Set districts, iterations, and enable desired scores\n"
                    "4. Click Start to begin optimization\n"
                    "5. Use Pause, Reset, or Revert to Best as needed\n"
                    "6. Save Assignments exports your district map as CSV",
                    wrap=540,
                )
                dpg.add_spacer(height=8)

                dpg.add_text("Exporting", color=(200, 200, 100))
                dpg.add_separator()
                dpg.add_text(
                    "Save Assignments exports a CSV mapping each precinct to its "
                    "district. This file can be joined back to your shapefile in "
                    "GIS software or uploaded to Dave's Redistricting App using "
                    "their 'Color Map from File' feature.",
                    wrap=540,
                )
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item("popup_help", show=False),
                width=80,
            )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            self._update_ui()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    # ── Frame update ──────────────────────────────────────────────────────────

    def _update_ui(self):
        snap = self.state.get_all()
        status  = snap["status"]
        is_running = status == AlgorithmStatus.RUNNING
        is_paused  = status == AlgorithmStatus.PAUSED
        is_busy    = status in (
            AlgorithmStatus.RUNNING,
            AlgorithmStatus.PARTITIONING,
            AlgorithmStatus.PAUSED,
        )

        # ── Inspection complete → show shapefile dialog ────────────────────────
        if snap["shp_inspect_ready"]:
            self.state.update(shp_inspect_ready=False)
            if self.runner and self.runner._pending_inspection is not None:
                self._shp_dialog.populate(self.runner._pending_inspection)

        # ── Load complete → update shapefile info label ────────────────────────
        if snap["gdf_ready"]:
            self.state.update(gdf_ready=False)
            self._update_shp_info_label()

        # ── Map: trigger background load when new shapefile is ready ──────────
        loaded_path = self.state.shapefile_path
        not_loading = status not in (AlgorithmStatus.LOADING,
                                     AlgorithmStatus.BUILDING_GRAPH)
        if (loaded_path
                and loaded_path != self._map_loaded_path
                and not self._map_loading
                and not_loading
                and self.runner
                and self.runner.gdf is not None):
            self._map_loading = True
            self._map_loaded_path = loaded_path
            gdf_ref = self.runner.gdf
            county_array_ref = self.runner.county_array
            dem_ref = (self.runner.election_arrays[0][0]
                       if self.runner.election_arrays else None)
            gop_ref = (self.runner.election_arrays[0][1]
                       if self.runner.election_arrays else None)
            pp_data_ref  = self.runner.pp_data
            pop_ref      = self.runner.populations
            mv = self.map_view
            def _bg_load():
                mv.load(gdf_ref, county_array=county_array_ref,
                        dem_votes=dem_ref, gop_votes=gop_ref,
                        pp_data=pp_data_ref, populations=pop_ref)
                self._map_loading = False
                self._map_ready = True
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
            else:
                _assignment = _initial = None
                _n_dist = 0

        if (map_needs
                and _assignment is not None
                and self.map_view is not None):
            self.map_view.render_assignment(_assignment, _n_dist, _initial)

        # ── Status / iteration ────────────────────────────────────────────────
        msg = snap["status_message"]
        dpg.set_value(
            self._status_txt,
            "Status: " + status.value + (" -- " + msg if msg else ""),
        )
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
            m, s = divmod(int(elapsed), 60)
            dpg.set_value(self._timer_txt,
                          f"Time: {m}:{s:02d}  |  {ips:.1f} iter/sec")

        # Score
        score   = snap["current_score"]
        best    = snap["best_score"]
        best_it = snap["best_iteration"]
        if score < float("inf"):
            dpg.set_value(self._score_txt, f"Score: {score:.2f}")
        if best < float("inf"):
            dpg.set_value(self._best_txt,
                          f"Best:  {best:.2f}   (iteration {best_it:,})")

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

        # ── Plots ─────────────────────────────────────────────────────────────
        # One lock acquisition, copying only the delta since the last frame.
        # At 1000 IPS / 60 fps that's ~17 new items regardless of run length.
        with self.state._lock:
            _sd   = list(self.state.score_history[self._buf_score.read:])
            _ad   = list(self.state.acceptance_rate_history[self._buf_acc.read:])
            _td   = list(self.state.temperature_history[self._buf_temp.read:])
            _csd  = list(self.state.county_splits_score_history[self._buf_cs_score.read:])
            _ced  = list(self.state.county_excess_splits_history[self._buf_cs_excess.read:])
            _cld  = list(self.state.county_clean_districts_history[self._buf_cs_clean.read:])
            _md   = list(self.state.mm_history[self._buf_mm.read:])
            _ed   = list(self.state.eg_history[self._buf_eg.read:])
            _sed  = list(self.state.dem_seats_history[self._buf_seats.read:])
            _cod  = list(self.state.competitive_count_history[self._buf_comp.read:])
            _pd   = list(self.state.pp_history[self._buf_pp.read:])
            _cutd = list(self.state.cut_edges_history[self._buf_cuts.read:])

        self._buf_score.add(_sd)
        self._buf_acc.add_pairs(_ad, scale=100.0)
        self._buf_temp.add(_td)
        self._buf_cs_score.add(_csd)
        self._buf_cs_excess.add(_ced)
        self._buf_cs_clean.add(_cld)
        self._buf_mm.add(_md)
        self._buf_eg.add(_ed)
        self._buf_seats.add(_sed)
        self._buf_comp.add(_cod)
        self._buf_pp.add(_pd)
        self._buf_cuts.add(_cutd)

        limit = dpg.get_value(self._limit_plots) if self._limit_plots else False

        def _render(buf: _SeriesBuffer, series_tag: str, x_tag: str, y_tag: str) -> None:
            if not buf.ys:
                return
            xs, ys = buf.plot_data(limit)
            dpg.set_value(series_tag, [xs, ys])
            dpg.fit_axis_data(x_tag)
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
                _render(self._buf_cs_score,  "cs_score_series",  "cs_score_x",  "cs_score_y")
                _render(self._buf_cs_excess, "cs_excess_series", "cs_excess_x", "cs_excess_y")
                _render(self._buf_cs_clean,  "cs_clean_series",  "cs_clean_x",  "cs_clean_y")
                if self._buf_cs_excess.ys:
                    hi = max(1, int(max(self._buf_cs_excess.ys)))
                    dpg.set_axis_limits("cs_excess_y", 0, hi + 1)
                if self._buf_cs_clean.ys:
                    lo = max(0, int(min(self._buf_cs_clean.ys)) - 1)
                    hi = int(max(self._buf_cs_clean.ys)) + 1
                    dpg.set_axis_limits("cs_clean_y", lo, hi)
            if (self.runner is not None and self.runner.county_pops is not None
                    and self.runner.populations is not None):
                n_dist = dpg.get_value(self._num_districts)
                tol = dpg.get_value(self._tolerance)
                ideal_pop = float(self.runner.populations.sum()) / n_dist if n_dist > 0 else 1.0
                min_dp = max(ideal_pop * (1.0 - tol), 1.0)
                max_clean = int(np.floor(self.runner.county_pops / min_dp).sum())
                dpg.set_value("cs_max_clean_note",
                              f"Maximum feasible single-county districts: {max_clean:,}")

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
        if dpg.is_item_shown("panel_dem_seats"):
            dpg.configure_item("seats_plot_grp",     show=self._has_elections)
            dpg.configure_item("seats_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_seats, "seats_series", "seats_x", "seats_y")
        if dpg.is_item_shown("panel_comp"):
            dpg.configure_item("comp_plot_grp",     show=self._has_elections)
            dpg.configure_item("comp_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_comp, "comp_series", "comp_x", "comp_y")
        if dpg.is_item_shown("panel_pp"):
            dpg.configure_item("pp_plot_grp",     show=pp_on)
            dpg.configure_item("pp_inactive_lbl", show=not pp_on)
            if pp_on:
                _render(self._buf_pp, "pp_series", "pp_x", "pp_y")
        if dpg.is_item_shown("panel_cut_edges"):
            _render(self._buf_cuts, "cuts_series", "cuts_x", "cuts_y")

        # Score Contributors panel — refresh every 500 iterations
        if dpg.is_item_shown("panel_score_contrib"):
            cur_it = snap["current_iteration"]
            if cur_it % 500 == 0 and cur_it != self._last_contrib_iter:
                self._last_contrib_iter = cur_it
                with self.state._lock:
                    bd = dict(self.state.score_breakdown)
                self._refresh_score_contrib(bd)

        # Partisanship panel — live bar chart from current assignment
        if dpg.is_item_shown("panel_partisanship"):
            from mosaic.gui.map_view import _PARTISAN_BREAKS, _PARTISAN_RGBA
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
                        [ranks[mask].tolist(), sorted_shares[mask].tolist()],
                    )
                n = len(sorted_shares)
                dpg.set_value("partisan_ref", [[0.5, n + 0.5], [0.5, 0.5]])
                median_x = (n + 1) / 2.0
                dpg.set_value("partisan_median", [[median_x, median_x], [0.0, 1.0]])
                dpg.fit_axis_data("partisan_x")
                dpg.set_axis_limits("partisan_y", 0.0, 1.0)

        # Keep dem seats target slider bounded by current district count
        n_dist_val = dpg.get_value(self._num_districts)
        dpg.configure_item(self._target_dem_seats, max_value=n_dist_val)

        # ── Button states ─────────────────────────────────────────────────────
        dpg.configure_item(self._run_btn,    enabled=not is_busy)
        dpg.configure_item(self._pause_btn,  enabled=is_running or is_paused)
        dpg.configure_item(self._pause_btn,
                           label="Resume" if is_paused else "Pause")
        dpg.configure_item(self._reset_btn,  enabled=True)
        has_result = self.state.best_assignment is not None
        dpg.configure_item(self._export_btn,
                           enabled=has_result and not is_busy)
        dpg.configure_item(self._metrics_btn,
                           enabled=has_result and not is_busy)
        can_revert = has_result and status in (
            AlgorithmStatus.IDLE, AlgorithmStatus.PAUSED,
            AlgorithmStatus.COMPLETED, AlgorithmStatus.ERROR,
        )
        dpg.configure_item(self._revert_btn, enabled=can_revert)

        # ── Button nudge / anti-nudge themes ─────────────────────────────────
        graph_ready = self.runner is not None and self.runner.graph is not None
        dpg.bind_item_theme(
            self._run_btn,
            self._nudge_theme if (not is_busy and graph_ready)
            else self._antinudge_theme,
        )
        dpg.bind_item_theme(
            self._pause_btn,
            self._nudge_theme if (is_running or is_paused)
            else self._antinudge_theme,
        )
        dpg.bind_item_theme(self._reset_btn, self._antinudge_theme)
        dpg.bind_item_theme(
            self._export_btn,
            self._nudge_theme if (has_result and not is_busy)
            else self._antinudge_theme,
        )
        dpg.bind_item_theme(
            self._metrics_btn,
            self._nudge_theme if (has_result and not is_busy)
            else self._antinudge_theme,
        )
        dpg.bind_item_theme(
            self._revert_btn,
            self._nudge_theme if can_revert else self._antinudge_theme,
        )

    def _refresh_score_contrib(self, bd: dict) -> None:
        any_shown = False
        for name, tag in _CONTRIB_METRICS:
            pct = bd.get(name, 0.0)
            if pct > 0.01:
                dots = max(1, 36 - len(name))
                dpg.set_value(tag, f"{name} {'.' * dots} {pct:5.1f}%")
                dpg.configure_item(tag, show=True)
                any_shown = True
            else:
                dpg.configure_item(tag, show=False)
        dpg.configure_item(self._contrib_status, show=not any_shown)

    def _clear_all_series(self) -> None:
        """Clear local history buffers and blank all DPG plot series."""
        for buf in (
            self._buf_score, self._buf_acc, self._buf_temp,
            self._buf_cs_score, self._buf_cs_excess, self._buf_cs_clean,
            self._buf_mm, self._buf_eg, self._buf_seats,
            self._buf_comp, self._buf_pp, self._buf_cuts,
        ):
            buf.clear()
        empty = [[], []]
        for tag in (
            "score_series", "acc_series", "panel_temp_series",
            "cs_score_series", "cs_excess_series", "cs_clean_series",
            "mm_series", "eg_series", "seats_series",
            "comp_series", "pp_series", "cuts_series",
        ):
            dpg.set_value(tag, empty)
        for s in self._partisan_bar_series:
            dpg.set_value(s, empty)

    # ── Shapefile info label ──────────────────────────────────────────────────

    def _update_shp_info_label(self) -> None:
        """Refresh the shapefile status line after a successful load."""
        if self.runner is None or self._loaded_config is None:
            return
        cfg  = self._loaded_config
        insp = self.runner._pending_inspection
        if insp is None:
            return
        stem = Path(insp.path).stem
        dpg.set_value(self._shp_info, f"Loaded: {stem}")
        dpg.configure_item(self._shp_info, color=(150, 200, 150))

        # Enable/disable county-dependent controls based on what was loaded
        has_county = cfg.county_col is not None
        self._has_county = has_county
        dpg.configure_item(self._cs_lbl,
                           color=(180, 180, 180) if has_county else (90, 90, 90))
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
        has_elections = bool(cfg.elections)
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

        # Compactness and Pop. Deviation map views
        has_pp   = self.runner is not None and self.runner.pp_data is not None
        has_pops = self.runner is not None and self.runner.populations is not None
        dpg.configure_item(self._compactness_view, enabled=has_pp)
        dpg.configure_item(self._pop_dev_view,     enabled=has_pops)
        if not has_pp and dpg.get_value(self._compactness_view):
            dpg.set_value(self._compactness_view, False)
            if self.map_view:
                self.map_view.compactness_view = False
        if not has_pops and dpg.get_value(self._pop_dev_view):
            dpg.set_value(self._pop_dev_view, False)
            if self.map_view:
                self.map_view.pop_dev_view = False

        # Enable/disable partisan metric controls based on election data
        _pt_color = (180, 180, 180) if has_elections else (90, 90, 90)
        for chk, lbl in [
            (self._mm_enabled,   self._mm_lbl),
            (self._eg_enabled,   self._eg_lbl),
            (self._comp_enabled, self._comp_lbl),
            (self._seats_enabled, self._seats_lbl),
        ]:
            dpg.configure_item(chk, enabled=has_elections)
            dpg.configure_item(lbl, color=_pt_color)
        if not has_elections:
            for chk, ctrl_tag in [
                (self._mm_enabled,    "mm_controls"),
                (self._eg_enabled,    "eg_controls"),
                (self._comp_enabled,  "comp_controls"),
                (self._seats_enabled, "seats_controls"),
            ]:
                dpg.set_value(chk, False)
                dpg.configure_item(ctrl_tag, show=False)
        # Enable/disable partisan panel menu items; close any open ones when unavailable
        for item_tag, panel_tag in [
            (self._panel_partisan_item, "panel_partisanship"),
            (self._panel_mm_item,       "panel_mm"),
            (self._panel_eg_item,       "panel_eg"),
            (self._panel_seats_item,    "panel_dem_seats"),
            (self._panel_comp_item,     "panel_comp"),
        ]:
            dpg.configure_item(item_tag, enabled=has_elections)
            if not has_elections and dpg.is_item_shown(panel_tag):
                dpg.set_value(item_tag, False)
                dpg.configure_item(panel_tag, show=False)

    # ── Popup toggle callbacks ────────────────────────────────────────────────

    def _on_ann_toggle(self):
        dpg.configure_item("ann_body",
                           show=dpg.get_value(self._ann_enabled))

    def _on_cool_mode(self):
        guided = dpg.get_value(self._cool_mode) == "Guided (recommended)"
        dpg.configure_item("guided_controls", show=guided)
        dpg.configure_item("static_controls", show=not guided)

    def _on_cut_toggle(self):
        en = dpg.get_value(self._cut_enabled)
        dpg.configure_item(self._cut_lbl,
                           color=(90, 220, 90) if en else (110, 110, 110))
        dpg.configure_item("cut_edge_controls", show=en)

    def _on_cs_toggle(self):
        en = dpg.get_value(self._cs_enabled)
        dpg.configure_item(self._cs_lbl,
                           color=(90, 220, 90) if en else (110, 110, 110))
        dpg.configure_item("cs_controls", show=en)

    def _on_pp_toggle(self):
        en = dpg.get_value(self._pp_enabled)
        dpg.configure_item(self._pp_lbl,
                           color=(90, 220, 90) if en else (110, 110, 110))
        dpg.configure_item("pp_controls", show=en)

    def _on_mm_toggle(self):
        en = dpg.get_value(self._mm_enabled)
        dpg.configure_item(self._mm_lbl,
                           color=(90, 220, 90) if en else (110, 110, 110))
        dpg.configure_item("mm_controls", show=en)

    def _on_eg_toggle(self):
        en = dpg.get_value(self._eg_enabled)
        dpg.configure_item(self._eg_lbl,
                           color=(90, 220, 90) if en else (110, 110, 110))
        dpg.configure_item("eg_controls", show=en)

    def _on_comp_toggle(self):
        en = dpg.get_value(self._comp_enabled)
        dpg.configure_item(self._comp_lbl,
                           color=(90, 220, 90) if en else (110, 110, 110))
        dpg.configure_item("comp_controls", show=en)

    def _on_seats_toggle(self):
        en = dpg.get_value(self._seats_enabled)
        dpg.configure_item(self._seats_lbl,
                           color=(90, 220, 90) if en else (110, 110, 110))
        dpg.configure_item("seats_controls", show=en)

    def _on_county_bias_toggle(self):
        dpg.configure_item("county_bias_controls",
                           show=dpg.get_value(self._county_bias_enabled))

    def _on_panel_temp_toggle(self):
        dpg.configure_item(
            "panel_temperature", show=dpg.get_value(self._panel_temp_item),
        )

    def _on_temp_panel_close(self):
        dpg.set_value(self._panel_temp_item, False)

    def _on_panel_cs_toggle(self):
        dpg.configure_item("panel_county_splits", show=dpg.get_value(self._panel_cs_item))

    def _on_panel_partisan_toggle(self):
        dpg.configure_item("panel_partisanship", show=dpg.get_value(self._panel_partisan_item))

    def _on_panel_mm_toggle(self):
        dpg.configure_item("panel_mm", show=dpg.get_value(self._panel_mm_item))

    def _on_panel_eg_toggle(self):
        dpg.configure_item("panel_eg", show=dpg.get_value(self._panel_eg_item))

    def _on_panel_seats_toggle(self):
        dpg.configure_item("panel_dem_seats", show=dpg.get_value(self._panel_seats_item))

    def _on_panel_comp_toggle(self):
        dpg.configure_item("panel_comp", show=dpg.get_value(self._panel_comp_item))

    def _on_panel_pp_toggle(self):
        dpg.configure_item("panel_pp", show=dpg.get_value(self._panel_pp_item))

    def _on_panel_cuts_toggle(self):
        dpg.configure_item("panel_cut_edges", show=dpg.get_value(self._panel_cuts_item))

    def _on_panel_contrib_toggle(self):
        show = dpg.get_value(self._panel_contrib_item)
        dpg.configure_item("panel_score_contrib", show=show)
        if show:
            with self.state._lock:
                bd = dict(self.state.score_breakdown)
            self._refresh_score_contrib(bd)

    def _on_county_overlay_toggle(self):
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.county_overlay = dpg.get_value(self._county_overlay)
        with self.state._lock:
            _a    = (self.state.current_assignment.copy()
                     if self.state.current_assignment is not None else None)
            _init = (self.state.initial_assignment.copy()
                     if self.state.initial_assignment is not None else None)
            _nd   = self.state.num_districts
        if _a is not None:
            self.map_view.render_assignment(_a, _nd, _init)
        else:
            self.map_view.draw_blank()

    def _on_partisan_overlay_toggle(self):
        if dpg.get_value(self._partisan_overlay):
            # Mutually exclusive with other base-layer views
            for cb, attr in [
                (self._district_partisan, "district_partisan_overlay"),
                (self._compactness_view,  "compactness_view"),
                (self._pop_dev_view,      "pop_dev_view"),
            ]:
                dpg.set_value(cb, False)
                if self.map_view:
                    setattr(self.map_view, attr, False)
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.partisan_overlay = dpg.get_value(self._partisan_overlay)
        with self.state._lock:
            _a    = (self.state.current_assignment.copy()
                     if self.state.current_assignment is not None else None)
            _init = (self.state.initial_assignment.copy()
                     if self.state.initial_assignment is not None else None)
            _nd   = self.state.num_districts
        if _a is not None:
            self.map_view.render_assignment(_a, _nd, _init)
        else:
            self.map_view.draw_blank()

    def _on_district_partisan_toggle(self):
        if dpg.get_value(self._district_partisan):
            # Mutually exclusive with other base-layer views
            for cb, attr in [
                (self._partisan_overlay, "partisan_overlay"),
                (self._compactness_view, "compactness_view"),
                (self._pop_dev_view,     "pop_dev_view"),
            ]:
                dpg.set_value(cb, False)
                if self.map_view:
                    setattr(self.map_view, attr, False)
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.district_partisan_overlay = dpg.get_value(self._district_partisan)
        with self.state._lock:
            _a    = (self.state.current_assignment.copy()
                     if self.state.current_assignment is not None else None)
            _init = (self.state.initial_assignment.copy()
                     if self.state.initial_assignment is not None else None)
            _nd   = self.state.num_districts
        if _a is not None:
            self.map_view.render_assignment(_a, _nd, _init)
        else:
            self.map_view.draw_blank()

    def _on_splits_view_toggle(self):
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.splits_view = dpg.get_value(self._splits_view)
        with self.state._lock:
            _a    = (self.state.current_assignment.copy()
                     if self.state.current_assignment is not None else None)
            _init = (self.state.initial_assignment.copy()
                     if self.state.initial_assignment is not None else None)
            _nd   = self.state.num_districts
        if _a is not None:
            self.map_view.render_assignment(_a, _nd, _init)
        else:
            self.map_view.draw_blank()

    def _on_compactness_toggle(self):
        if dpg.get_value(self._compactness_view):
            # Mutually exclusive with partisan base views
            for cb, attr in [
                (self._partisan_overlay, "partisan_overlay"),
                (self._district_partisan, "district_partisan_overlay"),
                (self._pop_dev_view, "pop_dev_view"),
            ]:
                dpg.set_value(cb, False)
                if self.map_view:
                    setattr(self.map_view, attr, False)
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.compactness_view = dpg.get_value(self._compactness_view)
        with self.state._lock:
            _a    = (self.state.current_assignment.copy()
                     if self.state.current_assignment is not None else None)
            _init = (self.state.initial_assignment.copy()
                     if self.state.initial_assignment is not None else None)
            _nd   = self.state.num_districts
        if _a is not None:
            self.map_view.render_assignment(_a, _nd, _init)
        else:
            self.map_view.draw_blank()

    def _on_pop_dev_toggle(self):
        if dpg.get_value(self._pop_dev_view):
            # Mutually exclusive with partisan base views
            for cb, attr in [
                (self._partisan_overlay, "partisan_overlay"),
                (self._district_partisan, "district_partisan_overlay"),
                (self._compactness_view, "compactness_view"),
            ]:
                dpg.set_value(cb, False)
                if self.map_view:
                    setattr(self.map_view, attr, False)
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.pop_dev_view = dpg.get_value(self._pop_dev_view)
        with self.state._lock:
            _a    = (self.state.current_assignment.copy()
                     if self.state.current_assignment is not None else None)
            _init = (self.state.initial_assignment.copy()
                     if self.state.initial_assignment is not None else None)
            _nd   = self.state.num_districts
        if _a is not None:
            self.map_view.render_assignment(_a, _nd, _init)
        else:
            self.map_view.draw_blank()

    # ── Action callbacks ──────────────────────────────────────────────────────

    def _on_import_shapefile(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select Shapefile",
            filetypes=[("Shapefiles", "*.shp"), ("All files", "*.*")],
            initialdir=Path.cwd() / "shapefiles",
        )
        root.destroy()
        if not path:
            return
        self.runner = AlgorithmRunner(self.state)
        self._loaded_config = None
        self._has_elections = False
        self._has_county    = False
        self.state.update(
            current_assignment=None,
            best_assignment=None,
            initial_assignment=None,
        )
        dpg.set_value(self._shp_info, "Reading shapefile...")
        dpg.configure_item(self._shp_info, color=(150, 150, 150))
        threading.Thread(
            target=self.runner.start_inspection, args=(path,), daemon=True,
        ).start()

    def _on_shp_confirm(self, inspection: ShapefileInspection,
                        config: ShapefileConfig) -> None:
        """Called by ShapefileDialog when the user clicks Confirm and Load."""
        self._loaded_config = config
        # Flush all history and series before the new load so charts start
        # fresh and the previous file's data doesn't bleed through.
        with self.state._lock:
            self.state.score_history = []
            self.state.temperature_history = []
            self.state.acceptance_rate_history = []
            self.state.county_splits_score_history = []
            self.state.county_excess_splits_history = []
            self.state.county_clean_districts_history = []
            self.state.mm_history = []
            self.state.eg_history = []
            self.state.dem_seats_history = []
            self.state.competitive_count_history = []
            self.state.pp_history = []
            self.state.cut_edges_history = []
        self._clear_all_series()
        dpg.set_value(self._shp_info, "Building graph...")
        dpg.configure_item(self._shp_info, color=(150, 150, 150))
        threading.Thread(
            target=self.runner.complete_load,
            args=(inspection, config),
            daemon=True,
        ).start()

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
            self.state.county_clean_districts_history = self.state.county_clean_districts_history[:n_score]
            self.state.mm_history               = self.state.mm_history[:n_score]
            self.state.eg_history               = self.state.eg_history[:n_score]
            self.state.dem_seats_history        = self.state.dem_seats_history[:n_score]
            self.state.competitive_count_history = self.state.competitive_count_history[:n_score]
            self.state.pp_history               = self.state.pp_history[:n_score]
            self.state.cut_edges_history        = self.state.cut_edges_history[:n_score]

        # Release any paused algorithm thread before overwriting state
        self.state.request_stop()

        self.state.update(
            status=AlgorithmStatus.IDLE,
            status_message=f"Reverted to best (iteration {best_iter:,})",
            current_assignment=best_assign,
            current_score=best_score,
            current_cut_edges=best_cuts,
            current_iteration=best_iter,
            current_temperature=best_temp,
            should_stop=False,
            should_pause=False,
        )
        if max_iter > 0:
            dpg.set_value(self._progress, best_iter / max_iter)

        # Trim local history buffers to match the reverted-to state
        for buf in (
            self._buf_score, self._buf_cs_score, self._buf_cs_excess,
            self._buf_cs_clean, self._buf_mm, self._buf_eg,
            self._buf_seats, self._buf_comp, self._buf_pp, self._buf_cuts,
        ):
            buf.trim_to(best_iter, n_score)
        self._buf_temp.trim_to(best_iter, n_temp)
        self._buf_acc.trim_to(best_iter, n_acc)

        if self.map_view is not None and self.map_view._loaded:
            self.map_view.render_assignment(best_assign, n_dist, _init)

    def _on_shp_cancel(self) -> None:
        """Called by ShapefileDialog when the user clicks Cancel."""
        self.state.update(
            status=AlgorithmStatus.IDLE,
            status_message="Load cancelled",
        )
        dpg.set_value(self._shp_info, "No shapefile loaded")
        dpg.configure_item(self._shp_info, color=(150, 150, 150))

    def _on_run(self):
        if self.runner is None or self.runner.graph is None:
            self.state.update(status=AlgorithmStatus.ERROR,
                              error_message="Please load a shapefile first")
            return

        cs_on    = dpg.get_value(self._cs_enabled)
        mm_on    = dpg.get_value(self._mm_enabled)
        eg_on    = dpg.get_value(self._eg_enabled)
        comp_on  = dpg.get_value(self._comp_enabled)
        seats_on = dpg.get_value(self._seats_enabled)
        robust_eg = dpg.get_value(self._eg_mode) == "Robust (recommended)"

        w_cut  = (dpg.get_value(self._w_cut_edges)
                  if dpg.get_value(self._cut_enabled) else 0.0)
        w_cs   = dpg.get_value(self._w_county_splits) if cs_on else 0.0
        w_pp   = (dpg.get_value(self._w_polsby_popper)
                  if dpg.get_value(self._pp_enabled) else 0.0)

        n_dist_run = dpg.get_value(self._num_districts)
        raw_target_seats = dpg.get_value(self._target_dem_seats)
        target_seats = float(max(1, min(raw_target_seats, n_dist_run - 1)))

        score_cfg = ScoreConfig(
            weight_cut_edges=w_cut,
            weight_county_splits=w_cs,
            weight_polsby_popper=w_pp,
            weight_mean_median=dpg.get_value(self._w_mean_median) if mm_on else 0.0,
            target_mean_median=dpg.get_value(self._target_mean_median) if mm_on else 0.0,
            weight_efficiency_gap=dpg.get_value(self._w_efficiency_gap) if eg_on else 0.0,
            target_efficiency_gap=dpg.get_value(self._target_efficiency_gap) if eg_on else 0.0,
            use_robust_eg=robust_eg,
            weight_dem_seats=dpg.get_value(self._w_dem_seats) if seats_on else 0.0,
            target_dem_seats=target_seats,
            weight_competitiveness=dpg.get_value(self._w_competitiveness) if comp_on else 0.0,
            election_win_prob_at_55=dpg.get_value(self._win_prob),
        )

        guided = dpg.get_value(self._cool_mode) == "Guided (recommended)"
        ann_cfg = AnnealingConfig(
            enabled=dpg.get_value(self._ann_enabled),
            initial_temp_factor=dpg.get_value(self._temp_factor),
            cooling_mode="GUIDED" if guided else "STATIC",
            guide_fraction=dpg.get_value(self._guide_frac),
            target_temp=dpg.get_value(self._target_temp),
            cooling_rate=dpg.get_value(self._cooling_rate),
        )
        seed = dpg.get_value(self._seed) or None

        cb_en  = cs_on and dpg.get_value(self._county_bias_enabled)
        cb_val = dpg.get_value(self._county_bias)

        self.state.update(
            num_districts=dpg.get_value(self._num_districts),
            pop_tolerance=dpg.get_value(self._tolerance),
            max_iterations=dpg.get_value(self._iterations),
            seed=seed,
            score_config=score_cfg,
            annealing_config=ann_cfg,
            map_render_interval=dpg.get_value(self._map_interval),
            county_bias_enabled=cb_en,
            county_bias=cb_val,
        )
        self.state.reset_run()

        self.algorithm_thread = threading.Thread(
            target=self.runner.run_algorithm, daemon=True,
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
            self.state.county_clean_districts_history = []
            self.state.mm_history = []
            self.state.eg_history = []
            self.state.dem_seats_history = []
            self.state.competitive_count_history = []
            self.state.pp_history = []
            self.state.cut_edges_history = []
        self.state.update(
            status=AlgorithmStatus.IDLE,
            status_message="",
            current_score=float("inf"),
            best_score=float("inf"),
            current_cut_edges=0,
            current_iteration=0,
            successful_steps=0,
            current_temperature=0.0,
            accepted_worse=0,
            rejected_worse=0,
            best_assignment=None,
            current_assignment=None,
            score_breakdown={},
        )
        self._last_contrib_iter = -1
        # Reset status bar and score readouts
        dpg.set_value(self._status_txt, "Status: Idle")
        dpg.set_value(self._iter_txt,   "Iteration: 0 / 0")
        dpg.set_value(self._timer_txt,  "Time: 0:00  |  0 iter/sec")
        dpg.set_value(self._progress,   0.0)
        dpg.set_value(self._score_txt,  "Score: --")
        dpg.set_value(self._best_txt,   "Best:  --   (iteration --)")
        dpg.set_value(self._temp_txt,   "Temperature: --")
        dpg.set_value(self._acc_txt,    "Entropy: --")
        dpg.set_value(self._succ_txt,   "Accepted steps: --")
        self._clear_all_series()
        self._refresh_score_contrib({})
        if self.map_view is not None and self.map_view._loaded:
            self.map_view.draw_blank()

    def _on_export(self):
        if self.runner is None or self.state.best_assignment is None:
            return
        from datetime import datetime
        from mosaic.io import save_assignments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("output") / f"assignments_{timestamp}.csv"
        output_path.parent.mkdir(exist_ok=True)

        precinct_ids = None
        id_col_name = "precinct_id"
        if self.runner.gdf is not None and self._loaded_config is not None:
            col = self._loaded_config.id_col
            if col and col in self.runner.gdf.columns:
                precinct_ids = self.runner.gdf[col].tolist()
                id_col_name = col

        save_assignments(
            self.state.best_assignment,
            output_path,
            precinct_ids=precinct_ids,
            id_col_name=id_col_name,
        )
        self.state.update(status_message=f"Assignments saved to {output_path}")

    def _on_export_metrics(self):
        if self.runner is None or self.state.best_assignment is None:
            return
        from datetime import datetime
        from mosaic.io import save_metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("output") / f"metrics_{timestamp}.csv"
        output_path.parent.mkdir(exist_ok=True)

        n_dist = self.state.num_districts
        ideal_pop = (float(self.runner.populations.sum()) / n_dist
                     if n_dist > 0 else 1.0)
        dem = self.runner.election_arrays[0][0] if self.runner.election_arrays else None
        gop = self.runner.election_arrays[0][1] if self.runner.election_arrays else None

        save_metrics(
            self.state.best_assignment,
            output_path,
            populations=self.runner.populations,
            ideal_pop=ideal_pop,
            dem_votes=dem,
            gop_votes=gop,
            pp_data=self.runner.pp_data,
        )
        self.state.update(status_message=f"Metrics saved to {output_path}")


def main():
    app = MosaicApp()
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
