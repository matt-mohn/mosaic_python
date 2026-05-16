"""Main Dear PyGui application -- two-column layout with live district map."""

import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

_DOCS_URL = "https://matt-mohn.github.io/mosaic_python/"
_DOCS_SHAPEFILE_URL = "https://matt-mohn.github.io/mosaic_python/shapefiles.html"

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_APP_ICON = _ASSETS_DIR / "mosaic_logo.ico"

import dearpygui.dearpygui as dpg
import numpy as np

from mosaic.gui.state import SharedState, AlgorithmStatus
from mosaic.gui.runner import AlgorithmRunner
from mosaic.gui.map_view import MapView
from mosaic.gui.shp_dialog import ShapefileDialog
from mosaic.gui.theme import ThemeManager
from mosaic.io.inspect import ShapefileConfig, ShapefileInspection
from mosaic.scoring.score import ScoreConfig
from mosaic.recom.annealing import AnnealingConfig

_PLOT_LIMIT   = 10_000   # max points rendered when limit-plots is on
_COMPACT_AT   = 20_000   # compact local buffer when it exceeds this
_COMPACT_KEEP = 10_000   # keep last N at full resolution after compaction
_COMPACT_THIN = 50       # keep 1-in-N for old data (~200 pts per 10k iters)
# DPG's set_value + fit_axis_data iterate the full point list under the GIL
# on every frame.  At 60 fps that single-thread cost steals time from the
# algorithm worker.  Cap how many points we *send* to DPG (the buffer keeps
# every value it already keeps — this is a render-time stride only).  A
# ~400-px-wide plot can't visually resolve more than ~1k points anyway.
_RENDER_TARGET = 1500


class _SeriesBuffer:
    """Incremental, self-compacting local history for one plot series.

    The GUI copies only the *delta* since the last frame under the lock
    (~IPS/fps items, not the full history), keeping lock hold time constant
    regardless of total run length.  When the buffer exceeds _COMPACT_AT
    entries, entries older than _COMPACT_KEEP that haven't been thinned yet
    are thinned to 1-in-_COMPACT_THIN.  Already-thinned data is never
    thinned again (_compact_end tracks the boundary).
    """

    __slots__ = ("xs", "ys", "read", "_compact_end")

    def __init__(self):
        self.xs: list = []   # iteration indices (survive thinning intact)
        self.ys: list = []   # values
        self.read: int = 0   # items consumed from SharedState list so far
        self._compact_end: int = 0  # xs[:_compact_end] has already been thinned

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
        if cut <= self._compact_end:
            return  # nothing new to thin
        thin_xs = self.xs[self._compact_end:cut:_COMPACT_THIN]
        thin_ys = self.ys[self._compact_end:cut:_COMPACT_THIN]
        self.xs = self.xs[:self._compact_end] + thin_xs + self.xs[cut:]
        self.ys = self.ys[:self._compact_end] + thin_ys + self.ys[cut:]
        self._compact_end += len(thin_xs)

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
        self._compact_end = min(self._compact_end, len(self.xs))

    def clear(self) -> None:
        self.xs.clear()
        self.ys.clear()
        self.read = 0
        self._compact_end = 0


# Score Contributor panel — bar chart metrics in display order (structural → partisan)
# (name, x-tick label, RGBA fill)
_CONTRIB_BAR_METRICS = [
    ("Cut Edges",       "Cuts",   (160, 160, 165, 220)),
    ("County Splits",   "Co.Spl", (190, 170, 130, 220)),
    ("Compactness",     "PP",    (90,  160, 220, 220)),
    ("Population Deviation", "PopDev", (220, 200, 70,  220)),
    ("Mean-Median",     "MM",     (240, 140, 60,  220)),
    ("Efficiency Gap",  "EG",     (225, 75,  75,  220)),
    ("Dem Seats",       "Seats",  (180, 80,  220, 220)),
    ("Competitiveness", "Comp",   (70,  200, 120, 220)),
    ("D Majority",      "D Maj",  (70,  130, 210, 220)),
    ("R Majority",      "R Maj",  (210, 70,  70,  220)),
    ("Hinge",           "Hinge",  (140, 90,  200, 220)),
]

# ── Layout constants ──────────────────────────────────────────────────────────
_VP_W         = 1340
_VP_H         = 1000
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


# Hover-help catalog: plain-language definitions for technical terms in the UI.
# Attach with self._hint(widget_tag, "key"). Keep entries one or two sentences.
_HINTS: dict[str, str] = {
    "entropy": (
        "Entropy is the share of iterations that Mosaic accepts that are "
        "technically worse - when high, Mosaic is conducting lots of "
        "exploration, and when low, Mosaic is settling on a solution."
    ),
    "score": (
        "The score is how the current map rates - totaling all the weighted "
        "penalties you've set. Mosaic tries to lower this number over the "
        "course of the annealing."
    ),
    "county_edge_bias": (
        "Makes Mosaic less likely to cut edges that cross county lines when "
        "drawing new districts, so counties stay whole more often. Higher "
        "multiplier = stronger preference for keeping counties intact."
    ),
    "cut_edges": (
        "Counts adjacent precinct-pairs that fall in different districts. "
        "Lower means more natural, compact-graph boundaries."
    ),
    "county_splits": (
        "Penalizes plans that split counties across multiple districts. "
        "Encourages keeping counties whole or nearly so."
    ),
    "pop_deviation": (
        "How far each district's population strays from the ideal (total / N). "
        "Districts inside the safe-harbor band are unpenalized; the rest "
        "contribute proportional to how far out they are."
    ),
    "compactness": (
        "Polsby-Popper: ratio of district area to a circle with the same perimeter. "
        "1.0 = perfectly round, lower means stretched or jagged. "
        "Optimizer uses (1 - PP) as the penalty."
    ),
    "mean_median": (
        "Gap between the mean and median Democratic vote share across districts. "
        "A nonzero value signals partisan skew baked into the plan."
    ),
    "efficiency_gap": (
        "Difference in wasted votes between parties, normalized by total votes. "
        "Captures classic gerrymander signatures (packing and cracking). "
        "0 = neutral; sign shows which party benefits."
    ),
    "competitiveness": (
        "Counts districts within a competitive vote-share band (close to 50/50). "
        "Penalty grows when too few districts are competitive."
    ),
    "dem_seats": (
        "Expected number of Democratic-won districts under a partisan-swing model. "
        "Penalty drives the plan toward a target seat count."
    ),
    "majority_chance": (
        "Probability that the selected party wins a majority of seats under "
        "the swing model. Useful when integer seat count is too coarse."
    ),
    "hinge": (
        "Probability the selected party reaches a chosen seat threshold "
        "(e.g. 2/3 supermajority). Targets minority-veto or override-proof seat counts."
    ),
}


class MosaicApp:
    """Main application -- Dear PyGui interface coordinating the algorithm thread."""

    def __init__(self):
        self.state = SharedState()
        self.runner: Optional[AlgorithmRunner] = None
        self.algorithm_thread: Optional[threading.Thread] = None
        self.map_view: Optional[MapView] = None
        self._shp_dialog: Optional[ShapefileDialog] = None
        self.theme = ThemeManager(initial="light")

        # Stored after the user confirms the shapefile dialog
        self._loaded_config: Optional[ShapefileConfig] = None

        # Map background-load tracking (app-local, no SharedState).
        # Tracking by (path, gdf id) so a re-import of the same path with
        # edited content still triggers a fresh MapView load — otherwise the
        # map keeps stale dimensions and render_assignment indexes past the
        # new assignment array.
        self._map_loading: bool = False
        self._map_ready: bool = False
        self._map_loaded_path: str = ""
        self._map_loaded_gdf_id: int = 0

        # Plot appearance toggle (app-local)
        self._limit_plots: int | str = ""   # DPG checkbox tag, set during setup

        self._contrib_bar_series: list = []

        # Track what data the current shapefile has
        self._has_elections: bool = False
        self._has_county: bool = False

        # Tracks whether the last frame was in a "running" state, so we can
        # trigger a one-shot precise-label re-render on transitions out of
        # running (cheap centroid labels -> pole-of-inaccessibility labels).
        self._labels_were_fast: bool = False

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
        self._buf_popdev     = _SeriesBuffer()
        self._buf_popdev_max = _SeriesBuffer()
        self._buf_popdev_mean = _SeriesBuffer()
        self._buf_cuts      = _SeriesBuffer()
        self._buf_maj_dem   = _SeriesBuffer()
        self._buf_maj_rep   = _SeriesBuffer()
        self._buf_hinge     = _SeriesBuffer()

    # ── Setup ─────────────────────────────────────────────────────────────────

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

        # ── Build palette themes and bind initial ─────────────────────────────
        self.theme.build()
        self.theme.apply(self.theme.palette.name)
        self._sync_map_bg_to_theme()

        self._shp_dialog = ShapefileDialog(
            confirm_cb=self._on_shp_confirm,
            cancel_cb=self._on_shp_cancel,
            theme=self.theme,
        )
        self._shp_dialog.build(_VP_W, _VP_H)

        self._build_population_popup()
        self._build_seed_popup()
        self._build_opt_popup()
        self._build_partisan_popup()
        self._build_help_popup()
        self._build_temperature_panel()
        self._build_score_contrib_panel()
        self._build_ref_line_themes()
        self._build_county_splits_panel()
        self._build_partisanship_panel()
        self._build_win_chance_panel()
        self._build_mm_panel()
        self._build_eg_panel()
        self._build_dem_seats_panel()
        self._build_comp_panel()
        self._build_pp_panel()
        self._build_popdev_panel()
        self._build_cut_edges_panel()
        self._build_majority_panel()
        self._build_hinge_panel()
        self._build_district_info_panel()

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
                    self._panel_win_chance_item = dpg.add_menu_item(
                        label="Win Chance", check=True, default_value=False,
                        callback=self._on_panel_win_chance_toggle,
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
                        label="Compactness", check=True, default_value=False,
                        callback=self._on_panel_pp_toggle,
                    )
                    self._panel_popdev_item = dpg.add_menu_item(
                        label="Population Deviation", check=True, default_value=False,
                        callback=self._on_panel_popdev_toggle,
                    )
                    self._panel_cuts_item = dpg.add_menu_item(
                        label="Cut Edges", check=True, default_value=False,
                        callback=self._on_panel_cuts_toggle,
                    )
                    self._panel_hinge_item = dpg.add_menu_item(
                        label="Supermajority/Hinge", check=True, default_value=False,
                        callback=self._on_panel_hinge_toggle,
                    )
                    self._panel_majority_item = dpg.add_menu_item(
                        label="Chance of Majority", check=True, default_value=False,
                        callback=self._on_panel_majority_toggle,
                    )
                    dpg.add_separator()
                    self._panel_contrib_item = dpg.add_menu_item(
                        label="Score Contributors", check=True, default_value=False,
                        callback=self._on_panel_contrib_toggle,
                    )
                    self._panel_district_item = dpg.add_menu_item(
                        label="District Info", check=True, default_value=False,
                        callback=self._on_panel_district_toggle,
                    )

                with dpg.menu(label="Scores", tag="menu_scores"):
                    self._svis_cuts = dpg.add_menu_item(
                        label="Cut Edges", check=True, default_value=False,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_cuts", dpg.get_value(self._svis_cuts),
                            self._cut_enabled, self._on_cut_toggle),
                    )
                    self._svis_cs = dpg.add_menu_item(
                        label="County Splits", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_cs", dpg.get_value(self._svis_cs),
                            self._cs_enabled, self._on_cs_toggle),
                    )
                    self._svis_popdev = dpg.add_menu_item(
                        label="Population Deviation", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_popdev", dpg.get_value(self._svis_popdev),
                            self._popdev_enabled, self._on_popdev_score_toggle),
                    )
                    dpg.add_separator()
                    self._svis_pp = dpg.add_menu_item(
                        label="Compactness", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_pp", dpg.get_value(self._svis_pp),
                            self._pp_enabled, self._on_pp_toggle),
                    )
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
                    dpg.add_separator()
                    self._svis_comp = dpg.add_menu_item(
                        label="Competitiveness", check=True, default_value=True,
                        callback=lambda: self._set_score_row_vis(
                            "score_row_comp", dpg.get_value(self._svis_comp),
                            self._comp_enabled, self._on_comp_toggle),
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

                with dpg.menu(label="Help"):
                    dpg.add_menu_item(
                        label="Open Help...",
                        callback=lambda: dpg.configure_item("popup_help", show=True),
                    )

            with dpg.child_window(height=_TOP_H, border=False,
                                  no_scrollbar=True):
                with dpg.group(horizontal=True):

                    # ── Left column ───────────────────────────────────────────
                    with dpg.child_window(width=_LEFT_W, border=False):

                        self.theme.text("Mosaic", "title")
                        dpg.add_separator()
                        dpg.add_spacer(height=6)

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

                        dpg.add_spacer(height=8)
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
                                )
                            with dpg.group():
                                self.theme.text("Iterations", "subheading")
                                self._iterations = dpg.add_input_int(
                                    label="##iter",
                                    default_value=2500, min_value=1,
                                    max_value=1_000_000,
                                    width=_inp_w, step=0,
                                )

                        dpg.add_spacer(height=8)
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
                        self.theme.text("Status", "heading")
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

                        self.theme.text("District Map", "heading")
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
                            self._show_labels = dpg.add_checkbox(
                                label="Labels",
                                default_value=False,
                                callback=self._on_labels_toggle,
                            )

                        dpg.add_spacer(height=4)

                        with dpg.group(horizontal=True):
                            with dpg.group():
                                self._hint(self.theme.text("Score", "heading"), "score")
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
                                self._hint(self.theme.text("Entropy", "heading"), "entropy")
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
                    self.theme.text("For full scores, see toolbar", "muted")
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

                        with dpg.group(tag="score_row_cs", show=True):
                            with dpg.group(horizontal=True):
                                self._cs_enabled = dpg.add_checkbox(
                                    default_value=False,
                                    callback=self._on_cs_toggle,
                                )
                                self._cs_lbl = self.theme.text(
                                    "County Splits and Bias", "disabled",
                                )
                                self._hint(self._cs_lbl, "county_splits")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_county_splits", self._panel_cs_item))
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
                                self._hint(self._county_bias_enabled, "county_edge_bias")
                                with dpg.group(tag="county_bias_controls", show=False):
                                    self._county_bias = dpg.add_slider_int(
                                        label="Multiplier",
                                        default_value=5, min_value=1, max_value=20,
                                        width=_SCORE_COL_W - 120,
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

                    # Col 2: shape + partisan bias
                    with dpg.child_window(width=_SCORE_COL_W, height=-1,
                                          border=False):
                        with dpg.group(tag="score_row_pp", show=True):
                            with dpg.group(horizontal=True):
                                self._pp_enabled = dpg.add_checkbox(
                                    default_value=True,
                                    callback=self._on_pp_toggle,
                                )
                                self._pp_lbl = self.theme.text(
                                    "Compactness (Polsby-Popper)", "accent_green",
                                )
                                self._hint(self._pp_lbl, "compactness")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_pp", self._panel_pp_item))
                            with dpg.group(tag="pp_controls", show=True):
                                self._w_polsby_popper = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=25, min_value=0, max_value=100,
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
                                self._tooltip(
                                    self._target_mean_median,
                                    "- = D advantage  |  + = R advantage",
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
                                self._tooltip(
                                    self._target_efficiency_gap,
                                    "- = D bias  |  + = R bias",
                                )

                    # Col 3: outcome metrics
                    with dpg.child_window(width=-1, height=-1, border=False):
                        with dpg.group(tag="score_row_comp", show=True):
                            with dpg.group(horizontal=True):
                                self._comp_enabled = dpg.add_checkbox(
                                    default_value=False, enabled=False,
                                    callback=self._on_comp_toggle,
                                )
                                self._comp_lbl = self.theme.text(
                                    "Competitiveness",
                                    "disabled_deep",
                                )
                                self._hint(self._comp_lbl, "competitiveness")
                                dpg.add_button(label="↗", width=24,
                                    callback=lambda: self._show_panel("panel_comp", self._panel_comp_item))
                            with dpg.group(tag="comp_controls", show=False):
                                self._w_competitiveness = dpg.add_slider_int(
                                    label="Weight",
                                    default_value=1, min_value=0, max_value=100,
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
                                    default_value=1, min_value=0, max_value=100,
                                    width=_SCORE_COL_W - 100,
                                )
                                self._target_dem_seats = dpg.add_slider_int(
                                    label="Target S",
                                    default_value=7, min_value=1, max_value=14,
                                    width=_SCORE_COL_W - 100,
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

    def _build_population_popup(self):
        with dpg.window(
            label="Run Config -- Population",
            tag="popup_population", show=False,
            modal=True, no_close=True,
            width=420, height=260,
            pos=[(_VP_W - 420) // 2, (_VP_H - 260) // 2],
        ):
            self._tolerance = dpg.add_slider_float(
                label="Population Tolerance",
                default_value=5.0, min_value=0.1, max_value=10.0,
                format="%.1f %%", width=260,
            )
            self._tooltip(
                self._tolerance,
                "Mosaic will only explore solutions where each district differs "
                "from ideal by no more than this percentage in either direction.",
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("Population Deviation Score - Safe Harbor", "heading")
            self._pop_dev_harbor = dpg.add_slider_float(
                label="Safe Harbor",
                default_value=0.25, min_value=0.0, max_value=5.0,
                format="%.2f %%", width=260,
            )
            self._tooltip(
                self._pop_dev_harbor,
                "Districts within this % of ideal are not penalized by the "
                "population deviation score. Cannot exceed Population Tolerance "
                "(clamped on run).",
            )
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
            width=340, height=160,
            pos=[(_VP_W - 340) // 2, (_VP_H - 160) // 2],
        ):
            self._seed = dpg.add_input_int(
                label="Random Seed  (0 = random)",
                default_value=0, min_value=0, width=120,
            )
            self._tooltip(
                self._seed,
                "Set a non-zero seed to make a run reproducible. 0 leaves "
                "the RNG random so each run differs.\n\n"
                "Reproducibility is best-effort: identical results require the "
                "same machine, same Mosaic version, and same shapefile. "
                "Cross-machine or cross-version runs may diverge slightly due "
                "to floating-point ordering in numpy/igraph.",
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
            width=460, height=460,
            pos=[(_VP_W - 460) // 2, (_VP_H - 460) // 2],
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
                self._tooltip(
                    self._temp_factor,
                    "initial_temp = factor x initial_score",
                )
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
                    self._tooltip(
                        self._guide_frac,
                        "Fraction of iterations to cool over.",
                    )
                    self._target_temp = dpg.add_input_float(
                        label="Target Temp",
                        default_value=1.0, min_value=0.001,
                        format="%.3f", width=120,
                    )
                    self._tooltip(
                        self._target_temp,
                        "Temperature at the guide point (absolute).",
                    )

                with dpg.group(tag="static_controls", show=False):
                    self._cooling_rate = dpg.add_slider_float(
                        label="Cooling Rate / iteration",
                        default_value=0.9995, min_value=0.990,
                        max_value=0.99999, format="%.5f", width=260,
                    )

                dpg.add_spacer(height=8)
                dpg.add_separator()
                dpg.add_spacer(height=4)
                self._launch_watch_enabled = dpg.add_checkbox(
                    label="Launch Watch",
                    default_value=True,
                    callback=self._on_launch_watch_toggle,
                )
                with dpg.group(tag="launch_watch_controls"):
                    self._launch_watch_iter = dpg.add_input_int(
                        label="Re-anchor after iter",
                        default_value=250, min_value=10, max_value=10_000,
                        step=50, width=120,
                    )
                    self._tooltip(
                        self._launch_watch_iter,
                        "Once past this iteration, reset initial_temp to "
                        "factor x current_score (instead of initial_score). "
                        "Helps when scores nose-dive in the first ~250 steps.",
                    )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("n=3 ReCom Mix", "heading")
            self._n3_pct = dpg.add_slider_int(
                label="% of iterations",
                default_value=5, min_value=0, max_value=25,
                format="%d %%", width=200,
            )
            self._tooltip(
                self._n3_pct,
                "Fraction of steps that merge 3 districts (vs 2) and re-split. "
                "Helps escape local minima at the cost of ~2.4x per-step time. "
                "Set to 0 for mass-generation runs (no overhead).",
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
            width=460, height=320,
            pos=[(_VP_W - 460) // 2, (_VP_H - 320) // 2],
        ):
            self.theme.text(
                "Applied when partisan metrics are enabled.",
                "muted",
            )
            dpg.add_separator()
            dpg.add_spacer(height=6)

            self._win_prob = dpg.add_slider_float(
                label="Win Prob at 55% vote share",
                default_value=0.9, min_value=0.51, max_value=0.999,
                format="%.3f", width=220,
            )
            self._tooltip(
                self._win_prob,
                "P(D wins district | D has 55% of two-party vote)",
            )
            dpg.add_spacer(height=6)

            self._swing_sigma = dpg.add_slider_float(
                label="Swing sigma (shared)",
                default_value=0.03, min_value=0.005, max_value=0.10,
                format="%.3f", width=220,
            )
            self._tooltip(
                self._swing_sigma,
                "Std dev of partisan-environment swing shared across all districts.",
            )
            dpg.add_spacer(height=8)

            dpg.add_text("Efficiency Gap mode:")
            self._eg_mode = dpg.add_radio_button(
                items=["Robust (recommended)", "Static"],
                default_value="Robust (recommended)",
                horizontal=True,
            )
            self._tooltip(
                self._eg_mode,
                "Robust EG integrates out both swing sigma and per-district noise.",
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
                self.theme.text("County Splits Score", "heading")
                with dpg.plot(height=140, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_score_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Score", tag="cs_score_y"):
                        dpg.add_line_series([], [], label="CS Score", tag="cs_score_series")
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    with dpg.group():
                        self.theme.text("Excess Splits", "heading")
                        with dpg.plot(height=150, width=248):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_excess_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="cs_excess_y"):
                                dpg.add_line_series([], [], label="Excess", tag="cs_excess_series")
                    with dpg.group():
                        self.theme.text("Single-County Districts", "heading")
                        with dpg.plot(height=150, width=-1):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_clean_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="cs_clean_y"):
                                dpg.add_line_series([], [], label="Single-County", tag="cs_clean_series")
                                _cs_max = dpg.add_line_series(
                                    [], [], label="##cs_max", tag="cs_clean_max",
                                )
                                dpg.bind_item_theme(_cs_max, self._partisan_ref_theme)
                self.theme.track(
                    dpg.add_text("", tag="cs_max_clean_note"),
                    "success_soft",
                )
            self.theme.track(
                dpg.add_text(
                    "Apply a score to use this panel.",
                    tag="cs_inactive_lbl", show=False,
                ),
                "muted",
            )

    def _build_ref_line_themes(self):
        """Build the palette-aware reference-line themes used across panels
        (50/50 guides, median guides, max-feasible line, etc.). Must run
        BEFORE any panel that binds these themes."""
        self._partisan_ref_themes: dict[str, int] = {}
        for mode, rgba in (("dark",  (255, 255, 255, 140)),
                           ("light", (0,   0,   0,   140))):
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line, rgba,
                        category=dpg.mvThemeCat_Plots,
                    )
                    dpg.add_theme_style(
                        dpg.mvPlotStyleVar_LineWeight, 1.0,
                        category=dpg.mvThemeCat_Plots,
                    )
            self._partisan_ref_themes[mode] = t
        self._partisan_ref_theme = self._partisan_ref_themes[
            self.theme.palette.name
        ]

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
                with dpg.plot_axis(dpg.mvYAxis, label="D Vote Share (%)", tag="partisan_y"):
                    self._partisan_bar_series = []
                    for i in range(12):
                        s = dpg.add_bar_series([], [], weight=0.85, show=True)
                        dpg.bind_item_theme(s, self._partisan_bar_themes[i])
                        self._partisan_bar_series.append(s)
                    # Reference lines drawn after bars so they render on top;
                    # ##-prefix suppresses legend entries.
                    _ref = dpg.add_line_series(
                        [0, 200], [50.0, 50.0], label="##ref50", tag="partisan_ref",
                    )
                    dpg.bind_item_theme(_ref, self._partisan_ref_theme)
                    _med = dpg.add_line_series(
                        [], [], label="##refmed", tag="partisan_median",
                    )
                    dpg.bind_item_theme(_med, self._partisan_ref_theme)
        dpg.set_axis_limits("partisan_y", 0.0, 100.0)

    def _build_win_chance_panel(self):
        with dpg.window(
            label="Win Chance", tag="panel_win_chance",
            show=False, width=540, height=320,
            pos=[_LEFT_W + 120, 110],
            on_close=lambda: dpg.set_value(self._panel_win_chance_item, False),
        ):
            with dpg.plot(height=-1, width=-1, tag="win_chance_plot"):
                dpg.add_plot_axis(
                    dpg.mvXAxis, tag="win_chance_x",
                    label="Districts (least to most likely D win)",
                    no_tick_labels=True,
                )
                with dpg.plot_axis(dpg.mvYAxis, label="P(D wins) (%)", tag="win_chance_y"):
                    self._win_chance_bar_series = []
                    for i in range(12):
                        s = dpg.add_bar_series([], [], weight=0.85, show=True)
                        dpg.bind_item_theme(s, self._partisan_bar_themes[i])
                        self._win_chance_bar_series.append(s)
                    _wref = dpg.add_line_series(
                        [0, 200], [50.0, 50.0], label="##wref50", tag="win_chance_ref",
                    )
                    dpg.bind_item_theme(_wref, self._partisan_ref_theme)
                    _wmed = dpg.add_line_series(
                        [], [], label="##wrefmed", tag="win_chance_median",
                    )
                    dpg.bind_item_theme(_wmed, self._partisan_ref_theme)
        dpg.set_axis_limits("win_chance_y", 0.0, 100.0)

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
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="mm_inactive_lbl", show=False,
                ),
                "muted",
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
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="eg_inactive_lbl", show=False,
                ),
                "muted",
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
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="seats_inactive_lbl", show=False,
                ),
                "muted",
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
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="comp_inactive_lbl", show=False,
                ),
                "muted",
            )

    def _build_pp_panel(self):
        with dpg.window(
            label="Compactness", tag="panel_pp",
            show=False, width=500, height=295,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_pp_item, False),
        ):
            with dpg.group(tag="pp_plot_grp"):
                with dpg.plot(height=240, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="pp_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Polsby-Popper (100 = circle)", tag="pp_y"):
                        dpg.add_line_series([], [], label="PP", tag="pp_series")
            self._tooltip(
                "pp_plot_grp",
                "Optimizer uses (1 - PP) as penalty; higher is more compact.",
            )
            self.theme.track(
                dpg.add_text(
                    "Apply a score to use this panel.",
                    tag="pp_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("pp_y", 0.0, 100.0)

    def _build_popdev_panel(self):
        with dpg.window(
            label="Population Deviation", tag="panel_popdev",
            show=False, width=500, height=300,
            pos=[_LEFT_W + 100, 100],
            on_close=lambda: dpg.set_value(self._panel_popdev_item, False),
        ):
            with dpg.group(tag="popdev_plot_grp"):
                with dpg.plot(height=240, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="popdev_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="% Deviation", tag="popdev_y"):
                        dpg.add_line_series([], [], label="Max %",
                                            tag="popdev_max_series")
                        dpg.add_line_series([], [], label="Mean %",
                                            tag="popdev_mean_series")
                # Constrain y-axis so pan/zoom can't go below 0; upper grows freely.
                dpg.set_axis_limits_constraints("popdev_y", 0.0, float("inf"))
            self._tooltip(
                "popdev_plot_grp",
                "Max and mean absolute deviation from ideal population. Lower is better.",
            )
            self.theme.track(
                dpg.add_text(
                    "Apply a score to use this panel.",
                    tag="popdev_inactive_lbl", show=False,
                ),
                "muted",
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

    def _build_majority_panel(self):
        with dpg.window(
            label="Chance of Majority", tag="panel_majority",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_majority_item, False),
        ):
            with dpg.group(tag="majority_plot_grp"):
                with dpg.plot(height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="maj_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Chance of Majority (%)", tag="maj_y"):
                        dpg.add_line_series([], [], label="Dem Majority", tag="maj_dem_series")
                        dpg.add_line_series([], [], label="Rep Majority", tag="maj_rep_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="majority_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("maj_y", 0.0, 100.0)

    def _build_hinge_panel(self):
        with dpg.window(
            label="Supermajority/Hinge", tag="panel_hinge",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hinge_item, False),
        ):
            with dpg.group(tag="hinge_plot_grp"):
                with dpg.plot(height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hinge_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="P(Hinge)", tag="hinge_y"):
                        dpg.add_line_series([], [], label="P(hinge)", tag="hinge_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="hinge_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hinge_y", 0.0, 1.0)

    # Column spec for the District Info table.  (header, pixel_width, key)
    _DIST_COLS = [
        ("Dist",        50,  "dist"),
        ("Population",  95,  "pop"),
        ("Pop Dev",     80,  "pdev"),
        ("Compactness", 100, "pp"),
        ("Dem %",       70,  "demp"),
        ("Rep %",       70,  "repp"),
        ("Margin",      75,  "margin"),
    ]

    def _build_district_info_panel(self):
        # Tracks how many rows the table currently has, so we only rebuild
        # rows when the district count changes (not every frame).
        self._dist_table_built_rows: int = 0
        # Iteration at which we last refreshed the table.  -1 forces an
        # update on first paint and after the panel re-opens.
        self._dist_info_last_iter: int = -1
        total_w = sum(w for _, w, _ in self._DIST_COLS) + 32  # column widths + chrome
        with dpg.window(
            label="District Info", tag="panel_district_info",
            show=False, width=total_w, height=400,
            pos=[_LEFT_W + 80, 80],
            on_close=self._on_district_panel_close,
        ):
            self.theme.text("Per-district metrics for the current plan.", "muted")
            with dpg.group(horizontal=True):
                self._dist_info_update_every = dpg.add_slider_int(
                    label="Update every (iters)",
                    default_value=1000, min_value=50, max_value=10000,
                    width=200,
                    callback=self._on_dist_info_interval_change,
                )
            dpg.add_spacer(height=4)
            with dpg.table(
                tag="district_info_table",
                header_row=True,
                borders_innerH=True,
                borders_innerV=False,
                borders_outerH=False,
                borders_outerV=False,
                row_background=True,
                policy=dpg.mvTable_SizingFixedFit,
                scrollY=True,
                no_host_extendX=True,
                width=-1, height=-1,
            ):
                for label, w, _ in self._DIST_COLS:
                    dpg.add_table_column(
                        label=label,
                        width_fixed=True,
                        init_width_or_weight=w,
                        no_resize=True,
                        no_reorder=True,
                    )
            self.theme.track(
                dpg.add_text(
                    "Load a shapefile to populate this table.",
                    tag="district_info_empty_lbl", show=False,
                ),
                "muted",
            )

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

        # Stable district -> displayed label (matches map labelling)
        d_to_label = {}
        if initial is not None and len(initial) == len(assignment):
            from mosaic.gui.map_view import stable_color_mapping
            stable_colors = stable_color_mapping(assignment, initial, n_dist)
            for d in range(n_dist):
                mask = assignment == d
                if mask.any():
                    d_to_label[d] = int(stable_colors[mask][0]) + 1
        else:
            for d in range(n_dist):
                d_to_label[d] = d + 1

        # Display order: row r shows the district whose stable label is r+1.
        label_to_d = {lab - 1: d for d, lab in d_to_label.items()}

        # Rebuild rows only when district count changes
        if self._dist_table_built_rows != n_dist:
            dpg.delete_item("district_info_table", children_only=True, slot=1)
            for r in range(n_dist):
                with dpg.table_row(parent="district_info_table"):
                    for _, _, key in self._DIST_COLS:
                        dpg.add_text("", tag=f"di_r{r}_{key}")
            self._dist_table_built_rows = n_dist

        for r in range(n_dist):
            d = label_to_d.get(r, r)
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

    def _build_score_contrib_panel(self):
        self._contrib_bar_themes = []
        for _, _, rgba in _CONTRIB_BAR_METRICS:
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvBarSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Fill, rgba,
                        category=dpg.mvThemeCat_Plots,
                    )
            self._contrib_bar_themes.append(t)

        with dpg.window(
            label="Score Contributors", tag="panel_score_contrib",
            show=False, width=480, height=300,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_contrib_item, False),
        ):
            self.theme.text(
                "Share of total score each metric contributes - the optimizer "
                "minimizes the total, so taller bars are driving the annealing harder.",
                "muted",
                wrap=460,
            )
            dpg.add_spacer(height=2)
            with dpg.plot(height=-1, width=-1, no_mouse_pos=True):
                dpg.add_plot_axis(
                    dpg.mvXAxis, tag="contrib_x",
                    no_gridlines=True,
                )
                dpg.set_axis_ticks(
                    "contrib_x",
                    tuple((lbl, float(i + 1))
                          for i, (_, lbl, _) in enumerate(_CONTRIB_BAR_METRICS)),
                )
                dpg.set_axis_limits("contrib_x", 0.5, len(_CONTRIB_BAR_METRICS) + 0.5)
                with dpg.plot_axis(dpg.mvYAxis, label="%", tag="contrib_y"):
                    self._contrib_bar_series = []
                    for i, (name, _, _) in enumerate(_CONTRIB_BAR_METRICS):
                        s = dpg.add_bar_series(
                            [float(i + 1)], [0.0], weight=0.7,
                            label="##cb{}".format(i),
                        )
                        dpg.bind_item_theme(s, self._contrib_bar_themes[i])
                        self._contrib_bar_series.append(s)
            dpg.set_axis_limits("contrib_y", 0.0, 100.0)

    def _build_help_popup(self):
        with dpg.window(
            label="Help", tag="popup_help",
            show=False, width=460, height=400,
            pos=[(_VP_W - 460) // 2, (_VP_H - 400) // 2],
        ):
            with dpg.child_window(height=-40, border=False):
                self.theme.text(
                    "Mosaic from Matt Mohn (@mattmxhn)",
                    "title",
                )
                dpg.add_spacer(height=6)
                dpg.add_text(
                    "Mosaic uses simulated annealing plus recombination to generate "
                    "redistricting plans. Cooling makes it more selective over time; "
                    "recombination (merge two districts, draw a new boundary) is the "
                    "edit it tries on each step.",
                    wrap=420,
                )
                dpg.add_spacer(height=14)

                self.theme.text("Basic usage", "heading")
                dpg.add_separator()
                dpg.add_text(
                    "1. Import a shapefile\n"
                    "2. Map columns (population, county, votes) in the picker\n"
                    "3. Set district count, iterations, and score weights\n"
                    "4. Start - pause / reset / revert to best as needed\n"
                    "5. Save Assignments writes a CSV",
                    wrap=420,
                )
                dpg.add_spacer(height=14)

                self.theme.text("Full documentation", "heading")
                dpg.add_separator()
                dpg.add_text(
                    "The website has the complete reference: scoring formulas, "
                    "shapefile sources and requirements, install steps, "
                    "troubleshooting, and methodology notes.",
                    wrap=420,
                )
                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Open docs",
                        callback=lambda: webbrowser.open(_DOCS_URL),
                        width=110,
                    )
                    dpg.add_button(
                        label="Shapefile guide",
                        callback=lambda: webbrowser.open(_DOCS_SHAPEFILE_URL),
                        width=140,
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
        is_running      = status == AlgorithmStatus.RUNNING
        is_paused       = status == AlgorithmStatus.PAUSED
        is_partitioning = status == AlgorithmStatus.PARTITIONING
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
        loaded_gdf_id = (id(self.runner.gdf)
                         if self.runner and self.runner.gdf is not None
                         else 0)
        not_loading = status not in (AlgorithmStatus.LOADING,
                                     AlgorithmStatus.BUILDING_GRAPH)
        if (loaded_path
                and (loaded_path != self._map_loaded_path
                     or loaded_gdf_id != self._map_loaded_gdf_id)
                and not self._map_loading
                and not_loading
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
            pop_ref      = self.runner.populations
            mv = self.map_view
            def _bg_load():
                try:
                    mv.load(gdf_ref, county_array=county_array_ref,
                            dem_votes=dem_ref, gop_votes=gop_ref,
                            pp_data=pp_data_ref, populations=pop_ref)
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
            else:
                _assignment = _initial = None
                _n_dist = 0

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

        # ── Plots & side panels ──────────────────────────────────────────────
        self._update_plots_and_panels()

        # Keep district-count-dependent sliders bounded
        n_dist_val = dpg.get_value(self._num_districts)
        dpg.configure_item(self._target_dem_seats, max_value=n_dist_val)
        dpg.configure_item(self._hinge_threshold,  max_value=n_dist_val)

        # ── Button states ─────────────────────────────────────────────────────
        dpg.configure_item("menu_scores",    enabled=not is_busy)
        dpg.configure_item(self._run_btn,    enabled=not is_busy)
        dpg.configure_item(self._pause_btn,  enabled=is_running or is_paused or is_partitioning)
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
            self.theme.nudge_theme if (not is_busy and graph_ready)
            else self.theme.antinudge_theme,
        )
        dpg.bind_item_theme(
            self._pause_btn,
            self.theme.nudge_theme if (is_running or is_paused)
            else self.theme.antinudge_theme,
        )
        dpg.bind_item_theme(self._reset_btn, self.theme.antinudge_theme)
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
        dpg.bind_item_theme(
            self._revert_btn,
            self.theme.nudge_theme if can_revert else self.theme.antinudge_theme,
        )

    def _update_plots_and_panels(self) -> None:
        """Per-frame plot redraws and side panel refreshes."""
        # ── Plots ─────────────────────────────────────────────────────────────
        # One lock acquisition, copying only the delta since the last call.
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
            _pvd  = list(self.state.pop_deviation_history[self._buf_popdev.read:])
            _pvd_max  = list(self.state.pop_dev_max_history[self._buf_popdev_max.read:])
            _pvd_mean = list(self.state.pop_dev_mean_history[self._buf_popdev_mean.read:])
            _cutd = list(self.state.cut_edges_history[self._buf_cuts.read:])
            _mjd  = list(self.state.majority_dem_history[self._buf_maj_dem.read:])
            _mjr  = list(self.state.majority_rep_history[self._buf_maj_rep.read:])
            _hgd  = list(self.state.hinge_history[self._buf_hinge.read:])

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
        self._buf_pp.add([v * 100.0 for v in _pd])
        self._buf_popdev.add(_pvd)
        self._buf_popdev_max.add(_pvd_max)
        self._buf_popdev_mean.add(_pvd_mean)
        self._buf_cuts.add(_cutd)
        self._buf_maj_dem.add([v * 100.0 for v in _mjd])
        self._buf_maj_rep.add([v * 100.0 for v in _mjr])
        self._buf_hinge.add(_hgd)

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
                _render(self._buf_cs_score,  "cs_score_series",  "cs_score_x",  "cs_score_y")
                _render(self._buf_cs_excess, "cs_excess_series", "cs_excess_x", "cs_excess_y")
                _render(self._buf_cs_clean,  "cs_clean_series",  "cs_clean_x",  "cs_clean_y")
                if self._buf_cs_score.ys:
                    _, w = self._buf_cs_score.plot_data(limit)
                    dpg.set_axis_limits("cs_score_y", 0, max(w) * 1.05)
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
        if dpg.is_item_shown("panel_majority"):
            dpg.configure_item("majority_plot_grp",     show=self._has_elections)
            dpg.configure_item("majority_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_maj_dem, "maj_dem_series", "maj_x", "maj_y", fit_y=False)
                _render(self._buf_maj_rep, "maj_rep_series", "maj_x", "maj_y", fit_y=False)
        if dpg.is_item_shown("panel_hinge"):
            dpg.configure_item("hinge_plot_grp",     show=self._has_elections)
            dpg.configure_item("hinge_inactive_lbl", show=not self._has_elections)
            if self._has_elections:
                _render(self._buf_hinge, "hinge_series", "hinge_x", "hinge_y", fit_y=False)
        if dpg.is_item_shown("panel_pp"):
            dpg.configure_item("pp_plot_grp",     show=pp_on)
            dpg.configure_item("pp_inactive_lbl", show=not pp_on)
            if pp_on:
                _render(self._buf_pp, "pp_series", "pp_x", "pp_y")
        if dpg.is_item_shown("panel_popdev"):
            popdev_on = dpg.get_value(self._popdev_enabled)
            dpg.configure_item("popdev_plot_grp",     show=popdev_on)
            dpg.configure_item("popdev_inactive_lbl", show=not popdev_on)
            if popdev_on:
                _render(self._buf_popdev_max,  "popdev_max_series",  "popdev_x", "popdev_y")
                _render(self._buf_popdev_mean, "popdev_mean_series", "popdev_x", "popdev_y")
                # Let DPG auto-fit y to data; the constraint pins lower bound to 0.
                dpg.fit_axis_data("popdev_y")
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

    def _clear_all_series(self) -> None:
        """Clear local history buffers and blank all DPG plot series."""
        for buf in (
            self._buf_score, self._buf_acc, self._buf_temp,
            self._buf_cs_score, self._buf_cs_excess, self._buf_cs_clean,
            self._buf_mm, self._buf_eg, self._buf_seats,
            self._buf_comp, self._buf_pp, self._buf_popdev,
            self._buf_popdev_max, self._buf_popdev_mean, self._buf_cuts,
            self._buf_maj_dem, self._buf_maj_rep, self._buf_hinge,
        ):
            buf.clear()
        empty = [[], []]
        for tag in (
            "score_series", "acc_series", "panel_temp_series",
            "cs_score_series", "cs_excess_series", "cs_clean_series",
            "mm_series", "eg_series", "seats_series",
            "comp_series", "pp_series",
            "popdev_max_series", "popdev_mean_series", "cuts_series",
            "maj_dem_series", "maj_rep_series", "hinge_series",
        ):
            dpg.set_value(tag, empty)
        for ax in (
            "score_x", "score_y", "acc_x", "acc_y",
            "panel_temp_x", "panel_temp_y",
            "cs_score_x", "cs_score_y",
            "cs_excess_x", "cs_excess_y",
            "cs_clean_x", "cs_clean_y",
            "mm_x", "mm_y", "eg_x", "eg_y",
            "seats_x", "seats_y", "comp_x", "comp_y",
            "pp_x", "pp_y", "popdev_x", "popdev_y", "cuts_x", "cuts_y",
            "maj_x", "maj_y", "hinge_x", "hinge_y",
        ):
            dpg.set_axis_limits_auto(ax)
        # Re-pin axes whose ranges should always be locked (probability bands).
        # Without this, set_axis_limits_auto above releases them and the next
        # render auto-fits to whatever the data happens to be, causing the
        # axis to "stick" at a small range like [0, 0.1].
        dpg.set_axis_limits("pp_y",    0.0, 100.0)
        dpg.set_axis_limits("maj_y",   0.0, 100.0)
        dpg.set_axis_limits("hinge_y", 0.0, 1.0)
        for s in self._partisan_bar_series:
            dpg.set_value(s, empty)
        for s in self._win_chance_bar_series:
            dpg.set_value(s, empty)
        for i, s in enumerate(self._contrib_bar_series):
            dpg.set_value(s, [[float(i + 1)], [0.0]])

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
        self.theme.retoken(self._shp_info, "success_pale")

        # Enable/disable county-dependent controls based on what was loaded
        has_county = cfg.county_col is not None
        self._has_county = has_county
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
        _pt_token = "secondary" if has_elections else "disabled_deep"
        for chk, lbl in [
            (self._mm_enabled,       self._mm_lbl),
            (self._eg_enabled,       self._eg_lbl),
            (self._comp_enabled,     self._comp_lbl),
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
                (self._comp_enabled,     "comp_controls"),
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
            (self._panel_seats_item,      "panel_dem_seats"),
            (self._panel_comp_item,       "panel_comp"),
            (self._panel_majority_item,   "panel_majority"),
            (self._panel_hinge_item,      "panel_hinge"),
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

    def _on_launch_watch_toggle(self):
        dpg.configure_item("launch_watch_controls",
                           show=dpg.get_value(self._launch_watch_enabled))

    def _on_cut_toggle(self):
        en = dpg.get_value(self._cut_enabled)
        self.theme.retoken(self._cut_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("cut_edge_controls", show=en)

    def _on_cs_toggle(self):
        en = dpg.get_value(self._cs_enabled)
        self.theme.retoken(self._cs_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("cs_controls", show=en)
        # Default intent: enabling County Splits also turns on County-Edge Bias.
        # User can still toggle bias off separately if they want.
        if en:
            dpg.set_value(self._county_bias_enabled, True)
            dpg.configure_item("county_bias_controls", show=True)

    def _on_pp_toggle(self):
        en = dpg.get_value(self._pp_enabled)
        self.theme.retoken(self._pp_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("pp_controls", show=en)

    def _on_popdev_score_toggle(self):
        en = dpg.get_value(self._popdev_enabled)
        self.theme.retoken(self._popdev_lbl,
                           "accent_green" if en else "secondary")
        dpg.configure_item("popdev_controls", show=en)

    def _on_mm_toggle(self):
        en = dpg.get_value(self._mm_enabled)
        self.theme.retoken(self._mm_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("mm_controls", show=en)

    def _on_eg_toggle(self):
        en = dpg.get_value(self._eg_enabled)
        self.theme.retoken(self._eg_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("eg_controls", show=en)

    def _on_comp_toggle(self):
        en = dpg.get_value(self._comp_enabled)
        self.theme.retoken(self._comp_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("comp_controls", show=en)

    def _on_seats_toggle(self):
        en = dpg.get_value(self._seats_enabled)
        self.theme.retoken(self._seats_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("seats_controls", show=en)

    def _on_majority_toggle(self):
        en = dpg.get_value(self._majority_enabled)
        self.theme.retoken(self._majority_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("majority_controls", show=en)

    def _on_majority_dem_chk(self):
        if dpg.get_value(self._majority_dem_chk):
            dpg.set_value(self._majority_rep_chk, False)

    def _on_majority_rep_chk(self):
        if dpg.get_value(self._majority_rep_chk):
            dpg.set_value(self._majority_dem_chk, False)

    def _on_hinge_toggle(self):
        en = dpg.get_value(self._hinge_enabled)
        self.theme.retoken(self._hinge_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("hinge_controls", show=en)

    def _on_hinge_dem_chk(self):
        if dpg.get_value(self._hinge_dem_chk):
            dpg.set_value(self._hinge_rep_chk, False)

    def _on_hinge_rep_chk(self):
        if dpg.get_value(self._hinge_rep_chk):
            dpg.set_value(self._hinge_dem_chk, False)

    def _set_score_row_vis(self, row_tag: str, show: bool,
                           chk_item: int, toggle_cb) -> None:
        """Show/hide a score row; if hiding, force-disable the checkbox."""
        dpg.configure_item(row_tag, show=show)
        if not show and dpg.get_value(chk_item):
            dpg.set_value(chk_item, False)
            toggle_cb()

    def _on_county_bias_toggle(self):
        dpg.configure_item("county_bias_controls",
                           show=dpg.get_value(self._county_bias_enabled))

    def _hint(self, widget: int | str, key: str, delay: float = 0.6) -> None:
        """Attach a hover tooltip to ``widget`` using the named entry from ``_HINTS``.

        ``delay`` is the hover time (seconds) before the tooltip appears.
        """
        text = _HINTS.get(key)
        if text is None:
            return
        with dpg.tooltip(parent=widget, delay=delay):
            dpg.add_text(text, wrap=320)

    def _tooltip(self, widget: int | str, text: str, delay: float = 0.4) -> None:
        """Attach a hover tooltip with literal ``text`` (no _HINTS lookup)."""
        with dpg.tooltip(parent=widget, delay=delay):
            dpg.add_text(text, wrap=320)

    def _show_panel(self, panel_tag: str, menu_item) -> None:
        visible = dpg.is_item_shown(panel_tag)
        dpg.set_value(menu_item, not visible)
        dpg.configure_item(panel_tag, show=not visible)

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

    def _on_panel_win_chance_toggle(self):
        dpg.configure_item("panel_win_chance", show=dpg.get_value(self._panel_win_chance_item))

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

    def _on_panel_popdev_toggle(self):
        dpg.configure_item("panel_popdev", show=dpg.get_value(self._panel_popdev_item))

    def _on_panel_cuts_toggle(self):
        dpg.configure_item("panel_cut_edges", show=dpg.get_value(self._panel_cuts_item))

    def _on_panel_majority_toggle(self):
        dpg.configure_item("panel_majority", show=dpg.get_value(self._panel_majority_item))

    def _on_panel_hinge_toggle(self):
        dpg.configure_item("panel_hinge", show=dpg.get_value(self._panel_hinge_item))

    def _on_panel_contrib_toggle(self):
        dpg.configure_item("panel_score_contrib", show=dpg.get_value(self._panel_contrib_item))

    def _on_panel_district_toggle(self):
        showing = dpg.get_value(self._panel_district_item)
        dpg.configure_item("panel_district_info", show=showing)
        # Force a fresh update next time we render the panel.
        self._dist_info_last_iter = -1

    def _on_district_panel_close(self):
        dpg.set_value(self._panel_district_item, False)
        self._dist_info_last_iter = -1

    def _on_dist_info_interval_change(self):
        # Lower threshold should take effect immediately.
        self._dist_info_last_iter = -1

    def _on_theme_change(self):
        choice = dpg.get_value(self._theme_radio)
        self.theme.apply("dark" if choice == "Dark" else "light")
        self._sync_map_bg_to_theme()
        self._sync_ref_lines_to_theme()

    def _sync_ref_lines_to_theme(self):
        """Rebind the partisan/win-chance 50/50 and median guide lines so
        they stay readable on the current plot background (white guides on
        dark mode, black on light)."""
        t = self._partisan_ref_themes[self.theme.palette.name]
        self._partisan_ref_theme = t
        for tag in ("partisan_ref", "partisan_median",
                    "win_chance_ref", "win_chance_median",
                    "cs_clean_max"):
            if dpg.does_item_exist(tag):
                dpg.bind_item_theme(tag, t)

    def _sync_map_bg_to_theme(self):
        """Push the theme's child_bg into the map view and refresh the texture."""
        r, g, b, _ = self.theme.color("child_bg")
        # Update MapView's bg so future LUT rebuilds use it.
        if self.map_view is not None:
            self.map_view._bg_color = np.array([r, g, b, 255], dtype=np.uint8)
        # If a shapefile is loaded, force the map to re-render with the new bg.
        if self.map_view is not None and self.map_view._loaded:
            self.state.update(map_needs_update=True)
            return
        # Otherwise paint the empty texture directly.
        rgba = np.tile(
            np.array([r / 255.0, g / 255.0, b / 255.0, 1.0], dtype=np.float32),
            _MAP_DW * _MAP_DH,
        )
        dpg.set_value("map_texture", rgba)

    def _on_county_overlay_toggle(self):
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.county_overlay = dpg.get_value(self._county_overlay)
        self.state.update(map_needs_update=True)

    def _on_partisan_overlay_toggle(self):
        if dpg.get_value(self._partisan_overlay):
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
        self.state.update(map_needs_update=True)

    def _on_district_partisan_toggle(self):
        if dpg.get_value(self._district_partisan):
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
        self.state.update(map_needs_update=True)

    def _on_splits_view_toggle(self):
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.splits_view = dpg.get_value(self._splits_view)
        self.state.update(map_needs_update=True)

    def _on_compactness_toggle(self):
        if dpg.get_value(self._compactness_view):
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
        self.state.update(map_needs_update=True)

    def _on_pop_dev_toggle(self):
        if dpg.get_value(self._pop_dev_view):
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
        self.state.update(map_needs_update=True)

    def _on_labels_toggle(self):
        if self.map_view is None or not self.map_view._loaded:
            return
        self.map_view.show_labels = dpg.get_value(self._show_labels)
        self.state.update(map_needs_update=True)

    # ── Action callbacks ──────────────────────────────────────────────────────

    def _on_import_shapefile(self):
        # Mixing tkinter's Tk root with Dear PyGui crashes on macOS (both
        # frameworks contend for the Cocoa main run loop). Use DPG's native
        # file dialog instead.
        if dpg.does_item_exist("__shp_file_dialog"):
            dpg.delete_item("__shp_file_dialog")
        from mosaic.paths import shapefiles_dir, mosaic_data_dir
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

    def _on_shapefile_selected(self, sender, app_data):
        path = app_data.get("file_path_name", "") if isinstance(app_data, dict) else ""
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
        self.theme.retoken(self._shp_info, "muted")
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
            self.state.pop_deviation_history = []
            self.state.pop_dev_max_history = []
            self.state.pop_dev_mean_history = []
            self.state.cut_edges_history = []
            self.state.majority_dem_history = []
            self.state.majority_rep_history = []
            self.state.hinge_history = []
        self._clear_all_series()
        dpg.set_value(self._shp_info, "Building graph...")
        self.theme.retoken(self._shp_info, "muted")
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
            self.state.pop_deviation_history = self.state.pop_deviation_history[:n_score]
            self.state.pop_dev_max_history   = self.state.pop_dev_max_history[:n_score]
            self.state.pop_dev_mean_history  = self.state.pop_dev_mean_history[:n_score]
            self.state.cut_edges_history        = self.state.cut_edges_history[:n_score]

        # Tell the worker to stop, then actually wait for it to exit before
        # we touch state.  Previously we set should_stop=False in the same
        # frame as request_stop(); if the worker was sitting in its
        # pause-wait loop it could observe the cleared flags and fall back
        # into normal iteration — i.e., "paused then unpaused on revert".
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
            self._buf_seats, self._buf_comp, self._buf_pp, self._buf_popdev,
            self._buf_popdev_max, self._buf_popdev_mean, self._buf_cuts,
            self._buf_maj_dem, self._buf_maj_rep,
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
        self.theme.retoken(self._shp_info, "muted")

    def _on_run(self):
        if self.runner is None or self.runner.graph is None:
            self.state.update(status=AlgorithmStatus.ERROR,
                              error_message="Please load a shapefile first")
            return

        cs_on      = dpg.get_value(self._cs_enabled)
        mm_on      = dpg.get_value(self._mm_enabled)
        eg_on      = dpg.get_value(self._eg_enabled)
        comp_on    = dpg.get_value(self._comp_enabled)
        seats_on   = dpg.get_value(self._seats_enabled)
        maj_on     = dpg.get_value(self._majority_enabled)
        robust_eg = dpg.get_value(self._eg_mode) == "Robust (recommended)"

        w_cut  = (dpg.get_value(self._w_cut_edges)
                  if dpg.get_value(self._cut_enabled) else 0.0)
        w_cs   = dpg.get_value(self._w_county_splits) if cs_on else 0.0
        w_pp   = (dpg.get_value(self._w_polsby_popper)
                  if dpg.get_value(self._pp_enabled) else 0.0)
        w_pd   = (dpg.get_value(self._w_pop_deviation)
                  if dpg.get_value(self._popdev_enabled) else 0.0)
        # Safe harbor cannot exceed population tolerance
        _tol    = dpg.get_value(self._tolerance) / 100.0
        _harbor = min(dpg.get_value(self._pop_dev_harbor) / 100.0, _tol)

        n_dist_run = dpg.get_value(self._num_districts)
        raw_target_seats = dpg.get_value(self._target_dem_seats)
        target_seats = float(max(1, min(raw_target_seats, n_dist_run - 1)))

        score_cfg = ScoreConfig(
            weight_cut_edges=w_cut,
            weight_county_splits=w_cs,
            weight_polsby_popper=w_pp,
            weight_pop_deviation=w_pd,
            pop_deviation_safe_harbor=_harbor,
            weight_mean_median=dpg.get_value(self._w_mean_median) if mm_on else 0.0,
            target_mean_median=dpg.get_value(self._target_mean_median) if mm_on else 0.0,
            weight_efficiency_gap=dpg.get_value(self._w_efficiency_gap) if eg_on else 0.0,
            target_efficiency_gap=dpg.get_value(self._target_efficiency_gap) if eg_on else 0.0,
            use_robust_eg=robust_eg,
            weight_dem_seats=dpg.get_value(self._w_dem_seats) if seats_on else 0.0,
            target_dem_seats=target_seats,
            weight_competitiveness=dpg.get_value(self._w_competitiveness) if comp_on else 0.0,
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

        cb_en  = cs_on and dpg.get_value(self._county_bias_enabled)
        cb_val = dpg.get_value(self._county_bias)

        self.state.update(
            num_districts=dpg.get_value(self._num_districts),
            pop_tolerance=_tol,
            max_iterations=dpg.get_value(self._iterations),
            seed=seed,
            score_config=score_cfg,
            annealing_config=ann_cfg,
            map_render_interval=dpg.get_value(self._map_interval),
            county_bias_enabled=cb_en,
            county_bias=cb_val,
            n3_probability=dpg.get_value(self._n3_pct) / 100.0,
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
            self.state.county_clean_districts_history = []
            self.state.mm_history = []
            self.state.eg_history = []
            self.state.dem_seats_history = []
            self.state.competitive_count_history = []
            self.state.pp_history = []
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
            current_temperature=0.0,
            accepted_worse=0,
            rejected_worse=0,
            best_assignment=None,
            current_assignment=None,
            score_breakdown={},
        )
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
        if self.map_view is not None and self.map_view._loaded:
            self.map_view.draw_blank()

    def _stable_labeled_best_assignment(self) -> "np.ndarray | None":
        """Best assignment relabeled to match the live map / District Info labels.

        ReCom internally renumbers districts over the course of a run; the GUI
        uses ``stable_color_mapping`` against the initial partition to keep the
        same physical region under the same number across iterations. CSV
        exports must apply the same permutation, otherwise users see one label
        in Mosaic and a different one in their spreadsheet.
        """
        best = self.state.best_assignment
        if best is None:
            return None
        initial = self.state.initial_assignment
        n_dist = self.state.num_districts or int(best.max()) + 1
        if initial is not None and len(initial) == len(best):
            from mosaic.gui.map_view import stable_color_mapping
            return stable_color_mapping(best, initial, n_dist)
        return best

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
            self._stable_labeled_best_assignment(),
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
            self._stable_labeled_best_assignment(),
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
