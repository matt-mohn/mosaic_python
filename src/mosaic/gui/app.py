"""Main Dear PyGui application -- two-column layout with live district map."""

import logging
import threading
import time
import warnings
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger("mosaic")

_DOCS_URL = "https://matt-mohn.github.io/mosaic_python/"
_UPDATE_CHECK_URL = (
    "https://raw.githubusercontent.com/matt-mohn/mosaic_python/main/pyproject.toml"
)
_DOCS_SHAPEFILE_URL = "https://matt-mohn.github.io/mosaic_python/shapefiles.html"
_DOWNLOAD_URL = "https://matt-mohn.github.io/mosaic_python/install.html"

_ASSETS_DIR   = Path(__file__).resolve().parent.parent / "assets"
# App-level settings/cache directory — separate from output/ which holds generated files.
_SETTINGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".mosaic"
_RECENT_FILE  = _SETTINGS_DIR / "recent_shapefiles.json"
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

from mosaic import __version__

_PLOT_LIMIT   = 10_000   # max points rendered when limit-plots is on
_COMPACT_AT   = 20_000   # compact local buffer when it exceeds this
_COMPACT_KEEP = 10_000   # keep last N at full resolution after compaction
_COMPACT_THIN = 50       # keep 1-in-N for old data (~200 pts per 10k iters)

# Phase plot: a metric-vs-metric trajectory whose tail cools with age (hot head
# -> dark blue). Histories append in lockstep, so any two are index-aligned.
_PHASE_BANDS  = 24        # recency-ramp line-series (static colours; bin by age)
_PHASE_HEAD   = 400       # newest points kept full-res (never strided away)
_PHASE_BUDGET = 1400      # total points plotted per redraw (head + strided tail)
_PHASE_TAU    = 2000.0    # recency falloff (iters): hot->mid ~1 TAU, cold by ~3
_PHASE_FOLLOW = 1000      # "Follow dot" fits the last this-many iterations
_PHASE_SMOOTH_WIN = 7     # moving-average window when Smooth is on
_PHASE_MIN_SPAN = 0.05    # floor on axis span so tiny-range metrics don't over-zoom

# (label, history attr, display transform). "rating" = 100 - penalty (higher is
# better), "x100" = 0-1 fraction -> 0-100, "raw" = as stored. Chosen so each
# label reads in its natural direction.
_PHASE_METRICS: list[tuple[str, str, str]] = [
    ("Cut Edges",            "cut_edges_history",                 "raw"),
    ("Compactness",          "holistic_compactness_history",      "rating"),
    ("Polsby-Popper",        "pp_history",                        "x100"),
    ("Reock",                "reock_history",                     "x100"),
    ("County Congruence",    "holistic_splitting_history",        "raw"),
    ("County Excess Splits", "county_excess_splits_history",      "raw"),
    ("Pop Dev max %",        "pop_dev_max_history",               "raw"),
    ("Pop Dev mean %",       "pop_dev_mean_history",              "raw"),
    ("Score (total)",        "score_history",                     "raw"),
    ("Temperature",          "temperature_history",               "raw"),
    # Partisan — dropped from the pickers when no election data is loaded.
    ("Efficiency Gap",       "eg_history",                        "raw"),
    ("Efficiency Gap (abs)", "eg_history",                        "abs"),
    ("Mean-Median",          "mm_history",                        "raw"),
    ("Mean-Median (abs)",    "mm_history",                        "abs"),
    ("Expected Dem Seats",   "dem_seats_history",                 "raw"),
    ("Proportionality",      "holistic_proportionality_history",  "rating"),
    ("Competitiveness",      "holistic_competitiveness_history",  "rating"),
    ("Dem Majority",         "majority_dem_history",              "x100"),
    ("Rep Majority",         "majority_rep_history",              "x100"),
    ("Hinge",                "hinge_history",                     "x100"),
]
_PHASE_ATTR   = {lbl: attr for lbl, attr, _ in _PHASE_METRICS}
_PHASE_KIND   = {lbl: kind for lbl, _, kind in _PHASE_METRICS}
_PHASE_LABELS = [lbl for lbl, _, _ in _PHASE_METRICS]

# Axis-title unit hints so the comet axes read like the per-metric panels
# (e.g. a probability shows "(%)"). Bare label if a metric isn't listed.
_PHASE_AXIS_UNIT = {
    "Compactness":        "(100 = best)",
    "Polsby-Popper":      "(100 = circle)",
    "Reock":              "(100 = circle)",
    "County Congruence": "(penalty, 0 = best)",
    "Proportionality":    "(100 = best)",
    "Competitiveness":    "(100 = best)",
    "Dem Majority":       "(%)",
    "Rep Majority":       "(%)",
    "Hinge":              "(%)",
}

# Recency ramps (old -> hot). Dark = magma (bright-on-black); light = mako
# reversed (pale mint old -> near-black recent) so recent reads dark on white.
_PHASE_RAMP = [
    (0.00, (48, 60, 135)),  (0.22, (90, 68, 165)),  (0.40, (170, 60, 120)),
    (0.60, (228, 90, 60)),  (0.80, (250, 160, 40)), (1.00, (252, 253, 191)),
]
_PHASE_RAMP_LIGHT = [
    (0.00, (222, 245, 229)),  # oldest: pale mint
    (0.22, (100, 200, 140)),  # green
    (0.42, ( 45, 148, 142)),  # teal
    (0.62, ( 54,  93, 141)),  # blue
    (0.80, ( 52,  38,  86)),  # indigo
    (1.00, ( 14,   5,   8)),  # hottest: near-black
]


def _ramp_color(t: float, ramp=_PHASE_RAMP) -> tuple[int, int, int]:
    """Sample a recency ramp at t in [0, 1] (old -> hot)."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    for i in range(1, len(ramp)):
        p1, c1 = ramp[i]
        if t <= p1:
            p0, c0 = ramp[i - 1]
            f = (t - p0) / (p1 - p0) if p1 > p0 else 0.0
            return tuple(int(a + (b - a) * f) for a, b in zip(c0, c1))
    return ramp[-1][1]


def _phase_transform(kind: str, arr):
    """Map a stored history array to its display units (see _PHASE_METRICS)."""
    if kind == "rating":
        return 100.0 - arr
    if kind == "x100":
        return arr * 100.0
    if kind == "abs":
        return np.abs(arr)
    return arr
# DPG's set_value + fit_axis_data iterate the full point list under the GIL
# on every frame.  At 60 fps that single-thread cost steals time from the
# algorithm worker.  Cap how many points we *send* to DPG (the buffer keeps
# every value it already keeps — this is a render-time stride only).  A
# ~400-px-wide plot can't visually resolve more than ~1k points anyway.
_RENDER_TARGET = 1500


_CAMERA_ICON_SIZE = 20


def _build_camera_icon(fg_rgb: tuple[int, int, int]) -> np.ndarray:
    """Procedural 20x20 RGBA camera glyph as a flat float32 array (for DPG raw texture).

    The 'hole' through the lens uses the inverse of fg so the icon stays
    legible on both light and dark button backgrounds.
    """
    from PIL import Image, ImageDraw
    w = h = _CAMERA_ICON_SIZE
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fg = (*fg_rgb, 255)
    inv = (255 - fg_rgb[0], 255 - fg_rgb[1], 255 - fg_rgb[2], 255)
    d.rectangle((4, 3, 9, 5), fill=fg)                       # viewfinder bump
    d.rounded_rectangle((1, 5, 18, 17), radius=2, fill=fg)   # body
    d.ellipse((6, 7, 14, 15), fill=inv)                      # lens hole
    d.ellipse((8, 9, 12, 13), fill=fg)                       # lens highlight
    return (np.asarray(img, dtype=np.float32) / 255.0).ravel()


def _build_more_icon(fg_rgb: tuple[int, int, int]) -> np.ndarray:
    """Procedural 20x20 RGBA three-dot 'more options' glyph (for raw texture)."""
    from PIL import Image, ImageDraw
    w = h = _CAMERA_ICON_SIZE
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fg = (*fg_rgb, 255)
    cy = 9
    for cx in (4, 9, 14):
        d.ellipse((cx, cy, cx + 3, cy + 3), fill=fg)
    return (np.asarray(img, dtype=np.float32) / 255.0).ravel()


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
_DIR_TO_MODE = {"Fair": "fair", "D": "favor_dem", "R": "favor_rep"}

_CONTRIB_BAR_METRICS = [
    ("Cut Edges",       "Cuts",   (160, 160, 165, 220)),
    ("Excess Splits",    "Co.Exc", (190, 170, 130, 220)),
    ("Single-County Districts", "1-Co Dist", (160, 200, 130, 220)),
    ("Population Deviation", "PopDev", (220, 200, 70,  220)),
    ("Alignment",       "Align", (150, 120, 210, 220)),
    ("Polsby-Popper",   "PP",    (90,  160, 220, 220)),
    ("Reock",           "Reock", (60,  190, 200, 220)),
    ("Mean-Median",     "MM",     (240, 140, 60,  220)),
    ("Efficiency Gap",  "EG",     (225, 75,  75,  220)),
    ("Dem Seats",       "Seats",  (180, 80,  220, 220)),
    ("D Majority",      "D Maj",  (70,  130, 210, 220)),
    ("R Majority",      "R Maj",  (210, 70,  70,  220)),
    ("Hinge",           "Hinge",  (140, 90,  200, 220)),
    # Single-rating composites
    ("Compactness",              "Comp",    (220, 160, 60,  220)),
    ("County Congruence",        "Cty Cong", (200, 140, 50, 220)),
    ("Proportionality",          "Prop",    (170, 90,  50,  220)),
    ("Competitiveness",          "Cmptv",   (50,  150, 90,  220)),
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

# ── Dialog design language ────────────────────────────────────────────────────
# Shared chrome/spacing for every little pop-out window (see _dialog / _dialog_
# footer). Keeps the secondary windows in one visual language instead of each
# reinventing size, centring, and button styling.
_DIALOG_PAD   = 10     # window_padding.x — matches theme _Style.window_padding
_DIALOG_GAP   = 8      # item_spacing.x between footer buttons
_DIALOG_BTN_W = 90     # standard footer button width
_DIALOG_RM    = 6      # right margin inside the content region for the footer
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
    "alignment": (
        "How close the plan stays to a loaded reference plan - a least-change "
        "penalty. A reference district kept whole costs nothing; one split "
        "apart costs more. Load a reference CSV first; 0 = identical to it."
    ),
    "alignment_focus": (
        "Whose retention to measure: all residents, or a party's voters - so a "
        "district scores on how many of its own partisans stayed together. "
        "Needs election data."
    ),
    "alignment_restrict": (
        "Score only the reference districts the focus party already wins, so "
        "you keep your own seats. Needs a party focus and election data."
    ),
    "compactness": (
        "Polsby-Popper: ratio of district area to a circle with the same perimeter. "
        "1.0 = perfectly round, lower means stretched or jagged. "
        "Optimizer uses (1 - PP) as the penalty."
    ),
    "reock": (
        "Reock: ratio of district area to its bounding-circle area. "
        "1.0 = perfect circle, lower means stretched. Complements Polsby-Popper, "
        "which measures boundary smoothness rather than overall roundness."
    ),
    "holistic_compactness": (
        "One 0-100 compactness dial blending Polsby-Popper and Reock 50/50. "
        "Higher = more compact. Use it instead of tuning the two separately."
    ),
    "holistic_splitting": (
        "One splitting penalty (0 = best) blending county and district splits "
        "50/50. Rewards lopsided splits over even ones, weighted by population so "
        "splitting a big city costs more. 'Unclipped' lets the penalty climb past "
        "the scorecard cap so the optimizer keeps a gradient on heavily-split plans."
    ),
    "holistic_proportionality": (
        "One 0-100 rating of seat share vs vote share. Pulls toward proportional "
        "seats for the statewide vote; antimajoritarian plans get an instant 100."
    ),
    "holistic_competitiveness": (
        "One 0-100 rating that rewards districts near a 50/50 win probability. "
        "The credit tapers off as seats get safer, and caps once about 75% of "
        "districts are competitive."
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
    "dem_seats": (
        "Expected Democratic-won districts under the swing model. Directional: "
        "the D toggle pulls toward more Dem seats, R toward fewer."
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


def _fmt_dur(secs: float) -> str:
    """Duration as M:SS, promoting to H:MM:SS only once it passes an hour (no
    leading 0: hour mark on short runs). Lives in fixed-width table columns, so
    a field growing a digit never reflows the rest of the readout."""
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


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

        # Recent shapefiles (path + column config), loaded from disk in setup()
        self._recent_shapefiles: list = []   # [{"path": str, "config": dict}]
        self._file_save_asgn_item = 0        # File > Save Assignments menu item
        self._file_save_metrics_item = 0     # File > Save District Info menu item
        self._save_name_type: str = ""       # "assignments" | "metrics"
        self._save_name_input = 0            # filename text-input in save dialog
        # When set, the next inspection-complete event skips the column picker
        # and uses this config directly (one-click recent-file open).
        self._pending_recent_config: Optional[ShapefileConfig] = None
        # If True, auto-enable partisan overlay after the next Recent load completes.
        self._restore_partisan_on_load: bool = False

        # Map background-load tracking (app-local, no SharedState).
        # Tracking by (path, gdf id) so a re-import of the same path with
        # edited content still triggers a fresh MapView load — otherwise the
        # map keeps stale dimensions and render_assignment indexes past the
        # new assignment array.
        self._map_loading: bool = False
        self._map_ready: bool = False
        self._map_loaded_path: str = ""
        self._map_loaded_gdf_id: int = 0
        # gdf id whose complete_load has fully finished (pulsed via gdf_ready).
        # The map may only load a gdf that reached this point, so it never
        # captures the runner mid-populate. See the map bg-load gate below.
        self._map_data_gdf_id: int = 0

        # Plot appearance toggle (app-local)
        self._limit_plots: int | str = ""   # DPG checkbox tag, set during setup

        self._contrib_bar_series: list = []

        # Track what data the current shapefile has
        self._has_elections: bool = False

        # Relight: a "continue-refining" mode that reseeds each run from the
        # current on-screen map (mutually exclusive with Hot Start). _saved holds
        # the annealing-control values captured when armed, restored on clear.
        self._relight_active: bool = False
        self._relight_saved: Optional[dict] = None

        # Tracks whether the last frame was in a "running" state, so we can
        # trigger a one-shot precise-label re-render on transitions out of
        # running (cheap centroid labels -> pole-of-inaccessibility labels).
        self._labels_were_fast: bool = False

        # Renumber settings. Source of truth for the Advanced > Renumber controls.
        # _renumber_enabled mirrors the "Renumber districts after run" check and
        # the radio's None option; _renumber_rule is the last chosen sweep.
        self._renumber_enabled: bool = True
        self._renumber_rule: str = "proximity"  # "nw_se" | "n_s" | "proximity"
        # Cached precinct centroids (x, y) in gdf CRS, keyed by gdf identity, so
        # repeated renumbers don't recompute geometry.centroid each time.
        self._renumber_centroids = None         # (gdf_id, x_arr, y_arr) | None

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
        self._buf_pp        = _SeriesBuffer()
        self._buf_reock     = _SeriesBuffer()
        self._buf_hc        = _SeriesBuffer()
        self._buf_hsplit    = _SeriesBuffer()
        self._buf_hprop     = _SeriesBuffer()
        self._buf_hcmp      = _SeriesBuffer()
        self._buf_popdev     = _SeriesBuffer()
        self._buf_popdev_max = _SeriesBuffer()
        self._buf_popdev_mean = _SeriesBuffer()
        self._buf_align_mean = _SeriesBuffer()
        self._buf_align_min  = _SeriesBuffer()
        self._buf_cuts      = _SeriesBuffer()
        self._buf_maj_dem   = _SeriesBuffer()
        self._buf_maj_rep   = _SeriesBuffer()
        self._buf_hinge     = _SeriesBuffer()

        # Phase plot: selected metric labels + view / smoothing prefs.
        self._phase_x_label = "Compactness"
        self._phase_y_label = "Efficiency Gap"
        self._phase_fit_all = True         # Fit all (sticky) vs Follow dot
        self._phase_smooth = True
        self._phase_fade = True            # alpha fade with age vs solid trail
        self._phase_metric_sig = None      # last active-metric set the combos synced to
        self._phase_lim = None             # sticky (xlo,xhi,ylo,yhi) for "Fit all"
        self._phase_prev_n = 0             # detect a run reset (history shrank)

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
        self._build_save_name_popup()
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
                    dpg.add_menu_item(
                        label="Open output directory...",
                        callback=self._on_open_output_dir,
                    )
                    dpg.add_menu_item(
                        label="Check for updates...",
                        callback=self._on_check_updates,
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
                                label="Save Assignments",
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
    def _dialog_pos(self, w: int, h: int) -> list:
        """Centre a w x h window on the live viewport (falls back to the design
        viewport size if it can't be queried yet)."""
        try:
            vp_w = dpg.get_viewport_client_width()
            vp_h = dpg.get_viewport_client_height()
        except Exception:
            vp_w, vp_h = _VP_W, _VP_H
        return [max(20, (vp_w - w) // 2), max(20, (vp_h - h) // 2)]

    @contextmanager
    def _dialog(self, title: str, tag: str, size, *,
                primary=None, secondary=None, buttons=None, show: bool = True,
                autosize: bool = True, modal: bool = True):
        """Standard modal dialog: locked chrome, **auto-fit height**, fixed width,
        centred, with a right-aligned themed footer built after the body.

        ``size`` is ``(width, height_hint)``. Width is enforced (``min_size`` +
        ``autosize``) so wrap widths and footer alignment stay predictable; height
        auto-fits the content, so a dialog can never show a scrollbar or clip its
        text -- size it once and it always fits. ``height_hint`` is used only to
        centre the window vertically. Wrap long/dynamic text at ``width - 2 *
        _DIALOG_PAD`` so it never forces the window wider.

        Footer buttons come from either ``primary``/``secondary`` ``(label,
        callback)`` tuples (primary blue, secondary grey) or a general ``buttons``
        list for 3+ buttons -- each item ``(label, callback)`` (grey) or
        ``(label, callback, "primary")`` (blue). Body widgets go in the ``with``
        block. Callers own the lifecycle via the callbacks they pass (hide with
        ``configure_item(tag, show=False)`` for build-once dialogs, or
        ``delete_item(tag)`` for transient ones)."""
        w, h = size
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        win_kwargs = dict(label=title, tag=tag, modal=modal, no_close=True,
                          no_collapse=True, no_resize=True, no_scrollbar=True,
                          show=show, pos=self._dialog_pos(w, h))
        if autosize:
            # Auto-fit height, and pin the width to exactly w (min == max on the
            # x axis). The window can never scroll, clip vertically, or -- the
            # part that bites -- balloon sideways when a body item is wider than
            # expected, which would leave the right-aligned footer floating away
            # from the edge. Wrap long body text at w - 2 * _DIALOG_PAD so it
            # doesn't clip against the pinned width.
            win_kwargs.update(autosize=True, min_size=[w, 1],
                              max_size=[w, 100_000])
        else:
            # Fixed size (readers that scroll their own inner child_window).
            win_kwargs.update(width=w, height=h)
        with dpg.window(**win_kwargs):
            yield
            self._dialog_footer(w, primary=primary, secondary=secondary,
                                buttons=buttons)

    @staticmethod
    def _dialog_btn_w(label: str) -> int:
        """Fit a footer button to its label while holding short labels to a
        uniform minimum. Estimated (not measured), so it works even for dialogs
        built before the first frame is rendered."""
        return max(_DIALOG_BTN_W, 8 * len(label) + 26)

    def _dialog_footer(self, w: int, *, primary=None, secondary=None,
                       buttons=None) -> None:
        """Right-aligned footer row: separator, then the buttons in order, the
        primary blue (nudge) and the rest grey (anti-nudge)."""
        rows = []
        if buttons is not None:
            for b in buttons:
                rows.append((b[0], b[1], len(b) > 2 and b[2] == "primary"))
        else:
            if primary:
                rows.append((primary[0], primary[1], True))
            if secondary:
                rows.append((secondary[0], secondary[1], False))
        if not rows:
            return
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=6)
        widths = [self._dialog_btn_w(r[0]) for r in rows]
        block = sum(widths) + (len(rows) - 1) * _DIALOG_GAP
        content_w = w - 2 * _DIALOG_PAD
        lead = max(0, content_w - _DIALOG_RM - block)
        with dpg.group(horizontal=True):
            if lead:
                dpg.add_spacer(width=lead)
            for (label, callback, is_primary), bw in zip(rows, widths):
                btag = dpg.add_button(label=label, width=bw, callback=callback)
                dpg.bind_item_theme(
                    btag,
                    self.theme.nudge_theme if is_primary
                    else self.theme.antinudge_theme,
                )

    def _build_population_popup(self):
        with self._dialog(
            "Population", "popup_population", (420, 340),
            show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_population", show=False)),
        ):
            self._tolerance = dpg.add_slider_float(
                label="Population Tolerance",
                default_value=2.5, min_value=0.1, max_value=10.0,
                format="%.1f %%", width=260,
            )
            self._tooltip(
                self._tolerance,
                "Mosaic will only explore solutions where each district differs "
                "from ideal by no more than this percentage in either direction.",
            )
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_text("Tolerance Ratchet:")
                self._tolerance_ratchet_mode = dpg.add_radio_button(
                    ["Off", "Standard", "Strict"],
                    default_value="Off", horizontal=True,
                    callback=self._on_popdev_score_toggle,
                )
            self._tooltip(
                self._tolerance_ratchet_mode,
                "Gradually tightens Population Tolerance toward 0.25% over the "
                "back of the run - never below the deviation the map already "
                "hit, so it can't strand a plan.\n"
                "  Off: fixed.\n"
                "  Standard: tightens on each new best.\n"
                "  Strict: tightens every eligible step.",
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("Population Deviation Score - Safe Harbor", "heading")
            self._pop_dev_harbor = dpg.add_slider_float(
                label="Safe Harbor",
                default_value=0.0, min_value=0.0, max_value=5.0,
                format="%.2f %%", width=260,
            )
            self._tooltip(
                self._pop_dev_harbor,
                "Districts within this % of ideal are not penalized by the "
                "population deviation score. Cannot exceed Population Tolerance "
                "(clamped on run).",
            )

    def _build_seed_popup(self):
        with self._dialog(
            "Seed", "popup_seed", (340, 160), show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_seed", show=False)),
        ):
            self._seed = dpg.add_input_int(
                label="Random Seed  (0 = random)",
                default_value=0, min_value=0, width=120,
            )
            self._tooltip(
                self._seed,
                "Set a non-zero seed to make a run reproducible; 0 leaves the "
                "RNG random. Best-effort: identical results need the same "
                "machine, Mosaic version, and shapefile - float ordering in "
                "numpy/igraph can differ across environments.",
            )

    def _build_advanced_save_popup(self):
        # Fixed size (not autosize): the inline spinner/status toggle on save and
        # we don't want the window resizing mid-export.
        with self._dialog(
            "Export District Map", "popup_adv_save", (420, 265),
            show=False, autosize=False,
        ):
            self._adv_save_title = dpg.add_input_text(
                label="Title (optional)", default_value="", width=260,
                hint="Leave blank for no title",
            )
            dpg.add_spacer(height=8)
            dpg.add_text("Format")
            self._adv_save_fmt = dpg.add_radio_button(
                ["PNG (raster)", "PDF (vector, slower)"],
                default_value="PNG (raster)", horizontal=True,
                callback=self._on_adv_fmt_changed,
            )
            with dpg.group() as self._adv_dpi_group:
                dpg.add_spacer(height=4)
                self._adv_save_dpi = dpg.add_combo(
                    label="Raster DPI",
                    items=["96 (screen)", "144 (1.5x)", "192 (2x)",
                           "288 (3x)", "384 (4x)", "576 (6x)"],
                    default_value="288 (3x)", width=200,
                )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                self._adv_save_btn = dpg.add_button(
                    label="Save", width=80,
                    callback=self._on_advanced_save_confirm,
                )
                dpg.bind_item_theme(self._adv_save_btn, self.theme.nudge_theme)
                self._adv_save_as_btn = dpg.add_button(
                    label="Save As...", width=90,
                    callback=self._on_advanced_save_as,
                )
                dpg.bind_item_theme(self._adv_save_as_btn,
                                    self.theme.nudge_theme)
                self._adv_close_btn = dpg.add_button(
                    label="Close", width=80,
                    callback=lambda: dpg.configure_item(
                        "popup_adv_save", show=False),
                )
                dpg.bind_item_theme(self._adv_close_btn,
                                    self.theme.antinudge_theme)
                dpg.add_spacer(width=4)
                self._adv_save_spinner = dpg.add_loading_indicator(
                    style=0, radius=2.0, show=False,
                    color=self.theme.color("body"),
                    secondary_color=self.theme.color("muted"),
                )
            self._adv_save_status = dpg.add_text("", show=False)

    def _build_opt_popup(self):
        with self._dialog(
            "Annealing Settings", "popup_opt", (460, 460),
            show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_opt", show=False)),
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
                    self._tooltip(
                        self._cooling_rate,
                        "Per-iteration temperature multiplier (default "
                        "0.9995). Lower values cool faster.",
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
                default_value=25, min_value=0, max_value=50,
                format="%d %%", width=200,
            )
            self._tooltip(
                self._n3_pct,
                "Fraction of steps that merge 3 districts (vs 2) and re-split. "
                "Escapes local minima better, at ~2.4x per-step cost. "
                "Set to 0 for mass-generation runs.",
            )

            dpg.add_spacer(height=6)
            self.theme.text("Polish Flips", "heading")
            self._flip_enabled = dpg.add_checkbox(
                label="Enable single-precinct flips in tail",
                default_value=True,
            )
            self._tooltip(
                self._flip_enabled,
                "Adds single-precinct boundary flips alongside ReCom to polish "
                "borders late in the run. The flip rate ramps from 5% early to "
                "85% at the end, always leaving at least 15% ReCom.",
            )
            self._flip_midpoint = dpg.add_slider_int(
                label="50% crossover (% of run)",
                default_value=84, min_value=1, max_value=99,
                format="%d %%", width=200,
            )
            self._tooltip(
                self._flip_midpoint,
                "Where in the run the flip rate hits 50%. "
                "Higher = flips stay rare until later.",
            )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("County Congruence", "heading")
            self._hsplit_unclipped = dpg.add_checkbox(
                label="Unclipped County Congruence", default_value=True,
            )
            self._tooltip(
                self._hsplit_unclipped,
                "Let the County Congruence penalty climb past the scorecard cap so "
                "the optimizer keeps a gradient on heavily-split plans. Off = clipped "
                "scorecard form (pins at the cap).",
            )

    def _build_alignment_settings_popup(self):
        # Non-modal exception: this window spawns a file dialog ("Load reference
        # plan..."), and a modal-over-modal stack misbehaves in DPG.
        with self._dialog(
            "Alignment Settings", "popup_alignment_settings", (400, 340),
            show=False, modal=False,
            secondary=("Close", lambda: dpg.configure_item(
                "popup_alignment_settings", show=False)),
        ):
            self.theme.text(
                "Load a reference plan, then choose whose voters to align "
                "and which districts to score.", "muted", wrap=380,
            )
            dpg.add_separator()
            dpg.add_spacer(height=6)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Load reference plan...",
                    callback=self._on_load_alignment,
                )
                dpg.add_button(
                    label="Clear",
                    callback=self._on_clear_alignment,
                )
            dpg.add_spacer(height=8)

            # Ask 1 — whose share retention is measured in (needs election data).
            dpg.add_text("Focus:")
            self._alignment_focus = dpg.add_radio_button(
                items=["All residents", "Republican", "Democratic"],
                default_value="All residents", horizontal=True,
                callback=self._on_alignment_focus,
            )
            self._hint(self._alignment_focus, "alignment_focus")
            dpg.add_spacer(height=6)

            # Ask 2 — restrict scoring to the focus party's won districts.
            # Meaningless without a party focus, so disabled when neutral.
            self._alignment_restrict = dpg.add_checkbox(
                label="Only districts that party wins",
                default_value=False, enabled=False,
            )
            self._hint(self._alignment_restrict, "alignment_restrict")
            dpg.add_spacer(height=6)

            self._alignment_win_threshold = dpg.add_slider_float(
                label="District win threshold",
                default_value=0.535, min_value=0.50, max_value=0.70,
                format="%.3f", width=200,
            )
            self._tooltip(
                self._alignment_win_threshold,
                "A reference district counts as 'won' (and is scored) when the "
                "focus party's two-party share exceeds this. 0.535 ~ win by 7pts; "
                "raise it to ignore near-coin-flip seats.",
            )

    def _build_partisan_popup(self):
        with self._dialog(
            "Partisanship Settings", "popup_partisan", (460, 320),
            show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_partisan", show=False)),
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
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            self.theme.text("Mean-Median / Efficiency Gap shape", "heading")
            self._partisan_quadratic_penalty = dpg.add_checkbox(
                label="Use quadratic penalty",
                default_value=False,
            )
            self._tooltip(
                self._partisan_quadratic_penalty,
                "When ON, all three modes (Fair / D / R) use a quadratic curve: "
                "less urgency near the favored end, more urgency near the unfavored "
                "end. Off = linear.",
            )
            dpg.add_spacer(height=8)

            self._mm_bound = dpg.add_slider_float(
                label="MM bound",
                default_value=0.20, min_value=0.05, max_value=0.30,
                format="%.2f", width=220,
            )
            self._tooltip(
                self._mm_bound,
                "MM penalty reaches 100 at this |raw| value, then saturates.",
            )
            self._eg_bound = dpg.add_slider_float(
                label="EG bound",
                default_value=0.35, min_value=0.10, max_value=0.50,
                format="%.2f", width=220,
            )
            self._tooltip(
                self._eg_bound,
                "EG penalty reaches 100 at this |raw| value, then saturates.",
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
            with dpg.plot(height=-1, width=-1, tag="panel_temp_plot", no_menus=True):
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
            label="Classic Splitting", tag="panel_county_splits",
            show=False, width=540, height=460,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_cs_item, False),
        ):
            with dpg.group(tag="cs_charts_grp"):
                # Two panels, one per real county score -- there is no single
                # combined "county splits score", so it isn't charted.
                self.theme.text("Excess Splits", "heading")
                with dpg.plot(height=165, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_excess_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="cs_excess_y"):
                        dpg.add_line_series([], [], label="Excess", tag="cs_excess_series")
                dpg.add_spacer(height=6)
                self.theme.text("Single-County Districts", "heading")
                with dpg.plot(height=165, width=-1, no_menus=True):
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

    def _build_phase_themes(self):
        """Recency-ramp line themes per palette, in faded and full-alpha variants
        (the Fade toggle), plus the palette's head-marker theme. Built for both
        palettes so a light/dark swap only rebinds -- see _phase_apply_fade."""
        self._phase_band_themes = {}    # palette -> {"fade": [...], "solid": [...]}
        self._phase_head_themes = {}    # palette -> marker theme
        for pal, ramp in (("dark", _PHASE_RAMP), ("light", _PHASE_RAMP_LIGHT)):
            fade_list, solid_list = [], []
            for k in range(_PHASE_BANDS):
                t = (k + 0.5) / _PHASE_BANDS
                rgb = _ramp_color(t, ramp)
                weight = 1.0 + 2.0 * t
                for alpha, store in ((int(75 + 180 * t), fade_list),
                                     (255, solid_list)):
                    with dpg.theme() as th:
                        with dpg.theme_component(dpg.mvLineSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Line, (*rgb, alpha),
                                                category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight,
                                                category=dpg.mvThemeCat_Plots)
                    store.append(th)
            self._phase_band_themes[pal] = {"fade": fade_list, "solid": solid_list}
            dot_fill = (0, 0, 0, 255) if pal == "dark" else (255, 255, 255, 255)
            dot_line = (255, 255, 255, 255) if pal == "dark" else (0, 0, 0, 255)
            with dpg.theme() as hd:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, dot_fill,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, dot_line,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 4.0,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerWeight, 2.0,
                                        category=dpg.mvThemeCat_Plots)
            self._phase_head_themes[pal] = hd

    def _build_phase_panel(self):
        """Comet Plot: a metric-vs-metric trajectory -- two axis pickers, a tail
        that cools with age, and a bright dot on the live plan."""
        self._build_phase_themes()
        with dpg.window(
            label="Comet Plot", tag="panel_phase", show=False,
            width=420, height=440, pos=[_LEFT_W + 100, 80],
            on_close=lambda: dpg.set_value(self._panel_phase_item, False),
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("X")
                dpg.add_combo(_PHASE_LABELS, default_value="Compactness",
                              width=200, tag="phase_x_combo",
                              callback=self._on_phase_x_change)
            with dpg.group(horizontal=True):
                dpg.add_text("Y")
                dpg.add_combo(_PHASE_LABELS, default_value="Efficiency Gap",
                              width=200, tag="phase_y_combo",
                              callback=self._on_phase_y_change)
            with dpg.group(horizontal=True):            # options row (all on by default)
                dpg.add_checkbox(label="Smooth", default_value=True,
                                 callback=self._on_phase_smooth_change)
                dpg.add_checkbox(label="Fit all", default_value=True,
                                 callback=self._on_phase_fit_change)
                dpg.add_checkbox(label="Fade", default_value=True,
                                 callback=self._on_phase_fade_change)
            dpg.add_spacer(height=4)
            with dpg.plot(height=-1, width=-1, tag="phase_plot", no_menus=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="Compactness", tag="phase_x")
                with dpg.plot_axis(dpg.mvYAxis, label="Efficiency Gap",
                                   tag="phase_y"):
                    for k in range(_PHASE_BANDS):
                        dpg.add_line_series([], [], label=f"##phase_band_{k}",
                                            tag=f"phase_band_{k}")
                    dpg.add_scatter_series([], [], label="##phase_head",
                                           tag="phase_head")
            # Stamp the default axis titles with their unit hints (later picks
            # route through _set_phase_axis, which does the same).
            self._set_phase_axis("x", self._phase_x_label)
            self._set_phase_axis("y", self._phase_y_label)
            self._phase_apply_fade()    # bind bands + head to current palette / fade

    def _set_phase_axis(self, which: str, label: str) -> None:
        """Point an axis at a metric (updates combo, stored label, axis title).
        The axis title carries a unit hint (e.g. "(%)") while the picker and the
        stored label stay bare -- see _PHASE_AXIS_UNIT."""
        self._phase_lim = None      # new metric/scale -> refit bounds
        unit = _PHASE_AXIS_UNIT.get(label)
        title = f"{label} {unit}" if unit else label
        if which == "x":
            self._phase_x_label = label
            dpg.set_value("phase_x_combo", label)
            dpg.configure_item("phase_x", label=title)
        else:
            self._phase_y_label = label
            dpg.set_value("phase_y_combo", label)
            dpg.configure_item("phase_y", label=title)

    def _on_phase_x_change(self, sender, app_data):
        self._set_phase_axis("x", app_data)

    def _on_phase_y_change(self, sender, app_data):
        self._set_phase_axis("y", app_data)

    def _on_phase_fit_change(self, sender, app_data):
        self._phase_fit_all = bool(app_data)
        self._phase_lim = None

    def _on_phase_fade_change(self, sender, app_data):
        self._phase_fade = bool(app_data)
        self._phase_apply_fade()

    def _phase_apply_fade(self) -> None:
        """Bind every trail band + head marker to the current palette's faded or
        full-alpha theme set. Called on Fade toggle and on light/dark swap."""
        pal = self.theme.palette.name
        if pal not in self._phase_band_themes:
            pal = "dark"
        bands = self._phase_band_themes[pal]["fade" if self._phase_fade else "solid"]
        for k in range(_PHASE_BANDS):
            dpg.bind_item_theme(f"phase_band_{k}", bands[k])
        dpg.bind_item_theme("phase_head", self._phase_head_themes[pal])

    def _on_phase_smooth_change(self, sender, app_data):
        self._phase_smooth = bool(app_data)

    def _on_panel_phase_toggle(self):
        dpg.configure_item("panel_phase",
                           show=dpg.get_value(self._panel_phase_item))

    def _phase_active_labels(self) -> list:
        """Metrics whose history carries real values right now.

        Non-partisan display metrics are 0.0 unless their weight (or a coupled
        weight / the pop-dev ratchet) is on, so they gate on the run's config --
        this keeps the picker from offering a metric that flatlines at 0. The
        coupling below mirrors scoring/score.py; keep the two in step when scores
        change. Partisan metrics are always computed once election data is
        loaded, so they gate on elections alone. Hinge is the exception: its
        value is P(party wins >= threshold), meaningless until the user
        establishes a hinge, so it also needs the Hinge score enabled. Cut Edges
        / Score / Temperature are intrinsic and always offered.
        """
        cfg = self.state.score_config
        has = getattr(self, "_has_elections", False)
        ratchet = self.state.tolerance_ratchet_mode not in (None, "off", "Off")
        active = set()
        if cfg.weight_holistic_compactness:
            active.add("holistic_compactness_history")
        if cfg.weight_polsby_popper or cfg.weight_holistic_compactness:
            active.add("pp_history")
        if cfg.weight_reock or cfg.weight_holistic_compactness:
            active.add("reock_history")
        if cfg.weight_holistic_splitting:
            active.add("holistic_splitting_history")
        if cfg.weight_county_excess or cfg.weight_county_unified:
            active.add("county_excess_splits_history")
        if cfg.weight_pop_deviation or ratchet:
            active.update(("pop_dev_max_history", "pop_dev_mean_history"))
        if has:
            # Always computed from the election data -> available with elections.
            active.update((
                "mm_history", "eg_history", "dem_seats_history",
                "holistic_proportionality_history",
                "holistic_competitiveness_history",
                "majority_dem_history", "majority_rep_history",
            ))
            # Hinge needs a user-established threshold (the Hinge score toggle),
            # not just election data -- else its value is against a stale default.
            if dpg.get_value(self._hinge_enabled):
                active.add("hinge_history")
        always = {"cut_edges_history", "score_history", "temperature_history"}
        return [l for l in _PHASE_LABELS
                if _PHASE_ATTR[l] in always or _PHASE_ATTR[l] in active]

    def _phase_sync_available_metrics(self) -> None:
        """Rebuild the axis pickers to the metrics active for this run's config;
        bump a now-hidden axis to a safe intrinsic. Re-syncs only on a change
        (config is set at run start, so the picker tracks the live histories)."""
        labels = self._phase_active_labels()
        sig = tuple(labels)
        if self._phase_metric_sig == sig:
            return
        self._phase_metric_sig = sig
        dpg.configure_item("phase_x_combo", items=labels)
        dpg.configure_item("phase_y_combo", items=labels)
        if self._phase_x_label not in labels:
            self._set_phase_axis("x", "Cut Edges")
        if self._phase_y_label not in labels:
            self._set_phase_axis("y", "Score (total)")

    def _update_phase_plot(self) -> None:
        """Redraw the phase trajectory. Free when hidden; bounded to _PHASE_HEAD
        full-res recent points plus a strided tail (<= _PHASE_BUDGET)."""
        if not dpg.is_item_shown("panel_phase"):
            return
        self._phase_sync_available_metrics()
        xa, ya = _PHASE_ATTR[self._phase_x_label], _PHASE_ATTR[self._phase_y_label]
        with self.state._lock:
            xh = getattr(self.state, xa)
            yh = getattr(self.state, ya)
            n = min(len(xh), len(yh))
            if n == 0:
                xs_l = ys_l = []
                head_len = tail_n = stride = 0
            else:
                head_len = min(n, _PHASE_HEAD)
                tail_n = n - head_len
                budget_tail = max(1, _PHASE_BUDGET - head_len)
                # Power-of-two stride: it only ever doubles, so the sampled tail
                # indices stay nested -- points thin out, never reshuffle (no shimmer).
                stride = 1
                while tail_n // stride > budget_tail:
                    stride *= 2
                xs_l = (xh[0:tail_n:stride] if tail_n else []) + xh[tail_n:n]
                ys_l = (yh[0:tail_n:stride] if tail_n else []) + yh[tail_n:n]
        if n < self._phase_prev_n:          # run reset -> forget sticky bounds
            self._phase_lim = None
        self._phase_prev_n = n
        if not xs_l:
            for k in range(_PHASE_BANDS):
                dpg.set_value(f"phase_band_{k}", [[], []])
            dpg.set_value("phase_head", [[], []])
            return
        xs = _phase_transform(_PHASE_KIND[self._phase_x_label],
                              np.asarray(xs_l, dtype=np.float64))
        ys = _phase_transform(_PHASE_KIND[self._phase_y_label],
                              np.asarray(ys_l, dtype=np.float64))
        hx, hy = float(xs[-1]), float(ys[-1])           # dot = true current plan
        # Age (iterations back from the head) per plotted point, oldest first, so
        # recency t = exp(-age/TAU) is ascending -> bands are contiguous by t.
        tail_ages = ((n - 1) - np.arange(0, tail_n, stride)) if tail_n else np.empty(0)
        head_ages = np.arange(head_len - 1, -1, -1)
        ages = np.concatenate([tail_ages, head_ages]).astype(np.float64)
        t = np.exp(-ages / _PHASE_TAU)
        if self._phase_smooth and len(xs) >= _PHASE_SMOOTH_WIN:
            # Edge-aware moving average: normalise by the real (non-padded) window
            # overlap, else convolve's zero-padding drags both ends to the origin.
            k = np.ones(_PHASE_SMOOTH_WIN)
            norm = np.convolve(np.ones(len(xs)), k, mode="same")
            xs = np.convolve(xs, k, mode="same") / norm
            ys = np.convolve(ys, k, mode="same") / norm
        edges = np.searchsorted(t, [k / _PHASE_BANDS for k in range(1, _PHASE_BANDS)])
        edges = [0, *edges.tolist(), len(t)]
        for k in range(_PHASE_BANDS):
            a, b = edges[k], edges[k + 1]
            if k < _PHASE_BANDS - 1:
                b = min(b + 1, len(t))          # overlap one point so bands join
            dpg.set_value(f"phase_band_{k}",
                          [xs[a:b].tolist(), ys[a:b].tolist()] if b - a >= 1
                          else [[], []])
        dpg.set_value("phase_head", [[hx], [hy]])
        self._phase_fit_axes(xs, ys, ages, hx, hy)

    def _phase_fit_axes(self, xs, ys, ages, hx: float, hy: float) -> None:
        """Axis limits with margin. 'Fit all' holds sticky total bounds that only
        ever grow, so the view never jitters or shrinks mid-run; 'Follow dot'
        fits the last _PHASE_FOLLOW iterations. The live dot always stays inside."""
        def _pad(lo, hi):
            lo, hi = min(lo, hi), max(lo, hi)
            if hi - lo < _PHASE_MIN_SPAN:      # floor tiny ranges (Follow dot zoom)
                c = 0.5 * (lo + hi)
                lo, hi = c - _PHASE_MIN_SPAN / 2, c + _PHASE_MIN_SPAN / 2
            pad = (hi - lo) * 0.08
            return lo - pad, hi + pad

        if not self._phase_fit_all:
            m = ages <= _PHASE_FOLLOW
            xv, yv = xs[m], ys[m]
            xlo, xhi = _pad(min(float(xv.min()), hx), max(float(xv.max()), hx))
            ylo, yhi = _pad(min(float(yv.min()), hy), max(float(yv.max()), hy))
        else:
            dxlo, dxhi = min(float(xs.min()), hx), max(float(xs.max()), hx)
            dylo, dyhi = min(float(ys.min()), hy), max(float(ys.max()), hy)
            if self._phase_lim is None:
                lim = (dxlo, dxhi, dylo, dyhi)
            else:
                pxlo, pxhi, pylo, pyhi = self._phase_lim
                lim = (min(pxlo, dxlo), max(pxhi, dxhi),      # sticky: grow only,
                       min(pylo, dylo), max(pyhi, dyhi))      # never shrink
            self._phase_lim = lim
            xlo, xhi = _pad(lim[0], lim[1])
            ylo, yhi = _pad(lim[2], lim[3])
        dpg.set_axis_limits("phase_x", xlo, xhi)
        dpg.set_axis_limits("phase_y", ylo, yhi)

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
        from mosaic.gui.map_view import _PARTISAN_RGBA
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
            with dpg.plot(height=-1, width=-1, tag="partisan_plot", no_menus=True):
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
            with dpg.plot(height=-1, width=-1, tag="win_chance_plot", no_menus=True):
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
                with dpg.plot(height=-1, width=-1, no_menus=True):
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
                with dpg.plot(height=-1, width=-1, no_menus=True):
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
                with dpg.plot(height=-1, width=-1, no_menus=True):
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

    def _build_pp_panel(self):
        with dpg.window(
            label="Polsby-Popper", tag="panel_pp",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_pp_item, False),
        ):
            with dpg.group(tag="pp_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
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

    def _build_reock_panel(self):
        with dpg.window(
            label="Reock", tag="panel_reock",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_reock_item, False),
        ):
            with dpg.group(tag="reock_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="reock_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Reock (100 = circle)", tag="reock_y"):
                        dpg.add_line_series([], [], label="Reock", tag="reock_series")
            self._tooltip(
                "reock_plot_grp",
                "Optimizer uses (1 - Reock) as penalty; higher is more compact.",
            )
            self.theme.track(
                dpg.add_text(
                    "Apply a score to use this panel.",
                    tag="reock_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("reock_y", 0.0, 100.0)

    def _build_hc_panel(self):
        with dpg.window(
            label="Compactness", tag="panel_hc",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hc_item, False),
        ):
            with dpg.group(tag="hc_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hc_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Compactness (100 = best)", tag="hc_y"):
                        dpg.add_line_series([], [], label="Compactness", tag="hc_series")
            self._tooltip(
                "hc_plot_grp",
                "Combined PP + Reock rating; higher is more compact.",
            )
            self.theme.track(
                dpg.add_text(
                    "Apply a score to use this panel.",
                    tag="hc_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hc_y", 0.0, 100.0)

    def _build_hsplit_panel(self):
        with dpg.window(
            label="County Congruence", tag="panel_hsplit",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hsplit_item, False),
        ):
            with dpg.group(tag="hsplit_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hsplit_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="County Congruence (penalty, 0 = best)", tag="hsplit_y"):
                        dpg.add_line_series([], [], label="Cty Cong", tag="hsplit_series")
            self._tooltip(
                "hsplit_plot_grp",
                "Combined county- and district-direction split penalty; lower = less "
                "split. Unclipped mode keeps climbing past 100 on heavily-split plans.",
            )
            self.theme.track(
                dpg.add_text(
                    "Apply a score to use this panel.",
                    tag="hsplit_inactive_lbl", show=False,
                ),
                "muted",
            )
        # Penalty is >= 0 and unbounded above: pin the floor at 0, auto-fit the top
        # each frame in the render block.
        dpg.set_axis_limits("hsplit_y", 0.0, 100.0)

    def _build_hprop_panel(self):
        with dpg.window(
            label="Proportionality", tag="panel_hprop",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hprop_item, False),
        ):
            with dpg.group(tag="hprop_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hprop_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Proportionality (100 = best)", tag="hprop_y"):
                        dpg.add_line_series([], [], label="Proportionality", tag="hprop_series")
            self._tooltip(
                "hprop_plot_grp",
                "Bias-from-proportional rating; higher = closer to proportional.",
            )
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="hprop_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hprop_y", 0.0, 100.0)

    def _build_hcmp_panel(self):
        with dpg.window(
            label="Competitiveness", tag="panel_hcmp",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hcmp_item, False),
        ):
            with dpg.group(tag="hcmp_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hcmp_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Competitiveness (100 = best)", tag="hcmp_y"):
                        dpg.add_line_series([], [], label="Competitiveness", tag="hcmp_series")
            self._tooltip(
                "hcmp_plot_grp",
                "Competitiveness rating; higher = more districts near a toss-up.",
            )
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="hcmp_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hcmp_y", 0.0, 100.0)

    def _build_popdev_panel(self):
        with dpg.window(
            label="Population Deviation", tag="panel_popdev",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 100, 100],
            on_close=lambda: dpg.set_value(self._panel_popdev_item, False),
        ):
            with dpg.group(tag="popdev_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="popdev_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="% Deviation", tag="popdev_y"):
                        dpg.add_line_series([], [], label="Max %",
                                            tag="popdev_max_series")
                        dpg.add_line_series([], [], label="Mean %",
                                            tag="popdev_mean_series")
                # Set y-axis floor to 0; constrain so pan/zoom can't go below.
                dpg.set_axis_limits("popdev_y", 0.0, 5.0)
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

    def _build_alignment_panel(self):
        with dpg.window(
            label="Alignment", tag="panel_alignment",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 100, 100],
            on_close=lambda: dpg.set_value(self._panel_alignment_item, False),
        ):
            with dpg.group(tag="alignment_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration",
                                      tag="alignment_x")
                    with dpg.plot_axis(dpg.mvYAxis,
                                       label="Cohesion % (100 = intact)",
                                       tag="alignment_y"):
                        dpg.add_line_series([], [], label="Mean",
                                            tag="alignment_mean_series")
                        dpg.add_line_series([], [], label="Worst district",
                                            tag="alignment_min_series")
            self._tooltip(
                "alignment_plot_grp",
                "Share of each reference district that stays intact. "
                "Mean across districts and the single worst-hit district. "
                "Higher is closer to the reference plan.",
            )
            self.theme.track(
                dpg.add_text(
                    "Load a reference plan and apply the score to use this panel.",
                    tag="alignment_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("alignment_y", 0.0, 100.0)

    def _build_cut_edges_panel(self):
        with dpg.window(
            label="Cut Edges", tag="panel_cut_edges",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_cuts_item, False),
        ):
            with dpg.plot(height=-1, width=-1, no_menus=True):
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
                with dpg.plot(height=-1, width=-1, no_menus=True):
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
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hinge_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Chance of Hinge (%)", tag="hinge_y"):
                        dpg.add_line_series([], [], label="Hinge", tag="hinge_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="hinge_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hinge_y", 0.0, 100.0)

    # Column spec for the District Info table.  (header, pixel_width, key)
    _DIST_COLS = [
        ("Dist",        50,  "dist"),
        ("Population",  95,  "pop"),
        ("Pop Dev",     80,  "pdev"),
        ("Polsby-Popper", 110, "pp"),
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
            with dpg.plot(height=-1, width=-1, no_mouse_pos=True, no_menus=True):
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

    @staticmethod
    def _version_tuple(v):
        """Parse 'X.Y.Z' to a comparable tuple; () if unparseable."""
        try:
            return tuple(int(x) for x in v.split("."))
        except (ValueError, AttributeError):
            return ()

    def _on_open_output_dir(self):
        """Open output/ (saved maps + assignment/metric CSVs) in the OS file browser."""
        import os, sys, subprocess
        out = Path("output").resolve()
        out.mkdir(exist_ok=True)  # create on first use so opening never fails
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
        import os, sys, subprocess
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
        """Manual update check (Advanced menu). Compares the local version to the
        public repo's pyproject version. Synchronous — it's user-initiated — with
        a short timeout so it can't hang the UI for long."""
        import urllib.request
        import re as _re
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

    def _build_help_popup(self):
        # Fixed-size reader: the doc text scrolls inside its own child_window,
        # so this one keeps a fixed height instead of auto-fitting.
        with self._dialog(
            "Help", "popup_help", (460, 400), show=False, autosize=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_help", show=False)),
        ):
            with dpg.child_window(height=-48, border=False):
                self.theme.text(
                    "Mosaic from Matt Mohn (@mattmxhn)",
                    "title",
                )
                self.theme.text(
                    f"Version {__version__}",
                    "muted",
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
                    "3. Set district count and iterations\n"
                    "4. Enable and weight the scores in the left-hand panel\n"
                    "5. Optionally load an existing plan as a hot start via "
                    "Advanced > Load Hot Start\n"
                    "6. Start - pause / reset / revert to best as needed\n"
                    "7. Save Assignments writes a CSV",
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

        # Recent menu is built directly from _recent_shapefiles during
        # window construction above; _refresh_recent_menu handles updates.

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        dpg.show_viewport()
        dpg.set_viewport_resize_callback(self._align_photo_icons)
        _aligned = False
        while dpg.is_dearpygui_running():
            self._update_ui()
            dpg.render_dearpygui_frame()
            if not _aligned and dpg.get_frame_count() >= 2:
                self._align_photo_icons()
                _aligned = True
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

        self._buf_score.add(_sd)
        self._buf_acc.add_pairs(_ad, scale=100.0)
        self._buf_temp.add(_td)
        self._buf_cs_score.add(_csd)
        self._buf_cs_excess.add(_ced)
        self._buf_cs_clean.add(_cld)
        self._buf_mm.add(_md)
        self._buf_eg.add(_ed)
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
            elif dpg.is_key_pressed(dpg.mvKey_S) and can_save:
                self._on_file_save_assignments()
            elif dpg.is_key_pressed(dpg.mvKey_W):
                self._on_close()

    def _clear_all_series(self) -> None:
        """Clear local history buffers and blank all DPG plot series."""
        for buf in (
            self._buf_score, self._buf_acc, self._buf_temp,
            self._buf_cs_score, self._buf_cs_excess, self._buf_cs_clean,
            self._buf_mm, self._buf_eg, self._buf_seats,
            self._buf_pp, self._buf_reock, self._buf_hc,
            self._buf_hsplit, self._buf_hprop, self._buf_hcmp,
            self._buf_popdev,
            self._buf_popdev_max, self._buf_popdev_mean, self._buf_cuts,
            self._buf_align_mean, self._buf_align_min,
            self._buf_maj_dem, self._buf_maj_rep, self._buf_hinge,
        ):
            buf.clear()
        empty = [[], []]
        for tag in (
            "score_series", "acc_series", "panel_temp_series",
            "cs_excess_series", "cs_clean_series",
            "mm_series", "eg_series", "seats_series",
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

    def _on_pp_toggle(self):
        en = dpg.get_value(self._pp_enabled)
        self.theme.retoken(self._pp_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("pp_controls", show=en)

    def _on_reock_toggle(self):
        en = dpg.get_value(self._reock_enabled)
        self.theme.retoken(self._reock_lbl,
                           "accent_green" if en else "secondary")
        dpg.configure_item("reock_controls", show=en)

    def _on_alignment_toggle(self):
        en = dpg.get_value(self._alignment_enabled)
        self.theme.retoken(self._alignment_lbl,
                           "accent_green" if en else "secondary")
        dpg.configure_item("alignment_controls", show=en)

    def _on_alignment_focus(self):
        # "Only districts that party wins" is meaningless without a party;
        # gray it out and force it off when focus is neutral.
        neutral = dpg.get_value(self._alignment_focus) == "All residents"
        if neutral:
            dpg.set_value(self._alignment_restrict, False)
        dpg.configure_item(self._alignment_restrict, enabled=not neutral)

    def _on_hc_toggle(self):
        en = dpg.get_value(self._hc_enabled)
        self.theme.retoken(self._hc_lbl,
                           "accent_green" if en else "secondary")
        dpg.configure_item("hc_controls", show=en)

    def _on_hsplit_toggle(self):
        en = dpg.get_value(self._hsplit_enabled)
        self.theme.retoken(self._hsplit_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("hsplit_controls", show=en)
        # County-Edge Bias is paired with County Congruence: enabling the score
        # turns the bias on too (user can still switch it off separately).
        if en:
            dpg.set_value(self._county_bias_enabled, True)
            dpg.configure_item("county_bias_controls", show=True)

    def _on_hprop_toggle(self):
        en = dpg.get_value(self._hprop_enabled)
        self.theme.retoken(self._hprop_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("hprop_controls", show=en)

    def _on_hcmp_toggle(self):
        en = dpg.get_value(self._hcmp_enabled)
        self.theme.retoken(self._hcmp_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("hcmp_controls", show=en)

    def _on_popdev_score_toggle(self):
        en = dpg.get_value(self._popdev_enabled)
        # Green when weighted OR when the Tolerance Ratchet is on -- the ratchet
        # drives population deviation even if this score's own weight is 0.
        ratchet_on = dpg.get_value(self._tolerance_ratchet_mode) != "Off"
        self.theme.retoken(self._popdev_lbl,
                           "accent_green" if (en or ratchet_on) else "secondary")
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

    def _on_panel_pp_toggle(self):
        dpg.configure_item("panel_pp", show=dpg.get_value(self._panel_pp_item))

    def _on_panel_reock_toggle(self):
        dpg.configure_item("panel_reock", show=dpg.get_value(self._panel_reock_item))

    def _on_panel_hc_toggle(self):
        dpg.configure_item("panel_hc", show=dpg.get_value(self._panel_hc_item))

    def _on_panel_hsplit_toggle(self):
        dpg.configure_item("panel_hsplit", show=dpg.get_value(self._panel_hsplit_item))

    def _on_panel_hprop_toggle(self):
        dpg.configure_item("panel_hprop", show=dpg.get_value(self._panel_hprop_item))

    def _on_panel_hcmp_toggle(self):
        dpg.configure_item("panel_hcmp", show=dpg.get_value(self._panel_hcmp_item))

    def _on_panel_popdev_toggle(self):
        dpg.configure_item("panel_popdev", show=dpg.get_value(self._panel_popdev_item))

    def _on_panel_alignment_toggle(self):
        dpg.configure_item("panel_alignment", show=dpg.get_value(self._panel_alignment_item))

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
        self._sync_camera_icon_to_theme()
        self._phase_apply_fade()        # repaint phase trail for the new palette

    def _align_photo_icons(self, *_):
        """Resize the overlay-row spacer so the photo icons hug the right edge.

        Called once after first render and again on every viewport resize.
        Uses the actual measured positions/sizes rather than guessing widths
        because checkbox widths depend on the font in use.
        """
        fill = getattr(self, "_overlay_fill", None)
        cam = getattr(self, "_cam_btn", None)
        more = getattr(self, "_more_btn", None)
        spinner = getattr(self, "_save_spinner", None)
        if not all(t is not None and dpg.does_item_exist(t)
                   for t in (fill, cam, more, spinner)):
            return
        if not dpg.does_item_exist("map_container"):
            return
        container_w = dpg.get_item_rect_size("map_container")[0]
        if container_w <= 1:
            return
        cam_x = dpg.get_item_pos(cam)[0]
        cam_w = dpg.get_item_rect_size(cam)[0] or 28
        more_w = dpg.get_item_rect_size(more)[0] or 28
        spinner_w = dpg.get_item_rect_size(spinner)[0] or 0
        photo_block = cam_w + more_w + spinner_w + 16
        right_pad = 12
        target_cam_x = container_w - photo_block - right_pad
        try:
            current_w = dpg.get_item_configuration(fill).get("width", 20)
        except SystemError:
            current_w = 20
        new_w = max(8, int(current_w + (target_cam_x - cam_x)))
        dpg.configure_item(fill, width=new_w)

    def _sync_camera_icon_to_theme(self):
        """Repaint the map-toolbar icon textures in the current palette's body color."""
        r, g, b, _ = self.theme.color("body")
        fg = (int(r), int(g), int(b))
        if dpg.does_item_exist("camera_icon_texture"):
            dpg.set_value("camera_icon_texture", _build_camera_icon(fg))
        if dpg.does_item_exist("more_icon_texture"):
            dpg.set_value("more_icon_texture", _build_more_icon(fg))
        body = self.theme.color("body")
        muted = self.theme.color("muted")
        for tag_attr in ("_save_spinner", "_adv_save_spinner"):
            tag = getattr(self, tag_attr, None)
            if tag is not None and dpg.does_item_exist(tag):
                dpg.configure_item(tag, color=body, secondary_color=muted)

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

    def _rerender_map(self) -> None:
        """Re-compose and upload the current map frame with the latest overlay flags.

        Overlay toggle callbacks call this instead of queuing map_needs_update so
        the response is immediate rather than deferred to the next render-loop tick
        (which can silently drop the update if current_assignment is transiently None).
        Falls back to queuing when no assignment is available yet.
        """
        if self.map_view is None:
            return
        with self.state._lock:
            asgn = (self.state.current_assignment.copy()
                    if self.state.current_assignment is not None else None)
            n    = self.state.num_districts
            init = (self.state.initial_assignment.copy()
                    if self.state.initial_assignment is not None else None)
        if asgn is not None:
            self.map_view.render_assignment(asgn, n, init)
        else:
            self.state.update(map_needs_update=True)

    def _on_county_overlay_toggle(self):
        if self.map_view is None:
            return
        self.map_view.county_overlay = dpg.get_value(self._county_overlay)
        self._rerender_map()

    def _on_precinct_overlay_toggle(self):
        if self.map_view is None:
            return
        self.map_view.precinct_overlay = dpg.get_value(self._precinct_overlay)
        self._rerender_map()

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
        if self.map_view is None:
            return
        self.map_view.partisan_overlay = dpg.get_value(self._partisan_overlay)
        self._rerender_map()

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
        if self.map_view is None:
            return
        self.map_view.district_partisan_overlay = dpg.get_value(self._district_partisan)
        self._rerender_map()

    def _on_splits_view_toggle(self):
        if self.map_view is None:
            return
        self.map_view.splits_view = dpg.get_value(self._splits_view)
        self._rerender_map()

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
        if self.map_view is None:
            return
        self.map_view.compactness_view = dpg.get_value(self._compactness_view)
        self._rerender_map()

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
        if self.map_view is None:
            return
        self.map_view.pop_dev_view = dpg.get_value(self._pop_dev_view)
        self._rerender_map()

    def _on_labels_toggle(self):
        if self.map_view is None:
            return
        self.map_view.show_labels = dpg.get_value(self._show_labels)
        self._rerender_map()

    # ── Action callbacks ──────────────────────────────────────────────────────

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
        from mosaic.paths import shapefiles_dir, mosaic_data_dir
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
        from mosaic.io.hot_start import read_csv_columns, HotStartError
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
        from mosaic.io.hot_start import load_hot_start, HotStartError
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
        from mosaic.io.hot_start import read_csv_columns, HotStartError
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
            precompute_alignment_data, AlignmentError,
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
            self.state.county_unified_districts_history = self.state.county_unified_districts_history[:n_score]
            self.state.mm_history               = self.state.mm_history[:n_score]
            self.state.eg_history               = self.state.eg_history[:n_score]
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
            weight_dem_seats=dpg.get_value(self._w_dem_seats) if seats_on else 0.0,
            dem_seats_favor_dem=(dpg.get_value(self._dem_seats_dir) == "D"),
            weight_holistic_proportionality=w_hprop,
            weight_holistic_competitiveness=w_hcmp,
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

    def _on_export(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("output") / f"assignments_{timestamp}.csv"
        self._do_export_to_path(output_path)

    def _do_export_to_path(self, output_path: Path) -> None:
        """Export labeled assignment CSV to a specific path."""
        with self.state._lock:
            current = (self.state.current_assignment.copy()
                       if self.state.current_assignment is not None else None)
        if self.runner is None or current is None:
            return
        from mosaic.io import save_assignments
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)

        precinct_ids = None
        id_col_name = "precinct_id"
        if self.runner.gdf is not None and self._loaded_config is not None:
            col = self._loaded_config.id_col
            if col and col in self.runner.gdf.columns:
                precinct_ids = self.runner.gdf[col].tolist()
                id_col_name = col

        save_assignments(
            self._export_labeled_assignment(current),
            output_path,
            precinct_ids=precinct_ids,
            id_col_name=id_col_name,
        )
        self.state.update(status_message=f"Assignments saved to {output_path}")

    def _on_export_metrics(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("output") / f"metrics_{timestamp}.csv"
        self._do_export_metrics_to_path(output_path)

    def _do_export_metrics_to_path(self, output_path: Path) -> None:
        """Export per-district metrics CSV to a specific path."""
        with self.state._lock:
            current = (self.state.current_assignment.copy()
                       if self.state.current_assignment is not None else None)
        if self.runner is None or current is None:
            return
        from mosaic.io import save_metrics
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)

        n_dist = self.state.num_districts
        ideal_pop = (float(self.runner.populations.sum()) / n_dist
                     if n_dist > 0 else 1.0)
        dem = self.runner.election_arrays[0][0] if self.runner.election_arrays else None
        gop = self.runner.election_arrays[0][1] if self.runner.election_arrays else None
        cfg = self.state.score_config

        save_metrics(
            self._export_labeled_assignment(current),
            output_path,
            populations=self.runner.populations,
            ideal_pop=ideal_pop,
            dem_votes=dem,
            gop_votes=gop,
            pp_data=self.runner.pp_data,
            reock_data=self.runner.reock_data,
            county_ids=self.runner.county_array,
            win_prob_at_55=cfg.election_win_prob_at_55,
            swing_sigma=cfg.election_swing_sigma,
        )
        self.state.update(status_message=f"Metrics saved to {output_path}")

    # ── File menu: New / recent files / named saves ───────────────────────────

    def _load_recent_shapefiles(self) -> None:
        """Read recent-shapefile list from disk."""
        import json
        if not _RECENT_FILE.exists():
            return
        try:
            data = json.loads(_RECENT_FILE.read_text(encoding="utf-8"))
            self._recent_shapefiles = data if isinstance(data, list) else []
        except Exception:
            self._recent_shapefiles = []

    def _save_recent_shapefiles(self) -> None:
        """Persist recent-shapefile list to disk."""
        import json
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            _RECENT_FILE.write_text(
                json.dumps(self._recent_shapefiles[:5], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _push_recent_shapefile(self, path: str,
                               config: "ShapefileConfig") -> None:
        """Prepend path+config to the recent list, cap at 5, refresh menu."""
        entry = {
            "path": path,
            "config": {
                "pop_col":    config.pop_col,
                "id_col":     config.id_col,
                "county_col": config.county_col,
                "elections":  config.elections,
            },
        }
        self._recent_shapefiles = [
            e for e in self._recent_shapefiles if e.get("path") != path
        ]
        self._recent_shapefiles.insert(0, entry)
        self._recent_shapefiles = self._recent_shapefiles[:5]
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
        if not user_data:
            return
        entry = user_data
        path = entry.get("path", "")
        if not path or not Path(path).exists():
            self.state.update(
                status_message=f"Recent file not found: {path}")
            return
        cfg_d = entry.get("config", {})
        # JSON round-trips tuples as lists; restore tuples so the runner's
        # for dem_col, gop_col in config.elections unpacking works correctly.
        raw_elections = cfg_d.get("elections", [])
        elections = [tuple(e) for e in raw_elections if len(e) == 2]
        config = ShapefileConfig(
            pop_col=cfg_d.get("pop_col", ""),
            id_col=cfg_d.get("id_col", ""),
            county_col=cfg_d.get("county_col"),
            elections=elections,
        )
        self._pending_recent_config = config
        self._restore_partisan_on_load = bool(elections)
        self._on_shapefile_selected(None, {"file_path_name": path})

    def _on_close(self) -> None:
        """File > Close / Ctrl+W: warn if there are unsaved results."""
        if self.state.best_assignment is not None:
            dpg.configure_item("popup_close_confirm", show=True)
        else:
            dpg.stop_dearpygui()

    def _on_new(self) -> None:
        """File > New: clear the map with an optional unsaved-work warning."""
        if self.state.best_assignment is not None:
            dpg.configure_item("popup_new_confirm", show=True)
        else:
            self._do_new()

    def _do_new(self) -> None:
        """Discard current results and return to a clean slate."""
        dpg.configure_item("popup_new_confirm", show=False)
        self._on_reset()
        self.runner = AlgorithmRunner(self.state)
        self._loaded_config = None
        self._has_elections = False
        self._map_loaded_path = ""
        self._map_loaded_gdf_id = 0
        self._map_data_gdf_id = 0
        self._map_loading = False
        self._map_ready = False
        self.state.update(shapefile_path="", status_message="")
        self._update_hot_start_display(None)
        dpg.set_value(self._shp_info, "Load a shapefile to begin.")
        self.theme.retoken(self._shp_info, "muted")
        if self.map_view is not None:
            self.map_view.wipe()

    def _build_new_confirm_popup(self) -> None:
        with self._dialog(
            "New Map", "popup_new_confirm", (380, 120),
            show=False,
            buttons=[
                ("Discard & New", self._do_new, "primary"),
                ("Cancel",
                 lambda: dpg.configure_item("popup_new_confirm", show=False)),
            ],
        ):
            dpg.add_text(
                "The current run has results that have not been saved.\n"
                "Discard them and start a new map?",
                wrap=380 - 2 * 16,
            )

    def _build_close_confirm_popup(self) -> None:
        with self._dialog(
            "Close Mosaic", "popup_close_confirm", (380, 120),
            show=False,
            buttons=[
                ("Close Anyway", lambda: dpg.stop_dearpygui(), "primary"),
                ("Cancel",
                 lambda: dpg.configure_item("popup_close_confirm", show=False)),
            ],
        ):
            dpg.add_text(
                "The current run has results that have not been saved.\n"
                "Close anyway?",
                wrap=380 - 2 * 16,
            )

    def _build_save_name_popup(self) -> None:
        """Shared filename-input dialog for Save Assignments / Save District Info."""
        with self._dialog(
            "Save", "popup_save_name", (400, 150),
            show=False,
            buttons=[
                ("Save", self._do_save_name_confirm, "primary"),
                ("Cancel",
                 lambda: dpg.configure_item("popup_save_name", show=False)),
            ],
        ):
            dpg.add_text("Save to: output/")
            with dpg.group(horizontal=True):
                self._save_name_input = dpg.add_input_text(
                    default_value="", width=318, hint="filename",
                )
                dpg.add_text(".csv")

    def _on_file_save_assignments(self) -> None:
        """File > Save Assignments: show filename dialog."""
        if self.state.best_assignment is None:
            return
        from datetime import datetime
        suggested = f"assignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dpg.configure_item("popup_save_name", label="Save Assignments")
        dpg.set_value(self._save_name_input, suggested)
        self._save_name_type = "assignments"
        dpg.configure_item("popup_save_name", show=True)

    def _on_file_save_metrics(self) -> None:
        """File > Save District Info: show filename dialog."""
        if self.state.best_assignment is None:
            return
        from datetime import datetime
        suggested = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dpg.configure_item("popup_save_name", label="Save District Info")
        dpg.set_value(self._save_name_input, suggested)
        self._save_name_type = "metrics"
        dpg.configure_item("popup_save_name", show=True)

    def _do_save_name_confirm(self) -> None:
        """Save Assignments / Save District Info: confirm filename and export."""
        name = dpg.get_value(self._save_name_input).strip()
        if not name:
            return
        if not name.endswith(".csv"):
            name += ".csv"
        output_path = Path("output") / name
        dpg.configure_item("popup_save_name", show=False)
        if self._save_name_type == "assignments":
            self._do_export_to_path(output_path)
        else:
            self._do_export_metrics_to_path(output_path)

    def _render_map_at_scale(self, scale: float,
                             state_outline: bool = False) -> Optional[np.ndarray]:
        """Re-rasterize the current map at `scale`x resolution, cropped to the
        state's aspect ratio, and return rgba.

        Long edge is `_MAP_DW * scale` regardless of orientation; short edge is
        derived from the gdf's bounding-box aspect. Falls back to LANCZOS
        upscale of the cached frame if the runner/gdf isn't available.
        """
        src = self.map_view
        if src is None or not src._loaded:
            return None

        if (self.runner is None or self.runner.gdf is None
                or self.state.current_assignment is None):
            if src._last_rgba is None:
                return None
            if scale == 1.0:
                return src._last_rgba.copy()
            from PIL import Image
            img = Image.fromarray(src._last_rgba, mode="RGBA")
            new_size = (int(img.width * scale), int(img.height * scale))
            return np.asarray(
                img.resize(new_size, Image.LANCZOS), dtype=np.uint8,
            )

        bounds = self.runner.gdf.total_bounds
        gw = max(float(bounds[2] - bounds[0]), 1e-9)
        gh = max(float(bounds[3] - bounds[1]), 1e-9)
        long_edge = max(1, int(_MAP_DW * scale))
        if gw >= gh:
            w = long_edge
            h = max(1, int(round(long_edge * gh / gw)))
        else:
            h = long_edge
            w = max(1, int(round(long_edge * gw / gh)))
        offscreen = MapView(texture_tag="__offscreen", draw_w=w, draw_h=h)
        offscreen._bg_color = src._bg_color.copy()
        for flag in ("county_overlay", "partisan_overlay",
                     "district_partisan_overlay", "splits_view",
                     "compactness_view", "pop_dev_view", "show_labels",
                     "precinct_overlay"):
            setattr(offscreen, flag, getattr(src, flag))
        offscreen.fast_labels = False  # precise label placement for export
        offscreen.state_outline = state_outline
        # Half-rate proportionality: doubling DPI grows lines/labels by ~50%,
        # not 100%. Keeps high-DPI exports from over-emphasizing borders.
        offscreen.border_thickness = max(1, int(round(1 + 0.5 * (scale - 1))))

        dem = (self.runner.election_arrays[0][0]
               if self.runner.election_arrays else None)
        gop = (self.runner.election_arrays[0][1]
               if self.runner.election_arrays else None)
        offscreen.load(
            self.runner.gdf,
            county_array=self.runner.county_array,
            dem_votes=dem, gop_votes=gop,
            pp_data=self.runner.pp_data,
            reock_data=self.runner.reock_data,
            populations=self.runner.populations,
        )

        with self.state._lock:
            assignment = self.state.current_assignment.copy()
            initial = (self.state.initial_assignment.copy()
                       if self.state.initial_assignment is not None else None)
            n_dist = self.state.num_districts
        return offscreen.compose_rgba(assignment, n_dist, initial)

    def _on_save_map(self):
        if getattr(self, "_saving", False):
            return
        if self.map_view is None or not self.map_view._loaded:
            self.state.update(status_message="Nothing to save: load a shapefile first.")
            return
        self._saving = True
        dpg.configure_item(self._save_spinner, show=True)
        self.state.update(status_message="Saving map...")

        def _worker():
            from datetime import datetime
            from PIL import Image
            try:
                rgba = self._render_map_at_scale(1.0)
                if rgba is None:
                    self.state.update(
                        status_message="Save failed: map not ready.",
                    )
                    return
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path("output") / f"map_{timestamp}.png"
                output_path.parent.mkdir(exist_ok=True)
                Image.fromarray(rgba, mode="RGBA").save(output_path)
                self.state.update(status_message=f"Map saved to {output_path}")
            finally:
                dpg.configure_item(self._save_spinner, show=False)
                self._saving = False
        threading.Thread(target=_worker, daemon=True).start()

    def _on_advanced_save_open(self):
        if self.map_view is None or self.map_view._last_rgba is None:
            self.state.update(status_message="Nothing to save: load a shapefile first.")
            return
        dpg.configure_item("popup_adv_save", show=True)

    def _on_adv_fmt_changed(self):
        """Show Raster DPI row only when PNG is selected."""
        is_pdf = "PDF" in dpg.get_value(self._adv_save_fmt)
        dpg.configure_item(self._adv_dpi_group, show=not is_pdf)

    def _adv_save_ready(self) -> bool:
        return (self.map_view is not None
                and self.map_view._loaded
                and self.state.current_assignment is not None)

    def _adv_begin(self, status: str) -> None:
        self._saving = True
        for btn in (self._adv_save_btn, self._adv_save_as_btn,
                    self._adv_close_btn):
            dpg.configure_item(btn, enabled=False)
        dpg.configure_item(self._adv_save_spinner, show=True)
        dpg.set_value(self._adv_save_status, status)
        dpg.configure_item(self._adv_save_status, show=True)

    def _adv_finish(self, close: bool = True) -> None:
        dpg.configure_item(self._adv_save_spinner, show=False)
        dpg.configure_item(self._adv_save_status, show=False)
        for btn in (self._adv_save_btn, self._adv_save_as_btn,
                    self._adv_close_btn):
            dpg.configure_item(btn, enabled=True)
        if close:
            dpg.configure_item("popup_adv_save", show=False)
        self._saving = False

    def _adv_get_dpi(self) -> int:
        if not hasattr(self, "_adv_save_dpi") or self._adv_save_dpi is None:
            return 192
        return int(dpg.get_value(self._adv_save_dpi).split()[0])

    def _adv_dispatch(self, output_path) -> None:
        """Start the PNG or PDF worker for the current format selection."""
        fmt_str = dpg.get_value(self._adv_save_fmt) if self._adv_save_fmt else ""
        is_pdf = "PDF" in fmt_str
        title = dpg.get_value(self._adv_save_title).strip()
        dpi = self._adv_get_dpi()
        if is_pdf:
            threading.Thread(
                target=self._pdf_vector_worker,
                args=(title, output_path),
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._png_worker,
                args=(title, dpi, output_path),
                daemon=True,
            ).start()

    def _on_advanced_save_confirm(self):
        """Save to an auto-timestamped path."""
        if getattr(self, "_saving", False):
            return
        if not self._adv_save_ready():
            self.state.update(status_message="Nothing to save: load a shapefile first.")
            dpg.configure_item("popup_adv_save", show=False)
            return
        fmt_str = dpg.get_value(self._adv_save_fmt) if self._adv_save_fmt else ""
        is_pdf = "PDF" in fmt_str
        dpi = self._adv_get_dpi()
        self._adv_begin("Rendering PDF..." if is_pdf else f"Rendering {dpi} DPI...")
        self._adv_dispatch(output_path=None)

    def _on_advanced_save_as(self):
        """Save to a chosen path (Windows native dialog); auto-name elsewhere."""
        if getattr(self, "_saving", False):
            return
        if not self._adv_save_ready():
            self.state.update(status_message="Nothing to save: load a shapefile first.")
            return
        fmt_str = dpg.get_value(self._adv_save_fmt) if self._adv_save_fmt else ""
        is_pdf = "PDF" in fmt_str
        import os
        if os.name != "nt":
            # No native save dialog off Windows: PowerShell is Windows-only, and a
            # tkinter dialog would risk the mac/DPG run-loop conflict the Open
            # picker already avoids. Fall back to an auto-named file in output/.
            dpi = self._adv_get_dpi()
            self._adv_begin("Rendering PDF..." if is_pdf else f"Rendering {dpi} DPI...")
            self._adv_dispatch(output_path=None)
            return
        self._adv_begin("Waiting for save dialog...")

        def _ask():
            import subprocess
            from datetime import datetime as _dt
            ext = ".pdf" if is_pdf else ".png"
            filt = ("PDF Document|*.pdf" if is_pdf else "PNG Image|*.png")
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"map_{ts}"
            output_dir = _ASSETS_DIR.parent.parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            init_dir = str(output_dir)
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$d = New-Object System.Windows.Forms.SaveFileDialog; "
                f"$d.Filter = '{filt}'; "
                f"$d.DefaultExt = '{ext.lstrip('.')}'; "
                f"$d.InitialDirectory = '{init_dir}'; "
                f"$d.FileName = '{default_name}'; "
                "$null = $d.ShowDialog(); "
                "Write-Output $d.FileName"
            )
            try:
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive",
                     "-Command", ps],
                    capture_output=True, text=True, timeout=120,
                )
                path = r.stdout.strip()
            except Exception:
                path = ""
            if not path:
                self._adv_finish(close=False)
                return
            dpi = self._adv_get_dpi()
            dpg.set_value(self._adv_save_status,
                          "Rendering PDF..." if is_pdf
                          else f"Rendering {dpi} DPI...")
            self._adv_dispatch(output_path=Path(path))

        threading.Thread(target=_ask, daemon=True).start()

    # ── Save workers ─────────────────────────────────────────────────────────

    def _pdf_vector_worker(self, title: str, output_path) -> None:
        """Build a true vector PDF on US Letter paper (auto landscape/portrait).
        Precinct fills and all outlines are real vector paths, not a raster."""
        from datetime import datetime
        try:
            if self.runner is None or self.runner.gdf is None:
                self.state.update(
                    status_message="PDF save failed: no shapefile loaded.")
                return
            with self.state._lock:
                assignment = self.state.current_assignment.copy()
                initial = (self.state.initial_assignment.copy()
                           if self.state.initial_assignment is not None
                           else None)
                n_dist = self.state.num_districts

            gdf = self.runner.gdf
            mv  = self.map_view
            n   = len(gdf)

            # ── Per-precinct fill colours (same logic as MapView LUTs) ────────
            if mv.partisan_overlay and mv._dem_votes is not None:
                lut = mv._build_partisan_lut()
                fill_rgba = lut[:n]
            elif mv.district_partisan_overlay and mv._dem_votes is not None:
                lut = mv._build_district_partisan_lut(assignment, n_dist)
                fill_rgba = lut[:n]
            elif mv.compactness_view and mv._pp_data is not None:
                lut = mv._build_compactness_lut(assignment, n_dist)
                fill_rgba = lut[:n]
            elif mv.pop_dev_view and mv._populations is not None:
                lut = mv._build_pop_dev_lut(assignment, n_dist)
                fill_rgba = lut[:n]
            else:
                from mosaic.gui.map_view import DISTRICT_COLORS, stable_color_mapping
                nc = len(DISTRICT_COLORS)
                ci_arr = (stable_color_mapping(assignment, initial, n_dist)
                          if initial is not None and len(initial) == len(assignment)
                          else assignment)
                fill_rgba = np.zeros((n, 4), dtype=np.uint8)
                for i in range(n):
                    r, g, b = DISTRICT_COLORS[int(ci_arr[i]) % nc]
                    fill_rgba[i] = (r, g, b, 255)

            fill_mpl = fill_rgba[:, :3] / 255.0

            # ── Matplotlib setup ──────────────────────────────────────────────
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
            import matplotlib.patheffects as _pe
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path as MplPath
            from matplotlib.collections import PatchCollection
            from matplotlib.font_manager import FontProperties
            from shapely.ops import unary_union as _shapely_union

            bounds = gdf.total_bounds
            bx0, by0, bx1, by1 = (float(v) for v in bounds)
            gw = max(bx1 - bx0, 1e-9)
            gh = max(by1 - by0, 1e-9)
            map_ratio = gw / gh

            br, bg_c, bb, _ = self.theme.color("child_bg")
            bg_mpl = (br / 255.0, bg_c / 255.0, bb / 255.0)

            # ── Page layout: build outward from the state's exact bbox ────────
            #   bbox (no margin) -> +10% height below (caption band)
            #                    -> +15% height above (title band)
            #                    -> +5% width each side (blank margin)
            # The resulting content box is fit to US Letter in whichever
            # orientation (portrait 8.5x11 / landscape 11x8.5) makes it larger,
            # then centered on the page inside a small physical safety margin.
            SIDE_FRAC   = 0.01                       # each side, of bbox width
            BOTTOM_FRAC = 0.10                       # caption band, of bbox height
            TOP_FRAC    = 0.15 if title else 0.0     # title band, of bbox height
            PAGE_MARGIN = 0.5                        # inches, printer safe area

            content_ratio = ((1.0 + 2 * SIDE_FRAC)
                             / (1.0 + BOTTOM_FRAC + TOP_FRAC)) * map_ratio

            def _fit_content(fw, fh):
                pw, ph = fw - 2 * PAGE_MARGIN, fh - 2 * PAGE_MARGIN
                if pw <= 0 or ph <= 0:
                    return 0.0, 0.0, 0.0
                if pw / ph >= content_ratio:
                    cw, ch = ph * content_ratio, ph
                else:
                    cw, ch = pw, pw / content_ratio
                return cw, ch, cw * ch

            _pw, _ph, p_area = _fit_content(8.5, 11.0)
            _lw, _lh, l_area = _fit_content(11.0, 8.5)
            if p_area >= l_area:
                fig_w, fig_h = 8.5, 11.0
                content_w, content_h = _pw, _ph
            else:
                fig_w, fig_h = 11.0, 8.5
                content_w, content_h = _lw, _lh

            map_w = content_w / (1.0 + 2 * SIDE_FRAC)
            map_h = content_h / (1.0 + BOTTOM_FRAC + TOP_FRAC)
            content_left   = (fig_w - content_w) / 2.0
            content_bottom = (fig_h - content_h) / 2.0

            # Map axes = bbox, offset by the left margin and the bottom band.
            ax_l   = (content_left + SIDE_FRAC * map_w) / fig_w
            ax_b   = (content_bottom + BOTTOM_FRAC * map_h) / fig_h
            ax_w   = map_w / fig_w
            ax_h_f = map_h / fig_h

            # Text anchors (figure fractions): caption at 80% width, centered in
            # the bottom band; title at 50% width, centered in the top band.
            cap_x   = (content_left + 0.80 * content_w) / fig_w
            cap_y   = (content_bottom + 0.5 * BOTTOM_FRAC * map_h) / fig_h
            title_x = (content_left + 0.50 * content_w) / fig_w
            title_y = (content_bottom
                       + (BOTTOM_FRAC + 1.0 + 0.5 * TOP_FRAC) * map_h) / fig_h

            fig = plt.figure(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor(bg_mpl)

            ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h_f])
            ax.set_facecolor(bg_mpl)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_xlim(bx0, bx1)
            ax.set_ylim(by0, by1)

            def _to_mpl_path(geom):
                """Shapely Polygon/MultiPolygon → matplotlib Path with holes."""
                if geom is None:
                    return None
                gtype = geom.geom_type
                if gtype == "Polygon":
                    polys = [geom]
                elif gtype == "MultiPolygon":
                    polys = list(geom.geoms)
                else:
                    return None
                verts, codes = [], []
                for poly in polys:
                    coords = np.asarray(poly.exterior.coords)
                    if len(coords) < 3:
                        continue
                    verts.append(coords)
                    codes.append(np.array(
                        [MplPath.MOVETO]
                        + [MplPath.LINETO] * (len(coords) - 2)
                        + [MplPath.CLOSEPOLY], dtype=np.uint8))
                    for ring in poly.interiors:
                        coords = np.asarray(ring.coords)
                        if len(coords) < 3:
                            continue
                        verts.append(coords)
                        codes.append(np.array(
                            [MplPath.MOVETO]
                            + [MplPath.LINETO] * (len(coords) - 2)
                            + [MplPath.CLOSEPOLY], dtype=np.uint8))
                if not verts:
                    return None
                return MplPath(np.concatenate(verts, axis=0),
                               np.concatenate(codes))

            # ── Gentle geometry simplification to shrink the vector PDF ───────
            # Douglas-Peucker at a fraction of a print pixel (fitted map size at
            # 300 DPI), so the deviation stays sub-pixel and invisible on paper.
            # Per-precinct vertex count is the dominant size driver and the
            # precinct grid reuses these same paths, so simplifying here shrinks
            # both the fills and the grid. The district/county dissolve below
            # keeps the original geometry (see note there). Raise _SIMPLIFY_PX to
            # shrink more aggressively (watch for hairline gaps between precincts).
            _SIMPLIFY_PX = 0.5
            _data_per_px = (gw / map_w) / 300.0 if map_w > 0 else 0.0
            _simp_tol = _data_per_px * _SIMPLIFY_PX
            geom_simplified = (gdf.geometry.simplify(_simp_tol)
                               if _simp_tol > 0 else gdf.geometry)

            # ── Precinct fills (batched PatchCollection) ──────────────────────
            patches, colors = [], []
            for i, geom in enumerate(geom_simplified):
                path = _to_mpl_path(geom)
                if path is None:
                    continue
                patches.append(PathPatch(path))
                colors.append(fill_mpl[i])
            if patches:
                # Fills: no edges (keeps PDF edge transparency independent)
                ax.add_collection(
                    PatchCollection(patches, facecolors=colors, edgecolors='none'))
                # Edges: stroke-only collection; set_alpha() writes a proper PDF /CA entry
                edge_coll = PatchCollection(patches, facecolors='none',
                                            edgecolors='white', linewidths=0.2)
                edge_coll.set_alpha(0.15)
                ax.add_collection(edge_coll)

            # ── Pre-dissolve district & county geometries (fast, avoids
            #    looping n_dist × unary_union over all precincts) ───────────────
            # Dissolve from ORIGINAL geometry: independently-simplified precincts
            # no longer share exact edges, so their union leaves interior slivers
            # that the district/county borders would trace as jagged lines.
            gdf_work = gdf[["geometry"]].copy()
            gdf_work["_dist"] = assignment
            have_county = (mv._county_array is not None
                           and (mv.county_overlay or mv.splits_view))
            if have_county:
                gdf_work["_cty"] = mv._county_array
                cty_geoms = gdf_work.dissolve(by="_cty").geometry
            dist_geoms = gdf_work.dissolve(by="_dist").geometry

            # ── Splits view: dim clean (un-split) counties ────────────────────
            if mv.splits_view and mv._county_array is not None:
                ca  = mv._county_array
                nct = int(ca.max()) + 1
                flt = (ca * n_dist + assignment).astype(np.int64)
                co_di = (np.bincount(flt, minlength=nct * n_dist)
                           .reshape(nct, n_dist))
                county_clean = (co_di > 0).sum(axis=1) <= 1
                dim_c = (28 / 255, 28 / 255, 28 / 255)
                dim_patches = []
                for ci_idx in range(nct):
                    if county_clean[ci_idx] and ci_idx in cty_geoms.index:
                        p = _to_mpl_path(cty_geoms.loc[ci_idx])
                        if p:
                            dim_patches.append(PathPatch(p))
                if dim_patches:
                    ax.add_collection(
                        PatchCollection(dim_patches,
                                        facecolors=[dim_c] * len(dim_patches),
                                        edgecolors="none", linewidths=0))

            # ── County borders ────────────────────────────────────────────────
            if have_county:
                cty_color = (180 / 255, 180 / 255, 180 / 255)
                for geom in cty_geoms:
                    p = _to_mpl_path(geom)
                    if p:
                        ax.add_patch(PathPatch(p, facecolor="none",
                                               edgecolor=cty_color,
                                               linewidth=0.5))

            # ── District borders (dissolved geoms, no per-precinct loop) ──────
            for geom in dist_geoms:
                p = _to_mpl_path(geom)
                if p:
                    ax.add_patch(PathPatch(p, facecolor="none",
                                           edgecolor="black", linewidth=0.7))

            # ── State outline (union of already-dissolved district geoms) ──────
            state_geom = _shapely_union(list(dist_geoms.values))
            p = _to_mpl_path(state_geom)
            if p:
                ax.add_patch(PathPatch(p, facecolor="none",
                                       edgecolor="black", linewidth=1.2))

            # ── District labels ───────────────────────────────────────────────
            if mv.show_labels:
                from mosaic.gui.map_view import stable_color_mapping
                from shapely.ops import polylabel as _polylabel
                ci_arr = (stable_color_mapping(assignment, initial, n_dist)
                          if initial is not None and len(initial) == len(assignment)
                          else assignment)
                lm     = mv.district_label_map
                use_lm = lm is not None and len(lm) == n_dist
                font_path = _ASSETS_DIR / "fonts" / "inter" / "Inter-SemiBold.ttf"
                fp = (FontProperties(fname=str(font_path))
                      if font_path.exists() else None)
                stroke_fx = [_pe.withStroke(linewidth=1.75, foreground="black")]
                # Pole of inaccessibility (deepest interior point) rather than
                # representative_point(), which can sit on a concave edge.
                _lbl_tol = max(gw, gh) / 1000.0
                for d in range(n_dist):
                    if d not in dist_geoms.index:
                        continue
                    mask = assignment == d
                    if not mask.any():
                        continue
                    si       = int(ci_arr[mask][0])
                    num      = int(lm[si]) if use_lm else si + 1
                    _dgeom = dist_geoms.loc[d]
                    if _dgeom.geom_type == "MultiPolygon":
                        _dgeom = max(_dgeom.geoms, key=lambda g: g.area)
                    try:
                        label_pt = _polylabel(_dgeom, tolerance=_lbl_tol)
                    except Exception:
                        label_pt = dist_geoms.loc[d].representative_point()
                    kw = dict(ha="center", va="center", fontsize=7,
                              color="white", path_effects=stroke_fx)
                    if fp:
                        ax.text(label_pt.x, label_pt.y, str(num),
                                fontproperties=fp, **kw)
                    else:
                        ax.text(label_pt.x, label_pt.y, str(num),
                                fontweight="bold", **kw)

            # ── Title (50% width, centered in the top band) ───────────────────
            if title:
                body_r, body_g, body_b, _ = self.theme.color("body")
                fig.text(title_x, title_y, title,
                         ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color=(body_r / 255, body_g / 255, body_b / 255))

            # ── Caption (80% width, centered in the bottom band) ──────────────
            cap_r, cap_g, cap_b, _ = self.theme.color("body")
            fig.text(cap_x, cap_y, "Made with Mosaic",
                     ha="center", va="center", fontsize=7,
                     color=(cap_r / 255, cap_g / 255, cap_b / 255))

            # ── Write file ────────────────────────────────────────────────────
            if output_path is None:
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = (_ASSETS_DIR.parent.parent.parent
                               / "output" / f"map_{timestamp}.pdf")
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)
            fig.savefig(str(output_path), format="pdf")
            plt.close(fig)
            self.state.update(status_message=f"Map saved to {output_path}")
            self._open_in_os(output_path)

        except Exception as exc:
            self.state.update(status_message=f"PDF export failed: {exc}")
        finally:
            self._adv_finish()

    def _png_worker(self, title: str, dpi: int, output_path) -> None:
        """Rasterise the map at the given DPI and save as PNG."""
        from datetime import datetime
        from PIL import Image, ImageDraw, ImageFont
        scale = dpi / 96.0
        try:
            rgba = self._render_map_at_scale(scale, state_outline=True)
            if rgba is None:
                self.state.update(status_message="Save failed: map not ready.")
                return

            img = Image.fromarray(rgba, mode="RGBA")
            font_path = _ASSETS_DIR / "fonts" / "inter" / "Inter-SemiBold.ttf"

            br, bg, bb, _ = self.theme.color("child_bg")

            cap_size       = max(10, int(11 * scale))
            cap_margin     = max(8,  int(12 * scale))
            bottom_strip_h = cap_size + 2 * cap_margin
            try:
                cap_font = ImageFont.truetype(str(font_path), cap_size)
            except OSError:
                cap_font = ImageFont.load_default()

            top_strip_h = 0
            title_font  = None
            if title:
                r, g, b, _ = self.theme.color("body")
                top_strip_h = max(28, int(36 * scale))
                font_size   = max(14, int(18 * scale))
                try:
                    title_font = ImageFont.truetype(str(font_path), font_size)
                except OSError:
                    title_font = ImageFont.load_default()

            new_img = Image.new(
                "RGBA",
                (img.width, img.height + top_strip_h + bottom_strip_h),
                (br, bg, bb, 255),
            )
            new_img.paste(img, (0, top_strip_h), img)
            draw = ImageDraw.Draw(new_img)
            if title:
                draw.text(
                    (img.width // 2, top_strip_h // 2), title,
                    fill=(r, g, b, 255), font=title_font, anchor="mm",
                )
            cap_r, cap_g, cap_b, _ = self.theme.color("body")
            draw.text(
                (new_img.width - cap_margin, new_img.height - cap_margin),
                "Made with Mosaic",
                fill=(cap_r, cap_g, cap_b, 255),
                font=cap_font, anchor="rs",
            )

            if output_path is None:
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path("output") / f"map_{timestamp}.png"
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)
            new_img.save(output_path, dpi=(dpi, dpi))
            self.state.update(status_message=f"Map saved to {output_path}")
            self._open_in_os(output_path)
        finally:
            self._adv_finish()


def main():
    # The .bat / .command / .sh launchers invoke this entry point directly
    # via `python -m mosaic.gui.app`, bypassing mosaic:main. Wire up logging
    # here too so the 'mosaic' logger always has a console + file handler.
    from mosaic import _setup_logging
    _setup_logging()

    app = MosaicApp()
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
