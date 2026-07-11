"""Shared imports, constants, and helpers for the Mosaic GUI app package."""

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

# NB: this module lives at gui/app/_common.py -- one level deeper than the old
# gui/app.py, so every __file__-relative path carries one extra .parent.
_ASSETS_DIR   = Path(__file__).resolve().parent.parent.parent / "assets"
# App-level settings/cache directory — separate from output/ which holds generated files.
_SETTINGS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / ".mosaic"
_RECENT_FILE  = _SETTINGS_DIR / "recent_shapefiles.json"
_APP_ICON = _ASSETS_DIR / "mosaic_logo.ico"
_PDF_PRECINCT_OFF_ALPHA = 0.05   # faint precinct hairlines in PDF even when the overlay is off

import dearpygui.dearpygui as dpg
import numpy as np

from mosaic import __version__
from mosaic.gui.map_view import PRECINCT_EDGE_ALPHA, MapView
from mosaic.gui.runner import AlgorithmRunner
from mosaic.gui.shp_dialog import ShapefileDialog
from mosaic.gui.state import AlgorithmStatus, SharedState
from mosaic.gui.theme import ThemeManager
from mosaic.io.inspect import ShapefileConfig, ShapefileInspection
from mosaic.paths import output_dir
from mosaic.recom.annealing import AnnealingConfig
from mosaic.scoring.score import ScoreConfig

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
    ("Partisan Bias",        "partisan_bias_history",             "raw"),
    ("Partisan Bias (abs)",  "partisan_bias_history",             "abs"),
    ("Partisan Gini",        "partisan_gini_history",             "raw"),
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
    (0.000, (82, 66, 159)), (0.049, (90, 68, 165)),  (0.268, (170, 60, 120)),
    (0.512, (228, 90, 60)), (0.756, (250, 160, 40)), (1.000, (252, 253, 191)),
]
_PHASE_RAMP_LIGHT = [
    (0.000, (166, 224, 188)),  # oldest: light mint-green
    (0.133, (100, 200, 140)),  # green
    (0.356, ( 45, 148, 142)),  # teal
    (0.578, ( 54,  93, 141)),  # blue
    (0.778, ( 52,  38,  86)),  # indigo
    (1.000, ( 14,   5,   8)),  # hottest: near-black
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
    ("Partisan Bias",   "P.Bias", (235, 110, 110, 220)),
    ("Partisan Gini",   "Gini",   (200, 120, 160, 220)),
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
    "partisan_bias": (
        "Seat-share tilt at a hypothetical 50/50 statewide vote (uniform swing). "
        "0 = symmetric; sign shows which party would hold the seat majority at a tie."
    ),
    "partisan_gini": (
        "Area between the seats-votes curve and its mirror image. Unsigned "
        "measure of asymmetry; 0 = a perfectly symmetric plan."
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


# Public surface re-exported to the app mixins. Listed explicitly so
# ruff F401 does not treat mixin-only names as dead imports.
__all__ = [
    "logging",
    "threading",
    "time",
    "warnings",
    "webbrowser",
    "contextmanager",
    "Path",
    "Optional",
    "log",
    "_DOCS_URL",
    "_UPDATE_CHECK_URL",
    "_DOCS_SHAPEFILE_URL",
    "_DOWNLOAD_URL",
    "_ASSETS_DIR",
    "_SETTINGS_DIR",
    "_RECENT_FILE",
    "_APP_ICON",
    "_PDF_PRECINCT_OFF_ALPHA",
    "dpg",
    "np",
    "output_dir",
    "SharedState",
    "AlgorithmStatus",
    "AlgorithmRunner",
    "MapView",
    "PRECINCT_EDGE_ALPHA",
    "ShapefileDialog",
    "ThemeManager",
    "ShapefileConfig",
    "ShapefileInspection",
    "ScoreConfig",
    "AnnealingConfig",
    "__version__",
    "_PLOT_LIMIT",
    "_COMPACT_AT",
    "_COMPACT_KEEP",
    "_COMPACT_THIN",
    "_PHASE_BANDS",
    "_PHASE_HEAD",
    "_PHASE_BUDGET",
    "_PHASE_TAU",
    "_PHASE_FOLLOW",
    "_PHASE_SMOOTH_WIN",
    "_PHASE_MIN_SPAN",
    "_PHASE_METRICS",
    "_PHASE_ATTR",
    "_PHASE_KIND",
    "_PHASE_LABELS",
    "_PHASE_AXIS_UNIT",
    "_PHASE_RAMP",
    "_PHASE_RAMP_LIGHT",
    "_ramp_color",
    "_phase_transform",
    "_RENDER_TARGET",
    "_CAMERA_ICON_SIZE",
    "_build_camera_icon",
    "_build_more_icon",
    "_SeriesBuffer",
    "_DIR_TO_MODE",
    "_CONTRIB_BAR_METRICS",
    "_VP_W",
    "_VP_H",
    "_LEFT_W",
    "_TOP_H",
    "_SCORE_H",
    "_SCORE_COL_W",
    "_MAP_H",
    "_MAP_DW",
    "_DIALOG_PAD",
    "_DIALOG_GAP",
    "_DIALOG_BTN_W",
    "_DIALOG_RM",
    "_MAP_DH",
    "_PLOT_H",
    "_HALF_PLOT_H",
    "_HALF_PLOT_W",
    "_HINTS",
    "_fmt_dur",
]
