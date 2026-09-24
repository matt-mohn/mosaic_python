"""
Targeting — tune an ensemble's chain settings from its own runs.

Every run draws each tuned setting at random inside a band around its
current centre. Every REFIT_EVERY runs (from FIRST_FIT) one regression of
score on all the settings, over a recent window of runs (WINDOW_PER_KNOB per tuned
setting, at least MIN_WINDOW), estimates score associations with bootstrap
intervals. This adaptive beta heuristic does not establish causal effects or
guarantee calibrated uncertainty. A
setting's centre moves toward the better side only when its interval excludes
zero. Steps start at a quarter band and grow STEP_GROWTH-fold with each move
in the same direction as the one before (capped at one band), so a persistent
effect accelerates; a move the other way resets the step, so noise that
flips direction stays at small steps. The window keeps the straight-line fit
local to where the centres are now.

With tail=True the regression is a quantile regression
at TAIL_QUANTILE instead of OLS: it estimates how each setting moves the score
of the better runs rather than the average one, for goals where the best maps
are rare outliers. The tail is thin, so its window is TAIL_WINDOW_FACTOR wider.

The first WARMUP_RUNS runs are labelled "warmup", later ones "learning".
Estimation begins at FIRST_FIT, so settings can change during warm-up.

Score is the objective, lower is better. Scoring settings (weights, targets,
county bias) are never tuned: they define what "better" means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

WARMUP_RUNS = 20
FIRST_FIT = 10
REFIT_EVERY = 5
WINDOW_PER_KNOB = 10         # rule of thumb: ~10 runs per estimated effect
MIN_WINDOW = 30
STEP_GROWTH = 1.5
TAIL_QUANTILE = 0.20
TAIL_WINDOW_FACTOR = 2
MAX_ITERATIONS = 300_000     # hard ceiling until run cost is part of the objective
_MIN_ITERATIONS = 100
CI_LEVEL = 0.90
BOOTSTRAPS = 400
TAIL_BOOTSTRAPS = 200        # fewer replicates to limit the cost of quantile fits


@dataclass
class Knob:
    """One tuned setting. Work happens in 'space' units: log for multiplicative
    settings (iterations, temperature), raw otherwise. half_band is the band's
    half-width in space units."""
    name: str
    label: str
    center: float            # space units
    half_band: float
    lo: float                # bounds, space units
    hi: float
    log: bool = False
    integer: bool = False
    effect: Optional[tuple[float, float, float]] = None   # (est, ci_lo, ci_hi) per band
    moved: int = 0           # +1 / -1 / 0 at the last fit
    step: float = 0.0        # current step, space units (0 = none taken yet)
    last_dir: int = 0        # direction of the most recent move (+1 / -1)
    ups: int = 0             # moves made up / down so far
    downs: int = 0

    @property
    def base_step(self) -> float:
        return self.half_band / 2

    def to_value(self, t: float) -> float:
        v = math.exp(t) if self.log else t
        return float(round(v)) if self.integer else float(v)

    def to_space(self, v: float) -> float:
        return math.log(v) if self.log else v

    def step_label(self) -> str:
        """One band width in the setting's own terms, e.g. '+32%' or '+16 pts'."""
        w = 2 * self.half_band
        if self.log:
            return f"+{(math.exp(w) - 1) * 100:.0f}%"
        return f"+{w * 100:.0f} pts"


@dataclass
class Targeter:
    knobs: list[Knob]
    rng: np.random.Generator
    warmup_runs: int = WARMUP_RUNS
    tail: bool = False       # aim at the best runs (quantile), not the average
    # Only the fitting window is retained; the ensemble files hold every run.
    history: list[tuple[dict[str, float], float]] = field(default_factory=list)
    runs_observed: int = 0

    @property
    def window(self) -> int:
        w = max(MIN_WINDOW, WINDOW_PER_KNOB * len(self.knobs))
        return w * TAIL_WINDOW_FACTOR if self.tail else w

    @classmethod
    def from_settings(cls, iterations: int, n3: float, flip_enabled: bool,
                      flip_midpoint: float, annealing_enabled: bool,
                      temp_factor: float, guided: bool = False,
                      guide_fraction: float = 0.9, tail: bool = False,
                      seed: Optional[int] = None) -> "Targeter":
        """Knobs centred on the user's own settings, bounded by the GUI's
        slider ranges (and the iteration ceiling)."""
        it = min(max(int(iterations), _MIN_ITERATIONS), MAX_ITERATIONS)
        knobs = [Knob("iterations", "Iterations", math.log(it), math.log(1.15),
                      math.log(_MIN_ITERATIONS), math.log(MAX_ITERATIONS),
                      log=True, integer=True),
                 Knob("n3_probability", "n3", min(max(n3, 0.0), 0.5), 0.08, 0.0, 0.5)]
        if flip_enabled:
            knobs.append(Knob("flip_midpoint", "Flip midpoint",
                              min(max(flip_midpoint, 0.05), 0.97), 0.06, 0.05, 0.97))
        if annealing_enabled:
            tf = min(max(temp_factor, 0.01), 2.0)
            knobs.append(Knob("temp_factor", "Temp factor", math.log(tf), math.log(1.3),
                              math.log(0.01), math.log(2.0), log=True))
            if guided:          # Static cooling has no guide point
                knobs.append(Knob("guide_fraction", "Guide point",
                                  min(max(guide_fraction, 0.5), 1.0), 0.04, 0.5, 1.0))
        return cls(knobs, np.random.default_rng(seed), tail=tail)

    # ── Per run ──────────────────────────────────────────────────────────────

    def phase(self, run: int) -> str:
        return "warmup" if run <= self.warmup_runs else "learning"

    def propose(self, run: int) -> dict[str, float]:
        """Settings for run `run`: a random draw inside each band."""
        out = {}
        for k in self.knobs:
            t = self.rng.uniform(max(k.lo, k.center - k.half_band),
                                 min(k.hi, k.center + k.half_band))
            out[k.name] = k.to_value(t)
        return out

    def observe(self, run: int, settings: dict[str, float], score: float,
                cancelled: Optional[Callable[[], bool]] = None) -> None:
        """Record a run; refit (and maybe move) on the schedule."""
        if not math.isfinite(score):
            return
        self.history.append((dict(settings), float(score)))
        self.runs_observed += 1
        del self.history[:-self.window]
        n = self.runs_observed
        if n >= FIRST_FIT and (n - FIRST_FIT) % REFIT_EVERY == 0:
            self.refit(cancelled=cancelled)

    # ── Estimation ───────────────────────────────────────────────────────────

    def refit(self, cancelled: Optional[Callable[[], bool]] = None) -> None:
        """Mean or lower-quantile regression with bootstrap intervals over runs.
        Effects are reported per band width so knobs compare on one scale."""
        recent = self.history[-self.window:]
        if not recent or (cancelled is not None and cancelled()):
            return
        X = np.array([[k.to_space(s[k.name]) for k in self.knobs] for s, _ in recent])
        y = np.array([sc for _, sc in recent])
        n, p = X.shape
        if n < p + 3:
            return
        spread = X.std(axis=0)
        use = spread > 1e-12                 # a pinned knob can't be estimated
        A = np.column_stack([np.ones(n), X[:, use]])

        def fit(rows):
            if self.tail:
                return quantile_fit(A[rows], y[rows], TAIL_QUANTILE)[1:]
            coef, *_ = np.linalg.lstsq(A[rows], y[rows], rcond=None)
            return coef[1:]

        est = fit(np.arange(n))
        reps = TAIL_BOOTSTRAPS if self.tail else BOOTSTRAPS
        samples = []
        for _ in range(reps):
            if cancelled is not None and cancelled():
                return
            samples.append(fit(self.rng.integers(0, n, n)))
        boots = np.array(samples)
        a = (1 - CI_LEVEL) / 2
        lo, hi = np.quantile(boots, [a, 1 - a], axis=0)
        j = 0
        for k, u in zip(self.knobs, use):
            k.moved = 0
            if not u:
                k.effect = None
                continue
            per_band = 2 * k.half_band
            k.effect = (est[j] * per_band, lo[j] * per_band, hi[j] * per_band)
            # Lower score is better: a CI wholly below 0 means "raise it".
            if hi[j] < 0:
                k.moved = +1
            elif lo[j] > 0:
                k.moved = -1
            if k.moved:
                # Same direction as the previous move (unclear fits in between
                # don't count against it): grow the step. A reversal resets it.
                if k.moved == k.last_dir:
                    k.step = min(k.step * STEP_GROWTH, 2 * k.half_band)
                else:
                    k.step = k.base_step
                k.last_dir = k.moved
                k.center = min(max(k.center + k.moved * k.step, k.lo), k.hi)
                k.ups += k.moved > 0
                k.downs += k.moved < 0
            j += 1

    # ── Reporting ────────────────────────────────────────────────────────────

    def status_header(self) -> str:
        n = self.runs_observed
        aim = ", lower tail" if self.tail else ""
        head = f"Targeting (Beta{aim}): {n} run{'s' if n != 1 else ''}"
        if n < FIRST_FIT:
            head += f" (first estimate after {FIRST_FIT})"
        elif n > self.window:
            head += f", fit on the last {self.window}"
        if n < self.warmup_runs:
            head += f"; warm-up {n} of {self.warmup_runs}"
        return head

    def status_rows(self) -> list[dict[str, str]]:
        """One row per setting: label, value, effect, interval, word, history.
        word is "raising" / "falling" (moved at the last fit), "unclear"
        (interval spans zero), or "--" before the first estimate."""
        rows = []
        for k in self.knobs:
            row = {"label": k.label, "value": _fmt_knob(k.name, k.to_value(k.center)),
                   "effect": "", "interval": "", "word": "--", "history": ""}
            if k.effect is not None:
                e, a, b = k.effect
                row["effect"] = f"{e:+,.1f} per {k.step_label()}"
                row["interval"] = f"({a:+,.1f}, {b:+,.1f})"
                row["word"] = {1: "raising", -1: "falling", 0: "unclear"}[k.moved]
            moves = [f"{k.ups} up"] * bool(k.ups) + [f"{k.downs} down"] * bool(k.downs)
            row["history"] = ", ".join(moves)
            rows.append(row)
        return rows

    def status_lines(self, run: int = 0) -> list[str]:
        """Plain-text status (logs, tests)."""
        return [self.status_header()] + [
            "  " + "  ".join(v for v in (r["label"] + " " + r["value"], r["effect"],
                                          r["interval"], r["word"], r["history"]) if v)
            for r in self.status_rows()]

    def summary(self) -> dict:
        """For the settings file: where each knob ended up."""
        return {"aim": "lower 20th percentile" if self.tail else "mean score",
                "warmup_runs": self.warmup_runs,
                "runs_observed": self.runs_observed,
                "fit_window": self.window,
                **{k.name: k.to_value(k.center) for k in self.knobs}}


def quantile_fit(A: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    """Linear quantile regression: coefficients b minimising the pinball loss
    sum(q * pos + (1 - q) * neg) where y - A b = pos - neg. Solved as a linear
    programme with HiGHS; falls back to least squares if it fails."""
    from scipy.optimize import linprog
    n, p = A.shape
    c = np.concatenate([np.zeros(p), np.full(n, q), np.full(n, 1 - q)])
    A_eq = np.hstack([A, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    return res.x[:p]


def _fmt_knob(name: str, v: float) -> str:
    if name == "iterations":
        return f"{int(v):,}"
    if name in ("n3_probability", "flip_midpoint", "guide_fraction"):
        return f"{v * 100:.0f}%"
    return f"{v:.3f}"
