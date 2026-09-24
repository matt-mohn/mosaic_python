"""
Score presets — which scores are on, their weights, and their targets, saved
as a small TOML file that travels across maps.

A preset holds scoring only. The map (shapefile, columns, reference plan),
plan size (districts, tolerance) and chain settings (iterations, seed, moves,
annealing) stay out, so one preset can be tried on any map. Hinge is left out
too: its threshold is a seat count tied to one map. County-edge bias is the one
sampling setting kept, since it pairs with County Congruence.

Sections are keyed by the ScoreConfig field stem. Each score stores `enabled`
separately from `weight`, so a disabled score keeps its weight on a round-trip.
Values use ScoreConfig units (fractions, "fair"/"favor_dem"/"favor_rep").

A preset is the scoring slice of a fuller run config. The `mosaic_preset` key
marks the file type, so a preset is never mistaken for a run config (or the
reverse); a run config can point at one with `preset = "<file>"`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the TOML parser backport.
    import tomli as tomllib

from mosaic.scoring.score import ScoreConfig

PRESET_VERSION = 1
PRESET_MARKER = "mosaic_preset"   # top-level key holding PRESET_VERSION

_DIRECTIONS = ("fair", "favor_dem", "favor_rep")
_PARTIES = ("dem", "rep")
_FOCUS = ("none", "rep", "dem")

# section -> key -> (default, allowed values or None). Defaults match the GUI's
# widget defaults, so a key missing from a file loads as a fresh app would.
SCHEMA: dict[str, dict[str, tuple[Any, tuple | None]]] = {
    "cut_edges": {"enabled": (False, None), "weight": (1, None)},
    "county_splits": {"enabled": (False, None), "weight_excess": (1, None),
                      "weight_unified": (1, None)},
    "holistic_splitting": {"enabled": (False, None), "weight": (25, None),
                           "unclipped": (True, None)},
    "county_bias": {"enabled": (False, None), "multiplier": (5, None)},
    "holistic_compactness": {"enabled": (True, None), "weight": (50, None),
                             "unclipped": (True, None)},
    "polsby_popper": {"enabled": (False, None), "weight": (25, None)},
    "reock": {"enabled": (False, None), "weight": (25, None)},
    "pop_deviation": {"enabled": (False, None), "weight": (1.0, None),
                      "safe_harbor": (0.0, None)},
    "alignment": {"enabled": (False, None), "weight": (25, None),
                  "party_focus": ("none", _FOCUS),
                  "restrict_to_party": (False, None),
                  "win_threshold": (0.535, None)},
    "mean_median": {"enabled": (False, None), "weight": (1, None),
                    "mode": ("fair", _DIRECTIONS), "bound": (0.20, None)},
    "efficiency_gap": {"enabled": (False, None), "weight": (1, None),
                       "mode": ("fair", _DIRECTIONS), "bound": (0.35, None),
                       "robust": (True, None)},
    "partisan_bias": {"enabled": (False, None), "weight": (1, None),
                      "mode": ("fair", _DIRECTIONS), "bound": (0.25, None)},
    "partisan_gini": {"enabled": (False, None), "weight": (25, None)},
    "dem_seats": {"enabled": (False, None), "weight": (25, None),
                  "favor": ("dem", _PARTIES)},
    "holistic_proportionality": {"enabled": (False, None), "weight": (25, None),
                                 "unclipped": (True, None)},
    "holistic_competitiveness": {"enabled": (False, None), "weight": (25, None),
                                 "unclipped": (True, None)},
    "majority_chance": {"enabled": (False, None), "weight": (1, None),
                        "party": ("dem", _PARTIES)},
    "representation": {"enabled": (False, None), "weight": (25, None),
                       "unclipped": (True, None)},
    "minority_cohesion": {"enabled": (False, None), "weight": (10, None)},
    "community_congruence": {"enabled": (False, None), "weight": (10, None)},
    "partisan": {"win_prob_at_55": (0.9, None), "swing_sigma": (0.03, None),
                 "quadratic_penalty": (False, None)},
    "opportunity": {"midpoint": (0.44, None), "steepness": (0.05, None),
                    "solid": (0.55, None), "smart_targets": (True, None)},
}


def default_preset() -> dict[str, dict[str, Any]]:
    return {sec: {k: d for k, (d, _) in keys.items()}
            for sec, keys in SCHEMA.items()}


# Portable presets use the same ranges and integer weights as the GUI.
NUMERIC_BOUNDS = {
    **{(s, k): (0, 100) for s, keys in SCHEMA.items() for k in keys
       if k.startswith("weight")},
    ("county_bias", "multiplier"): (1, 20),
    ("pop_deviation", "safe_harbor"): (0.0, 0.05),
    ("alignment", "win_threshold"): (0.50, 0.70),
    ("mean_median", "bound"): (0.05, 0.30),
    ("efficiency_gap", "bound"): (0.10, 0.50),
    ("partisan_bias", "bound"): (0.05, 0.50),
    ("partisan", "win_prob_at_55"): (0.51, 0.999),
    ("partisan", "swing_sigma"): (0.005, 0.10),
    ("opportunity", "midpoint"): (0.30, 0.65),
    ("opportunity", "steepness"): (0.01, 0.15),
    ("opportunity", "solid"): (0.45, 0.65),
}


def _coerce(value: Any, default: Any, allowed: tuple | None) -> Any:
    """Validate scalar values and normalize their numeric types."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError("expected true/false")
        return value
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a number")
        if not isinstance(value, int) and not math.isfinite(value):
            raise ValueError("must be finite")
        if isinstance(default, int):
            if value != int(value):
                raise ValueError("must be a whole number")
            return int(value)
        try:
            value = float(value)
        except OverflowError:
            raise ValueError("number is too large") from None
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value
    if not isinstance(value, str):
        raise ValueError("expected text")
    if allowed is not None and value not in allowed:
        raise ValueError(f"expected one of {', '.join(allowed)}")
    return value


def parse_preset(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate raw TOML data against SCHEMA.

    Missing keys take defaults and unknown entries produce warnings. Invalid
    known values reject the entire preset before it can change any settings.
    """
    if PRESET_MARKER not in data:
        raise ValueError("not a Mosaic preset file (no mosaic_preset key)")
    out = default_preset()
    warnings: list[str] = []
    errors: list[str] = []
    if data[PRESET_MARKER] != PRESET_VERSION:
        warnings.append(f"written by preset version {data[PRESET_MARKER]}")
    for sec, body in data.items():
        if sec == PRESET_MARKER:
            continue
        if sec not in SCHEMA:
            warnings.append(f"unknown section [{sec}]")
            continue
        if not isinstance(body, dict):
            errors.append(f"{sec}: expected a section")
            continue
        for key, value in body.items():
            if key not in SCHEMA[sec]:
                warnings.append(f"unknown key {sec}.{key}")
                continue
            default, allowed = SCHEMA[sec][key]
            try:
                value = _coerce(value, default, allowed)
                if (sec, key) in NUMERIC_BOUNDS:
                    lo, hi = NUMERIC_BOUNDS[sec, key]
                    # Native float sliders round decimal endpoints to float32.
                    # Accept only that tiny round-trip error, not out-of-range
                    # settings that would otherwise differ in headless runs.
                    if not lo <= value <= hi:
                        edge = lo if value < lo else hi
                        if isinstance(default, float) and math.isclose(
                            value, edge, rel_tol=1e-7, abs_tol=1e-10
                        ):
                            value = float(edge)
                        else:
                            raise ValueError(f"must be between {lo:g} and {hi:g}")
                out[sec][key] = value
            except ValueError as e:
                errors.append(f"{sec}.{key}: {e}")
    if errors:
        raise ValueError("; ".join(errors))
    return out, warnings


def validate_preset(preset: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate an in-memory preset using the same rules as file loading."""
    return parse_preset({**preset, PRESET_MARKER: PRESET_VERSION})[0]


def read_preset(path: Path | str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    with Path(path).open("rb") as f:
        return parse_preset(tomllib.load(f))


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(float(f"{v:.7g}"))   # GUI sliders are float32: 0.2 not 0.20000000298
    if isinstance(v, int):
        return repr(v)
    return json.dumps(v)   # JSON string escaping is valid TOML basic-string escaping


def dumps_preset(preset: dict[str, dict[str, Any]]) -> str:
    preset = validate_preset(preset)
    lines = [f"{PRESET_MARKER} = {PRESET_VERSION}"]
    for sec, keys in SCHEMA.items():
        lines += ["", f"[{sec}]"]
        for key in keys:
            lines.append(f"{key} = {_toml_value(preset[sec][key])}")
    return "\n".join(lines) + "\n"


def write_preset(path: Path | str, preset: dict[str, dict[str, Any]]) -> None:
    Path(path).write_text(dumps_preset(preset), encoding="utf-8")


def run_settings(preset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Preset values that live on RunConfig rather than ScoreConfig."""
    cb = validate_preset(preset)["county_bias"]
    return {"county_bias_enabled": cb["enabled"],
            "county_bias": float(cb["multiplier"])}


def to_score_config(preset: dict[str, dict[str, Any]], **overrides: Any) -> ScoreConfig:
    """Build a ScoreConfig the way the GUI runner does: disabled -> weight 0."""
    p = validate_preset(preset)

    def w(sec: str, key: str = "weight") -> float:
        return float(p[sec][key]) if p[sec]["enabled"] else 0.0

    maj = w("majority_chance")
    kwargs: dict[str, Any] = dict(
        weight_cut_edges=w("cut_edges"),
        weight_county_excess=w("county_splits", "weight_excess"),
        weight_county_unified=w("county_splits", "weight_unified"),
        weight_holistic_splitting=w("holistic_splitting"),
        holistic_splitting_unclipped=p["holistic_splitting"]["unclipped"],
        weight_polsby_popper=w("polsby_popper"),
        weight_reock=w("reock"),
        weight_holistic_compactness=w("holistic_compactness"),
        compactness_unclipped=p["holistic_compactness"]["unclipped"],
        weight_pop_deviation=w("pop_deviation"),
        pop_deviation_safe_harbor=p["pop_deviation"]["safe_harbor"],
        weight_alignment=w("alignment"),
        alignment_party_focus=p["alignment"]["party_focus"],
        alignment_restrict_to_party=p["alignment"]["restrict_to_party"],
        alignment_win_threshold=p["alignment"]["win_threshold"],
        weight_mean_median=w("mean_median"),
        mm_mode=p["mean_median"]["mode"],
        mm_bound=p["mean_median"]["bound"],
        weight_efficiency_gap=w("efficiency_gap"),
        eg_mode=p["efficiency_gap"]["mode"],
        eg_bound=p["efficiency_gap"]["bound"],
        use_robust_eg=p["efficiency_gap"]["robust"],
        partisan_quadratic_penalty=p["partisan"]["quadratic_penalty"],
        weight_partisan_bias=w("partisan_bias"),
        pbias_mode=p["partisan_bias"]["mode"],
        pbias_bound=p["partisan_bias"]["bound"],
        weight_partisan_gini=w("partisan_gini"),
        weight_dem_seats=w("dem_seats"),
        dem_seats_favor_dem=p["dem_seats"]["favor"] == "dem",
        weight_holistic_proportionality=w("holistic_proportionality"),
        proportionality_unclipped=p["holistic_proportionality"]["unclipped"],
        weight_holistic_competitiveness=w("holistic_competitiveness"),
        competitiveness_unclipped=p["holistic_competitiveness"]["unclipped"],
        weight_majority_chance_dem=maj if p["majority_chance"]["party"] == "dem" else 0.0,
        weight_majority_chance_rep=maj if p["majority_chance"]["party"] == "rep" else 0.0,
        election_win_prob_at_55=p["partisan"]["win_prob_at_55"],
        election_swing_sigma=p["partisan"]["swing_sigma"],
        weight_representation=w("representation"),
        representation_unclipped=p["representation"]["unclipped"],
        opportunity_midpoint=p["opportunity"]["midpoint"],
        opportunity_steepness=p["opportunity"]["steepness"],
        opportunity_solid=p["opportunity"]["solid"],
        opportunity_smart_targets=p["opportunity"]["smart_targets"],
        weight_minority_cohesion=w("minority_cohesion"),
        weight_community_congruence=w("community_congruence"),
    )
    kwargs.update(overrides)
    return ScoreConfig(**kwargs)
