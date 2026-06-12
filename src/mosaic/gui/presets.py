"""
Score presets: named, complete snapshots of the score-row GUI state.

A preset captures:
  - per-score enable checkbox
  - per-score weight slider(s)
  - row sub-controls (targets, favor-D/R toggles, county bias)

A preset does NOT capture:
  - num_districts, iterations, tolerance, seed
  - annealing config
  - shared tuning params (win_prob, swing_sigma, eg_mode, pop_dev_harbor)

Built-ins are defined below; user saves persist to <mosaic_data_dir>/presets.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mosaic.paths import mosaic_data_dir

log = logging.getLogger("mosaic")

PRESETS_VERSION = 1


# ── Default state: every key referenced by gather/apply must appear here ───

DEFAULT_STATE: dict[str, Any] = {
    # Cut Edges
    "cut_enabled": True,
    "weight_cut_edges": 1,
    # County Splits
    "cs_enabled": False,
    "weight_county_excess": 1,
    "weight_county_unified": 1,
    "county_bias_enabled": False,
    "county_bias": 5,
    # Compactness (PP)
    "pp_enabled": False,
    "weight_polsby_popper": 1,
    # Reock
    "reock_enabled": False,
    "weight_reock": 1,
    # Holistic Compactness
    "hc_enabled": False,
    "weight_holistic_compactness": 1,
    # Population Deviation
    "popdev_enabled": False,
    "weight_pop_deviation": 1,
    # Mean-Median (unified Fair/D/R)
    "mm_enabled": False,
    "weight_mean_median": 1,
    "mm_dir": "Fair",
    # Efficiency Gap (unified Fair/D/R)
    "eg_enabled": False,
    "weight_efficiency_gap": 1,
    "eg_dir": "Fair",
    # Shared partisanship settings (popup)
    "mm_bound": 0.20,
    "eg_bound": 0.35,
    "partisan_quadratic_penalty": False,
    # Competitiveness
    "comp_enabled": False,
    "weight_competitiveness": 1,
    # Expected Dem Seats
    "seats_enabled": False,
    "weight_dem_seats": 1,
    "target_dem_seats": 7,
    # Chance of Majority
    "majority_enabled": False,
    "weight_majority": 1,
    "majority_dem_chk": True,
    "majority_rep_chk": False,
    # Supermajority / Hinge
    "hinge_enabled": False,
    "weight_hinge": 1,
    "hinge_dem_chk": True,
    "hinge_rep_chk": False,
    # Pro: Dem Seats Alone
    "seats_alone_enabled": False,
    "weight_seats_alone": 25,
    "seats_alone_dem_chk": True,
    "seats_alone_rep_chk": False,
    # Pro: Efficiency Gap (Alone)
    "eg_alone_enabled": False,
    "weight_eg_alone": 25,
    "eg_alone_dem_chk": True,
    "eg_alone_rep_chk": False,
    # Pro: Mean Median Diff. (Alone)
    "mm_alone_enabled": False,
    "weight_mm_alone": 25,
    "mm_alone_dem_chk": True,
    "mm_alone_rep_chk": False,
}


def _preset(**overrides) -> dict[str, Any]:
    return {**DEFAULT_STATE, **overrides}


# ── Built-in presets ───────────────────────────────────────────────────────

BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "Default": _preset(),
    "Strict County Respect": _preset(
        cs_enabled=True,
        weight_county_excess=20,
        weight_county_unified=20,
        county_bias_enabled=True,
        county_bias=10,
    ),
    "Compactness Focus": _preset(
        pp_enabled=True, weight_polsby_popper=10,
        reock_enabled=True, weight_reock=10,
    ),
    "DRA Mimic": _preset(
        hc_enabled=True, weight_holistic_compactness=20,
    ),
    "Partisan: Favor Dem": _preset(
        pp_enabled=True, weight_polsby_popper=2,
        seats_alone_enabled=True, weight_seats_alone=10,
        eg_alone_enabled=True, weight_eg_alone=10,
        mm_alone_enabled=True, weight_mm_alone=10,
        seats_alone_dem_chk=True, seats_alone_rep_chk=False,
        eg_alone_dem_chk=True, eg_alone_rep_chk=False,
        mm_alone_dem_chk=True, mm_alone_rep_chk=False,
    ),
    "Partisan: Favor Rep": _preset(
        pp_enabled=True, weight_polsby_popper=2,
        seats_alone_enabled=True, weight_seats_alone=10,
        eg_alone_enabled=True, weight_eg_alone=10,
        mm_alone_enabled=True, weight_mm_alone=10,
        seats_alone_dem_chk=False, seats_alone_rep_chk=True,
        eg_alone_dem_chk=False, eg_alone_rep_chk=True,
        mm_alone_dem_chk=False, mm_alone_rep_chk=True,
    ),
}


# ── Row metadata: enable_attr, label_attr, controls_tag, off_token ─────────
# Used by apply() to refresh visibility + label color without firing the
# row's _on_xxx_toggle callback (which has side effects we don't want during
# bulk apply — e.g. _on_cs_toggle force-enables county bias).

_ROW_META = [
    ("cut_enabled",        "_cut_lbl",        "cut_edge_controls",   "disabled"),
    ("cs_enabled",         "_cs_lbl",         "cs_controls",         "disabled"),
    ("pp_enabled",         "_pp_lbl",         "pp_controls",         "disabled"),
    ("reock_enabled",  "_reock_lbl",  "reock_controls",  "secondary"),
    ("hc_enabled",        "_hc_lbl",        "dra_controls",        "secondary"),
    ("seats_alone_enabled","_seats_alone_lbl","seats_alone_controls","secondary"),
    ("eg_alone_enabled",   "_eg_alone_lbl",   "eg_alone_controls",   "secondary"),
    ("mm_alone_enabled",   "_mm_alone_lbl",   "mm_alone_controls",   "secondary"),
    ("popdev_enabled",     "_popdev_lbl",     "popdev_controls",     "secondary"),
    ("mm_enabled",         "_mm_lbl",         "mm_controls",         "disabled"),
    ("eg_enabled",         "_eg_lbl",         "eg_controls",         "disabled"),
    ("comp_enabled",       "_comp_lbl",       "comp_controls",       "disabled"),
    ("seats_enabled",      "_seats_lbl",      "seats_controls",      "disabled"),
    ("majority_enabled",   "_majority_lbl",   "majority_controls",   "disabled"),
    ("hinge_enabled",      "_hinge_lbl",      "hinge_controls",      "disabled"),
]

# Widget attr name -> state key. Sliders/inputs and checkboxes share this map.
_WIDGET_KEYS: list[tuple[str, str]] = [
    ("_cut_enabled",          "cut_enabled"),
    ("_w_cut_edges",          "weight_cut_edges"),
    ("_cs_enabled",           "cs_enabled"),
    ("_w_county_excess",      "weight_county_excess"),
    ("_w_county_clean",       "weight_county_unified"),
    ("_county_bias_enabled",  "county_bias_enabled"),
    ("_county_bias",          "county_bias"),
    ("_pp_enabled",           "pp_enabled"),
    ("_w_polsby_popper",      "weight_polsby_popper"),
    ("_reock_enabled",    "reock_enabled"),
    ("_w_reock",          "weight_reock"),
    ("_hc_enabled",          "hc_enabled"),
    ("_w_holistic_compactness",                "weight_holistic_compactness"),
    ("_popdev_enabled",       "popdev_enabled"),
    ("_w_pop_deviation",      "weight_pop_deviation"),
    ("_mm_enabled",           "mm_enabled"),
    ("_w_mean_median",        "weight_mean_median"),
    ("_mm_dir",               "mm_dir"),
    ("_eg_enabled",           "eg_enabled"),
    ("_w_efficiency_gap",     "weight_efficiency_gap"),
    ("_eg_dir",               "eg_dir"),
    ("_mm_bound",             "mm_bound"),
    ("_eg_bound",             "eg_bound"),
    ("_partisan_quadratic_penalty", "partisan_quadratic_penalty"),
    ("_comp_enabled",         "comp_enabled"),
    ("_w_competitiveness",    "weight_competitiveness"),
    ("_seats_enabled",        "seats_enabled"),
    ("_w_dem_seats",          "weight_dem_seats"),
    ("_target_dem_seats",     "target_dem_seats"),
    ("_majority_enabled",     "majority_enabled"),
    ("_w_majority",           "weight_majority"),
    ("_majority_dem_chk",     "majority_dem_chk"),
    ("_majority_rep_chk",     "majority_rep_chk"),
    ("_hinge_enabled",        "hinge_enabled"),
    ("_w_hinge",              "weight_hinge"),
    ("_hinge_dem_chk",        "hinge_dem_chk"),
    ("_hinge_rep_chk",        "hinge_rep_chk"),
    ("_seats_alone_enabled",  "seats_alone_enabled"),
    ("_w_seats_alone",        "weight_seats_alone"),
    ("_seats_alone_dem_chk",  "seats_alone_dem_chk"),
    ("_seats_alone_rep_chk",  "seats_alone_rep_chk"),
    ("_eg_alone_enabled",     "eg_alone_enabled"),
    ("_w_eg_alone",           "weight_eg_alone"),
    ("_eg_alone_dem_chk",     "eg_alone_dem_chk"),
    ("_eg_alone_rep_chk",     "eg_alone_rep_chk"),
    ("_mm_alone_enabled",     "mm_alone_enabled"),
    ("_w_mm_alone",           "weight_mm_alone"),
    ("_mm_alone_dem_chk",     "mm_alone_dem_chk"),
    ("_mm_alone_rep_chk",     "mm_alone_rep_chk"),
]


def gather_score_state(app) -> dict[str, Any]:
    """Read every score-related widget into a state dict."""
    import dearpygui.dearpygui as dpg
    state: dict[str, Any] = {}
    for attr, key in _WIDGET_KEYS:
        widget = getattr(app, attr, None)
        if widget is None:
            state[key] = DEFAULT_STATE.get(key)
            continue
        state[key] = dpg.get_value(widget)
    return state


def apply_score_state(app, state: dict[str, Any]) -> None:
    """Push every value in `state` to the matching GUI widget and refresh
    each row's visibility + label color. Missing keys fall back to DEFAULT_STATE.
    """
    import dearpygui.dearpygui as dpg
    full = {**DEFAULT_STATE, **state}

    # 1) Write values to widgets
    for attr, key in _WIDGET_KEYS:
        widget = getattr(app, attr, None)
        if widget is None:
            continue
        try:
            dpg.set_value(widget, full[key])
        except Exception as e:
            log.warning(f"preset apply: could not set {attr}={full[key]!r}: {e}")

    # 2) Refresh row visibility + label theme. Mirrors what each _on_xxx_toggle
    #    does, but without the cs->bias auto-link in _on_cs_toggle (we want
    #    county_bias controlled by its own preset key).
    for enable_key, label_attr, controls_tag, off_token in _ROW_META:
        en = bool(full.get(enable_key, False))
        try:
            dpg.configure_item(controls_tag, show=en)
        except Exception:
            pass
        label_widget = getattr(app, label_attr, None)
        if label_widget is not None:
            token = "accent_green" if en else off_token
            try:
                app.theme.retoken(label_widget, token)
            except Exception:
                pass

    # County-bias sub-controls (live inside cs_controls but with their own
    # show/hide group tied to county_bias_enabled).
    try:
        dpg.configure_item(
            "county_bias_controls",
            show=bool(full.get("county_bias_enabled", False)),
        )
    except Exception:
        pass


# ── Disk persistence for user-saved presets ────────────────────────────────

def _presets_path() -> Path:
    return mosaic_data_dir() / "presets.json"


def load_user_presets() -> dict[str, dict[str, Any]]:
    p = _presets_path()
    if not p.is_file():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"Could not read {p}: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    raw = data.get("presets", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, st in raw.items():
        if isinstance(name, str) and isinstance(st, dict):
            out[name] = st
    return out


def save_user_presets(presets: dict[str, dict[str, Any]]) -> None:
    p = _presets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": PRESETS_VERSION, "presets": presets}
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)
    log.info(f"Saved {len(presets)} user preset(s) to {p}")
