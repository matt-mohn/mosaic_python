"""
Ensemble — repeat full runs (partition to finish) until an end condition is
met, writing each run's final map and map-wide metrics as it completes.

Output lives in its own ``output/ensemble_<timestamp>_<id>/`` folder:

* ``ensemble_assignments.csv`` — one row per precinct, one column per run
  (``run_1`` ...), joinable in GIS like Save Assignments. Written when the
  ensemble ends.
* ``ensemble_summaries.csv`` — one row per run of map-wide metrics.
* ``ensemble_settings.toml`` — the scoring preset and run settings used.
* ``ensemble_results.sqlite3`` and ``ensemble_assignments.bin`` — disk-backed
  results used by the in-app views, with one assignment loaded at a time.

While running, each run is appended to ``ensemble_assignments_in_progress.csv``
(one row per run), since a per-precinct file can't grow by appending. It is
removed once finalization succeeds. A crash or write failure leaves the working
file available for recovery.

The loop drives the GUI runner's own ``run_algorithm`` so an ensemble run is
exactly a normal run. It holds no GUI code.
"""

from __future__ import annotations

import csv
import dataclasses
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from mosaic.ensemble_store import EnsembleStore, RunSequence
from mosaic.presets import _toml_value
from mosaic.scoring import score_plan
from mosaic.scoring.score import PlanScore, ScoreConfig

ENSEMBLE_VERSION = 1


@dataclass
class EnsembleSpec:
    mode: str = "count"               # "count" | "duration" | "unlimited" (until stopped)
    count: int = 10
    duration_s: float = 300.0
    base_seed: Optional[int] = None   # run i uses base_seed + i - 1; None = random
    keep_best: bool = False           # keep each run's best-scoring map, not its final one
    targeting: bool = False           # tune chain settings from the runs (mosaic.targeting)


@dataclass
class EnsembleProgress:
    """Shared between the ensemble thread and the GUI; guard with ``lock``."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    runs_done: int = 0
    done_elapsed: float = 0.0         # seconds from start to the last completed run
    current_run: int = 0
    started: float = 0.0
    stop_requested: bool = False
    finished: bool = False
    error: str = ""
    targeter: Any = None              # mosaic.targeting.Targeter when targeting
    target_header: str = ""
    target_rows: list[dict[str, str]] = field(default_factory=list)


# ── Map-wide metrics ─────────────────────────────────────────────────────────


def _measure_config(cfg: ScoreConfig, have: dict[str, bool]) -> ScoreConfig:
    """Every available metric on, in one fixed form so summaries don't depend
    on what the run optimized. Holistic ratings use the unclipped default: the
    clipped scorecard saturates (Reock above 0.50 pins its half of Compactness),
    which would hide variation the optimizer is still acting on. EG is static."""
    on = {f.name: 1.0 for f in dataclasses.fields(ScoreConfig)
          if f.name.startswith("weight_")}
    if not have.get("reference"):
        on["weight_alignment"] = 0.0
    if not have.get("hinge"):          # its threshold is the user's; no guessing one
        on["weight_hinge"] = 0.0
    return dataclasses.replace(
        cfg, **on,
        holistic_splitting_unclipped=True, compactness_unclipped=True,
        proportionality_unclipped=True, competitiveness_unclipped=True,
        representation_unclipped=True, use_robust_eg=False,
    )


def summary_columns(have: dict[str, bool], tuned: tuple = ()) -> list[str]:
    """Columns for ensemble_summaries.csv. `tuned`: targeting's knob names,
    recorded per run with the run's phase so tuning runs can be split out."""
    cols = ["run_id", "seed", "iterations", "seconds", "score",
            "cut_edges", "pop_dev_max_pct", "pop_dev_mean_pct"]
    if have["compactness"]:
        cols += ["compactness", "polsby_popper", "reock"]
    if have["county"]:
        cols += ["county_congruence_penalty", "county_excess_splits",
                 "county_unified_districts"]
    if have["elections"]:
        cols += ["mean_median", "efficiency_gap", "partisan_bias",
                 "partisan_gini_penalty", "expected_dem_seats", "proportionality",
                 "inversion_chance", "competitiveness", "dem_majority_chance"]
    if have.get("hinge"):
        cols += ["hinge_chance"]
    if have["demographics"]:
        cols += ["electoral_opportunity", "opportunity_black",
                 "opportunity_latino", "opportunity_asian",
                 "neighborhood_severance_penalty", "community_dispersion_penalty"]
    if have["reference"]:
        cols += ["alignment_mean_retention", "alignment_min_retention"]
    if tuned:
        cols += ["phase", *[k for k in tuned if k != "iterations"]]
    return cols


def summary_values(ps: PlanScore) -> dict[str, float]:
    """Map-wide values from a PlanScore scored with _measure_config."""
    return {
        "cut_edges": ps.cut_edges,
        "pop_dev_max_pct": ps.pop_dev_max,
        "pop_dev_mean_pct": ps.pop_dev_mean,
        "compactness": 100.0 - ps.holistic_compactness,
        "polsby_popper": 1.0 - ps.polsby_popper / 100.0,
        "reock": 1.0 - ps.reock / 100.0,
        "county_congruence_penalty": ps.holistic_splitting,
        "county_excess_splits": ps.county_excess_splits,
        "county_unified_districts": ps.county_unified_districts,
        "mean_median": ps.mean_median,
        "efficiency_gap": ps.efficiency_gap,
        "partisan_bias": ps.partisan_bias,
        "partisan_gini_penalty": ps.partisan_gini,
        "expected_dem_seats": ps.dem_seats,
        "proportionality": 100.0 - ps.holistic_proportionality,
        "inversion_chance": ps.inversion_chance,
        "competitiveness": 100.0 - ps.holistic_competitiveness,
        "dem_majority_chance": ps.majority_chance_dem,
        "hinge_chance": ps.hinge_chance,
        "electoral_opportunity": ps.representation_rating,
        "opportunity_black": ps.opportunity_black,
        "opportunity_latino": ps.opportunity_latino,
        "opportunity_asian": ps.opportunity_asian,
        "neighborhood_severance_penalty": ps.minority_cohesion,
        "community_dispersion_penalty": ps.community_congruence,
        "alignment_mean_retention": ps.alignment_mean_ret,
        "alignment_min_retention": ps.alignment_min_ret,
    }


def data_available(runner, score_config: Optional[ScoreConfig] = None) -> dict[str, bool]:
    """Which metric groups this map can report. Hinge also needs the run to use
    it: its threshold is a user choice, so there is none to measure otherwise."""
    elections = bool(runner.election_arrays)
    return {
        "compactness": runner.pp_data is not None and runner.reock_data is not None,
        "county": runner.county_array is not None,
        "elections": elections,
        "demographics": runner.vap_data is not None,
        "reference": runner.alignment_data is not None,
        "hinge": elections and bool(score_config and score_config.weight_hinge),
    }


# ── Writer ───────────────────────────────────────────────────────────────────


def dumps_toml(data: dict[str, Any]) -> str:
    """Scalars at top level, then tables (and one level of sub-tables).
    None values are dropped (TOML has no null)."""
    def _clean(d):
        return {k: (_clean(v) if isinstance(v, dict) else v)
                for k, v in d.items() if v is not None}
    data = _clean(data)
    lines = [f"{k} = {_toml_value(v)}" for k, v in data.items()
             if not isinstance(v, dict)]
    for name, table in data.items():
        if not isinstance(table, dict):
            continue
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if scalars:
            lines += ["", f"[{name}]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in scalars.items()]
        for sub, body in table.items():
            if isinstance(body, dict):
                lines += ["", f"[{name}.{sub}]"]
                lines += [f"{k} = {_toml_value(v)}" for k, v in body.items()]
    return "\n".join(lines) + "\n"


class EnsembleWriter:
    """Owns one ensemble folder and publishes completed, disk-backed runs."""

    def __init__(self, folder: Path, precinct_ids: list, id_col: str,
                 columns: list[str]):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.precinct_ids = list(precinct_ids)
        self.id_col = id_col
        self.columns = columns
        self.store = EnsembleStore(self.folder / "ensemble_results.sqlite3", columns,
                                   len(self.precinct_ids))
        self.assignments = RunSequence(self.store, assignments=True)
        self.rows = RunSequence(self.store)
        self._write_failed = False
        self.settings: Optional[dict[str, Any]] = None
        with self.working_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["run_id", *self.precinct_ids])
        with self.summary_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)

    @property
    def working_path(self) -> Path:
        """Row-per-run file appended during the ensemble; removed by finalize()."""
        return self.folder / "ensemble_assignments_in_progress.csv"

    @property
    def summary_path(self) -> Path:
        return self.folder / "ensemble_summaries.csv"

    @property
    def assign_path(self) -> Path:
        return self.folder / "ensemble_assignments.csv"

    def write_settings(self, settings: dict[str, Any]) -> None:
        body = {"mosaic_ensemble": ENSEMBLE_VERSION, **settings}
        path = self.folder / "ensemble_settings.toml"
        staging = path.with_suffix(".toml.tmp")
        staging.write_text(dumps_toml(body), encoding="utf-8")
        staging.replace(path)
        self.settings = settings

    def add_run(self, run_id: str, assignment: np.ndarray,
                row: dict[str, Any]) -> None:
        districts = assignment.astype(np.int32) + 1   # 1-indexed, as Save Assignments
        try:
            with self.working_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([run_id, *districts.tolist()])
            with self.summary_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([_fmt(row.get(c, "")) for c in self.columns])
            # The store publishes its count only after the disk transaction commits.
            self.store.add(run_id, districts, row)
        except Exception:
            self._write_failed = True
            raise

    def finalize(self) -> Optional[Path]:
        """Write the per-precinct assignments file, then drop the working file.
        The working file is removed only after the final one is on disk."""
        path = None
        if self.assignments:
            staging = self.assign_path.with_suffix(".csv.tmp")
            self.store.export_assignments(staging, self.id_col, self.precinct_ids)
            staging.replace(self.assign_path)
            path = self.assign_path
        if not self._write_failed:
            self.working_path.unlink(missing_ok=True)
        return path


def _fmt(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return v


# ── Loop ─────────────────────────────────────────────────────────────────────


def _apply_tuned(state, tuned: dict, base_ann) -> None:
    """Push one run's tuned settings into shared state before it starts."""
    upd = {"max_iterations": int(tuned["iterations"]),
           "n3_probability": tuned["n3_probability"]}
    if "flip_midpoint" in tuned:
        upd["flip_midpoint"] = tuned["flip_midpoint"]
    ann = {}
    if "temp_factor" in tuned:
        ann["initial_temp_factor"] = tuned["temp_factor"]
    if "guide_fraction" in tuned:
        ann["guide_fraction"] = tuned["guide_fraction"]
    if ann:
        upd["annealing_config"] = dataclasses.replace(base_ann, **ann)
    state.update(**upd)


def run_ensemble(runner, state, spec: EnsembleSpec, writer: EnsembleWriter,
                 progress: EnsembleProgress) -> None:
    """Run until the end condition, a stop request, or an error. Run settings
    must already be in ``state`` (the GUI's run-settings capture). ``spec`` is
    re-read before each run, so the GUI can extend it mid-ensemble."""
    num_districts, tolerance, max_iterations, score_config = state.get(
        "num_districts", "pop_tolerance", "max_iterations", "score_config")
    have = data_available(runner, score_config)
    measure_cfg = _measure_config(score_config, have)
    seed_rng = np.random.default_rng()
    targeter = progress.targeter
    base_ann = state.get("annealing_config")[0]
    with progress.lock:
        progress.started = time.time()
        if targeter is not None:
            progress.target_header = targeter.status_header()
            progress.target_rows = targeter.status_rows()

    def stopped():
        with progress.lock:
            return progress.stop_requested
    i = 0
    try:
        while True:
            with progress.lock:
                if progress.stop_requested:
                    break
                if spec.mode == "count" and i >= spec.count:
                    break
                if (spec.mode == "duration"
                        and time.time() - progress.started >= spec.duration_s):
                    break
                # Stop takes this lock too, so reset cannot erase a stop request.
                state.reset_run()
            i += 1
            seed = (spec.base_seed + i - 1 if spec.base_seed is not None
                    else int(seed_rng.integers(1, 2**31 - 1)))
            with progress.lock:
                progress.current_run = i
            state.update(seed=seed)
            tuned = targeter.propose(i) if targeter is not None else {}
            if tuned:
                _apply_tuned(state, tuned, base_ann)
            (run_iters,) = state.get("max_iterations")
            t0 = time.time()
            runner.run_algorithm()
            # Compared by name: importing mosaic.gui would pull in the whole app.
            status, err = state.get("status", "error_message")
            if status.name == "ERROR":
                with progress.lock:
                    progress.error = err or "run failed"
                break
            if status.name != "COMPLETED":
                break   # stopped mid-run; the partial run is dropped
            with state._lock:
                if spec.keep_best and state.best_assignment is not None:
                    final = state.best_assignment.copy()
                    total = state.best_score
                else:
                    final = state.current_assignment.copy()
                    total = state.current_score
            ps = score_plan(runner.graph_ctx.compute_cut_edges(final), measure_cfg,
                            assignment=final,
                            **runner.score_kwargs(num_districts, tolerance))
            row = {"run_id": f"run_{i}", "seed": seed,
                   "iterations": run_iters,
                   "seconds": time.time() - t0, "score": total,
                   **summary_values(ps)}
            if targeter is not None:
                row.update(tuned, phase=targeter.phase(i))
            writer.add_run(f"run_{i}", final, row)
            with progress.lock:
                progress.runs_done = i
                progress.done_elapsed = time.time() - progress.started
            if targeter is not None:
                # Fitting may be expensive; the GUI reads published snapshots.
                targeter.observe(i, tuned, total, cancelled=stopped)
                with progress.lock:
                    progress.target_header = targeter.status_header()
                    progress.target_rows = targeter.status_rows()
    except Exception as e:   # surfaced in the ensemble window, not swallowed
        from mosaic.crash import write_crash_log
        path = write_crash_log(e, context={"phase": "ensemble", "run": i})
        with progress.lock:
            progress.error = f"{type(e).__name__}: {e}  (log: {path})"
    finally:
        if writer.settings is not None:
            try:
                settings = dict(writer.settings)
                settings["ensemble"] = dict(settings["ensemble"])
                with progress.lock:
                    settings["ensemble"].update(
                        count=spec.count if spec.mode == "count" else None,
                        duration_minutes=spec.duration_s / 60 if spec.mode == "duration" else None)
                if targeter is not None:
                    settings["targeting"] = targeter.summary()
                writer.write_settings(settings)
            except Exception as e:
                with progress.lock:
                    progress.error = progress.error or f"Settings not saved: {e}"
        try:
            writer.finalize()
        except Exception as e:   # the working file survives with every run in it
            with progress.lock:
                progress.error = progress.error or (
                    f"Assignments file not written: {e}; runs are in "
                    f"{writer.working_path.name}")
        with progress.lock:
            progress.finished = True
