# CODEMAP

A one-screen index of `src/mosaic/` so you (or an assistant) can open the file
that owns a concern directly, instead of scanning. Line counts are approximate.

## Top level

| File | ~LOC | Owns |
|------|-----:|------|
| `__init__.py` | 170 | Package init, `__version__`, preflight dep checks, logging, `main()` (the `mosaic` console script) |
| `__main__.py` | — | `python -m mosaic` entry (preferred launcher path) |
| `paths.py` | — | `output_dir()` and path helpers |
| `renumber.py` | 258 | District renumbering / stable label mapping |
| `crash.py` | 77 | `write_crash_log()` — durable crash dumps to `crashes/` |
| `engine.py` | 421 | `MosaicEngine`, headless chain driver — **private attachment, never ships** |

## graph/ — adjacency

| File | ~LOC | Owns |
|------|-----:|------|
| `adjacency.py` | 235 | Build precinct adjacency graph |
| `cache.py` | 104 | Adjacency cache |

## recom/ — the ReCom algorithm

| File | ~LOC | Owns |
|------|-----:|------|
| `tree.py` | 785 | Spanning-tree build + balanced-cut (Numba Kruskal) |
| `recombination.py` | 308 | ReCom step (merge two districts, re-split) |
| `flip.py` | 224 | Boundary flip moves |
| `swap.py` | 171 | Swap moves |
| `partition.py` | 164 | Partition state container |
| `annealing.py` | 161 | `AnnealingConfig`, schedule/temperature |

## scoring/ — plan metrics

| File | ~LOC | Owns |
|------|-----:|------|
| `score.py` | 431 | `ScoreConfig`, `PlanScore`, the score aggregator |
| `partisan.py` | 457 | Partisan metrics (EG, MM, bias, seats, responsiveness, gini) |
| `alignment.py` | 294 | Alignment-to-reference scoring |
| `reock.py` | 260 | Reock compactness |
| `holistic_splitting.py` | 208 | Holistic county-congruence |
| `precompute.py` | 192 | Per-precinct precomputation |
| `population.py` | 128 | Population deviation |
| `cache.py` | 100 | Score cache |
| `county_splits.py` / `polsby_popper.py` / `holistic_proportionality.py` / `holistic_competitiveness.py` | <100 each | Named single metrics |

## io/ — data in/out

| File | ~LOC | Owns |
|------|-----:|------|
| `inspect.py` | 270 | `ShapefileConfig`, `ShapefileInspection` — column detection |
| `hot_start.py` | 212 | Load an existing assignment as a warm start |
| `export.py` | 185 | Assignment / metric CSV export |
| `validate.py` | 155 | Shapefile validation |

## gui/ — Dear PyGui front end

| File | ~LOC | Owns |
|------|-----:|------|
| `runner.py` | 840 | `AlgorithmRunner` — the worker thread that drives the chain |
| `map_view.py` | 696 | `MapView` — the live district map render |
| `theme.py` | 599 | `ThemeManager` — light/dark themes |
| `shp_dialog.py` | 457 | Shapefile import dialog |
| `state.py` | 280 | `SharedState`, `AlgorithmStatus` — thread-shared status snapshot |
| `app/` | — | **The application class — see below** |

### gui/app/ — `MosaicApp`, split from the old 7k-line `app.py`

`MosaicApp` is one class at runtime, assembled in `core.py` from mixins that
each own one concern. Any method may call any other via `self` (shared state,
no cross-mixin imports). To find a method, pick the concern:

| File | ~LOC | Owns (method families) |
|------|-----:|------|
| `core.py` | 290 | Class assembly + `_internal` seam, `__init__`, `run`, dialog infra (`_dialog*`), module `main()` |
| `_common.py` | 514 | Shared imports, constants (`_PHASE_*`, layout dims), module helpers, `_SeriesBuffer`. `__all__` is the re-export surface |
| `setup_mixin.py` | 1318 | `setup()` — the whole two-column window build (one large method) |
| `popups_mixin.py` | 525 | `_build_*_popup` modal builders (settings, help, confirm) |
| `panels_mixin.py` | 679 | `_build_*_panel` score/metric side-panels + ref-line themes |
| `phase_mixin.py` | 311 | Phase plot (metric-vs-metric comet) build + `_on_phase_*` controls |
| `updates_mixin.py` | 1027 | Per-frame refresh: `_update_ui`, `_update_plots_and_panels`, tables, status labels |
| `toggles_mixin.py` | 277 | Series/panel visibility toggles, `_hint`, `_tooltip`, `_show_panel` |
| `map_mixin.py` | 198 | Map overlay toggles (`_on_*_overlay/_toggle`), `_rerender_map`, theme sync |
| `io_mixin.py` | 634 | Shapefile / hot-start / alignment loading, column pickers, seed/relight |
| `runner_mixin.py` | 575 | `_on_run/_pause/_reset/_revert`, renumber wiring |
| `export_mixin.py` | 818 | CSV/metric export, map image save, PDF/PNG workers, advanced-save |
| `menu_mixin.py` | 234 | File/session menu: recent files, new/close, update check, output dir |

**Private-only extension seam:** `core.py` does `try: from ._internal import
INTERNAL_MIXINS`. Any internal-only GUI feature becomes a mixin in a private
`gui/app/_internal/` package (excluded from `/ship-to-public`); the public
checkout simply lacks it and falls back to `INTERNAL_MIXINS = ()`.

## headless/ — batch CLI (private attachment, never ships)

| File | ~LOC | Owns |
|------|-----:|------|
| `cli.py` | 273 | `mosaic-headless` argument parsing / entry |
| `output.py` | 247 | Ensemble output writing |
| `load.py` | 187 | Config-driven data load |
| `config.py` | 183 | Headless run config |
| `run.py` | 158 | Batch run loop (uses `engine.MosaicEngine`) |
