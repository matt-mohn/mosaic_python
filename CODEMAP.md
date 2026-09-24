# CODEMAP

Module ownership and invariants for `src/mosaic/`.

## Top level

| File | Owns |
|------|------|
| `__init__.py` | Package init, `__version__`, preflight dep checks, logging, `main()` (the `mosaic` console script) |
| `__main__.py` | `python -m mosaic` entry (preferred launcher path) |
| `paths.py` | `output_dir()` and path helpers |
| `renumber.py` | District renumbering / stable label mapping |
| `crash.py` | `write_crash_log()` — durable crash dumps to `crashes/` |
| `presets.py` | Score presets: TOML schema, read/write, `to_score_config()` / `run_settings()` for headless |
| `ensemble.py` | Ensemble loop (repeat full runs), per-run writer, map-wide summary metrics. No GUI imports |
| `ensemble_store.py` | SQLite summaries, disk-backed assignment records, streamed export and result-view queries |
| `targeting.py` | Beta Targeting: randomized per-run settings, continuous OLS/quantile fits over a bounded recent window, bootstrap intervals and adaptive centers |
| `engine.py` | `MosaicEngine`, headless chain driver — **private attachment, never ships** |

## graph/ — adjacency

| File | Owns |
|------|------|
| `adjacency.py` | Build precinct adjacency graph |
| `cache.py` | Adjacency cache |

## recom/ — the ReCom algorithm

| File | Owns |
|------|------|
| `tree.py` | Spanning-tree build + balanced-cut (Numba Kruskal) |
| `recombination.py` | ReCom step (merge two districts, re-split) |
| `flip.py` | Boundary flip moves |
| `swap.py` | Swap moves |
| `partition.py` | Population-balanced initial partition construction and local subgraph preparation |
| `annealing.py` | `AnnealingConfig`, schedule/temperature |

## scoring/ — plan metrics

| File | Owns |
|------|------|
| `score.py` | `ScoreConfig`, `PlanScore`, the score aggregator |
| `partisan.py` | Partisan metrics (EG, MM, bias, seats, gini) |
| `opportunity.py` | Shared opportunity-to-elect engine — per-district group curves, proportional benchmark `T`, drawability gate, achievable ceiling `f(state)`, smart targets. Feeds every demographic score |
| `community_congruence.py` | Community Dispersion — layered per-group cores, `N_eff / m` |
| `alignment.py` | Alignment-to-reference scoring |
| `minority_cohesion.py` | Neighborhood Severance — minority-weighted cut-edge enrichment ratio |
| `reock.py` | Reock compactness |
| `holistic_splitting.py` | Holistic county-congruence |
| `precompute.py` | Per-precinct precomputation |
| `representation.py` | Electoral Opportunity — top-`round(T)` credit vs `f(state)` |
| `population.py` | Population deviation |
| `cache.py` | Score cache |
| `county_splits.py` / `polsby_popper.py` / `holistic_proportionality.py` / `holistic_competitiveness.py` | Named single metrics |

The three demographic scores are independent axes, not proxies for one another:
Electoral Opportunity asks whether a group can elect, Neighborhood Severance
where district lines fall through a community, Community Dispersion how many
pieces a community lands in. Each returns its BEST value when it has nothing to
measure, so the GUI gates them on `state.race_score_applicable` rather than on
the score itself.

## io/ — data in/out

| File | Owns |
|------|------|
| `inspect.py` | `ShapefileConfig`, `ShapefileInspection` — column detection (population, id, county, election pairs, demographic groups) |
| `hot_start.py` | Load an existing assignment as a warm start |
| `export.py` | Assignment / metric CSV export |
| `validate.py` | Shapefile validation — geometry, columns, demographics, connectivity |

## gui/ — Dear PyGui front end

| File | Owns |
|------|------|
| `runner.py` | `AlgorithmRunner` — the worker thread that drives the chain |
| `map_view.py` | `MapView` — the live district map render |
| `theme.py` | `ThemeManager` — light/dark themes |
| `file_dialog.py` | Windows PowerShell/STA file picker, encoded arguments, cancellation and error handling |
| `shp_dialog.py` | Shapefile import dialog |
| `state.py` | `SharedState`, `AlgorithmStatus` — thread-shared status snapshot |
| `app/` | **The application class — see below** |

### gui/app/ — `MosaicApp` assembly

`MosaicApp` is one class at runtime, assembled in `core.py` from mixins that
each own one concern. Any method may call any other via `self` (shared state,
no cross-mixin imports). To find a method, pick the concern:

| File | Owns (method families) |
|------|------|
| `core.py` | Class assembly + `_internal` seam, `__init__`, `run`, dialog infra (`_dialog*`), module `main()` |
| `_common.py` | Shared imports, constants (`_PHASE_*`, layout dims), module helpers, `_SeriesBuffer`. `__all__` is the re-export surface |
| `setup_mixin.py` | `setup()` — the whole two-column window build (one large method) |
| `popups_mixin.py` | `_build_*_popup` modal builders (settings, help, confirm) |
| `panels_mixin.py` | `_build_*_panel` score/metric side-panels + ref-line themes |
| `phase_mixin.py` | Phase plot (metric-vs-metric comet) build + `_on_phase_*` controls |
| `updates_mixin.py` | Per-frame refresh: `_update_ui`, `_update_plots_and_panels`, tables, status labels |
| `toggles_mixin.py` | Series/panel visibility toggles, `_hint`, `_tooltip`, `_show_panel` |
| `map_mixin.py` | Map overlay toggles (`_on_*_overlay/_toggle`), `_rerender_map`, theme sync |
| `io_mixin.py` | Shapefile / hot-start / alignment loading, column pickers, seed/relight |
| `runner_mixin.py` | `_on_run/_pause/_reset/_revert`, renumber wiring |
| `export_mixin.py` | CSV/metric export, map image save, PDF/PNG workers, advanced-save |
| `menu_mixin.py` | File/session menu: recent files, new/close, update check, output dir; score presets (Save/Apply/Recent, Clear All Scores) |
| `ensemble_mixin.py` | Advanced > Ensemble window: start / force stop / extend, per-frame progress (replaces `_update_ui` while active) |
| `ensemble_views_mixin.py` | Ensemble pop-out views (Histograms, Scatterplot) and the main-window freeze during a run |
| `ensemble_roster_mixin.py` | Ensemble Roster: up to 5 metric criteria on drawn two-handle range tracks, filtered + sortable run table (click a run to map it), per-metric display formats |
| `ensemble_map_mixin.py` | Ensemble Map: static minimap of one run on its own `MapView` + texture (no zoom/overlays); shape follows the state |

**Private-only extension seam:** `core.py` does `try: from ._internal import
INTERNAL_MIXINS`. Any internal-only GUI feature becomes a mixin in a private
`gui/app/_internal/` package (excluded from `/ship-to-public`); the public
checkout simply lacks it and falls back to `INTERNAL_MIXINS = ()`.

## headless/ — batch CLI (private attachment, never ships)

| File | Owns |
|------|------|
| `cli.py` | `mosaic-headless` argument parsing / entry |
| `output.py` | Ensemble output writing |
| `load.py` | Config-driven data load |
| `config.py` | Headless run config |
| `run.py` | Batch run loop (uses `engine.MosaicEngine`) |

## Ownership and performance invariants

- Dear PyGui widget and texture mutations belong to the frame thread. Session
  requests wait for algorithm, data and map workers to exit before replacement.
  Callback signatures must remain explicit: Dear PyGui inspects argument counts.
- Map geometry, rasters, border masks and asynchronous results belong to one
  source and generation. A failed resize leaves the previous texture and grid
  together; successful resizing preserves geographic center and scale.
- The client width is at least 1,300 pixels. The upper map/control span retains
  its height; the score area absorbs vertical resizing.
- Adjacency caches fingerprint source files and retain shared boundary lengths.
  Virtual bridges are reconstructed from the selected population/county arrays.
  Compactness reuses the lengths; virtual bridges have zero boundary length.
- `_SeriesBuffer` amortizes numeric conversion and allocation; chart scanning
  still scales with the visible window. Map color matching is shared by fills
  and labels; static border masks are lazy and raster-owned.
- Ensemble results publish only after writes succeed. SQLite queries run off
  the frame thread, with one outstanding request per view type. Statistics use
  all finite results; scatter drawing is capped at 2,000 points and roster pages
  at 100 rows. Assignment selection reads one plan, and final CSV export uses
  bounded stripes. Unlimited mode still grows disk usage.
- Targeting fits outside the progress lock and publishes compact snapshots.
  Cancellation occurs between bootstrap replicates; an active solver call must
  return first. Fitting starts after 10 runs and continues past warm-up.
- Python 3.10 uses `tomli`; Python 3.11+ uses `tomllib`. Shared preset validation
  checks values before GUI mutation. The optional private headless config uses
  config-relative preset paths; explicit config values override preset values.
