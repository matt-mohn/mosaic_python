"""Thread-safe shared state for algorithm ↔ GUI communication."""

import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from mosaic.scoring.score import ScoreConfig
from mosaic.recom.annealing import AnnealingConfig


class AlgorithmStatus(Enum):
    IDLE = "idle"
    LOADING = "loading"
    BUILDING_GRAPH = "building_graph"
    PARTITIONING = "partitioning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SharedState:
    """
    Thread-safe container for state shared between algorithm and GUI.
    Algorithm thread writes; GUI thread reads via the public methods.
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Status
    status: AlgorithmStatus = AlgorithmStatus.IDLE
    status_message: str = ""
    error_message: str = ""

    # Loaded data
    num_precincts: int = 0
    total_population: int = 0
    shapefile_path: str = ""

    # Run parameters (set by GUI before run)
    num_districts: int = 0
    pop_tolerance: float = 0.05
    max_iterations: int = 1000
    seed: Optional[int] = None

    # Score config (set by GUI before run)
    score_config: ScoreConfig = field(default_factory=ScoreConfig)

    # Annealing config (set by GUI before run)
    annealing_config: AnnealingConfig = field(default_factory=AnnealingConfig)

    # Run progress
    current_iteration: int = 0
    successful_steps: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    # Current plan
    current_assignment: Optional[np.ndarray] = None
    current_score: float = float("inf")   # weighted total
    current_cut_edges: int = 0            # raw count

    # Best plan seen
    best_assignment: Optional[np.ndarray] = None
    best_score: float = float("inf")
    best_iteration: int = 0
    best_cut_edges: int = 0
    best_temperature: float = 0.0
    # History lengths at the moment best was recorded (for Revert to Best trimming)
    best_score_history_len: int = 0
    best_temp_history_len: int = 0
    best_acc_history_len: int = 0

    # Annealing runtime
    current_temperature: float = 0.0
    accepted_worse: int = 0
    rejected_worse: int = 0

    # Plot histories
    score_history: list = field(default_factory=list)
    temperature_history: list = field(default_factory=list)
    # Each entry: [iteration_float, rolling_acceptance_rate_float]
    acceptance_rate_history: list = field(default_factory=list)
    # Per-metric histories (one entry per iteration, same length as score_history)
    county_splits_score_history: list = field(default_factory=list)
    county_excess_splits_history: list = field(default_factory=list)
    county_clean_districts_history: list = field(default_factory=list)
    mm_history: list = field(default_factory=list)
    eg_history: list = field(default_factory=list)
    dem_seats_history: list = field(default_factory=list)
    competitive_count_history: list = field(default_factory=list)
    pp_history: list = field(default_factory=list)
    cut_edges_history: list = field(default_factory=list)

    # Score breakdown: metric name → % of total weighted score (updated by runner)
    score_breakdown: dict = field(default_factory=dict)

    # Map state
    gdf_ready: bool = False                 # pulsed True by runner after full load completes
    shp_inspect_ready: bool = False         # pulsed True by runner after inspection completes
    initial_assignment: Optional[np.ndarray] = None  # frozen after first partition
    map_needs_update: bool = False          # pulsed True by runner on interval
    map_render_interval: float = 0.75       # seconds between map renders

    # County-edge bias (set by GUI before run)
    county_bias_enabled: bool = False
    county_bias: float = 5.0

    # Control flags
    should_stop: bool = False
    should_pause: bool = False
    pause_time: float = 0.0      # wall-clock time when pause began (0 = not paused)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get(self, *fields) -> tuple:
        with self._lock:
            return tuple(getattr(self, f) for f in fields)

    def get_all(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "status_message": self.status_message,
                "error_message": self.error_message,
                "num_precincts": self.num_precincts,
                "total_population": self.total_population,
                "num_districts": self.num_districts,
                "pop_tolerance": self.pop_tolerance,
                "max_iterations": self.max_iterations,
                "current_iteration": self.current_iteration,
                "successful_steps": self.successful_steps,
                "current_score": self.current_score,
                "current_cut_edges": self.current_cut_edges,
                "best_score": self.best_score,
                "best_iteration": self.best_iteration,
                "current_temperature": self.current_temperature,
                "accepted_worse": self.accepted_worse,
                "rejected_worse": self.rejected_worse,
                "gdf_ready": self.gdf_ready,
                "shp_inspect_ready": self.shp_inspect_ready,
            }

    def reset_run(self) -> None:
        with self._lock:
            self.current_iteration = 0
            self.successful_steps = 0
            self.current_score = float("inf")
            self.current_cut_edges = 0
            self.best_score = float("inf")
            self.best_iteration = 0
            self.best_assignment = None
            self.best_cut_edges = 0
            self.best_temperature = 0.0
            self.best_score_history_len = 0
            self.best_temp_history_len = 0
            self.best_acc_history_len = 0
            self.current_assignment = None
            self.current_temperature = 0.0
            self.accepted_worse = 0
            self.rejected_worse = 0
            self.score_history = []
            self.temperature_history = []
            self.acceptance_rate_history = []
            self.county_splits_score_history = []
            self.county_excess_splits_history = []
            self.county_clean_districts_history = []
            self.mm_history = []
            self.eg_history = []
            self.dem_seats_history = []
            self.competitive_count_history = []
            self.pp_history = []
            self.cut_edges_history = []
            self.score_breakdown = {}
            self.initial_assignment = None
            self.map_needs_update = False
            self.shp_inspect_ready = False
            self.should_stop = False
            self.should_pause = False
            self.pause_time = 0.0
            self.error_message = ""
            self.start_time = 0.0
            self.end_time = 0.0

    def request_stop(self) -> None:
        with self._lock:
            self.should_stop = True

    def request_pause(self) -> None:
        with self._lock:
            self.should_pause = True

    def request_resume(self) -> None:
        with self._lock:
            self.should_pause = False

    def check_should_stop(self) -> bool:
        with self._lock:
            return self.should_stop

    def check_should_pause(self) -> bool:
        with self._lock:
            return self.should_pause
