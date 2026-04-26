"""
Simulated annealing acceptance and temperature schedule.

Cooling modes (user-facing labels):
  Guided  — rate auto-calculated so temperature reaches target_temp at
             guide_fraction x max_iterations  (default, recommended)
  Static  — user supplies a fixed per-iteration multiplier

Temperature initialisation modes:
  PROPORTIONAL -- initial_temp = initial_temp_factor x initial_score  (default)
  NOMINAL      -- initial_temp = initial_temp_factor  (absolute value)
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass


@dataclass
class AnnealingConfig:
    """User-facing annealing settings (set once before a run)."""
    enabled: bool = True

    # Temperature initialisation
    temp_mode: str = "PROPORTIONAL"   # "PROPORTIONAL" | "NOMINAL"
    initial_temp_factor: float = 0.2

    # Cooling schedule
    cooling_mode: str = "GUIDED"      # "GUIDED" | "STATIC"

    # Guided mode: reach target_temp (absolute) at guide_fraction of total iterations.
    guide_fraction: float = 0.9       # e.g. 0.9 -> cool to target_temp at 90% of run
    target_temp: float = 1.0          # absolute temperature at the guide point

    # Static mode: user-supplied fixed per-iteration multiplier
    cooling_rate: float = 0.9995


@dataclass
class AnnealingState:
    """Runtime annealing state -- updated each iteration."""
    temperature: float
    initial_temp: float
    cooling_rate: float        # resolved (may differ from config in Guided mode)
    accepted_worse: int = 0    # worse proposals accepted by Metropolis
    rejected_worse: int = 0    # worse proposals rejected


def init_annealing(
    initial_score: float,
    config: AnnealingConfig,
    max_iterations: int,
) -> AnnealingState:
    """
    Initialise annealing state from config and the first-iteration score.

    Args:
        initial_score:  Score of the starting partition.
        config:         User-facing configuration.
        max_iterations: Total iterations planned (used for Guided cooling).
    """
    if config.temp_mode == "PROPORTIONAL":
        initial_temp = config.initial_temp_factor * initial_score
    else:  # NOMINAL
        initial_temp = config.initial_temp_factor

    if initial_temp <= 0:
        initial_temp = 1.0  # guard against degenerate initial scores

    if config.cooling_mode == "GUIDED":
        guide = max(1, int(config.guide_fraction * max_iterations))
        target = max(config.target_temp, 1e-9)
        if target >= initial_temp:
            # target is warmer than start -- just use a near-zero cooling rate
            cooling_rate = 0.9999
        else:
            # Solve: initial_temp * rate^guide = target_temp
            # => rate = (target_temp / initial_temp)^(1/guide)
            cooling_rate = (target / initial_temp) ** (1.0 / guide)
    else:  # STATIC
        cooling_rate = config.cooling_rate

    return AnnealingState(
        temperature=initial_temp,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
    )


def cool_temperature(state: AnnealingState) -> None:
    """Apply one cooling step in-place."""
    state.temperature *= state.cooling_rate


def accept_proposal(
    current_score: float,
    proposed_score: float,
    state: AnnealingState,
) -> bool:
    """
    Metropolis acceptance criterion.

    Always accepts improvements.  Accepts degradations with probability
    exp(-delta / temperature), updating accepted/rejected counters in-place.
    """
    if proposed_score <= current_score:
        return True

    delta = proposed_score - current_score
    if state.temperature <= 0:
        state.rejected_worse += 1
        return False

    if _random.random() < math.exp(-delta / state.temperature):
        state.accepted_worse += 1
        return True
    else:
        state.rejected_worse += 1
        return False
