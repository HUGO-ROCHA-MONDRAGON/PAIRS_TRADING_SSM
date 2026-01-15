"""Threshold optimization module for pairs trading."""

from .threshold_grid import (
    threshold_grid_search,
    generate_threshold_grid,
    optimize_thresholds,
)
from .simulator import (
    simulate_ou_process,
    simulate_model,
    run_monte_carlo,
)
from .objective import (
    sharpe_objective,
    return_objective,
    calmar_objective,
    custom_objective,
)

__all__ = [
    # Grid search
    "threshold_grid_search",
    "generate_threshold_grid",
    "optimize_thresholds",
    # Simulation
    "simulate_ou_process",
    "simulate_model",
    "run_monte_carlo",
    # Objectives
    "sharpe_objective",
    "return_objective",
    "calmar_objective",
    "custom_objective",
]
