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
from .table1 import (
    replicate_table1,
    run_table1_optimization,
    simulate_paths,
    Table1Result,
    PAPER_TABLE1,
    NUMBA_AVAILABLE,
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
    # Table 1 replication
    "replicate_table1",
    "run_table1_optimization",
    "simulate_paths",
    "Table1Result",
    "PAPER_TABLE1",
    "NUMBA_AVAILABLE",
]
