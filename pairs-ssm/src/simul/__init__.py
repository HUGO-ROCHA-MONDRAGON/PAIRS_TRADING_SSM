"""
Pairs Trading with State-Space Models
======================================

A Python package for pairs trading using state-space models,
implementing the methodology from Zhang (2021).

Main components:
- trading: Trading strategy logic (strategies A, B, C, D, E)
- optimization: Threshold optimization (Table 1 & Table S1)
- utils: Simulation utilities (CIR spread simulation)

Example usage:
    >>> from simul.utils.simulation import simulate_cir_spread
    >>> from simul.trading.strategy import strategy_A_signals, find_trades
    >>> from simul.optimization.table1 import replicate_table1
"""

__version__ = "0.1.0"
__author__ = "Quantitative Finance Project"

# Trading strategies
from simul.trading.strategy import (
    strategy_A_signals,
    strategy_B_signals,
    strategy_C_signals,
    strategy_C_signals_timevarying,
    strategy_D_signals,
    strategy_E_signals,
    find_trades,
    find_trades_B,
    find_trades_safe,
)

# Optimization / Table 1 replication
from simul.optimization.table1 import (
    replicate_table1,
    replicate_tableS1,
    simulate_paths,
    simulate_paths_S1,
    Table1Result,
    TableS1Result,
    PAPER_TABLE1,
    NUMBA_AVAILABLE,
)

# Simulation utilities
from simul.utils.simulation import simulate_cir_spread, simulate_ou_spread

__all__ = [
    # Version
    "__version__",
    # Trading strategies
    "strategy_A_signals",
    "strategy_B_signals",
    "strategy_C_signals",
    "strategy_C_signals_timevarying",
    "strategy_D_signals",
    "strategy_E_signals",
    "find_trades",
    "find_trades_B",
    "find_trades_safe",
    # Optimization
    "replicate_table1",
    "replicate_tableS1",
    "simulate_paths",
    "simulate_paths_S1",
    "Table1Result",
    "TableS1Result",
    "PAPER_TABLE1",
    "NUMBA_AVAILABLE",
    # Simulation
    "simulate_cir_spread",
    "simulate_ou_spread",
]
