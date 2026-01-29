__version__ = "0.1.0"
__author__ = "Quantitative Finance Project"

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
