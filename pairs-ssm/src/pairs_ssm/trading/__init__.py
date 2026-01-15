"""Trading strategies and backtesting module."""

from .strategy import (
    strategy_A_signals,
    strategy_B_signals,
    strategy_C_signals,
    generate_signals,
    signals_to_positions,
)
from .costs import (
    TransactionCosts,
    zero_costs,
    paper_costs,
    retail_costs,
    institutional_costs,
)
from .portfolio import (
    BacktestResult,
    compute_pnl,
    backtest_signals,
    summary_statistics,
)

__all__ = [
    # Strategies
    "strategy_A_signals",
    "strategy_B_signals",
    "strategy_C_signals",
    "generate_signals",
    "signals_to_positions",
    # Costs
    "TransactionCosts",
    "zero_costs",
    "paper_costs",
    "retail_costs",
    "institutional_costs",
    # Portfolio
    "BacktestResult",
    "compute_pnl",
    "backtest_signals",
    "summary_statistics",
]
