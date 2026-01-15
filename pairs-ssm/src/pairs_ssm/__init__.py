"""
Pairs Trading with State-Space Models
======================================

A Python package for pairs trading using state-space models,
implementing the methodology from Zhang (2021).

Main components:
- data: Data loading and transformation
- models: State-space model definitions
- filtering: Kalman filter implementations
- trading: Trading strategy logic
- optimization: Threshold optimization
- backtest: Backtesting engine
- reporting: Results visualization

Example usage:
    >>> from pairs_ssm import load_pair, compute_spread, fit_model
    >>> pair_data = load_pair("path/to/data.xlsx", "PEP", "KO")
    >>> spread_data = compute_spread(pair_data.log_p1, pair_data.log_p2)
    >>> result = fit_model(spread_data.spread, model_type="model_I")
"""

__version__ = "0.1.0"
__author__ = "Quantitative Finance Project"

# Core imports
from pairs_ssm.utils.io import load_config, save_results
from pairs_ssm.data.loaders import load_pair, PairData
from pairs_ssm.data.transforms import compute_spread, SpreadData
from pairs_ssm.filtering.mle import fit_model, compare_models

# Trading
from pairs_ssm.trading import (
    generate_signals,
    strategy_A_signals,
    strategy_B_signals,
    strategy_C_signals,
    backtest_signals,
    BacktestResult,
    TransactionCosts,
)

# Backtest engine
from pairs_ssm.backtest import (
    BacktestEngine,
    run_backtest,
    WalkForwardConfig,
    walk_forward_backtest,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "load_config",
    "save_results",
    # Data
    "load_pair",
    "PairData",
    "compute_spread",
    "SpreadData",
    # Filtering
    "fit_model",
    "compare_models",
    # Trading
    "generate_signals",
    "strategy_A_signals",
    "strategy_B_signals",
    "strategy_C_signals",
    "backtest_signals",
    "BacktestResult",
    "TransactionCosts",
    # Backtest
    "BacktestEngine",
    "run_backtest",
    "WalkForwardConfig",
    "walk_forward_backtest",
]
