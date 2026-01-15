"""
Grid search for optimal trading thresholds.

Following Zhang (2021), thresholds are optimized via simulation
before being applied to real data.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Literal
from dataclasses import dataclass
from itertools import product

from ..trading import generate_signals, BacktestResult


@dataclass
class ThresholdResult:
    """
    Result from threshold optimization.
    
    Attributes
    ----------
    U : float
        Optimal upper threshold
    L : float
        Optimal lower threshold
    C : float
        Mean/center value
    objective_value : float
        Value of the objective function at optimal
    strategy : str
        Strategy used ('A', 'B', or 'C')
    grid_results : pd.DataFrame
        Full grid search results
    """
    U: float
    L: float
    C: float
    objective_value: float
    strategy: str
    grid_results: pd.DataFrame


def generate_threshold_grid(
    spread_mean: float,
    spread_std: float,
    n_std_U: tuple = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5),
    n_std_L: tuple = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5),
    symmetric: bool = True,
) -> list:
    """
    Generate grid of threshold combinations.
    
    Parameters
    ----------
    spread_mean : float
        Mean of the spread (C value)
    spread_std : float
        Standard deviation of the spread
    n_std_U : tuple
        Number of std devs above mean for U
    n_std_L : tuple
        Number of std devs below mean for L
    symmetric : bool
        If True, only use symmetric thresholds (U - C = C - L)
        
    Returns
    -------
    list
        List of (U, L, C) tuples
    """
    grid = []
    C = spread_mean
    
    if symmetric:
        for n in n_std_U:
            U = C + n * spread_std
            L = C - n * spread_std
            grid.append((U, L, C))
    else:
        for n_u, n_l in product(n_std_U, n_std_L):
            U = C + n_u * spread_std
            L = C - n_l * spread_std
            grid.append((U, L, C))
    
    return grid


def threshold_grid_search(
    spread: np.ndarray,
    log_p1: np.ndarray,
    log_p2: np.ndarray,
    gamma: float,
    grid: list,
    strategy: Literal["A", "B", "C"] = "C",
    objective: Callable[[BacktestResult], float] = None,
    cost_bp: float = 20.0,
) -> ThresholdResult:
    """
    Perform grid search over thresholds.
    
    Parameters
    ----------
    spread : np.ndarray
        Spread series for signal generation
    log_p1, log_p2 : np.ndarray
        Log prices for P&L
    gamma : float
        Hedge ratio
    grid : list
        List of (U, L, C) tuples
    strategy : str
        Trading strategy
    objective : callable
        Function mapping BacktestResult -> float (maximize)
    cost_bp : float
        Transaction costs in basis points
        
    Returns
    -------
    ThresholdResult
        Optimal thresholds and grid results
    """
    from ..trading import backtest_signals
    
    if objective is None:
        objective = lambda r: r.sharpe_ratio()
    
    # Create pandas series
    idx = pd.RangeIndex(len(spread))
    spread_s = pd.Series(spread, index=idx)
    log_p1_s = pd.Series(log_p1, index=idx)
    log_p2_s = pd.Series(log_p2, index=idx)
    
    results = []
    
    for U, L, C in grid:
        signals = generate_signals(spread_s, U, L, C, strategy)
        bt_result = backtest_signals(signals, log_p1_s, log_p2_s, gamma, cost_bp)
        obj_val = objective(bt_result)
        
        results.append({
            "U": U,
            "L": L,
            "C": C,
            "objective": obj_val,
            "sharpe": bt_result.sharpe_ratio(),
            "total_return": bt_result.total_return(),
            "max_drawdown": bt_result.max_drawdown(),
            "n_trades": bt_result.n_trades,
        })
    
    df = pd.DataFrame(results)
    
    # Find best
    best_idx = df["objective"].idxmax()
    best = df.loc[best_idx]
    
    return ThresholdResult(
        U=best["U"],
        L=best["L"],
        C=best["C"],
        objective_value=best["objective"],
        strategy=strategy,
        grid_results=df,
    )


def optimize_thresholds(
    spread_samples: list,
    log_p1_samples: list,
    log_p2_samples: list,
    gamma: float,
    spread_mean: float,
    spread_std: float,
    strategy: Literal["A", "B", "C"] = "C",
    n_std_range: tuple = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    cost_bp: float = 20.0,
    aggregation: str = "mean",
) -> ThresholdResult:
    """
    Optimize thresholds using Monte Carlo simulation.
    
    Average objective values across multiple simulated samples
    to find robust thresholds.
    
    Parameters
    ----------
    spread_samples : list
        List of simulated spread arrays
    log_p1_samples, log_p2_samples : list
        List of simulated log price arrays
    gamma : float
        Hedge ratio
    spread_mean, spread_std : float
        For threshold grid generation
    strategy : str
        Trading strategy
    n_std_range : tuple
        Range of std devs for thresholds
    cost_bp : float
        Transaction costs
    aggregation : str
        How to aggregate: 'mean' or 'median'
        
    Returns
    -------
    ThresholdResult
        Optimal thresholds based on simulation
    """
    from ..trading import backtest_signals
    
    grid = generate_threshold_grid(spread_mean, spread_std, n_std_range, n_std_range)
    n_samples = len(spread_samples)
    
    # Initialize storage
    obj_values = {(U, L, C): [] for U, L, C in grid}
    
    for i in range(n_samples):
        spread = spread_samples[i]
        log_p1 = log_p1_samples[i]
        log_p2 = log_p2_samples[i]
        
        idx = pd.RangeIndex(len(spread))
        spread_s = pd.Series(spread, index=idx)
        log_p1_s = pd.Series(log_p1, index=idx)
        log_p2_s = pd.Series(log_p2, index=idx)
        
        for U, L, C in grid:
            signals = generate_signals(spread_s, U, L, C, strategy)
            bt_result = backtest_signals(signals, log_p1_s, log_p2_s, gamma, cost_bp)
            obj_values[(U, L, C)].append(bt_result.sharpe_ratio())
    
    # Aggregate
    agg_func = np.mean if aggregation == "mean" else np.median
    
    results = []
    for (U, L, C), vals in obj_values.items():
        results.append({
            "U": U,
            "L": L,
            "C": C,
            "objective": agg_func(vals),
            "std": np.std(vals),
            "n_samples": len(vals),
        })
    
    df = pd.DataFrame(results)
    best_idx = df["objective"].idxmax()
    best = df.loc[best_idx]
    
    return ThresholdResult(
        U=best["U"],
        L=best["L"],
        C=best["C"],
        objective_value=best["objective"],
        strategy=strategy,
        grid_results=df,
    )
