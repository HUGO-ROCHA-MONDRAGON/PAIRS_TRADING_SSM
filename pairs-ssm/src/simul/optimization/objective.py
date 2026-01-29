"""
Objective functions for threshold optimization.

Various objectives to maximize during threshold search.
"""

import numpy as np
from typing import Callable

from ..trading import BacktestResult


def sharpe_objective(result: BacktestResult, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio objective (maximize risk-adjusted returns).
    
    This is the primary objective in Zhang (2021).
    """
    return result.sharpe_ratio(periods_per_year)


def return_objective(result: BacktestResult) -> float:
    """
    Total return objective (maximize absolute returns).
    """
    return result.total_return()


def calmar_objective(result: BacktestResult, periods_per_year: int = 252) -> float:
    """
    Calmar ratio objective (return / max drawdown).
    """
    return result.calmar_ratio(periods_per_year)


def sortino_objective(
    result: BacktestResult,
    periods_per_year: int = 252,
    target: float = 0.0,
) -> float:
    """
    Sortino ratio objective (downside risk-adjusted returns).
    
    Parameters
    ----------
    result : BacktestResult
        Backtest results
    periods_per_year : int
        Periods per year
    target : float
        Target return (default 0)
        
    Returns
    -------
    float
        Sortino ratio
    """
    excess = result.pnl.mean() - target / periods_per_year
    
    # Downside deviation
    downside = result.pnl[result.pnl < target / periods_per_year]
    if len(downside) == 0:
        return np.inf if excess > 0 else 0.0
    
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return np.inf if excess > 0 else 0.0
    
    return excess / downside_std * np.sqrt(periods_per_year)


def profit_factor_objective(result: BacktestResult) -> float:
    """
    Profit factor: gross profits / gross losses.
    """
    profits = result.pnl[result.pnl > 0].sum()
    losses = -result.pnl[result.pnl < 0].sum()
    
    if losses == 0:
        return np.inf if profits > 0 else 1.0
    
    return profits / losses


def custom_objective(
    sharpe_weight: float = 0.5,
    return_weight: float = 0.3,
    drawdown_weight: float = 0.2,
    trade_penalty: float = 0.001,
) -> Callable[[BacktestResult], float]:
    """
    Create custom weighted objective function.
    
    Parameters
    ----------
    sharpe_weight : float
        Weight on Sharpe ratio
    return_weight : float
        Weight on total return
    drawdown_weight : float
        Weight on max drawdown (negative)
    trade_penalty : float
        Penalty per trade
        
    Returns
    -------
    callable
        Objective function
    """
    def objective(result: BacktestResult) -> float:
        sharpe = result.sharpe_ratio()
        total_ret = result.total_return()
        mdd = result.max_drawdown()
        n_trades = result.n_trades
        
        # Normalize components
        score = (
            sharpe_weight * sharpe
            + return_weight * total_ret * 100  # Scale returns
            - drawdown_weight * mdd * 100      # Penalize drawdown
            - trade_penalty * n_trades         # Penalize excessive trading
        )
        
        return score
    
    return objective


def information_ratio_objective(
    result: BacktestResult,
    benchmark_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Information ratio relative to benchmark.
    
    Parameters
    ----------
    result : BacktestResult
        Backtest results
    benchmark_return : float
        Annualized benchmark return
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    float
        Information ratio
    """
    daily_benchmark = benchmark_return / periods_per_year
    tracking_error = (result.pnl - daily_benchmark).std()
    
    if tracking_error == 0:
        excess = result.pnl.mean() - daily_benchmark
        return np.inf if excess > 0 else -np.inf
    
    return (result.pnl.mean() - daily_benchmark) / tracking_error * np.sqrt(periods_per_year)
