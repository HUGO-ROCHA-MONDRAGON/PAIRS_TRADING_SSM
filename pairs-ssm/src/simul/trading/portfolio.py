"""
Portfolio construction and P&L calculation.

Implements P&L computation from log prices following Zhang (2021).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from dataclasses import dataclass

from .costs import TransactionCosts, paper_costs


@dataclass
class BacktestResult:
    """
    Results from backtesting a trading strategy.
    
    Attributes
    ----------
    pnl : pd.Series
        Daily P&L series
    cumulative_pnl : pd.Series
        Cumulative P&L
    signals : pd.Series
        Trading signals
    positions : pd.Series
        Position series
    trades : pd.Series
        Trade indicator (1 when position changes)
    costs : pd.Series
        Transaction costs incurred
    gross_pnl : pd.Series
        P&L before costs
    net_pnl : pd.Series
        P&L after costs (same as pnl)
    n_trades : int
        Total number of trades
    """
    pnl: pd.Series
    cumulative_pnl: pd.Series
    signals: pd.Series
    positions: pd.Series
    trades: pd.Series
    costs: pd.Series
    gross_pnl: pd.Series
    net_pnl: pd.Series
    n_trades: int
    
    def total_return(self) -> float:
        """Total return over the period."""
        return float(self.cumulative_pnl.iloc[-1])
    
    def annualized_return(self, periods_per_year: int = 252) -> float:
        """Annualized return."""
        n = len(self.pnl)
        years = n / periods_per_year
        return self.total_return() / years
    
    def annualized_volatility(self, periods_per_year: int = 252) -> float:
        """Annualized volatility of returns."""
        return float(self.pnl.std() * np.sqrt(periods_per_year))
    
    def sharpe_ratio(self, periods_per_year: int = 252, rf: float = 0.0) -> float:
        """Annualized Sharpe ratio."""
        excess_return = self.pnl.mean() - rf / periods_per_year
        if self.pnl.std() == 0:
            return 0.0
        return float(excess_return / self.pnl.std() * np.sqrt(periods_per_year))
    
    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        cummax = self.cumulative_pnl.cummax()
        drawdown = cummax - self.cumulative_pnl
        return float(drawdown.max())
    
    def calmar_ratio(self, periods_per_year: int = 252) -> float:
        """Calmar ratio (annual return / max drawdown)."""
        mdd = self.max_drawdown()
        if mdd == 0:
            return np.inf if self.annualized_return(periods_per_year) > 0 else 0.0
        return self.annualized_return(periods_per_year) / mdd


def compute_pnl(
    signals: pd.Series,
    log_p1: pd.Series,
    log_p2: pd.Series,
    gamma: float,
    costs: Optional[TransactionCosts] = None,
) -> BacktestResult:
    """
    Compute P&L from trading signals.
    
    The spread is defined as: S_t = log(P1_t) - gamma * log(P2_t)
    
    A LONG spread position means: buy P1, sell gamma units of P2
    A SHORT spread position means: sell P1, buy gamma units of P2
    
    P&L from spread = delta_log_p1 - gamma * delta_log_p2
    
    Parameters
    ----------
    signals : pd.Series
        Trading signals (+1 long, -1 short, 0 flat)
    log_p1 : pd.Series
        Log price of asset 1
    log_p2 : pd.Series
        Log price of asset 2
    gamma : float
        Hedge ratio
    costs : TransactionCosts, optional
        Transaction cost model (default: paper 20bp)
        
    Returns
    -------
    BacktestResult
        Complete backtest results
    """
    if costs is None:
        costs = paper_costs()
    
    # Align indices
    idx = signals.index.intersection(log_p1.index).intersection(log_p2.index)
    signals = signals.loc[idx]
    log_p1 = log_p1.loc[idx]
    log_p2 = log_p2.loc[idx]
    
    # Position is signal lagged by 1 (trade at close, hold next day)
    positions = signals.shift(1).fillna(0)
    
    # Log returns
    ret1 = log_p1.diff()
    ret2 = log_p2.diff()
    
    # Spread return
    spread_ret = ret1 - gamma * ret2
    
    # Gross P&L: position * spread return
    gross_pnl = positions * spread_ret
    gross_pnl = gross_pnl.fillna(0)
    
    # Trades: position changes
    trades = (positions.diff().abs() > 0).astype(int)
    trades.iloc[0] = int(positions.iloc[0] != 0)  # Initial entry
    
    # Transaction costs
    # Cost incurred when position changes
    # Cost = |position_change| * cost_per_pair_trade
    position_change = positions.diff().abs()
    position_change.iloc[0] = abs(positions.iloc[0])
    cost_series = position_change * costs.pair_trade_cost()
    
    # Net P&L
    net_pnl = gross_pnl - cost_series
    cumulative_pnl = net_pnl.cumsum()
    
    n_trades = int(trades.sum())
    
    return BacktestResult(
        pnl=net_pnl,
        cumulative_pnl=cumulative_pnl,
        signals=signals,
        positions=positions,
        trades=trades,
        costs=cost_series,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        n_trades=n_trades,
    )


def backtest_signals(
    signals: pd.Series,
    log_p1: pd.Series,
    log_p2: pd.Series,
    gamma: float,
    cost_bp: float = 20.0,
) -> BacktestResult:
    """
    Convenience function for backtesting.
    
    Parameters
    ----------
    signals : pd.Series
        Trading signals
    log_p1, log_p2 : pd.Series
        Log prices
    gamma : float
        Hedge ratio
    cost_bp : float
        Transaction cost in basis points per asset (default 20)
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    costs = TransactionCosts(cost_per_trade=cost_bp / 10000)
    return compute_pnl(signals, log_p1, log_p2, gamma, costs)


def summary_statistics(result: BacktestResult, periods_per_year: int = 252) -> dict:
    """
    Generate summary statistics for a backtest.
    
    Parameters
    ----------
    result : BacktestResult
        Backtest results
    periods_per_year : int
        Periods per year for annualization
        
    Returns
    -------
    dict
        Dictionary of performance metrics
    """
    return {
        "total_return": result.total_return(),
        "annualized_return": result.annualized_return(periods_per_year),
        "annualized_volatility": result.annualized_volatility(periods_per_year),
        "sharpe_ratio": result.sharpe_ratio(periods_per_year),
        "max_drawdown": result.max_drawdown(),
        "calmar_ratio": result.calmar_ratio(periods_per_year),
        "n_trades": result.n_trades,
        "total_costs": float(result.costs.sum()),
        "gross_return": float(result.gross_pnl.sum()),
        "net_return": float(result.net_pnl.sum()),
    }
