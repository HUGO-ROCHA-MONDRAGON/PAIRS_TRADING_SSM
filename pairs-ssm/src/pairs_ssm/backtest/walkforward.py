"""
Walk-forward and expanding window backtesting.

Implements proper out-of-sample testing methodology.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal, List
from dataclasses import dataclass, field

from .engine import BacktestEngine
from ..trading import BacktestResult, summary_statistics


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward analysis.
    
    Attributes
    ----------
    train_periods : int
        Number of periods for training window
    test_periods : int
        Number of periods for testing window
    step_size : int
        Step size for rolling window (default = test_periods)
    min_train_periods : int
        Minimum training periods (for expanding window)
    expanding : bool
        Use expanding window instead of rolling
    model : str
        Model type
    strategy : str
        Trading strategy
    n_std : float
        Threshold in std devs
    cost_bp : float
        Transaction costs
    """
    train_periods: int = 252  # 1 year
    test_periods: int = 63    # 3 months
    step_size: Optional[int] = None
    min_train_periods: int = 126  # 6 months
    expanding: bool = False
    model: str = "model_I"
    strategy: str = "C"
    n_std: float = 1.5
    cost_bp: float = 20.0
    
    def __post_init__(self):
        if self.step_size is None:
            self.step_size = self.test_periods


@dataclass
class WalkForwardResult:
    """
    Results from walk-forward analysis.
    
    Attributes
    ----------
    combined_pnl : pd.Series
        Combined out-of-sample P&L
    period_results : List[BacktestResult]
        Results for each test period
    period_params : List[dict]
        Fitted parameters for each period
    train_ranges : List[tuple]
        Training date ranges
    test_ranges : List[tuple]
        Test date ranges
    """
    combined_pnl: pd.Series
    period_results: List[BacktestResult]
    period_params: List[dict]
    train_ranges: List[tuple]
    test_ranges: List[tuple]
    
    def summary(self) -> dict:
        """Get summary statistics for combined results."""
        # Create combined BacktestResult-like object
        cumulative_pnl = self.combined_pnl.cumsum()
        
        return {
            "total_return": float(cumulative_pnl.iloc[-1]),
            "annualized_return": float(self.combined_pnl.mean() * 252),
            "annualized_volatility": float(self.combined_pnl.std() * np.sqrt(252)),
            "sharpe_ratio": float(self.combined_pnl.mean() / self.combined_pnl.std() * np.sqrt(252)) if self.combined_pnl.std() > 0 else 0.0,
            "max_drawdown": float((cumulative_pnl.cummax() - cumulative_pnl).max()),
            "n_periods": len(self.period_results),
            "total_trades": sum(r.n_trades for r in self.period_results),
        }


def walk_forward_backtest(
    log_p1: pd.Series,
    log_p2: pd.Series,
    config: WalkForwardConfig,
    gamma: Optional[float] = None,
) -> WalkForwardResult:
    """
    Perform walk-forward backtest.
    
    For each period:
    1. Fit model on training window
    2. Generate signals and compute P&L on test window
    3. Roll forward by step_size
    
    Parameters
    ----------
    log_p1, log_p2 : pd.Series
        Log prices (must have DatetimeIndex)
    config : WalkForwardConfig
        Walk-forward configuration
    gamma : float, optional
        Hedge ratio
        
    Returns
    -------
    WalkForwardResult
        Walk-forward backtest results
    """
    n = len(log_p1)
    
    if config.expanding:
        return expanding_window_backtest(log_p1, log_p2, config, gamma)
    
    period_results = []
    period_params = []
    train_ranges = []
    test_ranges = []
    combined_pnl_parts = []
    
    start = 0
    
    while start + config.train_periods + config.test_periods <= n:
        train_end = start + config.train_periods
        test_end = train_end + config.test_periods
        
        # Training data
        train_p1 = log_p1.iloc[start:train_end]
        train_p2 = log_p2.iloc[start:train_end]
        
        # Test data
        test_p1 = log_p1.iloc[train_end:test_end]
        test_p2 = log_p2.iloc[train_end:test_end]
        
        # Fit on training
        engine = BacktestEngine(train_p1, train_p2, gamma)
        engine.fit(config.model)
        
        # Extract params and thresholds
        params = engine.params
        mu = params.mu
        sigma = np.sqrt(params.q / (1 - params.phi ** 2))
        U = mu + config.n_std * sigma
        L = mu - config.n_std * sigma
        
        # Apply to test period
        test_engine = BacktestEngine(test_p1, test_p2, gamma)
        test_engine.params = params  # Use training params
        
        # Need to filter test data with trained model params
        from ..filtering import fit_model
        test_spread = test_engine.spread_data.spread.values
        
        # Filter test data using training params (no re-estimation)
        from ..filtering.kalman_linear import kalman_filter
        x_filt, P_filt, ll = kalman_filter(
            test_spread,
            params.mu,
            params.phi,
            params.q,
            params.r,
        )
        
        # Generate signals on filtered test spread
        from ..trading import generate_signals, backtest_signals
        test_spread_filtered = pd.Series(x_filt, index=test_p1.index)
        signals = generate_signals(test_spread_filtered, U, L, mu, config.strategy)
        
        # Compute P&L
        result = backtest_signals(
            signals,
            test_p1,
            test_p2,
            test_engine.spread_data.gamma,
            config.cost_bp,
        )
        
        period_results.append(result)
        period_params.append({
            "mu": params.mu,
            "phi": params.phi,
            "q": params.q,
            "r": params.r,
            "U": U,
            "L": L,
        })
        train_ranges.append((log_p1.index[start], log_p1.index[train_end - 1]))
        test_ranges.append((log_p1.index[train_end], log_p1.index[test_end - 1]))
        combined_pnl_parts.append(result.pnl)
        
        # Move forward
        start += config.step_size
    
    # Combine P&L
    combined_pnl = pd.concat(combined_pnl_parts)
    
    return WalkForwardResult(
        combined_pnl=combined_pnl,
        period_results=period_results,
        period_params=period_params,
        train_ranges=train_ranges,
        test_ranges=test_ranges,
    )


def expanding_window_backtest(
    log_p1: pd.Series,
    log_p2: pd.Series,
    config: WalkForwardConfig,
    gamma: Optional[float] = None,
) -> WalkForwardResult:
    """
    Perform expanding window backtest.
    
    Training window grows over time (anchored at start).
    
    Parameters
    ----------
    log_p1, log_p2 : pd.Series
        Log prices
    config : WalkForwardConfig
        Configuration
    gamma : float, optional
        Hedge ratio
        
    Returns
    -------
    WalkForwardResult
        Backtest results
    """
    n = len(log_p1)
    
    period_results = []
    period_params = []
    train_ranges = []
    test_ranges = []
    combined_pnl_parts = []
    
    train_end = config.min_train_periods
    
    while train_end + config.test_periods <= n:
        test_end = train_end + config.test_periods
        
        # Training: from start to train_end
        train_p1 = log_p1.iloc[:train_end]
        train_p2 = log_p2.iloc[:train_end]
        
        # Test
        test_p1 = log_p1.iloc[train_end:test_end]
        test_p2 = log_p2.iloc[train_end:test_end]
        
        # Fit on training
        engine = BacktestEngine(train_p1, train_p2, gamma)
        engine.fit(config.model)
        
        params = engine.params
        mu = params.mu
        sigma = np.sqrt(params.q / (1 - params.phi ** 2))
        U = mu + config.n_std * sigma
        L = mu - config.n_std * sigma
        
        # Apply to test
        test_engine = BacktestEngine(test_p1, test_p2, gamma)
        test_spread = test_engine.spread_data.spread.values
        
        from ..filtering.kalman_linear import kalman_filter
        x_filt, P_filt, ll = kalman_filter(
            test_spread,
            params.mu,
            params.phi,
            params.q,
            params.r,
        )
        
        from ..trading import generate_signals, backtest_signals
        test_spread_filtered = pd.Series(x_filt, index=test_p1.index)
        signals = generate_signals(test_spread_filtered, U, L, mu, config.strategy)
        
        result = backtest_signals(
            signals,
            test_p1,
            test_p2,
            test_engine.spread_data.gamma,
            config.cost_bp,
        )
        
        period_results.append(result)
        period_params.append({
            "mu": params.mu,
            "phi": params.phi,
            "q": params.q,
            "r": params.r,
            "U": U,
            "L": L,
        })
        train_ranges.append((log_p1.index[0], log_p1.index[train_end - 1]))
        test_ranges.append((log_p1.index[train_end], log_p1.index[test_end - 1]))
        combined_pnl_parts.append(result.pnl)
        
        train_end += config.step_size
    
    combined_pnl = pd.concat(combined_pnl_parts)
    
    return WalkForwardResult(
        combined_pnl=combined_pnl,
        period_results=period_results,
        period_params=period_params,
        train_ranges=train_ranges,
        test_ranges=test_ranges,
    )
