"""
Main backtest engine.

Integrates model fitting, filtering, signal generation, and P&L calculation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass

from ..data import SpreadData, compute_spread
from ..models import ModelParams
from ..filtering import fit_model, compare_models
from ..trading import generate_signals, backtest_signals, BacktestResult, summary_statistics


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    model: str = "model_I"
    strategy: str = "C"
    n_std_threshold: float = 1.5
    cost_bp: float = 20.0
    use_filtered_spread: bool = True
    fit_method: str = "mle"


class BacktestEngine:
    """
    Main backtest engine for pairs trading.
    
    Attributes
    ----------
    spread_data : SpreadData
        Spread and price data
    model : str
        Model type
    params : ModelParams, optional
        Fitted model parameters
    filter_result : FilterResult, optional
        Filtered state estimates
    """
    
    def __init__(
        self,
        log_p1: pd.Series,
        log_p2: pd.Series,
        gamma: Optional[float] = None,
    ):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        log_p1, log_p2 : pd.Series
            Log prices of the pair
        gamma : float, optional
            Hedge ratio (estimated if not provided)
        """
        self.spread_data = compute_spread(log_p1, log_p2, gamma)
        self.log_p1 = log_p1
        self.log_p2 = log_p2
        self.params: Optional[ModelParams] = None
        self.filter_result = None
        self._fitted_model = None
    
    def fit(
        self,
        model: str = "model_I",
        method: str = "mle",
        **kwargs,
    ) -> "BacktestEngine":
        """
        Fit state-space model to spread data.
        
        Parameters
        ----------
        model : str
            Model type: 'model_I' or 'model_II'
        method : str
            Fitting method: 'mle'
        **kwargs
            Additional arguments for fit_model
            
        Returns
        -------
        BacktestEngine
            Self for chaining
        """
        self._fitted_model = model
        result = fit_model(
            self.spread_data.spread,
            model_type=model,
            **kwargs,
        )
        self.params = result.params
        self.filter_result = result
        return self
    
    def generate_signals(
        self,
        strategy: Literal["A", "B", "C"] = "C",
        n_std: float = 1.5,
        use_filtered: bool = True,
    ) -> pd.Series:
        """
        Generate trading signals.
        
        Parameters
        ----------
        strategy : str
            Trading strategy
        n_std : float
            Number of std devs for thresholds
        use_filtered : bool
            Use filtered spread (True) or observed (False)
            
        Returns
        -------
        pd.Series
            Trading signals
        """
        if self.params is None:
            raise ValueError("Must fit model first. Call fit().")
        
        # Select spread series
        if use_filtered and self.filter_result is not None:
            spread = pd.Series(
                self.filter_result.x_filt,
                index=self.spread_data.spread.index,
            )
        else:
            spread = self.spread_data.spread
        
        # Thresholds using ModelParams properties
        mu = self.params.long_run_mean
        sigma = self.params.long_run_std
        
        U = mu + n_std * sigma
        L = mu - n_std * sigma
        C = mu
        
        return generate_signals(spread, U, L, C, strategy)
    
    def backtest(
        self,
        strategy: Literal["A", "B", "C"] = "C",
        n_std: float = 1.5,
        use_filtered: bool = True,
        cost_bp: float = 20.0,
    ) -> BacktestResult:
        """
        Run full backtest.
        
        Parameters
        ----------
        strategy : str
            Trading strategy
        n_std : float
            Number of std devs for thresholds
        use_filtered : bool
            Use filtered spread
        cost_bp : float
            Transaction costs in basis points
            
        Returns
        -------
        BacktestResult
            Backtest results
        """
        signals = self.generate_signals(strategy, n_std, use_filtered)
        
        return backtest_signals(
            signals,
            self.log_p1,
            self.log_p2,
            self.spread_data.gamma,
            cost_bp,
        )
    
    def summary(
        self,
        strategy: str = "C",
        n_std: float = 1.5,
        use_filtered: bool = True,
        cost_bp: float = 20.0,
    ) -> dict:
        """
        Get summary statistics for backtest.
        
        Returns
        -------
        dict
            Performance metrics
        """
        result = self.backtest(strategy, n_std, use_filtered, cost_bp)
        stats = summary_statistics(result)
        
        # Add model info
        stats["model"] = self._fitted_model
        stats["strategy"] = strategy
        stats["n_std"] = n_std
        stats["gamma"] = self.spread_data.gamma
        
        if self.params is not None:
            stats["mu"] = self.params.mu
            stats["phi"] = self.params.phi
            stats["q"] = self.params.q
            stats["r"] = self.params.r
        
        return stats


def run_backtest(
    log_p1: pd.Series,
    log_p2: pd.Series,
    gamma: Optional[float] = None,
    model: str = "model_I",
    strategy: str = "C",
    n_std: float = 1.5,
    cost_bp: float = 20.0,
) -> BacktestResult:
    """
    Convenience function to run a full backtest.
    
    Parameters
    ----------
    log_p1, log_p2 : pd.Series
        Log prices
    gamma : float, optional
        Hedge ratio
    model : str
        Model type
    strategy : str
        Trading strategy
    n_std : float
        Threshold in std devs
    cost_bp : float
        Transaction costs
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    engine = BacktestEngine(log_p1, log_p2, gamma)
    engine.fit(model)
    return engine.backtest(strategy, n_std, cost_bp=cost_bp)


def run_model_backtest(
    log_p1: pd.Series,
    log_p2: pd.Series,
    gamma: Optional[float] = None,
    models: list = ["model_I", "model_II"],
    strategy: str = "C",
    n_std: float = 1.5,
    cost_bp: float = 20.0,
) -> pd.DataFrame:
    """
    Run backtest for multiple models and compare.
    
    Parameters
    ----------
    log_p1, log_p2 : pd.Series
        Log prices
    gamma : float, optional
        Hedge ratio
    models : list
        List of model types
    strategy : str
        Trading strategy
    n_std : float
        Threshold
    cost_bp : float
        Transaction costs
        
    Returns
    -------
    pd.DataFrame
        Comparison of model performance
    """
    results = []
    
    for model in models:
        engine = BacktestEngine(log_p1, log_p2, gamma)
        engine.fit(model)
        stats = engine.summary(strategy, n_std, cost_bp=cost_bp)
        results.append(stats)
    
    return pd.DataFrame(results)
