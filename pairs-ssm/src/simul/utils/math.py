"""
Mathematical utilities for financial calculations.
"""

import numpy as np
from typing import Union, Optional
import warnings


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def annualize_return(total_return: float, n_periods: int, periods_per_year: int = 252) -> float:
    """
    Annualize a total return.
    
    Parameters
    ----------
    total_return : float
        Total return (e.g., 0.25 for 25%)
    n_periods : int
        Number of periods over which return was achieved
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly)
        
    Returns
    -------
    float
        Annualized return
    """
    if n_periods <= 0:
        return 0.0
    
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    
    # Handle negative returns
    if total_return < -1:
        return -1.0
    
    return (1 + total_return) ** (1 / years) - 1


def annualize_volatility(daily_vol: float, periods_per_year: int = 252) -> float:
    """Annualize volatility."""
    return daily_vol * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : array-like
        Period returns (not cumulative)
    risk_free_rate : float
        Annual risk-free rate (default: 0)
    periods_per_year : int
        Periods per year for annualization
        
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    if std_ret < 1e-10:
        return 0.0
    
    # Convert annual rf to period rf
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_return = mean_ret - rf_period
    
    # Annualize
    return (excess_return / std_ret) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio (using downside deviation).
    
    Parameters
    ----------
    returns : array-like
        Period returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    float
        Annualized Sortino ratio
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    mean_ret = np.mean(returns)
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) < 2:
        return np.inf if mean_ret > rf_period else 0.0
    
    downside_std = np.std(negative_returns, ddof=1)
    
    if downside_std < 1e-10:
        return np.inf if mean_ret > rf_period else 0.0
    
    excess_return = mean_ret - rf_period
    
    return (excess_return / downside_std) * np.sqrt(periods_per_year)


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    cumulative_returns : array-like
        Cumulative returns (wealth curve, starting from 0 or 1)
        
    Returns
    -------
    float
        Maximum drawdown as a positive number (e.g., 0.15 for 15% drawdown)
    """
    cumulative_returns = np.asarray(cumulative_returns, dtype=float)
    
    if len(cumulative_returns) < 2:
        return 0.0
    
    # Convert to wealth curve if starting from 0
    if abs(cumulative_returns[0]) < 1e-10:
        wealth = 1.0 + cumulative_returns
    else:
        wealth = cumulative_returns
    
    # Running maximum
    running_max = np.maximum.accumulate(wealth)
    
    # Drawdowns
    drawdowns = (running_max - wealth) / running_max
    
    return float(np.max(drawdowns))


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Parameters
    ----------
    returns : array-like
        Period returns
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    float
        Calmar ratio
    """
    returns = np.asarray(returns, dtype=float)
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate cumulative returns
    cum_ret = np.cumprod(1 + returns) - 1
    
    # Annualized return
    total_ret = cum_ret[-1]
    ann_ret = annualize_return(total_ret, len(returns), periods_per_year)
    
    # Max drawdown
    mdd = max_drawdown(cum_ret)
    
    if mdd < 1e-10:
        return np.inf if ann_ret > 0 else 0.0
    
    return ann_ret / mdd


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Parameters
    ----------
    returns : array-like
        Period returns
    confidence : float
        Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns
    -------
    float
        VaR as a positive number
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    return -np.percentile(returns, (1 - confidence) * 100)


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    
    Parameters
    ----------
    returns : array-like
        Period returns
    confidence : float
        Confidence level
        
    Returns
    -------
    float
        CVaR as a positive number
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    var = value_at_risk(returns, confidence)
    
    # Average of returns below VaR threshold
    tail_returns = returns[returns <= -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -np.mean(tail_returns)


def information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Information Ratio.
    
    Parameters
    ----------
    returns : array-like
        Strategy returns
    benchmark_returns : array-like
        Benchmark returns
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    float
        Information Ratio
    """
    returns = np.asarray(returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    
    # Align lengths
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Active returns
    active = returns - benchmark_returns
    
    if len(active) < 2:
        return 0.0
    
    tracking_error = np.std(active, ddof=1)
    
    if tracking_error < 1e-10:
        return 0.0
    
    return (np.mean(active) / tracking_error) * np.sqrt(periods_per_year)


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio.
    
    Parameters
    ----------
    returns : array-like
        Period returns
    threshold : float
        Return threshold (typically 0)
        
    Returns
    -------
    float
        Omega Ratio
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 2:
        return 1.0
    
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    
    sum_losses = np.sum(losses)
    
    if sum_losses < 1e-10:
        return np.inf if len(gains) > 0 else 1.0
    
    return np.sum(gains) / sum_losses
