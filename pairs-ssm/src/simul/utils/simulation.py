"""
Simulation utilities for generating synthetic spreads.
"""

import numpy as np
import pandas as pd
from typing import Optional


def simulate_cir_spread(
    n: int = 800,
    dt: float = 1/252,
    s0: float = 5.0,
    kappa: float = 3.0,
    theta: float = 5.0,
    sigma: float = 0.9,
    seed: Optional[int] = 42
) -> pd.Series:
    """
    Simulate a positive mean-reverting spread using the CIR process.
    
    The spread follows the Cox-Ingersoll-Ross (CIR) stochastic differential equation:
        dS = kappa * (theta - S) * dt + sigma * sqrt(S) * dW
    
    Uses full-truncation Euler scheme to ensure non-negativity.
    
    Parameters
    ----------
    n : int, default 800
        Number of time steps to simulate.
    dt : float, default 1/252
        Time step size (default is daily for 252 trading days/year).
    s0 : float, default 5.0
        Initial spread value.
    kappa : float, default 3.0
        Mean-reversion speed parameter.
    theta : float, default 5.0
        Long-term mean level.
    sigma : float, default 0.9
        Volatility parameter.
    seed : int or None, default 42
        Random seed for reproducibility. None for random initialization.
    
    Returns
    -------
    pd.Series
        Simulated spread as a pandas Series with integer index.
    
    Examples
    --------
    >>> S = simulate_cir_spread(n=900, dt=1/252, s0=5.2, kappa=2.5, theta=5.0, sigma=1.1, seed=7)
    >>> print(f"Mean: {S.mean():.2f}, Std: {S.std():.2f}")
    """
    rng = np.random.default_rng(seed)
    S = np.empty(n)
    S[0] = s0
    
    for t in range(1, n):
        z = rng.standard_normal()
        s_pos = max(S[t-1], 0.0)  # truncation for sqrt
        S[t] = S[t-1] + kappa * (theta - s_pos) * dt + sigma * np.sqrt(s_pos) * np.sqrt(dt) * z
        S[t] = max(S[t], 0.0)     # enforce non-negativity
    
    return pd.Series(S, name="spread")


def simulate_ou_spread(
    n: int = 800,
    dt: float = 1/252,
    s0: float = 0.0,
    kappa: float = 3.0,
    theta: float = 0.0,
    sigma: float = 0.1,
    seed: Optional[int] = 42
) -> pd.Series:
    """
    Simulate a mean-reverting spread using the Ornstein-Uhlenbeck (OU) process.
    
    The spread follows the OU stochastic differential equation:
        dS = kappa * (theta - S) * dt + sigma * dW
    
    Unlike CIR, the OU process can go negative and has constant volatility.
    
    Parameters
    ----------
    n : int, default 800
        Number of time steps to simulate.
    dt : float, default 1/252
        Time step size (default is daily for 252 trading days/year).
    s0 : float, default 0.0
        Initial spread value.
    kappa : float, default 3.0
        Mean-reversion speed parameter.
    theta : float, default 0.0
        Long-term mean level.
    sigma : float, default 0.1
        Volatility parameter.
    seed : int or None, default 42
        Random seed for reproducibility. None for random initialization.
    
    Returns
    -------
    pd.Series
        Simulated spread as a pandas Series with integer index.
    """
    rng = np.random.default_rng(seed)
    S = np.empty(n)
    S[0] = s0
    
    sqrt_dt = np.sqrt(dt)
    for t in range(1, n):
        z = rng.standard_normal()
        S[t] = S[t-1] + kappa * (theta - S[t-1]) * dt + sigma * sqrt_dt * z
    
    return pd.Series(S, name="spread")
