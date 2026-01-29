"""
Data transformation utilities for spread construction.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass

from pairs_ssm.data.loaders import PairData


@dataclass
class SpreadData:
    """Container for spread data and metadata."""
    
    spread: pd.Series          # The spread series
    gamma: float               # Hedge ratio
    method: str                # Estimation method
    use_log: bool              # Whether log prices were used
    pair: PairData             # Original pair data
    
    @property
    def mean(self) -> float:
        """Long-run mean of spread."""
        return float(self.spread.mean())
    
    @property
    def std(self) -> float:
        """Standard deviation of spread."""
        return float(self.spread.std(ddof=1))
    
    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return len(self.spread)


def estimate_gamma(
    PA: pd.Series,
    PB: pd.Series,
    method: str = "ols",
) -> float:
    """
    Estimate the hedge ratio (gamma) between two price series.
    
    The spread is defined as: S = PA - gamma * PB
    
    Parameters
    ----------
    PA : pd.Series
        Price series for asset A
    PB : pd.Series
        Price series for asset B
    method : str
        Estimation method:
        - "ols": Ordinary least squares (PA = alpha + gamma * PB)
        - "tls": Total least squares
        - "kalman": Time-varying via Kalman filter
        
    Returns
    -------
    float
        Estimated hedge ratio
    """
    # Align series
    df = pd.DataFrame({"PA": PA, "PB": PB}).dropna()
    pa = df["PA"].values
    pb = df["PB"].values
    
    if method == "ols":
        # OLS: PA = alpha + gamma * PB
        X = np.column_stack([np.ones(len(pb)), pb])
        beta = np.linalg.lstsq(X, pa, rcond=None)[0]
        gamma = float(beta[1])
        
    elif method == "tls":
        # Total least squares via SVD
        data = np.column_stack([pa, pb])
        data_centered = data - data.mean(axis=0)
        _, _, Vt = np.linalg.svd(data_centered)
        gamma = -Vt[-1, 0] / Vt[-1, 1]
        
    elif method == "kalman":
        # Time-varying Kalman filter estimate (use last value)
        gamma = _kalman_hedge_ratio(pa, pb)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return gamma


def _kalman_hedge_ratio(pa: np.ndarray, pb: np.ndarray) -> float:
    """
    Estimate time-varying hedge ratio using Kalman filter.
    Returns the final estimate.
    """
    n = len(pa)
    
    # State: [alpha, gamma]
    # Observation: PA_t = alpha + gamma * PB_t + noise
    
    # Initialize
    theta = np.array([0.0, 1.0])  # [alpha, gamma]
    P = np.eye(2) * 1.0
    
    # Process and observation noise
    Q = np.eye(2) * 1e-5
    R = np.var(pa) * 0.1
    
    for t in range(n):
        # Observation matrix
        H = np.array([[1.0, pb[t]]])
        
        # Predict
        theta_pred = theta
        P_pred = P + Q
        
        # Update
        y = pa[t]
        y_pred = H @ theta_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S
        
        theta = theta_pred + K.flatten() * (y - y_pred)
        P = (np.eye(2) - np.outer(K.flatten(), H)) @ P_pred
    
    return float(theta[1])  # Return gamma


def compute_spread(
    pair_or_p1: Union[PairData, pd.Series],
    p2: Optional[pd.Series] = None,
    gamma: Optional[float] = None,
    method: str = "ols",
    use_log: bool = True,
) -> SpreadData:
    """
    Compute the spread between two assets.
    
    Spread = log(PA) - gamma * log(PB)  [if use_log=True]
    Spread = PA - gamma * PB            [if use_log=False]
    
    Can be called in two ways:
    - compute_spread(pair_data) with a PairData object
    - compute_spread(log_p1, log_p2) with two Series (log prices)
    
    Parameters
    ----------
    pair_or_p1 : PairData or pd.Series
        Either a PairData object, or the first log price series
    p2 : pd.Series, optional
        Second log price series (only if first arg is a Series)
    gamma : float, optional
        Hedge ratio. If None, estimated from data.
    method : str
        Method for gamma estimation if gamma is None
    use_log : bool
        Whether to use log prices (only applies if PairData provided)
        If two Series provided, they are assumed to already be log prices.
        
    Returns
    -------
    SpreadData
        Spread data container
    """
    # Handle two calling conventions
    if isinstance(pair_or_p1, pd.Series):
        # Called with two Series (assumed to be log prices already)
        if p2 is None:
            raise ValueError("Second price series required when first argument is a Series")
        PA = pair_or_p1
        PB = p2
        pair = None
    else:
        # Called with PairData object
        pair = pair_or_p1
        if use_log:
            PA = np.log(pair.PA) if isinstance(pair.PA, pd.Series) else pd.Series(np.log(pair.PA))
            PB = np.log(pair.PB) if isinstance(pair.PB, pd.Series) else pd.Series(np.log(pair.PB))
        else:
            PA = pair.PA
            PB = pair.PB
    
    # Estimate gamma if not provided
    if gamma is None:
        gamma = estimate_gamma(PA.values if hasattr(PA, 'values') else PA, 
                               PB.values if hasattr(PB, 'values') else PB, 
                               method=method)
    
    # Compute spread
    spread = PA - gamma * PB
    if isinstance(spread, pd.Series):
        spread.name = "spread"
    else:
        spread = pd.Series(spread, name="spread")
    
    return SpreadData(
        spread=spread,
        gamma=gamma,
        method=method,
        use_log=use_log,
        pair=pair,
    )


def normalize_spread(
    spread: pd.Series,
    method: str = "zscore",
    window: Optional[int] = None,
) -> pd.Series:
    """
    Normalize a spread series.
    
    Parameters
    ----------
    spread : pd.Series
        Input spread
    method : str
        Normalization method:
        - "zscore": (x - mean) / std
        - "minmax": (x - min) / (max - min)
        - "rolling_zscore": Rolling window z-score
    window : int, optional
        Rolling window size (required for rolling methods)
        
    Returns
    -------
    pd.Series
        Normalized spread
    """
    if method == "zscore":
        return (spread - spread.mean()) / spread.std()
    
    elif method == "minmax":
        return (spread - spread.min()) / (spread.max() - spread.min())
    
    elif method == "rolling_zscore":
        if window is None:
            raise ValueError("window required for rolling_zscore")
        rolling_mean = spread.rolling(window).mean()
        rolling_std = spread.rolling(window).std()
        return (spread - rolling_mean) / rolling_std
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_returns(
    spread: pd.Series,
    method: str = "simple",
) -> pd.Series:
    """
    Compute returns from spread series.
    
    Parameters
    ----------
    spread : pd.Series
        Spread series
    method : str
        Return calculation method:
        - "simple": (x_t - x_{t-1}) / x_{t-1}
        - "log": log(x_t / x_{t-1})
        - "diff": x_t - x_{t-1}
        
    Returns
    -------
    pd.Series
        Returns series
    """
    if method == "simple":
        return spread.pct_change()
    elif method == "log":
        return np.log(spread / spread.shift(1))
    elif method == "diff":
        return spread.diff()
    else:
        raise ValueError(f"Unknown method: {method}")
