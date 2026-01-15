"""
Trading strategy implementations from Zhang (2021).

Three benchmark strategies:
- Strategy A: Enter at boundary, exit at mean
- Strategy B: Enter on boundary crossing, exit on opposite crossing  
- Strategy C: Enter on re-entry, exit at mean or stop-loss
"""

import numpy as np
import pandas as pd
from typing import Union, Literal


def _to_numpy(spread: Union[pd.Series, np.ndarray]) -> tuple:
    """Convert spread to numpy array and extract index if available."""
    if isinstance(spread, pd.Series):
        return spread.values.astype(float), spread.index
    return np.asarray(spread, dtype=float), None


def strategy_A_signals(
    spread: Union[pd.Series, np.ndarray],
    U: float,
    L: float,
    C: float,
) -> pd.Series:
    """
    Strategy A: Boundary entry, mean exit.
    
    Rules:
    - Enter SHORT (sell spread) when spread >= U
    - Enter LONG (buy spread) when spread <= L
    - Exit (close position) when spread crosses mean C
    - Hold until exit condition is met
    
    Parameters
    ----------
    spread : array-like
        Spread series (observed or filtered)
    U : float
        Upper threshold (short entry)
    L : float
        Lower threshold (long entry)
    C : float
        Mean/center (exit point)
        
    Returns
    -------
    pd.Series
        Signal series: +1 (long), -1 (short), 0 (flat)
    """
    x, idx = _to_numpy(spread)
    n = len(x)
    sig = np.zeros(n, dtype=np.int8)
    
    pos = 0  # Current position
    
    for t in range(n):
        if pos == 0:
            # No position - check for entry
            if x[t] >= U:
                pos = -1  # Short
            elif x[t] <= L:
                pos = +1  # Long
        else:
            # Have position - check for exit
            if (pos == -1 and x[t] <= C) or (pos == +1 and x[t] >= C):
                pos = 0  # Close
        
        sig[t] = pos
    
    if idx is None:
        return pd.Series(sig, name="signal")
    return pd.Series(sig, index=idx, name="signal")


def strategy_B_signals(
    spread: Union[pd.Series, np.ndarray],
    U: float,
    L: float,
) -> pd.Series:
    """
    Strategy B: Boundary crossing entry/exit.
    
    Rules:
    - Enter SHORT when spread crosses UP through U (prev < U, curr >= U)
    - Enter LONG when spread crosses DOWN through L (prev > L, curr <= L)
    - Can flip positions directly (no mean exit)
    
    Parameters
    ----------
    spread : array-like
        Spread series
    U : float
        Upper threshold
    L : float
        Lower threshold
        
    Returns
    -------
    pd.Series
        Signal series
    """
    x, idx = _to_numpy(spread)
    n = len(x)
    sig = np.zeros(n, dtype=np.int8)
    
    pos = 0
    
    for t in range(1, n):
        prev, curr = x[t - 1], x[t]
        
        # Cross up through U -> short
        if prev < U and curr >= U:
            pos = -1
        # Cross down through L -> long
        elif prev > L and curr <= L:
            pos = +1
        
        sig[t] = pos
    
    if idx is None:
        return pd.Series(sig, name="signal")
    return pd.Series(sig, index=idx, name="signal")


def strategy_C_signals(
    spread: Union[pd.Series, np.ndarray],
    U: float,
    L: float,
    C: float,
) -> pd.Series:
    """
    Strategy C: Re-entry with stop-loss (paper's main strategy).
    
    Entry rules:
    - Enter SHORT when spread re-enters from above U (prev > U, curr <= U)
    - Enter LONG when spread re-enters from below L (prev < L, curr >= L)
    
    Exit rules:
    - Take profit at mean C
    - Stop-loss if spread crosses boundary wrong way after entry
    
    Parameters
    ----------
    spread : array-like
        Spread series
    U : float
        Upper threshold
    L : float
        Lower threshold
    C : float
        Mean (take-profit level)
        
    Returns
    -------
    pd.Series
        Signal series
    """
    x, idx = _to_numpy(spread)
    n = len(x)
    sig = np.zeros(n, dtype=np.int8)
    
    pos = 0
    
    for t in range(1, n):
        prev, curr = x[t - 1], x[t]
        
        # Entry signals
        entry_short = (prev > U) and (curr <= U)  # Re-enter from above
        entry_long = (prev < L) and (curr >= L)   # Re-enter from below
        
        # Exit at mean
        cross_down_C = (prev > C) and (curr <= C)
        cross_up_C = (prev < C) and (curr >= C)
        
        # Stop-loss: wrong-way crossing after entry
        stop_short = (prev < U) and (curr >= U)  # Breaks out again
        stop_long = (prev > L) and (curr <= L)   # Breaks down again
        
        if pos == 0:
            if entry_short:
                pos = -1
            elif entry_long:
                pos = +1
        elif pos == -1:
            if cross_down_C or stop_short:
                pos = 0
        elif pos == +1:
            if cross_up_C or stop_long:
                pos = 0
        
        sig[t] = pos
    
    if idx is None:
        return pd.Series(sig, name="signal")
    return pd.Series(sig, index=idx, name="signal")


def generate_signals(
    spread: Union[pd.Series, np.ndarray],
    U: float,
    L: float,
    C: float,
    strategy: Literal["A", "B", "C"] = "C",
) -> pd.Series:
    """
    Generate trading signals for specified strategy.
    
    Parameters
    ----------
    spread : array-like
        Spread series
    U : float
        Upper threshold
    L : float
        Lower threshold
    C : float
        Mean/center
    strategy : str
        Strategy type: "A", "B", or "C"
        
    Returns
    -------
    pd.Series
        Signal series
    """
    strategy = strategy.upper()
    
    if strategy == "A":
        return strategy_A_signals(spread, U, L, C)
    elif strategy == "B":
        return strategy_B_signals(spread, U, L)
    elif strategy == "C":
        return strategy_C_signals(spread, U, L, C)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'A', 'B', or 'C'.")


def signals_to_positions(
    signals: pd.Series,
    position_size: float = 1.0,
) -> pd.Series:
    """
    Convert signals to position sizes.
    
    Parameters
    ----------
    signals : pd.Series
        Signal series (+1, -1, 0)
    position_size : float
        Base position size
        
    Returns
    -------
    pd.Series
        Position series
    """
    return signals * position_size
