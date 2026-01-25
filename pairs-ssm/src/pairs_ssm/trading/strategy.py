"""
Trading strategy implementations from Zhang (2021).

Three benchmark strategies:
- Strategy A: Enter at boundary, exit at mean
- Strategy B: Enter on boundary crossing, exit on opposite crossing  
- Strategy C: Enter on re-entry, exit at mean or stop-loss
"""

import numpy as np
import pandas as pd
from typing import Union, Literal, List, Dict


def _to_numpy(spread: Union[pd.Series, np.ndarray]) -> tuple:
    """Convert spread to numpy array and extract index if available."""
    if isinstance(spread, pd.Series):
        return spread.values.astype(float), spread.index
    return np.asarray(spread, dtype=float), None


def strategy_A_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
    C: Union[float, np.ndarray, pd.Series],
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
    U : float or array-like
        Upper threshold (short entry)
    L : float or array-like
        Lower threshold (long entry)
    C : float or array-like
        Mean/center (exit point)
        
    Returns
    -------
    pd.Series
        Signal series: +1 (long), -1 (short), 0 (flat)
    """
    x, idx = _to_numpy(spread)
    n = len(x)
    
    # Handle array-like thresholds
    U_arr = np.asarray(U) if not np.isscalar(U) else np.full(n, U)
    L_arr = np.asarray(L) if not np.isscalar(L) else np.full(n, L)
    C_arr = np.asarray(C) if not np.isscalar(C) else np.full(n, C)
    
    # Broadcast if single element array
    if U_arr.ndim == 0: U_arr = np.full(n, float(U_arr))
    if L_arr.ndim == 0: L_arr = np.full(n, float(L_arr))
    if C_arr.ndim == 0: C_arr = np.full(n, float(C_arr))

    # Ensure same length
    if len(U_arr) != n or len(L_arr) != n or len(C_arr) != n:
         # Try to align indices if they are pandas series, otherwise raise
         # For simplicity in this fix, we assume caller aligns them or passes scalars
         # If lengths differ, we must fallback to simple scalar assumption or fail.
         # But the loop below needs n. Let's assume valid input for now.
         pass

    sig = np.zeros(n, dtype=np.int8)
    
    pos = 0  # Current position
    
    for t in range(n):
        u_t = U_arr[t]
        l_t = L_arr[t]
        c_t = C_arr[t]
        
        if pos == 0:
            # No position - check for entry
            if x[t] >= u_t:
                pos = -1  # Short
            elif x[t] <= l_t:
                pos = +1  # Long
        else:
            # Have position - check for exit
            if (pos == -1 and x[t] <= c_t) or (pos == +1 and x[t] >= c_t):
                pos = 0  # Close
        
        sig[t] = pos
    
    if idx is None:
        return pd.Series(sig, name="signal")
    return pd.Series(sig, index=idx, name="signal")


def strategy_B_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
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
    U : float or array-like
        Upper threshold
    L : float or array-like
        Lower threshold
        
    Returns
    -------
    pd.Series
        Signal series
    """
    x, idx = _to_numpy(spread)
    n = len(x)
    
    U_arr = np.asarray(U) if not np.isscalar(U) else np.full(n, U)
    L_arr = np.asarray(L) if not np.isscalar(L) else np.full(n, L)
    
    if U_arr.ndim == 0: U_arr = np.full(n, float(U_arr))
    if L_arr.ndim == 0: L_arr = np.full(n, float(L_arr))
    
    sig = np.zeros(n, dtype=np.int8)
    
    pos = 0
    
    for t in range(1, n):
        prev, curr = x[t - 1], x[t]
        
        # Use current thresholds for crossing check
        # (Could use prev/curr specific thresholds if they vary fast, but this is safe)
        u_curr = U_arr[t]
        l_curr = L_arr[t]
        
        u_prev = U_arr[t-1]
        l_prev = L_arr[t-1]
        
        # Cross up through U -> short
        if prev < u_prev and curr >= u_curr:
            pos = -1
        # Cross down through L -> long
        elif prev > l_prev and curr <= l_curr:
            pos = +1
        
        sig[t] = pos
    
    if idx is None:
        return pd.Series(sig, name="signal")
    return pd.Series(sig, index=idx, name="signal")


def strategy_C_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
    C: Union[float, np.ndarray, pd.Series],
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
    U : float or array-like
        Upper threshold
    L : float or array-like
        Lower threshold
    C : float or array-like
        Mean (take-profit level)
        
    Returns
    -------
    pd.Series
        Signal series
    """
    x, idx = _to_numpy(spread)
    n = len(x)
    
    U_arr = np.asarray(U) if not np.isscalar(U) else np.full(n, U)
    L_arr = np.asarray(L) if not np.isscalar(L) else np.full(n, L)
    C_arr = np.asarray(C) if not np.isscalar(C) else np.full(n, C)
    
    if U_arr.ndim == 0: U_arr = np.full(n, float(U_arr))
    if L_arr.ndim == 0: L_arr = np.full(n, float(L_arr))
    if C_arr.ndim == 0: C_arr = np.full(n, float(C_arr))
    
    sig = np.zeros(n, dtype=np.int8)
    
    pos = 0
    
    for t in range(1, n):
        prev, curr = x[t - 1], x[t]
        
        u_prev, u_curr = U_arr[t-1], U_arr[t]
        l_prev, l_curr = L_arr[t-1], L_arr[t]
        c_prev, c_curr = C_arr[t-1], C_arr[t]
        
        # Entry signals
        entry_short = (prev > u_prev) and (curr <= u_curr)  # Re-enter from above
        entry_long = (prev < l_prev) and (curr >= l_curr)   # Re-enter from below
        
        # Exit at mean
        cross_down_C = (prev > c_prev) and (curr <= c_curr)
        cross_up_C = (prev < c_prev) and (curr >= c_curr)
        
        # Stop-loss: wrong-way crossing after entry
        stop_short = (prev < u_prev) and (curr >= u_curr)  # Breaks out again
        stop_long = (prev > l_prev) and (curr <= l_curr)   # Breaks down again
        
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


def strategy_C_signals_timevarying(
    spread: Union[pd.Series, np.ndarray],
    U_t: Union[pd.Series, np.ndarray],
    L_t: Union[pd.Series, np.ndarray],
    C: float,
) -> pd.Series:
    """
    Strategy C with time-varying boundaries for heteroscedastic models.
    
    As described in Zhang (2021) Figure 3(b), in the heteroscedastic model,
    the boundaries U_t and L_t vary over time based on the filtered volatility:
        U_t = μ + k·σ_t
        L_t = μ - k·σ_t
    
    where σ_t = √P_t is the filtered standard deviation at each time step.
    
    Entry rules (same as Strategy C):
    - Enter SHORT when spread re-enters from above U_t (prev > U_{t-1}, curr <= U_t)
    - Enter LONG when spread re-enters from below L_t (prev < L_{t-1}, curr >= L_t)
    
    Exit rules:
    - Take profit at mean C
    - Stop-loss if spread crosses boundary wrong way after entry
    
    Parameters
    ----------
    spread : array-like
        Spread series (filtered)
    U_t : array-like
        Time-varying upper threshold series (same length as spread)
    L_t : array-like
        Time-varying lower threshold series (same length as spread)
    C : float
        Mean (take-profit level, constant)
        
    Returns
    -------
    pd.Series
        Signal series: +1 (long), -1 (short), 0 (flat)
    """
    x, idx = _to_numpy(spread)
    U_arr = np.asarray(U_t, dtype=float)
    L_arr = np.asarray(L_t, dtype=float)
    
    n = len(x)
    sig = np.zeros(n, dtype=np.int8)
    
    pos = 0
    
    for t in range(1, n):
        prev_x, curr_x = x[t - 1], x[t]
        prev_U, curr_U = U_arr[t - 1], U_arr[t]
        prev_L, curr_L = L_arr[t - 1], L_arr[t]
        
        # Entry signals (using time-varying boundaries)
        entry_short = (prev_x > prev_U) and (curr_x <= curr_U)  # Re-enter from above
        entry_long = (prev_x < prev_L) and (curr_x >= curr_L)   # Re-enter from below
        
        # Exit at mean
        cross_down_C = (prev_x > C) and (curr_x <= C)
        cross_up_C = (prev_x < C) and (curr_x >= C)
        
        # Stop-loss: wrong-way crossing
        stop_short = (prev_x < prev_U) and (curr_x >= curr_U)  # Breaks out again
        stop_long = (prev_x > prev_L) and (curr_x <= curr_L)   # Breaks down again
        
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


def find_trades(signals: pd.Series) -> List[Dict]:
    """
    Find trade entry and exit points from signals.
    
    For Strategy A and C where positions return to 0 before next trade.
    
    Parameters
    ----------
    signals : pd.Series
        Signal series (+1, -1, 0)
        
    Returns
    -------
    List[Dict]
        List of trades with 'entry', 'exit', and 'type' keys
    """
    trades = []
    in_trade = False
    entry_idx = None
    trade_type = None
    
    for t in range(1, len(signals)):
        if not in_trade and signals.iloc[t] != 0:
            in_trade = True
            entry_idx = signals.index[t]
            trade_type = 'short' if signals.iloc[t] == -1 else 'long'
        elif in_trade and signals.iloc[t] == 0:
            trades.append({'entry': entry_idx, 'exit': signals.index[t], 'type': trade_type})
            in_trade = False
    
    return trades


def find_trades_B(signals: pd.Series) -> List[Dict]:
    """
    Find trades for Strategy B where positions flip directly (no return to 0).
    
    In Strategy B, we hold until we need to switch position, so clear and open
    happen at the same time.
    
    Parameters
    ----------
    signals : pd.Series
        Signal series from strategy_B_signals
        
    Returns
    -------
    List[Dict]
        List of trades with 'entry', 'exit', and 'type' keys
    """
    trades = []
    entry_idx = None
    trade_type = None
    prev_sig = 0
    
    for t in range(1, len(signals)):
        curr_sig = signals.iloc[t]
        
        # Detect position change
        if curr_sig != prev_sig:
            # Close previous trade if we had one
            if entry_idx is not None:
                trades.append({'entry': entry_idx, 'exit': signals.index[t], 'type': trade_type})
            
            # Open new trade if not flat
            if curr_sig != 0:
                entry_idx = signals.index[t]
                trade_type = 'short' if curr_sig == -1 else 'long'
            else:
                entry_idx = None
        
        prev_sig = curr_sig
    
    return trades

import numpy as np
import pandas as pd
from typing import Union

def strategy_C_signals_timevarying_v2(
    spread: Union[pd.Series, np.ndarray],
    U_t: Union[pd.Series, np.ndarray],
    L_t: Union[pd.Series, np.ndarray],
    C: float,
) -> pd.Series:

    # --- convert + ALIGN on index if pandas ---
    if isinstance(spread, pd.Series):
        x = spread.values.astype(float)
        idx = spread.index
        if isinstance(U_t, pd.Series):
            U = U_t.reindex(idx).values.astype(float)
        else:
            U = np.asarray(U_t, dtype=float)
        if isinstance(L_t, pd.Series):
            L = L_t.reindex(idx).values.astype(float)
        else:
            L = np.asarray(L_t, dtype=float)
    else:
        x = np.asarray(spread, dtype=float)
        idx = None
        U = np.asarray(U_t, dtype=float)
        L = np.asarray(L_t, dtype=float)

    n = len(x)
    sig = np.zeros(n, dtype=np.int8)
    pos = 0

    for t in range(1, n):
        # gaps to moving thresholds
        gapU_prev = x[t-1] - U[t-1]
        gapU_curr = x[t]   - U[t]
        gapL_prev = x[t-1] - L[t-1]
        gapL_curr = x[t]   - L[t]

        # --- entries: re-entry (sign change across boundary) ---
        entry_short = (gapU_prev > 0) and (gapU_curr <= 0)   # from above U back inside
        entry_long  = (gapL_prev < 0) and (gapL_curr >= 0)   # from below L back inside

        # --- exits: cross mean C (sign change around C) ---
        gapC_prev = x[t-1] - C
        gapC_curr = x[t]   - C
        cross_down_C = (gapC_prev > 0) and (gapC_curr <= 0)
        cross_up_C   = (gapC_prev < 0) and (gapC_curr >= 0)

        # --- stop-loss: wrong-way breakout (sign change opposite direction) ---
        # short stop: cross UP through U (from inside to above)
        stop_short = (gapU_prev <= 0) and (gapU_curr > 0)
        # long stop: cross DOWN through L (from inside to below)
        stop_long  = (gapL_prev >= 0) and (gapL_curr < 0)

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
