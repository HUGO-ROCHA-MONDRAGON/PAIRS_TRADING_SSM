"""
Trading strategy implementations from Zhang (2021).

Three benchmark strategies:
- Strategy A: Enter at boundary, exit at mean
- Strategy B: Enter on boundary crossing, exit on opposite crossing  
- Strategy C: Enter on re-entry, exit at mean or stop-loss
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Literal, List, Dict, Optional


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
    Strategy C with time-varying boundaries (heteroskedastic illustration).

    Positions:
      +1 = long spread (Long A Short B)
      -1 = short spread (Short A Long B)
       0 = flat

    Entry (re-entry / crossing rules):
      - Enter SHORT when spread crosses DOWN through the upper boundary:
            x_{t-1} > U_{t-1}  and  x_t <= U_t
      - Enter LONG when spread crosses UP through the lower boundary:
            x_{t-1} < L_{t-1}  and  x_t >= L_t

    Exit:
      - Take-profit when spread crosses the mean C (from either side, depending on position)
      - Stop-loss when spread crosses the boundary "wrong way" after entry:
            SHORT stop: x_{t-1} < U_{t-1} and x_t >= U_t
            LONG  stop: x_{t-1} > L_{t-1} and x_t <= L_t

    Notes:
      - Requires U_t and L_t aligned in length with spread.
      - For pd.Series inputs, indices must match exactly.

    Returns
    -------
    pd.Series
        Signal series (+1, -1, 0) with same index as `spread` if it is a Series.
    """
    # ---- Convert inputs ----
    if isinstance(spread, pd.Series):
        idx = spread.index
        x = spread.to_numpy(dtype=float)
        # Align boundaries by index if they are Series
        if isinstance(U_t, pd.Series):
            U_arr = U_t.reindex(idx).to_numpy(dtype=float)
        else:
            U_arr = np.asarray(U_t, dtype=float)

        if isinstance(L_t, pd.Series):
            L_arr = L_t.reindex(idx).to_numpy(dtype=float)
        else:
            L_arr = np.asarray(L_t, dtype=float)
    else:
        idx = None
        x = np.asarray(spread, dtype=float)
        U_arr = np.asarray(U_t, dtype=float)
        L_arr = np.asarray(L_t, dtype=float)

    # ---- Sanity checks ----
    n = len(x)
    if len(U_arr) != n or len(L_arr) != n:
        raise ValueError(
            f"Length mismatch: len(spread)={n}, len(U_t)={len(U_arr)}, len(L_t)={len(L_arr)}"
        )
    if not np.isfinite(x).all():
        raise ValueError("spread contains NaN/inf")
    if not np.isfinite(U_arr).all() or not np.isfinite(L_arr).all():
        raise ValueError("U_t or L_t contains NaN/inf after alignment/reindexing")

    # Optional but helpful: ensure U is above L most of the time
    if np.any(U_arr <= L_arr):
        raise ValueError("Found U_t <= L_t for some t; boundaries must satisfy U_t > L_t.")

    # ---- Core loop ----
    sig = np.zeros(n, dtype=np.int8)
    pos: int = 0  # current position

    for t in range(1, n):
        prev_x, curr_x = x[t - 1], x[t]
        prev_U, curr_U = U_arr[t - 1], U_arr[t]
        prev_L, curr_L = L_arr[t - 1], L_arr[t]

        # Entry crossings (re-entry)
        enter_short = (prev_x > prev_U) and (curr_x <= curr_U)
        enter_long  = (prev_x < prev_L) and (curr_x >= curr_L)

        # Mean crossings for take-profit
        cross_down_C = (prev_x > C) and (curr_x <= C)
        cross_up_C   = (prev_x < C) and (curr_x >= C)

        # Boundary crossings for stop-loss (wrong-way)
        stop_short = (prev_x < prev_U) and (curr_x >= curr_U)  # crosses up through U
        stop_long  = (prev_x > prev_L) and (curr_x <= curr_L)  # crosses down through L

        if pos == 0:
            if enter_short:
                pos = -1
            elif enter_long:
                pos = +1

        elif pos == -1:
            # short spread: profit when crossing down through mean, stop when breaking up through U
            if cross_down_C or stop_short:
                pos = 0

        else:  # pos == +1
            # long spread: profit when crossing up through mean, stop when breaking down through L
            if cross_up_C or stop_long:
                pos = 0

        sig[t] = pos

    return pd.Series(sig, index=idx, name="signal") if idx is not None else pd.Series(sig, name="signal")

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


def _as_array(x: Union[float, np.ndarray, pd.Series], n: int) -> np.ndarray:
    """Scalar -> (n,), Series -> values, ndarray -> array; best-effort broadcast."""
    if isinstance(x, pd.Series):
        arr = x.values.astype(float)
    else:
        arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        return np.full(n, float(arr))
    if len(arr) == 1 and n != 1:
        return np.full(n, float(arr[0]))
    return arr


def compute_Cpm(
    C: Union[float, np.ndarray, pd.Series],
    U: Union[float, np.ndarray, pd.Series],
    delta: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute C_minus and C_plus arrays from C, U, delta."""
    C_arr = _as_array(C, n)
    U_arr = _as_array(U, n)
    Delta = delta * U_arr
    return C_arr - Delta, C_arr + Delta


def strategy_D_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
    *,
    # You can pass either (C_minus, C_plus) OR (C and delta)
    C_minus: Optional[Union[float, np.ndarray, pd.Series]] = None,
    C_plus: Optional[Union[float, np.ndarray, pd.Series]] = None,
    C: Optional[Union[float, np.ndarray, pd.Series]] = None,
    delta: Optional[float] = None,
    exit_on_crossing: bool = True,
) -> pd.Series:
    """
    Strategy D: same entry as Strategy A, but exit at shifted levels:
      - short exits when spread crosses DOWN through C_minus
      - long exits when spread crosses UP through C_plus

    If exit_on_crossing=False, uses level checks (like your Strategy A).
    """
    x, idx = _to_numpy(spread)
    n = len(x)

    U_arr = _as_array(U, n)
    L_arr = _as_array(L, n)

    # Build C_minus / C_plus
    if C_minus is None or C_plus is None:
        if C is None or delta is None:
            raise ValueError("Provide either (C_minus, C_plus) or (C and delta).")
        C_minus_arr, C_plus_arr = compute_Cpm(C=C, U=U, delta=float(delta), n=n)
    else:
        C_minus_arr = _as_array(C_minus, n)
        C_plus_arr = _as_array(C_plus, n)

    sig = np.zeros(n, dtype=np.int8)
    pos = 0

    for t in range(n):
        u_t, l_t = U_arr[t], L_arr[t]
        cm_t, cp_t = C_minus_arr[t], C_plus_arr[t]

        if pos == 0:
            # Entry (same as A)
            if x[t] >= u_t:
                pos = -1
            elif x[t] <= l_t:
                pos = +1

        else:
            # Exit (shifted close)
            if exit_on_crossing and t > 0:
                cm_prev, cp_prev = C_minus_arr[t - 1], C_plus_arr[t - 1]
                prev, curr = x[t - 1], x[t]

                if pos == -1 and (prev > cm_prev) and (curr <= cm_t):
                    pos = 0
                elif pos == +1 and (prev < cp_prev) and (curr >= cp_t):
                    pos = 0
            else:
                # Level-based (A-style)
                if (pos == -1 and x[t] <= cm_t) or (pos == +1 and x[t] >= cp_t):
                    pos = 0

        sig[t] = pos

    return pd.Series(sig, index=idx, name="signal") if idx is not None else pd.Series(sig, name="signal")


def strategy_E_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
    *,
    # You can pass either (C_minus, C_plus) OR (C and delta)
    C_minus: Optional[Union[float, np.ndarray, pd.Series]] = None,
    C_plus: Optional[Union[float, np.ndarray, pd.Series]] = None,
    C: Optional[Union[float, np.ndarray, pd.Series]] = None,
    delta: Optional[float] = None,
) -> pd.Series:
    """
    Strategy E: same entry + stop-loss logic as Strategy C,
    but take-profit uses shifted levels:
      - short TP when cross DOWN through C_minus
      - long  TP when cross UP through C_plus
    Stop-loss remains "wrong-way breakout" as in Strategy C.
    """
    x, idx = _to_numpy(spread)
    n = len(x)

    U_arr = _as_array(U, n)
    L_arr = _as_array(L, n)

    # Build C_minus / C_plus
    if C_minus is None or C_plus is None:
        if C is None or delta is None:
            raise ValueError("Provide either (C_minus, C_plus) or (C and delta).")
        C_minus_arr, C_plus_arr = compute_Cpm(C=C, U=U, delta=float(delta), n=n)
    else:
        C_minus_arr = _as_array(C_minus, n)
        C_plus_arr = _as_array(C_plus, n)

    sig = np.zeros(n, dtype=np.int8)
    pos = 0

    for t in range(1, n):
        prev, curr = x[t - 1], x[t]

        u_prev, u_curr = U_arr[t - 1], U_arr[t]
        l_prev, l_curr = L_arr[t - 1], L_arr[t]

        cm_prev, cm_curr = C_minus_arr[t - 1], C_minus_arr[t]
        cp_prev, cp_curr = C_plus_arr[t - 1], C_plus_arr[t]

        # Entry (same as C)
        entry_short = (prev > u_prev) and (curr <= u_curr)   # re-enter from above U
        entry_long  = (prev < l_prev) and (curr >= l_curr)   # re-enter from below L

        # Take-profit at shifted center
        tp_short = (prev > cm_prev) and (curr <= cm_curr)    # cross down through C_minus
        tp_long  = (prev < cp_prev) and (curr >= cp_curr)    # cross up through C_plus

        # Stop-loss (same as C)
        stop_short = (prev < u_prev) and (curr >= u_curr)    # breaks out again above U
        stop_long  = (prev > l_prev) and (curr <= l_curr)    # breaks down again below L

        if pos == 0:
            if entry_short:
                pos = -1
            elif entry_long:
                pos = +1
        elif pos == -1:
            if tp_short or stop_short:
                pos = 0
        elif pos == +1:
            if tp_long or stop_long:
                pos = 0

        sig[t] = pos

    return pd.Series(sig, index=idx, name="signal") if idx is not None else pd.Series(sig, name="signal")


def find_trades_safe(signals: pd.Series, close_open_end: bool = True):
    trades = []
    in_trade = False
    entry_idx = None
    trade_type = None

    # Handle if we start already in a position at t=0
    if signals.iloc[0] != 0:
        in_trade = True
        entry_idx = signals.index[0]
        trade_type = 'short' if signals.iloc[0] == -1 else 'long'

    for t in range(1, len(signals)):
        if not in_trade and signals.iloc[t] != 0:
            in_trade = True
            entry_idx = signals.index[t]
            trade_type = 'short' if signals.iloc[t] == -1 else 'long'
        elif in_trade and signals.iloc[t] == 0:
            trades.append({'entry': entry_idx, 'exit': signals.index[t], 'type': trade_type})
            in_trade = False

    # Close open trade at the end if requested
    if close_open_end and in_trade:
        trades.append({'entry': entry_idx, 'exit': signals.index[-1], 'type': trade_type})

    return trades
