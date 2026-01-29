from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Literal, List, Dict, Optional


def _to_numpy(spread: Union[pd.Series, np.ndarray]) -> tuple:
    if isinstance(spread, pd.Series):
        return spread.values.astype(float), spread.index
    return np.asarray(spread, dtype=float), None


def strategy_A_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
    C: Union[float, np.ndarray, pd.Series],
) -> pd.Series:
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
    
    for t in range(n):
        u_t = U_arr[t]
        l_t = L_arr[t]
        c_t = C_arr[t]
        
        if pos == 0:
            if x[t] >= u_t:
                pos = -1
            elif x[t] <= l_t:
                pos = +1
        else:
            if (pos == -1 and x[t] <= c_t) or (pos == +1 and x[t] >= c_t):
                pos = 0
        
        sig[t] = pos
    
    if idx is None:
        return pd.Series(sig, name="signal")
    return pd.Series(sig, index=idx, name="signal")


def strategy_B_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
) -> pd.Series:
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
        
        u_curr = U_arr[t]
        l_curr = L_arr[t]
        
        u_prev = U_arr[t-1]
        l_prev = L_arr[t-1]
        
        if prev < u_prev and curr >= u_curr:
            pos = -1
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
        
        entry_short = (prev > u_prev) and (curr <= u_curr)
        entry_long = (prev < l_prev) and (curr >= l_curr)
        
        cross_down_C = (prev > c_prev) and (curr <= c_curr)
        cross_up_C = (prev < c_prev) and (curr >= c_curr)
        
        stop_short = (prev < u_prev) and (curr >= u_curr)
        stop_long = (prev > l_prev) and (curr <= l_curr)
        
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
    if isinstance(spread, pd.Series):
        idx = spread.index
        x = spread.to_numpy(dtype=float)
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

    n = len(x)
    if len(U_arr) != n or len(L_arr) != n:
        raise ValueError(
            f"Length mismatch: len(spread)={n}, len(U_t)={len(U_arr)}, len(L_t)={len(L_arr)}"
        )
    if not np.isfinite(x).all():
        raise ValueError("spread contains NaN/inf")
    if not np.isfinite(U_arr).all() or not np.isfinite(L_arr).all():
        raise ValueError("U_t or L_t contains NaN/inf after alignment/reindexing")

    if np.any(U_arr <= L_arr):
        raise ValueError("Found U_t <= L_t for some t; boundaries must satisfy U_t > L_t.")

    sig = np.zeros(n, dtype=np.int8)
    pos: int = 0

    for t in range(1, n):
        prev_x, curr_x = x[t - 1], x[t]
        prev_U, curr_U = U_arr[t - 1], U_arr[t]
        prev_L, curr_L = L_arr[t - 1], L_arr[t]

        enter_short = (prev_x > prev_U) and (curr_x <= curr_U)
        enter_long  = (prev_x < prev_L) and (curr_x >= curr_L)

        cross_down_C = (prev_x > C) and (curr_x <= C)
        cross_up_C   = (prev_x < C) and (curr_x >= C)

        stop_short = (prev_x < prev_U) and (curr_x >= curr_U)
        stop_long  = (prev_x > prev_L) and (curr_x <= curr_L)

        if pos == 0:
            if enter_short:
                pos = -1
            elif enter_long:
                pos = +1

        elif pos == -1:
            if cross_down_C or stop_short:
                pos = 0

        else:
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
    return signals * position_size


def find_trades(signals: pd.Series) -> List[Dict]:
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
    trades = []
    entry_idx = None
    trade_type = None
    prev_sig = 0
    
    for t in range(1, len(signals)):
        curr_sig = signals.iloc[t]
        
        if curr_sig != prev_sig:
            if entry_idx is not None:
                trades.append({'entry': entry_idx, 'exit': signals.index[t], 'type': trade_type})
            
            if curr_sig != 0:
                entry_idx = signals.index[t]
                trade_type = 'short' if curr_sig == -1 else 'long'
            else:
                entry_idx = None
        
        prev_sig = curr_sig
    
    return trades


def _as_array(x: Union[float, np.ndarray, pd.Series], n: int) -> np.ndarray:
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
    C_arr = _as_array(C, n)
    U_arr = _as_array(U, n)
    Delta = delta * U_arr
    return C_arr - Delta, C_arr + Delta


def strategy_D_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
    *,
    C_minus: Optional[Union[float, np.ndarray, pd.Series]] = None,
    C_plus: Optional[Union[float, np.ndarray, pd.Series]] = None,
    C: Optional[Union[float, np.ndarray, pd.Series]] = None,
    delta: Optional[float] = None,
    exit_on_crossing: bool = True,
) -> pd.Series:
    x, idx = _to_numpy(spread)
    n = len(x)

    U_arr = _as_array(U, n)
    L_arr = _as_array(L, n)

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
            if x[t] >= u_t:
                pos = -1
            elif x[t] <= l_t:
                pos = +1

        else:
            if exit_on_crossing and t > 0:
                cm_prev, cp_prev = C_minus_arr[t - 1], C_plus_arr[t - 1]
                prev, curr = x[t - 1], x[t]

                if pos == -1 and (prev > cm_prev) and (curr <= cm_t):
                    pos = 0
                elif pos == +1 and (prev < cp_prev) and (curr >= cp_t):
                    pos = 0
            else:
                if (pos == -1 and x[t] <= cm_t) or (pos == +1 and x[t] >= cp_t):
                    pos = 0

        sig[t] = pos

    return pd.Series(sig, index=idx, name="signal") if idx is not None else pd.Series(sig, name="signal")


def strategy_E_signals(
    spread: Union[pd.Series, np.ndarray],
    U: Union[float, np.ndarray, pd.Series],
    L: Union[float, np.ndarray, pd.Series],
    *,
    C_minus: Optional[Union[float, np.ndarray, pd.Series]] = None,
    C_plus: Optional[Union[float, np.ndarray, pd.Series]] = None,
    C: Optional[Union[float, np.ndarray, pd.Series]] = None,
    delta: Optional[float] = None,
) -> pd.Series:
    x, idx = _to_numpy(spread)
    n = len(x)

    U_arr = _as_array(U, n)
    L_arr = _as_array(L, n)

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

        entry_short = (prev > u_prev) and (curr <= u_curr)
        entry_long  = (prev < l_prev) and (curr >= l_curr)

        tp_short = (prev > cm_prev) and (curr <= cm_curr)
        tp_long  = (prev < cp_prev) and (curr >= cp_curr)

        stop_short = (prev < u_prev) and (curr >= u_curr)
        stop_long  = (prev > l_prev) and (curr <= l_curr)

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

    if close_open_end and in_trade:
        trades.append({'entry': entry_idx, 'exit': signals.index[-1], 'type': trade_type})

    return trades
