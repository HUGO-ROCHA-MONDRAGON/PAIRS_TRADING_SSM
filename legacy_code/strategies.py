# strategies.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_numpy_and_index(spread) -> tuple[np.ndarray, pd.Index | None]:
    """
    Accepts pd.Series or array-like.
    Returns (x_numpy, index_or_None).
    """
    if isinstance(spread, pd.Series):
        return spread.values.astype(float), spread.index
    return np.asarray(spread, dtype=float), None


def strategy_A_signals(spread: pd.Series, U: float, L: float, C: float) -> pd.Series:
    """
    Strategy A:
    - If spread >= U: open short spread -> signal = -1
    - If spread <= L: open long spread  -> signal = +1
    - Close when spread crosses mean C -> signal = 0
    Hold until close condition.
    """
    x, idx = _to_numpy_and_index(spread)
    sig = np.zeros_like(x, dtype=np.int8)

    pos = 0
    for t in range(x.shape[0]):
        if pos == 0:
            if x[t] >= U:
                pos = -1
            elif x[t] <= L:
                pos = +1
        else:
            if (pos == -1 and x[t] <= C) or (pos == +1 and x[t] >= C):
                pos = 0
        sig[t] = pos

    if idx is None:
        return pd.Series(sig, name="signal")
    return pd.Series(sig, index=idx, name="signal")


def strategy_B_signals(spread: pd.Series, U: float, L: float) -> pd.Series:
    """
    Strategy B:
    - Enter short when crossing up through U
    - Enter long  when crossing down through L
    - Hold until the next crossing triggers change (can flip).
    """
    x, idx = _to_numpy_and_index(spread)
    sig = np.zeros_like(x, dtype=np.int8)

    pos = 0
    for t in range(1, x.shape[0]):
        prev, cur = x[t - 1], x[t]
        if prev < U and cur >= U:
            pos = -1
        elif prev > L and cur <= L:
            pos = +1
        sig[t] = pos

    if idx is None:
        return pd.Series(sig, name="signal")
    return pd.Series(sig, index=idx, name="signal")

def strategy_C_signals(spread: pd.Series, U: float, L: float, C: float) -> pd.Series:
    """
    Strategy C (paper):
    Entry = re-enter the band:
      - short if crosses U from ABOVE: prev > U and cur <= U
      - long  if crosses L from BELOW: prev < L and cur >= L

    Exit:
      - at mean C (take profit)
      - OR stop if it crosses the boundary the wrong way after entry:
          short stops if crosses U from BELOW (prev < U and cur >= U)
          long  stops if crosses L from ABOVE (prev > L and cur <= L)
    """
    x, idx = _to_numpy_and_index(spread)
    sig = np.zeros_like(x, dtype=np.int8)

    pos = 0
    for t in range(1, x.shape[0]):
        prev, cur = x[t - 1], x[t]

        entry_short = (prev > U and cur <= U)
        entry_long  = (prev < L and cur >= L)

        cross_down_C = (prev > C and cur <= C)
        cross_up_C   = (prev < C and cur >= C)

        stop_short = (prev < U and cur >= U)
        stop_long  = (prev > L and cur <= L)

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
