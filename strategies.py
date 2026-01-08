# strategies.py
import numpy as np
import pandas as pd

def strategy_A_signals(spread: pd.Series, U: float, L: float, C: float) -> pd.Series:
    """
    Strategy A:
    - If spread >= U: open short spread (sell A, buy gamma*B) -> signal = -1
    - If spread <= L: open long spread (buy A, sell gamma*B)  -> signal = +1
    - Close when spread crosses mean C -> signal goes to 0
    Position is held until close condition.
    """
    x = spread.values
    sig = np.zeros_like(x, dtype=int)

    pos = 0  # -1 short spread, +1 long spread, 0 flat
    for t in range(len(x)):
        if pos == 0:
            if x[t] >= U:
                pos = -1
            elif x[t] <= L:
                pos = +1
        else:
            # close at mean
            if (pos == -1 and x[t] <= C) or (pos == +1 and x[t] >= C):
                pos = 0
        sig[t] = pos

    return pd.Series(sig, index=spread.index, name="signal")

def strategy_B_signals(spread: pd.Series, U: float, L: float) -> pd.Series:
    """
    Strategy B:
    - Enter short spread when spread crosses U from below  -> signal = -1
    - Enter long spread  when spread crosses L from above  -> signal = +1
    - Hold the position until the spread crosses the opposite boundary,
      then flip (close + open simultaneously).
    No explicit close at the mean.
    """
    x = spread.values
    sig = np.zeros_like(x, dtype=int)

    pos = 0
    for t in range(1, len(x)):
        prev, cur = x[t-1], x[t]

        # crossing up through U => short spread
        if prev < U and cur >= U:
            pos = -1

        # crossing down through L => long spread
        elif prev > L and cur <= L:
            pos = +1

        sig[t] = pos

    sig[0] = 0
    return pd.Series(sig, index=spread.index, name="signal")

