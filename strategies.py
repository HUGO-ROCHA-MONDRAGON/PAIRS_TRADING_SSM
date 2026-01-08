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

def strategy_C_signals(spread: pd.Series, U: float, L: float, C: float) -> pd.Series:
    """
    Strategy C (band + mean):
    - Entry is on boundary crossings (same as Strategy B):
        * cross up through U  -> enter short spread (-1)
        * cross down through L -> enter long spread (+1)
    - Exit rules:
        * if long (+1): exit when spread crosses up through mean C
                        OR flip to short if spread crosses up through U
        * if short (-1): exit when spread crosses down through mean C
                         OR flip to long if spread crosses down through L

    This produces shorter holding periods than B and usually lower drawdowns.
    """
    x = spread.values
    sig = np.zeros_like(x, dtype=int)

    pos = 0
    for t in range(1, len(x)):
        prev, cur = x[t-1], x[t]

        # helper booleans: crossings
        cross_up_U = (prev < U and cur >= U)
        cross_down_L = (prev > L and cur <= L)
        cross_up_C = (prev < C and cur >= C)
        cross_down_C = (prev > C and cur <= C)

        if pos == 0:
            # enter only on boundary crossings
            if cross_up_U:
                pos = -1
            elif cross_down_L:
                pos = +1

        elif pos == +1:
            # long spread: take profit at mean, or flip if it runs to U
            if cross_up_C:
                pos = 0
            elif cross_up_U:
                pos = -1  # close long and reverse to short

        elif pos == -1:
            # short spread: take profit at mean, or flip if it runs to L
            if cross_down_C:
                pos = 0
            elif cross_down_L:
                pos = +1  # close short and reverse to long

        sig[t] = pos

    sig[0] = 0
    return pd.Series(sig, index=spread.index, name="signal")