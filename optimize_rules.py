# optimize_rules.py
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies import strategy_A_signals, strategy_B_signals, strategy_C_signals

def pnl_from_spread_changes(spread: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """
    Simplified PnL on simulated spread:
    pnl_t = signal_{t-1} * (spread_t - spread_{t-1})
    This mirrors your pair PnL structure but avoids price-level complications.
    """
    d = np.diff(spread, axis=1)
    sig_lag = signal[:, :-1]  # already aligned with d
    pnl = sig_lag * d
    return pnl  # shape (n_paths, n_steps-1)

def sharpe_ratio(pnl: np.ndarray, eps: float = 1e-12) -> float:
    """
    Cross-path Sharpe proxy: mean(pnl) / std(pnl)
    computed on flattened pnl array.
    """
    x = pnl.reshape(-1)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    return m / (s + eps)

def grid_search_UL(
    paths: np.ndarray,
    strategy: str,
    C: float,
    U_grid: np.ndarray,
    L_grid: np.ndarray,
) -> dict:
    """
    Grid search for optimal (U,L) on simulated paths.
    """
    best = {"U": None, "L": None, "score": -np.inf}

    for U in U_grid:
        for L in L_grid:
            if L >= C or U <= C or L >= U:
                continue

            # build signals for each path
            # Strategy functions expect pd.Series, so we vectorize manually:
            sig = np.zeros_like(paths, dtype=int)

            # run each path through the chosen strategy
            for i in range(paths.shape[0]):
                s = pd.Series(paths[i, :])
                if strategy == "A":
                    sig[i, :] = strategy_A_signals(s, U=U, L=L, C=C).values
                elif strategy == "B":
                    sig[i, :] = strategy_B_signals(s, U=U, L=L).values
                elif strategy == "C":
                    sig[i, :] = strategy_C_signals(s, U=U, L=L, C=C).values
                else:
                    raise ValueError("strategy must be 'A','B', or 'C'")

            pnl = pnl_from_spread_changes(paths, sig)
            score = sharpe_ratio(pnl)

            if score > best["score"]:
                best = {"U": float(U), "L": float(L), "score": float(score)}

    return best
