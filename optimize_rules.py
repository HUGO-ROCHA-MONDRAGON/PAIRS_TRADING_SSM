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

def sharpe_ratio_annualized(pnl_daily: np.ndarray, ann_factor: int = 252) -> float:
    """
    pnl_daily: array shape (T,)
    """
    mu = pnl_daily.mean()
    sd = pnl_daily.std(ddof=1)
    if sd == 0:
        return 0.0
    return (mu / sd) * np.sqrt(ann_factor)


def grid_search_UL(
    paths: np.ndarray,          # shape (N, T)
    strategy: str,
    C: float,
    U_grid: np.ndarray,
    L_grid: np.ndarray,
    objective: str = "SR",      # "SR" or "CR"
    ann_factor: int = 252
) -> dict:
    """
    Paper-faithful grid search:
    - For each (U,L), compute CR^n and SR^n for each path n
    - Average across paths: CR = mean(CR^n), SR = mean(SR^n)
    - Choose argmax of objective (CR or SR)
    """
    best = {"U": None, "L": None, "CR": -np.inf, "SR": -np.inf, "score": -np.inf}

    N, T = paths.shape

    for U in U_grid:
        for L in L_grid:
            if L >= C or U <= C or L >= U:
                continue

            sig = np.zeros((N, T), dtype=int)

            # signals per path
            for i in range(N):
                s = pd.Series(paths[i, :])
                if strategy == "A":
                    sig[i, :] = strategy_A_signals(s, U=U, L=L, C=C).values
                elif strategy == "B":
                    sig[i, :] = strategy_B_signals(s, U=U, L=L).values
                elif strategy == "C":
                    sig[i, :] = strategy_C_signals(s, U=U, L=L, C=C).values
                else:
                    raise ValueError("strategy must be 'A','B', or 'C'")

            # pnl_daily should be shape (N, T)
            pnl_daily = pnl_from_spread_changes(paths, sig)

            if pnl_daily.ndim == 1:
                # if your function returns pooled pnl, we cannot replicate paper exactly
                # but we'll force shape (N, T) requirement
                raise ValueError("pnl_from_spread_changes must return shape (N, T) daily pnl per path")

            # per-path cumulative return (paper CR)
            CR_n = pnl_daily.sum(axis=1)          # shape (N,)

            # per-path Sharpe, then average (paper SR)
            SR_n = np.array([sharpe_ratio_annualized(pnl_daily[i, :], ann_factor=ann_factor) for i in range(N)])

            CR = float(CR_n.mean())
            SR = float(SR_n.mean())

            score = SR if objective.upper() == "SR" else CR

            if score > best["score"]:
                best = {"U": float(U), "L": float(L), "CR": CR, "SR": SR, "score": float(score)}

    return best
