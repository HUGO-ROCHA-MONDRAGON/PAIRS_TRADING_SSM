# simulation.py
from __future__ import annotations
import numpy as np
import pandas as pd

def fit_ar1(x: pd.Series) -> tuple[float, float, float]:
    """
    Fit AR(1): x_t = c + phi * x_{t-1} + eps_t
    Returns (c, phi, sigma) where sigma is std(eps).
    """
    x = x.dropna()
    y = x.iloc[1:].values
    X = np.column_stack([np.ones(len(y)), x.iloc[:-1].values])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    c, phi = float(beta[0]), float(beta[1])

    resid = y - (c + phi * X[:, 1])
    sigma = float(np.std(resid, ddof=1))
    return c, phi, sigma

def simulate_ar1_paths(
    c: float,
    phi: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    x0: float,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate AR(1) paths. Returns array shape (n_paths, n_steps).
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_steps), dtype=float)
    paths[:, 0] = x0

    eps = rng.normal(0.0, sigma, size=(n_paths, n_steps-1))
    for t in range(1, n_steps):
        paths[:, t] = c + phi * paths[:, t-1] + eps[:, t-1]
    return paths
