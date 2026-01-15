from __future__ import annotations
import numpy as np


def simulate_model1_paths(
    theta0: float,
    theta1: float,
    q: float,
    r: float | None,
    n_steps: int,
    n_paths: int,
    x0: float,
    seed: int = 42,
):
    """
    Simulate paths from KF Model 1:

      Latent state:
        x_{t+1} = theta0 + theta1 * x_t + w_t,   w_t ~ N(0, q)

      Observation (optional):
        y_t = x_t + v_t,                        v_t ~ N(0, r)

    If r is None, only latent paths x_t are returned.
    """
    rng = np.random.default_rng(seed)

    # --- latent state paths ---
    x_paths = np.zeros((n_paths, n_steps), dtype=float)
    x_paths[:, 0] = x0

    # state noise std = sqrt(q) (comes directly from KF)
    w = rng.normal(0.0, np.sqrt(q), size=(n_paths, n_steps - 1))

    for t in range(1, n_steps):
        x_paths[:, t] = theta0 + theta1 * x_paths[:, t - 1] + w[:, t - 1]

    # --- observation paths (optional) ---
    if r is None:
        return x_paths

    v = rng.normal(0.0, np.sqrt(r), size=(n_paths, n_steps))
    y_paths = x_paths + v

    return x_paths, y_paths
