import numpy as np
import pandas as pd


def simulate_cir_spread(
    n_steps: int = 252,
    dt: float = 1.0,
    kappa: float = 0.5,
    theta: float = 0.0,
    sigma: float = 0.1,
    x0: float = 0.0,
    seed: int | None = None,
) -> pd.Series:
    if seed is not None:
        np.random.seed(seed)
    
    x = np.zeros(n_steps + 1)
    x[0] = x0
    
    for t in range(n_steps):
        dW = np.random.randn() * np.sqrt(dt)
        x[t + 1] = x[t] + kappa * (theta - x[t]) * dt + sigma * np.sqrt(max(x[t] - theta, 0) + 0.01) * dW
    
    return pd.Series(x, name="spread")


def simulate_ou_spread(
    n_steps: int = 252,
    dt: float = 1.0,
    kappa: float = 0.5,
    theta: float = 0.0,
    sigma: float = 0.1,
    x0: float = 0.0,
    seed: int | None = None,
) -> pd.Series:
    if seed is not None:
        np.random.seed(seed)
    
    x = np.zeros(n_steps + 1)
    x[0] = x0
    
    for t in range(n_steps):
        dW = np.random.randn() * np.sqrt(dt)
        x[t + 1] = x[t] + kappa * (theta - x[t]) * dt + sigma * dW
    
    return pd.Series(x, name="spread")


def simulate_ou_spread_vectorized(
    n_steps: int = 252,
    dt: float = 1.0,
    kappa: float = 0.5,
    theta: float = 0.0,
    sigma: float = 0.1,
    x0: float = 0.0,
    n_paths: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
    
    x = np.zeros((n_paths, n_steps + 1))
    x[:, 0] = x0
    
    for t in range(n_steps):
        x[:, t + 1] = x[:, t] + kappa * (theta - x[:, t]) * dt + sigma * dW[:, t]
    
    return x


def simulate_cir_spread_vectorized(
    n_steps: int = 252,
    dt: float = 1.0,
    kappa: float = 0.5,
    theta: float = 0.0,
    sigma: float = 0.1,
    x0: float = 0.0,
    n_paths: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
    
    x = np.zeros((n_paths, n_steps + 1))
    x[:, 0] = x0
    
    for t in range(n_steps):
        vol = sigma * np.sqrt(np.maximum(x[:, t] - theta, 0) + 0.01)
        x[:, t + 1] = x[:, t] + kappa * (theta - x[:, t]) * dt + vol * dW[:, t]
    
    return x
