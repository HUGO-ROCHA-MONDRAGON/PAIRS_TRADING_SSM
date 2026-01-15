import numpy as np

def simulate_model1(T: int, N: int, seed: int = 1) -> np.ndarray:
    """
    Model 1: AR(1) simple
      x_{t+1} = 0.9590 * x_t + 0.0049 * eta_t,   eta_t ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((N, T), dtype=float)
    eps = rng.standard_normal((N, T-1))
    for t in range(T-1):
        x[:, t+1] = 0.9590 * x[:, t] + 0.0049 * eps[:, t]
    return x


def simulate_model2(T: int, N: int, seed: int = 1) -> np.ndarray:
    """
    Model 2: AR(1) avec terme quadratique (non-linéaire)
      x_{t+1} = 0.9 * x_t + 0.2590 * x_t^2 + 0.0049 * eta_t,   eta_t ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((N, T), dtype=float)
    eps = rng.standard_normal((N, T-1))
    for t in range(T-1):
        x[:, t+1] = 0.9 * x[:, t] + 0.2590 * (x[:, t] ** 2) + 0.0049 * eps[:, t]
    return x


def simulate_model3(T: int, N: int, seed: int = 1) -> np.ndarray:
    """
    Model 3: AR(1) hétéroscédastique (volatilité dépend de x)
      x_{t+1} = 0.9590 * x_t + sqrt(0.00089 + 0.08 * x_t^2) * eta_t,   eta_t ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((N, T), dtype=float)
    eps = rng.standard_normal((N, T-1))
    for t in range(T-1):
        vol = np.sqrt(0.00089 + 0.08 * (x[:, t] ** 2))
        x[:, t+1] = 0.9590 * x[:, t] + vol * eps[:, t]
    return x
