"""
Monte Carlo simulation for threshold optimization.

Simulates spread dynamics using estimated state-space model parameters.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SimulationParams:
    """Parameters for spread simulation."""
    mu: float      # Long-run mean
    phi: float     # AR(1) coefficient
    q: float       # State noise variance
    r: float       # Observation noise variance
    n_obs: int     # Number of observations
    
    
def simulate_ou_process(
    mu: float,
    kappa: float,
    sigma: float,
    x0: float,
    n_obs: int,
    dt: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate discretized Ornstein-Uhlenbeck process.
    
    dx = kappa * (mu - x) * dt + sigma * dW
    
    Discretized:
    x_{t+1} = mu + exp(-kappa*dt) * (x_t - mu) + sigma * sqrt((1 - exp(-2*kappa*dt))/(2*kappa)) * eps
    
    Parameters
    ----------
    mu : float
        Long-run mean
    kappa : float
        Mean-reversion speed
    sigma : float
        Volatility
    x0 : float
        Initial value
    n_obs : int
        Number of observations
    dt : float
        Time step
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Simulated process
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.zeros(n_obs)
    x[0] = x0
    
    # OU discretization
    phi = np.exp(-kappa * dt)
    sig_x = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
    
    for t in range(1, n_obs):
        x[t] = mu + phi * (x[t - 1] - mu) + sig_x * np.random.randn()
    
    return x


def simulate_model(
    params: SimulationParams,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate state-space model (Model I: linear Gaussian).
    
    State equation: x_t = mu + phi * (x_{t-1} - mu) + eta_t, eta_t ~ N(0, q)
    Observation:    y_t = x_t + eps_t, eps_t ~ N(0, r)
    
    Parameters
    ----------
    params : SimulationParams
        Model parameters
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple
        (true_state, observations)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = params.n_obs
    mu = params.mu
    phi = params.phi
    q = params.q
    r = params.r
    
    # State
    x = np.zeros(n)
    x[0] = mu  # Start at mean
    
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + np.sqrt(q) * np.random.randn()
    
    # Observations
    y = x + np.sqrt(r) * np.random.randn(n)
    
    return x, y


def simulate_model_II(
    mu: float,
    phi: float,
    q: float,
    beta_0: float,
    beta_1: float,
    n_obs: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Model II (heteroscedastic observation noise).
    
    State: x_t = mu + phi * (x_{t-1} - mu) + eta_t
    Obs:   y_t = x_t + eps_t, eps_t ~ N(0, exp(beta_0 + beta_1 * (x_t - mu)^2))
    
    Parameters
    ----------
    mu, phi, q : float
        State dynamics parameters
    beta_0, beta_1 : float
        Heteroscedasticity parameters
    n_obs : int
        Number of observations
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple
        (true_state, observations)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # State
    x = np.zeros(n_obs)
    x[0] = mu
    
    for t in range(1, n_obs):
        x[t] = mu + phi * (x[t - 1] - mu) + np.sqrt(q) * np.random.randn()
    
    # Observations with heteroscedastic noise
    y = np.zeros(n_obs)
    for t in range(n_obs):
        dev_sq = (x[t] - mu) ** 2
        r_t = np.exp(beta_0 + beta_1 * dev_sq)
        y[t] = x[t] + np.sqrt(r_t) * np.random.randn()
    
    return x, y


def simulate_spread_and_prices(
    mu: float,
    phi: float,
    q: float,
    r: float,
    gamma: float,
    n_obs: int,
    p1_vol: float = 0.02,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate spread and corresponding log prices.
    
    Spread: S_t = log(P1_t) - gamma * log(P2_t)
    
    We simulate the spread as a state-space model and back out prices.
    
    Parameters
    ----------
    mu, phi, q, r : float
        State-space model parameters
    gamma : float
        Hedge ratio
    n_obs : int
        Number of observations
    p1_vol : float
        Daily volatility of asset 1
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple
        (true_spread, observed_spread, log_p1, log_p2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Simulate state-space model
    params = SimulationParams(mu=mu, phi=phi, q=q, r=r, n_obs=n_obs)
    x_true, y_obs = simulate_model(params)
    
    # Generate P1 as random walk
    log_p1 = np.zeros(n_obs)
    log_p1[0] = 4.0  # Initial price ~55
    for t in range(1, n_obs):
        log_p1[t] = log_p1[t - 1] + p1_vol * np.random.randn()
    
    # Back out P2 from spread: log_p2 = (log_p1 - spread) / gamma
    log_p2 = (log_p1 - y_obs) / gamma
    
    return x_true, y_obs, log_p1, log_p2


def run_monte_carlo(
    n_simulations: int,
    mu: float,
    phi: float,
    q: float,
    r: float,
    gamma: float,
    n_obs: int,
    seed: Optional[int] = None,
) -> Tuple[list, list, list, list]:
    """
    Run Monte Carlo simulation for threshold optimization.
    
    Parameters
    ----------
    n_simulations : int
        Number of simulations
    mu, phi, q, r : float
        State-space model parameters
    gamma : float
        Hedge ratio
    n_obs : int
        Number of observations per simulation
    seed : int, optional
        Base seed
        
    Returns
    -------
    tuple
        Lists of (spread_true, spread_obs, log_p1, log_p2)
    """
    spread_true_list = []
    spread_obs_list = []
    log_p1_list = []
    log_p2_list = []
    
    for i in range(n_simulations):
        sim_seed = seed + i if seed is not None else None
        x_true, y_obs, log_p1, log_p2 = simulate_spread_and_prices(
            mu, phi, q, r, gamma, n_obs, seed=sim_seed
        )
        spread_true_list.append(x_true)
        spread_obs_list.append(y_obs)
        log_p1_list.append(log_p1)
        log_p2_list.append(log_p2)
    
    return spread_true_list, spread_obs_list, log_p1_list, log_p2_list
