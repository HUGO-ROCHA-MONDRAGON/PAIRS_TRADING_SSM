"""
Base class for state-space models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

from pairs_ssm.models.params import ModelParams


class BaseSSM(ABC):
    """
    Abstract base class for state-space models.
    
    State equation:  x_{t+1} = f(x_t) + g(x_t) * w_t,  w_t ~ p(w)
    Observation:     y_t = h(x_t) + v_t,               v_t ~ N(0, r)
    """
    
    def __init__(self, params: ModelParams):
        """
        Initialize state-space model.
        
        Parameters
        ----------
        params : ModelParams
            Model parameters
        """
        self.params = params
    
    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        """
        State transition function: E[x_{t+1} | x_t]
        
        Parameters
        ----------
        x : np.ndarray
            Current state(s)
            
        Returns
        -------
        np.ndarray
            Expected next state(s)
        """
        pass
    
    @abstractmethod
    def g(self, x: np.ndarray) -> np.ndarray:
        """
        State noise scaling function: Std[x_{t+1} | x_t]
        
        Parameters
        ----------
        x : np.ndarray
            Current state(s)
            
        Returns
        -------
        np.ndarray
            State noise standard deviation(s)
        """
        pass
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Observation function: E[y_t | x_t]
        Default is identity: h(x) = x
        
        Parameters
        ----------
        x : np.ndarray
            State(s)
            
        Returns
        -------
        np.ndarray
            Expected observation(s)
        """
        return x
    
    def df_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of state transition: df/dx
        Used for Extended Kalman Filter.
        
        Parameters
        ----------
        x : np.ndarray
            Current state(s)
            
        Returns
        -------
        np.ndarray
            Jacobian value(s)
        """
        # Default: numerical differentiation
        eps = 1e-6
        return (self.f(x + eps) - self.f(x - eps)) / (2 * eps)
    
    def dh_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation function: dh/dx
        Default is 1 for identity observation.
        """
        return np.ones_like(x)
    
    def state_variance(self, x: np.ndarray) -> np.ndarray:
        """
        State noise variance: g(x)²
        """
        return self.g(x) ** 2
    
    def observation_variance(self) -> float:
        """
        Observation noise variance.
        """
        return self.params.r
    
    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        x0: Optional[float] = None,
        seed: Optional[int] = None,
        return_observations: bool = False,
    ) -> np.ndarray:
        """
        Simulate paths from the model.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps
        n_paths : int
            Number of independent paths
        x0 : float, optional
            Initial state (default: long-run mean)
        seed : int, optional
            Random seed
        return_observations : bool
            If True, also return noisy observations
            
        Returns
        -------
        np.ndarray
            Simulated paths, shape (n_paths, n_steps)
            If return_observations, returns (x_paths, y_paths)
        """
        rng = np.random.default_rng(seed)
        
        # Initial state
        if x0 is None:
            x0 = self.params.long_run_mean
        
        # Allocate arrays
        x = np.zeros((n_paths, n_steps))
        x[:, 0] = x0
        
        # Simulate
        for t in range(n_steps - 1):
            # State transition
            x_mean = self.f(x[:, t])
            x_std = self.g(x[:, t])
            x[:, t + 1] = x_mean + x_std * rng.standard_normal(n_paths)
        
        if not return_observations:
            return x
        
        # Add observation noise
        y = x + np.sqrt(self.params.r) * rng.standard_normal((n_paths, n_steps))
        
        return x, y
    
    def stationary_mean(self) -> float:
        """Stationary mean if it exists."""
        return self.params.long_run_mean
    
    def stationary_variance(self) -> float:
        """Stationary variance if it exists."""
        if abs(self.params.theta1) >= 1.0:
            return np.inf
        return self.params.q_base / (1.0 - self.params.theta1**2)
