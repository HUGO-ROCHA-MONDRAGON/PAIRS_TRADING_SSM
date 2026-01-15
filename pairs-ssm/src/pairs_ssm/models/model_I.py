"""
Model I: Linear Gaussian AR(1) State-Space Model.

State equation:  x_{t+1} = θ₀ + θ₁ x_t + w_t,  w_t ~ N(0, q)
Observation:     y_t = x_t + v_t,              v_t ~ N(0, r)

This is the standard Kalman filter case.
"""

import numpy as np
from pairs_ssm.models.base_ssm import BaseSSM
from pairs_ssm.models.params import ModelParams


class ModelI(BaseSSM):
    """
    Linear Gaussian AR(1) model (Model I from Zhang 2021).
    
    Suitable for standard Kalman filtering with:
    - Linear state transition
    - Constant (homoscedastic) state noise
    - Gaussian innovations
    """
    
    name = "Model I (Linear Gaussian)"
    
    def f(self, x: np.ndarray) -> np.ndarray:
        """
        State transition: f(x) = θ₀ + θ₁ x
        """
        return self.params.theta0 + self.params.theta1 * x
    
    def g(self, x: np.ndarray) -> np.ndarray:
        """
        State noise std: g(x) = sqrt(q) [constant]
        """
        return np.full_like(x, np.sqrt(self.params.q_base), dtype=float)
    
    def df_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian: df/dx = θ₁ [constant]
        """
        return np.full_like(x, self.params.theta1, dtype=float)
    
    @staticmethod
    def n_free_params() -> int:
        """Number of free parameters."""
        return 4  # theta0, theta1, q, r
    
    @staticmethod
    def param_names() -> list:
        """Names of free parameters."""
        return ["theta0", "theta1", "q", "r"]
    
    def pack_params(self) -> np.ndarray:
        """
        Pack parameters into unconstrained vector for optimization.
        
        Transforms:
        - theta1 -> arctanh(theta1) to enforce (-1, 1)
        - q, r -> log(q), log(r) to enforce positivity
        """
        p = self.params
        return np.array([
            p.theta0,
            np.arctanh(np.clip(p.theta1, -0.999, 0.999)),
            np.log(max(p.q_base, 1e-12)),
            np.log(max(p.r, 1e-12)),
        ])
    
    @classmethod
    def unpack_params(cls, z: np.ndarray, var_y: float = 1.0) -> ModelParams:
        """
        Unpack unconstrained vector to ModelParams.
        
        Parameters
        ----------
        z : np.ndarray
            Unconstrained parameter vector
        var_y : float
            Variance of observations (for bounds)
            
        Returns
        -------
        ModelParams
            Model parameters
        """
        theta0 = float(z[0])
        theta1 = float(np.tanh(z[1]))
        q_base = float(np.exp(z[2]))
        r = float(np.exp(z[3]))
        
        # Apply floors to prevent degeneracy
        q_floor = 1e-6 * var_y
        r_floor = 1e-6 * var_y
        
        q_base = max(q_base, q_floor)
        r = max(r, r_floor)
        
        return ModelParams(
            theta0=theta0,
            theta1=theta1,
            theta2=0.0,
            q_base=q_base,
            q_het=0.0,
            r=r,
        )
    
    @classmethod
    def get_initial_params(cls, y: np.ndarray) -> ModelParams:
        """
        Get initial parameter estimates from data.
        
        Uses AR(1) regression on observations.
        """
        # Simple AR(1) fit
        y0 = y[:-1]
        y1 = y[1:]
        
        X = np.column_stack([np.ones(len(y1)), y0])
        beta = np.linalg.lstsq(X, y1, rcond=None)[0]
        
        theta0 = float(beta[0])
        theta1 = float(np.clip(beta[1], -0.98, 0.98))
        
        # Residual variance
        resid = y1 - (theta0 + theta1 * y0)
        sig2 = float(np.var(resid, ddof=2))
        
        # Split between state and observation noise
        q_init = max(0.5 * sig2, 1e-8)
        r_init = max(0.5 * sig2, 1e-8)
        
        return ModelParams(
            theta0=theta0,
            theta1=theta1,
            q_base=q_init,
            r=r_init,
        )
    
    @classmethod
    def get_bounds(cls) -> list:
        """Get parameter bounds for optimization."""
        return [
            (-np.inf, np.inf),  # theta0
            (-3.0, 3.0),        # arctanh(theta1)
            (-30.0, 10.0),      # log(q)
            (-30.0, 10.0),      # log(r)
        ]
