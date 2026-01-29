"""
Model II: Heteroscedastic AR(1) State-Space Model.

State equation:  x_{t+1} = θ₀ + θ₁ x_t + σ(x_t) w_t,  w_t ~ N(0, 1)
                 where σ(x) = sqrt(q_base + q_het * x²)
Observation:     y_t = x_t + v_t,                     v_t ~ N(0, r)

Requires Extended Kalman Filter or Unscented Kalman Filter.
"""

import numpy as np
from pairs_ssm.models.base_ssm import BaseSSM
from pairs_ssm.models.params import ModelParams


class ModelII(BaseSSM):
    """
    Heteroscedastic AR(1) model (Model II from Zhang 2021).
    
    Features state-dependent volatility where the spread has
    higher variance when it is far from the mean.
    """
    
    name = "Model II (Heteroscedastic)"
    
    def f(self, x: np.ndarray) -> np.ndarray:
        """
        State transition: f(x) = θ₀ + θ₁ x
        (Linear, same as Model I)
        """
        return self.params.theta0 + self.params.theta1 * x
    
    def g(self, x: np.ndarray) -> np.ndarray:
        """
        State noise std: g(x) = sqrt(q_base + q_het * x²)
        (State-dependent volatility)
        """
        var = self.params.q_base + self.params.q_het * x**2
        return np.sqrt(np.maximum(var, 1e-12))
    
    def df_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian: df/dx = θ₁ [constant]
        """
        return np.full_like(x, self.params.theta1, dtype=float)
    
    def dg_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of noise std w.r.t. state.
        dg/dx = q_het * x / sqrt(q_base + q_het * x²)
        """
        var = self.params.q_base + self.params.q_het * x**2
        return self.params.q_het * x / np.sqrt(np.maximum(var, 1e-12))
    
    @staticmethod
    def n_free_params() -> int:
        """Number of free parameters."""
        return 5  # theta0, theta1, q_base, q_het, r
    
    @staticmethod
    def param_names() -> list:
        """Names of free parameters."""
        return ["theta0", "theta1", "q_base", "q_het", "r"]
    
    def pack_params(self) -> np.ndarray:
        """
        Pack parameters into unconstrained vector.
        
        Transforms:
        - theta1 -> arctanh(theta1)
        - q_base, q_het, r -> log() for positivity
        """
        p = self.params
        return np.array([
            p.theta0,
            np.arctanh(np.clip(p.theta1, -0.999, 0.999)),
            np.log(max(p.q_base, 1e-12)),
            np.log(max(p.q_het + 1e-12, 1e-12)),  # q_het can be 0
            np.log(max(p.r, 1e-12)),
        ])
    
    @classmethod
    def unpack_params(cls, z: np.ndarray, var_y: float = 1.0) -> ModelParams:
        """
        Unpack unconstrained vector to ModelParams.
        """
        theta0 = float(z[0])
        theta1 = float(np.tanh(z[1]))
        q_base = float(np.exp(z[2]))
        q_het = float(np.exp(z[3]) - 1e-12)  # Can be close to 0
        r = float(np.exp(z[4]))
        
        # Apply floors
        q_floor = 1e-6 * var_y
        r_floor = 1e-6 * var_y
        
        q_base = max(q_base, q_floor)
        q_het = max(q_het, 0.0)
        r = max(r, r_floor)
        
        return ModelParams(
            theta0=theta0,
            theta1=theta1,
            theta2=0.0,
            q_base=q_base,
            q_het=q_het,
            r=r,
        )
    
    @classmethod
    def get_initial_params(cls, y: np.ndarray) -> ModelParams:
        """
        Get initial parameter estimates from data.
        """
        # Start with Model I estimates
        y0 = y[:-1]
        y1 = y[1:]
        
        X = np.column_stack([np.ones(len(y1)), y0])
        beta = np.linalg.lstsq(X, y1, rcond=None)[0]
        
        theta0 = float(beta[0])
        theta1 = float(np.clip(beta[1], -0.98, 0.98))
        
        # Residual analysis for heteroscedasticity
        resid = y1 - (theta0 + theta1 * y0)
        resid2 = resid**2
        
        # Regress squared residuals on squared lagged values
        # resid² = q_base + q_het * y0² + error
        X_het = np.column_stack([np.ones(len(y0)), y0**2])
        try:
            beta_het = np.linalg.lstsq(X_het, resid2, rcond=None)[0]
            q_base_init = max(float(beta_het[0]), 1e-8)
            q_het_init = max(float(beta_het[1]), 0.0)
        except:
            q_base_init = float(np.var(resid))
            q_het_init = 0.01
        
        r_init = max(0.1 * q_base_init, 1e-8)
        
        return ModelParams(
            theta0=theta0,
            theta1=theta1,
            q_base=q_base_init,
            q_het=q_het_init,
            r=r_init,
        )
    
    @classmethod
    def get_bounds(cls) -> list:
        """Get parameter bounds for optimization."""
        return [
            (-np.inf, np.inf),  # theta0
            (-3.0, 3.0),        # arctanh(theta1)
            (-30.0, 10.0),      # log(q_base)
            (-30.0, 10.0),      # log(q_het + eps)
            (-30.0, 10.0),      # log(r)
        ]


class ModelIINonlinear(BaseSSM):
    """
    Nonlinear + Heteroscedastic model.
    
    State equation:  x_{t+1} = θ₀ + θ₁ x_t + θ₂ x_t² + σ(x_t) w_t
    """
    
    name = "Model II+ (Nonlinear Heteroscedastic)"
    
    def f(self, x: np.ndarray) -> np.ndarray:
        """
        State transition: f(x) = θ₀ + θ₁ x + θ₂ x²
        """
        return self.params.theta0 + self.params.theta1 * x + self.params.theta2 * x**2
    
    def g(self, x: np.ndarray) -> np.ndarray:
        """
        State noise std: g(x) = sqrt(q_base + q_het * x²)
        """
        var = self.params.q_base + self.params.q_het * x**2
        return np.sqrt(np.maximum(var, 1e-12))
    
    def df_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian: df/dx = θ₁ + 2θ₂ x
        """
        return self.params.theta1 + 2.0 * self.params.theta2 * x
    
    @staticmethod
    def n_free_params() -> int:
        return 6  # theta0, theta1, theta2, q_base, q_het, r
    
    @staticmethod
    def param_names() -> list:
        return ["theta0", "theta1", "theta2", "q_base", "q_het", "r"]
