"""
Parameter dataclasses for state-space models.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class ModelParams:
    """
    Parameters for state-space models.
    
    State equation:  x_{t+1} = f(x_t, θ) + g(x_t, θ) * η_t
    Observation:     y_t = x_t + ε_t,  ε_t ~ N(0, r)
    
    For AR(1) models:
        f(x) = theta0 + theta1 * x + theta2 * x²
        g(x) = sqrt(q_base + q_het * x²)
    """
    # Mean reversion parameters
    theta0: float = 0.0          # Intercept (drift)
    theta1: float = 0.95         # AR(1) coefficient
    theta2: float = 0.0          # Quadratic term (nonlinearity)
    
    # State noise parameters
    q_base: float = 1e-4         # Base state variance
    q_het: float = 0.0           # Heteroscedasticity coefficient
    
    # Observation noise
    r: float = 1e-4              # Observation variance
    
    # Non-Gaussian parameters (optional)
    mix_prob: float = 0.0        # Mixture probability
    mix_scale: float = 3.0       # Outlier scale factor
    nu: float = float('inf')     # Degrees of freedom for t-distribution (inf = Gaussian)
    
    @property
    def q(self) -> float:
        """Backward compatibility: base state variance."""
        return self.q_base
    
    @property
    def is_linear(self) -> bool:
        """Check if model is linear."""
        return abs(self.theta2) < 1e-10
    
    @property
    def is_homoscedastic(self) -> bool:
        """Check if model has constant volatility."""
        return abs(self.q_het) < 1e-10
    
    @property
    def is_gaussian(self) -> bool:
        """Check if innovations are Gaussian."""
        return self.mix_prob < 1e-10 and self.nu > 100
    
    @property
    def long_run_mean(self) -> float:
        """Long-run mean for linear model."""
        if abs(self.theta1) >= 1.0:
            return 0.0
        if not self.is_linear:
            # Approximate for nonlinear case
            return self.theta0 / (1.0 - self.theta1)
        return self.theta0 / (1.0 - self.theta1)
    
    @property
    def long_run_std(self) -> float:
        """Long-run standard deviation for linear homoscedastic model."""
        if abs(self.theta1) >= 1.0:
            return np.sqrt(self.q_base)
        if not self.is_homoscedastic:
            # Return base volatility for heteroscedastic case
            return np.sqrt(self.q_base)
        return np.sqrt(self.q_base / (1.0 - self.theta1**2))
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "theta0": self.theta0,
            "theta1": self.theta1,
            "theta2": self.theta2,
            "q_base": self.q_base,
            "q_het": self.q_het,
            "r": self.r,
            "mix_prob": self.mix_prob,
            "mix_scale": self.mix_scale,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ModelParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FilterResult:
    """
    Results from Kalman/particle filtering.
    """
    params: ModelParams                # Estimated parameters
    x_filt: pd.Series                  # Filtered states E[x_t | y_{1:t}]
    x_pred: pd.Series                  # Predicted states E[x_t | y_{1:t-1}]
    P_filt: pd.Series                  # Filtered variance
    P_pred: pd.Series                  # Predicted variance
    loglik: float                      # Log-likelihood
    aic: float = 0.0                   # Akaike Information Criterion
    bic: float = 0.0                   # Bayesian Information Criterion
    model_type: str = "unknown"        # Model identifier
    
    @property
    def n_params(self) -> int:
        """Count estimated parameters."""
        n = 4  # theta0, theta1, q_base, r
        if not self.params.is_linear:
            n += 1  # theta2
        if not self.params.is_homoscedastic:
            n += 1  # q_het
        if not self.params.is_gaussian:
            n += 2  # mix_prob, mix_scale
        return n
    
    @property
    def C(self) -> float:
        """Long-run mean (trading center)."""
        return self.params.long_run_mean
    
    @property
    def sigma(self) -> float:
        """Long-run std (for threshold scaling)."""
        return self.params.long_run_std
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Model: {self.model_type}",
            f"Parameters:",
            f"  theta0  = {self.params.theta0:.6f}",
            f"  theta1  = {self.params.theta1:.6f}",
        ]
        if not self.params.is_linear:
            lines.append(f"  theta2  = {self.params.theta2:.6f}")
        lines.extend([
            f"  q_base  = {self.params.q_base:.2e}",
        ])
        if not self.params.is_homoscedastic:
            lines.append(f"  q_het   = {self.params.q_het:.2e}")
        lines.extend([
            f"  r       = {self.params.r:.2e}",
            f"",
            f"Derived:",
            f"  C (mean)= {self.C:.6f}",
            f"  σ (std) = {self.sigma:.6f}",
            f"",
            f"Fit:",
            f"  LogLik  = {self.loglik:.2f}",
            f"  AIC     = {self.aic:.2f}",
            f"  BIC     = {self.bic:.2f}",
        ])
        return "\n".join(lines)
