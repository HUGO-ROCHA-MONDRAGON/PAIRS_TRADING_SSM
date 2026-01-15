"""
Quasi Monte Carlo Kalman Filter (QMCKF) Implementation
=======================================================

Implementation of state-space filtering for pairs trading based on:
Zhang, G. (2021). "Pairs trading with general state space models"
Quantitative Finance, 21(9), 1567-1587.

This module implements:
- Standard Kalman Filter (Model 1: Linear Gaussian)
- Extended Kalman Filter (EKF) for nonlinear models
- Unscented Kalman Filter (UKF) for nonlinear models  
- Quasi Monte Carlo Kalman Filter (QMCKF) for general SSM
- Particle Filter for highly non-Gaussian cases

Models supported:
- Model 1: Linear AR(1) with Gaussian noise
- Model 2: Nonlinear mean reversion (quadratic term)
- Model 3: Heteroscedastic noise (state-dependent volatility)
- Model 4: Non-Gaussian innovations (mixture of normals)
- Model 5: Combined (nonlinear + heteroscedastic + non-Gaussian)

Author: Quantitative Finance Project
Date: January 2026
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Literal, Tuple, Optional, List
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t as student_t
from scipy.special import logsumexp
import warnings

# Try to import Sobol sequence generator
try:
    from scipy.stats import qmc
    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False
    warnings.warn("scipy.stats.qmc not available. Using pseudo-random sampling instead of Sobol.")


# =============================================================================
# DATA CLASSES FOR PARAMETERS AND RESULTS
# =============================================================================

@dataclass
class ModelParams:
    """
    General state-space model parameters.
    
    State equation: x_{t+1} = f(x_t, θ) + g(x_t, θ) * η_t
    Observation eq: y_t = h(x_t) + σ_v * ε_t
    
    For AR(1) models:
        f(x) = theta0 + theta1 * x + theta2 * x^2 (theta2=0 for linear)
        g(x) = sqrt(q_base + q_het * x^2)        (q_het=0 for homoscedastic)
    """
    # Mean reversion parameters
    theta0: float = 0.0          # Intercept
    theta1: float = 0.95         # AR(1) coefficient (persistence)
    theta2: float = 0.0          # Quadratic term (nonlinearity)
    
    # State noise parameters
    q_base: float = 1e-4         # Base state variance
    q_het: float = 0.0           # Heteroscedasticity coefficient
    
    # Observation noise
    r: float = 1e-4              # Observation variance
    
    # Non-Gaussian parameters (mixture of normals)
    mix_prob: float = 0.0        # Probability of outlier component
    mix_scale: float = 3.0       # Scale of outlier component relative to main
    
    # Degrees of freedom for t-distribution (inf = Gaussian)
    nu: float = float('inf')     
    
    @property
    def q(self) -> float:
        """Backward compatibility: return base state variance."""
        return self.q_base
    
    @property
    def is_linear(self) -> bool:
        return abs(self.theta2) < 1e-10
    
    @property
    def is_homoscedastic(self) -> bool:
        return abs(self.q_het) < 1e-10
    
    @property
    def is_gaussian(self) -> bool:
        return self.mix_prob < 1e-10 and self.nu > 100
    
    @property
    def long_run_mean(self) -> float:
        """Long-run mean for linear model (theta2=0)."""
        if abs(self.theta1) >= 1.0:
            return 0.0
        return self.theta0 / (1.0 - self.theta1)
    
    @property
    def long_run_std(self) -> float:
        """Long-run standard deviation for linear homoscedastic model."""
        if abs(self.theta1) >= 1.0 or not self.is_homoscedastic:
            return np.sqrt(self.q_base)
        return np.sqrt(self.q_base / (1.0 - self.theta1**2))


@dataclass
class FilterResult:
    """Results from Kalman/particle filtering."""
    params: ModelParams
    x_filt: pd.Series              # Filtered state estimates E[x_t | y_{1:t}]
    x_pred: pd.Series              # One-step predictions E[x_t | y_{1:t-1}]
    P_filt: pd.Series              # Filtered state variance Var[x_t | y_{1:t}]
    P_pred: pd.Series              # Prediction variance Var[x_t | y_{1:t-1}]
    loglik: float                  # Log-likelihood
    aic: float = 0.0               # Akaike Information Criterion
    bic: float = 0.0               # Bayesian Information Criterion
    model_type: str = "unknown"    # Model identifier
    
    @property
    def n_params(self) -> int:
        """Count number of estimated parameters based on model type."""
        # Base: theta0, theta1, q_base, r = 4
        n = 4
        if not self.params.is_linear:
            n += 1  # theta2
        if not self.params.is_homoscedastic:
            n += 1  # q_het
        if not self.params.is_gaussian:
            n += 2  # mix_prob, mix_scale or nu
        return n


# =============================================================================
# STATE TRANSITION AND OBSERVATION FUNCTIONS
# =============================================================================

def state_transition(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    State transition function: E[x_{t+1} | x_t]
    f(x) = theta0 + theta1 * x + theta2 * x^2
    """
    return params.theta0 + params.theta1 * x + params.theta2 * x**2


def state_transition_jacobian(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    Jacobian of state transition: df/dx
    df/dx = theta1 + 2 * theta2 * x
    """
    return params.theta1 + 2.0 * params.theta2 * x


def state_noise_std(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    State-dependent noise standard deviation.
    σ(x) = sqrt(q_base + q_het * x^2)
    """
    var = params.q_base + params.q_het * x**2
    return np.sqrt(np.maximum(var, 1e-12))


def observation_function(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    Observation function: E[y_t | x_t]
    h(x) = x (identity for our model)
    """
    return x


# =============================================================================
# STANDARD KALMAN FILTER (Model 1: Linear Gaussian)
# =============================================================================

def kalman_filter_standard(
    y: np.ndarray,
    params: ModelParams,
    x0: Optional[float] = None,
    P0: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Kalman filter for linear Gaussian state-space model.
    
    State:  x_{t+1} = theta0 + theta1 * x_t + w_t,  w_t ~ N(0, q)
    Obs:    y_t = x_t + v_t,                        v_t ~ N(0, r)
    
    Parameters
    ----------
    y : array-like
        Observations
    params : ModelParams
        Model parameters
    x0 : float, optional
        Initial state estimate (default: first observation)
    P0 : float, optional
        Initial state variance (default: diffuse)
        
    Returns
    -------
    loglik : float
        Log-likelihood
    x_filt : array
        Filtered states
    x_pred : array  
        Predicted states
    P_filt : array
        Filtered variances
    P_pred : array
        Predicted variances
    """
    n = len(y)
    theta0, theta1 = params.theta0, params.theta1
    q, r = params.q_base, params.r
    
    # Initialize
    if x0 is None:
        x0 = y[0]
    if P0 is None:
        P0 = 10.0 * max(np.var(y), q)
    
    x_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_filt = np.zeros(n)
    P_pred = np.zeros(n)
    
    # Initial prediction
    x = x0
    P = P0
    loglik = 0.0
    
    for t in range(n):
        # Store prediction
        x_pred[t] = x
        P_pred[t] = P
        
        # === UPDATE (measurement) ===
        # Innovation
        v = y[t] - x
        S = P + r  # Innovation variance
        
        # Kalman gain
        K = P / S if S > 1e-12 else 0.0
        
        # Update state estimate
        x_upd = x + K * v
        P_upd = (1.0 - K) * P
        
        # Store filtered
        x_filt[t] = x_upd
        P_filt[t] = P_upd
        
        # Log-likelihood contribution
        if S > 1e-12:
            loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + v**2 / S)
        
        # === PREDICT (time update) ===
        x = theta0 + theta1 * x_upd
        P = theta1**2 * P_upd + q
    
    return loglik, x_filt, x_pred, P_filt, P_pred


# =============================================================================
# EXTENDED KALMAN FILTER (Models 2-3: Nonlinear/Heteroscedastic)
# =============================================================================

def extended_kalman_filter(
    y: np.ndarray,
    params: ModelParams,
    x0: Optional[float] = None,
    P0: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extended Kalman Filter for nonlinear/heteroscedastic models.
    
    Linearizes the state transition around current estimate.
    Handles:
    - Nonlinear mean reversion (theta2 != 0)
    - State-dependent volatility (q_het != 0)
    """
    n = len(y)
    r = params.r
    
    # Initialize
    if x0 is None:
        x0 = y[0]
    if P0 is None:
        P0 = 10.0 * max(np.var(y), params.q_base)
    
    x_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_filt = np.zeros(n)
    P_pred = np.zeros(n)
    
    x = x0
    P = P0
    loglik = 0.0
    
    for t in range(n):
        # Store prediction
        x_pred[t] = x
        P_pred[t] = P
        
        # === UPDATE ===
        v = y[t] - x  # Innovation (h(x) = x)
        S = P + r
        
        K = P / S if S > 1e-12 else 0.0
        
        x_upd = x + K * v
        P_upd = (1.0 - K) * P
        
        x_filt[t] = x_upd
        P_filt[t] = P_upd
        
        if S > 1e-12:
            loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + v**2 / S)
        
        # === PREDICT ===
        # Nonlinear state transition
        x_next = state_transition(x_upd, params)
        
        # Jacobian for linearization
        F = state_transition_jacobian(x_upd, params)
        
        # State-dependent noise variance
        Q = state_noise_std(x_upd, params)**2
        
        x = x_next
        P = F**2 * P_upd + Q
    
    return loglik, x_filt, x_pred, P_filt, P_pred


# =============================================================================
# UNSCENTED KALMAN FILTER (Better for highly nonlinear models)
# =============================================================================

def unscented_kalman_filter(
    y: np.ndarray,
    params: ModelParams,
    x0: Optional[float] = None,
    P0: Optional[float] = None,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unscented Kalman Filter using sigma points.
    
    Better accuracy than EKF for highly nonlinear systems.
    Uses sigma points to propagate mean and covariance.
    
    Parameters
    ----------
    alpha : float
        Spread of sigma points (typically 1e-3)
    beta : float
        Prior knowledge of distribution (2 for Gaussian)
    kappa : float
        Secondary scaling parameter (typically 0)
    """
    n = len(y)
    r = params.r
    n_x = 1  # State dimension
    
    # UKF weights
    lam = alpha**2 * (n_x + kappa) - n_x
    gamma = np.sqrt(n_x + lam)
    
    # Weights for mean and covariance
    Wm = np.array([lam / (n_x + lam), 0.5 / (n_x + lam), 0.5 / (n_x + lam)])
    Wc = Wm.copy()
    Wc[0] += (1 - alpha**2 + beta)
    
    # Initialize
    if x0 is None:
        x0 = y[0]
    if P0 is None:
        P0 = 10.0 * max(np.var(y), params.q_base)
    
    x_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_filt = np.zeros(n)
    P_pred = np.zeros(n)
    
    x = x0
    P = P0
    loglik = 0.0
    
    for t in range(n):
        # Store prediction
        x_pred[t] = x
        P_pred[t] = P
        
        # === UPDATE ===
        # Generate sigma points
        sqrt_P = np.sqrt(max(P, 1e-12))
        sigma_pts = np.array([x, x + gamma * sqrt_P, x - gamma * sqrt_P])
        
        # Transform through observation function (identity)
        y_sigma = sigma_pts  # h(x) = x
        
        # Predicted observation
        y_pred = np.sum(Wm * y_sigma)
        
        # Innovation covariance
        Pyy = np.sum(Wc * (y_sigma - y_pred)**2) + r
        
        # Cross-covariance
        Pxy = np.sum(Wc * (sigma_pts - x) * (y_sigma - y_pred))
        
        # Kalman gain
        K = Pxy / Pyy if Pyy > 1e-12 else 0.0
        
        # Update
        v = y[t] - y_pred
        x_upd = x + K * v
        P_upd = P - K * Pyy * K
        P_upd = max(P_upd, 1e-12)
        
        x_filt[t] = x_upd
        P_filt[t] = P_upd
        
        if Pyy > 1e-12:
            loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(Pyy) + v**2 / Pyy)
        
        # === PREDICT ===
        # Generate sigma points from updated state
        sqrt_P_upd = np.sqrt(max(P_upd, 1e-12))
        sigma_pts = np.array([x_upd, x_upd + gamma * sqrt_P_upd, x_upd - gamma * sqrt_P_upd])
        
        # Propagate through state transition
        x_sigma_next = state_transition(sigma_pts, params)
        
        # Predicted state
        x_next = np.sum(Wm * x_sigma_next)
        
        # Predicted covariance
        P_next = np.sum(Wc * (x_sigma_next - x_next)**2)
        
        # Add process noise (state-dependent)
        Q = state_noise_std(x_upd, params)**2
        P_next += Q
        
        x = x_next
        P = max(P_next, 1e-12)
    
    return loglik, x_filt, x_pred, P_filt, P_pred


# =============================================================================
# QUASI MONTE CARLO KALMAN FILTER (Full generality)
# =============================================================================

def generate_qmc_points(n_points: int, seed: int = 42) -> np.ndarray:
    """Generate quasi-random points using Sobol sequence."""
    if SOBOL_AVAILABLE:
        sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
        points = sampler.random(n_points).flatten()
    else:
        rng = np.random.default_rng(seed)
        points = rng.random(n_points)
    return points


def mixture_normal_ppf(u: np.ndarray, mix_prob: float, mix_scale: float) -> np.ndarray:
    """
    Inverse CDF for mixture of normals.
    p(η) = (1-π) * N(0,1) + π * N(0, σ²)
    where σ = mix_scale
    """
    if mix_prob < 1e-10:
        return norm.ppf(u)
    
    # Approximate by sampling from mixture
    n = len(u)
    result = np.zeros(n)
    
    # Determine which component each sample comes from
    component = np.random.random(n) < mix_prob
    
    # Sample from appropriate component
    result[~component] = norm.ppf(u[~component])
    result[component] = mix_scale * norm.ppf(u[component])
    
    return result


def student_t_ppf(u: np.ndarray, nu: float) -> np.ndarray:
    """Inverse CDF for Student's t-distribution."""
    if nu > 100:  # Effectively Gaussian
        return norm.ppf(u)
    return student_t.ppf(u, df=nu)


def qmc_kalman_filter(
    y: np.ndarray,
    params: ModelParams,
    n_particles: int = 500,
    x0: Optional[float] = None,
    P0: Optional[float] = None,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Quasi Monte Carlo Kalman Filter for general state-space models.
    
    Handles:
    - Nonlinear state transitions
    - Heteroscedastic noise
    - Non-Gaussian innovations (mixture of normals, t-distribution)
    
    Uses QMC sampling (Sobol sequences) for better coverage of the
    state space compared to standard Monte Carlo.
    
    Parameters
    ----------
    y : array
        Observations
    params : ModelParams
        Model parameters
    n_particles : int
        Number of QMC particles
    x0 : float
        Initial state estimate
    P0 : float
        Initial state variance
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Same as other filters
    """
    n = len(y)
    r = params.r
    
    # Initialize
    if x0 is None:
        x0 = y[0]
    if P0 is None:
        P0 = 10.0 * max(np.var(y), params.q_base)
    
    x_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_filt = np.zeros(n)
    P_pred = np.zeros(n)
    
    loglik = 0.0
    
    # Initialize particles
    rng = np.random.default_rng(seed)
    
    # Initial particle distribution (Gaussian around x0)
    particles = rng.normal(x0, np.sqrt(P0), n_particles)
    weights = np.ones(n_particles) / n_particles
    
    for t in range(n):
        # === PREDICTION ===
        # Store current prediction (weighted mean/var before update)
        x_pred[t] = np.sum(weights * particles)
        P_pred[t] = np.sum(weights * (particles - x_pred[t])**2)
        
        # Generate QMC points for state noise
        u_state = generate_qmc_points(n_particles, seed=seed + t)
        
        # Transform to appropriate distribution
        if params.mix_prob > 1e-10:
            # Mixture of normals
            eta = mixture_normal_ppf(u_state, params.mix_prob, params.mix_scale)
        elif params.nu < 100:
            # Student's t
            eta = student_t_ppf(u_state, params.nu)
        else:
            # Standard normal
            eta = norm.ppf(np.clip(u_state, 1e-10, 1-1e-10))
        
        # Propagate particles through state equation
        # x_{t+1} = f(x_t) + g(x_t) * η
        f_x = state_transition(particles, params)
        g_x = state_noise_std(particles, params)
        particles_pred = f_x + g_x * eta
        
        # === UPDATE ===
        # Compute likelihood weights p(y_t | x_t)
        # y_t = x_t + v_t, v_t ~ N(0, r)
        innovations = y[t] - particles_pred
        log_weights = -0.5 * (innovations**2 / r + np.log(2 * np.pi * r))
        
        # Normalize weights (log-sum-exp for numerical stability)
        log_weights_max = np.max(log_weights)
        weights_unnorm = np.exp(log_weights - log_weights_max)
        weight_sum = np.sum(weights_unnorm)
        
        if weight_sum > 1e-12:
            weights = weights_unnorm / weight_sum
            # Log-likelihood contribution
            loglik += log_weights_max + np.log(weight_sum) - np.log(n_particles)
        else:
            weights = np.ones(n_particles) / n_particles
        
        # Filtered estimate (weighted mean and variance)
        x_filt[t] = np.sum(weights * particles_pred)
        P_filt[t] = np.sum(weights * (particles_pred - x_filt[t])**2)
        
        # === RESAMPLING ===
        # Effective sample size
        ESS = 1.0 / np.sum(weights**2)
        
        if ESS < n_particles / 2:
            # Systematic resampling
            indices = systematic_resample(weights, rng)
            particles = particles_pred[indices]
            weights = np.ones(n_particles) / n_particles
        else:
            particles = particles_pred
        
        # Add small jitter to prevent particle degeneracy
        jitter_std = 0.01 * np.sqrt(max(P_filt[t], 1e-10))
        particles += rng.normal(0, jitter_std, n_particles)
    
    return loglik, x_filt, x_pred, P_filt, P_pred


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling for particle filter."""
    n = len(weights)
    cumsum = np.cumsum(weights)
    
    # Starting point
    u0 = rng.random() / n
    u = u0 + np.arange(n) / n
    
    # Find indices
    indices = np.searchsorted(cumsum, u)
    indices = np.clip(indices, 0, n - 1)
    
    return indices


# =============================================================================
# PARTICLE FILTER (Bootstrap for highly non-Gaussian cases)
# =============================================================================

def particle_filter(
    y: np.ndarray,
    params: ModelParams,
    n_particles: int = 1000,
    x0: Optional[float] = None,
    P0: Optional[float] = None,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap Particle Filter (Sequential Monte Carlo).
    
    More robust than QMCKF for highly non-Gaussian distributions
    at the cost of higher variance.
    """
    n = len(y)
    r = params.r
    
    if x0 is None:
        x0 = y[0]
    if P0 is None:
        P0 = 10.0 * max(np.var(y), params.q_base)
    
    x_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_filt = np.zeros(n)
    P_pred = np.zeros(n)
    
    loglik = 0.0
    rng = np.random.default_rng(seed)
    
    # Initialize particles
    particles = rng.normal(x0, np.sqrt(P0), n_particles)
    weights = np.ones(n_particles) / n_particles
    
    for t in range(n):
        # Prediction
        x_pred[t] = np.sum(weights * particles)
        P_pred[t] = np.sum(weights * (particles - x_pred[t])**2)
        
        # Generate state noise
        if params.mix_prob > 1e-10:
            # Mixture of normals
            component = rng.random(n_particles) < params.mix_prob
            eta = rng.standard_normal(n_particles)
            eta[component] *= params.mix_scale
        elif params.nu < 100:
            # Student's t
            eta = rng.standard_t(params.nu, n_particles)
        else:
            eta = rng.standard_normal(n_particles)
        
        # Propagate
        f_x = state_transition(particles, params)
        g_x = state_noise_std(particles, params)
        particles_pred = f_x + g_x * eta
        
        # Update weights
        innovations = y[t] - particles_pred
        log_weights = -0.5 * innovations**2 / r
        
        log_weights_max = np.max(log_weights)
        weights_unnorm = np.exp(log_weights - log_weights_max)
        weight_sum = np.sum(weights_unnorm)
        
        if weight_sum > 1e-12:
            weights = weights_unnorm / weight_sum
            loglik += log_weights_max + np.log(weight_sum) - np.log(n_particles)
        else:
            weights = np.ones(n_particles) / n_particles
        
        # Filtered estimates
        x_filt[t] = np.sum(weights * particles_pred)
        P_filt[t] = np.sum(weights * (particles_pred - x_filt[t])**2)
        
        # Resample
        ESS = 1.0 / np.sum(weights**2)
        if ESS < n_particles / 3:
            indices = systematic_resample(weights, rng)
            particles = particles_pred[indices]
            weights = np.ones(n_particles) / n_particles
        else:
            particles = particles_pred
    
    return loglik, x_filt, x_pred, P_filt, P_pred


# =============================================================================
# PARAMETER ESTIMATION (MLE with constraints)
# =============================================================================

def pack_params(
    params: ModelParams,
    model_type: str = "model1"
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Pack parameters into optimization vector with bounds.
    
    Transforms:
    - theta1 -> arctanh (keeps in (-1, 1) for stationarity)
    - variances -> log (keeps positive)
    - mix_prob -> logit (keeps in (0, 1))
    """
    theta0 = params.theta0
    theta1_trans = np.arctanh(np.clip(params.theta1, -0.999, 0.999))
    log_q_base = np.log(max(params.q_base, 1e-12))
    log_r = np.log(max(params.r, 1e-12))
    
    if model_type == "model1":
        # Linear Gaussian: theta0, theta1, q, r
        z = np.array([theta0, theta1_trans, log_q_base, log_r])
        bounds = [(-1, 1), (-3, 3), (-20, 0), (-20, 0)]
        
    elif model_type == "model2":
        # Nonlinear: add theta2
        z = np.array([theta0, theta1_trans, params.theta2, log_q_base, log_r])
        bounds = [(-1, 1), (-3, 3), (-1, 1), (-20, 0), (-20, 0)]
        
    elif model_type == "model3":
        # Heteroscedastic: add q_het
        log_q_het = np.log(max(params.q_het, 1e-12)) if params.q_het > 0 else -20
        z = np.array([theta0, theta1_trans, log_q_base, log_q_het, log_r])
        bounds = [(-1, 1), (-3, 3), (-20, 0), (-20, 0), (-20, 0)]
        
    elif model_type == "model4":
        # Non-Gaussian: add mix_prob, mix_scale
        logit_mix = np.log(params.mix_prob / (1 - params.mix_prob)) if 0 < params.mix_prob < 1 else -5
        z = np.array([theta0, theta1_trans, log_q_base, log_r, logit_mix, np.log(params.mix_scale)])
        bounds = [(-1, 1), (-3, 3), (-20, 0), (-20, 0), (-10, 2), (0, 2)]
        
    elif model_type == "model5":
        # Full model: all parameters
        log_q_het = np.log(max(params.q_het, 1e-12)) if params.q_het > 0 else -20
        logit_mix = np.log(params.mix_prob / (1 - params.mix_prob)) if 0 < params.mix_prob < 1 else -5
        z = np.array([theta0, theta1_trans, params.theta2, log_q_base, log_q_het, log_r, logit_mix, np.log(params.mix_scale)])
        bounds = [(-1, 1), (-3, 3), (-0.5, 0.5), (-20, 0), (-20, 0), (-20, 0), (-10, 2), (0, 2)]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return z, bounds


def unpack_params(
    z: np.ndarray,
    model_type: str = "model1",
    var_y: float = 1.0
) -> ModelParams:
    """
    Unpack optimization vector to ModelParams.
    Applies floors to prevent degeneracy.
    """
    q_floor = 1e-4 * var_y
    r_floor = 1e-4 * var_y
    
    if model_type == "model1":
        theta0 = z[0]
        theta1 = np.tanh(z[1])
        q_base = max(np.exp(z[2]), q_floor)
        r = max(np.exp(z[3]), r_floor)
        return ModelParams(theta0=theta0, theta1=theta1, q_base=q_base, r=r)
    
    elif model_type == "model2":
        theta0 = z[0]
        theta1 = np.tanh(z[1])
        theta2 = z[2]
        q_base = max(np.exp(z[3]), q_floor)
        r = max(np.exp(z[4]), r_floor)
        return ModelParams(theta0=theta0, theta1=theta1, theta2=theta2, q_base=q_base, r=r)
    
    elif model_type == "model3":
        theta0 = z[0]
        theta1 = np.tanh(z[1])
        q_base = max(np.exp(z[2]), q_floor)
        q_het = max(np.exp(z[3]), 0)
        r = max(np.exp(z[4]), r_floor)
        return ModelParams(theta0=theta0, theta1=theta1, q_base=q_base, q_het=q_het, r=r)
    
    elif model_type == "model4":
        theta0 = z[0]
        theta1 = np.tanh(z[1])
        q_base = max(np.exp(z[2]), q_floor)
        r = max(np.exp(z[3]), r_floor)
        mix_prob = 1.0 / (1.0 + np.exp(-z[4]))
        mix_scale = np.exp(z[5])
        return ModelParams(theta0=theta0, theta1=theta1, q_base=q_base, r=r, 
                          mix_prob=mix_prob, mix_scale=mix_scale)
    
    elif model_type == "model5":
        theta0 = z[0]
        theta1 = np.tanh(z[1])
        theta2 = z[2]
        q_base = max(np.exp(z[3]), q_floor)
        q_het = max(np.exp(z[4]), 0)
        r = max(np.exp(z[5]), r_floor)
        mix_prob = 1.0 / (1.0 + np.exp(-z[6]))
        mix_scale = np.exp(z[7])
        return ModelParams(theta0=theta0, theta1=theta1, theta2=theta2,
                          q_base=q_base, q_het=q_het, r=r,
                          mix_prob=mix_prob, mix_scale=mix_scale)
    
    raise ValueError(f"Unknown model type: {model_type}")


def get_initial_params(y: np.ndarray, model_type: str = "model1") -> ModelParams:
    """
    Get good initial parameter estimates from data.
    Uses AR(1) regression for starting values.
    """
    # Fit AR(1) by OLS
    y0 = y[:-1]
    y1 = y[1:]
    X = np.column_stack([np.ones(len(y1)), y0])
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    
    c_init = float(beta[0])
    phi_init = float(np.clip(beta[1], -0.98, 0.98))
    
    resid = y1 - (c_init + phi_init * y0)
    sig2_init = float(np.var(resid, ddof=2))
    
    # Split variance between state and observation
    q_init = 0.7 * sig2_init
    r_init = 0.3 * sig2_init
    
    params = ModelParams(
        theta0=c_init,
        theta1=phi_init,
        q_base=max(q_init, 1e-6),
        r=max(r_init, 1e-6)
    )
    
    if model_type in ["model2", "model5"]:
        params.theta2 = 0.0
    if model_type in ["model3", "model5"]:
        params.q_het = 0.01 * sig2_init
    if model_type in ["model4", "model5"]:
        params.mix_prob = 0.05
        params.mix_scale = 2.0
    
    return params


def select_filter(model_type: str) -> Callable:
    """Select appropriate filter based on model type."""
    if model_type == "model1":
        return kalman_filter_standard
    elif model_type in ["model2", "model3"]:
        return extended_kalman_filter
    else:  # model4, model5
        return qmc_kalman_filter


def fit_model(
    y: pd.Series,
    model_type: Literal["model1", "model2", "model3", "model4", "model5"] = "model1",
    method: str = "Nelder-Mead",
    n_particles: int = 500,
    maxiter: int = 2000,
    verbose: bool = True,
    seed: int = 42,
) -> FilterResult:
    """
    Fit state-space model by Maximum Likelihood.
    
    Parameters
    ----------
    y : pd.Series
        Observed spread series
    model_type : str
        Model to fit: "model1" (linear), "model2" (nonlinear), 
        "model3" (heteroscedastic), "model4" (non-Gaussian),
        "model5" (combined)
    method : str
        Optimization method ("Nelder-Mead", "L-BFGS-B", "differential_evolution")
    n_particles : int
        Number of particles for QMCKF (model4, model5)
    maxiter : int
        Maximum optimization iterations
    verbose : bool
        Print fitting progress
    seed : int
        Random seed
        
    Returns
    -------
    FilterResult
        Fitted model results
    """
    y = y.dropna().astype(float)
    y_arr = y.values
    var_y = float(np.var(y_arr))
    n = len(y_arr)
    
    # Get initial parameters
    init_params = get_initial_params(y_arr, model_type)
    z0, bounds = pack_params(init_params, model_type)
    
    # Select filter
    filter_func = select_filter(model_type)
    
    # Objective function (negative log-likelihood with regularization)
    def neg_loglik(z):
        try:
            params = unpack_params(z, model_type, var_y)
            
            # Add regularization for stability
            reg = 0.0
            
            # Penalize very small observation noise (prevents degeneracy)
            if params.r < 0.001 * var_y:
                reg += 100 * (0.001 * var_y - params.r)**2
            
            # Penalize non-stationary AR
            if abs(params.theta1) > 0.999:
                reg += 1000 * (abs(params.theta1) - 0.999)**2
            
            # Run filter
            if model_type in ["model4", "model5"]:
                ll, _, _, _, _ = filter_func(y_arr, params, n_particles=n_particles, seed=seed)
            else:
                ll, _, _, _, _ = filter_func(y_arr, params)
            
            return -ll + reg
            
        except Exception:
            return 1e10
    
    # Optimize
    if verbose:
        print(f"Fitting {model_type} by MLE...")
    
    if method == "differential_evolution":
        result = differential_evolution(neg_loglik, bounds, seed=seed, maxiter=maxiter,
                                        polish=True, workers=-1)
    else:
        result = minimize(neg_loglik, z0, method=method,
                         options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
    
    # Extract final parameters
    params_hat = unpack_params(result.x, model_type, var_y)
    
    # Run final filter
    if model_type in ["model4", "model5"]:
        ll, x_f, x_p, P_f, P_p = filter_func(y_arr, params_hat, n_particles=n_particles, seed=seed)
    else:
        ll, x_f, x_p, P_f, P_p = filter_func(y_arr, params_hat)
    
    # Compute information criteria
    k = len(result.x)  # Number of parameters
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    
    if verbose:
        print(f"\n{model_type.upper()} Estimation Results:")
        print(f"  theta0  = {params_hat.theta0:.6f}")
        print(f"  theta1  = {params_hat.theta1:.6f}")
        if not params_hat.is_linear:
            print(f"  theta2  = {params_hat.theta2:.6f}")
        print(f"  q_base  = {params_hat.q_base:.6e}")
        if not params_hat.is_homoscedastic:
            print(f"  q_het   = {params_hat.q_het:.6e}")
        print(f"  r       = {params_hat.r:.6e}")
        if not params_hat.is_gaussian:
            print(f"  mix_prob= {params_hat.mix_prob:.4f}")
            print(f"  mix_scl = {params_hat.mix_scale:.4f}")
        print(f"  loglik  = {ll:.2f}")
        print(f"  AIC     = {aic:.2f}")
        print(f"  BIC     = {bic:.2f}")
        print(f"  C (mean)= {params_hat.long_run_mean:.6f}")
        print(f"  σ (std) = {params_hat.long_run_std:.6f}")
    
    return FilterResult(
        params=params_hat,
        x_filt=pd.Series(x_f, index=y.index, name="x_filt"),
        x_pred=pd.Series(x_p, index=y.index, name="x_pred"),
        P_filt=pd.Series(P_f, index=y.index, name="P_filt"),
        P_pred=pd.Series(P_p, index=y.index, name="P_pred"),
        loglik=ll,
        aic=aic,
        bic=bic,
        model_type=model_type,
    )


# =============================================================================
# MODEL SELECTION AND CROSS-VALIDATION
# =============================================================================

def compare_models(
    y: pd.Series,
    models: List[str] = ["model1", "model2", "model3"],
    verbose: bool = True,
    **fit_kwargs
) -> pd.DataFrame:
    """
    Compare multiple models using information criteria.
    
    Parameters
    ----------
    y : pd.Series
        Observed spread
    models : list
        Models to compare
    verbose : bool
        Print results
    **fit_kwargs
        Additional arguments for fit_model
        
    Returns
    -------
    pd.DataFrame
        Comparison table with AIC, BIC, loglik
    """
    results = []
    
    for model in models:
        try:
            res = fit_model(y, model_type=model, verbose=False, **fit_kwargs)
            results.append({
                "Model": model,
                "LogLik": res.loglik,
                "AIC": res.aic,
                "BIC": res.bic,
                "theta0": res.params.theta0,
                "theta1": res.params.theta1,
                "q": res.params.q_base,
                "r": res.params.r,
            })
        except Exception as e:
            if verbose:
                print(f"Failed to fit {model}: {e}")
    
    df = pd.DataFrame(results)
    
    # Rank by AIC and BIC
    df["AIC_rank"] = df["AIC"].rank()
    df["BIC_rank"] = df["BIC"].rank()
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(df.to_string(index=False))
        print("\nBest model by AIC:", df.loc[df["AIC"].idxmin(), "Model"])
        print("Best model by BIC:", df.loc[df["BIC"].idxmin(), "Model"])
    
    return df


def rolling_window_cv(
    y: pd.Series,
    model_type: str = "model1",
    train_size: int = 500,
    test_size: int = 50,
    step: int = 50,
    verbose: bool = True,
    **fit_kwargs
) -> dict:
    """
    Rolling window cross-validation for out-of-sample evaluation.
    
    Parameters
    ----------
    y : pd.Series
        Full time series
    model_type : str
        Model to evaluate
    train_size : int
        Training window size
    test_size : int
        Test window size
    step : int
        Step between windows
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        CV results with MSE, MAE, and predictions
    """
    y_arr = y.values
    n = len(y_arr)
    
    if train_size + test_size > n:
        raise ValueError("train_size + test_size exceeds data length")
    
    results = {
        "mse": [],
        "mae": [],
        "predictions": [],
        "actuals": [],
        "train_ends": [],
    }
    
    filter_func = select_filter(model_type)
    
    start_idx = 0
    fold = 0
    
    while start_idx + train_size + test_size <= n:
        fold += 1
        train_end = start_idx + train_size
        test_end = train_end + test_size
        
        # Training data
        y_train = y.iloc[start_idx:train_end]
        y_test = y.iloc[train_end:test_end]
        
        try:
            # Fit on training
            res = fit_model(y_train, model_type=model_type, verbose=False, **fit_kwargs)
            
            # Predict on test (using last filtered state)
            x_last = res.x_filt.iloc[-1]
            P_last = res.P_filt.iloc[-1]
            
            # One-step ahead predictions
            y_test_arr = y_test.values
            preds = np.zeros(len(y_test_arr))
            
            x = x_last
            for i in range(len(y_test_arr)):
                preds[i] = x  # Prediction is current state estimate
                # Update with observation
                v = y_test_arr[i] - x
                # Simple update (could use full filter)
                x = res.params.theta0 + res.params.theta1 * x
            
            # Compute errors
            mse = np.mean((preds - y_test_arr)**2)
            mae = np.mean(np.abs(preds - y_test_arr))
            
            results["mse"].append(mse)
            results["mae"].append(mae)
            results["predictions"].extend(preds.tolist())
            results["actuals"].extend(y_test_arr.tolist())
            results["train_ends"].append(y_train.index[-1])
            
            if verbose:
                print(f"Fold {fold}: MSE={mse:.6f}, MAE={mae:.6f}")
                
        except Exception as e:
            if verbose:
                print(f"Fold {fold} failed: {e}")
        
        start_idx += step
    
    # Summary statistics
    results["mean_mse"] = np.mean(results["mse"])
    results["mean_mae"] = np.mean(results["mae"])
    results["std_mse"] = np.std(results["mse"])
    
    if verbose:
        print(f"\nCV Summary for {model_type}:")
        print(f"  Mean MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
        print(f"  Mean MAE: {results['mean_mae']:.6f}")
    
    return results


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def simulate_paths(
    params: ModelParams,
    n_steps: int,
    n_paths: int,
    x0: float = 0.0,
    include_obs: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate paths from the state-space model.
    
    Parameters
    ----------
    params : ModelParams
        Model parameters
    n_steps : int
        Number of time steps
    n_paths : int  
        Number of Monte Carlo paths
    x0 : float
        Initial state
    include_obs : bool
        If True, return (x_paths, y_paths), else just x_paths
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Simulated paths of shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    
    x_paths = np.zeros((n_paths, n_steps))
    x_paths[:, 0] = x0
    
    for t in range(1, n_steps):
        # Generate state noise
        if params.mix_prob > 1e-10:
            # Mixture of normals
            component = rng.random(n_paths) < params.mix_prob
            eta = rng.standard_normal(n_paths)
            eta[component] *= params.mix_scale
        elif params.nu < 100:
            # Student's t
            eta = rng.standard_t(params.nu, n_paths)
        else:
            eta = rng.standard_normal(n_paths)
        
        # State transition
        f_x = state_transition(x_paths[:, t-1], params)
        g_x = state_noise_std(x_paths[:, t-1], params)
        x_paths[:, t] = f_x + g_x * eta
    
    if not include_obs:
        return x_paths
    
    # Generate observations
    v = rng.standard_normal((n_paths, n_steps)) * np.sqrt(params.r)
    y_paths = x_paths + v
    
    return x_paths, y_paths


# =============================================================================
# CONVENIENCE FUNCTION (backward compatible with kalman_model1.py)
# =============================================================================

def fit_kf_model1_qmc(y: pd.Series, verbose: bool = True) -> FilterResult:
    """
    Backward-compatible wrapper for Model 1 fitting.
    Drop-in replacement for kalman_model1.fit_kf_model1()
    """
    return fit_model(y, model_type="model1", verbose=verbose)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test with synthetic data
    print("Testing QMCKF implementation...")
    
    # Generate test data from Model 3 (heteroscedastic)
    true_params = ModelParams(
        theta0=0.01,
        theta1=0.95,
        q_base=0.0001,
        q_het=0.05,
        r=0.0001
    )
    
    np.random.seed(42)
    x, y = simulate_paths(true_params, n_steps=500, n_paths=1, x0=0.0, 
                         include_obs=True, seed=42)
    y_series = pd.Series(y.flatten(), name="spread")
    
    print("\nTrue parameters:")
    print(f"  theta0={true_params.theta0}, theta1={true_params.theta1}")
    print(f"  q_base={true_params.q_base}, q_het={true_params.q_het}, r={true_params.r}")
    
    # Fit models
    print("\n" + "="*60)
    res1 = fit_model(y_series, model_type="model1", verbose=True)
    
    print("\n" + "="*60)
    res3 = fit_model(y_series, model_type="model3", verbose=True)
    
    # Compare
    print("\n" + "="*60)
    compare_models(y_series, models=["model1", "model2", "model3"])
