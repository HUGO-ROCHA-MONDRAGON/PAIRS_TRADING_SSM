"""
Quasi Monte Carlo Kalman Filter (QMCKF)
=======================================

Implementation of QMCKF for general state-space models based on:
Zhang, G. (2021). "Pairs trading with general state space models"
Quantitative Finance, 21(9), 1567-1587.

The QMCKF handles:
- Nonlinear state transitions (quadratic mean reversion)
- Heteroscedastic noise (state-dependent volatility)
- Non-Gaussian innovations (mixture of normals, Student-t)

Uses Sobol sequences (quasi-random) for better coverage of the
state space compared to standard Monte Carlo sampling.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm, t as student_t
import warnings

from ..models.params import ModelParams

# Try to import Sobol sequence generator
try:
    from scipy.stats import qmc
    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False
    warnings.warn(
        "scipy.stats.qmc not available. Using pseudo-random sampling instead of Sobol."
    )


# =============================================================================
# STATE TRANSITION FUNCTIONS
# =============================================================================

def state_transition(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    State transition function: E[x_{t+1} | x_t]
    
    f(x) = θ₀ + θ₁x + θ₂x²
    
    For linear models (Model I), θ₂ = 0.
    """
    return params.theta0 + params.theta1 * x + params.theta2 * x**2


def state_transition_jacobian(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    Jacobian of state transition: df/dx
    
    df/dx = θ₁ + 2θ₂x
    """
    return params.theta1 + 2.0 * params.theta2 * x


def state_noise_std(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    State-dependent noise standard deviation.
    
    g(x) = √(q_base + q_het × x²)
    
    For homoscedastic models (Model I), q_het = 0.
    """
    var = params.q_base + params.q_het * x**2
    return np.sqrt(np.maximum(var, 1e-12))


# =============================================================================
# QMC SAMPLING UTILITIES
# =============================================================================

def generate_qmc_points(n_points: int, seed: int = 42) -> np.ndarray:
    """
    Generate quasi-random points using Sobol sequence.
    
    Falls back to pseudo-random if scipy.stats.qmc not available.
    
    Parameters
    ----------
    n_points : int
        Number of points to generate
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of quasi-random points in [0, 1]
    """
    if SOBOL_AVAILABLE:
        sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
        points = sampler.random(n_points).flatten()
    else:
        rng = np.random.default_rng(seed)
        points = rng.random(n_points)
    return points


def mixture_normal_ppf(
    u: np.ndarray, 
    mix_prob: float, 
    mix_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Inverse CDF sampling for mixture of normals.
    
    p(η) = (1-π)·N(0,1) + π·N(0, σ²)
    
    where π = mix_prob and σ = mix_scale.
    
    This approximation samples component membership randomly,
    then applies the inverse CDF of that component.
    """
    if mix_prob < 1e-10:
        return norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    
    n = len(u)
    result = np.zeros(n)
    
    # Determine which component each sample comes from
    component = rng.random(n) < mix_prob
    
    # Sample from appropriate component
    u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
    result[~component] = norm.ppf(u_clipped[~component])
    result[component] = mix_scale * norm.ppf(u_clipped[component])
    
    return result


def student_t_ppf(u: np.ndarray, nu: float) -> np.ndarray:
    """
    Inverse CDF for Student's t-distribution.
    
    For large nu (> 100), falls back to Gaussian.
    """
    if nu > 100:
        return norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    return student_t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)


# =============================================================================
# RESAMPLING
# =============================================================================

def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Systematic resampling for particle filter.
    
    More variance-efficient than multinomial resampling.
    
    Parameters
    ----------
    weights : np.ndarray
        Normalized particle weights
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Indices of resampled particles
    """
    n = len(weights)
    cumsum = np.cumsum(weights)
    
    # Single random starting point
    u0 = rng.random() / n
    u = u0 + np.arange(n) / n
    
    # Find indices via binary search
    indices = np.searchsorted(cumsum, u)
    indices = np.clip(indices, 0, n - 1)
    
    return indices


# =============================================================================
# QUASI MONTE CARLO KALMAN FILTER
# =============================================================================

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
    
    This filter handles the full generality of Zhang (2021):
    - Nonlinear state transitions (f(x) = θ₀ + θ₁x + θ₂x²)
    - Heteroscedastic noise (g(x) = √(q_base + q_het·x²))
    - Non-Gaussian innovations (mixture of normals or Student-t)
    
    Uses Sobol sequences (quasi-random) for better state space coverage.
    
    Parameters
    ----------
    y : np.ndarray
        Observations (spread series)
    params : ModelParams
        Model parameters including:
        - theta0, theta1, theta2: mean reversion
        - q_base, q_het: state noise
        - r: observation noise
        - mix_prob, mix_scale: mixture parameters
        - nu: degrees of freedom for t-distribution
    n_particles : int
        Number of QMC particles (default: 500)
    x0 : float, optional
        Initial state estimate (default: y[0])
    P0 : float, optional
        Initial state variance (default: diffuse)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    loglik : float
        Log-likelihood of the data given parameters
    x_filt : np.ndarray
        Filtered states E[x_t | y_{1:t}]
    x_pred : np.ndarray
        Predicted states E[x_t | y_{1:t-1}]
    P_filt : np.ndarray
        Filtered variances Var[x_t | y_{1:t}]
    P_pred : np.ndarray
        Predicted variances Var[x_t | y_{1:t-1}]
    """
    n = len(y)
    r = params.r
    
    # Initialize
    if x0 is None:
        x0 = float(y[0])
    if P0 is None:
        P0 = 10.0 * max(float(np.var(y)), params.q_base)
    
    # Allocate output arrays
    x_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_filt = np.zeros(n)
    P_pred = np.zeros(n)
    
    loglik = 0.0
    
    # Initialize random generator
    rng = np.random.default_rng(seed)
    
    # Initialize particles from Gaussian around x0
    particles = rng.normal(x0, np.sqrt(P0), n_particles)
    weights = np.ones(n_particles) / n_particles
    
    for t in range(n):
        # === PREDICTION (before update) ===
        # Store weighted mean and variance as prediction
        x_pred[t] = np.sum(weights * particles)
        P_pred[t] = np.sum(weights * (particles - x_pred[t])**2)
        
        # Generate QMC points for state noise
        u_state = generate_qmc_points(n_particles, seed=seed + t)
        
        # Transform to appropriate innovation distribution
        if params.mix_prob > 1e-10:
            # Mixture of normals (Model 4)
            eta = mixture_normal_ppf(u_state, params.mix_prob, params.mix_scale, rng)
        elif params.nu < 100:
            # Student's t-distribution
            eta = student_t_ppf(u_state, params.nu)
        else:
            # Standard Gaussian
            eta = norm.ppf(np.clip(u_state, 1e-10, 1 - 1e-10))
        
        # Propagate particles through state equation
        # x_{t+1} = f(x_t) + g(x_t) × η_t
        f_x = state_transition(particles, params)
        g_x = state_noise_std(particles, params)
        particles_pred = f_x + g_x * eta
        
        # === UPDATE (measurement) ===
        # Compute likelihood weights: p(y_t | x_t)
        # Observation model: y_t = x_t + v_t, v_t ~ N(0, r)
        innovations = y[t] - particles_pred
        log_weights = -0.5 * (innovations**2 / r + np.log(2 * np.pi * r))
        
        # Normalize weights using log-sum-exp for numerical stability
        log_weights_max = np.max(log_weights)
        weights_unnorm = np.exp(log_weights - log_weights_max)
        weight_sum = np.sum(weights_unnorm)
        
        if weight_sum > 1e-12:
            weights = weights_unnorm / weight_sum
            # Log-likelihood contribution
            loglik += log_weights_max + np.log(weight_sum) - np.log(n_particles)
        else:
            # Particle degeneracy - reset to uniform
            weights = np.ones(n_particles) / n_particles
        
        # Filtered estimate (weighted mean and variance)
        x_filt[t] = np.sum(weights * particles_pred)
        P_filt[t] = np.sum(weights * (particles_pred - x_filt[t])**2)
        
        # === RESAMPLING ===
        # Effective sample size
        ESS = 1.0 / np.sum(weights**2)
        
        if ESS < n_particles / 2:
            # Systematic resampling when ESS too low
            indices = systematic_resample(weights, rng)
            particles = particles_pred[indices].copy()
            weights = np.ones(n_particles) / n_particles
        else:
            particles = particles_pred.copy()
        
        # Add small jitter to prevent particle degeneracy
        jitter_std = 0.01 * np.sqrt(max(P_filt[t], 1e-10))
        particles += rng.normal(0, jitter_std, n_particles)
    
    return loglik, x_filt, x_pred, P_filt, P_pred


# =============================================================================
# BOOTSTRAP PARTICLE FILTER
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
    
    A standard particle filter using pseudo-random sampling.
    More robust than QMCKF for highly non-Gaussian distributions,
    but with higher variance.
    
    Parameters
    ----------
    y : np.ndarray
        Observations
    params : ModelParams
        Model parameters
    n_particles : int
        Number of particles (default: 1000)
    x0, P0 : float, optional
        Initial conditions
    seed : int
        Random seed
        
    Returns
    -------
    Same as qmc_kalman_filter
    """
    n = len(y)
    r = params.r
    
    if x0 is None:
        x0 = float(y[0])
    if P0 is None:
        P0 = 10.0 * max(float(np.var(y)), params.q_base)
    
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
        
        # Generate state noise (pseudo-random)
        if params.mix_prob > 1e-10:
            # Mixture of normals
            component = rng.random(n_particles) < params.mix_prob
            eta = rng.standard_normal(n_particles)
            eta[component] *= params.mix_scale
        elif params.nu < 100:
            # Student's t
            eta = rng.standard_t(params.nu, n_particles)
        else:
            # Standard Gaussian
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
        
        # Resample if ESS too low
        ESS = 1.0 / np.sum(weights**2)
        if ESS < n_particles / 3:
            indices = systematic_resample(weights, rng)
            particles = particles_pred[indices].copy()
            weights = np.ones(n_particles) / n_particles
        else:
            particles = particles_pred.copy()
    
    return loglik, x_filt, x_pred, P_filt, P_pred


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_filter(model_type: str):
    """
    Select appropriate filter based on model type.
    
    Parameters
    ----------
    model_type : str
        Model identifier:
        - "model_I", "model1": Linear Gaussian → use standard KF
        - "model_II", "model2", "model3": Nonlinear/heteroscedastic → use EKF
        - "model4", "model5": Non-Gaussian → use QMCKF
        
    Returns
    -------
    callable
        Filter function
    """
    from .kalman_linear import kalman_filter
    from .kalman_extended import extended_kalman_filter
    
    model_type = model_type.lower().replace("_", "")
    
    if model_type in ["modeli", "model1"]:
        return kalman_filter
    elif model_type in ["modelii", "model2", "model3"]:
        return extended_kalman_filter
    else:
        # model4, model5, or anything else
        return qmc_kalman_filter
