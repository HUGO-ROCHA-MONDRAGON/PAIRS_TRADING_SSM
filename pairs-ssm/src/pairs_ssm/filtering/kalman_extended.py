"""
Extended Kalman Filter (EKF).

For Model II: Heteroscedastic and/or nonlinear state-space models.
"""

import numpy as np
from typing import Tuple, Optional

from pairs_ssm.models.params import ModelParams


def extended_kalman_filter(
    y: np.ndarray,
    params: ModelParams,
    x0: Optional[float] = None,
    P0: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extended Kalman Filter for nonlinear/heteroscedastic models.
    
    Linearizes the state transition around current estimate.
    
    Model:
        State:  x_{t+1} = f(x_t) + g(x_t) * w_t,  w_t ~ N(0, 1)
                f(x) = θ₀ + θ₁ x + θ₂ x²
                g(x) = sqrt(q_base + q_het * x²)
        Obs:    y_t = x_t + v_t,  v_t ~ N(0, r)
    
    Parameters
    ----------
    y : np.ndarray
        Observations
    params : ModelParams
        Model parameters
    x0 : float, optional
        Initial state
    P0 : float, optional
        Initial variance
        
    Returns
    -------
    loglik : float
        Log-likelihood
    x_filt : np.ndarray
        Filtered states
    x_pred : np.ndarray
        Predicted states
    P_filt : np.ndarray
        Filtered variances
    P_pred : np.ndarray
        Predicted variances
    """
    n = len(y)
    r = params.r
    
    # Initialize
    if x0 is None:
        x0 = float(y[0])
    if P0 is None:
        P0 = 10.0 * max(float(np.var(y)), params.q_base)
    
    # Allocate
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
        
        # === MEASUREMENT UPDATE ===
        # Observation: y_t = x_t + v_t (linear)
        v = y[t] - x
        S = P + r
        
        if S > 1e-12:
            K = P / S
        else:
            K = 0.0
        
        x_upd = x + K * v
        P_upd = (1.0 - K) * P
        
        x_filt[t] = x_upd
        P_filt[t] = P_upd
        
        if S > 1e-12:
            loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + v**2 / S)
        
        # === TIME UPDATE (Prediction) ===
        # Nonlinear state transition
        x_next = _state_transition(x_upd, params)
        
        # Jacobian for linearization
        F = _state_jacobian(x_upd, params)
        
        # State-dependent noise variance
        Q = _state_noise_var(x_upd, params)
        
        x = x_next
        P = F**2 * P_upd + Q
    
    return loglik, x_filt, x_pred, P_filt, P_pred


def _state_transition(x: float, params: ModelParams) -> float:
    """f(x) = θ₀ + θ₁ x + θ₂ x²"""
    return params.theta0 + params.theta1 * x + params.theta2 * x**2


def _state_jacobian(x: float, params: ModelParams) -> float:
    """df/dx = θ₁ + 2θ₂ x"""
    return params.theta1 + 2.0 * params.theta2 * x


def _state_noise_var(x: float, params: ModelParams) -> float:
    """g(x)² = q_base + q_het * x²"""
    return params.q_base + params.q_het * x**2


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
    
    Parameters
    ----------
    y : np.ndarray
        Observations
    params : ModelParams
        Model parameters
    x0, P0 : float, optional
        Initial conditions
    alpha, beta, kappa : float
        UKF tuning parameters
        
    Returns
    -------
    Tuple of loglik, x_filt, x_pred, P_filt, P_pred
    """
    n = len(y)
    r = params.r
    n_x = 1  # State dimension
    
    # UKF weights
    lam = alpha**2 * (n_x + kappa) - n_x
    gamma = np.sqrt(n_x + lam)
    
    Wm = np.array([lam / (n_x + lam), 0.5 / (n_x + lam), 0.5 / (n_x + lam)])
    Wc = Wm.copy()
    Wc[0] += (1 - alpha**2 + beta)
    
    # Initialize
    if x0 is None:
        x0 = float(y[0])
    if P0 is None:
        P0 = 10.0 * max(float(np.var(y)), params.q_base)
    
    x_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_filt = np.zeros(n)
    P_pred = np.zeros(n)
    
    x = x0
    P = P0
    loglik = 0.0
    
    for t in range(n):
        x_pred[t] = x
        P_pred[t] = P
        
        # === MEASUREMENT UPDATE ===
        v = y[t] - x
        S = P + r
        
        if S > 1e-12:
            K = P / S
        else:
            K = 0.0
        
        x_upd = x + K * v
        P_upd = (1.0 - K) * P
        
        x_filt[t] = x_upd
        P_filt[t] = P_upd
        
        if S > 1e-12:
            loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + v**2 / S)
        
        # === TIME UPDATE via sigma points ===
        # Generate sigma points
        sqrt_P = np.sqrt(max(P_upd, 1e-12))
        sigma_pts = np.array([x_upd, x_upd + gamma * sqrt_P, x_upd - gamma * sqrt_P])
        
        # Transform sigma points
        sigma_transformed = np.array([_state_transition(s, params) for s in sigma_pts])
        
        # Predicted mean and covariance
        x = np.sum(Wm * sigma_transformed)
        
        # State noise (evaluated at mean)
        Q = _state_noise_var(x_upd, params)
        
        P = np.sum(Wc * (sigma_transformed - x)**2) + Q
    
    return loglik, x_filt, x_pred, P_filt, P_pred
