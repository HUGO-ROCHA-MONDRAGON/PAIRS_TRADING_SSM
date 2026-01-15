"""
Standard Linear Kalman Filter.

For Model I: Linear Gaussian state-space model.
"""

import numpy as np
from typing import Tuple, Optional

from pairs_ssm.models.params import ModelParams


def kalman_filter(
    y: np.ndarray,
    params: ModelParams,
    x0: Optional[float] = None,
    P0: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Kalman filter for linear Gaussian state-space model.
    
    Model:
        State:  x_{t+1} = θ₀ + θ₁ x_t + w_t,  w_t ~ N(0, q)
        Obs:    y_t = x_t + v_t,              v_t ~ N(0, r)
    
    Parameters
    ----------
    y : np.ndarray
        Observations
    params : ModelParams
        Model parameters (theta0, theta1, q_base, r)
    x0 : float, optional
        Initial state estimate (default: y[0])
    P0 : float, optional
        Initial variance (default: diffuse)
        
    Returns
    -------
    loglik : float
        Log-likelihood
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
    theta0, theta1 = params.theta0, params.theta1
    q, r = params.q_base, params.r
    
    # Initialize
    if x0 is None:
        x0 = float(y[0])
    if P0 is None:
        P0 = 10.0 * max(float(np.var(y)), q)
    
    # Allocate arrays
    x_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_filt = np.zeros(n)
    P_pred = np.zeros(n)
    
    # Initial state
    x = x0
    P = P0
    loglik = 0.0
    
    for t in range(n):
        # Store prediction
        x_pred[t] = x
        P_pred[t] = P
        
        # === MEASUREMENT UPDATE ===
        # Innovation
        v = y[t] - x
        S = P + r  # Innovation variance
        
        # Kalman gain
        if S > 1e-12:
            K = P / S
        else:
            K = 0.0
        
        # Update state
        x_upd = x + K * v
        P_upd = (1.0 - K) * P
        
        # Store filtered
        x_filt[t] = x_upd
        P_filt[t] = P_upd
        
        # Log-likelihood contribution
        if S > 1e-12:
            loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + v**2 / S)
        
        # === TIME UPDATE (Prediction) ===
        x = theta0 + theta1 * x_upd
        P = theta1**2 * P_upd + q
    
    return loglik, x_filt, x_pred, P_filt, P_pred


def kalman_smoother(
    y: np.ndarray,
    params: ModelParams,
    x0: Optional[float] = None,
    P0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel smoother for backward pass.
    
    Parameters
    ----------
    y : np.ndarray
        Observations
    params : ModelParams
        Model parameters
    x0, P0 : float, optional
        Initial conditions
        
    Returns
    -------
    x_smooth : np.ndarray
        Smoothed states E[x_t | y_{1:T}]
    P_smooth : np.ndarray
        Smoothed variances Var[x_t | y_{1:T}]
    """
    n = len(y)
    theta1 = params.theta1
    q = params.q_base
    
    # Forward pass
    _, x_filt, x_pred, P_filt, P_pred = kalman_filter(y, params, x0, P0)
    
    # Backward pass
    x_smooth = np.zeros(n)
    P_smooth = np.zeros(n)
    
    # Initialize at T
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    for t in range(n - 2, -1, -1):
        # Smoother gain
        P_pred_next = theta1**2 * P_filt[t] + q
        if P_pred_next > 1e-12:
            J = theta1 * P_filt[t] / P_pred_next
        else:
            J = 0.0
        
        # Smoothed estimates
        x_pred_next = params.theta0 + theta1 * x_filt[t]
        x_smooth[t] = x_filt[t] + J * (x_smooth[t + 1] - x_pred_next)
        P_smooth[t] = P_filt[t] + J**2 * (P_smooth[t + 1] - P_pred_next)
    
    return x_smooth, P_smooth
