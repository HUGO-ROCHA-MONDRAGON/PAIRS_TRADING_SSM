
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize

@dataclass
class KFModel1Params:
    theta0: float
    theta1: float
    q: float      # state noise variance
    r: float      # observation noise variance

@dataclass
class KFModel1Result:
    params: KFModel1Params
    x_filt: pd.Series
    P_filt: pd.Series
    loglik: float


def _kalman_filter_loglik(y: np.ndarray, theta0: float, theta1: float, q: float, r: float):
    """
    Standard Kalman filter for:
      x_{t+1} = theta0 + theta1 x_t + w_t,   w_t ~ N(0,q)
      y_t     = x_t + v_t,                  v_t ~ N(0,r)

    Returns:
      loglik, x_filt, P_filt
    """
    n = len(y)

    # diffuse-ish initialization (works fine for stationary AR(1))
    x = y[0]
    P = 10.0 * np.var(y) if np.var(y) > 0 else 1.0

    x_f = np.zeros(n)
    P_f = np.zeros(n)

    loglik = 0.0
    for t in range(n):
        # ----- Update (measurement) -----
        # y_t = x_t + v_t
        v = y[t] - x                 # innovation
        S = P + r                    # innovation variance
        K = P / S                    # Kalman gain

        x = x + K * v
        P = (1.0 - K) * P

        x_f[t] = x
        P_f[t] = P

        # log-likelihood contribution
        loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + (v * v) / S)

        # ----- Predict (time) -----
        # x_{t+1} = theta0 + theta1 x_t + w_t
        x = theta0 + theta1 * x
        P = (theta1 * theta1) * P + q

    return loglik, x_f, P_f


def fit_kf_model1(y: pd.Series, verbose: bool = True) -> KFModel1Result:
    """
    MLE fit of (theta0, theta1, q, r) by maximizing Kalman filter log-likelihood.
    Uses stable parameter transforms:
      theta1 = tanh(a)  -> keeps in (-1,1)
      q = exp(b), r = exp(c) -> positive
    """
    y = y.dropna().astype(float)
    y_arr = y.values
    var_y = float(np.var(y_arr))
    q_floor = 1e-3 * var_y  # 0.1% of var(y)
    r_floor = 1e-3 * var_y  # 0.1% of var(y)

    # Good starting values from AR(1) on y (your earlier fit)
    # y_t ≈ c + phi y_{t-1} + eps
    y0 = y.iloc[:-1].values
    y1 = y.iloc[1:].values
    X = np.column_stack([np.ones(len(y1)), y0])
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    c_init, phi_init = float(beta[0]), float(beta[1])
    resid = y1 - (c_init + phi_init * y0)
    sig_init = float(np.std(resid, ddof=1))

    # initial guesses
    theta0_init = c_init
    theta1_init = np.clip(phi_init, -0.98, 0.98)
    q_init = max(1e-6, 0.5 * sig_init**2)
    r_init = max(1e-6, 0.5 * sig_init**2)

    # transforms
    def pack(theta0, theta1, q, r):
        a = np.arctanh(np.clip(theta1, -0.999, 0.999))
        b = np.log(q)
        c = np.log(r)
        return np.array([theta0, a, b, c], dtype=float)

    def unpack(z):
        theta0 = float(z[0])
        theta1 = float(np.tanh(z[1]))

        q = float(np.exp(z[2]))
        r = float(np.exp(z[3]))

        # Prevent degenerate MLE solutions (r -> 0) and ensure meaningful filtering
        q = max(q, q_floor)
        r = max(r, r_floor)

        return theta0, theta1, q, r

    z0 = pack(theta0_init, theta1_init, q_init, r_init)

    def neg_loglik(z):
        theta0, theta1, q, r = unpack(z)
        ll, _, _ = _kalman_filter_loglik(y_arr, theta0, theta1, q, r)
        return -ll

    res = minimize(neg_loglik, z0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})

    theta0_hat, theta1_hat, q_hat, r_hat = unpack(res.x)
    ll, x_f, P_f = _kalman_filter_loglik(y_arr, theta0_hat, theta1_hat, q_hat, r_hat)

    if verbose:
        print("KF Model I fit:")
        print("  theta0 =", theta0_hat)
        print("  theta1 =", theta1_hat)
        print("  q      =", q_hat)
        print("  r      =", r_hat)
        print("  loglik =", ll)

    params = KFModel1Params(theta0_hat, theta1_hat, q_hat, r_hat)
    x_filt = pd.Series(x_f, index=y.index, name="x_hat")
    P_filt = pd.Series(P_f, index=y.index, name="P_hat")

    return KFModel1Result(params=params, x_filt=x_filt, P_filt=P_filt, loglik=float(ll))
