#!/usr/bin/env python3
"""
Table 2 & 3 Replication - Zhang (2021)
======================================

Script de réplication utilisant les modules du projet pairs-ssm.

Usage (depuis pairs-ssm/scripts/):
    python table2_replication_final.py ../data/dataGQ.xlsx
    python table2_replication_final.py ../data/yahoo_adj_close_2012-01-03_to_2019-06-28.xlsx

Structure attendue:
    pairs-ssm/
    ├── src/
    │   ├── pairs_ssm/
    │   │   ├── data/loaders.py
    │   │   ├── filtering/qmckf.py, kalman_linear.py
    │   │   ├── trading/strategy.py
    │   │   └── models/params.py
    ├── scripts/
    │   └── table2_replication_final.py  (ce fichier)
    └── data/
        └── dataGQ.xlsx, yahoo_adj_close_*.xlsx
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ajouter src au path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING (compatible Yahoo & Bloomberg formats)
# =============================================================================

@dataclass
class PairData:
    """Container for pair price data."""
    PA: pd.Series
    PB: pd.Series
    asset_a: str
    asset_b: str

    @property
    def n_obs(self) -> int:
        return len(self.PA)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.PA.index


def load_pair_data(filepath: str, col_a: str, col_b: str,
                   start_date: str, end_date: str) -> PairData:
    """Load and align pair data from Excel (Yahoo or Bloomberg format)."""
    df = pd.read_excel(filepath)

    # Detect format
    if col_a in df.columns:
        # Yahoo format: direct column names
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
        df.index = pd.to_datetime(df.index)
        PA = df[col_a].dropna()
        PB = df[col_b].dropna()
    else:
        # Bloomberg format: "TICKER US Equity" with interleaved date columns
        col_a_bb = f'{col_a} US Equity'
        col_b_bb = f'{col_b} US Equity'
        if col_a_bb not in df.columns:
            col_a_bb = f'{col_a} US Equity '
        if col_b_bb not in df.columns:
            col_b_bb = f'{col_b} US Equity '

        def get_series(df, col):
            col_idx = df.columns.get_loc(col)
            date_col = df.columns[col_idx - 1]
            result = pd.DataFrame({
                'date': pd.to_datetime(df[date_col], errors='coerce'),
                'price': pd.to_numeric(df[col], errors='coerce')
            }).dropna().drop_duplicates('date').set_index('date').sort_index()
            return result['price']

        PA = get_series(df, col_a_bb)
        PB = get_series(df, col_b_bb)

    # Align on common dates
    common_idx = PA.index.intersection(PB.index)
    PA, PB = PA.loc[common_idx], PB.loc[common_idx]

    # Filter date range
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    mask = (PA.index >= start) & (PA.index <= end)

    return PairData(PA.loc[mask], PB.loc[mask], col_a, col_b)


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

@dataclass
class ModelParams:
    """State-space model parameters (Zhang 2021 notation)."""
    theta0: float = 0.0  # θ₀: intercept
    theta1: float = 0.95  # θ₁: AR coefficient
    theta2: float = 0.0  # θ₂: quadratic term (Model II nonlinear)
    q_base: float = 1e-4  # Base state variance
    q_het: float = 0.0  # θ₃: heteroscedasticity coefficient
    r: float = 1e-4  # σ²_ε: observation noise variance

    # For non-Gaussian models (not used in basic replication)
    mix_prob: float = 0.0
    mix_scale: float = 1.0
    nu: float = 100.0

    @property
    def is_heteroscedastic(self) -> bool:
        return self.q_het > 1e-10

    @property
    def long_run_mean(self) -> float:
        if abs(self.theta1) >= 1.0:
            return 0.0
        return self.theta0 / (1.0 - self.theta1)


# =============================================================================
# KALMAN FILTERS
# =============================================================================

def halton_sequence(size: int, base: int) -> np.ndarray:
    """Generate Halton low-discrepancy sequence."""
    sequence = np.zeros(size)
    for i in range(size):
        n = i + 1
        f, result = 1.0, 0.0
        while n > 0:
            f = f / base
            result = result + f * (n % base)
            n = n // base
        sequence[i] = result
    return sequence


def kalman_filter(y: np.ndarray, params: ModelParams,
                  x0: Optional[float] = None, P0: Optional[float] = None
                  ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard Kalman Filter for Model I (homoscedastic)."""
    n = len(y)
    theta0, theta1, q, r = params.theta0, params.theta1, params.q_base, params.r

    # Initialize at unconditional distribution
    if x0 is None:
        x = theta0 / (1 - theta1) if abs(theta1) < 0.999 else y[0]
    else:
        x = x0
    P = P0 if P0 else (q / (1 - theta1 ** 2) if abs(theta1) < 0.999 else q * 10)

    x_filt, x_pred = np.zeros(n), np.zeros(n)
    P_filt, P_pred = np.zeros(n), np.zeros(n)
    loglik = 0.0

    for t in range(n):
        # Predict
        if t > 0:
            x = theta0 + theta1 * x
            P = theta1 ** 2 * P + q

        x_pred[t], P_pred[t] = x, P

        # Update
        v = y[t] - x
        S = P + r

        if S > 1e-12:
            K = P / S
            x = x + K * v
            P = (1 - K) * P
            loglik += -0.5 * (np.log(2 * np.pi) + np.log(S) + v ** 2 / S)

        x_filt[t], P_filt[t] = x, P

    return loglik, x_filt, x_pred, P_filt, P_pred


def qmckf(y: np.ndarray, params: ModelParams, n_particles: int = 200,
          x0: Optional[float] = None, P0: Optional[float] = None, seed: int = 42
          ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quasi-Monte Carlo Kalman Filter for Model II (heteroscedastic)."""
    n = len(y)
    theta0, theta1 = params.theta0, params.theta1
    q_base, q_het, r = params.q_base, params.q_het, params.r

    # Initialize
    x = x0 if x0 is not None else y[0]
    P = P0 if P0 is not None else q_base + q_het * x ** 2

    x_filt, x_pred = np.zeros(n), np.zeros(n)
    P_filt, P_pred = np.zeros(n), np.zeros(n)
    loglik = 0.0

    for t in range(n):
        if t == 0:
            x_p, P_p = x, P
        else:
            # QMC sampling via Halton + Box-Muller
            h1 = np.clip(halton_sequence(n_particles, 2), 1e-10, 1 - 1e-10)
            h2 = np.clip(halton_sequence(n_particles, 3), 1e-10, 1 - 1e-10)
            z = np.sqrt(-2 * np.log(h1)) * np.cos(2 * np.pi * h2)
            samples = x + np.sqrt(max(P, 1e-12)) * z

            # Propagate through state equation
            f_samples = theta0 + theta1 * samples
            x_p = np.mean(f_samples)

            # State-dependent variance: g(x)² = q_base + q_het * x²
            g_sq = q_base + q_het * samples ** 2
            P_p = np.mean((f_samples - x_p) ** 2) + np.mean(g_sq)

        x_pred[t], P_pred[t] = x_p, P_p

        # Kalman update
        v = y[t] - x_p
        S = P_p + r

        if S > 1e-12:
            K = P_p / S
            x = x_p + K * v
            P = (1 - K) * P_p
            loglik += -0.5 * (np.log(2 * np.pi) + np.log(S) + v ** 2 / S)
        else:
            x, P = x_p, P_p

        x_filt[t], P_filt[t] = x, P

    return loglik, x_filt, x_pred, P_filt, P_pred


# =============================================================================
# PARAMETER ESTIMATION (MLE)
# =============================================================================

def estimate_gamma_ols(log_PA: np.ndarray, log_PB: np.ndarray) -> float:
    """Estimate γ via OLS: log(P_A) = α + γ·log(P_B)."""
    X = np.column_stack([np.ones(len(log_PB)), log_PB])
    return float(np.linalg.lstsq(X, log_PA, rcond=None)[0][1])


def estimate_model_I(y: np.ndarray, verbose: bool = False) -> Tuple[ModelParams, np.ndarray, float]:
    """Estimate Model I parameters via MLE."""
    y_mean, y_var = np.mean(y), np.var(y)
    rho = np.corrcoef(y[:-1] - y_mean, y[1:] - y_mean)[0, 1]
    theta1_init = float(np.clip(rho, 0.8, 0.99))

    z0 = np.array([
        y_mean * (1 - theta1_init),
        np.arctanh(theta1_init),
        np.log(y_var * (1 - theta1_init ** 2) * 0.7 + 1e-10),
        np.log(y_var * 0.3 + 1e-10),
    ])

    def neg_ll(z):
        params = ModelParams(theta0=z[0], theta1=np.tanh(z[1]),
                             q_base=np.exp(z[2]), r=np.exp(z[3]))
        try:
            ll, _, _, _, _ = kalman_filter(y, params)
            return -ll if np.isfinite(ll) else 1e10
        except:
            return 1e10

    bounds = [(-0.5, 0.5), (np.arctanh(0.5), np.arctanh(0.999)),
              (np.log(1e-8), np.log(1.0)), (np.log(1e-8), np.log(1.0))]
    res = minimize(neg_ll, z0, method='L-BFGS-B', bounds=bounds)

    params = ModelParams(theta0=res.x[0], theta1=np.tanh(res.x[1]),
                         q_base=np.exp(res.x[2]), r=np.exp(res.x[3]))
    ll, x_filt, x_pred, P_filt, P_pred = kalman_filter(y, params)

    if verbose:
        print(f"  θ₀={params.theta0:.6f}, θ₁={params.theta1:.4f}, "
              f"q={params.q_base:.6f}, r={params.r:.6f}, LL={ll:.2f}")

    return params, x_filt, ll


def estimate_model_II(y: np.ndarray, verbose: bool = False) -> Tuple[ModelParams, np.ndarray, float]:
    """Estimate Model II parameters via MLE with QMCKF."""
    y_mean = np.mean(y)
    best_ll, best_params, best_filt = -np.inf, None, None

    # Multiple starting points (paper-like values)
    for t0, t1, q_b, q_h, r in [
        (y_mean * 0.01, 0.95, 0.0005, 0.10, 0.010),
        (y_mean * 0.01, 0.93, 0.0003, 0.13, 0.011),
        (y_mean * 0.01, 0.96, 0.0010, 0.08, 0.008),
    ]:
        z0 = np.array([t0, np.arctanh(t1), np.log(q_b), np.log(q_h), np.log(r)])

        def neg_ll(z):
            params = ModelParams(theta0=z[0], theta1=np.tanh(z[1]),
                                 q_base=np.exp(z[2]), q_het=np.exp(z[3]), r=np.exp(z[4]))
            try:
                ll, _, _, _, _ = qmckf(y, params, n_particles=50)
                return -ll if np.isfinite(ll) else 1e10
            except:
                return 1e10

        bounds = [(-0.1, 0.1), (np.arctanh(0.85), np.arctanh(0.99)),
                  (np.log(1e-6), np.log(0.005)), (np.log(0.05), np.log(0.3)),
                  (np.log(0.005), np.log(0.05))]

        try:
            res = minimize(neg_ll, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
            params = ModelParams(theta0=res.x[0], theta1=np.tanh(res.x[1]),
                                 q_base=np.exp(res.x[2]), q_het=np.exp(res.x[3]), r=np.exp(res.x[4]))
            ll, x_filt, _, _, _ = qmckf(y, params, n_particles=100)

            if ll > best_ll:
                best_ll, best_params, best_filt = ll, params, x_filt
        except:
            continue

    if best_params is None:
        best_params = ModelParams(theta0=0.0, theta1=0.95, q_base=0.0003, q_het=0.1, r=0.01)
        best_ll, best_filt, _, _, _ = qmckf(y, best_params, n_particles=100)

    if verbose:
        print(f"  θ₀={best_params.theta0:.6f}, θ₁={best_params.theta1:.4f}, "
              f"q_base={best_params.q_base:.6f}, q_het={best_params.q_het:.4f}, "
              f"r={best_params.r:.6f}, LL={best_ll:.2f}")

    return best_params, best_filt, best_ll


# =============================================================================
# TRADING STRATEGIES
# =============================================================================

def compute_thresholds(x_filt: np.ndarray, params: ModelParams, n_std: float
                       ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute trading thresholds (constant for M1, time-varying for M2)."""
    C = float(np.mean(x_filt))
    sigma_emp = float(np.std(x_filt))
    n = len(x_filt)

    if params.is_heteroscedastic:
        # Time-varying: g(x) = √(q_base + q_het·x²)
        g_x = np.sqrt(params.q_base + params.q_het * x_filt ** 2)
        sigma_t = g_x / np.mean(g_x) * sigma_emp  # Normalize to empirical scale
        U = C + n_std * sigma_t
        L = C - n_std * sigma_t
    else:
        U = np.full(n, C + n_std * sigma_emp)
        L = np.full(n, C - n_std * sigma_emp)

    return U, L, C


def strategy_A(x: np.ndarray, U: np.ndarray, L: np.ndarray, C: float) -> np.ndarray:
    """Strategy A: Open at boundaries, close at mean."""
    sig, pos = np.zeros(len(x)), 0
    for t in range(len(x)):
        if pos == 0:
            if x[t] >= U[t]:
                pos = -1
            elif x[t] <= L[t]:
                pos = 1
        elif pos == 1 and x[t] >= C:
            pos = 0
        elif pos == -1 and x[t] <= C:
            pos = 0
        sig[t] = pos
    return sig


def strategy_B(x: np.ndarray, U: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Strategy B: Boundary crossing."""
    sig, pos = np.zeros(len(x)), 0
    for t in range(1, len(x)):
        if x[t - 1] < U[t - 1] and x[t] >= U[t]:
            pos = -1
        elif x[t - 1] > L[t - 1] and x[t] <= L[t]:
            pos = 1
        sig[t] = pos
    return sig


def strategy_C(x: np.ndarray, U: np.ndarray, L: np.ndarray, C: float) -> np.ndarray:
    """Strategy C: Re-entry with stop-loss (Zhang's main strategy)."""
    sig, pos = np.zeros(len(x)), 0
    for t in range(1, len(x)):
        p, c = x[t - 1], x[t]
        # Entry: re-enter from outside
        entry_s = (p > U[t - 1]) and (c <= U[t])
        entry_l = (p < L[t - 1]) and (c >= L[t])
        # Exit: cross mean
        exit_l = (p < C) and (c >= C)
        exit_s = (p > C) and (c <= C)
        # Stop-loss: wrong-way crossing
        stop_s = (p < U[t - 1]) and (c >= U[t])
        stop_l = (p > L[t - 1]) and (c <= L[t])

        if pos == 0:
            if entry_s:
                pos = -1
            elif entry_l:
                pos = 1
        elif pos == 1 and (exit_l or stop_l):
            pos = 0
        elif pos == -1 and (exit_s or stop_s):
            pos = 0
        sig[t] = pos
    return sig


# =============================================================================
# BACKTESTING
# =============================================================================

@dataclass
class BacktestResult:
    """Backtest performance metrics."""
    pnl: np.ndarray
    cumulative_pnl: np.ndarray
    positions: np.ndarray
    n_trades: int

    def annualized_return(self, periods_per_year: int = 252) -> float:
        if len(self.cumulative_pnl) == 0: return 0.0
        return self.cumulative_pnl[-1] / (len(self.pnl) / periods_per_year)

    def annualized_std(self, periods_per_year: int = 252) -> float:
        return float(np.std(self.pnl)) * np.sqrt(periods_per_year)

    def sharpe_ratio(self, rf: float = 0.02) -> float:
        std = self.annualized_std()
        return (self.annualized_return() - rf) / std if std > 1e-10 else 0.0

    def max_drawdown(self) -> float:
        if len(self.cumulative_pnl) == 0: return 0.0
        cummax = np.maximum.accumulate(self.cumulative_pnl)
        return float(np.max(cummax - self.cumulative_pnl))

    def calmar_ratio(self) -> float:
        mdd = self.max_drawdown()
        return self.annualized_return() / mdd if mdd > 1e-10 else 0.0

    def pain_index(self) -> float:
        if len(self.cumulative_pnl) == 0: return 0.0
        cummax = np.maximum.accumulate(self.cumulative_pnl)
        return float(np.mean(cummax - self.cumulative_pnl))


def backtest(signals: np.ndarray, x_filt: np.ndarray, cost_bp: float = 20.0) -> BacktestResult:
    """Backtest trading signals on spread."""
    dx = np.diff(x_filt, prepend=x_filt[0])
    pos_changes = np.abs(np.diff(signals, prepend=0))
    net_pnl = signals * dx - pos_changes * (2 * cost_bp / 10000)
    return BacktestResult(net_pnl, np.cumsum(net_pnl), signals, int(np.sum(pos_changes > 0)))


def find_optimal_threshold(x_filt: np.ndarray, params: ModelParams,
                           strat_name: str, cost_bp: float = 20.0) -> Tuple[float, BacktestResult]:
    """Grid search for optimal n_std."""
    best_n, best_sr, best_r = 1.0, -np.inf, None

    for n_std in np.arange(0.1, 2.6, 0.1):
        U, L, C = compute_thresholds(x_filt, params, n_std)

        if strat_name == 'A':
            sig = strategy_A(x_filt, U, L, C)
        elif strat_name == 'B':
            sig = strategy_B(x_filt, U, L)
        else:
            sig = strategy_C(x_filt, U, L, C)

        r = backtest(sig, x_filt, cost_bp)
        if r.n_trades > 0 and r.sharpe_ratio() > best_sr:
            best_sr, best_n, best_r = r.sharpe_ratio(), n_std, r

    if best_r is None:
        best_r = BacktestResult(np.zeros(len(x_filt)), np.zeros(len(x_filt)), np.zeros(len(x_filt)), 0)
    return best_n, best_r


# =============================================================================
# MAIN REPLICATION
# =============================================================================

def replicate_pair(pair: PairData, name: str, cost_bp: float = 20.0) -> pd.DataFrame:
    """Replicate Table 2 & 3 for one pair."""
    print(f"\n{'#' * 70}\n# {name}\n{'#' * 70}")

    log_PA, log_PB = np.log(pair.PA.values), np.log(pair.PB.values)
    gamma = estimate_gamma_ols(log_PA, log_PB)
    print(f"\nOLS: γ = {gamma:.4f}")

    y = log_PA - gamma * log_PB

    print("\n" + "=" * 60 + "\nMODEL I\n" + "=" * 60)
    p1, f1, ll1 = estimate_model_I(y, verbose=True)

    print("\n" + "=" * 60 + "\nMODEL II (QMCKF)\n" + "=" * 60)
    p2, f2, ll2 = estimate_model_II(y, verbose=True)

    results = []
    for mname, p, f in [('Model I', p1, f1), ('Model II', p2, f2)]:
        for s in ['A', 'B', 'C']:
            n, r = find_optimal_threshold(f, p, s, cost_bp)
            results.append({
                'Pair': name, 'Model': mname, 'Strategy': s,
                'Return': r.annualized_return(), 'Std Dev': r.annualized_std(),
                'Sharpe': r.sharpe_ratio(), 'Calmar': r.calmar_ratio(),
                'Pain Index': r.pain_index(), 'n_std': n, 'Trades': r.n_trades,
            })

    df = pd.DataFrame(results)
    print(f"\n{'=' * 70}\nPERFORMANCE\n{'=' * 70}")
    print(df[['Model', 'Strategy', 'Return', 'Std Dev', 'Sharpe', 'Calmar']].to_string(index=False))
    return df


def main(data_path: str):
    """Main replication with all pairs including banks."""
    print("=" * 80)
    print("ZHANG (2021) TABLE 2 & 3 REPLICATION")
    print("AVEC PAIRES BANCAIRES (ANNEXE)")
    print("=" * 80)
    print(f"Data: {data_path}")

    all_results = []

    # Configuration des paires
    # Format: (ticker_A, ticker_B, start_date, end_date, pair_name, category)
    pairs_config = [
        # === PAIRES PRINCIPALES (Table 2 & 3) ===
        ('PEP', 'KO', '2012-01-03', '2019-06-28', 'PEP-KO', 'Main'),
        ('EWT', 'EWH', '2012-01-03', '2019-05-01', 'EWT-EWH', 'Main'),

        # === PAIRES BANCAIRES (Annexe du papier) ===
        ('JPM', 'BAC', '2012-01-03', '2019-06-28', 'JPM-BAC', 'Banks'),
        ('JPM', 'WFC', '2012-01-03', '2019-06-28', 'JPM-WFC', 'Banks'),
        ('WFC', 'BAC', '2012-01-03', '2019-06-28', 'WFC-BAC', 'Banks'),
        ('JPM', 'C', '2012-01-03', '2019-06-28', 'JPM-C', 'Banks'),
        ('USB', 'WFC', '2012-01-03', '2019-06-28', 'USB-WFC', 'Banks'),
    ]

    # Traiter par catégorie
    for category in ['Main', 'Banks']:
        cat_pairs = [p for p in pairs_config if p[5] == category]
        if cat_pairs:
            print(f"\n{'=' * 80}")
            print(f"  {category.upper()} PAIRS")
            print(f"{'=' * 80}")

            for col_a, col_b, start, end, name, cat in cat_pairs:
                try:
                    pair = load_pair_data(data_path, col_a, col_b, start, end)
                    print(f"\n📊 {name}: {pair.n_obs} observations")
                    all_results.append(replicate_pair(pair, name))
                except Exception as e:
                    print(f"\n❌ Error with {name}: {e}")

    if all_results:
        final = pd.concat(all_results, ignore_index=True)

        # Paper comparison
        paper = {
            ('PEP-KO', 'Model I', 'A'): (0.13, 1.10), ('PEP-KO', 'Model I', 'B'): (0.14, 1.01),
            ('PEP-KO', 'Model I', 'C'): (0.06, 0.76), ('PEP-KO', 'Model II', 'A'): (0.13, 1.08),
            ('PEP-KO', 'Model II', 'B'): (0.14, 1.04), ('PEP-KO', 'Model II', 'C'): (0.22, 2.95),
            ('EWT-EWH', 'Model I', 'A'): (0.15, 1.13), ('EWT-EWH', 'Model I', 'B'): (0.11, 0.65),
            ('EWT-EWH', 'Model I', 'C'): (0.13, 1.45), ('EWT-EWH', 'Model II', 'A'): (0.14, 0.96),
            ('EWT-EWH', 'Model II', 'B'): (0.11, 0.65), ('EWT-EWH', 'Model II', 'C'): (0.32, 3.89),
        }

        # === RÉSUMÉ PAIRES PRINCIPALES ===
        print("\n" + "=" * 90)
        print("RÉSUMÉ - PAIRES PRINCIPALES (vs PAPER)")
        print("=" * 90)
        main_pairs = final[final['Pair'].isin(['PEP-KO', 'EWT-EWH'])]
        print(
            f"\n{'Pair':<10} {'Model':<10} {'S':>3} {'Our Ret':>10} {'Paper Ret':>10} {'Our SR':>10} {'Paper SR':>10}")
        print('-' * 75)
        for _, r in main_pairs.iterrows():
            k = (r['Pair'], r['Model'], r['Strategy'])
            if k in paper:
                pr, ps = paper[k]
                print(f"{r['Pair']:<10} {r['Model']:<10} {r['Strategy']:>3} "
                      f"{r['Return']:>10.4f} {pr:>10.2f} {r['Sharpe']:>10.2f} {ps:>10.2f}")

        # === RÉSUMÉ PAIRES BANCAIRES ===
        print("\n" + "=" * 90)
        print("RÉSUMÉ - PAIRES BANCAIRES (ANNEXE)")
        print("=" * 90)
        bank_pairs = final[~final['Pair'].isin(['PEP-KO', 'EWT-EWH'])]
        if not bank_pairs.empty:
            print(f"\n{'Pair':<10} {'Model':<10} {'S':>3} {'Return':>10} {'Std Dev':>10} {'Sharpe':>10} {'Trades':>8}")
            print('-' * 75)
            for _, r in bank_pairs.iterrows():
                print(f"{r['Pair']:<10} {r['Model']:<10} {r['Strategy']:>3} "
                      f"{r['Return']:>10.4f} {r['Std Dev']:>10.4f} {r['Sharpe']:>10.2f} {r['Trades']:>8.0f}")

        # === TOP 5 STRATEGY C ===
        print("\n" + "=" * 90)
        print("TOP 5 - STRATEGY C (par Sharpe Ratio)")
        print("=" * 90)
        strat_c = final[final['Strategy'] == 'C']
        for model in ['Model I', 'Model II']:
            model_data = strat_c[strat_c['Model'] == model].sort_values('Sharpe', ascending=False)
            if not model_data.empty:
                print(f"\n{model}:")
                print(f"  {'Pair':<12} {'Sharpe':>10} {'Return':>10} {'Trades':>8}")
                print(f"  {'-' * 45}")
                for _, r in model_data.head(5).iterrows():
                    print(f"  {r['Pair']:<12} {r['Sharpe']:>10.2f} {r['Return']:>10.4f} {r['Trades']:>8.0f}")

        return final
    return pd.DataFrame()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/dataGQ.xlsx"
    main(path)