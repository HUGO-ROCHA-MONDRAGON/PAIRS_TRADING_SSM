#!/usr/bin/env python3
"""
RÉPLICATION COMPLÈTE - Zhang (2021)
====================================
Tables 2, 3 + Annexes A1-A6

Tables répliquées:
- Table 2 & 3: PEP-KO, EWT-EWH (Main pairs)
- Table A1: Large Banks (10 pairs) + Small Banks (10 pairs)
- Table A2: Large × Small Banks (25 pairs)
- Table A3: In-Sample / Out-of-Sample Large Banks
- Table A4: In-Sample / Out-of-Sample Small Banks
- Table A5: In-Sample Large × Small Banks
- Table A6: Out-of-Sample Large × Small Banks

Usage:
    python Replication_Full_Appendix.py
    python Replication_Full_Appendix.py ../data/dataGQ.xlsx
"""

from __future__ import annotations
import sys
from pathlib import Path
import itertools

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_FILE = DATA_DIR / "dataGQ.xlsx"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# NUMBA (optional)
# =============================================================================

try:
    from numba import njit

    NUMBA_AVAILABLE = True
    print("✅ Numba disponible - calculs accélérés")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba non disponible - utilisation NumPy")


    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# =============================================================================
# STOCK UNIVERSES (from Zhang 2021 Appendix)
# =============================================================================

# Large Banks
LARGE_BANKS = ['JPM', 'BAC', 'WFC', 'C', 'USB']

# Small Banks
SMALL_BANKS = ['CPF', 'BANC', 'CUBI', 'NBHC', 'FCF']

# Main pairs (Table 2 & 3)
MAIN_PAIRS = [('PEP', 'KO'), ('EWT', 'EWH')]

# =============================================================================
# DATE RANGES (from Zhang 2021)
# =============================================================================

# Full sample (Table 2, 3, A1, A2)
FULL_SAMPLE_START = '2012-01-03'
FULL_SAMPLE_END = '2019-06-28'

# EWT-EWH specific
EWT_EWH_END = '2019-05-01'

# In-Sample period (Table A3, A4, A5)
IN_SAMPLE_START = '2012-01-10'
IN_SAMPLE_END = '2018-01-01'

# Out-of-Sample period (Table A3, A4, A6)
OUT_SAMPLE_START = '2018-01-01'
OUT_SAMPLE_END = '2019-12-01'


# =============================================================================
# DATA LOADING
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


def load_pair_data(filepath: str, col_a: str, col_b: str,
                   start_date: str, end_date: str) -> PairData:
    """Load and align pair data from Excel."""
    df = pd.read_excel(filepath)

    if col_a in df.columns:
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
        df.index = pd.to_datetime(df.index)
        PA = df[col_a].dropna()
        PB = df[col_b].dropna()
    else:
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

    common_idx = PA.index.intersection(PB.index)
    PA, PB = PA.loc[common_idx], PB.loc[common_idx]

    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    mask = (PA.index >= start) & (PA.index <= end)

    return PairData(PA.loc[mask], PB.loc[mask], col_a, col_b)


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

@dataclass
class ModelParams:
    """State-space model parameters."""
    theta0: float = 0.0
    theta1: float = 0.95
    theta2: float = 0.0
    q_base: float = 1e-4
    q_het: float = 0.0
    r: float = 1e-4

    @property
    def is_homoscedastic(self) -> bool:
        return self.q_het < 1e-10


# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# =============================================================================

@njit(cache=True)
def halton_sequence_njit(size: int, base: int) -> np.ndarray:
    """Generate Halton sequence."""
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


@njit(cache=True)
def kalman_filter_njit(y: np.ndarray, theta0: float, theta1: float,
                       q: float, r: float) -> Tuple[float, np.ndarray]:
    """Kalman Filter for Model I."""
    n = len(y)

    if abs(theta1) < 0.999:
        x = theta0 / (1.0 - theta1)
        P = q / (1.0 - theta1 * theta1)
    else:
        x = y[0]
        P = q * 10.0

    x_filt = np.zeros(n)
    loglik = 0.0
    log_2pi = np.log(2.0 * np.pi)

    for t in range(n):
        if t > 0:
            x = theta0 + theta1 * x
            P = theta1 * theta1 * P + q

        v = y[t] - x
        S = P + r

        if S > 1e-12:
            K = P / S
            x = x + K * v
            P = (1.0 - K) * P
            loglik += -0.5 * (log_2pi + np.log(S) + v * v / S)

        x_filt[t] = x

    return loglik, x_filt


@njit(cache=True)
def qmckf_njit(y: np.ndarray, theta0: float, theta1: float,
               q_base: float, q_het: float, r: float,
               n_particles: int) -> Tuple[float, np.ndarray]:
    """QMCKF for Model II."""
    n = len(y)
    x = y[0]
    P = q_base + q_het * x * x

    x_filt = np.zeros(n)
    loglik = 0.0
    log_2pi = np.log(2.0 * np.pi)

    h1 = halton_sequence_njit(n_particles, 2)
    h2 = halton_sequence_njit(n_particles, 3)

    for i in range(n_particles):
        h1[i] = max(1e-10, min(1.0 - 1e-10, h1[i]))
        h2[i] = max(1e-10, min(1.0 - 1e-10, h2[i]))

    z = np.zeros(n_particles)
    for i in range(n_particles):
        z[i] = np.sqrt(-2.0 * np.log(h1[i])) * np.cos(2.0 * np.pi * h2[i])

    samples = np.zeros(n_particles)
    f_samples = np.zeros(n_particles)

    for t in range(n):
        if t == 0:
            x_p, P_p = x, P
        else:
            sqrt_P = np.sqrt(max(P, 1e-12))
            sum_f = 0.0
            for i in range(n_particles):
                samples[i] = x + sqrt_P * z[i]
                f_samples[i] = theta0 + theta1 * samples[i]
                sum_f += f_samples[i]
            x_p = sum_f / n_particles

            sum_var, sum_g = 0.0, 0.0
            for i in range(n_particles):
                diff = f_samples[i] - x_p
                sum_var += diff * diff
                sum_g += q_base + q_het * samples[i] * samples[i]
            P_p = sum_var / n_particles + sum_g / n_particles

        v = y[t] - x_p
        S = P_p + r

        if S > 1e-12:
            K = P_p / S
            x = x_p + K * v
            P = (1.0 - K) * P_p
            loglik += -0.5 * (log_2pi + np.log(S) + v * v / S)
        else:
            x, P = x_p, P_p

        x_filt[t] = x

    return loglik, x_filt


@njit(cache=True)
def strategy_A_njit(x: np.ndarray, U: np.ndarray, L: np.ndarray, C: float) -> np.ndarray:
    """Strategy A."""
    n = len(x)
    sig = np.zeros(n)
    pos = 0

    for t in range(n):
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


@njit(cache=True)
def strategy_C_njit(x: np.ndarray, U: np.ndarray, L: np.ndarray, C: float) -> np.ndarray:
    """Strategy C."""
    n = len(x)
    sig = np.zeros(n)
    pos = 0

    for t in range(1, n):
        prev, curr = x[t - 1], x[t]
        U_prev, U_curr = U[t - 1], U[t]
        L_prev, L_curr = L[t - 1], L[t]

        entry_short = (prev > U_prev) and (curr <= U_curr)
        entry_long = (prev < L_prev) and (curr >= L_curr)
        exit_long = (prev < C) and (curr >= C)
        exit_short = (prev > C) and (curr <= C)
        stop_short = (prev < U_prev) and (curr >= U_curr)
        stop_long = (prev > L_prev) and (curr <= L_curr)

        if pos == 0:
            if entry_short:
                pos = -1
            elif entry_long:
                pos = 1
        elif pos == 1 and (exit_long or stop_long):
            pos = 0
        elif pos == -1 and (exit_short or stop_short):
            pos = 0

        sig[t] = pos

    return sig


@njit(cache=True)
def compute_thresholds_njit(x_filt: np.ndarray, q_base: float, q_het: float,
                            n_std: float, is_hetero: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute thresholds."""
    n = len(x_filt)
    C = np.mean(x_filt)
    sigma_emp = np.std(x_filt)

    U = np.zeros(n)
    L = np.zeros(n)

    if is_hetero and q_het > 1e-10:
        g_x = np.sqrt(q_base + q_het * x_filt * x_filt)
        mean_g = np.mean(g_x)
        for t in range(n):
            sigma_t = g_x[t] / mean_g * sigma_emp
            U[t] = C + n_std * sigma_t
            L[t] = C - n_std * sigma_t
    else:
        threshold = n_std * sigma_emp
        for t in range(n):
            U[t] = C + threshold
            L[t] = C - threshold

    return U, L, C


@njit(cache=True)
def backtest_njit(signals: np.ndarray, x_filt: np.ndarray, cost_bp: float) -> Tuple[float, float, int]:
    """Backtest. Returns (annualized_return, sharpe, n_trades)."""
    n = len(signals)
    pnl = np.zeros(n)

    n_trades = 0
    cost_factor = 2.0 * cost_bp / 10000.0

    for t in range(1, n):
        dx = x_filt[t] - x_filt[t - 1]
        pos_change = abs(signals[t] - signals[t - 1])
        if pos_change > 0:
            n_trades += 1
        pnl[t] = signals[t] * dx - pos_change * cost_factor

    # Cumulative PnL
    cum_pnl = np.sum(pnl)

    # Annualized return
    ann_ret = cum_pnl / (n / 252.0)

    # Sharpe ratio
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    ann_std = std_pnl * np.sqrt(252.0)

    if ann_std > 1e-10:
        sharpe = (ann_ret - 0.02) / ann_std
    else:
        sharpe = 0.0

    return ann_ret, sharpe, n_trades


@njit(cache=True)
def grid_search_njit(x_filt: np.ndarray, q_base: float, q_het: float,
                     is_hetero: bool, use_strategy_C: bool, cost_bp: float) -> Tuple[float, float, float, int]:
    """Grid search. Returns (best_n_std, best_return, best_sharpe, best_trades)."""
    best_n = 1.0
    best_ret = -1e10
    best_sr = -1e10
    best_trades = 0

    for i in range(25):
        n_std = 0.1 + i * 0.1
        U, L, C = compute_thresholds_njit(x_filt, q_base, q_het, n_std, is_hetero)

        if use_strategy_C:
            sig = strategy_C_njit(x_filt, U, L, C)
        else:
            sig = strategy_A_njit(x_filt, U, L, C)

        ann_ret, sharpe, n_trades = backtest_njit(sig, x_filt, cost_bp)

        if n_trades > 0 and sharpe > best_sr:
            best_sr = sharpe
            best_ret = ann_ret
            best_n = n_std
            best_trades = n_trades

    return best_n, best_ret, best_sr, best_trades


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate_gamma_ols(log_PA: np.ndarray, log_PB: np.ndarray) -> float:
    """Estimate γ via OLS."""
    X = np.column_stack([np.ones(len(log_PB)), log_PB])
    return float(np.linalg.lstsq(X, log_PA, rcond=None)[0][1])


def estimate_model_I(y: np.ndarray) -> Tuple[ModelParams, np.ndarray, float]:
    """Estimate Model I."""
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
        try:
            ll, _ = kalman_filter_njit(y, z[0], np.tanh(z[1]), np.exp(z[2]), np.exp(z[3]))
            return -ll if np.isfinite(ll) else 1e10
        except:
            return 1e10

    bounds = [(-0.5, 0.5), (np.arctanh(0.5), np.arctanh(0.999)),
              (np.log(1e-8), np.log(1.0)), (np.log(1e-8), np.log(1.0))]
    res = minimize(neg_ll, z0, method='L-BFGS-B', bounds=bounds)

    params = ModelParams(theta0=res.x[0], theta1=np.tanh(res.x[1]),
                         q_base=np.exp(res.x[2]), r=np.exp(res.x[3]))
    ll, x_filt = kalman_filter_njit(y, params.theta0, params.theta1, params.q_base, params.r)

    return params, x_filt, ll


def estimate_model_II(y: np.ndarray) -> Tuple[ModelParams, np.ndarray, float]:
    """Estimate Model II."""
    y_mean = np.mean(y)
    best_ll, best_params, best_filt = -np.inf, None, None

    for t0, t1, q_b, q_h, r in [
        (y_mean * 0.01, 0.95, 0.0005, 0.10, 0.010),
        (y_mean * 0.01, 0.93, 0.0003, 0.13, 0.011),
        (y_mean * 0.01, 0.96, 0.0010, 0.08, 0.008),
    ]:
        z0 = np.array([t0, np.arctanh(t1), np.log(q_b), np.log(q_h), np.log(r)])

        def neg_ll(z):
            try:
                ll, _ = qmckf_njit(y, z[0], np.tanh(z[1]), np.exp(z[2]), np.exp(z[3]), np.exp(z[4]), 50)
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
            ll, x_filt = qmckf_njit(y, params.theta0, params.theta1,
                                    params.q_base, params.q_het, params.r, 100)
            if ll > best_ll:
                best_ll, best_params, best_filt = ll, params, x_filt
        except:
            continue

    if best_params is None:
        best_params = ModelParams(theta0=0.0, theta1=0.95, q_base=0.0003, q_het=0.1, r=0.01)
        best_ll, best_filt = qmckf_njit(y, 0.0, 0.95, 0.0003, 0.1, 0.01, 100)

    return best_params, best_filt, best_ll


# =============================================================================
# PAIR ANALYSIS
# =============================================================================

def analyze_pair(pair: PairData, cost_bp: float = 20.0) -> Dict:
    """Analyze a single pair. Returns results for Model I + Strategy A and Model II + Strategy C."""
    log_PA, log_PB = np.log(pair.PA.values), np.log(pair.PB.values)
    gamma = estimate_gamma_ols(log_PA, log_PB)
    y = log_PA - gamma * log_PB

    # Model I + Strategy A
    p1, f1, _ = estimate_model_I(y)
    _, ret_m1, sr_m1, trades_m1 = grid_search_njit(f1, p1.q_base, 0.0, False, False, cost_bp)

    # Model II + Strategy C
    p2, f2, _ = estimate_model_II(y)
    _, ret_m2, sr_m2, trades_m2 = grid_search_njit(f2, p2.q_base, p2.q_het, True, True, cost_bp)

    return {
        'Stock1': pair.asset_a,
        'Stock2': pair.asset_b,
        'M1_Return': ret_m1,
        'M1_Sharpe': sr_m1,
        'M2_Return': ret_m2,
        'M2_Sharpe': sr_m2,
        'Imp_Return': (ret_m2 / ret_m1 - 1) * 100 if abs(ret_m1) > 1e-6 else 0,
        'Imp_Sharpe': (sr_m2 / sr_m1 - 1) * 100 if abs(sr_m1) > 1e-6 else 0,
    }


def analyze_pair_insample_oos(filepath: str, col_a: str, col_b: str,
                              is_start: str, is_end: str,
                              oos_start: str, oos_end: str,
                              cost_bp: float = 20.0) -> Tuple[Dict, Dict]:
    """Analyze pair with in-sample and out-of-sample periods."""

    # In-Sample
    try:
        pair_is = load_pair_data(filepath, col_a, col_b, is_start, is_end)
        log_PA_is, log_PB_is = np.log(pair_is.PA.values), np.log(pair_is.PB.values)
        gamma_is = estimate_gamma_ols(log_PA_is, log_PB_is)
        y_is = log_PA_is - gamma_is * log_PB_is

        p1_is, f1_is, _ = estimate_model_I(y_is)
        _, ret_m1_is, sr_m1_is, _ = grid_search_njit(f1_is, p1_is.q_base, 0.0, False, False, cost_bp)

        p2_is, f2_is, _ = estimate_model_II(y_is)
        _, ret_m2_is, sr_m2_is, _ = grid_search_njit(f2_is, p2_is.q_base, p2_is.q_het, True, True, cost_bp)

        result_is = {
            'Stock1': col_a, 'Stock2': col_b,
            'M1_Return': ret_m1_is, 'M1_Sharpe': sr_m1_is,
            'M2_Return': ret_m2_is, 'M2_Sharpe': sr_m2_is,
            'Imp_Return': (ret_m2_is / ret_m1_is - 1) * 100 if abs(ret_m1_is) > 1e-6 else 0,
            'Imp_Sharpe': (sr_m2_is / sr_m1_is - 1) * 100 if abs(sr_m1_is) > 1e-6 else 0,
        }
    except:
        result_is = None

    # Out-of-Sample (use parameters from in-sample)
    try:
        pair_oos = load_pair_data(filepath, col_a, col_b, oos_start, oos_end)
        log_PA_oos, log_PB_oos = np.log(pair_oos.PA.values), np.log(pair_oos.PB.values)
        y_oos = log_PA_oos - gamma_is * log_PB_oos  # Use in-sample gamma

        # Apply in-sample parameters to OOS data
        _, f1_oos = kalman_filter_njit(y_oos, p1_is.theta0, p1_is.theta1, p1_is.q_base, p1_is.r)
        _, ret_m1_oos, sr_m1_oos, _ = grid_search_njit(f1_oos, p1_is.q_base, 0.0, False, False, cost_bp)

        _, f2_oos = qmckf_njit(y_oos, p2_is.theta0, p2_is.theta1, p2_is.q_base, p2_is.q_het, p2_is.r, 100)
        _, ret_m2_oos, sr_m2_oos, _ = grid_search_njit(f2_oos, p2_is.q_base, p2_is.q_het, True, True, cost_bp)

        result_oos = {
            'Stock1': col_a, 'Stock2': col_b,
            'M1_Return': ret_m1_oos, 'M1_Sharpe': sr_m1_oos,
            'M2_Return': ret_m2_oos, 'M2_Sharpe': sr_m2_oos,
            'Imp_Return': (ret_m2_oos / ret_m1_oos - 1) * 100 if abs(ret_m1_oos) > 1e-6 else 0,
            'Imp_Sharpe': (sr_m2_oos / sr_m1_oos - 1) * 100 if abs(sr_m1_oos) > 1e-6 else 0,
        }
    except:
        result_oos = None

    return result_is, result_oos


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_pairs_within_group(stocks: List[str]) -> List[Tuple[str, str]]:
    """Generate all pairs within a group (C(n,2) combinations)."""
    return list(itertools.combinations(stocks, 2))


def generate_pairs_between_groups(group1: List[str], group2: List[str]) -> List[Tuple[str, str]]:
    """Generate all pairs between two groups."""
    return list(itertools.product(group1, group2))


def replicate_table(filepath: str, pairs: List[Tuple[str, str]],
                    start_date: str, end_date: str, table_name: str) -> pd.DataFrame:
    """Replicate a table."""
    print(f"\n{'=' * 80}")
    print(f"  {table_name}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Pairs: {len(pairs)}")
    print(f"{'=' * 80}")

    results = []
    for i, (col_a, col_b) in enumerate(pairs):
        try:
            pair = load_pair_data(filepath, col_a, col_b, start_date, end_date)
            result = analyze_pair(pair)
            results.append(result)
            print(f"  {i + 1:2d}. {col_a}-{col_b}: M1 SR={result['M1_Sharpe']:.4f}, M2 SR={result['M2_Sharpe']:.4f}")
        except Exception as e:
            print(f"  {i + 1:2d}. {col_a}-{col_b}: ❌ Error - {e}")

    if results:
        df = pd.DataFrame(results)

        # Add summary statistics
        print(f"\n  Summary:")
        print(f"    Mean M1 Return: {df['M1_Return'].mean():.4f}")
        print(f"    Mean M1 Sharpe: {df['M1_Sharpe'].mean():.4f}")
        print(f"    Mean M2 Return: {df['M2_Return'].mean():.4f}")
        print(f"    Mean M2 Sharpe: {df['M2_Sharpe'].mean():.4f}")

        return df
    return pd.DataFrame()


def replicate_table_is_oos(filepath: str, pairs: List[Tuple[str, str]],
                           is_start: str, is_end: str,
                           oos_start: str, oos_end: str,
                           table_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate in-sample and out-of-sample table."""
    print(f"\n{'=' * 80}")
    print(f"  {table_name}")
    print(f"  In-Sample: {is_start} to {is_end}")
    print(f"  Out-of-Sample: {oos_start} to {oos_end}")
    print(f"  Pairs: {len(pairs)}")
    print(f"{'=' * 80}")

    results_is, results_oos = [], []

    for i, (col_a, col_b) in enumerate(pairs):
        try:
            res_is, res_oos = analyze_pair_insample_oos(
                filepath, col_a, col_b, is_start, is_end, oos_start, oos_end
            )
            if res_is:
                results_is.append(res_is)
            if res_oos:
                results_oos.append(res_oos)
            print(f"  {i + 1:2d}. {col_a}-{col_b}: IS SR={res_is['M2_Sharpe']:.4f}, OOS SR={res_oos['M2_Sharpe']:.4f}")
        except Exception as e:
            print(f"  {i + 1:2d}. {col_a}-{col_b}: ❌ Error - {e}")

    df_is = pd.DataFrame(results_is) if results_is else pd.DataFrame()
    df_oos = pd.DataFrame(results_oos) if results_oos else pd.DataFrame()

    return df_is, df_oos


# =============================================================================
# MAIN
# =============================================================================

def main(data_path: str = None):
    """Main function to replicate all tables."""
    start_time = time.time()

    print("=" * 80)
    print("ZHANG (2021) - FULL REPLICATION")
    print("Tables 2, 3 + Appendix A1-A6")
    print("=" * 80)

    if data_path is None:
        data_path = DEFAULT_DATA_FILE
    else:
        data_path = Path(data_path)

    print(f"\nData: {data_path}")
    print(f"Numba: {'✅ Enabled' if NUMBA_AVAILABLE else '❌ Disabled'}")

    # Warm-up JIT
    if NUMBA_AVAILABLE:
        print("\n⏳ JIT Compilation...")
        dummy = np.random.randn(100)
        _ = kalman_filter_njit(dummy, 0.0, 0.95, 0.001, 0.001)
        _ = qmckf_njit(dummy, 0.0, 0.95, 0.001, 0.1, 0.01, 50)
        print("✅ Done!")

    all_tables = {}

    # =========================================================================
    # TABLE 2 & 3: Main Pairs
    # =========================================================================
    print("\n" + "#" * 80)
    print("# TABLES 2 & 3: MAIN PAIRS")
    print("#" * 80)

    main_results = []
    for col_a, col_b in MAIN_PAIRS:
        end_date = EWT_EWH_END if col_a == 'EWT' else FULL_SAMPLE_END
        try:
            pair = load_pair_data(data_path, col_a, col_b, FULL_SAMPLE_START, end_date)
            print(f"\n📊 {col_a}-{col_b}: {pair.n_obs} observations")
            result = analyze_pair(pair)
            main_results.append(result)
            print(f"   Model I + Strategy A: Return={result['M1_Return']:.4f}, Sharpe={result['M1_Sharpe']:.4f}")
            print(f"   Model II + Strategy C: Return={result['M2_Return']:.4f}, Sharpe={result['M2_Sharpe']:.4f}")
        except Exception as e:
            print(f"\n❌ {col_a}-{col_b}: Error - {e}")

    all_tables['Table_2_3'] = pd.DataFrame(main_results)

    # =========================================================================
    # TABLE A1: Large Banks + Small Banks (within group)
    # =========================================================================
    print("\n" + "#" * 80)
    print("# TABLE A1: PAIRS WITHIN LARGE BANKS + SMALL BANKS")
    print("#" * 80)

    # Panel A: Large Banks
    large_pairs = generate_pairs_within_group(LARGE_BANKS)
    df_a1_large = replicate_table(data_path, large_pairs, FULL_SAMPLE_START, FULL_SAMPLE_END,
                                  "Table A1 - Panel A: Large Banks")

    # Panel B: Small Banks
    small_pairs = generate_pairs_within_group(SMALL_BANKS)
    df_a1_small = replicate_table(data_path, small_pairs, FULL_SAMPLE_START, FULL_SAMPLE_END,
                                  "Table A1 - Panel B: Small Banks")

    all_tables['Table_A1_Large'] = df_a1_large
    all_tables['Table_A1_Small'] = df_a1_small

    # =========================================================================
    # TABLE A2: Large Banks × Small Banks
    # =========================================================================
    print("\n" + "#" * 80)
    print("# TABLE A2: PAIRS BETWEEN LARGE AND SMALL BANKS")
    print("#" * 80)

    cross_pairs = generate_pairs_between_groups(LARGE_BANKS, SMALL_BANKS)
    df_a2 = replicate_table(data_path, cross_pairs, FULL_SAMPLE_START, FULL_SAMPLE_END,
                            "Table A2: Large × Small Banks")
    all_tables['Table_A2'] = df_a2

    # =========================================================================
    # TABLE A3: In-Sample / Out-of-Sample Large Banks
    # =========================================================================
    print("\n" + "#" * 80)
    print("# TABLE A3: IN-SAMPLE / OUT-OF-SAMPLE LARGE BANKS")
    print("#" * 80)

    df_a3_is, df_a3_oos = replicate_table_is_oos(
        data_path, large_pairs,
        IN_SAMPLE_START, IN_SAMPLE_END,
        OUT_SAMPLE_START, OUT_SAMPLE_END,
        "Table A3: Large Banks IS/OOS"
    )
    all_tables['Table_A3_IS'] = df_a3_is
    all_tables['Table_A3_OOS'] = df_a3_oos

    # =========================================================================
    # TABLE A4: In-Sample / Out-of-Sample Small Banks
    # =========================================================================
    print("\n" + "#" * 80)
    print("# TABLE A4: IN-SAMPLE / OUT-OF-SAMPLE SMALL BANKS")
    print("#" * 80)

    df_a4_is, df_a4_oos = replicate_table_is_oos(
        data_path, small_pairs,
        IN_SAMPLE_START, IN_SAMPLE_END,
        OUT_SAMPLE_START, OUT_SAMPLE_END,
        "Table A4: Small Banks IS/OOS"
    )
    all_tables['Table_A4_IS'] = df_a4_is
    all_tables['Table_A4_OOS'] = df_a4_oos

    # =========================================================================
    # TABLE A5 & A6: In-Sample / Out-of-Sample Large × Small
    # =========================================================================
    print("\n" + "#" * 80)
    print("# TABLES A5 & A6: IN-SAMPLE / OUT-OF-SAMPLE LARGE × SMALL BANKS")
    print("#" * 80)

    df_a5, df_a6 = replicate_table_is_oos(
        data_path, cross_pairs,
        IN_SAMPLE_START, IN_SAMPLE_END,
        OUT_SAMPLE_START, OUT_SAMPLE_END,
        "Tables A5/A6: Large × Small IS/OOS"
    )
    all_tables['Table_A5_IS'] = df_a5
    all_tables['Table_A6_OOS'] = df_a6

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for name, df in all_tables.items():
        if not df.empty:
            print(f"\n{name}:")
            print(f"  Pairs: {len(df)}")
            print(f"  Mean M1 Sharpe: {df['M1_Sharpe'].mean():.4f}")
            print(f"  Mean M2 Sharpe: {df['M2_Sharpe'].mean():.4f}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")

    return all_tables


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    tables = main(path)