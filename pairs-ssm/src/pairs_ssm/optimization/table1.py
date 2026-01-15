"""
Table 1 Replication from Zhang (2021).

Optimal Selection of Trading Rule for Cumulative Return and Sharpe Ratio.
Uses numba for performance (~100x faster than pure Python).
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Try to use numba for speed
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator if func is None else func
    def prange(x):
        return range(x)


# =============================================================================
# Sharpe Ratio Helper (non-annualized as in paper)
# =============================================================================

@njit
def _sharpe_from_sums(sum_p: float, sum_p2: float, n: int) -> float:
    """Compute Sharpe ratio from running sums (non-annualized)."""
    if n <= 1:
        return 0.0
    mean = sum_p / n
    var = (sum_p2 - n * mean * mean) / (n - 1)
    if var <= 0.0:
        return 0.0
    return mean / np.sqrt(var)


# =============================================================================
# Strategy Evaluators (following Zhang 2021 exactly)
# =============================================================================

@njit
def _eval_path_A(x: np.ndarray, U: float, L: float, C: float, tc: float) -> Tuple[float, float, float, int]:
    """
    Strategy A: Open at extremes (U/L), close at mean (C).
    - Short when spread >= U
    - Long when spread <= L  
    - Close position when spread returns to C
    """
    T = len(x)
    pos = 0  # 0: flat, 1: long, -1: short
    
    cr = 0.0
    sum_p = 0.0
    sum_p2 = 0.0
    n_steps = T - 1
    
    for t in range(1, T):
        dx = x[t] - x[t-1]
        pnl = pos * dx
        
        new_pos = pos
        if pos == 0:
            if x[t] >= U:
                new_pos = -1  # Short at upper bound
            elif x[t] <= L:
                new_pos = 1   # Long at lower bound
        elif pos == 1:
            if x[t] >= C:
                new_pos = 0   # Close long at mean
        elif pos == -1:
            if x[t] <= C:
                new_pos = 0   # Close short at mean
        
        # Transaction cost on position change
        if new_pos != pos:
            pnl -= tc * abs(new_pos - pos)
        
        pos = new_pos
        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl
    
    return cr, sum_p, sum_p2, n_steps


@njit
def _eval_path_B(x: np.ndarray, U: float, L: float, tc: float) -> Tuple[float, float, float, int]:
    """
    Strategy B: Position changes when crossing boundaries (can flip directly).
    - Cross U from below -> go short (or flip from long to short)
    - Cross L from above -> go long (or flip from short to long)
    No explicit close - position reverses at boundaries.
    """
    T = len(x)
    pos = 0
    
    cr = 0.0
    sum_p = 0.0
    sum_p2 = 0.0
    n_steps = T - 1
    
    for t in range(1, T):
        dx = x[t] - x[t-1]
        pnl = pos * dx
        
        prev = x[t-1]
        cur = x[t]
        
        new_pos = pos
        # Direct position changes on boundary crossing
        if prev < U and cur >= U:
            new_pos = -1  # Short
        elif prev > L and cur <= L:
            new_pos = 1   # Long
        
        if new_pos != pos:
            pnl -= tc * abs(new_pos - pos)
        
        pos = new_pos
        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl
    
    return cr, sum_p, sum_p2, n_steps


@njit
def _eval_path_C(x: np.ndarray, U: float, L: float, C: float, tc: float) -> Tuple[float, float, float, int]:
    """
    Strategy C: Re-entry with stop-loss.
    - Open when crossing boundary toward mean
    - Close at mean (profit) OR stop-loss if crosses boundary away from mean
    
    KEY DIFFERENCE from A/B: PnL calculated using NEW position (after decision).
    This is because crossing-based entry captures the move that triggered entry.
    """
    T = len(x)
    pos = 0
    
    cr = 0.0
    sum_p = 0.0
    sum_p2 = 0.0
    n_steps = T - 1
    
    for t in range(1, T):
        dx = x[t] - x[t-1]
        prev = x[t-1]
        cur = x[t]
        
        # Entry: crossing boundary from outside toward mean
        entry_short = (prev > U and cur <= U)  # Cross U from above
        entry_long = (prev < L and cur >= L)   # Cross L from below
        
        # Exit at mean
        cross_down_C = (prev > C and cur <= C)
        cross_up_C = (prev < C and cur >= C)
        
        # Stop-loss: crossing back outside
        stop_short = (prev < U and cur >= U)   # Cross U from below
        stop_long = (prev > L and cur <= L)    # Cross L from above
        
        new_pos = pos
        if pos == 0:
            if entry_short:
                new_pos = -1
            elif entry_long:
                new_pos = 1
        elif pos == 1:
            if cross_up_C or stop_long:
                new_pos = 0
        elif pos == -1:
            if cross_down_C or stop_short:
                new_pos = 0
        
        # KEY: Use NEW position for PnL (crossing-based entry)
        pnl = new_pos * dx
        
        if new_pos != pos:
            pnl -= tc * abs(new_pos - pos)
        
        pos = new_pos
        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl
    
    return cr, sum_p, sum_p2, n_steps


# =============================================================================
# Grid Search with Numba Parallelization
# =============================================================================

@njit(parallel=True)
def _grid_search_numba(
    paths: np.ndarray,      # (N, T) - raw paths
    C: float,               # Center (mean across all paths)
    sigma_avg: float,       # Average std across all paths
    U_grid: np.ndarray,     # Upper thresholds in sigma units
    L_grid: np.ndarray,     # Lower thresholds in sigma units  
    strategy_id: int,       # 1=A, 2=B, 3=C
    tc: float,              # Transaction cost (total round-trip)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel grid search over all (U, L) combinations.
    
    U and L are in sigma units - converted to absolute using sigma_avg.
    Returns CR and SR arrays of shape (nU * nL,).
    """
    N, T = paths.shape
    nU = len(U_grid)
    nL = len(L_grid)
    K = nU * nL
    
    CR_out = np.full(K, -np.inf)
    SR_out = np.full(K, -np.inf)
    
    for k in prange(K):
        i = k // nL
        j = k % nL
        u_sigma = U_grid[i]  # In sigma units
        l_sigma = L_grid[j]  # In sigma units
        
        # Skip invalid combinations (L must be < 0, U must be > 0)
        if l_sigma >= 0 or u_sigma <= 0 or l_sigma >= u_sigma:
            continue
        
        # Convert to absolute thresholds (same for all paths)
        U = C + u_sigma * sigma_avg
        L = C + l_sigma * sigma_avg
        
        cr_sum = 0.0
        sr_sum = 0.0
        
        for n in range(N):
            x = paths[n, :]
            
            if strategy_id == 1:
                cr, s1, s2, n_steps = _eval_path_A(x, U, L, C, tc)
            elif strategy_id == 2:
                cr, s1, s2, n_steps = _eval_path_B(x, U, L, tc)
            else:
                cr, s1, s2, n_steps = _eval_path_C(x, U, L, C, tc)
            
            sr = _sharpe_from_sums(s1, s2, n_steps)
            cr_sum += cr
            sr_sum += sr
        
        CR_out[k] = cr_sum / N
        SR_out[k] = sr_sum / N
    
    return CR_out, SR_out


# =============================================================================
# Model Simulators (Models 1-5 from Zhang 2021)
# =============================================================================

@njit
def _simulate_model1(T: int, x0: float, eta: np.ndarray) -> np.ndarray:
    """Model 1: Linear + Gaussian + Homoscedastic"""
    theta1 = 0.9590
    sigma = 0.0049
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        x[t + 1] = theta1 * x[t] + sigma * eta[t]
    return x


@njit
def _simulate_model2(T: int, x0: float, eta: np.ndarray) -> np.ndarray:
    """Model 2: Nonlinear + Gaussian"""
    theta1 = 0.9
    theta2 = 0.2590
    sigma = 0.0049
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        # Clip to prevent explosion
        x_clipped = max(min(x[t], 1.0), -1.0)
        x[t + 1] = theta1 * x_clipped + theta2 * x_clipped**2 + sigma * eta[t]
    return x


@njit
def _simulate_model3(T: int, x0: float, eta: np.ndarray) -> np.ndarray:
    """Model 3: Linear + Heteroscedastic"""
    theta1 = 0.9590
    q_base = 0.00089
    q_het = 0.08
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        sigma_t = np.sqrt(q_base + q_het * x[t]**2)
        x[t + 1] = theta1 * x[t] + sigma_t * eta[t]
    return x


@njit
def _simulate_model4(T: int, x0: float, eta_t: np.ndarray) -> np.ndarray:
    """Model 4: Linear + t-distributed (nu=3)"""
    theta1 = 0.9590
    sigma = 0.0049 / np.sqrt(3)  # Scale for variance matching
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        x[t + 1] = theta1 * x[t] + sigma * eta_t[t]
    return x


@njit
def _simulate_model5(T: int, x0: float, eta_t: np.ndarray) -> np.ndarray:
    """Model 5: Nonlinear + t-distributed"""
    theta1 = 0.9
    theta2 = 0.2590
    sigma = 0.0049 / np.sqrt(3)
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        # Clip x[t] to prevent explosion from quadratic term
        x_clipped = max(min(x[t], 1.0), -1.0)
        x[t + 1] = theta1 * x_clipped + theta2 * x_clipped**2 + sigma * eta_t[t]
    return x


def simulate_paths(
    model: str,
    N: int,
    T: int,
    seed: int = 42,
    standardize: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Simulate N paths of length T for the specified model.
    
    Parameters
    ----------
    model : str
        One of 'model1', 'model2', 'model3', 'model4', 'model5'
    N : int
        Number of paths
    T : int
        Length of each path
    seed : int
        Random seed
    standardize : bool
        If True, standardize paths (for internal use)
        
    Returns
    -------
    paths : ndarray of shape (N, T)
        Simulated paths (raw, not standardized)
    C : float
        Global mean across ALL paths and time steps
    sigma : float
        Mean of per-path standard deviations (better matches paper results)
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((N, T))
    
    for i in range(N):
        if model in ['model1', 'model2', 'model3']:
            eta = rng.standard_normal(T)
        else:  # model4, model5 use t-distribution
            eta = rng.standard_t(df=3, size=T)
        
        if model == 'model1':
            x = _simulate_model1(T, 0.0, eta)
        elif model == 'model2':
            x = _simulate_model2(T, 0.0, eta)
        elif model == 'model3':
            x = _simulate_model3(T, 0.0, eta)
        elif model == 'model4':
            x = _simulate_model4(T, 0.0, eta)
        elif model == 'model5':
            x = _simulate_model5(T, 0.0, eta)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        paths[i] = x
    
    # Compute global mean across all paths and time steps
    C = float(paths.mean())
    
    # Compute MEAN of per-path standard deviations
    # This better matches paper results, especially for heteroscedastic models
    sigma = float(np.mean([np.std(paths[i]) for i in range(N)]))
    
    return paths, C, sigma


# =============================================================================
# Main Table 1 Replication Function
# =============================================================================

@dataclass
class Table1Result:
    """Result for one (model, strategy) combination."""
    model: str
    strategy: str
    U_star_cr: float
    L_star_cr: float
    CR: float
    U_star_sr: float
    L_star_sr: float
    SR: float


def run_table1_optimization(
    model: str,
    strategy: str,
    N: int = 10000,
    T: int = 1000,
    cost_bp: float = 20.0,
    seed: int = 42,
) -> Table1Result:
    """
    Run Table 1 optimization for one (model, strategy) combination.
    
    Parameters
    ----------
    model : str
        Model name ('model1' to 'model5')
    strategy : str
        Strategy ('A', 'B', or 'C')
    N : int
        Number of Monte Carlo simulations
    T : int
        Length of each simulation
    cost_bp : float
        Transaction cost in basis points (per asset)
    seed : int
        Random seed
        
    Returns
    -------
    Table1Result
        Optimal thresholds and performance metrics
    """
    # Simulate paths - returns global C and sigma
    paths, C, sigma_avg = simulate_paths(model, N, T, seed)
    
    # Grid definition (sigma units)
    U_grid = np.arange(0.1, 2.55, 0.1)  # [0.1, 0.2, ..., 2.5]
    L_grid = np.arange(-2.5, -0.05, 0.1)  # [-2.5, -2.4, ..., -0.1]
    
    # Transaction cost: 20bp per trade
    # Paper says "20bp per asset, 40bp per complete trading (round-trip)"
    # This means each single trade (open OR close) costs 20bp
    tc = cost_bp / 10000
    
    # Strategy ID
    strategy_id = {'A': 1, 'B': 2, 'C': 3}[strategy.upper()]
    
    # Run grid search
    CR_arr, SR_arr = _grid_search_numba(paths, C, sigma_avg, U_grid, L_grid, strategy_id, tc)
    
    # Find optimal for CR
    k_cr = np.nanargmax(CR_arr)
    i_cr = k_cr // len(L_grid)
    j_cr = k_cr % len(L_grid)
    
    # Find optimal for SR
    k_sr = np.nanargmax(SR_arr)
    i_sr = k_sr // len(L_grid)
    j_sr = k_sr % len(L_grid)
    
    return Table1Result(
        model=model,
        strategy=strategy,
        U_star_cr=U_grid[i_cr],
        L_star_cr=L_grid[j_cr],
        CR=CR_arr[k_cr],
        U_star_sr=U_grid[i_sr],
        L_star_sr=L_grid[j_sr],
        SR=SR_arr[k_sr],
    )


def replicate_table1(
    N: int = 10000,
    T: int = 1000,
    cost_bp: float = 20.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[Tuple[str, str], Table1Result]:
    """
    Replicate full Table 1 from Zhang (2021).
    
    Parameters
    ----------
    N : int
        Number of Monte Carlo simulations
    T : int
        Length of each simulation
    cost_bp : float
        Transaction cost in basis points
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Dictionary mapping (model, strategy) to Table1Result
    """
    import time
    
    models = ['model1', 'model2', 'model3', 'model4', 'model5']
    strategies = ['A', 'B', 'C']
    
    results = {}
    
    if verbose:
        print(f"Replicating Table 1: N={N}, T={T}, cost={cost_bp}bp")
        print(f"Numba available: {NUMBA_AVAILABLE}")
        print("=" * 70)
    
    total_start = time.time()
    
    for model in models:
        if verbose:
            print(f"\n{model.upper()}")
        
        model_start = time.time()
        
        # Simulate paths once per model - returns GLOBAL C and sigma
        paths, C, sigma_avg = simulate_paths(model, N, T, seed)
        
        if verbose:
            print(f"  Paths simulated in {time.time() - model_start:.1f}s (C={C:.6f}, σ={sigma_avg:.6f})")
        
        # Grid (in sigma units)
        U_grid = np.arange(0.1, 2.55, 0.1)
        L_grid = np.arange(-2.5, -0.05, 0.1)
        # Transaction cost: 20bp per trade (paper: "40bp per complete trading" = round-trip)
        tc = cost_bp / 10000
        
        for strategy in strategies:
            strat_start = time.time()
            strategy_id = {'A': 1, 'B': 2, 'C': 3}[strategy]
            
            CR_arr, SR_arr = _grid_search_numba(paths, C, sigma_avg, U_grid, L_grid, strategy_id, tc)
            
            k_cr = np.nanargmax(CR_arr)
            i_cr, j_cr = k_cr // len(L_grid), k_cr % len(L_grid)
            
            k_sr = np.nanargmax(SR_arr)
            i_sr, j_sr = k_sr // len(L_grid), k_sr % len(L_grid)
            
            res = Table1Result(
                model=model,
                strategy=strategy,
                U_star_cr=U_grid[i_cr],
                L_star_cr=L_grid[j_cr],
                CR=CR_arr[k_cr],
                U_star_sr=U_grid[i_sr],
                L_star_sr=L_grid[j_sr],
                SR=SR_arr[k_sr],
            )
            results[(model, strategy)] = res
            
            if verbose:
                print(f"  Strategy {strategy} ({time.time() - strat_start:.1f}s): "
                      f"CR: U*={res.U_star_cr:.1f}σ, L*={res.L_star_cr:.1f}σ, CR={res.CR:.4f} | "
                      f"SR: U*={res.U_star_sr:.1f}σ, L*={res.L_star_sr:.1f}σ, SR={res.SR:.4f}")
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✅ Table 1 complete in {time.time() - total_start:.1f}s")
    
    return results


# Reference values from paper
PAPER_TABLE1 = {
    ('model1', 'A'): {'U_cr': 0.7, 'L_cr': -0.7, 'CR': 0.3868, 'U_sr': 1.1, 'L_sr': -1.1, 'SR': 0.0882},
    ('model1', 'B'): {'U_cr': 0.5, 'L_cr': -0.5, 'CR': 0.4245, 'U_sr': 0.5, 'L_sr': -0.5, 'SR': 0.0807},
    ('model1', 'C'): {'U_cr': 1.0, 'L_cr': -1.0, 'CR': 0.2990, 'U_sr': 0.9, 'L_sr': -0.9, 'SR': 0.1044},
    ('model2', 'A'): {'U_cr': 0.8, 'L_cr': -0.8, 'CR': 0.5562, 'U_sr': 1.2, 'L_sr': -1.3, 'SR': 0.1308},
    ('model2', 'B'): {'U_cr': 0.6, 'L_cr': -0.6, 'CR': 0.6085, 'U_sr': 0.6, 'L_sr': -0.6, 'SR': 0.1203},
    ('model2', 'C'): {'U_cr': 1.2, 'L_cr': -1.3, 'CR': 0.3300, 'U_sr': 1.2, 'L_sr': -1.3, 'SR': 0.1163},
    ('model3', 'A'): {'U_cr': 0.3, 'L_cr': -0.2, 'CR': 3.9413, 'U_sr': 0.4, 'L_sr': -0.4, 'SR': 0.0751},
    ('model3', 'B'): {'U_cr': 0.1, 'L_cr': -0.1, 'CR': 4.0139, 'U_sr': 0.1, 'L_sr': -0.1, 'SR': 0.0743},
    ('model3', 'C'): {'U_cr': 0.8, 'L_cr': -0.8, 'CR': 6.6763, 'U_sr': 0.1, 'L_sr': -0.1, 'SR': 0.2499},
    ('model4', 'A'): {'U_cr': 0.6, 'L_cr': -0.6, 'CR': 0.3792, 'U_sr': 1.0, 'L_sr': -1.0, 'SR': 0.0881},
    ('model4', 'B'): {'U_cr': 0.4, 'L_cr': -0.5, 'CR': 0.4071, 'U_sr': 0.5, 'L_sr': -0.5, 'SR': 0.0782},
    ('model4', 'C'): {'U_cr': 1.0, 'L_cr': -1.0, 'CR': 0.2243, 'U_sr': 1.0, 'L_sr': -1.0, 'SR': 0.0829},
    ('model5', 'A'): {'U_cr': 0.7, 'L_cr': -0.7, 'CR': 0.5359, 'U_sr': 1.2, 'L_sr': -1.2, 'SR': 0.1293},
    ('model5', 'B'): {'U_cr': 0.5, 'L_cr': -0.5, 'CR': 0.5760, 'U_sr': 0.5, 'L_sr': -0.5, 'SR': 0.1145},
    ('model5', 'C'): {'U_cr': 1.2, 'L_cr': -1.2, 'CR': 0.2423, 'U_sr': 1.4, 'L_sr': -1.4, 'SR': 0.0961},
}
