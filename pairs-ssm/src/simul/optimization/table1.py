import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

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


@njit
def _sharpe_from_sums(sum_p: float, sum_p2: float, n: int) -> float:
    if n <= 1:
        return 0.0
    mean = sum_p / n
    var = (sum_p2 - n * mean * mean) / (n - 1)
    if var <= 0.0:
        return 0.0
    return mean / np.sqrt(var)


@njit
def _eval_path_A(x: np.ndarray, U: float, L: float, C: float, tc: float) -> Tuple[float, float, float, int]:
    T = len(x)
    pos = 0
    
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
                new_pos = -1
            elif x[t] <= L:
                new_pos = 1
        elif pos == 1:
            if x[t] >= C:
                new_pos = 0
        elif pos == -1:
            if x[t] <= C:
                new_pos = 0
        
        if new_pos != pos:
            pnl -= tc * abs(new_pos - pos)
        
        pos = new_pos
        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl
    
    return cr, sum_p, sum_p2, n_steps


@njit
def _eval_path_B(x: np.ndarray, U: float, L: float, tc: float) -> Tuple[float, float, float, int]:
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
        if prev < U and cur >= U:
            new_pos = -1
        elif prev > L and cur <= L:
            new_pos = 1
        
        if new_pos != pos:
            pnl -= tc * abs(new_pos - pos)
        
        pos = new_pos
        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl
    
    return cr, sum_p, sum_p2, n_steps


@njit
def _eval_path_C(x: np.ndarray, U: float, L: float, C: float, tc: float) -> Tuple[float, float, float, int]:
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
        
        entry_short = (prev > U and cur <= U)
        entry_long = (prev < L and cur >= L)
        
        cross_down_C = (prev > C and cur <= C)
        cross_up_C = (prev < C and cur >= C)
        
        stop_short = (prev < U and cur >= U)
        stop_long = (prev > L and cur <= L)
        
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
        
        pnl = new_pos * dx
        
        if new_pos != pos:
            pnl -= tc * abs(new_pos - pos)
        
        pos = new_pos
        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl
    
    return cr, sum_p, sum_p2, n_steps


@njit(parallel=True)
def _grid_search_numba(
    paths: np.ndarray,
    C: float,
    sigma_avg: float,
    U_grid: np.ndarray,
    L_grid: np.ndarray,
    strategy_id: int,
    tc: float,
) -> Tuple[np.ndarray, np.ndarray]:
    N, T = paths.shape
    nU = len(U_grid)
    nL = len(L_grid)
    K = nU * nL
    
    CR_out = np.full(K, -np.inf)
    SR_out = np.full(K, -np.inf)
    
    for k in prange(K):
        i = k // nL
        j = k % nL
        u_sigma = U_grid[i]
        l_sigma = L_grid[j]
        
        if l_sigma >= 0 or u_sigma <= 0 or l_sigma >= u_sigma:
            continue
        
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


@njit
def _simulate_model1(T: int, x0: float, eta: np.ndarray) -> np.ndarray:
    theta1 = 0.9590
    sigma = 0.0049
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        x[t + 1] = theta1 * x[t] + sigma * eta[t]
    return x


@njit
def _simulate_model2(T: int, x0: float, eta: np.ndarray) -> np.ndarray:
    theta1 = 0.9
    theta2 = 0.2590
    sigma = 0.0049
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        x_clipped = max(min(x[t], 1.0), -1.0)
        x[t + 1] = theta1 * x_clipped + theta2 * x_clipped**2 + sigma * eta[t]
    return x


@njit
def _simulate_model3(T: int, x0: float, eta: np.ndarray) -> np.ndarray:
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
    theta1 = 0.9590
    sigma = 0.0049 / np.sqrt(3)
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        x[t + 1] = theta1 * x[t] + sigma * eta_t[t]
    return x


@njit
def _simulate_model5(T: int, x0: float, eta_t: np.ndarray) -> np.ndarray:
    theta1 = 0.9
    theta2 = 0.2590
    sigma = 0.0049 / np.sqrt(3)
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
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
    rng = np.random.default_rng(seed)
    paths = np.zeros((N, T))
    
    for i in range(N):
        if model in ['model1', 'model2', 'model3']:
            eta = rng.standard_normal(T)
        else:
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
    
    C = float(paths.mean())
    sigma = float(np.mean([np.std(paths[i]) for i in range(N)]))
    
    return paths, C, sigma


@dataclass
class Table1Result:
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
    paths, C, sigma_avg = simulate_paths(model, N, T, seed)
    
    U_grid = np.arange(0.1, 2.55, 0.1)
    L_grid = np.arange(-2.5, -0.05, 0.1)
    
    tc = cost_bp / 10000
    
    strategy_id = {'A': 1, 'B': 2, 'C': 3}[strategy.upper()]
    
    CR_arr, SR_arr = _grid_search_numba(paths, C, sigma_avg, U_grid, L_grid, strategy_id, tc)
    
    k_cr = np.nanargmax(CR_arr)
    i_cr = k_cr // len(L_grid)
    j_cr = k_cr % len(L_grid)
    
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
        
        paths, C, sigma_avg = simulate_paths(model, N, T, seed)
        
        if verbose:
            print(f"  Paths simulated in {time.time() - model_start:.1f}s (C={C:.6f}, σ={sigma_avg:.6f})")
        
        U_grid = np.arange(0.1, 2.55, 0.1)
        L_grid = np.arange(-2.5, -0.05, 0.1)
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

@njit
def _simulate_model5_noclip(T: int, x0: float, eta_t: np.ndarray) -> np.ndarray:
    theta1 = 0.9
    theta2 = 0.2590
    sigma = 0.0049 / np.sqrt(3)
    x = np.zeros(T)
    x[0] = x0
    for t in range(T - 1):
        x[t + 1] = theta1 * x[t] + theta2 * x[t]**2 + sigma * eta_t[t]
    return x

def _simulate_one_path_s1(model: str, T: int, rng: np.random.Generator, max_tries: int = 50) -> np.ndarray:
    for _ in range(max_tries):
        if model in ["A1", "A2"]:
            eta = rng.standard_normal(T)
        else:
            eta = rng.standard_t(df=3, size=T)

        if model == "A1":
            x = _simulate_model1(T, 0.0, eta)
        elif model == "A2":
            x = _simulate_model3(T, 0.0, eta)
        elif model == "A3":
            x = _simulate_model5_noclip(T, 0.0, eta)
        else:
            raise ValueError(f"Unknown S1 model: {model}")

        if np.isfinite(x).all():
            return x

    return x

def simulate_paths_S1(
    model: str,
    N: int,
    T: int,
    seed: int = 42,
) -> Tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    paths = np.zeros((N, T))

    for i in range(N):
        paths[i] = _simulate_one_path_s1(model, T, rng)

    mask = np.isfinite(paths).all(axis=1)
    paths = paths[mask]
    if paths.shape[0] == 0:
        raise RuntimeError("All simulated paths became non-finite (A3 exploded). Increase max_tries or adjust T.")

    C = float(paths.mean())
    sigma = float(np.mean([np.std(paths[i]) for i in range(paths.shape[0])]))

    return paths, C, sigma

@njit
def _eval_path_D(
    x: np.ndarray, U: float, L: float, C: float,
    C_minus: float, C_plus: float,
    tc: float
) -> Tuple[float, float, float, int]:
    T = len(x)
    pos = 0

    cr = 0.0
    sum_p = 0.0
    sum_p2 = 0.0
    n_steps = T - 1

    for t in range(1, T):
        dx = x[t] - x[t - 1]
        pnl = pos * dx

        prev = x[t - 1]
        cur = x[t]

        new_pos = pos
        if pos == 0:
            if cur >= U:
                new_pos = -1
            elif cur <= L:
                new_pos = 1
        elif pos == 1:
            if prev < C_plus and cur >= C_plus:
                new_pos = 0
        elif pos == -1:
            if prev > C_minus and cur <= C_minus:
                new_pos = 0

        if new_pos != pos:
            pnl -= tc * abs(new_pos - pos)

        pos = new_pos
        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl

    return cr, sum_p, sum_p2, n_steps


@njit
def _eval_path_E(
    x: np.ndarray, U: float, L: float, C: float,
    C_minus: float, C_plus: float,
    tc: float
) -> Tuple[float, float, float, int]:
    T = len(x)
    pos = 0

    cr = 0.0
    sum_p = 0.0
    sum_p2 = 0.0
    n_steps = T - 1

    for t in range(1, T):
        dx = x[t] - x[t - 1]
        prev = x[t - 1]
        cur = x[t]

        entry_short = (prev > U and cur <= U)
        entry_long  = (prev < L and cur >= L)

        tp_short = (prev > C_minus and cur <= C_minus)
        tp_long  = (prev < C_plus  and cur >= C_plus)

        stop_short = (prev < U and cur >= U)
        stop_long  = (prev > L and cur <= L)

        new_pos = pos
        if pos == 0:
            if entry_short:
                new_pos = -1
            elif entry_long:
                new_pos = 1
        elif pos == 1:
            if tp_long or stop_long:
                new_pos = 0
        elif pos == -1:
            if tp_short or stop_short:
                new_pos = 0

        pnl = new_pos * dx

        if new_pos != pos:
            pnl -= tc * abs(new_pos - pos)

        pos = new_pos
        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl

    return cr, sum_p, sum_p2, n_steps

@njit(parallel=True)
def _grid_search_numba_S1_DE(
    paths: np.ndarray,
    C: float,
    sigma_avg: float,
    U_grid: np.ndarray,
    delta_grid: np.ndarray,
    strategy_id: int,
    tc: float
) -> Tuple[np.ndarray, np.ndarray]:
    N, T = paths.shape
    nU = len(U_grid)
    nD = len(delta_grid)
    K = nU * nD

    CR_out = np.full(K, -np.inf)
    SR_out = np.full(K, -np.inf)

    for k in prange(K):
        i = k // nD
        j = k % nD

        u_sigma = U_grid[i]
        delta = delta_grid[j]

        if u_sigma <= 0:
            continue

        U_level = C + u_sigma * sigma_avg
        L_level = C - u_sigma * sigma_avg

        U_dist = U_level - C
        Delta = delta * U_dist
        C_minus = C - Delta
        C_plus  = C + Delta

        cr_sum = 0.0
        sr_sum = 0.0

        for n in range(N):
            x = paths[n, :]
            if not np.isfinite(x).all():
                continue

            if strategy_id == 4:
                cr, s1, s2, n_steps = _eval_path_D(x, U_level, L_level, C, C_minus, C_plus, tc)
            else:
                cr, s1, s2, n_steps = _eval_path_E(x, U_level, L_level, C, C_minus, C_plus, tc)

            sr = _sharpe_from_sums(s1, s2, n_steps)
            cr_sum += cr
            sr_sum += sr

        CR_out[k] = cr_sum / N
        SR_out[k] = sr_sum / N

    return CR_out, SR_out

@dataclass
class TableS1Result:
    model: str
    strategy: str
    U_star_cr: float
    delta_star_cr: float
    CR: float
    U_star_sr: float
    delta_star_sr: float
    SR: float


def run_tableS1_optimization(
    model: str,
    strategy: str,
    N: int = 10000,
    T: int = 1000,
    cost_bp: float = 20.0,
    seed: int = 42,
) -> TableS1Result:

    paths, C, sigma_avg = simulate_paths_S1(model, N, T, seed)

    U_grid = np.arange(0.1, 2.55, 0.1)
    delta_grid = np.arange(-0.5, 1.01, 0.1)

    tc = cost_bp / 10000
    strategy_id = {"D": 4, "E": 5}[strategy.upper()]

    CR_arr, SR_arr = _grid_search_numba_S1_DE(paths, C, sigma_avg, U_grid, delta_grid, strategy_id, tc)

    k_cr = np.nanargmax(CR_arr)
    i_cr, j_cr = k_cr // len(delta_grid), k_cr % len(delta_grid)

    k_sr = np.nanargmax(SR_arr)
    i_sr, j_sr = k_sr // len(delta_grid), k_sr % len(delta_grid)

    return TableS1Result(
        model=model,
        strategy=strategy.upper(),
        U_star_cr=float(U_grid[i_cr]),
        delta_star_cr=float(delta_grid[j_cr]),
        CR=float(CR_arr[k_cr]),
        U_star_sr=float(U_grid[i_sr]),
        delta_star_sr=float(delta_grid[j_sr]),
        SR=float(SR_arr[k_sr]),
    )

def replicate_tableS1(
    N: int = 10000,
    T: int = 1000,
    cost_bp: float = 20.0,
    seed: int = 42,
    verbose: bool = True,
):
    import time

    models = ["A1", "A2", "A3"]
    strategies = ["D", "E"]
    results = {}

    if verbose:
        print(f"Replicating Table S1: N={N}, T={T}, cost={cost_bp}bp")
        print(f"Numba available: {NUMBA_AVAILABLE}")
        print("=" * 70)

    t0 = time.time()

    for model in models:
        if verbose:
            print(f"\n{model}")
        paths, C, sigma_avg = simulate_paths_S1(model, N, T, seed)
        if verbose:
            print(f"  Paths simulated: N_eff={paths.shape[0]} (C={C:.6f}, σ={sigma_avg:.6f})")

        U_grid = np.arange(0.1, 2.55, 0.1)
        delta_grid = np.arange(-0.5, 1.01, 0.1)
        tc = cost_bp / 10000

        for strat in strategies:
            sid = {"D": 4, "E": 5}[strat]
            CR_arr, SR_arr = _grid_search_numba_S1_DE(paths, C, sigma_avg, U_grid, delta_grid, sid, tc)

            k_cr = np.nanargmax(CR_arr)
            i_cr, j_cr = k_cr // len(delta_grid), k_cr % len(delta_grid)

            k_sr = np.nanargmax(SR_arr)
            i_sr, j_sr = k_sr // len(delta_grid), k_sr % len(delta_grid)

            res = TableS1Result(
                model=model, strategy=strat,
                U_star_cr=float(U_grid[i_cr]), delta_star_cr=float(delta_grid[j_cr]), CR=float(CR_arr[k_cr]),
                U_star_sr=float(U_grid[i_sr]), delta_star_sr=float(delta_grid[j_sr]), SR=float(SR_arr[k_sr]),
            )
            results[(model, strat)] = res

            if verbose:
                print(f"  {strat}: "
                      f"CR: U*={res.U_star_cr:.1f}σ, δ*={res.delta_star_cr:.1f}, CR={res.CR:.4f} | "
                      f"SR: U*={res.U_star_sr:.1f}σ, δ*={res.delta_star_sr:.1f}, SR={res.SR:.4f}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"✅ Table S1 complete in {time.time() - t0:.1f}s")

    return results
