# optimize_rules.py
from __future__ import annotations
import numpy as np

# Try to use numba for speed. If not available, fallback to Python loops.
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(func=None, **kwargs):
        """Fallback decorator when numba is not available."""
        def decorator(f):
            return f
        if func is not None:
            # Called as @njit without arguments
            return func
        # Called as @njit(...) with arguments
        return decorator
    def prange(x):  # type: ignore
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
def _eval_path_A(x: np.ndarray, U: float, L: float, C: float, tc_complete: float):
    T = x.shape[0]
    n_steps = T - 1
    pos = 0

    cr = 0.0
    sum_p = 0.0
    sum_p2 = 0.0

    for t in range(1, T):
        dx = x[t] - x[t - 1]
        pnl = pos * dx

        new_pos = pos
        if pos == 0:
            if x[t] >= U:
                new_pos = -1
            elif x[t] <= L:
                new_pos = +1
        else:
            if (pos == -1 and x[t] <= C) or (pos == +1 and x[t] >= C):
                new_pos = 0

        if new_pos != pos:
            pnl -= tc_complete * abs(new_pos - pos)

        pos = new_pos

        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl

    return cr, sum_p, sum_p2, n_steps


@njit
def _eval_path_B(x: np.ndarray, U: float, L: float, tc_complete: float):
    T = x.shape[0]
    n_steps = T - 1
    pos = 0

    cr = 0.0
    sum_p = 0.0
    sum_p2 = 0.0

    for t in range(1, T):
        dx = x[t] - x[t - 1]
        pnl = pos * dx

        prev = x[t - 1]
        cur = x[t]

        new_pos = pos
        if prev < U and cur >= U:
            new_pos = -1
        elif prev > L and cur <= L:
            new_pos = +1

        if new_pos != pos:
            pnl -= tc_complete * abs(new_pos - pos)

        pos = new_pos

        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl

    return cr, sum_p, sum_p2, n_steps


@njit
def _eval_path_C(x: np.ndarray, U: float, L: float, C: float, tc_complete: float):
    T = x.shape[0]
    n_steps = T - 1
    pos = 0

    cr = 0.0
    sum_p = 0.0
    sum_p2 = 0.0

    for t in range(1, T):
        dx = x[t] - x[t - 1]
        pnl = pos * dx

        prev = x[t - 1]
        cur = x[t]

        # Entry: outside -> inside
        entry_short = (prev > U and cur <= U)   # cross U from above
        entry_long  = (prev < L and cur >= L)   # cross L from below

        # Profit at mean
        cross_down_C = (prev > C and cur <= C)
        cross_up_C   = (prev < C and cur >= C)

        # Stop: inside -> outside
        stop_short = (prev < U and cur >= U)    # cross U from below
        stop_long  = (prev > L and cur <= L)    # cross L from above

        new_pos = pos
        if pos == 0:
            if entry_short:
                new_pos = -1
            elif entry_long:
                new_pos = +1
        elif pos == -1:
            if cross_down_C or stop_short:
                new_pos = 0
        else:  # pos == +1
            if cross_up_C or stop_long:
                new_pos = 0

        if new_pos != pos:
            pnl -= tc_complete * abs(new_pos - pos)

        pos = new_pos

        cr += pnl
        sum_p += pnl
        sum_p2 += pnl * pnl

    return cr, sum_p, sum_p2, n_steps



@njit(parallel=True)
def _grid_metrics(
    paths: np.ndarray,      # (N, T)
    U_grid: np.ndarray,
    L_grid: np.ndarray,
    C: float,
    strategy_id: int,       # 1=A,2=B,3=C
    ann_factor: int,
    tc_complete: float,
):
    N, T = paths.shape
    nU = U_grid.shape[0]
    nL = L_grid.shape[0]
    K = nU * nL

    CR_out = np.full(K, -np.inf)
    SR_out = np.full(K, -np.inf)

    for k in prange(K):
        i = k // nL
        j = k - i * nL
        U = U_grid[i]
        L = L_grid[j]

        if L >= C or U <= C or L >= U:
            continue

        cr_sum = 0.0
        sr_sum = 0.0

        for n in range(N):
            x = paths[n, :]

            if strategy_id == 1:
                cr, s1, s2, n_steps = _eval_path_A(x, U, L, C, tc_complete)
            elif strategy_id == 2:
                cr, s1, s2, n_steps = _eval_path_B(x, U, L, tc_complete)
            else:
                cr, s1, s2, n_steps = _eval_path_C(x, U, L, C, tc_complete)

            sr = _sharpe_from_sums(s1, s2, n_steps)

            cr_sum += cr
            sr_sum += sr

        CR_out[k] = cr_sum / N
        SR_out[k] = sr_sum / N

    return CR_out, SR_out


def grid_search_UL(
    paths: np.ndarray,          # (N, T)
    strategy: str,
    C: float,
    U_grid: np.ndarray,
    L_grid: np.ndarray,
    objective: str = "SR",
    ann_factor: int = 252,
    tc_complete: float = 0.004,  # 40bp per complete trading
) -> dict:
    """
    Returns best {U,L,CR,SR,score} + grid indices (iU,iL) to recover sigma-units.
    """
    strategy = strategy.upper()
    obj = objective.upper()
    if strategy not in {"A", "B", "C"}:
        raise ValueError("strategy must be 'A','B','C'")
    if obj not in {"SR", "CR"}:
        raise ValueError("objective must be 'SR' or 'CR'")

    strategy_id = 1 if strategy == "A" else (2 if strategy == "B" else 3)

    paths_ = np.asarray(paths, dtype=np.float64)
    U_ = np.asarray(U_grid, dtype=np.float64)
    L_ = np.asarray(L_grid, dtype=np.float64)

    CR_arr, SR_arr = _grid_metrics(paths_, U_, L_, float(C), strategy_id, int(ann_factor), float(tc_complete))

    score = SR_arr if obj == "SR" else CR_arr
    k_best = int(np.nanargmax(score))

    nL = len(L_grid)
    i = k_best // nL
    j = k_best - i * nL

    return {
        "U": float(U_grid[i]),
        "L": float(L_grid[j]),
        "CR": float(CR_arr[k_best]),
        "SR": float(SR_arr[k_best]),
        "score": float(score[k_best]),
        "iU": int(i),
        "iL": int(j),
        "objective": obj,
        "strategy": strategy,
        "numba": bool(NUMBA_AVAILABLE),
    }
