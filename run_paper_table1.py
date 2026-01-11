# run_paper_table1.py
from __future__ import annotations

import numpy as np
import pandas as pd

from optimize_rules import grid_search_UL
from paper_models import simulate_model1, simulate_model2, simulate_model3


def make_grids(C: float, sd: float):
    """
    Paper grid in sigma units:
      U_sigma in [0.1, 2.5], step 0.1
      L_sigma in [-2.5, -0.1], step 0.1
    """
    U_sig = np.arange(0.1, 2.5 + 1e-12, 0.1)
    L_sig = np.arange(-2.5, -0.1 + 1e-12, 0.1)
    U_grid = C + sd * U_sig
    L_grid = C + sd * L_sig
    return U_grid, L_grid, U_sig, L_sig


def _run_model(sim_func, model_name: str, T: int, N: int, seed: int, ann_factor: int, tc_complete: float):
    """
    Runs the 2 optimizations (CR then SR) for strategies A/B/C, returns rows for the final table.
    """
    paths = sim_func(T=T, N=N, seed=seed)  # (N, T)
    C = float(paths.mean())
    sd = float(paths.std(ddof=1))

    U_grid, L_grid, U_sig, L_sig = make_grids(C, sd)

    rows = []
    for strat in ["A", "B", "C"]:
        # Optimize CR
        best_CR = grid_search_UL(paths, strat, C, U_grid, L_grid, objective="CR",
                                 ann_factor=ann_factor, tc_complete=tc_complete)
        U_star_CR_sig = float(U_sig[best_CR["iU"]])
        L_star_CR_sig = float(L_sig[best_CR["iL"]])

        # Optimize SR
        best_SR = grid_search_UL(paths, strat, C, U_grid, L_grid, objective="SR",
                                 ann_factor=ann_factor, tc_complete=tc_complete)
        U_star_SR_sig = float(U_sig[best_SR["iU"]])
        L_star_SR_sig = float(L_sig[best_SR["iL"]])

        rows.append({
            "Model": model_name,
            "Strategy": strat,
            "U*_CR": U_star_CR_sig,
            "L*_CR": L_star_CR_sig,
            "CR": float(best_CR["CR"]),
            "U*_SR": U_star_SR_sig,
            "L*_SR": L_star_SR_sig,
            "SR": float(best_SR["SR"]),
        })

    return rows


def run_paper_table1(
    T: int = 1000,
    N: int = 2000,
    seed: int = 1,
    ann_factor: int = 252,
    tc_complete: float = 0.004,
) -> pd.DataFrame:
    """
    Returns a DataFrame matching the Table 1 structure for Models 1-3.
    """
    all_rows = []
    all_rows += _run_model(simulate_model1, "Model 1", T, N, seed, ann_factor, tc_complete)
    all_rows += _run_model(simulate_model2, "Model 2", T, N, seed, ann_factor, tc_complete)
    all_rows += _run_model(simulate_model3, "Model 3", T, N, seed, ann_factor, tc_complete)

    df = pd.DataFrame(all_rows, columns=["Model", "Strategy", "U*_CR", "L*_CR", "CR", "U*_SR", "L*_SR", "SR"])
    return df
