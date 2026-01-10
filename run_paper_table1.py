import numpy as np
from optimize_rules import grid_search_UL
from paper_models import simulate_model1, simulate_model2, simulate_model3

def make_grids(C, sd):
    U_sig = np.r_[np.arange(0.0, 1.21, 0.1), 1.25]      # include 1.25 exactly
    L_sig = np.arange(-2.5, -0.1 + 1e-9, 0.1)
    U_grid = C + sd * U_sig
    L_grid = C + sd * L_sig
    return U_grid, L_grid

def run_one_model(sim_func, model_name, T=1000, N=2000, seed=1, objective="SR"):
    paths = sim_func(T=T, N=N, seed=seed)  # (N, T)
    C = float(paths.mean())
    sd = float(paths.std(ddof=1))
    U_grid, L_grid = make_grids(C, sd)

    bestA = grid_search_UL(paths, "A", C, U_grid, L_grid, objective=objective)
    bestB = grid_search_UL(paths, "B", C, U_grid, L_grid, objective=objective)
    bestC = grid_search_UL(paths, "C", C, U_grid, L_grid, objective=objective)

    print(f"\n=== {model_name} objective={objective} ===")
    print("Best A:", bestA)
    print("Best B:", bestB)
    print("Best C:", bestC)

def main():
    T = 1000
    N = 2000  # debug; later 10_000
    seed = 1

    for obj in ["CR", "SR"]:
        run_one_model(simulate_model1, "Model 1", T=T, N=N, seed=seed, objective=obj)
        run_one_model(simulate_model2, "Model 2", T=T, N=N, seed=seed, objective=obj)
        run_one_model(simulate_model3, "Model 3", T=T, N=N, seed=seed, objective=obj)

if __name__ == "__main__":
    main()
