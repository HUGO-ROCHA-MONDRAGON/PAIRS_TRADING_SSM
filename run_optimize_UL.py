# run_optimize_UL.py
import numpy as np
import pandas as pd

from data_loader import extract_pair
from backtest import estimate_gamma_ols, compute_spread, backtest_pair
from simulation import fit_ar1, simulate_ar1_paths
from optimize_rules import grid_search_UL
from strategies import strategy_A_signals, strategy_B_signals, strategy_C_signals
from kalman_model1 import fit_kf_model1
from plots import plot_spread_with_signals


def main():
    # -----------------------------
    # 1) Load data + build pair
    # -----------------------------
    df = pd.read_excel("dataGQ.xlsx")
    

    pair = extract_pair(df, "PEP US Equity", "KO US Equity")
    pair = pair.loc["2012-01-03":"2019-06-28"]

    gamma = estimate_gamma_ols(pair["PA"], pair["PB"])
    spread = compute_spread(pair["PA"], pair["PB"], gamma)

    print("Sample:", pair.index.min(), "->", pair.index.max(), pair.shape)
    print("gamma:", gamma)

    SIM_T = 1000
    N_PATHS = 2000
    U_SIGMA_GRID = np.arange(0.0, 1.25 + 1e-9, 0.1)  # [0, 1.25] step 0.1
    L_SIGMA_GRID = np.arange(-2.5, -0.1 + 1e-9, 0.1)  # [-2.5, -0.1] step 0.1
    SEED = 1
    TC_BPS_PER_ASSET = 20

    # -----------------------------
    # 2) Model I: Kalman filter -> x_hat
    # -----------------------------
    y = spread.copy()
    kf_res = fit_kf_model1(y, verbose=True)
    x_hat = kf_res.x_filt  # filtered latent spread

    # -----------------------------
    # 3) Optimize U/L using x_hat (simulation-based rule selection)
    # -----------------------------
    C_opt = float(x_hat.mean())
    sd_opt = float(x_hat.std(ddof=1))

    c, phi, sigma = fit_ar1(x_hat)
    print("AR(1) on x_hat: c=", c, "phi=", phi, "sigma=", sigma)

    paths = simulate_ar1_paths(
        c, phi, sigma,
        n_steps=SIM_T,
        n_paths=N_PATHS,
        x0=float(x_hat.iloc[0]),
        seed=SEED
    )

    U_grid = C_opt + sd_opt * U_SIGMA_GRID
    L_grid = C_opt + sd_opt * L_SIGMA_GRID

    best_A = grid_search_UL(paths, strategy="A", C=C_opt, U_grid=U_grid, L_grid=L_grid, objective="SR")
    best_B = grid_search_UL(paths, strategy="B", C=C_opt, U_grid=U_grid, L_grid=L_grid, objective="SR")
    best_C = grid_search_UL(paths, strategy="C", C=C_opt, U_grid=U_grid, L_grid=L_grid, objective="SR")


    print("Best A (x_hat):", best_A)
    print("Best B (x_hat):", best_B)
    print("Best C (x_hat):", best_C)

    print("A/C sigma units:",
          (best_A["U"] - C_opt) / sd_opt,
          (best_A["L"] - C_opt) / sd_opt)

    print("B sigma units:",
          (best_B["U"] - C_opt) / sd_opt,
          (best_B["L"] - C_opt) / sd_opt)

    # -----------------------------
    # 4) Backtest on real prices using signals from x_hat
    # -----------------------------
    sig_A = strategy_A_signals(x_hat, U=best_A["U"], L=best_A["L"], C=C_opt)
    sig_B = strategy_B_signals(x_hat, U=best_B["U"], L=best_B["L"])
    sig_C = strategy_C_signals(x_hat, U=best_C["U"], L=best_C["L"], C=C_opt)

    bt_A = backtest_pair(pair["PA"], pair["PB"], gamma, sig_A, tc_bps=20)
    bt_B = backtest_pair(pair["PA"], pair["PB"], gamma, sig_B, tc_bps=20)
    bt_C = backtest_pair(pair["PA"], pair["PB"], gamma, sig_C, tc_bps=20)

    print("Real PnL A (x_hat rules):", bt_A["cum_pnl_net"].iloc[-1])
    print("Real PnL B (x_hat rules):", bt_B["cum_pnl_net"].iloc[-1])
    print("Real PnL C (x_hat rules):", bt_C["cum_pnl_net"].iloc[-1])

    # -----------------------------
    # 5) Plot (Model I + optimal U/L, Strategy C)
    # -----------------------------
    plot_spread_with_signals(
        x_hat,
        best_C["U"],
        best_C["L"],
        C_opt,
        sig_C,
        "Strategy C on filtered spread (Model I) + optimal U/L"
    )


if __name__ == "__main__":
    main()
