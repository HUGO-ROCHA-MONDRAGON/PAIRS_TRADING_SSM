import numpy as np
import pandas as pd

from data_loader import extract_pair
from backtest import estimate_gamma_ols, compute_spread, backtest_pair
from simulation import fit_ar1, simulate_ar1_paths
from optimize_rules import grid_search_UL
from strategies import strategy_A_signals, strategy_B_signals, strategy_C_signals

# Load data
df = pd.read_excel("dataGQ.xlsx")

# Pair + sample
pair = extract_pair(df, "PEP US Equity", "KO US Equity")
pair = pair.loc["2012-01-03":"2019-06-28"]

gamma = estimate_gamma_ols(pair["PA"], pair["PB"])
spread = compute_spread(pair["PA"], pair["PB"], gamma)

C = float(spread.mean())
sd = float(spread.std(ddof=1))

# Fit AR(1) to spread (proxy Model I)
c, phi, sigma = fit_ar1(spread)
print("AR(1): c=", c, "phi=", phi, "sigma=", sigma)

# Simulate
n_steps = len(spread)
paths = simulate_ar1_paths(c, phi, sigma, n_steps=n_steps, n_paths=300, x0=float(spread.iloc[0]), seed=1)

# Grid (relative to C and sd)
U_grid = C + sd * np.linspace(0.3, 2.0, 18)
L_grid = C - sd * np.linspace(0.3, 2.0, 18)

best_A = grid_search_UL(paths, strategy="A", C=C, U_grid=U_grid, L_grid=L_grid)
best_B = grid_search_UL(paths, strategy="B", C=C, U_grid=U_grid, L_grid=L_grid)
best_C = grid_search_UL(paths, strategy="C", C=C, U_grid=U_grid, L_grid=L_grid)

print("Best A:", best_A)
print("Best B:", best_B)
print("Best C:", best_C)

# Apply best thresholds on REAL data
U_A, L_A = best_A["U"], best_A["L"]
U_B, L_B = best_B["U"], best_B["L"]
U_C, L_C = best_C["U"], best_C["L"]

sig_A = strategy_A_signals(spread, U_A, L_A, C)
sig_B = strategy_B_signals(spread, U_B, L_B)
sig_C = strategy_C_signals(spread, U_C, L_C, C)

bt_A = backtest_pair(pair["PA"], pair["PB"], gamma, sig_A, tc_bps=20)
bt_B = backtest_pair(pair["PA"], pair["PB"], gamma, sig_B, tc_bps=20)
bt_C = backtest_pair(pair["PA"], pair["PB"], gamma, sig_C, tc_bps=20)

print("Real PnL A:", bt_A["cum_pnl_net"].iloc[-1])
print("Real PnL B:", bt_B["cum_pnl_net"].iloc[-1])
print("Real PnL C:", bt_C["cum_pnl_net"].iloc[-1])
