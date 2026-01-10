import pandas as pd

from data_loader import extract_pair
from strategies import strategy_A_signals,strategy_B_signals,strategy_C_signals
from backtest import backtest_pair
from backtest import estimate_gamma_ols, compute_spread  # or wherever you placed them
from plots import plot_spread_with_signals

def restrict_dates(df, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)]
    return df

# Load data
df = pd.read_excel("dataGQ.xlsx")

# Extract pair
pair = extract_pair(df, "PEP US Equity", "KO US Equity")

# Restrict to paper sample
pair = restrict_dates(pair, "2012-01-03", "2019-06-28")

# Sanity checks
print(pair.index.min(), pair.index.max(), pair.shape)
print(pair.head())
print(pair.isna().sum())

gamma = estimate_gamma_ols(pair["PA"], pair["PB"])
spread = compute_spread(pair["PA"], pair["PB"], gamma)

mu = float(spread.mean())
sd = float(spread.std(ddof=1))
#U = mu + 1.0 * sd
#L = mu - 1.0 * sd
U = mu + 0.3 * sd
L = mu - 0.3 * sd

C = mu

signal_A = strategy_A_signals(spread, U=U, L=L, C=C)
signal_B = strategy_B_signals(spread, U=U, L=L)
signal_C = strategy_C_signals(spread, U=U, L=L, C=C)

bt_A = backtest_pair(pair["PA"], pair["PB"], gamma, signal_A, tc_bps=20)
bt_B = backtest_pair(pair["PA"], pair["PB"], gamma, signal_B, tc_bps=20)
bt_C = backtest_pair(pair["PA"], pair["PB"], gamma, signal_C, tc_bps=20)

print("Final net PnL A:", bt_A["cum_pnl_net"].iloc[-1])
print("Final net PnL B:", bt_B["cum_pnl_net"].iloc[-1])
print("Final net PnL C:", bt_C["cum_pnl_net"].iloc[-1])

plot_spread_with_signals(spread, U, L, C, signal_A, "Strategy A (baseline)")
plot_spread_with_signals(spread, U, L, None, signal_B, "Strategy B (baseline)")
plot_spread_with_signals(spread, U, L, C, signal_C, "Strategy C (baseline)")