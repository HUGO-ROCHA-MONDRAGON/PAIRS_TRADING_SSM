import pandas as pd
import matplotlib.pyplot as plt

from data_loader import extract_pair
from strategies import strategy_A_signals
from backtest import backtest_pair
from backtest import estimate_gamma_ols, compute_spread  # or wherever you placed them

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
U = mu + 1.0 * sd
L = mu - 1.0 * sd
C = mu

signal = strategy_A_signals(spread, U=U, L=L, C=C)
bt = backtest_pair(pair["PA"], pair["PB"], gamma, signal, tc_bps=20)

print("gamma:", gamma)
print("Final net PnL (price units):", bt["cum_pnl_net"].iloc[-1])

# Plot spread + thresholds + signals
fig, ax = plt.subplots()
ax.plot(spread.index, spread.values, label="spread")
ax.axhline(U, linestyle="--", label="U")
ax.axhline(L, linestyle="--", label="L")
ax.axhline(C, linestyle="-", label="mean")

# mark entries
entries = signal.diff().fillna(0) != 0
ax.scatter(spread.index[entries], spread[entries], s=10, label="signal change")
ax.legend()
plt.show()
