import numpy as np
import pandas as pd

def estimate_gamma_ols(PA: pd.Series, PB: pd.Series) -> float:
    """
    OLS of PA on PB with intercept:
      PA = alpha + gamma * PB + error
    Return gamma.
    """
    X = np.column_stack([np.ones(len(PB)), PB.values])
    y = PA.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    gamma = float(beta[1])
    return gamma

def compute_spread(PA: pd.Series, PB: pd.Series, gamma: float) -> pd.Series:
    return PA - gamma * PB


def backtest_pair(PA: pd.Series, PB: pd.Series, gamma: float, signal: pd.Series, tc_bps: float = 20.0) -> pd.DataFrame:
    """
    Compute daily P&L for a self-financing portfolio:
      position_t = signal_t * [ +1 share A  - gamma shares B ] for long-spread
                 = signal_t * [ -1 share A  + gamma shares B ] for short-spread (since signal=-1)
    Equivalent:
      pnl_t = signal_{t-1} * (dPA - gamma*dPB)

    Transaction cost:
      apply cost when signal changes (enter/exit/flip)
      paper uses 20bp per asset per transaction; round-trip includes both legs.
    We'll approximate: cost = (abs(delta_signal) > 0) * 2 legs * tc
    """
    df = pd.DataFrame({"PA": PA, "PB": PB, "signal": signal}).dropna()
    df["dPA"] = df["PA"].diff()
    df["dPB"] = df["PB"].diff()

    df["signal_lag"] = df["signal"].shift(1).fillna(0)

    # raw pnl in price units
    df["pnl"] = df["signal_lag"] * (df["dPA"] - gamma * df["dPB"])

    # transaction cost approximation
    tc = tc_bps / 10000.0
    delta = df["signal"].diff().abs().fillna(0)

    # 2 legs: A and B. Use notional approx based on prices.
    # Cost in price units: tc*(|trade in A|*PA + |trade in B|*PB*gamma)
    traded_A = delta  # 1 share
    traded_B = delta * abs(gamma)

    df["tcost"] = tc * (traded_A * df["PA"] + traded_B * df["PB"])
    df["pnl_net"] = df["pnl"] - df["tcost"]

    df["cum_pnl_net"] = df["pnl_net"].cumsum()
    return df
