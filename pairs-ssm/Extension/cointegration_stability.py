# cointegration_stability.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# ============================================================
# CONFIG OBJECT
# ============================================================

@dataclass
class RollingConfig:
    start_date: str = "2012-01-03"
    end_date: str = "2019-06-28"
    formation_len: int = 252
    test_len: int = 252
    step: int = 63
    pval_threshold: float = 0.05
    adf_regression: str = "c"
    adf_autolag: str = "AIC"


# ============================================================
# DATA LOADING (compatible with your Excel structure)
# ============================================================

def load_price_series_from_excel(filepath: Path, ticker: str) -> pd.Series:
    """
    Loads a ticker series from an Excel file.

    Supports:
      1) Wide format: columns include 'Date' (or 'Unnamed: 0') and 'TICKER'
      2) Bloomberg-like: 'TICKER US Equity' with an adjacent date column

    Returns:
      pd.Series with DatetimeIndex.
    """
    df = pd.read_excel(filepath)

    # Case 1: wide format with ticker column
    if ticker in df.columns:
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif "Unnamed: 0" in df.columns:
            df = df.set_index("Unnamed: 0")
        df.index = pd.to_datetime(df.index)
        s = pd.to_numeric(df[ticker], errors="coerce").dropna().sort_index()
        return s

    # Case 2: Bloomberg-like columns
    candidates = [f"{ticker} US Equity", f"{ticker} US Equity "]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break

    if col is None:
        raise KeyError(f"Column for ticker '{ticker}' not found in {filepath.name}")

    col_idx = df.columns.get_loc(col)
    date_col = df.columns[col_idx - 1]

    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "price": pd.to_numeric(df[col], errors="coerce"),
        }
    ).dropna().drop_duplicates("date").set_index("date").sort_index()

    return tmp["price"]


def load_pair_prices(
    filepath: Path,
    a: str,
    b: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Returns aligned price DataFrame with columns [a, b] and date index."""
    pa = load_price_series_from_excel(filepath, a)
    pb = load_price_series_from_excel(filepath, b)

    idx = pa.index.intersection(pb.index)
    pa, pb = pa.loc[idx], pb.loc[idx]

    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    mask = (pa.index >= start) & (pa.index <= end)

    out = pd.DataFrame({a: pa.loc[mask], b: pb.loc[mask]}).dropna()
    return out


# ============================================================
# ENGLE–GRANGER HELPERS
# ============================================================

def fit_beta_ols(log_pa: np.ndarray, log_pb: np.ndarray) -> float:
    """OLS: log(PA) = alpha + beta log(PB) + e."""
    X = sm.add_constant(log_pb)
    res = sm.OLS(log_pa, X).fit()
    return float(res.params[1])


def residuals_from_beta(log_pa: np.ndarray, log_pb: np.ndarray, beta: float) -> np.ndarray:
    return log_pa - beta * log_pb


def adf_pvalue(x: np.ndarray, regression: str = "c", autolag: str = "AIC") -> float:
    """
    Standard ADF p-value on residuals.
    Note: EG has non-standard critical values in theory, but for
    'stability/non-persistence' demonstration, p-values are fine.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 40:
        return np.nan
    try:
        return float(adfuller(x, regression=regression, autolag=autolag)[1])
    except Exception:
        return np.nan


# ============================================================
# ROLLING NON-PERSISTENCE STUDY
# ============================================================

def rolling_cointegration_for_pair(
    prices: pd.DataFrame,
    a: str,
    b: str,
    cfg: RollingConfig,
) -> pd.DataFrame:
    """
    Rolling formation -> next test window.

    For each rolling start:
      - formation window: [t, t+formation_len)
      - test window:      [t+formation_len, t+formation_len+test_len)

    Compute:
      p_form: ADF p-value on formation residuals (beta fitted on formation)
      p_test_refit_beta: ADF p-value on test residuals (beta re-fitted on test)
      p_test_oos_beta:   ADF p-value on test residuals (beta fixed = formation beta)
    """
    logp = np.log(prices[[a, b]]).dropna()
    n = len(logp)

    rows: List[Dict] = []
    Lf, Lt, step = cfg.formation_len, cfg.test_len, cfg.step

    for start in range(0, n - (Lf + Lt) + 1, step):
        mid = start + Lf
        end = mid + Lt

        form = logp.iloc[start:mid]
        test = logp.iloc[mid:end]

        beta_form = fit_beta_ols(form[a].values, form[b].values)
        e_form = residuals_from_beta(form[a].values, form[b].values, beta_form)
        p_form = adf_pvalue(e_form, regression=cfg.adf_regression, autolag=cfg.adf_autolag)

        # optimistic: refit beta on test window
        beta_test = fit_beta_ols(test[a].values, test[b].values)
        e_test_refit = residuals_from_beta(test[a].values, test[b].values, beta_test)
        p_test_refit = adf_pvalue(e_test_refit, regression=cfg.adf_regression, autolag=cfg.adf_autolag)

        # realistic: keep formation beta (OOS)
        e_test_oos = residuals_from_beta(test[a].values, test[b].values, beta_form)
        p_test_oos = adf_pvalue(e_test_oos, regression=cfg.adf_regression, autolag=cfg.adf_autolag)

        rows.append(
            {
                "pair": f"{a}-{b}",
                "formation_start": form.index[0],
                "formation_end": form.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "beta_form": beta_form,
                "beta_test": beta_test,
                "p_form": p_form,
                "p_test_refit_beta": p_test_refit,
                "p_test_oos_beta": p_test_oos,
                "form_coint_5%": bool(np.isfinite(p_form) and (p_form <= cfg.pval_threshold)),
                "persist_refit_5%": bool(np.isfinite(p_form) and np.isfinite(p_test_refit)
                                         and (p_form <= cfg.pval_threshold) and (p_test_refit <= cfg.pval_threshold)),
                "persist_oos_5%": bool(np.isfinite(p_form) and np.isfinite(p_test_oos)
                                       and (p_form <= cfg.pval_threshold) and (p_test_oos <= cfg.pval_threshold)),
            }
        )

    return pd.DataFrame(rows)


def run_rolling_cointegration(
    filepath: Path,
    pairs: List[Tuple[str, str]],
    cfg: Optional[RollingConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs rolling non-persistence test for multiple pairs.
    Returns:
      - df_all: block-level results
      - df_summary: persistence summary per pair
    """
    if cfg is None:
        cfg = RollingConfig()

    all_dfs = []
    for a, b in pairs:
        prices = load_pair_prices(filepath, a, b, cfg.start_date, cfg.end_date)
        df_pair = rolling_cointegration_for_pair(prices, a, b, cfg)
        all_dfs.append(df_pair)

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    df_summary = summarize_persistence(df_all, cfg)
    return df_all, df_summary


def summarize_persistence(df_all: pd.DataFrame, cfg: RollingConfig) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame()

    out = []
    for pair, d in df_all.groupby("pair"):
        n_blocks = len(d)
        n_form = int(d["form_coint_5%"].sum())
        persist_refit = int(d["persist_refit_5%"].sum())
        persist_oos = int(d["persist_oos_5%"].sum())

        out.append(
            {
                "pair": pair,
                "n_blocks": n_blocks,
                "n_form_cointegrated_5%": n_form,
                "persistence_rate_next_refit_beta": (persist_refit / n_form) if n_form else np.nan,
                "persistence_rate_next_oos_beta": (persist_oos / n_form) if n_form else np.nan,
                "pval_threshold": cfg.pval_threshold,
                "formation_len": cfg.formation_len,
                "test_len": cfg.test_len,
                "step": cfg.step,
            }
        )

    return pd.DataFrame(out).sort_values("pair")


# ============================================================
# PLOTTING
# ============================================================

def plot_pvalues(
    df_all: pd.DataFrame,
    pair: str,
    cfg: RollingConfig,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot formation p-values vs next-period p-values (refit vs OOS beta) for one pair.
    Returns the matplotlib axis to display in notebook.
    """
    d = df_all[df_all["pair"] == pair].copy()
    if d.empty:
        raise ValueError(f"No rows for pair={pair}")

    x = d["formation_end"]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, d["p_form"], marker="o", label="ADF p-value (formation)")
    ax.plot(x, d["p_test_refit_beta"], marker="o", label="ADF p-value (next, refit β)")
    ax.plot(x, d["p_test_oos_beta"], marker="o", label="ADF p-value (next, OOS β fixed)")
    ax.axhline(cfg.pval_threshold, linestyle="--", label=f"threshold {cfg.pval_threshold:.2f}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{pair} — Rolling cointegration (non-persistence)")
    ax.set_xlabel("End of formation window")
    ax.set_ylabel("ADF p-value on residuals")
    ax.legend()
    return ax
