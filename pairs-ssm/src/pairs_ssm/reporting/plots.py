"""
Plotting functions for pairs trading analysis.

Creates publication-quality figures.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

# Import matplotlib lazily to avoid issues
_plt = None
_mpl_dates = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_mdates():
    global _mpl_dates
    if _mpl_dates is None:
        import matplotlib.dates as mdates
        _mpl_dates = mdates
    return _mpl_dates


def plot_spread(
    spread: pd.Series,
    thresholds: Optional[Tuple[float, float, float]] = None,
    title: str = "Spread",
    figsize: Tuple[int, int] = (12, 6),
    ax=None,
):
    """
    Plot spread with optional thresholds.
    
    Parameters
    ----------
    spread : pd.Series
        Spread series
    thresholds : tuple, optional
        (U, L, C) threshold values
    title : str
        Plot title
    figsize : tuple
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(spread.index, spread.values, "b-", linewidth=0.8, label="Spread")
    
    if thresholds is not None:
        U, L, C = thresholds
        ax.axhline(U, color="r", linestyle="--", linewidth=1, label=f"U = {U:.4f}")
        ax.axhline(L, color="g", linestyle="--", linewidth=1, label=f"L = {L:.4f}")
        ax.axhline(C, color="gray", linestyle="-", linewidth=1, label=f"C = {C:.4f}")
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_filtered_vs_observed(
    observed: pd.Series,
    filtered: pd.Series,
    thresholds: Optional[Tuple[float, float, float]] = None,
    title: str = "Observed vs Filtered Spread",
    figsize: Tuple[int, int] = (12, 6),
    ax=None,
):
    """
    Plot observed and filtered spread together.
    
    Parameters
    ----------
    observed : pd.Series
        Observed spread
    filtered : pd.Series
        Filtered spread estimate
    thresholds : tuple, optional
        (U, L, C) values
    title : str
        Plot title
    figsize : tuple
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(observed.index, observed.values, "b-", alpha=0.5, linewidth=0.5, label="Observed")
    ax.plot(filtered.index, filtered.values, "r-", linewidth=1.0, label="Filtered")
    
    if thresholds is not None:
        U, L, C = thresholds
        ax.axhline(U, color="darkred", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(L, color="darkgreen", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(C, color="gray", linestyle="-", linewidth=1, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_cumulative_pnl(
    pnl: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Cumulative P&L",
    figsize: Tuple[int, int] = (12, 5),
    ax=None,
):
    """
    Plot cumulative P&L.
    
    Parameters
    ----------
    pnl : pd.Series
        Daily P&L series
    benchmark : pd.Series, optional
        Benchmark P&L for comparison
    title : str
        Plot title
    figsize : tuple
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    cum_pnl = pnl.cumsum()
    ax.plot(cum_pnl.index, cum_pnl.values, "b-", linewidth=1.5, label="Strategy")
    
    if benchmark is not None:
        cum_bench = benchmark.cumsum()
        ax.plot(cum_bench.index, cum_bench.values, "k--", linewidth=1, alpha=0.7, label="Benchmark")
    
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_drawdown(
    pnl: pd.Series,
    title: str = "Drawdown",
    figsize: Tuple[int, int] = (12, 4),
    ax=None,
):
    """
    Plot drawdown over time.
    
    Parameters
    ----------
    pnl : pd.Series
        Daily P&L series
    title : str
        Plot title
    figsize : tuple
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    cum_pnl = pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = running_max - cum_pnl
    
    ax.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, "r-", linewidth=0.5)
    
    max_dd = drawdown.max()
    max_dd_idx = drawdown.idxmax()
    ax.axhline(max_dd, color="darkred", linestyle="--", linewidth=1, label=f"Max DD = {max_dd:.4f}")
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_signals(
    spread: pd.Series,
    signals: pd.Series,
    thresholds: Optional[Tuple[float, float, float]] = None,
    title: str = "Trading Signals",
    figsize: Tuple[int, int] = (12, 6),
    ax=None,
):
    """
    Plot spread with trading signals overlaid.
    
    Parameters
    ----------
    spread : pd.Series
        Spread series
    signals : pd.Series
        Signal series (+1, -1, 0)
    thresholds : tuple, optional
        (U, L, C) values
    title : str
        Plot title
    figsize : tuple
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spread
    ax.plot(spread.index, spread.values, "b-", linewidth=0.8, alpha=0.7, label="Spread")
    
    # Highlight long positions
    long_mask = signals == 1
    if long_mask.any():
        ax.fill_between(
            spread.index,
            spread.min(),
            spread.max(),
            where=long_mask,
            alpha=0.2,
            color="green",
            label="Long",
        )
    
    # Highlight short positions
    short_mask = signals == -1
    if short_mask.any():
        ax.fill_between(
            spread.index,
            spread.min(),
            spread.max(),
            where=short_mask,
            alpha=0.2,
            color="red",
            label="Short",
        )
    
    if thresholds is not None:
        U, L, C = thresholds
        ax.axhline(U, color="r", linestyle="--", linewidth=1)
        ax.axhline(L, color="g", linestyle="--", linewidth=1)
        ax.axhline(C, color="gray", linestyle="-", linewidth=1)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_backtest_summary(
    spread: pd.Series,
    filtered: pd.Series,
    signals: pd.Series,
    pnl: pd.Series,
    thresholds: Tuple[float, float, float],
    title: str = "Backtest Summary",
    figsize: Tuple[int, int] = (14, 12),
):
    """
    Create a multi-panel summary plot.
    
    Parameters
    ----------
    spread : pd.Series
        Observed spread
    filtered : pd.Series
        Filtered spread
    signals : pd.Series
        Trading signals
    pnl : pd.Series
        Daily P&L
    thresholds : tuple
        (U, L, C) values
    title : str
        Overall title
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    plt = _get_plt()
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # Panel 1: Filtered vs Observed
    plot_filtered_vs_observed(spread, filtered, thresholds, "Filtered vs Observed", ax=axes[0])
    
    # Panel 2: Signals
    plot_signals(filtered, signals, thresholds, "Trading Signals", ax=axes[1])
    
    # Panel 3: Cumulative P&L
    plot_cumulative_pnl(pnl, title="Cumulative P&L", ax=axes[2])
    
    # Panel 4: Drawdown
    plot_drawdown(pnl, title="Drawdown", ax=axes[3])
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    return fig
