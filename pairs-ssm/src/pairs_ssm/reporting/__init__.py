"""Reporting module for tables, plots, and exports."""

from .tables import (
    format_backtest_table,
    format_model_comparison,
    format_parameter_table,
    format_summary_stats,
)
from .plots import (
    plot_spread,
    plot_filtered_vs_observed,
    plot_cumulative_pnl,
    plot_drawdown,
    plot_signals,
)
from .export import (
    export_results_csv,
    export_results_latex,
    export_results_excel,
)

__all__ = [
    # Tables
    "format_backtest_table",
    "format_model_comparison",
    "format_parameter_table",
    "format_summary_stats",
    # Plots
    "plot_spread",
    "plot_filtered_vs_observed",
    "plot_cumulative_pnl",
    "plot_drawdown",
    "plot_signals",
    # Export
    "export_results_csv",
    "export_results_latex",
    "export_results_excel",
]
