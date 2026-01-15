"""
Table formatting for backtest results.

Creates publication-quality tables following Zhang (2021) format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def format_backtest_table(
    results: Dict[str, dict],
    metrics: Optional[List[str]] = None,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Format backtest results as a comparison table.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model/strategy names to result dictionaries
    metrics : list, optional
        Metrics to include (default: common metrics)
    decimals : int
        Decimal places for formatting
        
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    if metrics is None:
        metrics = [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "n_trades",
        ]
    
    rows = []
    for name, res in results.items():
        row = {"Model/Strategy": name}
        for metric in metrics:
            if metric in res:
                row[metric] = res[metric]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index("Model/Strategy")
    
    return df.round(decimals)


def format_model_comparison(
    model_results: List[dict],
    sort_by: str = "AIC",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Format model comparison table (like Table 3 in Zhang 2021).
    
    Parameters
    ----------
    model_results : list
        List of dicts with model info and fit statistics
    sort_by : str
        Column to sort by
    ascending : bool
        Sort order
        
    Returns
    -------
    pd.DataFrame
        Model comparison table
    """
    df = pd.DataFrame(model_results)
    
    # Standard columns
    cols = ["model", "log_likelihood", "n_params", "AIC", "BIC"]
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    # Rename for display
    rename_map = {
        "model": "Model",
        "log_likelihood": "Log-Lik",
        "n_params": "K",
        "AIC": "AIC",
        "BIC": "BIC",
    }
    df = df.rename(columns=rename_map)
    
    return df


def format_parameter_table(
    params: dict,
    standard_errors: Optional[dict] = None,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Format estimated parameters table.
    
    Parameters
    ----------
    params : dict
        Parameter estimates
    standard_errors : dict, optional
        Standard errors for parameters
    decimals : int
        Decimal places
        
    Returns
    -------
    pd.DataFrame
        Parameter table
    """
    rows = []
    
    for param, value in params.items():
        row = {
            "Parameter": param,
            "Estimate": value,
        }
        if standard_errors and param in standard_errors:
            row["Std. Error"] = standard_errors[param]
            row["t-stat"] = value / standard_errors[param] if standard_errors[param] != 0 else np.nan
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index("Parameter")
    
    return df.round(decimals)


def format_summary_stats(
    stats: dict,
    title: str = "Performance Summary",
) -> str:
    """
    Format summary statistics as a string table.
    
    Parameters
    ----------
    stats : dict
        Summary statistics
    title : str
        Table title
        
    Returns
    -------
    str
        Formatted string
    """
    lines = [
        title,
        "=" * len(title),
        "",
    ]
    
    # Format each statistic
    format_map = {
        "total_return": ("Total Return", "{:.4f}"),
        "annualized_return": ("Annual Return", "{:.4f}"),
        "annualized_volatility": ("Annual Vol", "{:.4f}"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.2f}"),
        "max_drawdown": ("Max Drawdown", "{:.4f}"),
        "calmar_ratio": ("Calmar Ratio", "{:.2f}"),
        "n_trades": ("Trades", "{:d}"),
        "total_costs": ("Total Costs", "{:.4f}"),
    }
    
    max_label_len = max(len(label) for label, _ in format_map.values())
    
    for key, (label, fmt) in format_map.items():
        if key in stats:
            val = stats[key]
            if isinstance(val, (int, np.integer)):
                val_str = fmt.format(int(val))
            else:
                val_str = fmt.format(float(val))
            lines.append(f"{label:<{max_label_len}} : {val_str}")
    
    return "\n".join(lines)


def create_table1_zhang(
    backtest_results: Dict[str, dict],
) -> pd.DataFrame:
    """
    Recreate Table 1 from Zhang (2021).
    
    Shows performance of different models across strategies.
    
    Parameters
    ----------
    backtest_results : dict
        Nested dict: {model: {strategy: results}}
        
    Returns
    -------
    pd.DataFrame
        Table 1 format
    """
    rows = []
    
    for model, strategies in backtest_results.items():
        for strategy, results in strategies.items():
            row = {
                "Model": model,
                "Strategy": strategy,
                "Return (%)": results.get("total_return", np.nan) * 100,
                "Sharpe": results.get("sharpe_ratio", np.nan),
                "MaxDD (%)": results.get("max_drawdown", np.nan) * 100,
                "Trades": results.get("n_trades", np.nan),
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Multi-index for nice display
    df = df.set_index(["Model", "Strategy"])
    
    return df.round(2)
