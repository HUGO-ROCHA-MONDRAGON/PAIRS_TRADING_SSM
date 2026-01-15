"""
Export functions for results.

Supports CSV, LaTeX, and Excel formats.
"""

import pandas as pd
from typing import Union, Dict, Optional
from pathlib import Path


def export_results_csv(
    results: Union[pd.DataFrame, Dict[str, dict]],
    filepath: Union[str, Path],
    index: bool = True,
) -> None:
    """
    Export results to CSV.
    
    Parameters
    ----------
    results : DataFrame or dict
        Results to export
    filepath : str or Path
        Output file path
    index : bool
        Include index in output
    """
    if isinstance(results, dict):
        results = pd.DataFrame(results).T
    
    results.to_csv(filepath, index=index)


def export_results_latex(
    results: pd.DataFrame,
    filepath: Optional[Union[str, Path]] = None,
    caption: str = "Results",
    label: str = "tab:results",
    float_format: str = "%.4f",
) -> str:
    """
    Export results to LaTeX table format.
    
    Parameters
    ----------
    results : DataFrame
        Results table
    filepath : str or Path, optional
        Output file path (returns string if None)
    caption : str
        Table caption
    label : str
        LaTeX label
    float_format : str
        Float formatting string
        
    Returns
    -------
    str
        LaTeX table string
    """
    latex_str = results.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
    )
    
    if filepath is not None:
        with open(filepath, "w") as f:
            f.write(latex_str)
    
    return latex_str


def export_results_excel(
    results: Dict[str, pd.DataFrame],
    filepath: Union[str, Path],
) -> None:
    """
    Export multiple result tables to Excel workbook.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping sheet names to DataFrames
    filepath : str or Path
        Output file path (.xlsx)
    """
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name[:31])  # Excel sheet name limit


def export_backtest_report(
    backtest_result,
    model_params: dict,
    filepath: Union[str, Path],
    format: str = "csv",
) -> None:
    """
    Export comprehensive backtest report.
    
    Parameters
    ----------
    backtest_result : BacktestResult
        Backtest results
    model_params : dict
        Fitted model parameters
    filepath : str or Path
        Output path
    format : str
        Output format ('csv', 'latex', 'excel')
    """
    from ..trading import summary_statistics
    
    stats = summary_statistics(backtest_result)
    
    # Combine with params
    full_report = {
        "Performance Metrics": stats,
        "Model Parameters": model_params,
    }
    
    filepath = Path(filepath)
    
    if format == "csv":
        # Export summary stats
        pd.DataFrame([stats]).to_csv(filepath, index=False)
    elif format == "latex":
        df = pd.DataFrame([stats])
        export_results_latex(df, filepath)
    elif format == "excel":
        export_results_excel({
            "Summary": pd.DataFrame([stats]),
            "Parameters": pd.DataFrame([model_params]),
            "Daily_PnL": backtest_result.pnl.to_frame("pnl"),
        }, filepath)


def generate_paper_table1(
    results_dict: Dict[str, Dict[str, dict]],
    filepath: Optional[Union[str, Path]] = None,
    format: str = "latex",
) -> Union[str, pd.DataFrame]:
    """
    Generate Table 1 from Zhang (2021).
    
    Shows strategy performance across different models.
    
    Parameters
    ----------
    results_dict : dict
        Nested dict: {model: {strategy: results}}
    filepath : str or Path, optional
        Output path
    format : str
        'latex' or 'dataframe'
        
    Returns
    -------
    str or DataFrame
        Table output
    """
    rows = []
    
    for model, strategies in results_dict.items():
        for strategy, results in strategies.items():
            rows.append({
                "Model": model,
                "Strategy": strategy,
                "Return (%)": results.get("total_return", 0) * 100,
                "Vol (%)": results.get("annualized_volatility", 0) * 100,
                "Sharpe": results.get("sharpe_ratio", 0),
                "MaxDD (%)": results.get("max_drawdown", 0) * 100,
                "Calmar": results.get("calmar_ratio", 0),
                "Trades": results.get("n_trades", 0),
            })
    
    df = pd.DataFrame(rows)
    df = df.round(2)
    
    if format == "dataframe":
        return df
    
    latex_str = export_results_latex(
        df,
        filepath=filepath,
        caption="Backtest Results by Model and Strategy",
        label="tab:results",
    )
    
    return latex_str
