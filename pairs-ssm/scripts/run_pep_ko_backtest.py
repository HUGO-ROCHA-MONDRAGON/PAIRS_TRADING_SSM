#!/usr/bin/env python
"""
Example script: Run pairs trading backtest on PEP-KO.

Usage:
    python scripts/run_pep_ko_backtest.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from pairs_ssm import (
    load_pair,
    compute_spread,
    fit_model,
    compare_models,
    generate_signals,
    backtest_signals,
    BacktestEngine,
)
from pairs_ssm.reporting import format_backtest_table, format_summary_stats


def main():
    """Run PEP-KO pairs trading backtest."""
    
    print("=" * 60)
    print("Pairs Trading with State-Space Models")
    print("Replicating Zhang (2021)")
    print("=" * 60)
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / "dataGQ.xlsx"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    print(f"\n1. Loading data from {data_path}...")
    pair_data = load_pair(str(data_path), "PEP US Equity", "KO US Equity")
    print(f"   Loaded {len(pair_data.PA)} observations")
    print(f"   Date range: {pair_data.PA.index[0]} to {pair_data.PA.index[-1]}")
    
    # Convert to log prices
    log_p1 = np.log(pair_data.PA)
    log_p2 = np.log(pair_data.PB)
    
    # Compute spread
    print("\n2. Computing spread...")
    spread_data = compute_spread(log_p1, log_p2)
    print(f"   Estimated gamma: {spread_data.gamma:.6f}")
    print(f"   Spread mean: {spread_data.spread.mean():.6f}")
    print(f"   Spread std: {spread_data.spread.std():.6f}")
    
    # Fit models
    print("\n3. Fitting state-space models...")
    
    y = spread_data.spread
    
    # Compare models
    comparison = compare_models(y, models=["model_I", "model_II"])
    print("\nModel Comparison:")
    print(comparison.to_string())
    
    # Use best model (Model I or II based on AIC)
    best_model = comparison.loc[comparison["AIC"].idxmin(), "Model"]
    print(f"\nBest model by AIC: {best_model}")
    
    # Fit best model
    result = fit_model(y, model_type=best_model)
    params = result.params
    
    print(f"\nEstimated parameters ({best_model}):")
    print(f"   theta0  = {params.theta0:.6f}")
    print(f"   theta1  = {params.theta1:.6f}")
    print(f"   q_base  = {params.q_base:.6e}")
    print(f"   r       = {params.r:.6e}")
    
    # Run backtests
    print("\n4. Running backtests...")
    
    results_all = {}
    
    for strategy in ["A", "B", "C"]:
        for n_std in [1.0, 1.5, 2.0]:
            engine = BacktestEngine(log_p1, log_p2)
            engine.fit(best_model)
            
            bt_result = engine.backtest(
                strategy=strategy,
                n_std=n_std,
                use_filtered=True,
                cost_bp=20.0,
            )
            
            stats = {
                "strategy": strategy,
                "n_std": n_std,
                "total_return": bt_result.total_return(),
                "sharpe": bt_result.sharpe_ratio(),
                "max_drawdown": bt_result.max_drawdown(),
                "n_trades": bt_result.n_trades,
            }
            
            key = f"{strategy}_{n_std}"
            results_all[key] = stats
    
    # Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    df_results = pd.DataFrame(results_all).T
    df_results = df_results.round(4)
    print(df_results.to_string())
    
    # Best configuration
    best_key = max(results_all, key=lambda k: results_all[k]["sharpe"])
    best_result = results_all[best_key]
    
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Strategy: {best_result['strategy']}")
    print(f"Threshold: {best_result['n_std']} std devs")
    print(f"Sharpe Ratio: {best_result['sharpe']:.4f}")
    print(f"Total Return: {best_result['total_return']:.4f}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.4f}")
    print(f"Number of Trades: {best_result['n_trades']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
