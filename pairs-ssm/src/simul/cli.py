"""
Command Line Interface for pairs-ssm.

Usage:
    pairs-ssm fit --config CONFIG_PATH
    pairs-ssm optimize --config CONFIG_PATH
    pairs-ssm backtest --config CONFIG_PATH
    pairs-ssm report --config CONFIG_PATH
    pairs-ssm run --config CONFIG_PATH  # Full pipeline
"""

import argparse
import sys
from pathlib import Path

from pairs_ssm.utils.io import load_config, save_results
from pairs_ssm.utils.logging import setup_logging, get_logger


def cmd_fit(args):
    """Fit the state-space model to data."""
    from pairs_ssm.data.loaders import load_pair
    from pairs_ssm.data.transforms import compute_spread
    from pairs_ssm.filtering.mle import fit_model
    
    logger = get_logger(__name__)
    config = load_config(args.config)
    
    logger.info("Loading data...")
    pair = load_pair(
        Path(config["data"]["raw_path"]) / config["data"]["source_file"],
        config["pair"]["asset_a"],
        config["pair"]["asset_b"],
        start_date=config["pair"]["start_date"],
        end_date=config["pair"]["end_date"],
    )
    
    logger.info("Computing spread...")
    spread = compute_spread(pair, use_log=config["pair"]["use_log_prices"])
    
    logger.info(f"Fitting {config['model']['type']}...")
    result = fit_model(
        spread,
        model_type=config["model"]["type"],
        method=config["model"]["optimizer"]["method"],
        maxiter=config["model"]["optimizer"]["maxiter"],
        verbose=True,
    )
    
    # Save results
    output_dir = Path(config["reporting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(result, output_dir / "fit_result.pkl")
    
    logger.info(f"Model fit complete. Results saved to {output_dir}/fit_result.pkl")
    return result


def cmd_optimize(args):
    """Optimize trading thresholds via Monte Carlo simulation."""
    from pairs_ssm.utils.io import load_results
    from pairs_ssm.optimization.objective import optimize_thresholds
    from pairs_ssm.optimization.simulator import simulate_paths
    
    logger = get_logger(__name__)
    config = load_config(args.config)
    
    # Load fitted model
    output_dir = Path(config["reporting"]["output_dir"])
    fit_result = load_results(output_dir / "fit_result.pkl")
    
    logger.info("Simulating paths for optimization...")
    paths = simulate_paths(
        fit_result.params,
        n_paths=config["optimization"]["simulation"]["n_paths"],
        n_steps=config["optimization"]["simulation"]["n_steps"],
        seed=config["seed"],
    )
    
    logger.info(f"Optimizing thresholds for Strategy {config['trading']['strategy']}...")
    optimal = optimize_thresholds(
        paths,
        strategy=config["trading"]["strategy"],
        objective=config["optimization"]["objective"],
        grid_config=config["optimization"]["grid"],
        tc_bps=config["trading"]["tc_bps"],
    )
    
    save_results(optimal, output_dir / "optimal_thresholds.pkl")
    logger.info(f"Optimal U={optimal['U']:.4f}, L={optimal['L']:.4f}")
    logger.info(f"Expected {config['optimization']['objective']}: {optimal['score']:.4f}")
    
    return optimal


def cmd_backtest(args):
    """Run backtest with optimized parameters."""
    from pairs_ssm.utils.io import load_results
    from pairs_ssm.data.loaders import load_pair
    from pairs_ssm.data.transforms import compute_spread
    from pairs_ssm.backtest.engine import run_backtest
    
    logger = get_logger(__name__)
    config = load_config(args.config)
    
    # Load data
    pair = load_pair(
        Path(config["data"]["raw_path"]) / config["data"]["source_file"],
        config["pair"]["asset_a"],
        config["pair"]["asset_b"],
        start_date=config["pair"]["start_date"],
        end_date=config["pair"]["end_date"],
    )
    spread = compute_spread(pair, use_log=config["pair"]["use_log_prices"])
    
    # Load fitted model and optimal thresholds
    output_dir = Path(config["reporting"]["output_dir"])
    fit_result = load_results(output_dir / "fit_result.pkl")
    optimal = load_results(output_dir / "optimal_thresholds.pkl")
    
    logger.info("Running backtest...")
    performance = run_backtest(
        pair=pair,
        spread=spread,
        fit_result=fit_result,
        thresholds=optimal,
        strategy=config["trading"]["strategy"],
        tc_bps=config["trading"]["tc_bps"],
    )
    
    save_results(performance, output_dir / "backtest_results.pkl")
    
    logger.info("Backtest complete!")
    logger.info(f"  Cumulative Return: {performance['cumulative_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
    
    return performance


def cmd_report(args):
    """Generate performance report."""
    from pairs_ssm.utils.io import load_results
    from pairs_ssm.reporting.tables import generate_summary_table
    from pairs_ssm.reporting.plots import plot_backtest_results
    from pairs_ssm.reporting.export import export_results
    
    logger = get_logger(__name__)
    config = load_config(args.config)
    
    output_dir = Path(config["reporting"]["output_dir"])
    
    # Load all results
    fit_result = load_results(output_dir / "fit_result.pkl")
    optimal = load_results(output_dir / "optimal_thresholds.pkl")
    performance = load_results(output_dir / "backtest_results.pkl")
    
    logger.info("Generating report...")
    
    # Generate summary table
    summary = generate_summary_table(fit_result, optimal, performance)
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(summary.to_string())
    
    # Generate plots
    plot_backtest_results(performance, output_dir / "figures")
    
    # Export
    export_results(summary, performance, output_dir, formats=config["reporting"]["formats"])
    
    logger.info(f"Report saved to {output_dir}")


def cmd_run(args):
    """Run full pipeline: fit -> optimize -> backtest -> report."""
    logger = get_logger(__name__)
    logger.info("Running full pipeline...")
    
    cmd_fit(args)
    cmd_optimize(args)
    cmd_backtest(args)
    cmd_report(args)
    
    logger.info("Pipeline complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pairs-ssm",
        description="Pairs Trading with State-Space Models",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit state-space model")
    fit_parser.add_argument("--config", "-c", required=True, help="Path to config file")
    fit_parser.set_defaults(func=cmd_fit)
    
    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize thresholds")
    opt_parser.add_argument("--config", "-c", required=True, help="Path to config file")
    opt_parser.set_defaults(func=cmd_optimize)
    
    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--config", "-c", required=True, help="Path to config file")
    bt_parser.set_defaults(func=cmd_backtest)
    
    # Report command
    rep_parser = subparsers.add_parser("report", help="Generate report")
    rep_parser.add_argument("--config", "-c", required=True, help="Path to config file")
    rep_parser.set_defaults(func=cmd_report)
    
    # Run command (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--config", "-c", required=True, help="Path to config file")
    run_parser.set_defaults(func=cmd_run)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    config = load_config(args.config)
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    
    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
