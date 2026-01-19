#!/usr/bin/env python
"""
Replication of Table 1: Simulation Study from Zhang (2021).

This script performs Monte Carlo simulations to evaluate the performance
of strategies A, B, and C under different market conditions.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pairs_ssm.optimization.table1 import replicate_table1

def run_simulation_table():
    print("=" * 80)
    print("REPLICATION TABLE 1: SIMULATION STUDY")
    print("=" * 80)
    
    # We use a smaller N than the paper (10000) for speed in this demonstration
    # N=1000 gives reasonable precision in seconds instead of minutes
    N_SIMS = 1000
    
    print(f"Running simulation with N={N_SIMS} paths per model...")
    results = replicate_table1(N=N_SIMS, T=1000, cost_bp=20.0, verbose=True)
    
    print("\n" + "=" * 80)
    print(f"{'Model':<10} {'Strat':<6} {'SR':<8} {'U*':<6} {'L*':<6}")
    print("-" * 80)
    
    # Sort by model then strategy
    sorted_keys = sorted(results.keys())
    
    for key in sorted_keys:
        res = results[key]
        print(f"{res.model:<10} {res.strategy:<6} {res.SR:.4f}   {res.U_star_sr:.2f}   {res.L_star_sr:.2f}")
        
    print("=" * 80)

if __name__ == "__main__":
    run_simulation_table()
