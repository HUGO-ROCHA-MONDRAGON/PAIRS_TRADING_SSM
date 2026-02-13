# Pairs Trading with State-Space Models

Replication and extension of **Zhang (2021)** — *Pairs Trading with General State Space Models*.

## Project Structure

```
├── configs/              # Model & backtest configuration (YAML)
├── data/                 # Raw market data (Excel)
│   └── outputs/          # Generated results (CSV, PNG) — not tracked
├── notebooks/            # Jupyter notebooks (analysis & results)
│   ├── 01_simulation             — CIR spread simulation
│   ├── 02_table1_replication     — Table 1 replication (Zhang 2021)
│   ├── 03_strategy_extensions    — Extended trading strategies
│   ├── 04_table1_extensions      — Table 1 with extensions
│   ├── 05_tables_and_appendix    — Tables 2, 3 & Appendix
│   ├── 06_extended_study_period  — Study period extension (→ 2025)
│   ├── 07_kim_filter_ms_ssm      — Kim filter & Markov-switching SSM
│   ├── 08_cointegration_persistence — Cointegration persistence tests
│   └── 09_portfolio_construction — Multi-asset portfolio construction
├── src/simul/            # Python package
│   ├── trading/          — Strategy signal generators (A, B, C, D, E)
│   ├── optimization/     — Threshold optimization (Table 1 & S1)
│   └── utils/            — Simulation utilities (CIR / OU spreads)
└── pyproject.toml
```

## Installation

```bash
pip install -e .
```

## Key Features

- **Kalman filter** estimation of state-space spread models
- **5 trading strategies** (A–E) with configurable thresholds
- **Numba-accelerated** Monte Carlo simulations
- **Kim filter** extension with Markov-switching regimes
- **Multi-asset** portfolio construction across crypto, commodities & equities

## Reference

Zhang, H. (2021). *Pairs Trading with General State Space Models*. Quantitative Finance.
