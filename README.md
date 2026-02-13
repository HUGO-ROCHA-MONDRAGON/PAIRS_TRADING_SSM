# Pairs Trading with State-Space Models

Replication and extension of **Zhang (2021)** — *Pairs Trading with General State Space Models*, Quantitative Finance, 21(9):1567–1587.

> **M2 Quantitative Asset Management Project** — Çağla Naz Amiklioğlu, Hugo Rocha Mondragon, Sacha Guerin, Marius Calaque

## Abstract

This project replicates and empirically evaluates Zhang (2021), which proposes a general pairs trading framework based on state-space models with state-dependent volatility. We reproduce the optimal threshold selection via simulation, estimate the models on real market data, and analyze the performance of different trading strategies.

The simulation results confirm the superiority of **Strategy C** in terms of Sharpe ratio, especially when combined with the **heteroscedastic model (Model II)**. On real data, this configuration outperforms classical approaches on the PEP–KO and EWT–EWH pairs, primarily by improving risk control rather than raw returns. These results generalize to a broader universe of U.S. banks listed on the NYSE, both in-sample and out-of-sample, but do not remain robust over an extended time period covering more recent market conditions.

The project also proposes several extensions:

- **Flexible exit rules (Strategies D & E)** — optimizing exit timing can enhance performance under heteroscedasticity by adjusting the trade-off between trading frequency and mean-reversion capture
- **Rolling cointegration tests** — reveal limited out-of-sample persistence of cointegration, motivating dynamic pair selection
- **Unsupervised pair selection & portfolio construction** — periodic rebalancing via clustering improves portfolio stability; the heteroscedastic model retains an edge in risk-adjusted performance
- **Markov-switching regime model (Kim filter)** — provides a useful stress indicator for risk management, improving strategies A/B by avoiding entries during extreme volatility, while revealing structural incompatibilities with sequential rules like Strategy C

Overall, this work confirms the relevance of Zhang's state-space framework and highlights two practical insights: (1) pairs trading performance is largely a matter of **conditional risk management** rather than directional prediction, and (2) robustness depends on the coherence between the assumed statistical dynamics (cointegration, mean-reversion, regimes) and the trading rule employed.

## Project Structure

```
├── configs/              # Model & backtest configuration (YAML)
├── data/                 # Raw market data (Excel)
│   └── outputs/          # Generated results (CSV, PNG) — not tracked
├── notebooks/            # Jupyter notebooks (analysis & results)
│   ├── 01_simulation             — CIR spread simulation
│   ├── 02_table1_replication     — Table 1 replication (Zhang 2021)
│   ├── 03_strategy_extensions    — Extended trading strategies (D, E)
│   ├── 04_table1_extensions      — Table S1 with extensions
│   ├── 05_tables_and_appendix    — Tables 2, 3 & Appendix (real data)
│   ├── 06_extended_study_period  — Study period extension (→ 2025)
│   ├── 07_kim_filter_ms_ssm      — Kim filter & Markov-switching SSM
│   ├── 08_cointegration_persistence — Rolling cointegration tests
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

## Reference

Zhang, G. (2021). *Pairs Trading with General State Space Models*. Quantitative Finance, 21(9):1567–1587. [DOI](https://doi.org/10.1080/14697688.2021.1890806)
