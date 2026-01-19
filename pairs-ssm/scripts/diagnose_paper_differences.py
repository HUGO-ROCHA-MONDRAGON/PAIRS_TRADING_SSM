#!/usr/bin/env python
"""
Diagnostic détaillé : Pourquoi les résultats diffèrent du papier Zhang (2021)
=============================================================================

Ce script identifie les différences potentielles entre notre implémentation
et le papier original.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pairs_ssm import load_pair, compute_spread, fit_model
from pairs_ssm.backtest import BacktestEngine


def restrict_dates(df, start, end):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.loc[pd.to_datetime(start): pd.to_datetime(end)]


def main():
    print("=" * 80)
    print("DIAGNOSTIC : DIFFÉRENCES AVEC LE PAPIER ZHANG (2021)")
    print("=" * 80)
    
    # Charger les données
    data_path = Path(__file__).parent.parent / "data" / "dataGQ.xlsx"
    pair_data = load_pair(str(data_path), "PEP US Equity", "KO US Equity")
    
    df = pd.DataFrame({"PA": pair_data.PA, "PB": pair_data.PB})
    df = restrict_dates(df, "2012-01-03", "2019-06-28")
    
    print(f"\n1. VÉRIFICATION DES DONNÉES")
    print("-" * 80)
    print(f"   Période: {df.index[0]} à {df.index[-1]}")
    print(f"   Observations: {len(df)}")
    print(f"   PEP (PA) - Premier prix: ${df['PA'].iloc[0]:.2f}, Dernier: ${df['PA'].iloc[-1]:.2f}")
    print(f"   KO (PB) - Premier prix: ${df['PB'].iloc[0]:.2f}, Dernier: ${df['PB'].iloc[-1]:.2f}")
    
    # =========================================================================
    # TEST 1: LOG PRICES VS PRICES
    # =========================================================================
    print(f"\n2. TEST : LOG PRICES vs PRICES NORMAUX")
    print("-" * 80)
    
    # Avec log prices (notre implémentation)
    log_p1 = np.log(df["PA"])
    log_p2 = np.log(df["PB"])
    spread_log = compute_spread(log_p1, log_p2)
    
    # Avec prices normaux (possible dans le papier?)
    spread_normal = compute_spread(df["PA"], df["PB"])
    
    print(f"   LOG PRICES:")
    print(f"      Gamma: {spread_log.gamma:.6f}")
    print(f"      Spread mean: {spread_log.spread.mean():.6f}")
    print(f"      Spread std: {spread_log.spread.std():.6f}")
    
    print(f"\n   PRICES NORMAUX:")
    print(f"      Gamma: {spread_normal.gamma:.6f}")
    print(f"      Spread mean: {spread_normal.spread.mean():.2f}")
    print(f"      Spread std: {spread_normal.spread.std():.2f}")
    
    # =========================================================================
    # TEST 2: VÉRIFIER LES SIGNAUX DE TRADING
    # =========================================================================
    print(f"\n3. TEST : VÉRIFICATION DES SIGNAUX DE TRADING")
    print("-" * 80)
    
    # Utiliser log prices
    y = spread_log.spread
    
    # Fit model
    result = fit_model(y, model_type="model_I")  # Tester Model I d'abord
    
    print(f"   Model I - Paramètres:")
    print(f"      theta0: {result.params.theta0:.6f}")
    print(f"      theta1: {result.params.theta1:.6f}")
    print(f"      q: {result.params.q_base:.6e}")
    print(f"      r: {result.params.r:.6e}")
    
    # Backtest avec différents seuils pour voir l'effet
    print(f"\n   Test des seuils (Stratégie A):")
    engine = BacktestEngine(log_p1, log_p2)
    engine.fit("model_I")
    
    # Calculer les seuils théoriques
    mu_lr = result.params.long_run_mean
    sigma_lr = result.params.long_run_std
    
    for n_std in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        bt = engine.backtest(strategy="A", n_std=n_std, use_filtered=True, cost_bp=20.0)
        
        # Seuils théoriques
        U = mu_lr + n_std * sigma_lr
        L = mu_lr - n_std * sigma_lr
        
        print(f"      {n_std:.1f}σ -> U={U:.6f}, L={L:.6f} | Trades: {bt.n_trades:3d} | Sharpe: {bt.sharpe_ratio():.3f}")
    
    # =========================================================================
    # TEST 3: COMPARER FILTERED VS STATIC THRESHOLDS
    # =========================================================================
    print(f"\n4. TEST : FILTERED vs STATIC THRESHOLDS")
    print("-" * 80)
    
    engine = BacktestEngine(log_p1, log_p2)
    engine.fit("model_I")
    
    # Avec filtered (time-varying)
    bt_filtered = engine.backtest(strategy="A", n_std=1.5, use_filtered=True, cost_bp=20.0)
    
    # Avec static (constant)
    bt_static = engine.backtest(strategy="A", n_std=1.5, use_filtered=False, cost_bp=20.0)
    
    print(f"   FILTERED thresholds (time-varying):")
    print(f"      Sharpe: {bt_filtered.sharpe_ratio():.4f}")
    print(f"      Return: {bt_filtered.total_return():.2%}")
    print(f"      Trades: {bt_filtered.n_trades}")
    
    print(f"\n   STATIC thresholds (constant):")
    print(f"      Sharpe: {bt_static.sharpe_ratio():.4f}")
    print(f"      Return: {bt_static.total_return():.2%}")
    print(f"      Trades: {bt_static.n_trades}")
    
    # =========================================================================
    # TEST 4: VÉRIFIER LA STRATÉGIE C
    # =========================================================================
    print(f"\n5. TEST : ANALYSE DE LA STRATÉGIE C (problématique)")
    print("-" * 80)
    
    for strat in ["A", "B", "C"]:
        bt = engine.backtest(strategy=strat, n_std=1.5, use_filtered=True, cost_bp=20.0)
        print(f"   Stratégie {strat}:")
        print(f"      Sharpe: {bt.sharpe_ratio():7.3f}")
        print(f"      Return: {bt.total_return():7.2%}")
        print(f"      Trades: {bt.n_trades:3d}")
        print(f"      Max DD: {bt.max_drawdown():7.2%}")
    
    # =========================================================================
    # TEST 5: VISUALISATION DES SIGNAUX
    # =========================================================================
    print(f"\n6. VISUALISATION DES SIGNAUX")
    print("-" * 80)
    
    # Créer un graphique pour comprendre
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Spread et seuils
    ax = axes[0]
    ax.plot(y.index, y.values, label='Spread', linewidth=0.8, alpha=0.7)
    
    # Seuils statiques
    mu = y.mean()
    sigma = y.std()
    ax.axhline(mu, color='green', linestyle='--', label='Mean', alpha=0.5)
    ax.axhline(mu + 1.5*sigma, color='red', linestyle='--', label='U (1.5σ)', alpha=0.5)
    ax.axhline(mu - 1.5*sigma, color='red', linestyle='--', label='L (1.5σ)', alpha=0.5)
    
    ax.set_ylabel('Spread')
    ax.set_title('Spread avec seuils STATIQUES')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Spread filtré
    ax = axes[1]
    if engine.filter_result is not None:
        x_filt = pd.Series(engine.filter_result.x_filt, index=y.index)
        P_filt = pd.Series(engine.filter_result.P_filt, index=y.index)
        
        ax.plot(x_filt.index, x_filt.values, 
                label='Filtered Mean', linewidth=1, color='blue')
        ax.fill_between(x_filt.index,
                         x_filt - 1.5 * np.sqrt(P_filt),
                         x_filt + 1.5 * np.sqrt(P_filt),
                         alpha=0.2, label='±1.5σ filtered')
    ax.plot(y.index, y.values, label='Spread', linewidth=0.5, alpha=0.5, color='black')
    
    ax.set_ylabel('Spread')
    ax.set_title('Spread avec seuils FILTRÉS (time-varying)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Signaux de trading (Stratégie A)
    ax = axes[2]
    bt_static = engine.backtest(strategy="A", n_std=1.5, use_filtered=False, cost_bp=20.0)
    
    # Calculer la courbe PnL cumulée
    pnl_cum = bt_static.pnl.cumsum()
    ax.plot(pnl_cum.index, pnl_cum.values, 
            label='PnL Curve', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_ylabel('Cumulative PnL')
    ax.set_title('Performance de la Stratégie A (seuils statiques)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / "diagnostic_signals.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Graphique sauvegardé: {output_path}")
    
    # =========================================================================
    # CONCLUSIONS
    # =========================================================================
    print(f"\n7. CONCLUSIONS ET HYPOTHÈSES")
    print("=" * 80)
    print("""
DIFFÉRENCES POSSIBLES AVEC LE PAPIER:

1. LOG PRICES vs PRICES:
   - Notre implémentation utilise log(prices)
   - Le papier pourrait utiliser prices directement
   - Impact: Échelle du spread et des seuils complètement différente

2. SEUILS FILTRÉS vs STATIQUES:
   - Seuils time-varying (filtrés) vs constants (statiques)
   - Le papier n'est pas clair sur ce point
   - Impact: Nombre de trades et performance différents

3. STRATÉGIE C:
   - Logique "re-entry" potentiellement mal implémentée
   - Stop-loss pourrait être trop agressif
   - À vérifier dans le code source

4. ESTIMATION DES PARAMÈTRES:
   - MLE peut converger vers différents optima locaux
   - Sensible aux valeurs initiales
   - Impact sur filtered mean et std

5. COÛTS DE TRANSACTION:
   - Implémentation exacte des 20 bps
   - Round-trip vs one-way
   - Impact sur le nombre de trades rentables

RECOMMANDATIONS:
1. Tester avec prices normaux (sans log)
2. Tester avec seuils statiques simples: mean ± k*std
3. Vérifier l'implémentation de la Stratégie C
4. Comparer visuellement les signaux avec le papier
5. Vérifier les valeurs initiales du MLE
    """)
    
    print("\nDiagnostic terminé!")


if __name__ == "__main__":
    main()
