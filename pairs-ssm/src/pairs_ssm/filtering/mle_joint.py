"""
Joint Maximum Likelihood Estimation - Version du papier Zhang (2021).

Le papier traite le spread comme une VARIABLE LATENTE avec bruit d'observation:
    Observation: y_t = x_t + ε_t,  où ε_t ~ N(0, σε²)
    État:        x_{t+1} = θ₀ + θ₁ x_t + σ(x_t) η_t

Cette implémentation:
1. Estime σε² (variance d'observation) comme paramètre LIBRE (non fixé à ~0)
2. Estime γ (hedge ratio) CONJOINTEMENT avec les autres paramètres
3. Traite le spread comme non observé directement

DIFFÉRENCE CRITIQUE avec mle.py:
- mle.py: r = σε² est forcé vers ~0 (spread observé exactement)
- Ici:    r = σε² est un paramètre libre significatif (~0.01)
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass

from pairs_ssm.models.params import ModelParams, FilterResult
from pairs_ssm.filtering.kalman_linear import kalman_filter
from pairs_ssm.filtering.kalman_extended import extended_kalman_filter
from pairs_ssm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class JointFilterResult:
    """Résultat de l'estimation jointe γ + paramètres state-space."""
    
    gamma: float              # Hedge ratio estimé
    params: ModelParams       # Paramètres state-space
    x_filt: pd.Series        # État filtré
    x_pred: pd.Series        # État prédit
    P_filt: pd.Series        # Variance filtrée
    P_pred: pd.Series        # Variance prédite
    loglik: float            # Log-vraisemblance
    aic: float               # AIC
    bic: float               # BIC
    model_type: str          # Type de modèle
    spread: pd.Series        # Spread calculé avec γ estimé


def fit_joint(
    log_PA: pd.Series,
    log_PB: pd.Series,
    model_type: Literal["model_I", "model_II"] = "model_I",
    method: str = "L-BFGS-B",
    maxiter: int = 3000,
    verbose: bool = True,
    seed: Optional[int] = 42,
    r_init: float = 0.01,
    fix_gamma: bool = False,
) -> JointFilterResult:
    """
    Estimation JOINTE de γ et des paramètres state-space.
    
    C'est la méthode du papier Zhang (2021):
    - γ n'est PAS estimé séparément par OLS
    - σε² (variance d'observation) n'est PAS forcée à ~0
    
    Parameters
    ----------
    log_PA : pd.Series
        Log prix de l'actif A
    log_PB : pd.Series  
        Log prix de l'actif B
    model_type : str
        "model_I" (homoscédastique) ou "model_II" (hétéroscédastique)
    method : str
        Méthode d'optimisation ("L-BFGS-B", "Nelder-Mead", "differential_evolution")
    maxiter : int
        Nombre max d'itérations
    verbose : bool
        Afficher les résultats
    seed : int
        Graine aléatoire
    r_init : float
        Valeur initiale pour σε² (variance d'observation)
    fix_gamma : bool
        Si True, garder γ fixé à sa valeur OLS (pour debugging)
        
    Returns
    -------
    JointFilterResult
        Résultats de l'estimation
    """
    # Préparer les données
    df = pd.DataFrame({"PA": log_PA, "PB": log_PB}).dropna()
    log_PA = df["PA"]
    log_PB = df["PB"]
    n = len(log_PA)
    
    # Estimation initiale de gamma par OLS
    X = np.column_stack([np.ones(n), log_PB.values])
    beta = np.linalg.lstsq(X, log_PA.values, rcond=None)[0]
    gamma_init = float(beta[1])
    
    # Spread initial
    spread_init = log_PA.values - gamma_init * log_PB.values
    var_spread = float(np.var(spread_init))
    mean_spread = float(np.mean(spread_init))
    
    if verbose:
        logger.info(f"Estimation jointe {model_type} + γ...")
        logger.info(f"  γ initial (OLS): {gamma_init:.4f}")
        logger.info(f"  Observations:    {n}")
    
    # Sélectionner le filtre
    if model_type.lower() == "model_i":
        filter_func = kalman_filter
        n_params = 5  # gamma, theta0, theta1, q, r
    else:
        filter_func = extended_kalman_filter
        n_params = 6  # gamma, theta0, theta1, q_base, q_het, r
    
    # AMÉLIORATION: Estimer θ1 initial par autocorrélation
    spread_centered = spread_init - mean_spread
    autocorr = np.corrcoef(spread_centered[:-1], spread_centered[1:])[0, 1]
    theta1_init = min(0.99, max(0.5, autocorr))  # Clip entre 0.5 et 0.99
    
    # Estimer q initial: variance * (1 - θ1²) approximativement
    q_init = var_spread * (1 - theta1_init**2)
    
    # θ0 initial: mean * (1 - θ1)
    theta0_init = mean_spread * (1 - theta1_init)
    
    # Paramètres initiaux
    if model_type.lower() == "model_i":
        z0 = np.array([
            gamma_init,                              # gamma
            theta0_init,                             # theta0
            np.arctanh(theta1_init),                 # theta1 (transformé)
            np.log(q_init + 1e-10),                  # log(q)
            np.log(r_init),                          # log(r) - NON FIXÉ À ~0
        ])
        
        # Bounds plus serrés autour des valeurs initiales
        gamma_lb = max(0.5, gamma_init - 0.5)
        gamma_ub = min(3.0, gamma_init + 0.5)
        
        bounds = [
            (gamma_lb, gamma_ub) if not fix_gamma else (gamma_init - 0.001, gamma_init + 0.001),
            (-0.05, 0.05),                           # theta0
            (np.arctanh(0.8), np.arctanh(0.999)),    # theta1 (0.8 to 0.999)
            (np.log(1e-6), np.log(0.01)),            # log(q)
            (np.log(0.0001), np.log(0.05)),          # log(r) - PEUT ÊTRE SIGNIFICATIF
        ]
    else:
        z0 = np.array([
            gamma_init,                              # gamma
            theta0_init,                             # theta0
            np.arctanh(theta1_init),                 # theta1
            np.log(q_init * 0.01 + 1e-10),           # log(q_base) - petit pour Model II
            np.log(0.1),                             # log(q_het)
            np.log(r_init),                          # log(r)
        ])
        
        gamma_lb = max(0.5, gamma_init - 0.5)
        gamma_ub = min(3.0, gamma_init + 0.5)
        
        bounds = [
            (gamma_lb, gamma_ub) if not fix_gamma else (gamma_init - 0.001, gamma_init + 0.001),
            (-0.05, 0.05),                           # theta0
            (np.arctanh(0.8), np.arctanh(0.999)),    # theta1
            (np.log(1e-10), np.log(0.001)),          # log(q_base)
            (np.log(0.01), np.log(1.0)),             # log(q_het)
            (np.log(0.0001), np.log(0.05)),          # log(r)
        ]
    
    # Fonction objectif
    def neg_loglik(z: np.ndarray) -> float:
        try:
            # Extraire les paramètres
            gamma = z[0]
            theta0 = z[1]
            theta1 = np.tanh(z[2])
            
            if model_type.lower() == "model_i":
                q_base = np.exp(z[3])
                q_het = 0.0
                r = np.exp(z[4])
            else:
                q_base = np.exp(z[3])
                q_het = np.exp(z[4])
                r = np.exp(z[5])
            
            # Calculer le spread avec ce gamma
            spread = log_PA.values - gamma * log_PB.values
            
            # Créer les paramètres
            params = ModelParams(
                theta0=theta0,
                theta1=theta1,
                theta2=0.0,
                q_base=q_base,
                q_het=q_het,
                r=r,
            )
            
            # Régularisation
            reg = 0.0
            
            # Pénaliser racine unitaire
            if abs(theta1) > 0.999:
                reg += 1000.0 * (abs(theta1) - 0.999)**2
            
            # Run filter
            ll, _, _, _, _ = filter_func(spread, params)
            
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            
            return -ll + reg
            
        except Exception as e:
            return 1e10
    
    # Optimiser
    if method.lower() == "differential_evolution":
        result = differential_evolution(
            neg_loglik, bounds, seed=seed, maxiter=maxiter,
            polish=True, tol=1e-8, atol=1e-8, workers=1
        )
    else:
        result = minimize(
            neg_loglik, z0, method=method, bounds=bounds,
            options={"maxiter": maxiter}
        )
    
    # Extraire les paramètres optimaux
    z_opt = result.x
    gamma = z_opt[0]
    theta0 = z_opt[1]
    theta1 = np.tanh(z_opt[2])
    
    if model_type.lower() == "model_i":
        q_base = np.exp(z_opt[3])
        q_het = 0.0
        r = np.exp(z_opt[4])
    else:
        q_base = np.exp(z_opt[3])
        q_het = np.exp(z_opt[4])
        r = np.exp(z_opt[5])
    
    params = ModelParams(
        theta0=theta0,
        theta1=theta1,
        theta2=0.0,
        q_base=q_base,
        q_het=q_het,
        r=r,
    )
    
    # Calculer le spread final et filtrer
    spread = log_PA - gamma * log_PB
    spread.name = "spread"
    
    ll, x_f, x_p, P_f, P_p = filter_func(spread.values, params)
    
    # Critères d'information
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n)
    
    if verbose:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{model_type.upper()} - Estimation Jointe")
        logger.info('=' * 60)
        logger.info(f"  γ (gamma)   = {gamma:.4f}")
        logger.info(f"  θ0          = {theta0:.6f}")
        logger.info(f"  θ1          = {theta1:.6f}")
        logger.info(f"  √q (θ2)     = {np.sqrt(q_base):.6f}")
        if q_het > 0:
            logger.info(f"  q_het (θ3)  = {q_het:.6f}")
        logger.info(f"  σε² (r)     = {r:.6f}")
        logger.info(f"  log-lik     = {ll:.2f}")
        logger.info(f"  AIC         = {aic:.2f}")
        logger.info(f"  BIC         = {bic:.2f}")
        logger.info(f"  C (mean)    = {params.long_run_mean:.6f}")
        logger.info(f"  σ (std)     = {params.long_run_std:.6f}")
    
    return JointFilterResult(
        gamma=gamma,
        params=params,
        x_filt=pd.Series(x_f, index=log_PA.index, name="x_filt"),
        x_pred=pd.Series(x_p, index=log_PA.index, name="x_pred"),
        P_filt=pd.Series(P_f, index=log_PA.index, name="P_filt"),
        P_pred=pd.Series(P_p, index=log_PA.index, name="P_pred"),
        loglik=ll,
        aic=aic,
        bic=bic,
        model_type=model_type,
        spread=spread,
    )


def compare_estimation_methods(
    log_PA: pd.Series,
    log_PB: pd.Series,
    paper_values: dict,
    verbose: bool = True,
) -> dict:
    """
    Comparer l'estimation standard vs l'estimation jointe.
    
    Parameters
    ----------
    log_PA, log_PB : pd.Series
        Log prix des deux actifs
    paper_values : dict
        Valeurs du papier pour comparaison
    verbose : bool
        Afficher les résultats
        
    Returns
    -------
    dict
        Résultats des deux méthodes
    """
    from pairs_ssm import compute_spread, fit_model
    
    # Méthode 1: Standard (γ par OLS, puis MLE)
    spread_std = compute_spread(log_PA, log_PB)
    result_std = fit_model(spread_std.spread, model_type="model_I", verbose=False)
    
    # Méthode 2: Jointe (γ et params ensemble)
    result_joint = fit_joint(log_PA, log_PB, model_type="model_I", verbose=False)
    
    if verbose:
        print("\n" + "=" * 70)
        print("COMPARAISON: Standard vs Joint vs Papier")
        print("=" * 70)
        print(f"\n{'Paramètre':<12} {'Standard':>15} {'Joint':>15} {'Papier':>15}")
        print("-" * 58)
        
        params = [
            ("γ", spread_std.gamma, result_joint.gamma, paper_values.get('gamma_m1', 'N/A')),
            ("θ0", result_std.params.theta0, result_joint.params.theta0, paper_values.get('theta0_m1', 'N/A')),
            ("θ1", result_std.params.theta1, result_joint.params.theta1, paper_values.get('theta1_m1', 'N/A')),
            ("√q", np.sqrt(result_std.params.q_base), np.sqrt(result_joint.params.q_base), paper_values.get('theta2_m1', 'N/A')),
            ("σε²", result_std.params.r, result_joint.params.r, paper_values.get('sigma_eps_m1', 'N/A')),
        ]
        
        for name, std, joint, paper in params:
            if isinstance(paper, float):
                print(f"{name:<12} {std:>15.6f} {joint:>15.6f} {paper:>15.6f}")
            else:
                print(f"{name:<12} {std:>15.6f} {joint:>15.6f} {paper:>15}")
    
    return {
        "standard": {"gamma": spread_std.gamma, "params": result_std.params, "result": result_std},
        "joint": {"gamma": result_joint.gamma, "params": result_joint.params, "result": result_joint},
    }
