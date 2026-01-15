"""
Maximum Likelihood Estimation for state-space models.
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional, List, Tuple, Callable
from scipy.optimize import minimize, differential_evolution

from pairs_ssm.models.params import ModelParams, FilterResult
from pairs_ssm.models.model_I import ModelI
from pairs_ssm.models.model_II import ModelII
from pairs_ssm.filtering.kalman_linear import kalman_filter
from pairs_ssm.filtering.kalman_extended import extended_kalman_filter
from pairs_ssm.utils.logging import get_logger

logger = get_logger(__name__)


def fit_model(
    y: pd.Series,
    model_type: Literal["model_I", "model_II"] = "model_I",
    method: str = "Nelder-Mead",
    maxiter: int = 2000,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> FilterResult:
    """
    Fit state-space model by Maximum Likelihood Estimation.
    
    Parameters
    ----------
    y : pd.Series
        Observed spread series
    model_type : str
        Model type: "model_I" (linear) or "model_II" (heteroscedastic)
    method : str
        Optimization method ("Nelder-Mead", "L-BFGS-B", "differential_evolution")
    maxiter : int
        Maximum iterations
    verbose : bool
        Print progress
    seed : int, optional
        Random seed for DE optimizer
        
    Returns
    -------
    FilterResult
        Fitted model with parameters and filtered states
    """
    # Prepare data
    y = y.dropna().astype(float)
    y_arr = y.values
    var_y = float(np.var(y_arr))
    n = len(y_arr)
    
    # Select model class and filter
    if model_type.lower() == "model_i":
        model_cls = ModelI
        filter_func = kalman_filter
        n_params = 4
    elif model_type.lower() == "model_ii":
        model_cls = ModelII
        filter_func = extended_kalman_filter
        n_params = 5
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Get initial parameters
    init_params = model_cls.get_initial_params(y_arr)
    model = model_cls(init_params)
    z0 = model.pack_params()
    bounds = model_cls.get_bounds()
    
    if verbose:
        logger.info(f"Fitting {model_type} by MLE...")
    
    # Objective function
    def neg_loglik(z: np.ndarray) -> float:
        try:
            params = model_cls.unpack_params(z, var_y)
            
            # Regularization for stability
            reg = 0.0
            
            # Penalize very small observation noise
            if params.r < 1e-4 * var_y:
                reg += 100.0 * (1e-4 * var_y - params.r)**2
            
            # Penalize unit root
            if abs(params.theta1) > 0.999:
                reg += 1000.0 * (abs(params.theta1) - 0.999)**2
            
            # Run filter
            ll, _, _, _, _ = filter_func(y_arr, params)
            
            return -ll + reg
            
        except Exception as e:
            return 1e10
    
    # Optimize
    if method.lower() == "differential_evolution":
        result = differential_evolution(
            neg_loglik, bounds, seed=seed, maxiter=maxiter,
            polish=True, workers=-1, tol=1e-6
        )
    else:
        result = minimize(
            neg_loglik, z0, method=method,
            options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6}
        )
    
    # Extract final parameters
    params_hat = model_cls.unpack_params(result.x, var_y)
    
    # Run filter with optimized parameters
    ll, x_f, x_p, P_f, P_p = filter_func(y_arr, params_hat)
    
    # Information criteria
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n)
    
    if verbose:
        logger.info(f"\n{model_type.upper()} Estimation Results:")
        logger.info(f"  theta0  = {params_hat.theta0:.6f}")
        logger.info(f"  theta1  = {params_hat.theta1:.6f}")
        if not params_hat.is_linear:
            logger.info(f"  theta2  = {params_hat.theta2:.6f}")
        logger.info(f"  q_base  = {params_hat.q_base:.2e}")
        if not params_hat.is_homoscedastic:
            logger.info(f"  q_het   = {params_hat.q_het:.2e}")
        logger.info(f"  r       = {params_hat.r:.2e}")
        logger.info(f"  loglik  = {ll:.2f}")
        logger.info(f"  AIC     = {aic:.2f}")
        logger.info(f"  BIC     = {bic:.2f}")
        logger.info(f"  C (mean)= {params_hat.long_run_mean:.6f}")
        logger.info(f"  σ (std) = {params_hat.long_run_std:.6f}")
    
    return FilterResult(
        params=params_hat,
        x_filt=pd.Series(x_f, index=y.index, name="x_filt"),
        x_pred=pd.Series(x_p, index=y.index, name="x_pred"),
        P_filt=pd.Series(P_f, index=y.index, name="P_filt"),
        P_pred=pd.Series(P_p, index=y.index, name="P_pred"),
        loglik=ll,
        aic=aic,
        bic=bic,
        model_type=model_type,
    )


def compare_models(
    y: pd.Series,
    models: List[str] = ["model_I", "model_II"],
    verbose: bool = True,
    **fit_kwargs,
) -> pd.DataFrame:
    """
    Compare multiple models using information criteria.
    
    Parameters
    ----------
    y : pd.Series
        Observed spread
    models : list of str
        Model types to compare
    verbose : bool
        Print comparison
    **fit_kwargs
        Additional arguments to fit_model
        
    Returns
    -------
    pd.DataFrame
        Comparison table with AIC, BIC, log-likelihood
    """
    results = []
    
    for model in models:
        try:
            result = fit_model(y, model_type=model, verbose=False, **fit_kwargs)
            results.append({
                "Model": model,
                "LogLik": result.loglik,
                "AIC": result.aic,
                "BIC": result.bic,
                "theta0": result.params.theta0,
                "theta1": result.params.theta1,
                "q": result.params.q_base,
                "r": result.params.r,
            })
        except Exception as e:
            logger.warning(f"Failed to fit {model}: {e}")
            results.append({
                "Model": model,
                "LogLik": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
            })
    
    df = pd.DataFrame(results)
    
    # Rank by AIC and BIC
    df["AIC_rank"] = df["AIC"].rank()
    df["BIC_rank"] = df["BIC"].rank()
    
    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(df.to_string(index=False))
        
        best_aic = df.loc[df["AIC_rank"] == 1, "Model"].values[0]
        best_bic = df.loc[df["BIC_rank"] == 1, "Model"].values[0]
        print(f"\nBest by AIC: {best_aic}")
        print(f"Best by BIC: {best_bic}")
    
    return df


def cross_validate(
    y: pd.Series,
    model_type: str = "model_I",
    n_folds: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Time-series cross-validation for model selection.
    
    Uses expanding window approach.
    
    Parameters
    ----------
    y : pd.Series
        Observations
    model_type : str
        Model type
    n_folds : int
        Number of validation folds
    verbose : bool
        Print results
        
    Returns
    -------
    dict
        Cross-validation results
    """
    n = len(y)
    fold_size = n // (n_folds + 1)
    
    train_logliks = []
    test_logliks = []
    
    for fold in range(n_folds):
        # Training set: expanding window
        train_end = (fold + 2) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        
        y_train = y.iloc[:train_end]
        y_test = y.iloc[test_start:test_end]
        
        if len(y_test) < 10:
            continue
        
        # Fit on training data
        result = fit_model(y_train, model_type=model_type, verbose=False)
        train_logliks.append(result.loglik / len(y_train))
        
        # Evaluate on test data
        if model_type.lower() == "model_i":
            filter_func = kalman_filter
        else:
            filter_func = extended_kalman_filter
        
        test_ll, _, _, _, _ = filter_func(
            y_test.values,
            result.params,
            x0=float(result.x_filt.iloc[-1]),
            P0=float(result.P_filt.iloc[-1]),
        )
        test_logliks.append(test_ll / len(y_test))
    
    results = {
        "model": model_type,
        "n_folds": len(train_logliks),
        "train_loglik_mean": np.mean(train_logliks),
        "train_loglik_std": np.std(train_logliks),
        "test_loglik_mean": np.mean(test_logliks),
        "test_loglik_std": np.std(test_logliks),
    }
    
    if verbose:
        print(f"\nCross-Validation Results for {model_type}:")
        print(f"  Train LogLik: {results['train_loglik_mean']:.4f} ± {results['train_loglik_std']:.4f}")
        print(f"  Test LogLik:  {results['test_loglik_mean']:.4f} ± {results['test_loglik_std']:.4f}")
    
    return results
