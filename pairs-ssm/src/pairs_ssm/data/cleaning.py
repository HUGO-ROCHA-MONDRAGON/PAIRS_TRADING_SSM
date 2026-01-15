"""
Data cleaning utilities.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


def align_series(
    *series: pd.Series,
    how: str = "inner",
) -> Tuple[pd.Series, ...]:
    """
    Align multiple series to common dates.
    
    Parameters
    ----------
    *series : pd.Series
        Series to align
    how : str
        Join method: "inner", "outer", "left", "right"
        
    Returns
    -------
    tuple of pd.Series
        Aligned series
    """
    if len(series) == 0:
        return tuple()
    
    if len(series) == 1:
        return (series[0].dropna(),)
    
    # Create DataFrame for alignment
    df = pd.DataFrame({f"s{i}": s for i, s in enumerate(series)})
    
    if how == "inner":
        df = df.dropna()
    elif how == "outer":
        pass  # Keep all
    
    return tuple(df[f"s{i}"] for i in range(len(series)))


def fill_missing(
    series: pd.Series,
    method: str = "ffill",
    limit: Optional[int] = None,
) -> pd.Series:
    """
    Fill missing values in a series.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    method : str
        Fill method: "ffill", "bfill", "interpolate", "mean"
    limit : int, optional
        Maximum consecutive fills
        
    Returns
    -------
    pd.Series
        Series with filled values
    """
    series = series.copy()
    
    if method == "ffill":
        return series.ffill(limit=limit)
    elif method == "bfill":
        return series.bfill(limit=limit)
    elif method == "interpolate":
        return series.interpolate(method="linear", limit=limit)
    elif method == "mean":
        return series.fillna(series.mean())
    else:
        raise ValueError(f"Unknown fill method: {method}")


def remove_outliers(
    series: pd.Series,
    method: str = "zscore",
    threshold: float = 3.0,
    replace_with: str = "nan",
) -> pd.Series:
    """
    Remove or replace outliers in a series.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    method : str
        Detection method: "zscore", "iqr", "percentile"
    threshold : float
        Threshold for outlier detection
        - zscore: number of standard deviations
        - iqr: multiplier for IQR
        - percentile: percentile bounds (e.g., 0.01 for 1%/99%)
    replace_with : str
        How to handle outliers: "nan", "clip", "median"
        
    Returns
    -------
    pd.Series
        Series with outliers handled
    """
    series = series.copy()
    
    if method == "zscore":
        mean = series.mean()
        std = series.std()
        zscore = np.abs((series - mean) / std)
        is_outlier = zscore > threshold
        
    elif method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        is_outlier = (series < lower) | (series > upper)
        
    elif method == "percentile":
        lower = series.quantile(threshold)
        upper = series.quantile(1 - threshold)
        is_outlier = (series < lower) | (series > upper)
        
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    
    if replace_with == "nan":
        series[is_outlier] = np.nan
    elif replace_with == "clip":
        if method == "zscore":
            lower = mean - threshold * std
            upper = mean + threshold * std
        series = series.clip(lower=lower, upper=upper)
    elif replace_with == "median":
        series[is_outlier] = series.median()
    else:
        raise ValueError(f"Unknown replace method: {replace_with}")
    
    return series


def check_stationarity(
    series: pd.Series,
    significance: float = 0.05,
) -> dict:
    """
    Check stationarity of a series using ADF test.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    significance : float
        Significance level for test
        
    Returns
    -------
    dict
        Test results including statistic, p-value, and conclusion
    """
    from scipy import stats
    
    series = series.dropna()
    
    # Simple ADF-like test (lagged difference regression)
    # For proper ADF, use statsmodels if available
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, autolag="AIC")
        return {
            "statistic": result[0],
            "pvalue": result[1],
            "lags": result[2],
            "nobs": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < significance,
        }
    except ImportError:
        # Fallback: simple variance ratio test
        n = len(series)
        half = n // 2
        var1 = series.iloc[:half].var()
        var2 = series.iloc[half:].var()
        
        return {
            "statistic": var1 / var2 if var2 > 0 else np.inf,
            "pvalue": np.nan,
            "is_stationary": None,
            "note": "statsmodels not available, using variance ratio",
        }
