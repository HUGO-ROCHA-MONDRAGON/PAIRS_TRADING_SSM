"""
Data loading utilities for pairs trading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PairData:
    """Container for pair price data."""
    
    PA: pd.Series  # Price series for asset A
    PB: pd.Series  # Price series for asset B
    asset_a: str   # Asset A identifier
    asset_b: str   # Asset B identifier
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not self.PA.index.equals(self.PB.index):
            raise ValueError("Price series must have aligned indices")
    
    @property
    def dates(self) -> pd.DatetimeIndex:
        """Return date index."""
        return self.PA.index
    
    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return len(self.PA)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({"PA": self.PA, "PB": self.PB})


def _find_column(df: pd.DataFrame, name: str) -> str:
    """Find column by exact or stripped match."""
    if name in df.columns:
        return name
    
    # Try stripped matches
    stripped = {c.strip(): c for c in df.columns if isinstance(c, str)}
    if name.strip() in stripped:
        return stripped[name.strip()]
    
    raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)[:10]}")


def _detect_date_column(df: pd.DataFrame, price_col: str) -> str:
    """Detect date column for a price column."""
    # Check for common date column names
    for date_name in ["Date", "date", "DATE", "Dates", "dates"]:
        if date_name in df.columns:
            return date_name
    
    # Assume date column is immediately to the left of price column
    idx = df.columns.get_loc(price_col)
    if idx == 0:
        raise ValueError(f"Cannot infer date column for {price_col}")
    
    return df.columns[idx - 1]


def load_excel(
    path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
) -> pd.DataFrame:
    """
    Load data from Excel file.
    
    Parameters
    ----------
    path : str or Path
        Path to Excel file
    sheet_name : str or int
        Sheet name or index
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return pd.read_excel(path, sheet_name=sheet_name)


def load_csv(
    path: Union[str, Path],
    date_col: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Parameters
    ----------
    path : str or Path
        Path to CSV file
    date_col : str, optional
        Date column name (will be parsed and set as index)
    **kwargs
        Additional arguments to pd.read_csv
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path, **kwargs)
    
    if date_col is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    
    return df


def load_pair(
    path: Union[str, Path],
    col_a: str,
    col_b: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_col: Optional[str] = None,
) -> PairData:
    """
    Load a pair of price series from file.
    
    Parameters
    ----------
    path : str or Path
        Path to data file (Excel or CSV)
    col_a : str
        Column name for asset A prices
    col_b : str
        Column name for asset B prices
    start_date : str, optional
        Start date for sample (inclusive)
    end_date : str, optional
        End date for sample (inclusive)
    date_col : str, optional
        Explicit date column name
        
    Returns
    -------
    PairData
        Container with aligned price series
    """
    path = Path(path)
    
    # Load based on file extension
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = load_excel(path)
    elif path.suffix.lower() == ".csv":
        df = load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    # Find price columns
    col_a = _find_column(df, col_a)
    col_b = _find_column(df, col_b)
    
    # Detect date columns
    if date_col is None:
        date_col_a = _detect_date_column(df, col_a)
    else:
        date_col_a = _find_column(df, date_col)
    
    date_col_b = date_col_a  # Assume same date column
    
    # Parse dates
    dates_a = pd.to_datetime(df[date_col_a], errors="coerce")
    dates_b = pd.to_datetime(df[date_col_b], errors="coerce")
    
    # Get prices
    prices_a = pd.to_numeric(df[col_a], errors="coerce")
    prices_b = pd.to_numeric(df[col_b], errors="coerce")
    
    # Create series
    series_a = pd.DataFrame({"date": dates_a, "price": prices_a}).dropna()
    series_b = pd.DataFrame({"date": dates_b, "price": prices_b}).dropna()
    
    # Inner join on dates
    merged = series_a.merge(series_b, on="date", how="inner", suffixes=("_a", "_b"))
    merged = merged.sort_values("date").drop_duplicates("date")
    merged = merged.set_index("date")
    
    PA = merged["price_a"].astype(float)
    PB = merged["price_b"].astype(float)
    
    PA.name = col_a
    PB.name = col_b
    
    # Apply date filter
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        PA = PA[PA.index >= start_date]
        PB = PB[PB.index >= start_date]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        PA = PA[PA.index <= end_date]
        PB = PB[PB.index <= end_date]
    
    return PairData(
        PA=PA,
        PB=PB,
        asset_a=col_a,
        asset_b=col_b,
    )
