# data_loader.py
from __future__ import annotations
import pandas as pd

def _find_column(df: pd.DataFrame, name: str) -> str:
    """Find a column by exact match; raise a clear error if missing."""
    if name in df.columns:
        return name
    # try stripped matches
    stripped = {c.strip(): c for c in df.columns if isinstance(c, str)}
    if name.strip() in stripped:
        return stripped[name.strip()]
    raise KeyError(f"Column '{name}' not found. Available columns sample: {list(df.columns)[:10]}")

def extract_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    date_col_a: str | None = None,
    date_col_b: str | None = None,
) -> pd.DataFrame:
    """
    Extract a clean pair dataframe with columns:
      date, PA, PB
    Works even if each asset has its own date column.

    Parameters
    ----------
    df : wide dataframe
    col_a, col_b : price column names (e.g., 'PEP US Equity', 'KO US Equity')
    date_col_a, date_col_b : optional explicit date column names; if None, tries:
        - 'Date' (single date column)
        - or assumes the date column is immediately left of the price column (as in your screenshot)

    Returns
    -------
    pd.DataFrame indexed by date with PA, PB floats, no missing dates.
    """
    df = df.copy()

    # Locate price columns
    col_a = _find_column(df, col_a)
    col_b = _find_column(df, col_b)

    # Detect date columns
    if date_col_a is None:
        if "Date" in df.columns:
            date_col_a = "Date"
        else:
            # assume date col is immediately to the left
            idx = df.columns.get_loc(col_a)
            if idx == 0:
                raise ValueError(f"Cannot infer date col for {col_a} (it is first column).")
            date_col_a = df.columns[idx - 1]

    if date_col_b is None:
        if date_col_a == "Date":
            date_col_b = "Date"
        else:
            idx = df.columns.get_loc(col_b)
            if idx == 0:
                raise ValueError(f"Cannot infer date col for {col_b} (it is first column).")
            date_col_b = df.columns[idx - 1]

    date_col_a = _find_column(df, date_col_a)
    date_col_b = _find_column(df, date_col_b)

    # Parse dates
    da = pd.to_datetime(df[date_col_a], errors="coerce")
    db = pd.to_datetime(df[date_col_b], errors="coerce")

    pa = pd.to_numeric(df[col_a], errors="coerce")
    pb = pd.to_numeric(df[col_b], errors="coerce")

    a = pd.DataFrame({"date": da, "PA": pa}).dropna(subset=["date"]).dropna(subset=["PA"])
    b = pd.DataFrame({"date": db, "PB": pb}).dropna(subset=["date"]).dropna(subset=["PB"])

    # Inner join on dates
    out = a.merge(b, on="date", how="inner").sort_values("date").drop_duplicates("date")
    out = out.set_index("date")
    out = out.astype(float)

    return out
