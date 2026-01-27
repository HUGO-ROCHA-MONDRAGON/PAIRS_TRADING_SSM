# utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Read 2-col Excel
# =========================================================

def read_excel_2cols_per_ticker(excel_path: str, sheet_name=0) -> Dict[str, pd.Series]:

    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")

    out = {}
    c = 0

    while c + 1 < df.shape[1]:

        ticker = str(df.columns[c + 1]).strip()

        if ticker == "" or ticker.lower().startswith("unnamed"):
            c += 2
            continue

        tmp = pd.DataFrame({
            "date": df.iloc[:, c],
            "val": df.iloc[:, c + 1]
        })

        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.dropna().sort_values("date")

        if not tmp.empty:
            s = pd.Series(tmp["val"].values, index=tmp["date"].values, name=ticker)
            out[ticker] = s

        c += 2

    return out


# =========================================================
# Availability plot (identique à ton ancien)
# =========================================================

def ticker_start_summary(series_dict: Dict[str, pd.Series], plot=True, title=None):

    rows = []

    for t, s in series_dict.items():
        s = s.dropna()
        if not s.empty:
            rows.append({"ticker": t, "first_date": s.index.min()})

    starts_df = pd.DataFrame(rows).sort_values("first_date")

    years = range(
        starts_df.first_date.dt.year.min(),
        starts_df.first_date.dt.year.max() + 1
    )

    cum = [(starts_df.first_date <= f"{y}-01-01").sum() for y in years]

    yearly_df = pd.DataFrame({"year": years, "cum_starts": cum})
    yearly_df["new_starts"] = yearly_df["cum_starts"].diff().fillna(yearly_df["cum_starts"])

    if plot:
        plt.figure(figsize=(7,4))
        plt.bar(yearly_df.year, yearly_df.new_starts, alpha=0.3)
        plt.plot(yearly_df.year, yearly_df.cum_starts, marker="o")
        plt.title(title or "Ticker availability")
        plt.grid()
        plt.show()

    return starts_df, yearly_df


# =========================================================
# Build + Harmonize (IDENTIQUE ancien comportement)
# =========================================================




def build_and_save_harmonized_wides_from_start(
    input_files: Dict[str, str],
    out_dir: Union[str, Path] = "Data",
    start_date: str = "2019-01-01",
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    - Garde uniquement les variables dont la 1ère valeur (first_valid_index) est <= start_date
    - Conserve les dates à partir de start_date
    - Harmonise les dates entre datasets (union) à partir de start_date
    - Remplit uniquement par ffill (pas de bfill)
    - Retourne les DataFrames finaux (et optionnellement les sauvegarde)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.to_datetime(start_date)

    wides: Dict[str, pd.DataFrame] = {}

    # 1) Load + filter columns based on first valid date <= start_date
    for name, path in input_files.items():
        series_dict = read_excel_2cols_per_ticker(path)

        # Construire wide "brut" (pas de fill ici)
        wide_raw = pd.DataFrame(series_dict).sort_index()

        # Déterminer la première date non-NaN par colonne
        first_valid = {col: wide_raw[col].first_valid_index() for col in wide_raw.columns}

        # Garder uniquement colonnes dont first_valid_date <= start_date
        keep_cols = [
            col for col, dt in first_valid.items()
            if (dt is not None) and (pd.to_datetime(dt) <= start_ts)
        ]

        wide_raw = wide_raw[keep_cols]

        # ffill en gardant l'historique (important pour remplir dès start_date)
        wide_ff = wide_raw.ffill()

        # couper à partir de start_date
        wide_2019 = wide_ff.loc[wide_ff.index >= start_ts].copy()

        wides[name] = wide_2019

    # 2) Union des dates (à partir de start_date) + s'assurer que start_date est inclus
    all_dates = None
    for df in wides.values():
        all_dates = df.index if all_dates is None else all_dates.union(df.index)

    all_dates = all_dates.sort_values()
    if len(all_dates) == 0 or all_dates[0] > start_ts:
        all_dates = all_dates.union(pd.DatetimeIndex([start_ts])).sort_values()
    else:
        # même si start_date est dans la plage mais pas exactement présent, on l'ajoute
        all_dates = all_dates.union(pd.DatetimeIndex([start_ts])).sort_values()

    # 3) Reindex each dataset on common dates + ffill only
    finals: Dict[str, pd.DataFrame] = {}

    for name, df in wides.items():
        df_final = df.reindex(all_dates).ffill()

        finals[name] = df_final

        if save:
            out_path = out_dir / f"{name}_2019_wide_harmonized.xlsx"
            df_final.reset_index().rename(columns={"index": "Date"}).to_excel(out_path, index=False)

    return finals


