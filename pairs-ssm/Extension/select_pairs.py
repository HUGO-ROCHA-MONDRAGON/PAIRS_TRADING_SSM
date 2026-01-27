# select_pairs.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from statsmodels.tsa.stattools import coint


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class PairSelectionConfig:
    # --- clustering ---
    k_min: int = 2
    k_max: int = 10
    mds_dim: int = 5
    random_state: int = 42

    # --- cointegration ---
    alpha: float = 0.05
    min_obs: int = 252
    max_pairs_per_cluster: Optional[int] = None

    # --- preprocessing ---
    series_kind: str = "log_price"     # price | log_price | cumret
    normalize_for_dtw: str = "zscore"  # zscore | none
    dropna_how: str = "inner"          # inner | outer

    # --- dtw ---
    use_tslearn: bool = True

    # --- caching ---
    cache_dir: str = "Data/.cache_pairs"   # dossier cache
    cache: bool = True                     # active / désactive


# ============================================================
# Public API
# ============================================================

def select_pairs_from_wide_df(
    X0: pd.DataFrame,
    cfg: Optional[PairSelectionConfig] = None,
    top_k: int = 3,
    method: str = "optimal",
) -> pd.DataFrame:
    """
    Input:
        wide df (index=Date, columns=tickers)

    Output:
        best top_k pairs
    """
    cfg = cfg or PairSelectionConfig()

    # --- prefilter ---
    keep = [
        c for c in X0.columns
        if X0[c].dropna().shape[0] >= cfg.min_obs
        and X0[c].nunique() > 1
    ]
    X0 = X0[keep]
    if X0.shape[1] < 3:
        return pd.DataFrame()

    # --- preprocess ---
    X, tickers = _to_matrix(X0, cfg)

    # --- cache key (stable) ---
    cache_paths = _get_cache_paths(X0, tickers, cfg)

    # --- DTW distances (cached) ---
    D = _load_npy(cache_paths["dtw"]) if cfg.cache else None
    if D is None:
        D = _dtw_distance_matrix(X, cfg)
        if cfg.cache:
            _save_npy(cache_paths["dtw"], D)

    # --- MDS embedding (cached) ---
    Z = _load_npy(cache_paths["mds"]) if cfg.cache else None
    if Z is None:
        Z = _mds_embedding(D, cfg)
        if cfg.cache:
            _save_npy(cache_paths["mds"], Z)

    # --- clustering ---
    labels = _cluster_from_embedding(Z, cfg)

    # --- cointegration within clusters ---
    pairs = _cointegration_pairs(X, tickers, labels, cfg)

    # --- top k selection (tradability + score) ---
    return select_top_k_pairs(pairs, X, k=top_k, method=method, alpha=cfg.alpha)


# ============================================================
# Preprocessing
# ============================================================

def _to_matrix(df: pd.DataFrame, cfg: PairSelectionConfig) -> Tuple[pd.DataFrame, List[str]]:
    if cfg.dropna_how == "inner":
        df = df.dropna(how="any")
    else:
        df = df.sort_index().interpolate(method="time").ffill().bfill()

    if cfg.series_kind == "log_price":
        df = np.log(df.where(df > 0))
    elif cfg.series_kind == "cumret":
        df = (1.0 + df.pct_change()).cumprod()

    if cfg.normalize_for_dtw == "zscore":
        df = (df - df.mean()) / df.std(ddof=0)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    return df, list(df.columns)


# ============================================================
# DTW
# ============================================================

def _dtw_distance_matrix(X: pd.DataFrame, cfg: PairSelectionConfig) -> np.ndarray:
    data = X.values

    if cfg.use_tslearn:
        try:
            from tslearn.metrics import cdist_dtw
            series = data.T[:, :, None]  # (N, T, 1)
            # n_jobs existe selon versions tslearn -> on tente
            try:
                return cdist_dtw(series, n_jobs=-1)
            except TypeError:
                return cdist_dtw(series)
        except Exception:
            pass

    # fallback simple dtw (lent) - mais ton cas normal utilise tslearn
    def dtw(a, b):
        n, m = len(a), len(b)
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = (a[i - 1] - b[j - 1]) ** 2
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(np.sqrt(dp[n, m]))

    N = data.shape[1]
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            d = dtw(data[:, i], data[:, j])
            D[i, j] = D[j, i] = d
    return D


# ============================================================
# MDS embedding
# ============================================================

def _mds_embedding(D: np.ndarray, cfg: PairSelectionConfig) -> np.ndarray:
    mds = MDS(
        n_components=cfg.mds_dim,
        dissimilarity="precomputed",
        random_state=cfg.random_state,
        n_init=4,
        max_iter=300,
    )
    return mds.fit_transform(D)


# ============================================================
# Clustering
# ============================================================

def _cluster_from_embedding(Z: np.ndarray, cfg: PairSelectionConfig) -> np.ndarray:
    best_score = -np.inf
    best_labels = None

    k_min = max(2, cfg.k_min)
    k_max = min(cfg.k_max, Z.shape[0] - 1)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=cfg.random_state)
        labels = km.fit_predict(Z)
        try:
            score = silhouette_score(Z, labels)
        except Exception:
            score = -np.inf

        if score > best_score:
            best_score = score
            best_labels = labels

    if best_labels is None:
        raise RuntimeError("Clustering failed (silhouette).")

    return best_labels
# ============================================================
# Correlation
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def drop_highly_correlated_columns(
    df: pd.DataFrame,
    k: float = 0.9995,
    method: str = "logret",   # "logret" or "level"
    min_obs: int = 252,
    plot: bool = True,
    title: str = None,
):
    """
    Retourne:
      - df_kept : df avec colonnes filtrées (on garde 1 colonne par groupe corr>k)
      - corr    : matrice de corr utilisée
      - dropped : liste des colonnes supprimées

    Règle:
      - on calcule la corr
      - on parcourt la matrice (triangle supérieur)
      - si corr(i,j) > k, on drop la colonne "j" (ou celle que tu veux)
    """

    X = df.copy()

    # 1) construire la matrice pour corr
    if method == "logret":
        Xc = np.log(X).diff()
    elif method == "level":
        Xc = X.copy()
    else:
        raise ValueError("method must be 'logret' or 'level'")

    # garder seulement obs suffisantes
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    Xc = Xc.dropna(axis=0, how="any")

    # mini garde-fou
    if Xc.shape[0] < min_obs or Xc.shape[1] < 2:
        return df, pd.DataFrame(), []

    corr = Xc.corr()

    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, aspect="auto")  # pas seaborn
        ax.set_title(title or f"Correlation matrix ({method})")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    # 2) drop greedy
    to_drop = set()
    cols = corr.columns.tolist()

    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if abs(corr.iloc[i, j]) > k:
                # règle simple: on drop j
                to_drop.add(cols[j])

    dropped = sorted(to_drop)
    kept = [c for c in df.columns if c not in to_drop]

    return df[kept], corr, dropped


# ============================================================
# Cointegration
# ============================================================

def _cointegration_pairs(X: pd.DataFrame, tickers: List[str], labels: np.ndarray, cfg: PairSelectionConfig) -> pd.DataFrame:
    results = []
    clusters: Dict[int, List[int]] = {}

    for i, l in enumerate(labels):
        clusters.setdefault(int(l), []).append(i)

    for cl, idxs in clusters.items():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):

                s1 = X.iloc[:, idxs[a]]
                s2 = X.iloc[:, idxs[b]]
                df2 = pd.concat([s1, s2], axis=1).dropna()

                if len(df2) < cfg.min_obs:
                    continue
                if df2.iloc[:,0].corr(df2.iloc[:,1]) > 0.999:
                    continue

                try:
                    stat, pval, _ = coint(df2.iloc[:, 0], df2.iloc[:, 1])
                except Exception:
                    continue

                if pval <= cfg.alpha:
                    results.append({
                        "cluster": cl,
                        "ticker_1": tickers[idxs[a]],
                        "ticker_2": tickers[idxs[b]],
                        "pvalue": float(pval),
                        "coint_stat": float(stat),
                        "n_obs": int(len(df2)),
                    })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values(["cluster", "pvalue"], ascending=[True, True]).reset_index(drop=True)

    if cfg.max_pairs_per_cluster is not None:
        out = out.groupby("cluster", group_keys=False).head(cfg.max_pairs_per_cluster).reset_index(drop=True)

    return out


# ============================================================
# Tradability + scoring (inchangé vs ton style)
# ============================================================

def _half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 50:
        return np.inf

    ds = spread.diff().dropna()
    lag = spread.shift(1).loc[ds.index]
    b = np.polyfit(lag.values, ds.values, 1)[0]

    if not np.isfinite(b) or b >= 0:
        return np.inf
    return float(-np.log(2) / b)


def select_top_k_pairs(pairs_df: pd.DataFrame, X: pd.DataFrame, k: int = 3, method: str = "greedy", alpha: float = 0.05) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df

    df = pairs_df.copy()

    hl, stds = [], []
    for _, r in df.iterrows():
        a, b = r["ticker_1"], r["ticker_2"]
        y, x = X[a], X[b]

        # OLS y ~ x + c
        Xreg = np.column_stack([np.ones_like(x.values), x.values])
        beta = np.linalg.lstsq(Xreg, y.values, rcond=None)[0]
        spread = y.values - (Xreg @ beta)
        spread = pd.Series(spread, index=y.index)

        hl.append(_half_life(spread))
        stds.append(float(np.nanstd(spread.values, ddof=0)))

    df["half_life"] = hl
    df["spread_std"] = stds

    df = df[(df["half_life"] < 120) & (df["spread_std"] > 0) & (df["pvalue"] <= alpha)].copy()

    if df.empty:
        return df

    df["score"] = -np.log(df["pvalue"].clip(1e-12, 1.0)) + df["spread_std"] - df["half_life"] / 100.0
    df = df.sort_values("score", ascending=False)

    used = set()
    chosen = []
    for _, r in df.iterrows():
        a, b = r["ticker_1"], r["ticker_2"]
        if a in used or b in used:
            continue
        chosen.append(r)
        used.add(a); used.add(b)
        if len(chosen) >= k:
            break

    return pd.DataFrame(chosen).reset_index(drop=True)


# ============================================================
# Cache helpers
# ============================================================

def _get_cache_paths(X0: pd.DataFrame, tickers: List[str], cfg: PairSelectionConfig) -> Dict[str, Path]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "tickers": tickers,
        "shape": list(X0.shape),
        "start": str(pd.to_datetime(X0.index.min()).date()) if len(X0.index) else None,
        "end": str(pd.to_datetime(X0.index.max()).date()) if len(X0.index) else None,
        "cfg": {
            "k_min": cfg.k_min, "k_max": cfg.k_max, "mds_dim": cfg.mds_dim, "random_state": cfg.random_state,
            "alpha": cfg.alpha, "min_obs": cfg.min_obs, "max_pairs_per_cluster": cfg.max_pairs_per_cluster,
            "series_kind": cfg.series_kind, "normalize_for_dtw": cfg.normalize_for_dtw, "dropna_how": cfg.dropna_how,
            "use_tslearn": cfg.use_tslearn,
        }
    }
    key = hashlib.sha1(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    return {
        "dtw": cache_dir / f"dtw_{key}.npy",
        "mds": cache_dir / f"mds_{key}.npy",
    }


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _load_npy(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        return np.load(path, allow_pickle=False)
    return None
