"""Diagnostics pour choisir k et juger la stabilité des clusters.

Trois familles, à croiser (aucune ne suffit seule) :
  - CDF / aire sous la CDF / delta-K  (Monti et al. 2003)
  - PAC : proportion of ambiguous clustering (Senbabaoglu et al. 2014) —
    plus fiable que delta-K, qui a un biais monotone bien documenté
  - consensus par item / par cluster + silhouette sur la distance consensus
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score

from consensus import ConsensusResult

logger = logging.getLogger(__name__)


def _upper(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(C, k=1)
    return C[iu]


def consensus_cdf(C: np.ndarray, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """CDF empirique des valeurs de consensus hors diagonale."""
    vals = _upper(C)
    grid = np.linspace(0, 1, n_points)
    cdf = np.array([(vals <= c).mean() for c in grid])
    return grid, cdf


def auc_cdf(C: np.ndarray) -> float:
    """Aire sous la CDF (A(k) de Monti). Une CDF proche d'une marche
    0/1 (consensus parfait) donne une aire élevée."""
    vals = np.sort(_upper(C))
    # A = sum_{i=2}^{m} (x_i - x_{i-1}) * CDF(x_i)
    m = vals.size
    cdf = np.arange(1, m + 1) / m
    return float(np.sum(np.diff(vals) * cdf[1:]))


def delta_k(result: ConsensusResult) -> pd.DataFrame:
    """Gain relatif d'aire sous la CDF entre k-1 et k."""
    ks = sorted(result.consensus)
    areas = {k: auc_cdf(result.consensus[k]) for k in ks}
    rows = []
    for i, k in enumerate(ks):
        if i == 0:
            d = areas[k]
        else:
            prev = areas[ks[i - 1]]
            d = (areas[k] - prev) / prev if prev > 0 else np.nan
        rows.append({"k": k, "auc_cdf": areas[k], "delta_k": d})
    return pd.DataFrame(rows)


def pac(C: np.ndarray, u1: float = 0.1, u2: float = 0.9) -> float:
    """Proportion of Ambiguous Clustering : fraction des paires dont le
    consensus tombe dans l'intervalle intermédiaire ]u1, u2[.
    **Plus c'est bas, mieux c'est** — le k optimal minimise le PAC."""
    vals = _upper(C)
    return float(((vals > u1) & (vals < u2)).mean())


def item_consensus(result: ConsensusResult, k: int) -> pd.DataFrame:
    """Consensus moyen de chaque patient avec les membres de son cluster.

    Un item consensus faible signale une tumeur "intermédiaire" ou mal classée,
    typiquement les cas atypiques qu'on veut repérer.
    """
    C = result.consensus[k]
    labels = result.labels(k)
    rows = []
    for i in range(C.shape[0]):
        own = labels == labels[i]
        own[i] = False
        rows.append(
            {
                "sample": result.sample_names[i],
                "cluster": int(labels[i]),
                "item_consensus": float(C[i, own].mean()) if own.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cluster_consensus(result: ConsensusResult, k: int) -> pd.DataFrame:
    """Consensus moyen intra-cluster (compacité de chaque groupe)."""
    C = result.consensus[k]
    labels = result.labels(k)
    rows = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if idx.size < 2:
            rows.append({"cluster": int(c), "n": idx.size, "cluster_consensus": np.nan})
            continue
        sub = C[np.ix_(idx, idx)]
        iu = np.triu_indices_from(sub, k=1)
        rows.append(
            {"cluster": int(c), "n": int(idx.size),
             "cluster_consensus": float(sub[iu].mean())}
        )
    return pd.DataFrame(rows)


def summary(result: ConsensusResult, pac_bounds: tuple[float, float] = (0.1, 0.9)
            ) -> pd.DataFrame:
    """Tableau récapitulatif par k : PAC, aire CDF, delta-K, silhouette,
    taille du plus petit cluster."""
    dk = delta_k(result).set_index("k")
    rows = []
    for k in sorted(result.consensus):
        C = result.consensus[k]
        D = result.distance(k)
        labels = result.labels(k)
        sizes = np.bincount(labels)[1:]
        try:
            sil = silhouette_score(D, labels, metric="precomputed")
        except ValueError:
            sil = np.nan
        rows.append(
            {
                "k": k,
                "PAC": pac(C, *pac_bounds),
                "auc_cdf": dk.loc[k, "auc_cdf"],
                "delta_k": dk.loc[k, "delta_k"],
                "silhouette_consensus": sil,
                "min_cluster_size": int(sizes.min()),
                "mean_cluster_consensus": float(
                    cluster_consensus(result, k)["cluster_consensus"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def silhouette_per_sample(result: ConsensusResult, k: int) -> pd.DataFrame:
    D = result.distance(k)
    labels = result.labels(k)
    return pd.DataFrame(
        {
            "sample": result.sample_names,
            "cluster": labels,
            "silhouette": silhouette_samples(D, labels, metric="precomputed"),
        }
    )


def suggest_k(
    result: ConsensusResult,
    min_cluster_size: int = 10,
    method: str = "both",
    pac_tol: float = 0.01,
    min_delta: float = 0.05,
) -> int:
    """Heuristique de choix de k. **À valider par la heatmap, jamais à suivre
    aveuglément.**

    - ``"pac"`` : k qui **minimise le PAC** (proportion de paires ambiguës), sur
      **tous** les k testés. Fiable, mais peut sous-estimer k quand le PAC sature.
    - ``"deltak"`` : **coude de Δ(K)** — le plus grand k qui apporte encore un
      gain d'aire sous la CDF d'au moins `min_delta`. Sur-estime souvent k.
    - ``"both"`` (défaut) : parmi les k à `pac_tol` près du PAC minimal **et**
      dont le plus petit cluster fait ≥ `min_cluster_size` tumeurs, le plus grand
      qui garde un Δ(K) ≥ `min_delta`.

    `min_cluster_size` ne **filtre en dur que pour `both`** (l'auto par défaut).
    Pour les critères explicites `pac` / `deltak`, le critère est respecté à la
    lettre et un **avertissement** est émis si le k retenu produit un petit
    cluster — l'utilisateur garde le contrôle.
    """
    tab = summary(result)

    def _warn_small(k: int) -> int:
        row = tab.loc[tab["k"] == k].iloc[0]
        if row["min_cluster_size"] < min_cluster_size:
            logger.warning(
                "suggest_k(%s) : k=%d retenu (PAC=%.3f) mais son plus petit cluster "
                "ne fait que %d tumeurs (< min_cluster_size=%d). Baisse "
                "min_cluster_size ou fixe k_final si ce n'est pas voulu.",
                method, k, row["PAC"], int(row["min_cluster_size"]), min_cluster_size)
        return k

    if method == "pac":
        return _warn_small(int(tab.loc[tab["PAC"].idxmin(), "k"]))

    if method == "deltak":
        gain = tab[tab["delta_k"] >= min_delta]
        k = int(gain["k"].max()) if len(gain) else int(tab.loc[tab["delta_k"].idxmax(), "k"])
        return _warn_small(k)

    if method == "both":
        ok = tab[tab["min_cluster_size"] >= min_cluster_size]
        base = ok if len(ok) else tab
        near = base[base["PAC"] <= base["PAC"].min() + pac_tol]
        gain = near[near["delta_k"] >= min_delta]
        if len(gain):
            return int(gain["k"].max())
        return int(near.loc[near["PAC"].idxmin(), "k"])

    raise ValueError(f"method inconnu : {method!r} (pac | deltak | both).")
