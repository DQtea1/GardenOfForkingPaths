"""Consensus clustering (Monti et al., Machine Learning 2003) étendu à un
rééchantillonnage **double** : patients ET gènes.

Principe
--------
Pour b = 1..B :
  1. tirer un sous-ensemble de patients  (subsample 80 % ou bootstrap)
  2. tirer un sous-ensemble de gènes     (subsample 80 % ou bootstrap)
  3. clusteriser la sous-matrice pour chaque k testé
  4. incrémenter la matrice de connectivité M^(k)[i,j] si i et j tombent
     dans le même cluster, et la matrice d'indicatrice I[i,j] si i et j ont
     été tirés ensemble
Consensus : C^(k) = M^(k) / I,  puis distance consensus D = 1 - C.

Le rééchantillonnage des gènes est ce qui distingue cette implémentation de
ConsensusClusterPlus par défaut : il teste la stabilité des clusters vis-à-vis
du choix des features, pas seulement des individus. C'est important quand les
groupes reposent sur un petit nombre de programmes transcriptionnels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------
def pairwise_distance(X: np.ndarray, metric: str = "pearson") -> np.ndarray:
    """Matrice de distance `n x n` entre échantillons (X : samples x genes)."""
    if metric == "pearson":
        D = 1.0 - np.corrcoef(X)
    elif metric == "spearman":
        D = 1.0 - np.corrcoef(rankdata(X, axis=1))
    elif metric == "euclidean":
        D = squareform(pdist(X, metric="euclidean"))
    elif metric == "cosine":
        D = squareform(pdist(X, metric="cosine"))
    else:
        raise ValueError(f"metric inconnue : {metric}")
    D = np.nan_to_num(D, nan=1.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------------------
# Clusteriseurs de base
# --------------------------------------------------------------------------
def _kmedoids(D: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 5,
              max_iter: int = 100) -> np.ndarray:
    """PAM simplifié (alternating k-medoids) sur distance précalculée.

    Évite une dépendance à scikit-learn-extra ; suffisant comme apprenant de
    base puisque le consensus lisse le bruit de chaque run individuel.
    """
    n = D.shape[0]
    best_labels, best_cost = None, np.inf
    for _ in range(n_init):
        medoids = rng.choice(n, size=k, replace=False)
        labels = np.argmin(D[:, medoids], axis=1)
        for _ in range(max_iter):
            new_medoids = medoids.copy()
            for c in range(k):
                members = np.flatnonzero(labels == c)
                if members.size == 0:
                    new_medoids[c] = rng.integers(n)
                    continue
                sub = D[np.ix_(members, members)]
                new_medoids[c] = members[np.argmin(sub.sum(axis=1))]
            new_labels = np.argmin(D[:, new_medoids], axis=1)
            if np.array_equal(new_labels, labels) and np.array_equal(new_medoids, medoids):
                break
            medoids, labels = new_medoids, new_labels
        cost = D[np.arange(n), medoids[labels]].sum()
        if cost < best_cost:
            best_cost, best_labels = cost, labels
    return best_labels


def _cluster(X: np.ndarray, D: np.ndarray | None, k: int, base: str,
             linkage_method: str, rng: np.random.Generator) -> np.ndarray:
    """Renvoie des labels 0..k-1 pour une sous-matrice donnée."""
    if base == "hierarchical":
        Z = linkage(squareform(D, checks=False), method=linkage_method)
        return fcluster(Z, t=k, criterion="maxclust") - 1
    if base == "kmeans":
        seed = int(rng.integers(0, 2**31 - 1))
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    if base == "kmedoids":
        return _kmedoids(D, k, rng)
    raise ValueError(f"base inconnue : {base}")


# --------------------------------------------------------------------------
# Rééchantillonnage
# --------------------------------------------------------------------------
def _draw(n: int, prop: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    size = max(2, int(round(prop * n)))
    if mode == "subsample":
        return rng.choice(n, size=size, replace=False)
    if mode == "bootstrap":
        return rng.choice(n, size=size, replace=True)
    raise ValueError(f"mode inconnu : {mode}")


def _one_resample(seed: int, X: np.ndarray, k_values: tuple[int, ...],
                  prop_samples: float, prop_genes: float,
                  sample_mode: str, gene_mode: str,
                  base: str, metric: str, linkage_method: str):
    """Un tirage : renvoie (indices patients uniques, {k: labels})."""
    rng = np.random.default_rng(seed)
    n, p = X.shape

    idx_s = _draw(n, prop_samples, sample_mode, rng)
    idx_g = _draw(p, prop_genes, gene_mode, rng)

    # En bootstrap patients, un même patient tiré plusieurs fois donnerait des
    # lignes dupliquées : on ne garde qu'une occurrence pour la connectivité.
    idx_s = np.unique(idx_s)

    Xb = X[np.ix_(idx_s, idx_g)]
    D = pairwise_distance(Xb, metric) if base != "kmeans" else None

    labels = {k: _cluster(Xb, D, k, base, linkage_method, rng) for k in k_values}
    return idx_s, labels


# --------------------------------------------------------------------------
# API principale
# --------------------------------------------------------------------------
@dataclass
class ConsensusResult:
    """Résultat d'un consensus clustering multi-k."""

    consensus: dict[int, np.ndarray]          # k -> matrice consensus n x n
    indicator: np.ndarray                     # n x n, nb de co-tirages
    sample_names: np.ndarray
    params: dict = field(default_factory=dict)

    def distance(self, k: int) -> np.ndarray:
        """Distance consensus D = 1 - C (diagonale nulle, symétrique)."""
        D = 1.0 - self.consensus[k]
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        return np.clip(D, 0.0, 1.0)

    def labels(self, k: int, linkage_method: str = "average") -> np.ndarray:
        """Partition finale : CAH sur la distance consensus, coupée à k."""
        Z = linkage(squareform(self.distance(k), checks=False), method=linkage_method)
        return fcluster(Z, t=k, criterion="maxclust")

    def order(self, k: int, linkage_method: str = "average") -> np.ndarray:
        """Ordre des échantillons issu du dendrogramme (pour les heatmaps)."""
        from scipy.cluster.hierarchy import leaves_list

        Z = linkage(squareform(self.distance(k), checks=False), method=linkage_method)
        return leaves_list(Z)


def consensus_clustering(
    X: np.ndarray,
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    n_resamples: int = 1000,
    prop_samples: float = 0.8,
    prop_genes: float = 0.8,
    sample_mode: str = "subsample",
    gene_mode: str = "subsample",
    base: str = "hierarchical",
    metric: str = "pearson",
    linkage_method: str = "average",
    sample_names: np.ndarray | None = None,
    random_state: int = 0,
    n_jobs: int = -1,
) -> ConsensusResult:
    """Consensus clustering avec rééchantillonnage patients + gènes.

    Parameters
    ----------
    X : array `samples x genes`, déjà normalisé et filtré.
    k_values : nombres de clusters à évaluer.
    n_resamples : B, nombre de tirages (>= 500 recommandé, 1000 par défaut).
    prop_samples, prop_genes : proportion tirée à chaque itération.
    sample_mode, gene_mode : "subsample" (sans remise, Monti) ou "bootstrap"
        (avec remise). Le bootstrap sur les gènes duplique des features et
        repondère donc implicitement la distance — informatif, mais moins
        classique ; le subsample est le défaut recommandé.
    base : "hierarchical" | "kmeans" | "kmedoids".
    metric : distance de base entre patients ("pearson" par défaut, standard
        en transcriptomique car insensible aux effets d'échelle par échantillon).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    k_values = tuple(sorted(set(int(k) for k in k_values)))
    if sample_names is None:
        sample_names = np.array([f"S{i}" for i in range(n)])

    seeds = np.random.SeedSequence(random_state).generate_state(n_resamples)

    logger.info(
        "Consensus clustering : B=%d, k=%s, base=%s, metric=%s, "
        "patients=%s %.0f%%, gènes=%s %.0f%%",
        n_resamples, k_values, base, metric,
        sample_mode, 100 * prop_samples, gene_mode, 100 * prop_genes,
    )

    results = Parallel(n_jobs=n_jobs, batch_size=8, verbose=0)(
        delayed(_one_resample)(
            int(s), X, k_values, prop_samples, prop_genes,
            sample_mode, gene_mode, base, metric, linkage_method,
        )
        for s in seeds
    )

    M = {k: np.zeros((n, n)) for k in k_values}
    I = np.zeros((n, n))

    for idx_s, labels in results:
        I[np.ix_(idx_s, idx_s)] += 1.0
        for k, lab in labels.items():
            # connectivité = produit de la matrice one-hot par sa transposée
            onehot = np.zeros((idx_s.size, lab.max() + 1))
            onehot[np.arange(idx_s.size), lab] = 1.0
            M[k][np.ix_(idx_s, idx_s)] += onehot @ onehot.T

    denom = np.maximum(I, 1e-12)
    consensus = {}
    for k in k_values:
        C = M[k] / denom
        C[I == 0] = 0.0          # paires jamais tirées ensemble (rare si B grand)
        C = (C + C.T) / 2.0
        np.fill_diagonal(C, 1.0)
        consensus[k] = C

    never = int((I == 0).sum() - 0)
    if never:
        logger.warning("%d paires jamais co-tirées : augmente n_resamples", never // 2)

    return ConsensusResult(
        consensus=consensus,
        indicator=I,
        sample_names=np.asarray(sample_names),
        params=dict(
            n_resamples=n_resamples, prop_samples=prop_samples,
            prop_genes=prop_genes, sample_mode=sample_mode, gene_mode=gene_mode,
            base=base, metric=metric, linkage_method=linkage_method,
            random_state=random_state, k_values=k_values,
        ),
    )
