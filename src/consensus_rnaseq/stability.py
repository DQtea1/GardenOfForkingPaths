"""Stabilité des branches par bootstrap Jaccard (façon pvclust / Hennig).

Le consensus clustering donne un arbre (CAH sur la distance consensus), mais ne
dit pas quelles **branches** de cet arbre sont reproductibles. Ce module répond
à cette question par un bootstrap des gènes, comme pvclust :

  1. on construit B arbres bootstrap : à chaque tirage on rééchantillonne les
     gènes *avec remise* (les tumeurs, elles, restent toutes présentes — c'est
     ce qui garantit que tous les arbres partitionnent le même ensemble de
     tumeurs et rend le Jaccard entre branches exact) ;
  2. une « branche » = l'ensemble des tumeurs sous un nœud interne du
     dendrogramme. Pour chaque branche de l'arbre consensus et chaque branche
     d'un arbre bootstrap on calcule le Jaccard J = |A ∩ B| / |A ∪ B| ;
  3. score d'une branche de l'arbre consensus = moyenne, sur les B arbres
     bootstrap, du Jaccard **maximal** avec une branche de cet arbre bootstrap.

Lecture (seuils de Hennig 2007) : < 0,5 branche instable (artefact probable) ;
0,6–0,75 motif présent mais incertain ; > 0,75 branche stable ; > 0,85 très
stable. Une branche = un cluster candidat : ce score dit à quel point ce cluster
survit au rééchantillonnage des features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from .consensus import _draw, pairwise_distance

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Énumération des branches d'un dendrogramme
# --------------------------------------------------------------------------
def branch_members(
    Z: np.ndarray, min_size: int = 2, include_root: bool = False
) -> dict[int, np.ndarray]:
    """{id du nœud interne -> indices des tumeurs sous ce nœud}.

    On ne garde que les branches non triviales : taille >= `min_size` et, sauf
    `include_root`, on exclut la racine (tout l'échantillon), dont le Jaccard
    vaut trivialement 1 dans tout arbre bootstrap.
    """
    n = Z.shape[0] + 1
    _, nodelist = to_tree(Z, rd=True)
    out: dict[int, np.ndarray] = {}
    for node in nodelist:
        if node.is_leaf():
            continue
        members = np.asarray(node.pre_order(), dtype=np.intp)
        if members.size < min_size or (not include_root and members.size >= n):
            continue
        out[node.id] = members
    return out


def _bool_matrix(members: dict[int, np.ndarray], n: int) -> tuple[np.ndarray, list[int]]:
    """Matrice d'appartenance `(n_branches x n)` (0/1) et liste des ids alignée."""
    ids = list(members)
    M = np.zeros((len(ids), n), dtype=np.float64)
    for r, nid in enumerate(ids):
        M[r, members[nid]] = 1.0
    return M, ids


# --------------------------------------------------------------------------
# Un arbre bootstrap -> Jaccard max par branche consensus
# --------------------------------------------------------------------------
def _bootstrap_max_jaccard(
    seed: int, X: np.ndarray, Cons: np.ndarray, cons_sizes: np.ndarray,
    prop_genes: float, gene_mode: str, metric: str, linkage_method: str,
    min_size: int,
) -> np.ndarray:
    """Renvoie, pour chaque branche consensus, le Jaccard max avec une branche
    de cet arbre bootstrap (vecteur aligné sur les lignes de `Cons`)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape

    idx_g = _draw(p, prop_genes, gene_mode, rng)
    Xb = X[:, idx_g]
    D = pairwise_distance(Xb, metric)
    Z = linkage(squareform(D, checks=False), method=linkage_method)

    members = branch_members(Z, min_size=min_size, include_root=True)
    if not members:
        return np.zeros(Cons.shape[0])

    Boot, _ = _bool_matrix(members, n)          # (n_boot_branches x n)
    inter = Cons @ Boot.T                        # (n_cons x n_boot)
    boot_sizes = Boot.sum(axis=1)
    union = cons_sizes[:, None] + boot_sizes[None, :] - inter
    jaccard = inter / np.maximum(union, 1e-12)
    return jaccard.max(axis=1)


# --------------------------------------------------------------------------
# Résultat + API
# --------------------------------------------------------------------------
@dataclass
class BranchStability:
    """Stabilité Jaccard de chaque branche de l'arbre consensus."""

    linkage: np.ndarray               # CAH sur la distance consensus (pour le tracé)
    node_ids: list[int]               # id de nœud interne de chaque branche
    members: dict[int, np.ndarray]    # id -> indices des tumeurs de la branche
    sizes: np.ndarray                 # taille de chaque branche (aligné sur node_ids)
    stability: np.ndarray             # score de stabilité (aligné sur node_ids)

    def to_frame(self, sample_names: np.ndarray | None = None,
                 final_labels: np.ndarray | None = None) -> pd.DataFrame:
        """Table triée : une ligne par branche, avec sa stabilité et ses membres.

        `final_labels` (partition finale à k) permet de marquer les branches qui
        correspondent exactement à un cluster retenu (`is_final_cluster`).
        """
        names = np.asarray(sample_names) if sample_names is not None else None
        final_sets = set()
        if final_labels is not None:
            final_labels = np.asarray(final_labels)
            final_sets = {
                frozenset(np.flatnonzero(final_labels == c).tolist())
                for c in np.unique(final_labels)
            }
        rows = []
        for i, nid in enumerate(self.node_ids):
            mem = self.members[nid]
            member_names = names[mem] if names is not None else mem.astype(str)
            rows.append({
                "branch_id": int(nid),
                "size": int(self.sizes[i]),
                "stability": float(self.stability[i]),
                "is_final_cluster": frozenset(mem.tolist()) in final_sets,
                "members": "|".join(map(str, member_names)),
            })
        return (
            pd.DataFrame(rows)
            .sort_values(["size", "stability"], ascending=[False, False])
            .reset_index(drop=True)
        )


def branch_stability(
    X: np.ndarray,
    consensus_distance: np.ndarray,
    n_resamples: int = 1000,
    prop_genes: float = 1.0,
    gene_mode: str = "bootstrap",
    metric: str = "pearson",
    linkage_method: str = "average",
    min_size: int = 2,
    random_state: int = 0,
    n_jobs: int = -1,
) -> BranchStability:
    """Stabilité Jaccard des branches de l'arbre consensus par bootstrap des gènes.

    Parameters
    ----------
    X : matrice `samples x genes` prétraitée (celle qui a servi au consensus).
    consensus_distance : distance consensus `n x n` du k retenu (`result.distance(k)`).
    n_resamples : B, nombre d'arbres bootstrap (réutilise `--n-resamples`).
    prop_genes, gene_mode : par défaut bootstrap classique (tirage de p gènes
        avec remise) ; les tumeurs restent toutes présentes.
    metric, linkage_method : mêmes réglages que l'arbre consensus, pour comparer
        des arbres construits de la même façon.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]

    Zc = linkage(squareform(consensus_distance, checks=False), method=linkage_method)
    cons_members = branch_members(Zc, min_size=min_size, include_root=False)
    Cons, node_ids = _bool_matrix(cons_members, n)
    cons_sizes = Cons.sum(axis=1)

    logger.info(
        "Stabilité Jaccard : %d branches consensus, B=%d arbres bootstrap "
        "(gènes %s, metric=%s, linkage=%s)",
        len(node_ids), n_resamples, gene_mode, metric, linkage_method,
    )

    seeds = np.random.SeedSequence(random_state).generate_state(n_resamples)
    vecs = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(_bootstrap_max_jaccard)(
            int(s), X, Cons, cons_sizes, prop_genes, gene_mode,
            metric, linkage_method, min_size,
        )
        for s in seeds
    )
    stability = np.vstack(vecs).mean(axis=0)

    return BranchStability(
        linkage=Zc,
        node_ids=node_ids,
        members={nid: cons_members[nid] for nid in node_ids},
        sizes=cons_sizes.astype(int),
        stability=stability,
    )
