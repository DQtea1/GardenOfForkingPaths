"""t-SNE et UMAP **à partir de la matrice de distance consensus**.

Point clé : on ne réembedde pas l'expression brute, mais D = 1 - C.
La distance consensus encode « à quelle fréquence deux tumeurs ont été
co-clusterisées sur B rééchantillonnages » ; l'embedding montre donc la
structure stable, pas la géométrie bruitée de l'espace des gènes.

Deux conséquences à garder en tête pour l'interprétation :
  - D est bornée dans [0, 1] et fortement non euclidienne, avec beaucoup
    d'ex aequo (0 et 1) ; les distances *entre* clusters bien séparés y sont
    toutes égales à 1, donc les positions relatives des clusters sur le plan
    n'ont aucune signification. Seule la structure locale est informative.
  - l'embedding est calculé sur la même donnée qui a servi au clustering :
    c'est une **visualisation**, pas une validation. Pour valider, il faut un
    embedding indépendant (sur l'expression) ou une cohorte externe.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def _check_distance(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, None)


def tsne_from_distance(
    D: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1500,
    learning_rate: str | float = "auto",
    early_exaggeration: float = 12.0,
    random_state: int = 0,
) -> np.ndarray:
    """t-SNE 2D ou 3D sur distance précalculée.

    `init="random"` est obligatoire avec `metric="precomputed"` (une PCA
    n'est pas définie sans coordonnées). Perplexité < n_samples / 3.
    `n_components` doit valoir 2 ou 3 (limite de l'algorithme barnes-hut).
    """
    D = _check_distance(D)
    n = D.shape[0]
    perplexity = float(min(perplexity, max(5.0, (n - 1) / 3.0)))
    kwargs = dict(
        n_components=n_components,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        random_state=random_state,
    )
    try:  # sklearn >= 1.5
        emb = TSNE(max_iter=n_iter, **kwargs).fit_transform(D)
    except TypeError:  # sklearn < 1.5
        emb = TSNE(n_iter=n_iter, **kwargs).fit_transform(D)
    logger.info("t-SNE calculé (perplexity=%.1f, %dD)", perplexity, n_components)
    return emb


def umap_from_distance(
    D: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> np.ndarray:
    """UMAP 2D ou 3D sur distance précalculée. Nécessite `umap-learn`."""
    try:
        import umap
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "umap-learn absent : `pip install umap-learn` "
            "(ou lance le pipeline avec --no-umap)"
        ) from exc

    D = _check_distance(D)
    n_neighbors = int(min(n_neighbors, max(2, D.shape[0] - 1)))
    reducer = umap.UMAP(
        n_components=n_components,
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    emb = reducer.fit_transform(D)
    logger.info("UMAP calculé (n_neighbors=%d, min_dist=%.2f, %dD)",
                n_neighbors, min_dist, n_components)
    return np.asarray(emb)


def embeddings_table(
    D: np.ndarray,
    sample_names: np.ndarray,
    labels: np.ndarray,
    run_umap: bool = True,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """DataFrame prêt pour le plot : coordonnées t-SNE (+ UMAP) et cluster.

    Produit `n_components` colonnes par méthode : `tsne1..tsneN` (et
    `umap1..umapN`), avec N = 2 ou 3. t-SNE et UMAP sont deux calculs
    indépendants : si `run_umap` et `n_jobs != 1`, ils sont lancés en parallèle
    (joblib) plutôt que l'un après l'autre.
    """
    out = pd.DataFrame({"sample": sample_names, "cluster": labels})

    if run_umap and n_jobs != 1:
        try:
            import umap  # noqa: F401  (juste pour échouer tôt si absent)
            have_umap = True
        except ImportError as exc:
            logger.warning("UMAP ignoré : %s", exc)
            have_umap = False
        if have_umap:
            ts, um = Parallel(n_jobs=2)(
                delayed(fn)(D, n_components=n_components, random_state=random_state, **kw)
                for fn, kw in (
                    (tsne_from_distance, dict(perplexity=perplexity)),
                    (umap_from_distance, dict(n_neighbors=n_neighbors, min_dist=min_dist)),
                )
            )
            for j in range(n_components):
                out[f"tsne{j + 1}"] = ts[:, j]
            for j in range(n_components):
                out[f"umap{j + 1}"] = um[:, j]
            return out

    ts = tsne_from_distance(D, n_components=n_components, perplexity=perplexity,
                            random_state=random_state)
    for j in range(n_components):
        out[f"tsne{j + 1}"] = ts[:, j]
    if run_umap:
        try:
            um = umap_from_distance(
                D, n_components=n_components, n_neighbors=n_neighbors,
                min_dist=min_dist, random_state=random_state,
            )
            for j in range(n_components):
                out[f"umap{j + 1}"] = um[:, j]
        except ImportError as exc:
            logger.warning("UMAP ignoré : %s", exc)
    return out


def stability_of_embedding(
    D: np.ndarray, seeds: tuple[int, ...] = (0, 1, 2), method: str = "tsne", **kwargs
) -> list[np.ndarray]:
    """Recalcule l'embedding avec plusieurs graines.

    À faire systématiquement avant de raconter une histoire sur la forme du
    nuage : si la topologie change d'une graine à l'autre, elle n'est pas réelle.
    """
    fn = tsne_from_distance if method == "tsne" else umap_from_distance
    return [fn(D, random_state=s, **kwargs) for s in seeds]
