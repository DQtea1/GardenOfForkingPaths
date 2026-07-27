"""ICA stabilisée et choix de la *Most Stable Transcriptome Dimension*.

Ce module constitue une branche indépendante du consensus clustering. Il reçoit
la matrice d'expression déjà prétraitée dans l'orientation ``échantillons x
gènes`` et utilise :mod:`stabilized-ica` pour :

* répéter l'ICA pour plusieurs nombres de composantes ;
* classer les composantes de chaque décomposition par stabilité ;
* estimer la MSTD (``Most/Maximally Stable Transcriptome Dimension``) ;
* conserver la MSTD, ses deux voisines testées et la meilleure stabilité moyenne ;
* écrire les tables et les cinq diagnostics requis par le rapport.

``sica.base.MSTD`` dessine deux graphiques mais ne retourne ni les profils ni
la dimension sélectionnée. Le balayage est donc réalisé ici avec l'API publique
``StabilizedICA`` afin de conserver toutes les sorties et d'appliquer une règle
de sélection MSTD explicite et reproductible.

La sélection suit d'abord le critère publié par Kairov *et al.* (2017) :
deux droites sont ajustées aux profils de stabilité (k-lines), et leur
intersection donne la MSTD. Si cette intersection est indéfinie ou hors de la
plage testée, un coude déterministe sur le profil de stabilité moyenne est
utilisé. En dernier recours, la plus grande dimension dont la stabilité moyenne
atteint ``mean_stability_threshold`` est retenue.
"""

from __future__ import annotations

import json
import inspect
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import matplotlib

# Le pipeline est non interactif ; définir le backend avant pyplot rend aussi
# le module utilisable seul sur un serveur sans écran.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MSTDSelection:
    """Détail de la règle ayant sélectionné la MSTD."""

    mstd: int
    raw_intersection: float | None
    method: str
    fallback_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ICADecomposition:
    """Une ICA stabilisée sauvegardée pour une dimension donnée.

    ``metagenes`` est orientée ``composantes x gènes`` et ``metasamples``
    ``échantillons x composantes``. Les composantes sont déjà rangées par
    stabilité décroissante par ``stabilized-ica``.
    """

    n_components: int
    metagenes: pd.DataFrame
    metasamples: pd.DataFrame
    stability: pd.DataFrame
    output_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class ICAResult:
    """Résultats complets de la branche ICA.

    ``decompositions`` contient les dimensions retenues pour les branches
    aval : la MSTD, la dimension testée immédiatement inférieure, celle
    immédiatement supérieure, puis la meilleure stabilité moyenne. Une même
    dimension peut remplir plusieurs rôles ; elle est alors persistée une seule
    fois et ses rôles restent explicitement tracés dans ``dimension_roles``.
    """

    mstd: int
    selection: MSTDSelection
    scan_summary: pd.DataFrame
    stability_profiles: pd.DataFrame
    decompositions: dict[int, ICADecomposition]
    mstd_diagnostic: ICADecomposition
    selected_dimensions: tuple[int, ...]
    dimension_roles: dict[int, tuple[str, ...]]
    persisted_dimensions: tuple[int, ...]
    params: dict[str, Any]
    output_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def mstd_decomposition(self) -> ICADecomposition:
        """Décomposition MSTD utilisée par les diagnostics de qualité."""
        return self.mstd_diagnostic

    @property
    def top_stable_dimensions(self) -> tuple[int, ...]:
        """Alias de compatibilité pour les consommateurs antérieurs.

        Les dimensions persistées ne sont plus un classement de stabilité
        moyenne : utiliser ``selected_dimensions`` et ``dimension_roles`` dans
        tout nouveau code.
        """
        return self.selected_dimensions


def _require_stabilized_ica():
    """Import différé : le reste du pipeline reste utilisable sans ICA."""
    try:
        import sica.base as sica_base
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "La branche ICA requiert le package `stabilized-ica`. "
            "Installe-le avec : pip install stabilized-ica"
        ) from exc
    _patch_agglomerative_clustering_compat(sica_base)
    return sica_base.StabilizedICA


def _patch_agglomerative_clustering_compat(sica_base) -> None:
    """Compatibilise stabilized-ica 2.0.0 avec les sklearn récents.

    La roue PyPI appelle ``AgglomerativeClustering(affinity=...)`` tandis que
    scikit-learn >= 1.4 attend ``metric``. Le shim est local au module ``sica``
    et traduit seulement ce mot-clé ; il gère aussi le sens inverse si une
    future version de stabilized-ica utilise ``metric`` avec un vieux sklearn.
    """
    marker = "_consensus_cluster_bulk_agglomerative_compat"
    if getattr(sica_base, marker, False):
        return

    original = sica_base.AgglomerativeClustering
    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError):  # pragma: no cover - API sklearn inhabituelle
        return

    accepts_affinity = "affinity" in parameters
    accepts_metric = "metric" in parameters
    if accepts_affinity == accepts_metric:
        # Deux mots-clés (version de transition) ou aucun : pas de traduction.
        setattr(sica_base, marker, True)
        return

    def _compatible_agglomerative(*args, **kwargs):
        kwargs = dict(kwargs)
        if accepts_metric and not accepts_affinity and "affinity" in kwargs:
            kwargs["metric"] = kwargs.pop("affinity")
        elif accepts_affinity and not accepts_metric and "metric" in kwargs:
            kwargs["affinity"] = kwargs.pop("metric")
        return original(*args, **kwargs)

    _compatible_agglomerative.__name__ = getattr(
        original, "__name__", "AgglomerativeClustering"
    )
    sica_base.AgglomerativeClustering = _compatible_agglomerative
    setattr(sica_base, marker, True)
    old_name, new_name = (
        ("affinity", "metric") if accepts_metric else ("metric", "affinity")
    )
    logger.info(
        "Compatibilité stabilized-ica/sklearn active : %s est traduit vers %s.",
        old_name,
        new_name,
    )


@contextmanager
def _temporary_numpy_seed(seed: int) -> Iterator[None]:
    """Isole la graine globale utilisée implicitement par FastICA.

    ``StabilizedICA`` 2.0.0 n'expose pas de paramètre ``random_state``. Avec
    ``n_jobs=1``, fixer temporairement la graine NumPy rend donc chaque fit
    reproductible sans modifier l'état aléatoire du reste du pipeline.
    """
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        yield
    finally:
        np.random.set_state(state)


def _as_expression_matrix(
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, pd.Index, pd.Index]:
    """Valide et matérialise une matrice ``samples x genes`` finie."""
    if isinstance(X, pd.DataFrame):
        sample_names = pd.Index(X.index)
        gene_names = pd.Index(X.columns)
        try:
            values = X.to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("La matrice ICA doit être entièrement numérique.") from exc
    else:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("X doit être une matrice 2D échantillons x gènes.")
        sample_names = pd.Index([f"sample_{i + 1}" for i in range(values.shape[0])])
        gene_names = pd.Index([f"gene_{j + 1}" for j in range(values.shape[1])])

    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("X doit contenir au moins un échantillon et un gène.")
    if sample_names.has_duplicates:
        raise ValueError("Les identifiants d'échantillons doivent être uniques pour l'ICA.")
    if gene_names.has_duplicates:
        raise ValueError("Les identifiants de gènes doivent être uniques pour l'ICA.")
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"La matrice ICA contient {n_bad} valeur(s) non finie(s).")
    if values.shape[0] < 3:
        raise ValueError("L'ICA requiert au moins trois échantillons.")
    if np.all(np.nanstd(values, axis=0) == 0):
        raise ValueError("Tous les gènes sont constants : l'ICA est impossible.")
    return np.ascontiguousarray(values, dtype=np.float64), sample_names, gene_names


def _candidate_dimensions(
    min_components: int,
    max_components: int,
    step: int,
    n_samples: int,
    n_genes: int,
) -> tuple[int, ...]:
    """Construit la grille en respectant la limite de rang après centrage."""
    if isinstance(min_components, bool) or isinstance(max_components, bool):
        raise ValueError("min_components et max_components doivent être des entiers.")
    min_components = int(min_components)
    max_components = int(max_components)
    step = int(step)
    if min_components < 2:
        raise ValueError("min_components doit être >= 2 pour une analyse MSTD.")
    if max_components < min_components:
        raise ValueError("max_components doit être >= min_components.")
    if step < 1:
        raise ValueError("step doit être un entier strictement positif.")

    # Après centrage, le rang ne peut pas dépasser n_samples - 1. La borne est
    # aussi limitée par le nombre de gènes ; elle évite une erreur opaque de SVD.
    upper = min(max_components, n_samples - 1, n_genes)
    if upper < min_components:
        raise ValueError(
            "La plage ICA demandée est impossible : nombre maximal admissible "
            f"= {upper} (échantillons={n_samples}, gènes={n_genes})."
        )

    dimensions = list(range(min_components, upper + 1, step))
    # Inclure explicitement la borne supérieure rend l'intervalle [min, max]
    # intuitif même si `step` ne le divise pas exactement.
    if dimensions[-1] != upper:
        dimensions.append(upper)
    return tuple(dimensions)


@dataclass(frozen=True)
class _MSTDWhitening:
    """Whitening unique du scan officiel MSTD.

    ``X_w`` a l'orientation employée par :func:`sica.base.MSTD`, soit
    ``gènes x max_components``. Pour une dimension ``i``, le fit ICA reçoit
    donc ``X_w[:, :i].T`` (``i x gènes``) avec ``whiten=False``.
    """

    X_w: np.ndarray
    mean: np.ndarray | None
    max_components: int


def _prepare_mstd_whitening(
    matrix: np.ndarray,
    max_components: int,
    *,
    seed: int,
) -> _MSTDWhitening:
    """Applique exactement le whitening préalable de ``sica.base.MSTD``.

    La fonction officielle transpose d'abord l'expression ``samples x genes``
    puis exécute une seule PCA blanchie au maximum testé. Les dimensions plus
    petites réutilisent les premières composantes de cette même PCA, plutôt que
    de recalculer un whitening par valeur de ``n_components``.
    """
    try:
        from sica._whitening import whitening
    except ImportError as exc:  # pragma: no cover - dépend du package installé
        raise ImportError(
            "La version installée de `stabilized-ica` ne fournit pas son "
            "whitening MSTD interne. Réinstallez `stabilized-ica` depuis PyPI."
        ) from exc

    with _temporary_numpy_seed(seed):
        X_w, mean = whitening(
            matrix.T,
            n_components=int(max_components),
            svd_solver="auto",
            chunked=False,
            chunk_size=None,
            zero_center=True,
        )
    X_w = np.ascontiguousarray(X_w, dtype=np.float64)
    if X_w.shape != (matrix.shape[1], int(max_components)):
        raise RuntimeError(
            "Le whitening MSTD a retourné une matrice de forme inattendue "
            f"{X_w.shape}; attendu {(matrix.shape[1], int(max_components))}."
        )
    if not np.isfinite(X_w).all():
        raise RuntimeError("Le whitening MSTD a retourné des valeurs non finies.")

    if mean is not None:
        mean = np.asarray(mean, dtype=np.float64)
        if mean.ndim != 1 or mean.shape[0] != matrix.shape[0]:
            raise RuntimeError(
                "La moyenne du whitening MSTD a une forme inattendue : "
                f"{mean.shape}; attendu ({matrix.shape[0]},)."
            )
        if not np.isfinite(mean).all():
            raise RuntimeError("La moyenne du whitening MSTD contient des valeurs non finies.")
    return _MSTDWhitening(X_w=X_w, mean=mean, max_components=int(max_components))


def _fit_stabilized_ica(
    StabilizedICA,
    fit_matrix: np.ndarray,
    n_components: int,
    n_runs: int,
    *,
    algorithm: str,
    fun: str,
    resampling: str | None,
    max_iter: int,
    n_jobs: int,
    seed: int,
    whiten: bool = True,
):
    """Ajuste une ICA stabilisée avec une graine déterministe.

    ``whiten=False`` est réservé au chemin MSTD pré-blanchi : ``fit_matrix``
    est alors ``n_components x gènes`` et ne peut pas être combiné à un
    bootstrap de stabilized-ica.
    """
    if not whiten and resampling is not None:
        raise ValueError(
            "Un fit ICA pré-blanchi (whiten=False) ne prend pas en charge "
            "resampling ; utiliser le repli par dimension."
        )
    with _temporary_numpy_seed(seed):
        model = StabilizedICA(
            n_components=int(n_components),
            n_runs=int(n_runs),
            resampling=resampling,
            algorithm=algorithm,
            fun=fun,
            whiten=bool(whiten),
            max_iter=int(max_iter),
            plot=False,
            normalize=True,
            reorientation=True,
            pca_solver="auto",
            n_jobs=int(n_jobs),
            verbose=0,
        )
        model.fit(fit_matrix)

    indices = np.asarray(getattr(model, "stability_indexes_", None), dtype=float)
    sources = np.asarray(getattr(model, "S_", None), dtype=float)
    if indices.ndim != 1 or indices.size != n_components:
        raise RuntimeError("stabilized-ica n'a pas retourné les indices de stabilité attendus.")
    if sources.shape != (n_components, fit_matrix.shape[1]):
        raise RuntimeError("stabilized-ica n'a pas retourné les métagènes attendus.")
    if not (np.isfinite(indices).all() and np.isfinite(sources).all()):
        raise RuntimeError("stabilized-ica a retourné des valeurs non finies.")
    return model


def _scan_dimensions(
    StabilizedICA,
    matrix: np.ndarray,
    dimensions: tuple[int, ...],
    *,
    n_runs: int,
    algorithm: str,
    fun: str,
    resampling: str | None,
    max_iter: int,
    n_jobs: int,
    seeds: dict[int, int],
    mean_stability_threshold: float,
    mstd_whitening: _MSTDWhitening | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lance le scan et ne garde en mémoire que les statistiques légères.

    Sans resampling, ``mstd_whitening`` reproduit le chemin officiel MSTD :
    une PCA blanchie unique au maximum, puis un fit pré-blanchi pour chaque
    dimension. Avec bootstrap/fast_bootstrap, il est volontairement absent car
    stabilized-ica impose son propre whitening à chaque rééchantillonnage.
    """
    summary_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for pos, n_components in enumerate(dimensions, start=1):
        logger.info(
            "ICA MSTD : dimension %d/%d (n_components=%d, n_runs=%d)…",
            pos,
            len(dimensions),
            n_components,
            n_runs,
        )
        try:
            if mstd_whitening is None:
                fit_matrix = matrix
                whiten = True
                whitening_strategy = "per_dimension"
            else:
                fit_matrix = mstd_whitening.X_w[:, :n_components].T
                whiten = False
                whitening_strategy = "single_max_pca"
            model = _fit_stabilized_ica(
                StabilizedICA,
                fit_matrix,
                n_components,
                n_runs,
                algorithm=algorithm,
                fun=fun,
                resampling=resampling,
                max_iter=max_iter,
                n_jobs=n_jobs,
                seed=seeds[n_components],
                whiten=whiten,
            )
            stability = np.asarray(model.stability_indexes_, dtype=float)
            for rank, value in enumerate(stability, start=1):
                profile_rows.append(
                    {
                        "n_components": int(n_components),
                        "component_rank": int(rank),
                        "stability_index": float(value),
                    }
                )
            summary_rows.append(
                {
                    "n_components": int(n_components),
                    "seed": int(seeds[n_components]),
                    "status": "ok",
                    "whitening_strategy": whitening_strategy,
                    "mean_stability": float(np.mean(stability)),
                    "median_stability": float(np.median(stability)),
                    "min_stability": float(np.min(stability)),
                    "max_stability": float(np.max(stability)),
                    "n_components_above_threshold": int(
                        np.sum(stability >= mean_stability_threshold)
                    ),
                    "error": "",
                }
            )
        except Exception as exc:
            # Une dimension peut échouer par difficulté numérique sans invalider
            # tout le scan. L'erreur est exportée afin qu'elle reste visible.
            message = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            logger.warning("ICA MSTD : n_components=%d ignoré (%s)", n_components, message)
            summary_rows.append(
                {
                    "n_components": int(n_components),
                    "seed": int(seeds[n_components]),
                    "status": "failed",
                    "whitening_strategy": (
                        "single_max_pca" if mstd_whitening is not None else "per_dimension"
                    ),
                    "mean_stability": np.nan,
                    "median_stability": np.nan,
                    "min_stability": np.nan,
                    "max_stability": np.nan,
                    "n_components_above_threshold": np.nan,
                    "error": message,
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values("n_components").reset_index(drop=True)
    profiles = pd.DataFrame(
        profile_rows,
        columns=["n_components", "component_rank", "stability_index"],
    )
    return summary, profiles


def _fit_output_dimension(
    StabilizedICA,
    matrix: np.ndarray,
    dimension: int,
    n_runs: int,
    *,
    algorithm: str,
    fun: str,
    resampling: str | None,
    max_iter: int,
    n_jobs: int,
    seed: int,
    mstd_whitening: _MSTDWhitening | None,
) -> tuple[Any, np.ndarray | None]:
    """Refait une dimension retenue avec le même chemin que son scan.

    Le scan MSTD sans rééchantillonnage utilise une unique PCA blanchie. Les
    sorties retenues doivent impérativement la réutiliser : un whitening par
    dimension modifierait les métagènes et ne correspondrait plus aux indices
    de stabilité évalués. La moyenne de cette PCA est retournée pour projeter
    ensuite les métasamples dans l'espace d'expression d'origine.
    """
    if mstd_whitening is None:
        model = _fit_stabilized_ica(
            StabilizedICA,
            matrix,
            dimension,
            n_runs,
            algorithm=algorithm,
            fun=fun,
            resampling=resampling,
            max_iter=max_iter,
            n_jobs=n_jobs,
            seed=seed,
            whiten=True,
        )
        return model, None

    fit_matrix = mstd_whitening.X_w[:, :dimension].T
    model = _fit_stabilized_ica(
        StabilizedICA,
        fit_matrix,
        dimension,
        n_runs,
        algorithm=algorithm,
        fun=fun,
        resampling=None,
        max_iter=max_iter,
        n_jobs=n_jobs,
        seed=seed,
        whiten=False,
    )
    return model, mstd_whitening.mean


def _fit_tls_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Droite des moindres carrés orthogonaux (centre, direction unitaire)."""
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    centre = np.mean(points, axis=0)
    centered = points - centre
    if np.allclose(centered, 0.0):
        return None
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = np.asarray(vt[0], dtype=float)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return None
    return centre, direction / norm


def _distance_to_line(points: np.ndarray, line: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Distance orthogonale de points 2D à une droite 2D."""
    centre, direction = line
    delta = points - centre
    return np.abs(delta[:, 0] * direction[1] - delta[:, 1] * direction[0])


def _line_slope(direction: np.ndarray) -> float | None:
    if abs(direction[0]) <= np.finfo(float).eps:
        return None
    return float(direction[1] / direction[0])


def _k_lines_intersection(profiles: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    """Estime l'intersection des deux lignes du critère MSTD publié.

    Le papier décrit un k-lines avec des droites initialisées comme les axes.
    Les abscisses (rangs de composantes) sont normalisées dans [0, 1] afin que
    cette initialisation ne privilégie pas mécaniquement l'axe horizontal.
    L'algorithme est entièrement déterministe : pas de redémarrage aléatoire,
    et les égalités sont affectées à la première droite.
    """
    if profiles.empty or profiles["component_rank"].nunique() < 2:
        return None, {"reason": "profils insuffisants pour ajuster deux droites"}

    max_rank = float(profiles["component_rank"].max())
    points = np.column_stack(
        [
            profiles["component_rank"].to_numpy(dtype=float) / max_rank,
            profiles["stability_index"].to_numpy(dtype=float),
        ]
    )
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 4:
        return None, {"reason": "points finis insuffisants pour k-lines"}

    # Droites initiales : axe des abscisses puis axe des ordonnées.
    lines: list[tuple[np.ndarray, np.ndarray]] = [
        (np.array([0.5, 0.0]), np.array([1.0, 0.0])),
        (np.array([0.0, 0.5]), np.array([0.0, 1.0])),
    ]
    labels: np.ndarray | None = None
    converged = False
    max_iter = 100

    for iteration in range(1, max_iter + 1):
        distances = np.column_stack([_distance_to_line(points, line) for line in lines])
        new_labels = (distances[:, 1] < distances[:, 0]).astype(int)
        counts = [int(np.sum(new_labels == group)) for group in (0, 1)]
        if min(counts) < 2:
            return None, {
                "reason": "k-lines a produit un groupe de moins de deux points",
                "cluster_sizes": counts,
            }

        updated: list[tuple[np.ndarray, np.ndarray]] = []
        for group in (0, 1):
            line = _fit_tls_line(points[new_labels == group])
            if line is None:
                return None, {"reason": "droite dégénérée pendant k-lines"}
            updated.append(line)

        if labels is not None and np.array_equal(new_labels, labels):
            lines = updated
            labels = new_labels
            converged = True
            break
        lines = updated
        labels = new_labels
    else:
        iteration = max_iter

    direction0, direction1 = lines[0][1], lines[1][1]
    system = np.column_stack([direction0, -direction1])
    determinant = float(np.linalg.det(system))
    details: dict[str, Any] = {
        "converged": converged,
        "iterations": int(iteration),
        "cluster_sizes": [int(np.sum(labels == group)) for group in (0, 1)],
        "line_0_slope": _line_slope(direction0),
        "line_1_slope": _line_slope(direction1),
    }
    if abs(determinant) <= 1e-8:
        details["reason"] = "droites k-lines quasi parallèles"
        return None, details

    try:
        parameters = np.linalg.solve(system, lines[1][0] - lines[0][0])
    except np.linalg.LinAlgError:
        details["reason"] = "intersection k-lines non résoluble"
        return None, details
    intersection = lines[0][0] + parameters[0] * direction0
    raw_dimension = float(intersection[0] * max_rank)
    details["intersection_stability"] = float(intersection[1])
    details["raw_dimension"] = raw_dimension
    return raw_dimension, details


def _segmented_profile_selection(summary: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    """Coude déterministe à deux segments sur la stabilité moyenne."""
    valid = summary.loc[summary["status"] == "ok", ["n_components", "mean_stability"]]
    valid = valid.dropna().sort_values("n_components")
    if len(valid) < 4:
        return None, {"reason": "moins de quatre dimensions réussies"}

    x = valid["n_components"].to_numpy(dtype=float)
    y = valid["mean_stability"].to_numpy(dtype=float)
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None

    # Chaque côté doit disposer d'au moins deux dimensions pour ajuster une
    # droite ; les égalités de SSE choisissent le plus petit coude.
    for split in range(1, len(x) - 1):
        left_x, left_y = x[: split + 1], y[: split + 1]
        right_x, right_y = x[split:], y[split:]
        try:
            left_coef = np.polyfit(left_x, left_y, deg=1)
            right_coef = np.polyfit(right_x, right_y, deg=1)
        except np.linalg.LinAlgError:
            continue
        sse = float(
            np.sum((left_y - np.polyval(left_coef, left_x)) ** 2)
            + np.sum((right_y - np.polyval(right_coef, right_x)) ** 2)
        )
        candidate = (sse, int(x[split]), left_coef, right_coef)
        if best is None or candidate[0] < best[0] - 1e-12 or (
            np.isclose(candidate[0], best[0]) and candidate[1] < best[1]
        ):
            best = candidate

    if best is None:
        return None, {"reason": "ajustement segmenté impossible"}

    sse, breakpoint, left_coef, right_coef = best
    slope_delta = float(left_coef[0] - right_coef[0])
    raw_intersection: float | None
    if abs(slope_delta) <= 1e-12:
        raw_intersection = None
    else:
        raw_intersection = float((right_coef[1] - left_coef[1]) / slope_delta)
    details = {
        "sse": sse,
        "breakpoint": breakpoint,
        "left_slope": float(left_coef[0]),
        "right_slope": float(right_coef[0]),
        "raw_intersection": raw_intersection,
    }
    return raw_intersection, details


def _nearest_dimension(value: float, candidates: np.ndarray) -> int:
    """Dimension testée la plus proche (égalité résolue vers la plus petite)."""
    return int(min(candidates, key=lambda d: (abs(float(d) - value), int(d))))


def _select_mstd(
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    mean_stability_threshold: float,
) -> MSTDSelection:
    """Sélectionne la MSTD avec deux repli explicites et auditables."""
    valid = summary.loc[summary["status"] == "ok"].copy()
    valid = valid.dropna(subset=["mean_stability"]).sort_values("n_components")
    if valid.empty:
        errors = summary.loc[summary["status"] != "ok", "error"].dropna().tolist()
        detail = "; ".join(map(str, errors[:3]))
        raise RuntimeError(
            "Aucune décomposition ICA de la grille n'a abouti. "
            f"Premières erreurs : {detail or 'non renseignées'}"
        )

    candidates = valid["n_components"].to_numpy(dtype=int)
    minimum, maximum = int(candidates.min()), int(candidates.max())

    raw, details = _k_lines_intersection(profiles)
    if raw is not None and np.isfinite(raw) and minimum <= raw <= maximum:
        return MSTDSelection(
            mstd=_nearest_dimension(float(raw), candidates),
            raw_intersection=float(raw),
            method="k_lines_profiles",
            details=details,
        )

    first_reason = str(details.get("reason", "intersection k-lines hors plage"))
    if raw is not None and not (minimum <= raw <= maximum):
        first_reason = (
            f"intersection k-lines hors plage ({raw:.3g}, plage [{minimum}, {maximum}])"
        )

    raw_segmented, segmented_details = _segmented_profile_selection(valid)
    if (
        raw_segmented is not None
        and np.isfinite(raw_segmented)
        and minimum <= raw_segmented <= maximum
    ):
        return MSTDSelection(
            mstd=_nearest_dimension(float(raw_segmented), candidates),
            raw_intersection=float(raw_segmented),
            method="segmented_mean_stability",
            fallback_reason=first_reason,
            details=segmented_details,
        )

    second_reason = str(segmented_details.get("reason", "intersection segmentée hors plage"))
    if raw_segmented is not None and not (minimum <= raw_segmented <= maximum):
        second_reason = (
            f"intersection segmentée hors plage ({raw_segmented:.3g}, "
            f"plage [{minimum}, {maximum}])"
        )

    stable = valid.loc[valid["mean_stability"] >= mean_stability_threshold]
    if not stable.empty:
        selected = int(stable["n_components"].max())
        method = "mean_stability_threshold"
    else:
        # Dernier filet de sécurité : la meilleure stabilité moyenne, puis la
        # plus petite dimension à égalité pour ne pas sur-décomposer.
        best = valid.sort_values(
            ["mean_stability", "n_components"], ascending=[False, True]
        ).iloc[0]
        selected = int(best["n_components"])
        method = "max_mean_stability"
    return MSTDSelection(
        mstd=selected,
        raw_intersection=None,
        method=method,
        fallback_reason=f"{first_reason}; {second_reason}",
        details={"mean_stability_threshold": float(mean_stability_threshold)},
    )


def _select_branch_dimensions(
    summary: pd.DataFrame,
    mstd: int,
    maximum: int,
) -> tuple[tuple[int, ...], dict[int, tuple[str, ...]]]:
    """Choisit les dimensions ICA à analyser en aval, avec leurs rôles.

    La MSTD est la dimension principale. Ses voisins *sur la grille réellement
    réussie* sont gardés pour vérifier localement la robustesse du choix ; ils
    ne sont donc pas forcément ``mstd - step`` et ``mstd + step`` si un fit a
    échoué. La quatrième position conserve la meilleure stabilité moyenne à
    titre de diagnostic complémentaire. Les doublons de dimensions sont
    volontairement fusionnés, sans perdre les étiquettes de sélection.
    """
    if maximum < 1 or maximum > 4:
        raise ValueError("top_n_dimensions doit être compris entre 1 et 4.")

    successful = summary.loc[summary["status"] == "ok"].dropna(
        subset=["n_components", "mean_stability"]
    )
    dimensions = sorted({int(value) for value in successful["n_components"]})
    if int(mstd) not in dimensions:
        raise RuntimeError("La MSTD doit appartenir aux dimensions ICA réussies.")

    lower = [dimension for dimension in dimensions if dimension < int(mstd)]
    higher = [dimension for dimension in dimensions if dimension > int(mstd)]
    best_mean = int(
        successful.sort_values(
            ["mean_stability", "n_components"], ascending=[False, True]
        ).iloc[0]["n_components"]
    )
    candidates: list[tuple[int | None, str]] = [
        (int(mstd), "mstd"),
        (lower[-1] if lower else None, "nearest_lower"),
        (higher[0] if higher else None, "nearest_higher"),
        (best_mean, "best_mean_stability"),
    ]

    selected: list[int] = []
    role_lists: dict[int, list[str]] = {}
    for dimension, role in candidates:
        if dimension is None:
            continue
        if dimension not in selected:
            if len(selected) >= int(maximum):
                continue
            selected.append(dimension)
            role_lists[dimension] = []
        # Même lorsqu'une dimension est déjà présente, tous ses rôles doivent
        # être visibles dans le rapport et le fichier de sélection.
        if dimension in role_lists:
            role_lists[dimension].append(role)

    return tuple(selected), {
        dimension: tuple(roles) for dimension, roles in role_lists.items()
    }


def _component_names(n_components: int) -> list[str]:
    width = max(2, len(str(n_components)))
    return [f"IC{i:0{width}d}" for i in range(1, n_components + 1)]


def _materialize_decomposition(
    model,
    matrix: np.ndarray,
    sample_names: pd.Index,
    gene_names: pd.Index,
    n_components: int,
    *,
    projection_mean: np.ndarray | None = None,
) -> ICADecomposition:
    """Convertit les attributs publics de ``StabilizedICA`` en DataFrames."""
    names = _component_names(n_components)
    stability_values = np.asarray(model.stability_indexes_, dtype=float)
    sources = np.asarray(model.S_, dtype=float)
    # ``StabilizedICA.transform`` 2.0.0 applique directement
    # ``X.T @ pinv(S_)`` et son propre code laisse volontairement désactivée
    # la soustraction de ``mean_``. Or l'ajustement avec ``whiten=True`` a
    # bien centré les observations (les gènes) suivant cette moyenne, de taille
    # ``n_samples``. Recentrer l'entrée ici rend les métasamples cohérents avec
    # les métagènes appris ; sans cela, toutes les analyses aval ICA sont biaisées.
    #
    # Dans le scan MSTD officiel, l'ICA est fitée sur les PC déjà blanchies et
    # son ``model.mean_`` vaut donc None. ``projection_mean`` transmet alors la
    # moyenne de la PCA unique : on projette bien l'expression originale centrée
    # (patients x gènes), et non les seules coordonnées PCA.
    projection_input = matrix
    fit_mean = projection_mean
    if fit_mean is None:
        fit_mean = getattr(model, "mean_", None)
    if fit_mean is not None:
        fit_mean = np.asarray(fit_mean, dtype=float)
        if fit_mean.ndim != 1 or fit_mean.shape[0] != matrix.shape[0]:
            raise RuntimeError(
                "La moyenne de centrage retournée par stabilized-ica a une forme inattendue."
            )
        # ``matrix.T`` est la matrice centrée dans le package ; après retour à
        # l'orientation patients x gènes, la moyenne par patient se diffuse sur
        # les colonnes de gènes.
        projection_input = matrix - fit_mean.reshape(-1, 1)
    scores = np.asarray(model.transform(projection_input), dtype=float)
    if scores.shape != (matrix.shape[0], n_components):
        raise RuntimeError("La matrice de métasamples ICA a une forme inattendue.")

    stability = pd.DataFrame(
        {
            "component": names,
            "component_rank": np.arange(1, n_components + 1, dtype=int),
            "stability_index": stability_values,
        }
    )
    metagenes = pd.DataFrame(sources, index=names, columns=gene_names)
    metagenes.index.name = "component"
    metasamples = pd.DataFrame(scores, index=sample_names, columns=names)
    metasamples.index.name = "sample"
    return ICADecomposition(
        n_components=n_components,
        metagenes=metagenes,
        metasamples=metasamples,
        stability=stability,
    )


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_index_stability_distribution(
    profiles: pd.DataFrame,
    selection: MSTDSelection,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    dimensions = sorted(profiles["n_components"].unique()) if not profiles.empty else []
    cmap = plt.get_cmap("Greys")
    for i, dimension in enumerate(dimensions):
        frame = profiles.loc[profiles["n_components"] == dimension]
        selected = int(dimension) == selection.mstd
        color = "#c0392b" if selected else cmap(0.45 + 0.45 * i / max(1, len(dimensions) - 1))
        ax.plot(
            frame["component_rank"],
            frame["stability_index"],
            color=color,
            linewidth=2.2 if selected else 0.85,
            alpha=1.0 if selected else 0.62,
            label=f"MSTD = {dimension}" if selected else None,
        )
    ax.axvline(selection.mstd, color="#c0392b", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("rang de la composante ICA")
    ax.set_ylabel("indice de stabilité")
    ax.set_title("Index stability distribution")
    ax.set_ylim(bottom=min(-0.02, float(profiles["stability_index"].min()) - 0.02) if not profiles.empty else -0.02)
    if any(int(d) == selection.mstd for d in dimensions):
        ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_figure(fig, path)


def _plot_mean_stability(
    summary: pd.DataFrame,
    selection: MSTDSelection,
    mean_stability_threshold: float,
    path: Path,
) -> Path:
    frame = summary.loc[summary["status"] == "ok"].dropna(subset=["mean_stability"])
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot(frame["n_components"], frame["mean_stability"], "o-", color="#2878b5", lw=1.8)
    chosen = frame.loc[frame["n_components"] == selection.mstd]
    if not chosen.empty:
        ax.scatter(
            chosen["n_components"],
            chosen["mean_stability"],
            s=85,
            color="#c0392b",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label=f"MSTD = {selection.mstd}",
        )
    ax.axhline(
        mean_stability_threshold,
        color="#555",
        linestyle="--",
        linewidth=0.9,
        label=f"seuil de repli = {mean_stability_threshold:g}",
    )
    if selection.raw_intersection is not None:
        ax.axvline(selection.raw_intersection, color="#c0392b", linestyle=":", lw=0.9)
    ax.set_xlabel("nombre de composantes")
    ax.set_ylabel("stabilité moyenne")
    ax.set_title("Mean stability")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, path)


def _plot_component_stability(
    decomposition: ICADecomposition,
    path: Path,
) -> Path:
    frame = decomposition.stability
    fig, ax = plt.subplots(figsize=(max(7.0, 0.30 * len(frame)), 4.8))
    colors = ["#2878b5"] * len(frame)
    if colors:
        colors[0] = "#c0392b"
    ax.bar(frame["component_rank"], frame["stability_index"], color=colors, width=0.78)
    ax.axhline(frame["stability_index"].mean(), color="#555", linestyle="--", lw=0.9,
               label=f"moyenne = {frame['stability_index'].mean():.3f}")
    ax.set_xlabel("rang de la composante ICA")
    ax.set_ylabel("indice de stabilité")
    ax.set_title(f"Stability of ICA components — MSTD = {decomposition.n_components}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, path)


def _fallback_mds_projection(model, ax: plt.Axes, seed: int) -> None:
    """MDS de repli si ``StabilizedICA.projection`` échoue.

    Les attributs privés sont employés seulement ici, après tentative de
    l'API publique. Ils existent dans stabilized-ica 2.0.0 et permettent de
    conserver le diagnostic plutôt que d'interrompre toute la branche ICA.
    """
    sim = np.asarray(getattr(model, "_Sim", None), dtype=float)
    clusters = np.asarray(getattr(model, "_clusters", None))
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise RuntimeError("La matrice de similarité ICA n'est pas disponible pour le MDS.")
    from sklearn.manifold import MDS

    distance = np.sqrt(np.clip(1.0 - sim, 0.0, None))
    kwargs = dict(n_components=2, dissimilarity="precomputed", random_state=int(seed), n_init=1)
    try:
        coords = MDS(normalized_stress="auto", **kwargs).fit_transform(distance)
    except TypeError:  # scikit-learn < 1.4
        coords = MDS(**kwargs).fit_transform(distance)
    ax.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap="viridis", s=12, alpha=0.8)


def _plot_mds_components(model, decomposition: ICADecomposition, path: Path, seed: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.7))
    fallback_note = ""
    try:
        # API publique stabilized-ica : MDS des n_components * n_runs ICs,
        # colorés selon le cluster Icasso auquel elles appartiennent.
        with _temporary_numpy_seed(seed):
            model.projection(method="mds", ax=ax)
    except Exception as exc:
        logger.warning("Projection MDS native stabilized-ica en échec : %s", exc)
        try:
            _fallback_mds_projection(model, ax, seed)
            fallback_note = " (repli déterministe)"
        except Exception as fallback_exc:
            ax.text(
                0.5,
                0.5,
                "Projection MDS indisponible\n"
                f"{type(fallback_exc).__name__}: {fallback_exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            fallback_note = " (indisponible)"
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(
        f"Multidimensional scaling for ICA components — MSTD = "
        f"{decomposition.n_components}{fallback_note}"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_figure(fig, path)


def _plot_metagene_distributions(decomposition: ICADecomposition, path: Path) -> Path:
    """Histogrammes des poids de gènes de tous les métagènes MSTD."""
    metagenes = decomposition.metagenes
    n_components = len(metagenes)
    ncols = min(4, max(1, n_components))
    nrows = int(np.ceil(n_components / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.3 * ncols, 2.45 * nrows),
        squeeze=False,
    )
    stability_by_component = decomposition.stability.set_index("component")["stability_index"]
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, component in zip(axes.flat, metagenes.index):
        ax.set_visible(True)
        values = metagenes.loc[component].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        bins: int | str = "fd" if values.size >= 4 and np.ptp(values) > 0 else 12
        ax.hist(values, bins=bins, color="#4292c6", edgecolor="white", linewidth=0.35)
        ax.axvline(0.0, color="#555", lw=0.7)
        ax.set_title(f"{component}  ·  Iq={stability_by_component[component]:.3f}", fontsize=8.5)
        ax.set_xlabel("poids du gène", fontsize=7.5)
        ax.set_ylabel("n gènes", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Distribution des métagènes — MSTD = {decomposition.n_components}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return _save_figure(fig, path)


def _persist_decomposition(decomposition: ICADecomposition, table_dir: Path) -> dict[str, Path]:
    """Écrit les matrices biologiquement utiles, plutôt qu'un pickle fragile."""
    table_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ica_dim{decomposition.n_components:02d}"
    paths = {
        "metagenes": table_dir / f"{stem}_metagenes.csv.gz",
        "metasamples": table_dir / f"{stem}_metasamples.csv.gz",
        "stability": table_dir / f"{stem}_stability.csv",
    }
    decomposition.metagenes.to_csv(paths["metagenes"], compression="gzip")
    decomposition.metasamples.to_csv(paths["metasamples"], compression="gzip")
    decomposition.stability.to_csv(paths["stability"], index=False)
    decomposition.output_paths.update(paths)
    return paths


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Objet non sérialisable en JSON : {type(value).__name__}")


def run_ica(
    X: pd.DataFrame | np.ndarray,
    outdir: str | Path,
    *,
    min_components: int = 6,
    max_components: int = 30,
    step: int = 1,
    n_runs: int = 50,
    top_n_dimensions: int = 4,
    algorithm: str = "fastica_par",
    fun: str = "cube",
    resampling: str | None = None,
    max_iter: int = 2000,
    n_jobs: int = 1,
    random_state: int = 42,
    mean_stability_threshold: float = 0.60,
    deterministic: bool = True,
) -> ICAResult:
    """Exécute le scan ICA stabilisé et persiste les résultats.

    Parameters
    ----------
    X
        Matrice prétraitée ``échantillons x gènes``. Pour des données RNA-seq,
        utiliser la matrice VST/logCPM filtrée et centrée produite par le
        prétraitement du pipeline, jamais les counts bruts.
    outdir
        Racine du run. Les résultats sont écrits dans ``tables/ica`` et
        ``figures/ica``.
    min_components, max_components, step
        Intervalle des dimensions ICA évaluées. La borne haute est réduite si
        nécessaire à ``min(n_samples - 1, n_genes)``.
    n_runs
        Nombre de redémarrages ICA par dimension. 50 est un compromis ; 100
        correspond au protocole MSTD publié, au prix d'un calcul plus long.
    top_n_dimensions
        Nombre maximal de branches ICA conservées (quatre par défaut), dans
        cet ordre : MSTD, dimension testée immédiatement inférieure, dimension
        testée immédiatement supérieure, meilleure stabilité moyenne. Les
        doublons sont fusionnés ; une borne de grille peut donc produire moins
        de quatre branches.
    algorithm, fun, resampling, max_iter
        Paramètres transmis à :class:`sica.base.StabilizedICA`. ``cube`` est la
        non-linéarité *pow3* du protocole MSTD de référence.
    n_jobs
        Parallélisme interne de ``stabilized-ica``. Avec ``deterministic=True``
        (défaut), il est volontairement forcé à 1 : le package ne propose pas
        de ``random_state`` public et son parallélisme rend les initialisations
        de FastICA non garanties reproductibles.
    random_state
        Graine dont dérivent une graine par dimension.
    mean_stability_threshold
        Seuil utilisé seulement par le dernier repli MSTD. La littérature
        rapporte souvent une MSTD vers une stabilité moyenne de 0.6.
    deterministic
        Force les fits en série pour rendre le scan et la sélection répétables.

    Returns
    -------
    ICAResult
        Tables en mémoire et chemins de toutes les sorties sauvegardées.
    """
    if n_runs < 2:
        raise ValueError("n_runs doit être >= 2 pour estimer une stabilité ICA.")
    if top_n_dimensions < 1 or top_n_dimensions > 4:
        raise ValueError("top_n_dimensions doit être compris entre 1 et 4.")
    if max_iter < 1:
        raise ValueError("max_iter doit être >= 1.")
    if n_jobs == 0:
        raise ValueError("n_jobs ne peut pas valoir 0.")
    if int(random_state) < 0:
        raise ValueError("random_state doit être >= 0.")
    if not 0.0 <= float(mean_stability_threshold) <= 1.0:
        raise ValueError("mean_stability_threshold doit être compris entre 0 et 1.")

    if isinstance(resampling, str) and resampling.strip().lower() in {"", "none", "null"}:
        resampling = None
    if resampling not in {None, "bootstrap", "fast_bootstrap"}:
        raise ValueError("resampling doit être None, 'bootstrap' ou 'fast_bootstrap'.")

    StabilizedICA = _require_stabilized_ica()
    matrix, sample_names, gene_names = _as_expression_matrix(X)
    dimensions = _candidate_dimensions(
        min_components,
        max_components,
        step,
        matrix.shape[0],
        matrix.shape[1],
    )
    if dimensions[-1] < int(max_components):
        logger.warning(
            "ICA : max_components ramené de %d à %d (limite de rang des données).",
            max_components,
            dimensions[-1],
        )

    effective_n_jobs = 1 if deterministic else int(n_jobs)
    if deterministic and int(n_jobs) != 1:
        logger.info(
            "ICA : deterministic=True, n_jobs forcé à 1 pour stabiliser les "
            "initialisations FastICA (n_jobs demandé=%s).",
            n_jobs,
        )

    seed_sequence = np.random.SeedSequence(int(random_state))
    # Une graine est réservée au whitening MSTD commun et une autre au MDS ;
    # elles ne dépendent donc pas du nombre de dimensions persistées.
    child_sequences = seed_sequence.spawn(len(dimensions) + 2)
    seeds = {
        dimension: int(sequence.generate_state(1, dtype=np.uint32)[0])
        for dimension, sequence in zip(dimensions, child_sequences[:-2])
    }
    whitening_seed = int(child_sequences[-2].generate_state(1, dtype=np.uint32)[0])
    mds_seed = int(child_sequences[-1].generate_state(1, dtype=np.uint32)[0])

    if resampling is None:
        # Chemin identique à ``sica.base.MSTD`` : une PCA blanchie au maximum
        # testé, puis des ICA ``whiten=False`` sur ses premières composantes.
        mstd_whitening = _prepare_mstd_whitening(
            matrix,
            dimensions[-1],
            seed=whitening_seed,
        )
        scan_whitening = "single_max_pca"
        logger.info(
            "ICA MSTD : whitening PCA unique à %d composantes, partagé par "
            "toutes les dimensions.",
            mstd_whitening.max_components,
        )
    else:
        # stabilized-ica rééchantillonne les observations en interne. Un
        # whitening commun ferait perdre ce rééchantillonnage ; cette variante
        # est donc volontairement un repli avec whitening par dimension/run.
        mstd_whitening = None
        scan_whitening = "per_dimension_resampling"
        logger.warning(
            "ICA MSTD : resampling=%s ; dérogation au whitening unique "
            "officiel, un whitening est recalculé par dimension et "
            "rééchantillonnage.",
            resampling,
        )

    logger.info(
        "ICA stabilisée : %d tumeurs x %d gènes ; dimensions=%s ; n_runs=%d ; "
        "algorithme=%s/%s.",
        matrix.shape[0],
        matrix.shape[1],
        list(dimensions),
        n_runs,
        algorithm,
        fun,
    )
    summary, profiles = _scan_dimensions(
        StabilizedICA,
        matrix,
        dimensions,
        n_runs=int(n_runs),
        algorithm=algorithm,
        fun=fun,
        resampling=resampling,
        max_iter=int(max_iter),
        n_jobs=effective_n_jobs,
        seeds=seeds,
        mean_stability_threshold=float(mean_stability_threshold),
        mstd_whitening=mstd_whitening,
    )
    selection = _select_mstd(summary, profiles, float(mean_stability_threshold))

    persisted_dimensions, dimension_roles = _select_branch_dimensions(
        summary, selection.mstd, int(top_n_dimensions)
    )
    summary["is_mstd"] = summary["n_components"].eq(selection.mstd)
    summary["selection_roles"] = summary["n_components"].map(
        lambda value: ";".join(dimension_roles.get(int(value), ()))
    )
    summary["is_persisted"] = summary["n_components"].isin(persisted_dimensions)

    root = Path(outdir)
    table_dir = root / "tables" / "ica"
    figure_dir = root / "figures" / "ica"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {
        "scan_summary": table_dir / "ica_mstd_scan.csv",
        "stability_profiles": table_dir / "ica_stability_profiles.csv.gz",
        "selection": table_dir / "ica_mstd_selection.json",
    }
    summary.to_csv(output_paths["scan_summary"], index=False)
    profiles.to_csv(output_paths["stability_profiles"], index=False, compression="gzip")

    decompositions: dict[int, ICADecomposition] = {}
    # Un modèle stabilized-ica conserve les composantes de tous les runs et sa
    # matrice de similarité. Seul celui de la MSTD sert au graphique MDS ; ne
    # pas retenir ceux des autres dimensions évite un surcoût mémoire important
    # avec n_runs=100 et des matrices transcriptomiques réelles.
    mstd_model: Any | None = None
    mstd_diagnostic: ICADecomposition | None = None
    fit_dimensions = tuple(dict.fromkeys([*persisted_dimensions, selection.mstd]))
    for dimension in fit_dimensions:
        if dimension in persisted_dimensions:
            logger.info("ICA : sauvegarde de la décomposition n_components=%d…", dimension)
        else:
            logger.info("ICA : recalcul diagnostic MSTD n_components=%d…", dimension)
        try:
            # Même graine que pendant le scan : avec deterministic=True, cette
            # décomposition est identique à celle dont provient son profil.
            model, projection_mean = _fit_output_dimension(
                StabilizedICA,
                matrix,
                dimension,
                int(n_runs),
                algorithm=algorithm,
                fun=fun,
                resampling=resampling,
                max_iter=int(max_iter),
                n_jobs=effective_n_jobs,
                seed=seeds[dimension],
                mstd_whitening=mstd_whitening,
            )
            decomposition = _materialize_decomposition(
                model,
                matrix,
                sample_names,
                gene_names,
                dimension,
                projection_mean=projection_mean,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Impossible de recalculer la décomposition ICA retenue "
                f"n_components={dimension}."
            ) from exc
        if dimension == selection.mstd:
            mstd_model = model
            mstd_diagnostic = decomposition
        else:
            del model
        if dimension in persisted_dimensions:
            _persist_decomposition(decomposition, table_dir)
            decompositions[dimension] = decomposition
            for key, value in decomposition.output_paths.items():
                output_paths[f"dim{dimension}_{key}"] = value

    diagnostic = mstd_diagnostic
    if diagnostic is None or mstd_model is None:  # garde-fou pour les futurs changements
        raise RuntimeError("Le modèle ICA MSTD n'est pas disponible pour le diagnostic MDS.")
    output_paths.update(
        {
            "index_stability_distribution": figure_dir / "ica_index_stability_distribution.png",
            "mean_stability": figure_dir / "ica_mean_stability.png",
            "component_stability": figure_dir / f"ica_component_stability_m{selection.mstd}.png",
            "mds_components": figure_dir / f"ica_component_mds_m{selection.mstd}.png",
            "metagene_distributions": figure_dir / f"ica_metagene_distribution_m{selection.mstd}.png",
        }
    )
    _plot_index_stability_distribution(profiles, selection, output_paths["index_stability_distribution"])
    _plot_mean_stability(
        summary,
        selection,
        float(mean_stability_threshold),
        output_paths["mean_stability"],
    )
    _plot_component_stability(diagnostic, output_paths["component_stability"])
    _plot_mds_components(mstd_model, diagnostic, output_paths["mds_components"], mds_seed)
    _plot_metagene_distributions(diagnostic, output_paths["metagene_distributions"])

    params = {
        "min_components": int(min_components),
        "max_components_requested": int(max_components),
        "max_components_tested": int(dimensions[-1]),
        "step": int(step),
        "candidate_dimensions": list(dimensions),
        "n_runs": int(n_runs),
        "top_n_dimensions": int(top_n_dimensions),
        "algorithm": algorithm,
        "fun": fun,
        "resampling": resampling,
        "max_iter": int(max_iter),
        "n_jobs_requested": int(n_jobs),
        "n_jobs_effective": int(effective_n_jobs),
        "random_state": int(random_state),
        "mean_stability_threshold": float(mean_stability_threshold),
        "deterministic": bool(deterministic),
        "scan_whitening": scan_whitening,
        "scan_whitening_max_components": (
            int(mstd_whitening.max_components) if mstd_whitening is not None else None
        ),
        "scan_whitening_seed": whitening_seed if mstd_whitening is not None else None,
        "selection_roles": {
            str(dimension): list(roles)
            for dimension, roles in dimension_roles.items()
        },
        "selection_order": [
            "mstd", "nearest_lower", "nearest_higher", "best_mean_stability"
        ],
    }
    selection_payload = {
        "selection": asdict(selection),
        "selected_dimensions": list(persisted_dimensions),
        "dimension_roles": {
            str(dimension): list(roles)
            for dimension, roles in dimension_roles.items()
        },
        "persisted_dimensions": list(persisted_dimensions),
        "mstd_diagnostic_persisted": True,
        "params": params,
    }
    with output_paths["selection"].open("w", encoding="utf-8") as handle:
        json.dump(selection_payload, handle, indent=2, ensure_ascii=False, default=_json_default)

    logger.info(
        "ICA MSTD retenue : %d (%s). Dimensions sauvegardées (rôles : %s) : %s. "
        "Sorties : %s",
        selection.mstd,
        selection.method,
        {dimension: list(roles) for dimension, roles in dimension_roles.items()},
        list(persisted_dimensions),
        table_dir,
    )
    return ICAResult(
        mstd=selection.mstd,
        selection=selection,
        scan_summary=summary,
        stability_profiles=profiles,
        decompositions=decompositions,
        mstd_diagnostic=diagnostic,
        selected_dimensions=persisted_dimensions,
        dimension_roles=dimension_roles,
        persisted_dimensions=persisted_dimensions,
        params=params,
        output_paths=output_paths,
    )


# Nom explicite pratique pour les appels de pipeline, tout en conservant
# ``run_ica`` court pour les usages directs/notebooks.
run_ica_mstd = run_ica


__all__ = [
    "ICADecomposition",
    "ICAResult",
    "MSTDSelection",
    "run_ica",
    "run_ica_mstd",
]
