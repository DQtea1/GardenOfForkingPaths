"""Comparaisons descriptives entre clusters dans l'espace des métasamples ICA.

Ce module est volontairement indépendant du consensus clustering : il reçoit
une matrice échantillons × composantes et des partitions déjà calculées. Pour
chaque ``k``, il exporte les contrastes de centroïdes, de médoïdes, les tailles
d'effet de Cohen ainsi que la distance euclidienne entre centroïdes après
z-score global des composantes.

Les contrastes sont descriptifs. Ils ne constituent ni des tests
d'hypothèse, ni une validation prédictive des clusters ayant servi à les
construire.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


METHODS = ("centroid", "medoid", "cohen")


def _finite_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _cluster_key(value: Any) -> tuple[int, Any]:
    """Trie naturellement les labels numériques, puis les autres labels."""
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _warning(*messages: str | None) -> str:
    return " ; ".join(dict.fromkeys(m for m in messages if m))


def _cluster_warning(
    *,
    n_cluster: int,
    n_valid: int,
    min_cluster_size: int,
    label: str,
) -> str:
    messages = []
    if n_cluster < min_cluster_size:
        messages.append(
            f"{label}: petit cluster (n={n_cluster} < {min_cluster_size})"
        )
    if n_valid < n_cluster:
        messages.append(
            f"{label}: {n_cluster - n_valid} valeur(s) manquante(s)"
        )
    return _warning(*messages)


def _medoid(
    values: np.ndarray,
    samples: np.ndarray,
) -> tuple[np.ndarray | None, str | None, str]:
    """Retourne le médoïde euclidien dans l'espace ICA brut.

    Les observations incomplètes sont exclues du calcul plutôt qu'imputées.
    L'ICA produit normalement une matrice complète ; cette règle rend néanmoins
    le comportement explicite en cas de données importées ou altérées.
    """
    complete = np.isfinite(values).all(axis=1)
    messages = []
    if not complete.all():
        messages.append(
            f"{int((~complete).sum())} métasample(s) incomplet(s) exclu(s) du médoïde"
        )
    x = values[complete]
    ids = samples[complete]
    if not len(x):
        return None, None, _warning(*messages, "médoïde non calculable")
    if len(x) == 1:
        return x[0], str(ids[0]), _warning(*messages)
    # Matrice des distances euclidiennes. En cas d'égalité, argmin conserve le
    # premier échantillon dans l'ordre d'entrée, ce qui est reproductible.
    distances = np.sqrt(np.square(x[:, None, :] - x[None, :, :]).sum(axis=2))
    index = int(np.argmin(distances.sum(axis=1)))
    return x[index], str(ids[index]), _warning(*messages)


def _component_rows(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    *,
    min_cluster_size: int,
    variance_epsilon: float,
) -> dict[str, list[dict]]:
    x = matrix.to_numpy(dtype=float)
    sample_ids = matrix.index.astype(str).to_numpy()
    components = list(map(str, matrix.columns))
    clusters = sorted(pd.unique(labels).tolist(), key=_cluster_key)

    positions = {cluster: np.flatnonzero(labels == cluster) for cluster in clusters}
    medoids = {}
    for cluster, idx in positions.items():
        medoids[cluster] = _medoid(x[idx], sample_ids[idx])

    rows = {method: [] for method in METHODS}
    for cluster_a, cluster_b in combinations(clusters, 2):
        idx_a, idx_b = positions[cluster_a], positions[cluster_b]
        values_a, values_b = x[idx_a], x[idx_b]
        med_a, med_sample_a, med_warning_a = medoids[cluster_a]
        med_b, med_sample_b, med_warning_b = medoids[cluster_b]

        for component_index, component in enumerate(components):
            a = values_a[:, component_index]
            b = values_b[:, component_index]
            a_finite, b_finite = a[np.isfinite(a)], b[np.isfinite(b)]
            n_valid_a, n_valid_b = len(a_finite), len(b_finite)
            mean_a = float(np.mean(a_finite)) if n_valid_a else np.nan
            mean_b = float(np.mean(b_finite)) if n_valid_b else np.nan
            sd_a = (
                float(np.std(a_finite, ddof=1)) if n_valid_a >= 2 else np.nan
            )
            sd_b = (
                float(np.std(b_finite, ddof=1)) if n_valid_b >= 2 else np.nan
            )
            raw_difference = mean_a - mean_b
            base_warning = _warning(
                _cluster_warning(
                    n_cluster=len(idx_a),
                    n_valid=n_valid_a,
                    min_cluster_size=min_cluster_size,
                    label=f"C{cluster_a}",
                ),
                _cluster_warning(
                    n_cluster=len(idx_b),
                    n_valid=n_valid_b,
                    min_cluster_size=min_cluster_size,
                    label=f"C{cluster_b}",
                ),
            )
            common = {
                "component": component,
                "cluster_a": str(cluster_a),
                "cluster_b": str(cluster_b),
                "n_a": int(len(idx_a)),
                "n_b": int(len(idx_b)),
                "n_valid_a": int(n_valid_a),
                "n_valid_b": int(n_valid_b),
            }

            centroid_value = _finite_or_none(raw_difference)
            rows["centroid"].append({
                **common,
                "method": "centroid",
                "representative_a": _finite_or_none(mean_a),
                "representative_b": _finite_or_none(mean_b),
                "representative_sample_a": None,
                "representative_sample_b": None,
                "signed_difference": centroid_value,
                "absolute_difference": (
                    abs(centroid_value) if centroid_value is not None else None
                ),
                "effect_size": None,
                "sd_a": _finite_or_none(sd_a),
                "sd_b": _finite_or_none(sd_b),
                "pooled_sd": None,
                "warning": base_warning,
            })

            med_value_a = (
                med_a[component_index] if med_a is not None else np.nan
            )
            med_value_b = (
                med_b[component_index] if med_b is not None else np.nan
            )
            med_difference = med_value_a - med_value_b
            med_value = _finite_or_none(med_difference)
            rows["medoid"].append({
                **common,
                "method": "medoid",
                "representative_a": _finite_or_none(med_value_a),
                "representative_b": _finite_or_none(med_value_b),
                "representative_sample_a": med_sample_a,
                "representative_sample_b": med_sample_b,
                "signed_difference": med_value,
                "absolute_difference": (
                    abs(med_value) if med_value is not None else None
                ),
                "effect_size": None,
                "sd_a": _finite_or_none(sd_a),
                "sd_b": _finite_or_none(sd_b),
                "pooled_sd": None,
                "warning": _warning(
                    base_warning, med_warning_a, med_warning_b
                ),
            })

            cohen_messages = [base_warning]
            pooled_sd = np.nan
            effect = np.nan
            if n_valid_a < 2 or n_valid_b < 2:
                cohen_messages.append(
                    "Cohen d non calculable: moins de 2 valeurs valides dans un cluster"
                )
            else:
                denominator = n_valid_a + n_valid_b - 2
                pooled_variance = (
                    (n_valid_a - 1) * sd_a**2
                    + (n_valid_b - 1) * sd_b**2
                ) / denominator
                pooled_sd = float(np.sqrt(max(0.0, pooled_variance)))
                if pooled_sd <= variance_epsilon:
                    cohen_messages.append(
                        "Cohen d non calculable: variance poolée nulle ou quasi nulle"
                    )
                else:
                    effect = raw_difference / pooled_sd
                var_a, var_b = sd_a**2, sd_b**2
                if min(var_a, var_b) <= variance_epsilon < max(var_a, var_b):
                    cohen_messages.append(
                        "variances intra-cluster fortement différentes (une variance quasi nulle)"
                    )
                elif min(var_a, var_b) > variance_epsilon:
                    ratio = max(var_a, var_b) / min(var_a, var_b)
                    if ratio >= 4:
                        cohen_messages.append(
                            f"variances intra-cluster fortement différentes (rapport={ratio:.2f})"
                        )
            effect_value = _finite_or_none(effect)
            # Correction de petit échantillon proposée par Hedges. Elle est
            # exportée en parallèle du d brut : l'interface peut ainsi
            # l'activer sans recalculer ni modifier les données du rapport.
            degrees_of_freedom = n_valid_a + n_valid_b - 2
            hedges_correction = (
                1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
                if degrees_of_freedom >= 2
                else np.nan
            )
            hedges_g = (
                effect * hedges_correction
                if np.isfinite(effect) and np.isfinite(hedges_correction)
                else np.nan
            )
            rows["cohen"].append({
                **common,
                "method": "cohen",
                "representative_a": _finite_or_none(mean_a),
                "representative_b": _finite_or_none(mean_b),
                "representative_sample_a": None,
                "representative_sample_b": None,
                "signed_difference": _finite_or_none(raw_difference),
                "absolute_difference": (
                    abs(raw_difference) if np.isfinite(raw_difference) else None
                ),
                "effect_size": effect_value,
                "absolute_effect_size": (
                    abs(effect_value) if effect_value is not None else None
                ),
                "hedges_g": _finite_or_none(hedges_g),
                "hedges_correction": _finite_or_none(hedges_correction),
                "sd_a": _finite_or_none(sd_a),
                "sd_b": _finite_or_none(sd_b),
                "pooled_sd": _finite_or_none(pooled_sd),
                "warning": _warning(*cohen_messages),
            })
    return rows


def _global_rows(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    *,
    min_cluster_size: int,
    variance_epsilon: float,
) -> tuple[list[dict], list[dict], dict]:
    """Distance euclidienne entre centroïdes après z-score global.

    La moyenne et l'écart-type (échantillonnal, ``ddof=1``) sont calculés une
    seule fois par composante sur l'ensemble des métasamples, avant toute
    partition par cluster.
    """
    x = matrix.to_numpy(dtype=float)
    components = list(map(str, matrix.columns))
    counts = np.isfinite(x).sum(axis=0)
    # Calcul explicite pour éviter les RuntimeWarning de ``nanmean/nanstd``
    # lorsqu'une composante ne possède aucune ou une seule valeur valide.
    means = np.full(x.shape[1], np.nan, dtype=float)
    sds = np.full(x.shape[1], np.nan, dtype=float)
    for component_index in range(x.shape[1]):
        finite = x[np.isfinite(x[:, component_index]), component_index]
        if len(finite):
            means[component_index] = float(np.mean(finite))
        if len(finite) >= 2:
            sds[component_index] = float(np.std(finite, ddof=1))
    valid_components = (
        (counts >= 2)
        & np.isfinite(means)
        & np.isfinite(sds)
        & (sds > variance_epsilon)
    )
    z = np.full_like(x, np.nan, dtype=float)
    z[:, valid_components] = (
        x[:, valid_components] - means[valid_components]
    ) / sds[valid_components]

    clusters = sorted(pd.unique(labels).tolist(), key=_cluster_key)
    positions = {cluster: np.flatnonzero(labels == cluster) for cluster in clusters}
    centroids = {}
    for cluster, idx in positions.items():
        centroid = np.full(z.shape[1], np.nan, dtype=float)
        for component_index in range(z.shape[1]):
            finite = z[idx, component_index]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                centroid[component_index] = float(np.mean(finite))
        centroids[cluster] = centroid
    global_rows: list[dict] = []
    contribution_rows: list[dict] = []
    excluded_components = [
        component for component, valid in zip(components, valid_components)
        if not valid
    ]
    settings = {
        "metric": "euclidean",
        "standardization": "global_zscore",
        "standard_deviation_ddof": 1,
        "n_components_total": int(len(components)),
        "n_components_valid": int(valid_components.sum()),
        "excluded_components": excluded_components,
    }

    for cluster_a, cluster_b in combinations(clusters, 2):
        idx_a, idx_b = positions[cluster_a], positions[cluster_b]
        delta = centroids[cluster_a] - centroids[cluster_b]
        pair_valid = valid_components & np.isfinite(delta)
        missing_values = int(
            (~np.isfinite(x[np.r_[idx_a, idx_b]][:, valid_components])).sum()
        )
        q = np.where(pair_valid, np.square(delta), np.nan)
        q_sum = float(np.nansum(q))
        distance = float(np.sqrt(q_sum)) if pair_valid.any() else np.nan
        pair_warning = _warning(
            (
                f"C{cluster_a}: petit cluster "
                f"(n={len(idx_a)} < {min_cluster_size})"
                if len(idx_a) < min_cluster_size else None
            ),
            (
                f"C{cluster_b}: petit cluster "
                f"(n={len(idx_b)} < {min_cluster_size})"
                if len(idx_b) < min_cluster_size else None
            ),
            (
                f"{len(excluded_components)} composante(s) exclue(s): "
                "variance globale nulle/quasi nulle ou données insuffisantes"
                if excluded_components else None
            ),
            (
                f"{int(valid_components.sum() - pair_valid.sum())} composante(s) "
                "supplémentaire(s) exclue(s) pour cette paire (données manquantes)"
                if pair_valid.sum() < valid_components.sum() else None
            ),
            (
                f"{missing_values} valeur(s) manquante(s) dans la paire ; "
                "centroïdes calculés sur les valeurs disponibles"
                if missing_values else None
            ),
        )
        pair_contributions = []
        for component_index, component in enumerate(components):
            if not pair_valid[component_index]:
                continue
            relative = q[component_index] / q_sum if q_sum > 0 else 0.0
            item = {
                "component": component,
                "signed_standardized_contrast": float(delta[component_index]),
                "quadratic_contribution": float(q[component_index]),
                "relative_contribution": float(relative),
            }
            pair_contributions.append(item)
        pair_contributions.sort(
            key=lambda item: item["quadratic_contribution"], reverse=True
        )
        for rank, item in enumerate(pair_contributions, start=1):
            contribution_rows.append({
                "cluster_a": str(cluster_a),
                "cluster_b": str(cluster_b),
                "rank": rank,
                **item,
            })
        global_rows.append({
            "cluster_a": str(cluster_a),
            "cluster_b": str(cluster_b),
            "n_a": int(len(idx_a)),
            "n_b": int(len(idx_b)),
            "distance": _finite_or_none(distance),
            "n_components": int(pair_valid.sum()),
            "top_components": [
                item["component"] for item in pair_contributions[:3]
            ],
            "warning": pair_warning,
        })
    return global_rows, contribution_rows, settings


def compare_ica_clusters(
    metasamples: pd.DataFrame,
    labels_by_k: Mapping[int, Any],
    output_dir: str | Path,
    *,
    min_cluster_size: int = 10,
    variance_epsilon: float = 1e-12,
    clustering_method: str | None = None,
) -> dict:
    """Calcule et exporte toutes les comparaisons ICA pour tous les ``k``.

    Parameters
    ----------
    metasamples:
        Matrice échantillons × composantes ICA.
    labels_by_k:
        Dictionnaire ``k -> labels`` aligné sur les lignes de ``metasamples``.
    output_dir:
        Dossier racine de la dimension ICA. Les résultats sont écrits dans
        ``cluster_comparisons/k<k>/``.
    min_cluster_size:
        Seuil purement diagnostique sous lequel un avertissement est ajouté.
    variance_epsilon:
        Seuil de variance quasi nulle.
    clustering_method:
        Méthode ayant produit les partitions, conservée dans les métadonnées
        pour contextualiser l'interprétation descriptive dans le rapport.
    """
    if not isinstance(metasamples, pd.DataFrame):
        raise TypeError("metasamples doit être un DataFrame échantillons × composantes.")
    if metasamples.empty or metasamples.shape[1] == 0:
        raise ValueError("metasamples ne peut pas être vide.")
    matrix = metasamples.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if matrix.index.has_duplicates:
        raise ValueError("Les identifiants de métasamples doivent être uniques.")

    root = Path(output_dir) / "cluster_comparisons"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "min_cluster_size": int(min_cluster_size),
            "variance_epsilon": float(variance_epsilon),
            "medoid_metric": "euclidean",
            "scope": "all_retained_ica_dimensions_and_all_tested_k",
            "descriptive_only": True,
            "clustering_method": (
                str(clustering_method) if clustering_method is not None else None
            ),
        },
        "perK": {},
    }

    for k in sorted(labels_by_k):
        labels = np.asarray(labels_by_k[k])
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError(
                f"k={k}: {len(labels)} labels pour {len(matrix)} métasamples."
            )
        if pd.isna(labels).any():
            raise ValueError(f"k={k}: les labels de cluster contiennent des valeurs manquantes.")
        expected_pairs = int(len(pd.unique(labels)) * (len(pd.unique(labels)) - 1) / 2)
        componentwise = _component_rows(
            matrix,
            labels,
            min_cluster_size=min_cluster_size,
            variance_epsilon=variance_epsilon,
        )
        global_rows, contributions, global_settings = _global_rows(
            matrix,
            labels,
            min_cluster_size=min_cluster_size,
            variance_epsilon=variance_epsilon,
        )
        k_dir = root / f"k{int(k)}"
        k_dir.mkdir(parents=True, exist_ok=True)
        for method, rows in componentwise.items():
            pd.DataFrame(rows).to_csv(
                k_dir / f"componentwise_{method}.csv", index=False
            )
        pd.DataFrame(global_rows).to_csv(
            k_dir / "global_standardized_distance.csv", index=False
        )
        pd.DataFrame(contributions).to_csv(
            k_dir / "global_component_contributions.csv", index=False
        )
        payload["perK"][str(int(k))] = {
            "n_clusters": int(len(pd.unique(labels))),
            "n_pairs": expected_pairs,
            "componentwise": componentwise,
            "global": global_rows,
            "contributions": contributions,
            "global_settings": global_settings,
        }
    return payload
