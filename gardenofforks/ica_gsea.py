"""Annotation fonctionnelle des métagènes ICA par GSEA pré-classé.

Chaque ligne de la matrice ``metagenes`` est un axe ICA signé : les gènes à
poids négatif définissent un pôle et ceux à poids positif l'autre. Cette étape
classe tous les gènes selon leur poids, exécute un GSEA pour chaque collection
configurée et conserve le NES, les p/q-valeurs et le leading edge.

Le signe d'une composante ICA est arbitraire au sens mathématique. Les pôles
positif et négatif sont donc interprétables l'un relativement à l'autre, sans
leur attribuer intrinsèquement une direction d'activation universelle.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from .degsea import gsea_prerank_scores, resolve_gene_sets

logger = logging.getLogger(__name__)


def _safe_filename(value: object) -> str:
    """Produit un fragment de chemin stable sans modifier le libellé métier."""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "unnamed"


def _run_one_metagene(
    component: str,
    scores: pd.Series,
    collection: str,
    gene_set_path: str,
    output_path: Path,
    *,
    permutations: int,
    min_size: int,
    max_size: int,
    threads: int,
    seed: int,
) -> tuple[str, str, pd.DataFrame | None]:
    table = gsea_prerank_scores(
        scores,
        gene_set_path,
        permutations=permutations,
        min_size=min_size,
        max_size=max_size,
        threads=threads,
        seed=seed,
    )
    if table is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    return component, collection, table


def run_ica_metagene_gsea(
    metagenes: pd.DataFrame,
    gene_sets,
    outdir: Path,
    *,
    permutations: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    n_jobs: int = 1,
    seed: int = 0,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Exécute un GSEA sur chaque métagène et chaque collection.

    Parameters
    ----------
    metagenes
        Matrice ``composantes × gènes`` de poids ICA signés.
    gene_sets
        Chemin GMT ou dictionnaire ``{collection: chemin GMT}``.
    outdir
        Dossier de cette dimension ICA. Les résultats sont écrits sous
        ``outdir/metagene_gsea/<composante>/gsea_<collection>.csv``.

    Returns
    -------
    dict
        ``{composante: {collection: table GSEA}}``. Une collection dont le
        calcul a échoué n'est pas ajoutée au dictionnaire.
    """
    if not isinstance(metagenes, pd.DataFrame) or metagenes.empty:
        logger.warning("GSEA ICA : matrice de métagènes vide, étape sautée.")
        return {}
    if int(min_size) < 1 or int(max_size) < int(min_size):
        raise ValueError("GSEA ICA : min_size/max_size invalides.")
    if int(permutations) < 0:
        raise ValueError("GSEA ICA : permutations doit être positif ou nul.")

    collections = resolve_gene_sets(gene_sets)
    if not collections:
        logger.warning("GSEA ICA : aucune collection GMT existante, étape sautée.")
        return {}

    matrix = metagenes.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    output_root = Path(outdir) / "metagene_gsea"

    tasks = []
    for component in matrix.index:
        component_dir = output_root / _safe_filename(component)
        for collection, path in collections.items():
            output_path = component_dir / f"gsea_{_safe_filename(collection)}.csv"
            tasks.append((component, matrix.loc[component].copy(), collection, path, output_path))

    parallel_tasks = n_jobs != 1 and len(tasks) > 1
    inner_threads = 1 if parallel_tasks else (
        (os.cpu_count() or 1) if n_jobs in (-1, 0, None)
        else max(1, int(n_jobs))
    )
    logger.info(
        "GSEA ICA : %d métagène(s) × %d collection(s), %s (n_jobs=%s).",
        len(matrix), len(collections),
        "en parallèle" if parallel_tasks else "séquentiellement", n_jobs,
    )
    computed = Parallel(n_jobs=n_jobs if parallel_tasks else 1)(
        delayed(_run_one_metagene)(
            component, scores, collection, path, output_path,
            permutations=int(permutations), min_size=int(min_size),
            max_size=int(max_size), threads=inner_threads, seed=int(seed),
        )
        for component, scores, collection, path, output_path in tasks
    )

    results: dict[str, dict[str, pd.DataFrame]] = {
        component: {} for component in matrix.index
    }
    summary_rows = []
    manifest_rows = []
    for component, collection, table in computed:
        manifest_rows.append({
            "component": component,
            "collection": collection,
            "status": "ok" if table is not None else "failed_or_empty",
            "n_pathways": 0 if table is None else int(len(table)),
        })
        if table is None:
            continue
        results[component][collection] = table
        if "Term" not in table:
            continue
        for row in table.to_dict(orient="records"):
            fdr = pd.to_numeric(row.get("FDR q-val"), errors="coerce")
            if pd.isna(fdr) or float(fdr) >= 0.25:
                continue
            summary_rows.append({
                "component": component,
                "collection": collection,
                "term": row.get("Term"),
                "NES": row.get("NES"),
                "NOM_pval": row.get("NOM p-val"),
                "FDR": float(fdr),
                "lead_genes": row.get("Lead_genes"),
            })
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(output_root / "gsea_run_manifest.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_root / "gsea_summary.csv", index=False)
    return results


__all__ = ["run_ica_metagene_gsea"]
