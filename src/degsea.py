"""Étape 6 — DEGSEA : expression différentielle (DESeq2) + GSEA par cluster.

Pour chaque cluster de la partition finale, on identifie les gènes
différentiellement exprimés avec **DESeq2** (via PyDESeq2), puis on fait un
**GSEA pré-classé** (gseapy) sur la statistique de Wald, selon deux schémas :

  - **one-vs-all** : cluster *c* contre toutes les autres tumeurs réunies ;
  - **one-vs-one** : chaque paire de clusters (*c*, *c'*).

On travaille sur les **counts bruts** (DESeq2 modélise la surdispersion des
comptages) ; les identifiants de gènes doivent être des **symboles HGNC** pour
matcher les gene sets GSEA (hallmarks MSigDB par défaut).

⚠️ **Double-dipping.** Les clusters sont définis à partir des mêmes données que
le test : les p-valeurs sont anticonservatives (Gao, Bien & Witten 2022). À lire
comme une **caractérisation** des programmes transcriptionnels de chaque groupe,
pas comme un test d'hypothèse valide. Pour de l'inférence, valider par
data-splitting ou sur une cohorte externe.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Briques : DESeq2 et GSEA sur un contraste
# --------------------------------------------------------------------------
def deseq2_contrast(counts: pd.DataFrame, groups: pd.Series,
                    target: str, ref: str, n_cpus: int | None = None) -> pd.DataFrame:
    """DESeq2 sur deux groupes. `counts` : tumeurs × gènes (counts entiers) ;
    `groups` : labels alignés sur `counts.index`. `n_cpus` borne le parallélisme
    interne de PyDESeq2 (mets 1 quand plusieurs contrastes tournent déjà en
    parallèle, pour ne pas sursouscrire les cœurs). Renvoie le tableau de
    résultats (index = gène : baseMean, log2FoldChange, stat, pvalue, padj)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    meta = pd.DataFrame({"group": groups.astype(str).to_numpy()}, index=counts.index)
    with contextlib.redirect_stdout(io.StringIO()):
        dds = DeseqDataSet(counts=counts, metadata=meta, design="~group",
                           n_cpus=n_cpus, quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["group", str(target), str(ref)],
                        n_cpus=n_cpus, quiet=True)
        st.summary()
    return st.results_df.sort_values("stat", ascending=False)


def gsea_prerank(results_df: pd.DataFrame, gene_sets: str | Path,
                 permutations: int = 1000, min_size: int = 15,
                 max_size: int = 500, threads: int = 4,
                 seed: int = 0) -> pd.DataFrame | None:
    """GSEA pré-classé sur la statistique de Wald DESeq2. Renvoie `res2d`
    (Term, NES, NOM p-val, FDR q-val, Lead_genes…) ou `None` si indisponible."""
    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy absent : GSEA sauté (`pip install gseapy`).")
        return None

    rnk = (results_df["stat"].dropna().sort_values(ascending=False)
           .rename_axis("gene").reset_index())
    rnk.columns = ["gene", "score"]
    if len(rnk) < min_size:
        return None
    gene_sets = os.path.expanduser(str(gene_sets))   # gère les chemins en ~/
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pre = gp.prerank(rnk=rnk, gene_sets=gene_sets, outdir=None,
                             min_size=min_size, max_size=max_size,
                             permutation_num=permutations, threads=threads,
                             seed=seed, no_plot=True, verbose=False)
        res = pre.res2d.copy()
        for col in ("NES", "ES", "NOM p-val", "FDR q-val"):
            if col in res:
                res[col] = pd.to_numeric(res[col], errors="coerce")
        return res
    except Exception as exc:  # gseapy lève sur données dégénérées
        logger.warning("GSEA échoué pour un contraste : %s", exc)
        return None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _run_one(cnt: pd.DataFrame, groups: pd.Series, target: str, ref: str,
             tag: str, scheme: str, de_dir: Path, gs_dir: Path, gene_sets: dict,
             permutations, min_count: int, threads: int,
             seed: int) -> tuple[str, str, dict]:
    """Un contraste : DESeq2 **une fois**, puis GSEA **pour chaque collection**
    de `gene_sets` ({nom: chemin .gmt}). Renvoie (tag, scheme, {nom: res2d|None}).
    Pensé pour un worker joblib indépendant ; `threads` borne PyDESeq2 et GSEA."""
    keep = cnt.columns[cnt.sum(axis=0) >= min_count]
    res = deseq2_contrast(cnt[keep], groups, target, ref, n_cpus=threads)
    res.to_csv(de_dir / f"deseq2_{tag}.csv", index_label="gene")

    gseas = {}
    for name, path in gene_sets.items():
        g = gsea_prerank(res, path, permutations=permutations,
                         threads=threads, seed=seed)
        if g is not None:
            g.to_csv(gs_dir / f"gsea_{name}_{tag}.csv", index=False)
        gseas[name] = g
    return tag, scheme, gseas


def run_degsea(
    counts: pd.DataFrame,
    labels: np.ndarray,
    sample_names: np.ndarray,
    outdir: Path,
    gene_sets,
    mode: str = "both",
    min_group: int = 3,
    min_count: int = 10,
    permutations: int = 1000,
    heatmap_pval: float = 0.05,
    subdir: str = "",
    n_jobs: int = -1,
    seed: int = 0,
) -> dict:
    """Lance DESeq2 + GSEA sur tous les contrastes demandés.

    Parameters
    ----------
    counts : matrice de counts **bruts**, tumeurs × gènes (symboles HGNC).
    labels : cluster de chaque tumeur, aligné sur `sample_names`.
    sample_names : ordre des tumeurs (partition finale).
    gene_sets : soit un chemin `.gmt` unique, soit un **dict {nom: chemin .gmt}**
        (une collection de gene sets par entrée). DESeq2 n'est calculé qu'une
        fois par contraste ; le GSEA est relancé pour chaque collection.
    mode : "ova" (one-vs-all), "ovo" (one-vs-one) ou "both".
    heatmap_pval : seuil de p-valeur nominale ; les matrices renvoyées ne gardent
        que les pathways significatifs (p < seuil) dans au moins un cluster.
    n_jobs : contrastes exécutés en parallèle (joblib, un contraste = une tâche).
        `1` = séquentiel ; chaque contraste reçoit alors plus de threads internes.

    Renvoie `{collection: matrice NES (pathways × clusters)}` (one-vs-all), pour
    les heatmaps de synthèse — dict vide si aucun résultat GSEA.
    """
    # normalisation gene_sets -> dict {nom: chemin}, en ne gardant que l'existant
    if isinstance(gene_sets, (str, Path)):
        gene_sets = {Path(gene_sets).stem: str(gene_sets)}
    resolved = {}
    for name, path in gene_sets.items():
        p = Path(os.path.expanduser(str(path)))
        if p.exists():
            resolved[name] = str(p)
        else:
            logger.warning("DEGSEA : gene set introuvable, ignoré : %s (%s)", name, p)
    gene_sets = resolved
    logger.info("DEGSEA : GSEA sur %d collection(s) : %s", len(gene_sets),
                ", ".join(gene_sets) or "aucune (DESeq2 seul)")

    base = Path(outdir) / "tables" / "degsea"
    if subdir:
        base = base / subdir      # une sous-arbo par k quand on balaie tous les k
    cnt = counts.loc[sample_names].round().astype(int)
    lab = pd.Series(np.asarray(labels), index=list(sample_names))

    sizes = lab.value_counts()
    clusters = sorted(c for c in sizes.index if sizes[c] >= min_group)
    dropped = sorted(c for c in sizes.index if sizes[c] < min_group)
    if dropped:
        logger.warning("DEGSEA : clusters ignorés (< %d tumeurs) : %s",
                       min_group, dropped)
    if len(clusters) < 2:
        logger.warning("DEGSEA : moins de 2 clusters exploitables, étape sautée.")
        return {}

    do_ova = mode in ("ova", "both")
    do_ovo = mode in ("ovo", "both")

    n_pairs = len(clusters) * (len(clusters) - 1) // 2 if do_ovo else 0
    n_contrasts = (len(clusters) if do_ova else 0) + n_pairs
    logger.info("DEGSEA : %d clusters -> %d contrastes one-vs-all + %d one-vs-one",
                len(clusters), len(clusters) if do_ova else 0, n_pairs)
    if n_pairs > 45:
        logger.warning("DEGSEA : %d paires one-vs-one, ça peut être long "
                       "(k élevé). Envisage degsea_mode=ova.", n_pairs)

    # n_jobs != 1 : le parallélisme se fait au niveau des contrastes (un worker
    # par contraste), donc chaque contraste est borné à 1 thread interne (DESeq2
    # ET GSEA) pour ne pas sursouscrire les cœurs. n_jobs == 1 : pas de
    # parallélisme externe, on donne alors tous les cœurs à chaque contraste.
    parallel_contrasts = n_jobs != 1 and n_contrasts > 1
    inner_threads = 1 if parallel_contrasts else (
        os.cpu_count() if n_jobs in (-1, 0, None) else max(1, int(n_jobs)))

    tasks = []
    ova_de = ovo_de = None
    if do_ova:
        ova_de = base / "ova"; ova_de.mkdir(parents=True, exist_ok=True)
        for c in clusters:
            groups = pd.Series(np.where(lab.values == c, f"c{c}", "rest"),
                               index=lab.index)
            tasks.append(dict(cnt=cnt, groups=groups, target=f"c{c}", ref="rest",
                              tag=f"c{c}_vs_rest", scheme="one-vs-all",
                              de_dir=ova_de, gs_dir=ova_de))
    if do_ovo:
        ovo_de = base / "ovo"; ovo_de.mkdir(parents=True, exist_ok=True)
        for a, b in combinations(clusters, 2):
            mask = lab.isin([a, b]).values
            groups = pd.Series([f"c{x}" for x in lab.values[mask]],
                               index=lab.index[mask])
            tasks.append(dict(cnt=cnt.loc[mask], groups=groups, target=f"c{a}",
                              ref=f"c{b}", tag=f"c{a}_vs_c{b}", scheme="one-vs-one",
                              de_dir=ovo_de, gs_dir=ovo_de))

    logger.info("DEGSEA : %d contrastes, %s (n_jobs=%s)", len(tasks),
               "en parallèle" if parallel_contrasts else "séquentiel", n_jobs)

    results = Parallel(n_jobs=n_jobs if parallel_contrasts else 1)(
        delayed(_run_one)(t["cnt"], t["groups"], t["target"], t["ref"], t["tag"],
                          t["scheme"], t["de_dir"], t["gs_dir"], gene_sets,
                          permutations, min_count, inner_threads, seed)
        for t in tasks
    )

    summary_rows: list[dict] = []
    ova_nes = defaultdict(dict)     # collection -> {cluster: Series(NES)}
    ova_fdr = defaultdict(dict)     # collection -> {cluster: Series(FDR q-val du GSEA)}
    for tag, scheme, gseas in results:
        cluster_id = tag.split("_vs_rest")[0] if scheme == "one-vs-all" else None
        for coll, gsea in gseas.items():
            if gsea is None:
                continue
            summary_rows += _collect(gsea, coll, tag, scheme)
            if cluster_id is not None:
                g = gsea.set_index("Term")
                ova_nes[coll][cluster_id] = g["NES"]
                ova_fdr[coll][cluster_id] = g["FDR q-val"]   # FDR (permutations GSEA), pas p brute

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(base / "gsea_summary.csv", index=False)
        logger.info("DEGSEA : synthèse GSEA (%d collections) -> %s",
                    len({r['collection'] for r in summary_rows}), base / "gsea_summary.csv")

    nes_by_collection: dict = {}
    for coll in ova_nes:
        m = _nes_matrix(ova_nes[coll], ova_fdr[coll], heatmap_pval)
        if not m.empty:
            nes_by_collection[coll] = m
            logger.info("DEGSEA [%s] : %d pathways significatifs (FDR q < %.3g).",
                        coll, len(m), heatmap_pval)
    return nes_by_collection


def _collect(res2d: pd.DataFrame, collection: str, contrast: str, scheme: str,
             fdr_max: float = 0.25) -> list[dict]:
    """Lignes de synthèse : pathways significatifs (FDR < seuil) d'un contraste."""
    sig = res2d[res2d["FDR q-val"] < fdr_max]
    return [{"collection": collection, "contrast": contrast, "scheme": scheme,
             "term": r["Term"], "NES": r["NES"], "NOM_pval": r.get("NOM p-val"),
             "FDR": r["FDR q-val"], "lead_genes": r.get("Lead_genes")}
            for _, r in sig.iterrows()]


def _nes_matrix(ova_nes: dict, ova_fdr: dict, fdr_max: float = 0.05) -> pd.DataFrame:
    """Matrice pathways × clusters (NES one-vs-all) pour la heatmap.

    Ne garde que les pathways **significatifs après correction** (FDR q-valeur du
    GSEA, issue des permutations, `< fdr_max`) dans au moins un cluster, triés par
    |NES| max décroissant.
    """
    nes = pd.DataFrame(ova_nes)
    fdr = pd.DataFrame(ova_fdr).reindex(index=nes.index, columns=nes.columns)
    sig = (fdr < fdr_max).any(axis=1)
    kept = nes.loc[sig]
    order = kept.abs().max(axis=1).sort_values(ascending=False).index
    return kept.loc[order]
