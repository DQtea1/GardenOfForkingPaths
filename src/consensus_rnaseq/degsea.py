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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Briques : DESeq2 et GSEA sur un contraste
# --------------------------------------------------------------------------
def deseq2_contrast(counts: pd.DataFrame, groups: pd.Series,
                    target: str, ref: str) -> pd.DataFrame:
    """DESeq2 sur deux groupes. `counts` : tumeurs × gènes (counts entiers) ;
    `groups` : labels alignés sur `counts.index`. Renvoie le tableau de
    résultats (index = gène : baseMean, log2FoldChange, stat, pvalue, padj)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    meta = pd.DataFrame({"group": groups.astype(str).to_numpy()}, index=counts.index)
    with contextlib.redirect_stdout(io.StringIO()):
        dds = DeseqDataSet(counts=counts, metadata=meta, design="~group", quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["group", str(target), str(ref)], quiet=True)
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
             tag: str, de_dir: Path, gs_dir: Path, gene_sets, permutations,
             min_count: int, threads: int, seed: int) -> pd.DataFrame | None:
    """Un contraste : DESeq2 + GSEA, écriture des deux tables. Renvoie le
    res2d GSEA (ou None)."""
    keep = cnt.columns[cnt.sum(axis=0) >= min_count]
    res = deseq2_contrast(cnt[keep], groups, target, ref)
    res.to_csv(de_dir / f"deseq2_{tag}.csv", index_label="gene")

    gsea = gsea_prerank(res, gene_sets, permutations=permutations,
                        threads=threads, seed=seed)
    if gsea is not None:
        gsea.to_csv(gs_dir / f"gsea_{tag}.csv", index=False)
    return gsea


def run_degsea(
    counts: pd.DataFrame,
    labels: np.ndarray,
    sample_names: np.ndarray,
    outdir: Path,
    gene_sets: str | Path,
    mode: str = "both",
    min_group: int = 3,
    min_count: int = 10,
    permutations: int = 1000,
    n_jobs: int = -1,
    seed: int = 0,
) -> pd.DataFrame | None:
    """Lance DESeq2 + GSEA sur tous les contrastes demandés.

    Parameters
    ----------
    counts : matrice de counts **bruts**, tumeurs × gènes (symboles HGNC).
    labels : cluster de chaque tumeur, aligné sur `sample_names`.
    sample_names : ordre des tumeurs (partition finale).
    gene_sets : chemin d'un fichier .gmt (hallmarks MSigDB par défaut).
    mode : "ova" (one-vs-all), "ovo" (one-vs-one) ou "both".

    Renvoie une matrice NES (pathways × clusters) issue du one-vs-all pour la
    heatmap de synthèse, ou None.
    """
    base = Path(outdir) / "tables" / "degsea"
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
        return None

    threads = os.cpu_count() if n_jobs in (-1, 0, None) else max(1, int(n_jobs))
    do_ova = mode in ("ova", "both")
    do_ovo = mode in ("ovo", "both")

    n_pairs = len(clusters) * (len(clusters) - 1) // 2 if do_ovo else 0
    logger.info("DEGSEA : %d clusters -> %d contrastes one-vs-all + %d one-vs-one",
                len(clusters), len(clusters) if do_ova else 0, n_pairs)
    if n_pairs > 45:
        logger.warning("DEGSEA : %d paires one-vs-one, ça peut être long "
                       "(k élevé). Envisage degsea_mode=ova.", n_pairs)

    summary_rows: list[dict] = []
    ova_nes: dict = {}

    if do_ova:
        ova_de = base / "ova"; ova_de.mkdir(parents=True, exist_ok=True)
        for c in clusters:
            tag = f"c{c}_vs_rest"
            logger.info("DEGSEA one-vs-all : cluster %s (%d tumeurs)", c, sizes[c])
            groups = pd.Series(np.where(lab.values == c, f"c{c}", "rest"),
                               index=lab.index)
            gsea = _run_one(cnt, groups, f"c{c}", "rest", tag, ova_de, ova_de,
                            gene_sets, permutations, min_count, threads, seed)
            if gsea is not None:
                ova_nes[f"c{c}"] = gsea.set_index("Term")["NES"]
                summary_rows += _collect(gsea, tag, "one-vs-all")

    if do_ovo:
        ovo_de = base / "ovo"; ovo_de.mkdir(parents=True, exist_ok=True)
        for a, b in combinations(clusters, 2):
            tag = f"c{a}_vs_c{b}"
            logger.info("DEGSEA one-vs-one : %s vs %s", a, b)
            mask = lab.isin([a, b]).values
            groups = pd.Series([f"c{x}" for x in lab.values[mask]],
                               index=lab.index[mask])
            gsea = _run_one(cnt.loc[mask], groups, f"c{a}", f"c{b}", tag,
                            ovo_de, ovo_de, gene_sets, permutations,
                            min_count, threads, seed)
            if gsea is not None:
                summary_rows += _collect(gsea, tag, "one-vs-one")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(base / "gsea_summary.csv", index=False)
        logger.info("DEGSEA : synthèse GSEA -> %s", base / "gsea_summary.csv")

    return _nes_matrix(ova_nes) if ova_nes else None


def _collect(res2d: pd.DataFrame, contrast: str, scheme: str,
             fdr_max: float = 0.25) -> list[dict]:
    """Lignes de synthèse : pathways significatifs (FDR < seuil) d'un contraste."""
    sig = res2d[res2d["FDR q-val"] < fdr_max]
    return [{"contrast": contrast, "scheme": scheme, "term": r["Term"],
             "NES": r["NES"], "NOM_pval": r.get("NOM p-val"),
             "FDR": r["FDR q-val"], "lead_genes": r.get("Lead_genes")}
            for _, r in sig.iterrows()]


def _nes_matrix(ova_nes: dict, fdr_source: dict | None = None,
                top: int = 25) -> pd.DataFrame:
    """Matrice pathways × clusters (NES one-vs-all), restreinte aux `top`
    pathways les plus marqués, pour la heatmap de synthèse."""
    nes = pd.DataFrame(ova_nes)
    order = nes.abs().max(axis=1).sort_values(ascending=False).index[:top]
    return nes.loc[order]
