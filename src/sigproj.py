"""Étape 7 — Projection de signatures : scoring par patient + association clinique.

7.1  Score chaque signature (gene set) pour chaque tumeur, de **deux façons** :
       - **ssGSEA** (gseapy) : enrichissement rang-based par échantillon (NES) ;
       - **expression moyenne** : moyenne des z-scores (par gène) des gènes de
         la signature présents dans la matrice.

7.2  Teste l'**association** score de signature ↔ variable clinique (métadonnées) :
       - variable **catégorielle** : test de Wilcoxon rang-somme (Mann-Whitney)
         entre chaque paire de modalités ;
       - variable **continue** : corrélation (Spearman par défaut — robuste,
         monotone ; Pearson en option).
     Correction BH (FDR) des p-valeurs, séparément par méthode de score.

7.2bis  Tests de Wilcoxon *one-vs-rest* (score de signature vs le reste) pour
     chaque modalité de chaque stratification affichée dans le rapport — le
     **cluster** (pour **chaque k**) et chaque **variable clinique catégorielle**.
     Sert aux **étoiles de significativité** au-dessus des boxplots de l'onglet
     « Signatures détaillé » (`stratified_signature_tests`).

7.3  Figures, par variable clinique catégorielle et pour chaque méthode :
       - **boxplots** des top signatures qui séparent le mieux les modalités ;
       - **heatmap** de l'activation des signatures par échantillon.

Comme dans DEGSEA, les gènes doivent être des **symboles HGNC**. Le scoring se
fait sur une matrice d'expression **normalisée, tous gènes** (logCPM), pour que
les gènes des signatures soient présents (≠ matrice top-variable du clustering).
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
# Chargement des signatures
# --------------------------------------------------------------------------
def load_signatures(gmt_path: str | Path) -> dict[str, list[str]]:
    """Charge un `.gmt` -> {signature: [gènes]} (symboles HGNC)."""
    from gseapy.parser import read_gmt
    return read_gmt(str(Path(os.path.expanduser(str(gmt_path)))))


def _load_one_source(name: str, spec: dict) -> tuple[dict, dict]:
    """Charge **une** source de signatures selon son `format`. Renvoie
    ({signature: [gènes]}, {signature: détail}). Formats gérés :

    - ``gmt``  : fichier .gmt standard ;
    - ``csv`` / ``tsv`` : une signature par ligne, colonnes configurables
      (`name_col`, `genes_col`, `detail_col`) et séparateur de gènes `genes_sep`.

    C'est ce qui **harmonise** des sources hétérogènes (chacune sa nomenclature)
    vers une représentation commune {nom: gènes}.
    """
    fmt = str(spec.get("format", "gmt")).lower()
    path = Path(os.path.expanduser(str(spec["path"])))
    if not path.exists():
        logger.warning("Signatures : source '%s' introuvable, ignorée : %s", name, path)
        return {}, {}

    if fmt == "gmt":
        sigs = load_signatures(path)
        return sigs, {k: "" for k in sigs}

    if fmt in ("csv", "tsv"):
        sep = "\t" if fmt == "tsv" else spec.get("sep", ",")
        df = pd.read_csv(path, sep=sep)
        name_col = spec.get("name_col", "Geneset")
        genes_col = spec.get("genes_col", "Gene Listing")
        detail_col = spec.get("detail_col")
        genes_sep = spec.get("genes_sep", ";")
        for c in (name_col, genes_col):
            if c not in df.columns:
                logger.warning("Signatures : colonne '%s' absente de %s (%s) — source "
                               "ignorée. Colonnes : %s", c, name, path, list(df.columns))
                return {}, {}
        sigs, details = {}, {}
        for _, row in df.iterrows():
            nm = str(row[name_col]).strip()
            genes = [g.strip() for g in str(row[genes_col]).split(genes_sep) if g.strip()]
            if nm and genes:
                sigs[nm] = genes
                details[nm] = (str(row[detail_col])
                               if detail_col and detail_col in df.columns else "")
        return sigs, details

    logger.warning("Signatures : format inconnu '%s' pour la source '%s' — ignorée.",
                   fmt, name)
    return {}, {}


def load_signature_sources(sources: dict) -> tuple[dict, pd.DataFrame]:
    """Charge et **harmonise** plusieurs sources de signatures hétérogènes.

    `sources` : {nom_source: {format, path, ...}}. Renvoie
    ({signature: [gènes]}, table de provenance). En cas de collision de nom
    entre sources, la signature est préfixée par le nom de source ; la table de
    provenance garde la trace complète (source, nb de gènes, détail)."""
    combined: dict = {}
    prov = []
    for src_name, spec in sources.items():
        sigs, details = _load_one_source(src_name, spec)
        for nm, genes in sigs.items():
            key = nm if nm not in combined else f"{src_name}:{nm}"
            combined[key] = genes
            prov.append({"signature": key, "source": src_name,
                         "n_genes": len(genes), "detail": details.get(nm, "")})
        if sigs:
            logger.info("Signatures : source '%s' (%s) -> %d signatures",
                        src_name, spec.get("format", "gmt"), len(sigs))
    return combined, pd.DataFrame(prov)


# --------------------------------------------------------------------------
# 7.1  Scoring
# --------------------------------------------------------------------------
def score_ssgsea(expr_genes_x_samples: pd.DataFrame, signatures: dict,
                 min_size: int = 3, threads: int = 4, seed: int = 0) -> pd.DataFrame:
    """ssGSEA (gseapy) : renvoie une matrice **signatures × échantillons** (NES)."""
    import gseapy as gp
    if any(not isinstance(g, str) for g in expr_genes_x_samples.index):
        expr_genes_x_samples = expr_genes_x_samples.copy()
        expr_genes_x_samples.index = expr_genes_x_samples.index.astype(str)
    with contextlib.redirect_stdout(io.StringIO()):
        ss = gp.ssgsea(data=expr_genes_x_samples, gene_sets=signatures, outdir=None,
                       sample_norm_method="rank", min_size=min_size,
                       threads=threads, seed=seed, no_plot=True, verbose=False)
    df = ss.res2d.copy()
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    mat = df.pivot(index="Term", columns="Name", values="NES")
    return mat.reindex(columns=expr_genes_x_samples.columns)


def score_mean_expression(expr_samples_x_genes: pd.DataFrame,
                          signatures: dict) -> pd.DataFrame:
    """Score = moyenne des **z-scores** (par gène, à travers les tumeurs) des
    gènes de la signature présents. Renvoie **signatures × échantillons**."""
    X = expr_samples_x_genes
    sd = X.std(axis=0).replace(0, 1.0)
    z = (X - X.mean(axis=0)) / sd                      # z-score par gène
    rows = {}
    for sig, genes in signatures.items():
        present = [g for g in genes if g in z.columns]
        if present:
            rows[sig] = z[present].mean(axis=1)
    return pd.DataFrame(rows).T.reindex(columns=X.index)


# --------------------------------------------------------------------------
# 7.2  Association aux variables cliniques
# --------------------------------------------------------------------------
def _bh(pvals: np.ndarray) -> np.ndarray:
    """Correction Benjamini-Hochberg (FDR) -> q-valeurs."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _is_categorical(series: pd.Series, max_levels: int = 6) -> bool | None:
    """True = catégorielle, False = continue, None = inexploitable (vide)."""
    s = series.dropna()
    if s.empty:
        return None
    if not pd.api.types.is_numeric_dtype(s):
        return True
    return s.nunique() <= max_levels        # numérique à peu de niveaux -> catégoriel


def associate(scores: pd.DataFrame, metadata: pd.DataFrame,
              corr_method: str = "spearman", min_group: int = 3,
              max_levels: int = 6) -> pd.DataFrame:
    """Teste chaque (signature × variable). `scores` : signatures × échantillons ;
    `metadata` : échantillons × variables (aligné). Renvoie une table longue."""
    from scipy.stats import mannwhitneyu, pearsonr, spearmanr

    meta = metadata.reindex(scores.columns)
    rows = []
    for var in meta.columns:
        col = meta[var]
        kind = _is_categorical(col, max_levels)
        if kind is None:
            continue

        if kind:  # ---- catégorielle : Wilcoxon (Mann-Whitney) par paire ----
            counts = col.value_counts()
            levels = [m for m in counts.index if counts[m] >= min_group]
            if len(levels) < 2:
                continue
            for sig in scores.index:
                sc = scores.loc[sig]
                for a, b in combinations(levels, 2):
                    xa = sc[col.index[col == a]].dropna()
                    xb = sc[col.index[col == b]].dropna()
                    if len(xa) < min_group or len(xb) < min_group:
                        continue
                    try:
                        stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
                    except ValueError:
                        continue
                    rbc = 1.0 - 2.0 * stat / (len(xa) * len(xb))   # rank-biserial
                    rows.append(dict(signature=sig, variable=str(var),
                                     var_type="categorical",
                                     comparison=f"{a} vs {b}", test="mannwhitney",
                                     n1=len(xa), n2=len(xb), statistic=float(stat),
                                     effect=float(rbc),
                                     median_diff=float(xa.median() - xb.median()),
                                     pvalue=float(p)))

        else:  # ---- continue : corrélation ----
            x = pd.to_numeric(col, errors="coerce")
            fn = pearsonr if corr_method == "pearson" else spearmanr
            for sig in scores.index:
                d = pd.concat([scores.loc[sig], x], axis=1).dropna()
                if len(d) < 4:
                    continue
                r, p = fn(d.iloc[:, 0], d.iloc[:, 1])
                rows.append(dict(signature=sig, variable=str(var),
                                 var_type="continuous", comparison=corr_method,
                                 test=corr_method, n1=len(d), n2=np.nan,
                                 statistic=float(r), effect=float(r),
                                 median_diff=np.nan, pvalue=float(p)))

    df = pd.DataFrame(rows)
    if len(df):
        df["padj"] = _bh(df["pvalue"].to_numpy())
        df = df.sort_values("pvalue").reset_index(drop=True)
    return df


def _stars(p) -> str:
    """Notation étoilée d'une p-valeur : *** < 0.001, ** < 0.01, * < 0.05."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def group_pvals(score_row, groups, min_group: int = 3) -> dict:
    """Test de Wilcoxon rang-somme (Mann-Whitney) **une modalité contre le reste**
    pour chaque modalité. `score_row` est aligné sur `groups` (labels de groupe,
    `None` = valeur manquante). Renvoie `{modalité: p-valeur | None}` — `None` si
    l'un des deux échantillons a moins de `min_group` tumeurs (test non fiable)."""
    from scipy.stats import mannwhitneyu

    x = np.asarray(score_row, dtype=float)
    g = np.asarray([None if v is None else str(v) for v in groups], dtype=object)
    keep = np.array([(gi is not None) and np.isfinite(xi) for gi, xi in zip(g, x)])
    x, g = x[keep], g[keep]
    out = {}
    for lvl in pd.unique(g):
        a, b = x[g == lvl], x[g != lvl]
        if len(a) < min_group or len(b) < min_group:
            out[str(lvl)] = None
            continue
        try:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            p = None
        out[str(lvl)] = p if (p is not None and np.isfinite(p)) else None
    return out


def pair_pvals(score_row, groups, min_group: int = 3) -> dict:
    """Test de Wilcoxon rang-somme (Mann-Whitney) **entre chaque paire de
    modalités**. Renvoie un dict imbriqué `{modalité_a: {modalité_b: p}}` — chaque
    paire une seule fois (`a` avant `b` dans l'ordre d'apparition) ; paires trop
    petites (< `min_group` de part ou d'autre) omises."""
    from scipy.stats import mannwhitneyu

    x = np.asarray(score_row, dtype=float)
    g = np.asarray([None if v is None else str(v) for v in groups], dtype=object)
    keep = np.array([(gi is not None) and np.isfinite(xi) for gi, xi in zip(g, x)])
    x, g = x[keep], g[keep]
    levels = list(pd.unique(g))
    out: dict = {}
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a, b = x[g == levels[i]], x[g == levels[j]]
            if len(a) < min_group or len(b) < min_group:
                continue
            try:
                p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except ValueError:
                continue
            if np.isfinite(p):
                out.setdefault(str(levels[i]), {})[str(levels[j])] = p
    return out


def stratified_signature_tests(scores: dict, cluster_labels_by_k: dict,
                               metadata: pd.DataFrame | None,
                               max_levels: int = 6, min_group: int = 3):
    """Tests de Wilcoxon (Mann-Whitney) des scores de signature entre modalités,
    pour **chaque stratification affichée dans le rapport** :

      - le **cluster** (partition consensus) pour **chaque k** ;
      - chaque **variable clinique catégorielle** (indépendante de k).

    Deux familles de tests par (signature, stratification) : **one-vs-rest** (une
    modalité contre le reste) et **pairwise** (chaque paire de modalités).

    `scores` : `{method: DataFrame signatures × tumeurs}` (colonnes = ordre des
    tumeurs). `cluster_labels_by_k` : `{k: labels}` alignés sur ces colonnes.

    Renvoie `(tests, tidy_df)` avec ``tests[method][stratKey][kkey][signature] =
    {"ovr": {modalité: p}, "pair": {a: {b: p}}}`` où ``stratKey`` vaut
    ``"__cluster__"`` (alors ``kkey = str(k)``) ou le nom de la variable (alors
    ``kkey = "*"``). Les **clés de modalité sont exactement celles des boxplots** :
    ``"C<label>"`` pour les clusters, ``str(valeur)`` pour les variables cliniques.
    """
    methods = list(scores)
    tests = {m: {"__cluster__": {}} for m in methods}
    tidy = []

    def _fill(mat, groups, m, stratify_by, kval):
        d = {}
        for sig in mat.index:
            row = mat.loc[sig].to_numpy()
            ovr = group_pvals(row, groups, min_group)
            pair = pair_pvals(row, groups, min_group)
            d[str(sig)] = {"ovr": ovr, "pair": pair}
            for grp, p in ovr.items():
                tidy.append(dict(method=m, stratify_by=stratify_by, k=kval,
                                 signature=str(sig), test_type="one_vs_rest",
                                 comparison=grp, pvalue=p, stars=_stars(p)))
            for a, inner in pair.items():
                for b, p in inner.items():
                    tidy.append(dict(method=m, stratify_by=stratify_by, k=kval,
                                     signature=str(sig), test_type="pairwise",
                                     comparison=f"{a} vs {b}", pvalue=p, stars=_stars(p)))
        return d

    # ---- clusters : un test par k
    for k in sorted(cluster_labels_by_k):
        groups = ["C" + str(int(l)) for l in cluster_labels_by_k[k]]
        for m in methods:
            tests[m]["__cluster__"][str(int(k))] = _fill(
                scores[m], groups, m, "cluster", int(k))

    # ---- variables cliniques catégorielles : indépendantes de k (clé "*")
    if metadata is not None and metadata.shape[1]:
        meta = metadata.reindex(scores[methods[0]].columns)
        for var in meta.columns:
            col = meta[var]
            if _is_categorical(col, max_levels) is not True:
                continue
            if sum(col.value_counts() >= min_group) < 2:  # < 2 modalités exploitables
                continue
            groups = [None if pd.isna(v) else str(v) for v in col]
            for m in methods:
                tests[m].setdefault(str(var), {})["*"] = _fill(
                    scores[m], groups, m, str(var), None)

    return tests, pd.DataFrame(tidy)


def top_signatures(assoc: pd.DataFrame, variable: str, top_n: int,
                   padj_max: float) -> list[str]:
    """Signatures les plus significativement associées à `variable` (min padj),
    limitées à celles sous le seuil, jusqu'à `top_n`."""
    sub = assoc[assoc["variable"] == variable]
    if sub.empty:
        return []
    best = sub.groupby("signature")["padj"].min().sort_values()
    sig = best[best <= padj_max]
    return list((sig if len(sig) else best).index[:top_n])


# --------------------------------------------------------------------------
# Orchestration 7.1 -> 7.3
# --------------------------------------------------------------------------
def run_signature_projection(
    expr: pd.DataFrame,
    signatures: dict,
    metadata: pd.DataFrame | None,
    outdir: Path,
    corr_method: str = "spearman",
    top_n: int = 8,
    sig_pval: float = 0.05,
    min_group: int = 3,
    n_jobs: int = -1,
    seed: int = 0,
) -> dict:
    """`expr` : tumeurs × gènes normalisé (tous gènes). `metadata` : tumeurs ×
    variables cliniques (ou None -> seul le scoring 7.1 est fait). Renvoie
    `{"ssgsea": df, "mean": df}` (signatures × tumeurs)."""
    import plots as pl

    sig_dir = Path(outdir) / "tables" / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)
    figs = Path(outdir) / "figures"
    threads = os.cpu_count() if n_jobs in (-1, 0, None) else max(1, int(n_jobs))

    if len(signatures) > 500:
        logger.warning("Projection : %d signatures — le ssGSEA peut être long "
                       "(collection volumineuse ?).", len(signatures))

    # 7.1  scoring (deux méthodes)
    logger.info("Projection 7.1 : scoring de %d signatures (ssGSEA + expression moyenne)…",
                len(signatures))
    scores = {
        "ssgsea": score_ssgsea(expr.T, signatures, threads=threads, seed=seed),
        "mean": score_mean_expression(expr, signatures),
    }
    for method, mat in scores.items():
        mat.to_csv(sig_dir / f"scores_{method}.csv")
    logger.info("Projection 7.1 : scores sauvegardés (%d signatures scorées).",
                scores["ssgsea"].shape[0])

    if metadata is None or metadata.shape[1] == 0:
        logger.info("Projection : pas de métadonnées -> association (7.2) et "
                    "figures (7.3) sautées.")
        return scores

    # 7.2  association + 7.3  figures, pour chaque méthode
    for method, mat in scores.items():
        assoc = associate(mat, metadata, corr_method=corr_method, min_group=min_group)
        assoc.to_csv(sig_dir / f"association_{method}.csv", index=False)
        n_sig = int((assoc["padj"] <= sig_pval).sum()) if len(assoc) else 0
        logger.info("Projection 7.2 [%s] : %d tests, %d significatifs (FDR <= %.3g).",
                    method, len(assoc), n_sig, sig_pval)

        cat_vars = (assoc.loc[assoc["var_type"] == "categorical", "variable"].unique()
                    if len(assoc) else [])
        for var in cat_vars:
            top = top_signatures(assoc, var, top_n, sig_pval)
            if not top:
                continue
            pl.plot_signature_boxplots(mat, metadata[var], top, var, method,
                                       assoc, figs)
            pl.plot_signature_heatmap(mat, metadata[var], top, var, method, figs)
        logger.info("Projection 7.3 [%s] : figures par variable -> %s", method, figs)
    return scores
