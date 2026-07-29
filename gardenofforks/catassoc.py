"""Étape 9a — Association entre variables catégorielles (khi², Fisher, résidus).

Pour chaque paire de variables catégorielles — la **partition en clusters** (pour
chaque k) croisée avec chaque variable clinique, et les variables cliniques entre
elles — on construit le **tableau de contingence** (avec marges), on teste
l'**indépendance** par un **khi² de Pearson**, et, si les conditions de Cochran ne
sont pas réunies (> 20 % de cellules à effectif attendu < 5, ou un attendu < 1), on
bascule vers :

  - un **test exact de Fisher** pour les tables **2×2** ;
  - un **khi² de Monte-Carlo par permutation** pour les tables **R×C** (le Fisher
    R×C / Freeman-Halton n'existe pas dans scipy ; la permutation en est
    l'équivalent robuste, façon ``simulate.p.value`` de R : on permute une des
    deux étiquettes, ce qui fixe exactement les marges, et on compte la fraction
    de permutations dont le khi² dépasse l'observé).

On rapporte aussi la taille d'effet (**V de Cramér**) et surtout les **résidus
standardisés ajustés** (Haberman) : sous indépendance ils suivent ~ N(0,1), donc
|résidu| > 1.96 (resp. 2.58) marque une modalité **sur- ou sous-représentée** au
seuil 5 % (resp. 1 %) — c'est ce qui dit *quelles* combinaisons portent
l'association. Correction **BH (FDR)** sur l'ensemble des paires testées.

Pour les variables **ordinales** déclarées dans le YAML (stade, grade…), un test
de tendance linéaire-par-linéaire — Cochran-Armitage dans le cas binaire ×
ordinal — remplace le khi² nominal lorsque les conditions asymptotiques sont
satisfaites. Si elles ne le sont pas, le repli exact/permutation reste
prioritaire : Fisher pour une table 2×2, Monte-Carlo pour une table R×C.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from .stats_utils import benjamini_hochberg as _bh
from .stats_utils import is_categorical as _is_categorical

logger = logging.getLogger(__name__)


def _chi2_stat(obs: np.ndarray) -> float:
    """Statistique du khi² de Pearson d'une table (sans correction)."""
    n = obs.sum()
    if n == 0:
        return 0.0
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0)
    return float(term.sum())


def _mc_pvalue(ai: np.ndarray, bi: np.ndarray, R: int, C: int,
               obs_stat: float, n_resamples: int, seed: int) -> float:
    """p-valeur de Monte-Carlo par permutation (khi² à marges fixées)."""
    rng = np.random.default_rng(seed)
    b = bi.copy()
    ge = 1  # +1 (Davison-Hinkley) : la table observée compte comme une permutation
    for _ in range(n_resamples):
        rng.shuffle(b)
        tab = np.zeros((R, C))
        np.add.at(tab, (ai, b), 1)
        if _chi2_stat(tab) >= obs_stat - 1e-9:
            ge += 1
    return ge / (n_resamples + 1)


def _axis_scores(categories, order):
    """Scores numériques d'un axe pour le test de tendance.

    `order` = liste ordonnée des modalités (variable **ordinale**) -> rang de
    chaque catégorie ; `None` = variable nominale -> scores 0/1 seulement si
    l'axe est **binaire** (2 modalités), sinon non « scorable » -> renvoie None.
    """
    cats = [str(x) for x in categories]
    if order is not None:
        pos = {str(m): i for i, m in enumerate(order)}
        if not all(c in pos for c in cats):
            return None          # l'ordre déclaré ne couvre pas les données
        return np.array([pos[c] for c in cats], dtype=float)
    if len(cats) == 2:
        return np.array([0.0, 1.0])
    return None                  # nominale à > 2 modalités : pas d'ordre


def _trend_test(obs, u, v):
    """Test de tendance linéaire-par-linéaire (Mantel) : M² = (N-1)·r² ~ χ²(1).

    Cas particulier binaire × ordinal = **Cochran-Armitage**. `u`/`v` : scores
    des lignes/colonnes. Renvoie (M², p, r) ; r>0 = tendance croissante."""
    from scipy.stats import chi2 as chi2dist
    N = obs.sum()
    ru, cv = obs.sum(1), obs.sum(0)
    Su, Sv = float((u * ru).sum()), float((v * cv).sum())
    Suv = float((u[:, None] * v[None, :] * obs).sum())
    cov = Suv - Su * Sv / N
    varu = float((u ** 2 * ru).sum()) - Su ** 2 / N
    varv = float((v ** 2 * cv).sum()) - Sv ** 2 / N
    if varu <= 0 or varv <= 0:
        return 0.0, 1.0, 0.0
    r = cov / np.sqrt(varu * varv)
    m2 = (N - 1) * r * r
    return float(m2), float(chi2dist.sf(m2, 1)), float(r)


def association_test(a: pd.Series, b: pd.Series, min_expected: int = 5,
                     max_lowexp_frac: float = 0.2, mc_resamples: int = 2000,
                     seed: int = 0, a_order=None, b_order=None) -> dict | None:
    """Test d'indépendance de deux variables catégorielles alignées.

    L'ordre de décision protège d'abord contre les faibles effectifs attendus :
    Fisher pour une table 2×2, Monte-Carlo pour une table R×C. Lorsque les
    conditions de Cochran sont satisfaites, une variable **ordinale** déclarée
    (avec deux axes scorables) déclenche le test de tendance
    linéaire-par-linéaire / Cochran-Armitage ; sinon on utilise Pearson.

    Renvoie un dict (contingence, attendus, résidus ajustés, test utilisé, p, V de
    Cramér, diagnostic) ou ``None`` si une variable a < 2 modalités présentes.
    """
    df = pd.DataFrame({"a": np.asarray(a, dtype=object),
                       "b": np.asarray(b, dtype=object)}).dropna()
    if df.empty:
        return None
    tab = pd.crosstab(df["a"], df["b"])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None

    obs = tab.values.astype(float)
    n = obs.sum()
    chi2stat, p_chi2, dof, expected = chi2_contingency(obs, correction=False)
    frac_low = float((expected < min_expected).mean())
    min_exp = float(expected.min())
    conditions = (min_exp >= 1.0) and (frac_low <= max_lowexp_frac)
    r, c = obs.shape

    # test de tendance si au moins une variable ordinale et les deux axes scorables
    u = _axis_scores(tab.index, a_order)
    v = _axis_scores(tab.columns, b_order)
    trend_ok = ((a_order is not None) or (b_order is not None)) and (u is not None) and (v is not None)
    trend_r = None

    # Les replis exact/permutation ont priorité sur tout test asymptotique,
    # y compris le test de tendance. Auparavant, ``trend_ok`` court-circuitait
    # Fisher sur une table 2×2 clairsemée déclarée ordinale.
    if not conditions and r == 2 and c == 2:
        _, p = fisher_exact(obs)
        used, p = "fisher_exact", float(p)
    elif not conditions:
        ai = pd.Categorical(df["a"]).codes
        bi = pd.Categorical(df["b"]).codes
        p = _mc_pvalue(ai, bi, r, c, float(chi2stat), mc_resamples, seed)
        used = "chi2_montecarlo"
    elif trend_ok:
        m2, p, trend_r = _trend_test(obs, u, v)
        used, chi2stat = "trend", m2
    else:
        used, p = "chi2", float(p_chi2)

    # V de Cramér (toujours à partir du khi² de Pearson, comme taille d'effet)
    k = min(r - 1, c - 1)
    pearson_chi2 = chi2_contingency(obs, correction=False)[0]
    cramers_v = float(np.sqrt(pearson_chi2 / (n * k))) if (n > 0 and k > 0) else np.nan

    # résidus standardisés ajustés (Haberman) ~ N(0,1) sous indépendance
    row, col = obs.sum(1, keepdims=True), obs.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        adj = (obs - expected) / np.sqrt(expected * (1 - row / n) * (1 - col / n))
    adj = np.where(np.isfinite(adj), adj, 0.0)

    return {
        "table": tab,
        "expected": pd.DataFrame(expected, index=tab.index, columns=tab.columns),
        "adj_residuals": pd.DataFrame(adj, index=tab.index, columns=tab.columns),
        "test": used, "chi2": float(chi2stat), "dof": int(dof),
        "pvalue": float(p), "cramers_v": cramers_v, "n": int(n),
        "conditions_met": bool(conditions), "trend_r": trend_r,
        "min_expected": min_exp, "frac_cells_low_expected": frac_low,
        "expected_threshold": float(min_expected),
        "max_low_expected_fraction": float(max_lowexp_frac),
        "mc_resamples": int(mc_resamples),
    }


def _with_margins(tab: pd.DataFrame) -> pd.DataFrame:
    """Tableau d'effectifs + marges (ligne/colonne 'Total')."""
    m = tab.copy()
    m.loc["Total"] = tab.sum(0)
    m["Total"] = m.sum(1)
    return m


def _save_detail(detail_dir: Path, name: str, res: dict) -> None:
    detail_dir.mkdir(parents=True, exist_ok=True)
    _with_margins(res["table"]).to_csv(detail_dir / f"{name}__counts.csv")
    res["adj_residuals"].round(3).to_csv(detail_dir / f"{name}__adjresiduals.csv")


def _tidy_row(v1, v2, k, res):
    is_trend = res["test"] == "trend"
    return {
        "var1": v1, "var2": v2, "k": k, "test": res["test"],
        "statistic": round(res["chi2"], 4),          # khi² de Pearson, ou M² si tendance
        "dof": 1 if is_trend else res["dof"], "n": res["n"],
        "cramers_v": round(res["cramers_v"], 4) if np.isfinite(res["cramers_v"]) else np.nan,
        "trend_r": res.get("trend_r"),
        "pvalue": res["pvalue"], "conditions_met": res["conditions_met"],
        "min_expected": round(res["min_expected"], 2),
        "frac_cells_low_expected": round(res["frac_cells_low_expected"], 3),
    }


DUP_WARNING = (
    "Le khi² suppose l'indépendance des observations : 1 tumeur = 1 patient. "
    "Si un patient a plusieurs prélèvements (primaire + métastase, lésions "
    "multiples…), il est compté plusieurs fois et les p-valeurs deviennent "
    "anticonservatives (trop optimistes). Vérifie l'absence de duplicats patient "
    "avant d'interpréter.")


def _pairdata(res: dict, padj: float) -> dict:
    """Données d'un croisement, prêtes pour le JSON du rapport."""
    tab, adj, cv = res["table"], res["adj_residuals"], res["cramers_v"]
    return {
        "rows": [str(x) for x in tab.index],
        "cols": [str(x) for x in tab.columns],
        "counts": tab.values.astype(int).tolist(),
        "resid": np.round(adj.values, 3).tolist(),
        "test": res["test"], "pvalue": float(res["pvalue"]), "padj": float(padj),
        "cramers_v": None if not np.isfinite(cv) else round(float(cv), 4),
        "conditions_met": bool(res["conditions_met"]), "n": int(res["n"]),
        "trend_r": None if res.get("trend_r") is None else round(float(res["trend_r"]), 3),
        "min_expected": round(float(res["min_expected"]), 3),
        "frac_cells_low_expected": round(float(res["frac_cells_low_expected"]), 4),
        "expected_threshold": float(res.get("expected_threshold", 5.0)),
        "max_low_expected_fraction": float(res.get("max_low_expected_fraction", 0.2)),
        "mc_resamples": int(res.get("mc_resamples", 2000)),
    }


def run_categorical_association(
    cluster_labels_by_k: dict, metadata: pd.DataFrame, sample_names,
    outdir: Path, k_final: int, *, ordinal: dict | None = None,
    max_levels: int = 12, min_expected: int = 5, max_lowexp_frac: float = 0.2,
    mc_resamples: int = 2000, seed: int = 0, output_subdir: str | None = None,
) -> dict:
    """Croise cluster (chaque k) × clinique et clinique × clinique.

    `cluster_labels_by_k` : {k: labels alignés sur `sample_names`}. `metadata` :
    tumeurs × variables (réindexé sur `sample_names`). `ordinal` : `{variable:
    [modalités ordonnées]}` (ou `{variable: None}` -> ordre = tri) déclarant les
    variables **ordinales** ; un croisement impliquant une ordinale et un axe
    scorable (ordinal ou binaire) utilise alors un **test de tendance**.

    Sauve les tableaux détaillés (contingence + résidus) pour k_final et les paires
    cliniques, une figure de résidus par variable (cluster k_final), et un
    `chi2_summary.csv` (toutes paires, tous k, FDR). **Renvoie la structure prête
    pour le rapport** (`byClusterK`, `byClinical`, avertissement duplicats).

    `output_subdir`, s'il est renseigné, isole les fichiers sous
    ``tables/<output_subdir>/chi2`` et ``figures/<output_subdir>``. Il permet aux
    branches parallèles (p. ex. ICA) de produire leurs propres analyses sans
    écraser les sorties du consensus clustering principal.
    """
    from . import plots as pl

    ordinal = ordinal or {}
    meta = metadata.reindex(sample_names)
    cat_vars = []
    for v in meta.columns:
        if _is_categorical(meta[v], max_levels) is not True:
            continue
        nlev = meta[v].dropna().nunique()
        if 2 <= nlev <= max_levels:
            cat_vars.append(str(v))
    if not cat_vars:
        logger.info("9a. Khi² : aucune variable clinique catégorielle exploitable "
                    "(2..%d modalités) — étape sautée.", max_levels)
        return {}

    def _order_for(v):
        if v not in ordinal:
            return None
        o = ordinal[v]
        if o:                              # liste ordonnée explicite (recommandé)
            return [str(x) for x in o]
        return sorted(meta[v].dropna().astype(str).unique())   # repli : tri

    ord_declared = [v for v in cat_vars if v in ordinal]
    root = Path(outdir)
    if output_subdir:
        detail_dir = root / "tables" / output_subdir / "chi2"
        figs = root / "figures" / output_subdir
    else:
        detail_dir = root / "tables" / "chi2"
        figs = root / "figures"
    logger.info("9a. Khi² d'indépendance : %d variable(s) catégorielle(s) (%s) × "
                "clusters (k=%s) + paires cliniques%s",
                len(cat_vars), ", ".join(cat_vars), list(sorted(cluster_labels_by_k)),
                (" ; ordinales (test de tendance) : " + ", ".join(ord_declared))
                if ord_declared else "")
    logger.warning("9a. Khi² — %s", DUP_WARNING)

    rows, records = [], []      # records : (bucket, kkey, vkey, res), même ordre que rows
    for k in sorted(cluster_labels_by_k):
        clus = pd.Series(["C" + str(int(x)) for x in cluster_labels_by_k[k]],
                         index=list(sample_names), name="cluster")
        for v in cat_vars:
            res = association_test(clus, meta[v], min_expected, max_lowexp_frac,
                                   mc_resamples, seed, a_order=None, b_order=_order_for(v))
            if res is None:
                continue
            rows.append(_tidy_row("cluster", v, int(k), res))
            records.append(("clusterK", str(int(k)), v, res))
            if k == k_final:
                _save_detail(detail_dir, f"clusterK{k}__{v}", res)
                pl.plot_chi2_residuals(res, v, f"cluster k={k}", figs,
                                       f"chi2_residuals_clusterK{k}_{v}.png")

    for v1, v2 in combinations(cat_vars, 2):
        res = association_test(meta[v1], meta[v2], min_expected, max_lowexp_frac,
                               mc_resamples, seed, a_order=_order_for(v1), b_order=_order_for(v2))
        if res is None:
            continue
        rows.append(_tidy_row(v1, v2, None, res))
        records.append(("clinical", f"{v1}|{v2}", None, res))
        _save_detail(detail_dir, f"{v1}__{v2}", res)

    tidy = pd.DataFrame(rows)
    if not len(tidy):
        return {"warning": DUP_WARNING, "clusterVars": [], "clinicalPairs": [],
                "byClusterK": {}, "byClinical": {}}

    padj = _bh(tidy["pvalue"].to_numpy())        # avant tri -> aligné sur `records`
    by_cluster_k, by_clinical = {}, {}
    for (bucket, kkey, vkey, res), pa in zip(records, padj):
        pdat = _pairdata(res, pa)
        if bucket == "clusterK":
            by_cluster_k.setdefault(kkey, {})[vkey] = pdat
        else:
            by_clinical[kkey] = pdat

    detail_dir.mkdir(parents=True, exist_ok=True)
    tidy = tidy.assign(padj=padj).sort_values("pvalue").reset_index(drop=True)
    tidy.to_csv(detail_dir / "chi2_summary.csv", index=False)
    n_sig = int((tidy["padj"] <= 0.05).sum())
    logger.info("9a. Khi² : %d paires testées, %d significatives (FDR <= 0.05) -> %s",
                len(tidy), n_sig, detail_dir / "chi2_summary.csv")
    top = tidy.loc[tidy["padj"] <= 0.05,
                   ["var1", "var2", "k", "test", "pvalue", "cramers_v"]].head(8)
    if len(top):
        logger.info("Associations les plus fortes :\n%s", top.to_string(index=False))

    clus_vars = [v for v in cat_vars if any(v in d for d in by_cluster_k.values())]
    return {
        "warning": DUP_WARNING,
        "clusterVars": clus_vars,
        "clinicalPairs": [key.split("|") for key in by_clinical],
        "byClusterK": by_cluster_k,
        "byClinical": by_clinical,
    }
