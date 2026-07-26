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

Idées non implémentées ici mais pertinentes selon les données : pour des variables
**ordinales** (stade, grade…), un test de **tendance de Cochran-Armitage** ou un
**tau de Kendall** serait plus puissant que le khi² nominal ; la détection
automatique de l'ordre étant ambiguë, on reste sur le khi² nominal par défaut.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from sigproj import _is_categorical

logger = logging.getLogger(__name__)


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


def association_test(a: pd.Series, b: pd.Series, min_expected: int = 5,
                     max_lowexp_frac: float = 0.2, mc_resamples: int = 2000,
                     seed: int = 0) -> dict | None:
    """Test d'indépendance de deux variables catégorielles alignées.

    Renvoie un dict avec le tableau de contingence, les effectifs attendus, les
    résidus standardisés ajustés (Haberman), le test utilisé, la statistique, la
    p-valeur, le V de Cramér et le diagnostic des conditions d'application — ou
    ``None`` si l'une des variables a moins de 2 modalités présentes.
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

    if conditions:
        used, p = "chi2", float(p_chi2)
    elif r == 2 and c == 2:
        _, p = fisher_exact(obs)
        used, p = "fisher_exact", float(p)
    else:
        ai = pd.Categorical(df["a"]).codes
        bi = pd.Categorical(df["b"]).codes
        p = _mc_pvalue(ai, bi, r, c, float(chi2stat), mc_resamples, seed)
        used = "chi2_montecarlo"

    k = min(r - 1, c - 1)
    cramers_v = float(np.sqrt(chi2stat / (n * k))) if (n > 0 and k > 0) else np.nan

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
        "conditions_met": bool(conditions),
        "min_expected": min_exp, "frac_cells_low_expected": frac_low,
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
    return {
        "var1": v1, "var2": v2, "k": k, "test": res["test"],
        "chi2": round(res["chi2"], 4), "dof": res["dof"], "n": res["n"],
        "cramers_v": round(res["cramers_v"], 4) if np.isfinite(res["cramers_v"]) else np.nan,
        "pvalue": res["pvalue"], "conditions_met": res["conditions_met"],
        "min_expected": round(res["min_expected"], 2),
        "frac_cells_low_expected": round(res["frac_cells_low_expected"], 3),
    }


def run_categorical_association(
    cluster_labels_by_k: dict, metadata: pd.DataFrame, sample_names,
    outdir: Path, k_final: int, *, max_levels: int = 12, min_expected: int = 5,
    max_lowexp_frac: float = 0.2, mc_resamples: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    """Croise cluster (chaque k) × clinique et clinique × clinique.

    `cluster_labels_by_k` : {k: labels alignés sur `sample_names`}. `metadata` :
    tumeurs × variables (réindexé sur `sample_names`). Sauve les tableaux détaillés
    (contingence + résidus) pour k_final et pour les paires cliniques, une figure
    de résidus par variable clinique (cluster k_final), et un `chi2_summary.csv`
    (toutes paires, tous k, avec FDR). Renvoie la table de synthèse.
    """
    import plots as pl

    meta = metadata.reindex(sample_names)
    # variables cliniques catégorielles exploitables (2..max_levels modalités)
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
        return pd.DataFrame()

    detail_dir = Path(outdir) / "tables" / "chi2"
    figs = Path(outdir) / "figures"
    logger.info("9a. Khi² d'indépendance : %d variable(s) clinique(s) catégorielle(s) "
                "(%s) × clusters (k=%s) + paires cliniques",
                len(cat_vars), ", ".join(cat_vars), list(sorted(cluster_labels_by_k)))
    logger.warning(
        "9a. Khi² — le test suppose l'INDÉPENDANCE des observations (1 tumeur = "
        "1 patient). Si des patients ont plusieurs prélèvements (primaire + "
        "métastase, lésions multiples…), ils sont comptés plusieurs fois : les "
        "p-valeurs deviennent anticonservatives (trop optimistes). Vérifie "
        "l'absence de duplicats patient avant d'interpréter.")

    rows = []
    # ---- cluster (chaque k) × clinique ; détails + figures pour k_final
    for k in sorted(cluster_labels_by_k):
        clus = pd.Series(["C" + str(int(x)) for x in cluster_labels_by_k[k]],
                         index=list(sample_names), name="cluster")
        for v in cat_vars:
            res = association_test(clus, meta[v], min_expected, max_lowexp_frac,
                                   mc_resamples, seed)
            if res is None:
                continue
            rows.append(_tidy_row("cluster", v, int(k), res))
            if k == k_final:
                _save_detail(detail_dir, f"clusterK{k}__{v}", res)
                pl.plot_chi2_residuals(res, v, f"cluster k={k}", figs,
                                       f"chi2_residuals_clusterK{k}_{v}.png")

    # ---- clinique × clinique (indépendant de k)
    for v1, v2 in combinations(cat_vars, 2):
        res = association_test(meta[v1], meta[v2], min_expected, max_lowexp_frac,
                               mc_resamples, seed)
        if res is None:
            continue
        rows.append(_tidy_row(v1, v2, None, res))
        _save_detail(detail_dir, f"{v1}__{v2}", res)

    tidy = pd.DataFrame(rows)
    if len(tidy):
        tidy["padj"] = _bh(tidy["pvalue"].to_numpy())
        tidy = tidy.sort_values("pvalue").reset_index(drop=True)
        detail_dir.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(detail_dir / "chi2_summary.csv", index=False)
        n_sig = int((tidy["padj"] <= 0.05).sum())
        logger.info("9a. Khi² : %d paires testées, %d significatives (FDR <= 0.05) "
                    "-> %s", len(tidy), n_sig, detail_dir / "chi2_summary.csv")
        top = tidy.loc[tidy["padj"] <= 0.05, ["var1", "var2", "k", "test", "pvalue",
                                              "cramers_v"]].head(8)
        if len(top):
            logger.info("Associations les plus fortes :\n%s", top.to_string(index=False))
    return tidy
