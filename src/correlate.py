"""Étape 9b — Corrélations entre variables continues à l'échelle du patient.

Croise chaque variable clinique **continue** des métadonnées avec les autres scores
continus calculés par patient (signatures **ssGSEA** / **expression moyenne**,
scores/fractions de **déconvolution**), plus **signatures × déconvolution**.
Corrélation de **Spearman** par défaut (robuste, monotone, sans hypothèse de
normalité — adaptée à des scores bornés/asymétriques) ; Pearson en option.
Observations **complètes par paire** (gestion des valeurs manquantes cliniques),
correction **BH (FDR)** sur l'ensemble des paires testées.

Pertinence / garde-fous (d'où le périmètre par défaut) :
  - **déconvolution compositionnelle** : les fractions cellulaires somment ≈ 1, donc
    les corrélations **déconv × déconv** portent un biais négatif artéfactuel
    (Aitchison). Elles ne sont calculées qu'avec `all_pairs=True`.
  - **redondance signatures** : ssGSEA et moyenne d'une même signature mesurent le
    même gene set ; on saute **signature × signature** par défaut (bloc redondant).
  - **pureté tumorale** : elle confond expression et déconvolution ; une corrélation
    forte peut la refléter (cf. corrélation partielle dans les analyses
    complémentaires).

Par défaut on garde donc : clinique × clinique, clinique × signatures,
clinique × déconvolution, et signatures × déconvolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .stats_utils import benjamini_hochberg as _bh
from .stats_utils import is_continuous as _is_continuous

logger = logging.getLogger(__name__)


def assemble_features(sig_scores: dict | None, deconv: dict | None,
                      metadata: pd.DataFrame | None, sample_names,
                      max_levels: int = 6):
    """Matrice patients × variables continues, + tags de bloc et de famille.

    Familles : ``clinic`` (métadonnées continues), ``sig`` (signatures ssGSEA /
    moyenne), ``deconv`` (types cellulaires). Blocs (plus fins) : ``clinic``,
    ``sig_ssgsea``, ``sig_mean``, ``deconv:<méthode>``.
    """
    idx = [str(s) for s in sample_names]
    feats, block, family = {}, {}, {}

    if metadata is not None and metadata.shape[1]:
        m = metadata.copy()
        m.index = m.index.astype(str)
        m = m.reindex(idx)
        for v in map(str, m.columns):
            if _is_continuous(m[v], max_levels):
                name = f"clinic:{v}"
                feats[name] = pd.to_numeric(m[v], errors="coerce").to_numpy()
                block[name], family[name] = "clinic", "clinic"

    for meth, blk in (("ssgsea", "sig_ssgsea"), ("mean", "sig_mean")):
        df = (sig_scores or {}).get(meth)
        if df is None:
            continue
        d = df.reindex(columns=idx)
        for sig in map(str, d.index):
            name = f"{meth}:{sig}"
            feats[name] = pd.to_numeric(d.loc[sig], errors="coerce").to_numpy()
            block[name], family[name] = blk, "sig"

    for method, df in (deconv or {}).items():
        d = df.reindex(columns=idx)
        for ct in map(str, d.index):
            name = f"deconv:{method}:{ct}"
            feats[name] = pd.to_numeric(d.loc[ct], errors="coerce").to_numpy()
            block[name], family[name] = f"deconv:{method}", "deconv"

    feat_df = pd.DataFrame(feats, index=idx)
    return feat_df, block, family


def _keep_pair(fa: str, fb: str, all_pairs: bool) -> bool:
    if all_pairs:
        return True
    if fa == "deconv" and fb == "deconv":   # compositionnel -> biais artéfactuel
        return False
    if fa == "sig" and fb == "sig":         # ssGSEA/moyenne de mêmes signatures -> redondant
        return False
    return True


def pairwise_correlations(feat_df: pd.DataFrame, block: dict, family: dict,
                          method: str = "spearman", all_pairs: bool = False,
                          min_n: int = 8) -> pd.DataFrame:
    """Corrélations deux à deux (observations complètes par paire). Table longue."""
    from scipy.stats import pearsonr, spearmanr

    corr = pearsonr if method == "pearson" else spearmanr
    names = list(feat_df.columns)
    cols = {nm: feat_df[nm].to_numpy(dtype=float) for nm in names}
    rows = []
    for i in range(len(names)):
        a = names[i]
        xa = cols[a]
        for j in range(i + 1, len(names)):
            b = names[j]
            if not _keep_pair(family[a], family[b], all_pairs):
                continue
            xb = cols[b]
            mask = np.isfinite(xa) & np.isfinite(xb)
            if int(mask.sum()) < min_n:
                continue
            try:
                r, p = corr(xa[mask], xb[mask])
            except Exception:
                continue
            if not np.isfinite(r):
                continue
            rows.append({"var1": a, "block1": block[a], "var2": b, "block2": block[b],
                         "method": method, "rho": float(r), "pvalue": float(p),
                         "n": int(mask.sum())})
    df = pd.DataFrame(rows)
    if len(df):
        df["padj"] = _bh(df["pvalue"].to_numpy())
    return df


def run_correlations(sig_scores: dict | None, deconv: dict | None,
                     metadata: pd.DataFrame | None, sample_names, outdir: Path, *,
                     method: str = "spearman", all_pairs: bool = False,
                     min_n: int = 8, max_levels: int = 6, top_fig: int = 40) -> dict:
    """Assemble les variables continues, corrèle deux à deux, sauve tables + figure.

    Renvoie une structure (blocs, table longue) réutilisable par le rapport.
    """
    from . import plots as pl

    feat_df, block, family = assemble_features(sig_scores, deconv, metadata,
                                               sample_names, max_levels)
    if feat_df.shape[1] < 2:
        logger.info("9b. Corrélations : moins de 2 variables continues à l'échelle "
                    "du patient — étape sautée.")
        return {}

    n_clin = sum(1 for f in family.values() if f == "clinic")
    n_sig = sum(1 for f in family.values() if f == "sig")
    n_dec = sum(1 for f in family.values() if f == "deconv")
    logger.info("9b. Corrélations (%s) : %d variables continues (clinique %d, "
                "signatures %d, déconvolution %d)%s", method, feat_df.shape[1],
                n_clin, n_sig, n_dec, "" if not all_pairs else " — TOUTES paires")
    if n_dec and all_pairs:
        logger.warning("9b. Corrélations — déconv × déconv incluses (all_pairs) : "
                       "fractions COMPOSITIONNELLES, corrélations négatives possiblement "
                       "artéfactuelles. À interpréter avec prudence (CLR / corrélation "
                       "partielle recommandées).")

    tab = pairwise_correlations(feat_df, block, family, method, all_pairs, min_n)
    cdir = Path(outdir) / "tables" / "correlations"
    cdir.mkdir(parents=True, exist_ok=True)
    if not len(tab):
        logger.info("9b. Corrélations : aucune paire exploitable (n < %d).", min_n)
        return {"features": list(feat_df.columns), "table": tab}

    tab = tab.sort_values("padj").reset_index(drop=True)
    tab.to_csv(cdir / "correlations.csv", index=False)
    n_sig_pairs = int((tab["padj"] <= 0.05).sum())
    logger.info("9b. Corrélations : %d paires testées, %d significatives (FDR <= 0.05) "
                "-> %s", len(tab), n_sig_pairs, cdir / "correlations.csv")
    top = tab.reindex(columns=["var1", "var2", "rho", "padj", "n"]).head(8)
    if len(top):
        logger.info("Corrélations les plus fortes :\n%s", top.to_string(index=False))

    # matrice clinique × dérivé (rho) : figure + CSV (si des variables cliniques continues)
    clin = [f for f in feat_df.columns if family[f] == "clinic"]
    if clin:
        rho_m, pad_m = {}, {}
        for r in tab.itertuples():
            if family[r.var1] == "clinic" and family[r.var2] != "clinic":
                c, dv = r.var1, r.var2
            elif family[r.var2] == "clinic" and family[r.var1] != "clinic":
                c, dv = r.var2, r.var1
            else:
                continue
            rho_m.setdefault(c, {})[dv] = r.rho
            pad_m.setdefault(c, {})[dv] = r.padj
        if rho_m:
            M = pd.DataFrame(rho_m).T.reindex(index=clin)      # clin × dérivé
            Pm = pd.DataFrame(pad_m).T.reindex(index=clin, columns=M.columns)
            # colonnes : dérivés significatifs (≥1 clinique), sinon top |rho| ; plafond top_fig
            sig_cols = Pm.columns[(Pm <= 0.05).any(axis=0)]
            order = (M[sig_cols] if len(sig_cols) else M).abs().max(axis=0) \
                .sort_values(ascending=False)
            keep = list(order.index[:top_fig])
            M, Pm = M[keep], Pm[keep]
            M.to_csv(cdir / "correlations_clinic_matrix.csv")
            pl.plot_correlation_heatmap(M, Pm, method, cdir.parent.parent / "figures",
                                        "correlations_clinic.png")

    return {"features": list(feat_df.columns), "family": family, "block": block,
            "method": method, "table": tab}
