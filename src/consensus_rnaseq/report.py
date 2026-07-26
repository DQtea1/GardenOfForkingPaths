"""Étape 10 — Rapport d'analyse HTML interactif (autonome).

Rassemble tous les résultats du run (matrices de consensus par k, labels, item
consensus, arbres, scores de signatures, déconvolution, DEGSEA, métadonnées
cliniques, embeddings, tables, figures de pré-analyse) et les **exporte en JSON**
embarqué dans un unique fichier `report.html` autonome (aucune dépendance, aucun
accès réseau).

Le HTML dessine les heatmaps en **canvas** partageant un même **ordre
d'échantillons** (celui du dendrogramme du k choisi) : on peut donc empiler des
panneaux au-dessus / en dessous de la matrice de consensus **en restant alignés
au niveau des patients**. Trois onglets : Résultats (non-supervisé / signatures /
t-SNE-UMAP), Tableaux (triables/filtrables), Pré-analyse (figures de filtrage et
de choix de k).
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TEMPLATE = Path(__file__).with_name("report_template.html")


def _is_continuous(series: pd.Series, max_levels: int = 8) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    return pd.api.types.is_numeric_dtype(s) and s.nunique() > max_levels


def _b64img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _clean(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _link_stability(Z, dend, stab_by_id):
    """Rattache le score de stabilité Jaccard de chaque **branche** (nœud interne)
    à la liste `icoord`/`dcoord` du dendrogramme, dans le **même ordre** que celle-ci
    (donc directement indexable côté JS). On identifie chaque U du dendrogramme par
    (hauteur de fusion, abscisse de l'apex) — l'abscisse suit la convention scipy
    (feuille i à x=5+10·i, nœud interne = milieu de ses deux enfants). Renvoie une
    liste alignée sur `dend['icoord']`, `None` là où la branche n'a pas de score
    (racine, branches < min_size)."""
    from scipy.cluster.hierarchy import to_tree

    leaves = dend["leaves"]
    pos = {int(orig): i for i, orig in enumerate(leaves)}
    _, nodelist = to_tree(Z, rd=True)
    xof: dict = {}

    def x_of(nd):
        if nd.id in xof:
            return xof[nd.id]
        v = (5.0 + 10.0 * pos[nd.id]) if nd.is_leaf() else (x_of(nd.left) + x_of(nd.right)) / 2.0
        xof[nd.id] = v
        return v

    for nd in nodelist:
        x_of(nd)
    key2id = {(round(float(nd.dist), 6), round(xof[nd.id], 3)): nd.id
              for nd in nodelist if not nd.is_leaf()}

    out = []
    ic, dc = dend["icoord"], dend["dcoord"]
    for l in range(len(ic)):
        key = (round(dc[l][1], 6), round((ic[l][1] + ic[l][2]) / 2.0, 3))
        nid = key2id.get(key)
        out.append(_clean(stab_by_id.get(nid)) if nid is not None else None)
    return out


def _gather(result, k_final, outdir, coords, meta, sig_scores, sig_tests, deconv,
            degsea_by_k, linkage_method, min_cluster_size, k_criterion,
            branch_stability_by_k):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    from . import metrics as mt

    # stabilité Jaccard par k : {k: {id de nœud -> score}} (pour l'arbre de chaque k)
    stab_by_id_per_k = {}
    for k, bs in (branch_stability_by_k or {}).items():
        stab_by_id_per_k[int(k)] = {int(i): float(s)
                                    for i, s in zip(bs.node_ids, bs.stability)}

    outdir = Path(outdir)
    samples = [str(s) for s in result.sample_names]
    n = len(samples)
    kvals = sorted(result.consensus)
    summ = mt.summary(result)
    pac_by_k = {int(r.k): float(r.PAC) for r in summ.itertuples()}
    minclust_by_k = {int(r.k): int(r.min_cluster_size) for r in summ.itertuples()}
    ks_by_pac = sorted(kvals, key=lambda k: pac_by_k.get(k, 9.0))

    # k "recommandé" au sens PAC+Δ(K) (critère 'both'), à surligner en bleu
    try:
        best_both = int(mt.suggest_k(result, min_cluster_size, method="both"))
    except Exception:
        best_both = int(k_final)

    data = {
        "samples": samples, "n": n, "kFinal": int(k_final),
        "minClusterSize": int(min_cluster_size), "bestBoth": best_both,
        "kCriterion": str(k_criterion),
        "ks": [{"k": int(k), "pac": round(pac_by_k.get(k, float("nan")), 4),
                "minClust": minclust_by_k.get(k, 0)}
               for k in ks_by_pac],
        "perK": {}, "consensus": {},
    }

    for k in kvals:
        order = [int(i) for i in result.order(k, linkage_method)]
        labels = [int(x) for x in result.labels(k, linkage_method)]
        item = mt.item_consensus(result, k)
        imap = dict(zip(item["sample"].astype(str), item["item_consensus"]))
        Z = linkage(squareform(result.distance(k), checks=False), method=linkage_method)
        dend = dendrogram(Z, no_plot=True)
        data["perK"][str(k)] = {
            "order": order, "labels": labels,
            "item": [_clean(imap.get(s)) for s in samples],
            "icoord": dend["icoord"], "dcoord": dend["dcoord"],
        }
        # stabilité Jaccard des branches, rattachée à l'arbre de CE k
        if int(k) in stab_by_id_per_k:
            data["perK"][str(k)]["stability"] = _link_stability(
                Z, dend, stab_by_id_per_k[int(k)])
        C = result.consensus[k]
        data["consensus"][str(k)] = [[int(round(float(x) * 100)) for x in row] for row in C]

    # signatures (signatures × échantillons)
    data["signatures"] = {}
    for method, df in (sig_scores or {}).items():
        df = df.reindex(columns=samples)
        data["signatures"][method] = {
            "names": [str(x) for x in df.index],
            "values": [[_clean(v) for v in df.loc[nm]] for nm in df.index],
        }

    # tests de Wilcoxon score de signature × modalité (one-vs-rest ET pairwise),
    # pour les étoiles/barres au-dessus des boxplots. Structure :
    #   {method:{stratKey:{kkey:{sig:{"ovr":{group:p},"pair":{a:{b:p}}}}}}}
    #   stratKey "__cluster__" : kkey = str(k) ; variable clinique : kkey = "*"
    def _clean_sig(o):
        return {"ovr": {str(g): _clean(p) for g, p in (o.get("ovr") or {}).items()},
                "pair": {str(a): {str(b): _clean(p) for b, p in inner.items()}
                         for a, inner in (o.get("pair") or {}).items()}}
    data["sigTests"] = {}
    for method, strat in (sig_tests or {}).items():
        dm = {}
        for strat_key, byk in strat.items():
            dm[str(strat_key)] = {
                str(kk): {str(sig): _clean_sig(o) for sig, o in sigmap.items()}
                for kk, sigmap in byk.items()}
        data["sigTests"][str(method)] = dm

    # déconvolution (types cellulaires × échantillons)
    data["deconv"] = {}
    for method, df in (deconv or {}).items():
        df = df.reindex(columns=samples)
        data["deconv"][method] = {
            "types": [str(x) for x in df.index],
            "values": [[_clean(v) for v in df.loc[t]] for t in df.index],
        }

    # DEGSEA (NES pathway × cluster), par k -> {str(k): {collection: {...}}}.
    # Un seul k (k_final) en mode normal, tous les k si --degsea_all_k y.
    data["degsea"] = {}
    for k, coll_map in (degsea_by_k or {}).items():
        d = {}
        for coll, df in (coll_map or {}).items():
            clusters = [int(str(c).lstrip("c")) for c in df.columns]
            d[coll] = {
                "terms": [str(t) for t in df.index], "clusters": clusters,
                "nes": [[_clean(v) for v in df.loc[t]] for t in df.index],
            }
        data["degsea"][str(int(k))] = d
    data["degseaK"] = int(k_final)

    # métadonnées cliniques
    data["meta"], data["metaTypes"] = {}, {}
    if meta is not None and meta.shape[1]:
        m = meta.reindex(samples)
        for var in map(str, m.columns):
            col = m[var]
            if col.dropna().empty or (not _is_continuous(col) and col.nunique() > 40):
                continue                      # ignore vides / identifiants uniques
            if _is_continuous(col):
                num = pd.to_numeric(col, errors="coerce")
                data["meta"][var] = [_clean(v) for v in num]
                data["metaTypes"][var] = "continuous"
            else:
                data["meta"][var] = [None if pd.isna(v) else str(v) for v in col]
                data["metaTypes"][var] = "categorical"

    # embeddings — 3 coordonnées (z=0 si l'embedding est 2D) pour un rendu 3D
    data["embed"] = {}
    if coords is not None and "sample" in coords:
        c = coords.copy()
        c["sample"] = c["sample"].astype(str)
        c = c.set_index("sample").reindex(samples)
        for m, (a, b, zc) in {"tsne": ("tsne1", "tsne2", "tsne3"),
                              "umap": ("umap1", "umap2", "umap3")}.items():
            if a in c and b in c and c[a].notna().any():
                z = c[zc] if (zc in c and c[zc].notna().any()) else pd.Series(0.0, index=c.index)
                data["embed"][m] = [[_clean(x), _clean(y), _clean(zz)]
                                    for x, y, zz in zip(c[a], c[b], z)]

    # tables (toutes les .csv sous outdir/tables)
    data["tables"] = {}
    tdir = outdir / "tables"
    if tdir.exists():
        for f in sorted(tdir.rglob("*.csv")):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            truncated = len(df) > 3000
            if truncated:
                df = df.head(3000)
            rel = str(f.relative_to(tdir))
            data["tables"][rel] = {
                "columns": [str(c) for c in df.columns],
                "rows": df.astype(object).where(pd.notna(df), None).values.tolist(),
                "truncated": truncated,
            }

    # figures de pré-analyse
    data["preAnalysis"] = []
    figs = outdir / "figures"
    for name, title in [("pca_outliers.png", "Détection d'outliers (ACP)"),
                        ("purity_puree.png", "Pureté tumorale (PUREE)"),
                        ("cdf_pac_deltak.png", "Choix de k — CDF / PAC / Δ(K)"),
                        ("tracking_plot.png", "Tracking plot des affectations")]:
        p = figs / name
        if p.exists():
            data["preAnalysis"].append({"title": title, "img": _b64img(p)})

    return data


def build_report(result, k_final, outdir, *, coords=None, meta=None,
                 sig_scores=None, sig_tests=None, deconv=None, degsea_by_k=None,
                 branch_stability_by_k=None, min_cluster_size=10, k_criterion="both",
                 linkage_method="average") -> Path:
    """Construit `outdir/report.html`. Voir le module pour les entrées.

    `degsea_by_k` : dict {k: {collection: matrice NES}} — un seul k (k_final) en
    mode normal, tous les k si le run a activé `--degsea_all_k y`.
    `sig_tests` : tests de Wilcoxon (voir `sigproj.stratified_signature_tests`),
    pour les étoiles de significativité au-dessus des boxplots.
    `branch_stability_by_k` : dict `{k: stability.BranchStability}` — affiche la
    stabilité Jaccard des branches sur l'arbre consensus de **chaque** k.
    """
    outdir = Path(outdir)
    data = _gather(result, k_final, outdir, coords, meta, sig_scores, sig_tests,
                   deconv, degsea_by_k, linkage_method, min_cluster_size, k_criterion,
                   branch_stability_by_k)
    html = _TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    out = outdir / "report.html"
    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size / 1e6
    logger.info("Rapport HTML : %s (%.1f Mo)", out, size)
    return out
