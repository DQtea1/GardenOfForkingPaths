"""Étape 18 — Rapport d'analyse HTML interactif (autonome).

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

from . import config as cf
from .stats_utils import is_continuous

logger = logging.getLogger(__name__)

_TEMPLATE = Path(__file__).with_name("report_template.html")

# seuil d'affichage des annotations cliniques du rapport (cf. stats_utils : par contexte)
_META_MAX_LEVELS = 8


def _is_continuous(series: pd.Series) -> bool:
    return is_continuous(series, _META_MAX_LEVELS)


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


def _clean_deep(v):
    """Version récursive de `_clean` pour les métadonnées structurées ICA."""
    if isinstance(v, dict):
        return {str(k): _clean_deep(value) for k, value in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean_deep(value) for value in v]
    return _clean(v)


def _embedding_payload(coords: pd.DataFrame | None, samples) -> dict:
    """Sérialise une table d'embedding, dans l'ordre des échantillons fourni."""
    embed = {}
    if not isinstance(coords, pd.DataFrame) or "sample" not in coords:
        return embed
    coord = coords.copy()
    coord["sample"] = coord["sample"].astype(str)
    coord = coord.set_index("sample").reindex(samples)
    for name, (x, y, z) in {
        "tsne": ("tsne1", "tsne2", "tsne3"),
        "umap": ("umap1", "umap2", "umap3"),
    }.items():
        if x in coord and y in coord and coord[x].notna().any():
            zz = coord[z] if z in coord else pd.Series(0.0, index=coord.index)
            embed[name] = [
                [_clean(a), _clean(b), _clean(c)]
                for a, b, c in zip(coord[x], coord[y], zz)
            ]
    return embed


def _mostly_numeric(col: pd.Series, min_ratio: float = 0.8) -> bool:
    """True si `col`, non numérique au sens de pandas, est en fait une variable
    quantitative saisie avec quelques mentions textuelles ("NC", "inconnu", "NA").

    Une telle colonne a un dtype `object` : `is_continuous` la déclare
    catégorielle et, passé `max_levels`, `_meta_payload` la SUPPRIME du rapport —
    un âge disparaissait ainsi entièrement (couleur, filtres, boxplots) à cause
    de deux cellules non renseignées. On la récupère quand la conversion réussit
    sur au moins `min_ratio` des valeurs présentes ET que le résultat a trop de
    valeurs distinctes pour être un code (grade, statut 0/1, qui doivent rester
    catégoriels pour les filtres et les tests d'association).
    """
    present = col.dropna()
    if present.empty or pd.api.types.is_numeric_dtype(present):
        return False
    numeric = pd.to_numeric(present, errors="coerce")
    if numeric.notna().sum() < min_ratio * present.size:
        return False
    return numeric.nunique() > _META_MAX_LEVELS


def _meta_payload(metadata: pd.DataFrame | None, samples,
                  max_levels: int = 40) -> tuple[dict, dict]:
    """Métadonnées cliniques sérialisées : ``({var: valeurs}, {var: type})``.

    Deux colonnes sont écartées : celles entièrement vides, et les catégorielles
    à plus de `max_levels` modalités — un identifiant déguisé (numéro de bloc,
    date) n'apporte rien à une couleur ni à un filtre, et ferait exploser les
    menus. Le même sérialiseur sert à la branche historique et aux branches ICA :
    les deux vues du rapport montrent donc exactement les mêmes variables.

    Les colonnes quantitatives « sales » (cf. `_mostly_numeric`) sont converties
    en continu plutôt qu'écartées.
    """
    meta: dict[str, list] = {}
    meta_types: dict[str, str] = {}
    if not isinstance(metadata, pd.DataFrame) or not metadata.shape[1]:
        return meta, meta_types

    frame = metadata.copy()
    frame.index = frame.index.astype(str)
    frame = frame.reindex([str(s) for s in samples])
    for variable in map(str, frame.columns):
        col = frame[variable]
        continuous = _is_continuous(col) or _mostly_numeric(col)
        if col.dropna().empty or (not continuous and col.nunique() > max_levels):
            continue
        if continuous:
            meta[variable] = [_clean(v) for v in pd.to_numeric(col, errors="coerce")]
            meta_types[variable] = "continuous"
        else:
            meta[variable] = [None if pd.isna(v) else str(v) for v in col]
            meta_types[variable] = "categorical"
    return meta, meta_types


def _degsea_contrast_label(tag: str) -> tuple[str, str, str]:
    """Retourne libellé, schéma et cible d'un nom de contraste DEGSEA."""
    import re

    match = re.fullmatch(r"c(\d+)_vs_(rest|c\d+)", str(tag))
    if not match:
        return str(tag), "unknown", ""
    target, reference = int(match.group(1)), match.group(2)
    if reference == "rest":
        return f"C{target} vs all", "one-vs-all", f"C{target}"
    return f"C{target} vs C{int(reference[1:])}", "one-vs-one", f"C{target}"


def _number_or_none(value):
    """Convertit une cellule CSV DEGSEA en nombre JSON ou ``None``."""
    try:
        if pd.isna(value):
            return None
        return _clean(float(value))
    except (TypeError, ValueError):
        return None


def _gsea_rows_payload(table: pd.DataFrame | None) -> list[dict]:
    """Sérialise une table gseapy pour les tableaux GSEA interactifs."""
    if not isinstance(table, pd.DataFrame) or table.empty:
        return []
    frame = table.copy()
    if "Term" not in frame and frame.index.name == "Term":
        frame = frame.reset_index()
    if "Term" not in frame:
        return []
    pval_col = (
        "NOM p-val" if "NOM p-val" in frame
        else "pvalue" if "pvalue" in frame else None
    )
    padj_col = (
        "FDR q-val" if "FDR q-val" in frame
        else "padj" if "padj" in frame else None
    )
    lead_col = (
        "Lead_genes" if "Lead_genes" in frame
        else "leading_edge" if "leading_edge" in frame else None
    )
    rows = []
    for row in frame.to_dict(orient="records"):
        term = row.get("Term")
        if pd.isna(term):
            continue
        leading_edge = row.get(lead_col) if lead_col else None
        rows.append({
            "term": str(term),
            "NES": _number_or_none(row.get("NES")),
            "pvalue": _number_or_none(row.get(pval_col)) if pval_col else None,
            "padj": _number_or_none(row.get(padj_col)) if padj_col else None,
            "leadingEdge": (
                None if leading_edge is None or pd.isna(leading_edge)
                else str(leading_edge)
            ),
        })
    return rows


def _gsea_files_payload(directory: Path, pattern: str, strip: tuple[str, str]) -> dict:
    """Sérialise les `gsea_*.csv` d'un dossier -> ``{collection: lignes}``.

    `strip` donne le préfixe et le suffixe à retirer du nom de fichier pour
    retrouver le nom de la collection : ``("gsea_", "")`` côté clinique,
    ``("gsea_", "_<contraste>")`` côté clusters, où le contraste est collé au
    nom du fichier.
    """
    prefix, suffix = strip
    out: dict[str, list] = {}
    for path in sorted(directory.glob(pattern)):
        stem = path.stem
        if not stem.startswith(prefix) or (suffix and not stem.endswith(suffix)):
            continue
        collection = stem[len(prefix):len(stem) - len(suffix)] if suffix \
            else stem[len(prefix):]
        try:
            table = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Rapport GSEA : lecture impossible de %s : %s", path, exc)
            continue
        out[collection] = _gsea_rows_payload(table)
    return out


def _deseq2_genes_payload(de: pd.DataFrame) -> list[dict]:
    """Sérialise une table DESeq2 déjà lue -> une entrée par gène.

    Le drapeau `dispersion_suspecte` n'est transporté que pour les gènes qui le
    portent : le volcano les montre sans les étiqueter (il teste
    `!row.suspect`, donc l'absence vaut faux), et une table de 12 000 gènes ne
    gagne pas 12 000 booléens à `false`.
    """
    gene_col = ("gene" if "gene" in de.columns
                else de.columns[0] if len(de.columns) else None)
    if gene_col is None:
        return []
    lfc_col = "log2FoldChange" if "log2FoldChange" in de else None
    p_col = "pvalue" if "pvalue" in de else None
    padj_col = "padj" if "padj" in de else None
    flag_col = "dispersion_suspecte" if "dispersion_suspecte" in de else None

    genes = []
    for row in de.to_dict(orient="records"):
        gene = row.get(gene_col)
        if pd.isna(gene):
            continue
        entry = {
            "gene": str(gene),
            "log2FoldChange": _number_or_none(row.get(lfc_col)) if lfc_col else None,
            "pvalue": _number_or_none(row.get(p_col)) if p_col else None,
            "padj": _number_or_none(row.get(padj_col)) if padj_col else None,
        }
        if flag_col and bool(row.get(flag_col)):
            entry["suspect"] = True
        genes.append(entry)
    return genes


def _ica_metagene_gsea_payload(results: dict | None) -> dict:
    """Convertit ``composante → collection → DataFrame`` en JSON."""
    payload = {}
    for component, collections in (results or {}).items():
        serialized = {
            str(collection): _gsea_rows_payload(table)
            for collection, table in (collections or {}).items()
        }
        payload[str(component)] = serialized
    return payload


def _degsea_default_k(k_final: int, recommended_k: int,
                      degsea_by_k: dict | None) -> int:
    """Choisit le K DEGSEA ouvert par défaut parmi les résultats disponibles."""
    available = sorted(int(k) for k in (degsea_by_k or {}))
    if int(recommended_k) in available:
        return int(recommended_k)
    if int(k_final) in available:
        return int(k_final)
    return available[0] if available else int(recommended_k)


def _degsea_detail_payload(outdir: Path, default_k: int,
                           degsea_by_k: dict | None) -> dict:
    """Embarque les résultats complets DESeq2 et GSEA pour l'onglet volcano.

    Ce lecteur sérialise les CSV déjà exportés par :mod:`degsea`. Il reconnaît
    les deux dispositions possibles :
    ``tables/degsea/{ova,ovo}`` (K recommandé seul) et
    ``tables/degsea/k<K>/{ova,ovo}`` (``degsea_all_k = y``).
    """
    root = Path(outdir) / "tables" / "degsea"
    if not root.exists():
        return {}

    bases: dict[int, Path] = {}
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("k"):
            continue
        try:
            bases[int(path.name[1:])] = path
        except ValueError:
            continue
    # Ne pas découvrir par erreur de vieux dossiers résiduels ne faisant pas
    # partie du run courant quand le pipeline a fourni l'information de K.
    known_k = {int(k) for k in (degsea_by_k or {})}
    # En mode ciblé, les sorties sont directement sous tables/degsea/{ova,ovo}.
    # Elles ont priorité sur d'éventuels dossiers k<K> laissés par un ancien run.
    if (
        any((root / scheme).is_dir() for scheme in ("ova", "ovo"))
        and len(known_k) <= 1
    ):
        bases[int(default_k)] = root
    if known_k:
        bases = {k: path for k, path in bases.items() if k in known_k}

    payload: dict[str, dict] = {}
    for k, base in sorted(bases.items()):
        contrasts = []
        for dirname in ("ova", "ovo"):
            directory = base / dirname
            if not directory.exists():
                continue
            for de_path in sorted(directory.glob("deseq2_*.csv")):
                tag = de_path.stem.removeprefix("deseq2_")
                try:
                    de = pd.read_csv(de_path)
                except Exception as exc:
                    logger.warning("Rapport DEGSEA : lecture impossible de %s : %s", de_path, exc)
                    continue
                genes = _deseq2_genes_payload(de)
                if not genes and not len(de.columns):
                    continue
                gsea = _gsea_files_payload(directory, f"gsea_*_{tag}.csv",
                                           ("gsea_", f"_{tag}"))
                label, scheme, target = _degsea_contrast_label(tag)
                contrasts.append({
                    "id": tag, "label": label,
                    "scheme": scheme if scheme != "unknown" else dirname,
                    "target": target, "genes": genes, "gsea": gsea,
                })
        if contrasts:
            payload[str(k)] = {"contrasts": contrasts}
    return payload


def _clinical_degsea_detail_payload(outdir: Path,
                                    clinical_degsea: dict | None) -> dict:
    """Embarque les sorties des expériences DEGSEA cliniques dans le rapport.

    Une expérience clinique correspond à un seul contraste ajusté et à ses
    collections GSEA. Contrairement au DEGSEA des clusters, elle n'a donc ni
    axe ``k`` ni comparaisons one-vs-one : le nom de l'expérience devient le
    sélecteur principal de l'interface.
    """
    root = Path(outdir) / "tables" / "clinical_degsea"
    if not root.exists() or not clinical_degsea:
        return {}

    payload: dict[str, dict] = {}
    for group_col, by_modality in sorted(clinical_degsea.items()):
        for modality, experiments in sorted(by_modality.items()):
            block = _clinical_stratum_payload(root, group_col, modality, experiments)
            if block:
                payload.setdefault(str(group_col), {})[str(modality)] = block
    return payload


def _clinical_stratum_payload(root: Path, group_col: str, modality: str,
                              clinical_degsea: dict) -> dict:
    """Charge les sorties d'une strate (colonne × modalité) du DEGSEA clinique."""
    payload: dict[str, dict] = {}
    for name, summary in sorted(clinical_degsea.items()):
        directory = root / cf.slug(group_col) / cf.slug(modality) / str(name)
        de_path = directory / "deseq2.csv"
        if not de_path.exists():
            logger.warning("Rapport DEGSEA clinique : fichier absent : %s", de_path)
            continue
        try:
            de = pd.read_csv(de_path)
        except Exception as exc:
            logger.warning("Rapport DEGSEA clinique : lecture impossible de %s : %s", de_path, exc)
            continue
        # Les gènes à dispersion effondrée arrivent marqués (`suspect`) plutôt
        # que supprimés : le volcano peut les montrer sans les présenter comme
        # des découvertes.
        genes = _deseq2_genes_payload(de)
        gsea = _gsea_files_payload(directory, "gsea_*.csv", ("gsea_", ""))

        summary = summary or {}
        design = str(summary.get("design", ""))
        contrast = str(summary.get("contrast", ""))
        control, test = str(summary.get("control", "")), str(summary.get("test", ""))
        details = " · ".join(part for part in (f"{test} vs {control}" if test or control else "", design) if part)
        n_samples = summary.get("n_samples")
        payload[str(name)] = {
            "id": str(name),
            "label": f"{name} — {details}" if details else str(name),
            "design": design,
            "contrast": contrast,
            "control": control,
            "test": test,
            "groupCol": str(group_col),
            "modality": str(modality),
            # Formule RÉELLEMENT ajustée sur cette strate : elle peut différer du
            # design demandé, un terme constant sur la strate étant écarté. Sans
            # cette information, deux volcanos de strates différentes ne sont pas
            # comparables et rien ne le signale.
            "designUsed": str(summary.get("design_used", design)),
            "droppedTerms": str(summary.get("dropped_terms", "") or ""),
            "nSamples": n_samples,
            "nTest": summary.get("n_test"),
            "nControl": summary.get("n_control"),
            "genes": genes,
            "gsea": gsea,
        }
    return payload


def _outrider_payload(outrider: dict | None, enabled: bool = False) -> dict:
    """Sérialise les runs OUTRIDER (7b) pour le menu déroulant du rapport.

    Un run = un sous-groupe de `subset_by`. Le rapport doit pouvoir dire
    pourquoi un groupe attendu n'apparaît pas : les écartés voyagent donc avec
    leur motif, à côté des runs aboutis.
    """
    if not outrider or not (outrider.get("runs") or outrider.get("skipped")):
        return {
            "status": "not_run",
            "message": (
                "OUTRIDER était demandé, mais aucun run n'a abouti. Voir run.log "
                "et tables/outrider/plan.csv." if enabled else
                "OUTRIDER n'a pas été exécuté pour ce run (run_outrider = n). "
                "Déclare les découpages voulus dans `subset_by` et relance avec "
                "--run_outrider y."),
            "runs": [], "skipped": [], "settings": {},
        }

    def rows(frame, columns):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return []
        keep = [c for c in columns if c in frame.columns]
        return [{c: _clean(row[c]) for c in keep}
                for row in frame[keep].to_dict(orient="records")]

    runs = []
    for run in outrider.get("runs", []):
        summary = {str(k): _clean(v) for k, v in (run.get("summary") or {}).items()}
        runs.append({
            "key": str(run.get("key", "")),
            "subset": str(run.get("subset", "")),
            # Données des figures : déjà arrondies et bornées par le runner, on
            # les transporte telles quelles (les retraiter ici les alourdirait).
            "figures": run.get("figures") or {},
            "label": str(run.get("label", run.get("key", ""))),
            "columns": [str(c) for c in run.get("columns", [])],
            "modalities": [str(m) for m in run.get("modalities", [])],
            "summary": summary,
            "samples": rows(run.get("per_sample"),
                            ["sample", "n_aberrant", "n_sur", "n_sous"]),
            "genes": rows(run.get("per_gene"), ["gene", "n_aberrant"]),
            "events": rows(run.get("events"),
                           ["sample", "gene", "padj", "pvalue", "l2fc", "zscore",
                            "raw_count", "predicted", "direction"]),
        })

    return {
        "status": "complete" if runs else "empty",
        "message": None if runs else (
            "Aucun sous-groupe n'a pu être analysé : tous ont été écartés. "
            "Motifs dans tables/outrider/plan.csv."),
        "runs": runs,
        "skipped": [{str(k): _clean(v) for k, v in entry.items()}
                    for entry in outrider.get("skipped", [])],
        "settings": {str(k): _clean(v)
                     for k, v in (outrider.get("settings") or {}).items()},
    }


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


def _consensus_payload(result, k_final: int, linkage_method: str,
                       min_cluster_size: int, k_criterion: str,
                       branch_stability_by_k: dict | None = None) -> dict:
    """Sérialise un résultat consensus pour le rapport.

    Le consensus historique et chaque branche ICA partagent ce contrat JSON. Le
    helper évite que l'onglet ICA dépende des variables globales de la première
    analyse et maintient une séparation stricte entre les deux branches.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    from . import metrics as mt

    samples = [str(s) for s in result.sample_names]
    kvals = sorted(result.consensus)
    summ = mt.summary(result)
    pac_by_k = {int(r.k): float(r.PAC) for r in summ.itertuples()}
    minclust_by_k = {int(r.k): int(r.min_cluster_size) for r in summ.itertuples()}
    ks_by_pac = sorted(kvals, key=lambda k: pac_by_k.get(k, 9.0))
    try:
        best_both = int(mt.suggest_k(result, min_cluster_size, method="both"))
    except Exception:
        best_both = int(k_final)

    stab_by_id_per_k = {}
    for k, bs in (branch_stability_by_k or {}).items():
        stab_by_id_per_k[int(k)] = {int(i): float(s)
                                    for i, s in zip(bs.node_ids, bs.stability)}

    payload = {
        "samples": samples, "n": len(samples), "kFinal": int(k_final),
        "minClusterSize": int(min_cluster_size), "bestBoth": best_both,
        "kCriterion": str(k_criterion),
        "ks": [{"k": int(k), "pac": round(pac_by_k.get(k, float("nan")), 4),
                "minClust": minclust_by_k.get(k, 0)} for k in ks_by_pac],
        "perK": {}, "consensus": {},
    }
    for k in kvals:
        order = [int(i) for i in result.order(k, linkage_method)]
        labels = [int(x) for x in result.labels(k, linkage_method)]
        item = mt.item_consensus(result, k)
        imap = dict(zip(item["sample"].astype(str), item["item_consensus"]))
        Z = result.linkage_tree(k, linkage_method)   # mémoïsé côté ConsensusResult
        dend = dendrogram(Z, no_plot=True)
        payload["perK"][str(k)] = {
            "order": order, "labels": labels,
            "item": [_clean(imap.get(s)) for s in samples],
            "icoord": dend["icoord"], "dcoord": dend["dcoord"],
        }
        if int(k) in stab_by_id_per_k:
            payload["perK"][str(k)]["stability"] = _link_stability(
                Z, dend, stab_by_id_per_k[int(k)])
        C = result.consensus[k]
        payload["consensus"][str(k)] = [
            [int(round(float(x) * 100)) for x in row] for row in C]
    return payload


def _corr_payload(corr) -> dict:
    """Convertit les corrélations en JSON, y compris les valeurs nécessaires au
    nuage de points de la branche ICA."""
    if not corr or "table" not in corr or not len(corr["table"]):
        return {}
    tab, blk = corr["table"], corr.get("block", {})
    blocks = {}
    for f in corr.get("features", []):
        if f in blk:
            blocks.setdefault(blk[f], []).append(str(f))
    pairs = [{"a": str(r.var1), "b": str(r.var2),
              "rho": _clean(round(float(r.rho), 4)), "p": _clean(float(r.pvalue)),
              "padj": _clean(float(r.padj)), "n": int(r.n)}
             for r in tab.itertuples()]
    values = {str(name): [_clean(v) for v in vals]
              for name, vals in (corr.get("values") or {}).items()}
    return {"method": corr.get("method", "spearman"), "blocks": blocks,
            "pairs": pairs, "values": values}


def _ica_preanalysis(outdir: Path) -> list[dict]:
    """Embarque uniquement les cinq diagnostics ICA demandés dans le sous-onglet
    Pré-analyse > ICA. Les figures de consensus ICA restent dans leurs résultats."""
    titles = {
        "ica_index_stability_distribution": "Distribution de l’indice de stabilité",
        "ica_mean_stability": "Stabilité moyenne",
        "ica_component_stability": "Stabilité des composantes ICA",
        "ica_component_mds": "Mise à l’échelle multidimensionnelle des composantes ICA",
        "ica_metagene_distribution": "Distribution des métagènes",
    }
    figdir = Path(outdir) / "figures" / "ica"
    cards = []
    if not figdir.exists():
        return cards
    for path in sorted(figdir.glob("*.png")):
        stem = path.stem
        key = next((k for k in titles if stem.startswith(k)), None)
        if key:
            cards.append({"title": titles[key], "img": _b64img(path)})
    return cards


def _ica_payload(ica, outdir: Path, *, linkage_method: str,
                 min_cluster_size: int, k_criterion: str,
                 fallback_meta: dict, fallback_meta_types: dict) -> dict:
    """Construit le payload isolé de la branche ICA avec un état vide sûr."""
    if not ica or not ica.get("result"):
        enabled = bool(ica and ica.get("enabled"))
        message = (
            "L’ICA était demandée, mais aucun résultat n’a été transmis au rapport. "
            "Consultez run.log pour l’erreur du pipeline."
            if enabled else
            "Branche ICA désactivée pour ce run (run_ica = n). Relancez le pipeline "
            "avec --run_ica y pour produire les résultats ICA."
        )
        return {"status": "not_run", "message": message,
                "quality": {}, "topDimensions": [], "branches": {},
                "preAnalysis": []}

    result = ica["result"]
    scan = getattr(result, "scan_summary", None)
    scan_rows = []
    if isinstance(scan, pd.DataFrame):
        for row in scan.to_dict(orient="records"):
            scan_rows.append({str(k): _clean(v) for k, v in row.items()})
    profiles_obj = getattr(result, "stability_profiles", {})
    if isinstance(profiles_obj, pd.DataFrame):
        profiles = {
            str(int(dimension)): [_clean(v) for v in frame.sort_values("component_rank")["stability_index"]]
            for dimension, frame in profiles_obj.groupby("n_components")
        }
    else:
        profiles = {
            str(k): [_clean(v) for v in values]
            for k, values in (profiles_obj or {}).items()
        }
    selection = getattr(result, "selection", None)
    selection_data = {}
    if selection is not None:
        selection_data = {str(k): _clean_deep(v) for k, v in vars(selection).items()}
    top_dimensions = [int(d) for d in getattr(result, "persisted_dimensions", ())]
    roles_obj = getattr(result, "dimension_roles", {}) or {}
    dimension_roles = {
        str(int(dimension)): [str(role) for role in roles]
        for dimension, roles in roles_obj.items()
    }
    params = getattr(result, "params", {}) or {}
    tested_dimensions = params.get("candidate_dimensions")
    if tested_dimensions is None and isinstance(scan, pd.DataFrame) and "n_components" in scan:
        tested_dimensions = scan["n_components"].dropna().astype(int).tolist()
    payload = {
        "status": "complete", "message": None,
        "gseaEnabled": bool(ica.get("gseaEnabled", False)),
        "quality": {
            "testedDimensions": [int(x) for x in (tested_dimensions or [])],
            "nRuns": _clean(params.get("n_runs")),
            "mostStableDimension": int(getattr(result, "mstd")),
            "topDimensions": top_dimensions,
            "dimensionRoles": dimension_roles,
            "scan": scan_rows, "stabilityProfiles": profiles,
            "selection": selection_data,
        },
        "topDimensions": top_dimensions, "branches": {},
        "preAnalysis": _ica_preanalysis(outdir),
    }

    for dimension, branch in (ica.get("branches") or {}).items():
        dim = int(dimension)
        projection = branch["projection"].copy()
        projection.index = projection.index.astype(str)
        component_names = [str(c) for c in projection.columns]
        stability = branch.get("stability")
        if isinstance(stability, pd.DataFrame):
            if {"component", "stability_index"}.issubset(stability.columns):
                indexed = stability.set_index("component")["stability_index"]
                comp_stability = [_clean(v) for v in indexed.reindex(component_names)]
            else:
                comp_stability = []
        elif isinstance(stability, pd.Series):
            comp_stability = [_clean(v) for v in stability.reindex(component_names)]
        elif stability is None:
            comp_stability = []
        else:
            comp_stability = [_clean(v) for v in stability]

        top_genes = {}
        metagenes = branch.get("metagenes")
        if isinstance(metagenes, pd.DataFrame):
            for component in component_names:
                if component not in metagenes.index:
                    continue
                vals = pd.to_numeric(metagenes.loc[component], errors="coerce")
                sel = vals.abs().sort_values(ascending=False).head(12).index
                top_genes[component] = [
                    {"gene": str(g), "loading": _clean(vals.loc[g])} for g in sel]

        # Les projections sont distinctes pour chaque K car elles sont apprises
        # sur D_K = 1 - C_K. Le champ historique ``embed`` reste le K final
        # pour conserver la compatibilité avec les rapports plus anciens.
        embed_by_k = {
            str(int(k)): _embedding_payload(coords_k, projection.index)
            for k, coords_k in (branch.get("coords_by_k") or {}).items()
        }
        embed = embed_by_k.get(
            str(int(branch["k_final"])),
            _embedding_payload(branch.get("coords"), projection.index),
        )
        # Espaces de distance alternatifs (euclidien, corrélation, Manhattan)
        # calculés sur les scores de composantes : indépendants de K.
        embed_by_distance = {
            str(name): _embedding_payload(coords_m, projection.index)
            for name, coords_m in (branch.get("coords_by_distance") or {}).items()
        }

        meta, meta_types = fallback_meta, fallback_meta_types
        branch_meta = branch.get("meta")
        if isinstance(branch_meta, pd.DataFrame):
            meta, meta_types = _meta_payload(branch_meta, projection.index)

        branch_samples = list(projection.index)
        payload["branches"][str(dim)] = {
            "samples": branch_samples, "n": int(len(projection)),
            "selectionRoles": dimension_roles.get(str(dim), []),
            "meta": meta, "metaTypes": meta_types,
            "projection": {
                "componentNames": component_names,
                "scores": [[_clean(v) for v in projection.loc[s]] for s in projection.index],
                "componentStability": comp_stability, "topGenes": top_genes,
            },
            "metageneGsea": _ica_metagene_gsea_payload(
                branch.get("metagene_gsea")
            ),
            "clusterComparisons": _clean_deep(
                branch.get("cluster_comparisons") or {}
            ),
            "consensus": _consensus_payload(
                branch["result"], branch["k_final"], linkage_method,
                min_cluster_size, k_criterion,
                branch.get("branch_stability_by_k")),
            "embed": embed, "embedByK": embed_by_k,
            "embedByDistance": embed_by_distance,
            "assoc": branch.get("assoc") or {},
            "corr": _corr_payload(branch.get("corr")),
        }
    return payload


def _gather(res, outdir):
    # déballage du conteneur (cf. results.PipelineResults)
    result, k_final = res.result, res.k_final
    coords, coords_by_k, meta = res.coords, (res.coords_by_k or {}), res.meta
    sig_scores, sig_tests, deconv = res.sig_scores, res.sig_tests, res.deconv
    sig_provenance = res.sig_provenance
    degsea_by_k, clinical_degsea = res.degsea_by_k, res.clinical_degsea
    linkage_method, min_cluster_size, k_criterion = (
        res.linkage_method, res.min_cluster_size, res.k_criterion)

    outdir = Path(outdir)
    # Bloc consensus (matrices par k, arbres, PAC, stabilité des branches) : la
    # branche historique et les branches ICA passent par le MÊME sérialiseur,
    # sans quoi les deux vues du rapport peuvent diverger sans que rien ne le
    # signale. `bestBoth` et `samples` en ressortent pour la suite.
    data = _consensus_payload(result, k_final, linkage_method, min_cluster_size,
                              k_criterion, res.branch_stability_by_k)
    samples, best_both = data["samples"], data["bestBoth"]

    # signatures (signatures × échantillons). `sources` = collection d'origine
    # (source du signature_sources : IPRES / sigGeNeHetX / select…) alignée sur
    # `names` -> permet au rapport de regrouper les signatures par collection.
    sig_src_map = {}
    if sig_provenance is not None and len(sig_provenance):
        sig_src_map = {str(s): str(src) for s, src in
                       zip(sig_provenance["signature"], sig_provenance["source"])}
    data["signatures"] = {}
    for method, df in (sig_scores or {}).items():
        df = df.reindex(columns=samples)
        names = [str(x) for x in df.index]
        data["signatures"][method] = {
            "names": names,
            "sources": [sig_src_map.get(nm, "") for nm in names],
            "values": [[_clean(v) for v in df.loc[nm]] for nm in df.index],
        }

    # tests de Wilcoxon score de signature × modalité (one-vs-rest ET pairwise),
    # pour les étoiles/barres au-dessus des boxplots. Structure :
    #   {method:{stratKey:{kkey:{sig:{"ovr":{group:p},"pair":{a:{b:p}}}}}}}
    #   stratKey "__cluster__" : kkey = str(k) ; variable clinique : kkey = "*"
    def _clean_sig(o):
        def _mm(key):   # {group: val} nettoyé
            return {str(g): _clean(p) for g, p in (o.get(key) or {}).items()}
        def _nn(key):   # {a: {b: val}} nettoyé
            return {str(a): {str(b): _clean(p) for b, p in inner.items()}
                    for a, inner in (o.get(key) or {}).items()}
        return {"ovr": _mm("ovr"), "pair": _nn("pair"),
                "ovrq": _mm("ovrq"), "pairq": _nn("pairq")}   # q = FDR (BH)
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
    # Un seul k (le recommandé PAC+Δ(K)) en mode normal, tous les k si
    # --degsea_all_k y.
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
    degsea_default_k = _degsea_default_k(
        int(k_final), int(best_both), degsea_by_k
    )
    data["degseaK"] = degsea_default_k
    data["degseaDetail"] = _degsea_detail_payload(
        outdir, degsea_default_k, degsea_by_k
    )
    data["clinicalDegseaDetail"] = _clinical_degsea_detail_payload(
        outdir, clinical_degsea
    )

    # OUTRIDER (7b) : un run par sous-groupe, choisi dans un menu du rapport.
    data["outrider"] = _outrider_payload(res.outrider,
                                         enabled=bool(res.outrider))

    # métadonnées cliniques (vides et identifiants uniques écartés)
    data["meta"], data["metaTypes"] = _meta_payload(meta, samples)

    # `filter_columns` a déjà restreint les métadonnées en amont (cf.
    # run_pipeline._restrict_metadata) : data["meta"] ne contient donc plus que
    # les colonnes autorisées, et aucune vue du rapport ne peut en afficher
    # d'autres. Ce champ ne sert plus qu'à fixer l'ORDRE des menus de filtrage
    # sur celui du YAML, plus parlant qu'un ordre alphabétique quand les colonnes
    # ont une hiérarchie (histo_classe, subclass, subtype).
    wanted = [str(c) for c in (getattr(res, "filter_columns", ()) or ())]
    if wanted:
        data["filterColumns"] = [c for c in wanted if c in data["meta"]]
        missing = [c for c in wanted if c not in data["meta"]]
        if missing:
            logger.warning("filter_columns : %d colonne(s) ignorée(s), absentes des "
                           "métadonnées exploitables du rapport — %s",
                           len(missing), ", ".join(missing))
    else:
        data["filterColumns"] = None          # null = aucune restriction

    # Embeddings propres à chaque K, calculés sur D_K = 1 - C_K. ``embed``
    # reste volontairement un alias du K final pour les rapports existants.
    data["embedByK"] = {
        str(int(k)): _embedding_payload(coords_k, samples)
        for k, coords_k in coords_by_k.items()
    }
    data["embed"] = data["embedByK"].get(
        str(int(k_final)), _embedding_payload(coords, samples)
    )
    data["embedByDistance"] = {
        str(name): _embedding_payload(coords_m, samples)
        for name, coords_m in (res.coords_by_distance or {}).items()
    }

    # tables (toutes les .csv sous outdir/tables) — sauf celles qui feraient
    # doublon avec un onglet dédié. Les sorties OUTRIDER sont déjà dans
    # `data["outrider"]`, et son `counts.csv` est la matrice d'ENTRÉE envoyée à
    # py_outrider : l'embarquer reviendrait à recopier les counts bruts dans le
    # rapport (mesuré : 4,1 Mo sur 6,2 pour cinq sous-groupes). Seul `plan.csv`,
    # qui récapitule les runs retenus et écartés, reste consultable ici.
    data["tables"] = {}
    tdir = outdir / "tables"
    if tdir.exists():
        for f in sorted(tdir.rglob("*.csv")):
            rel_parts = f.relative_to(tdir).parts
            if rel_parts and rel_parts[0] == "outrider" and f.name != "plan.csv":
                continue
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

    data["assoc"] = res.assoc or {}

    # 9b corrélations : table précalculée complète (heatmap bloc×bloc + scatter)
    data["corr"] = _corr_payload(res.corr)

    # Branche ICA : payload séparé, même si l'ICA n'a pas été activée. Ainsi le
    # rapport garde une structure stable et peut afficher un état vide explicite.
    data["ica"] = _ica_payload(
        res.ica, outdir, linkage_method=linkage_method,
        min_cluster_size=min_cluster_size, k_criterion=k_criterion,
        fallback_meta=data["meta"], fallback_meta_types=data["metaTypes"],
    )
    return data


def build_report(res, outdir) -> Path:
    """Construit `outdir/report.html` à partir d'un `results.PipelineResults`."""
    outdir = Path(outdir)
    data = _gather(res, outdir)
    html = _TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    out = outdir / "report.html"
    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size / 1e6
    logger.info("Rapport HTML : %s (%.1f Mo)", out, size)
    return out
