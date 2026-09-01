"""Étape 1a — harmonisation des identifiants de gènes vers les symboles HGNC.

Une matrice issue de la fusion de plusieurs lots peut mélanger plusieurs espaces
d'identifiants : Entrez NCBI (``7157``), Ensembl (``ENSG00000141510``), RefSeq
(``NM_000546``), symboles courants (``TP53``) et symboles périmés (``p53``,
``LFS1``). Comme la fusion se fait par jointure externe sur l'index, chaque lot
est **non nul uniquement sur son propre espace** et à zéro partout ailleurs : la
matrice est en blocs disjoints.

Les conséquences sont silencieuses et sévères :

  - tout filtre de prévalence (« exprimé chez >= 30 % des tumeurs ») élimine par
    construction l'espace minoritaire — un lot de 7 tumeurs sur 618 ne peut pas
    atteindre 30 % ;
  - les tumeurs de ce lot deviennent alors entièrement nulles, ce qui fait
    échouer DESeq2 (size factors ``poscounts`` = NaN) ;
  - GSEA, signatures et déconvolution ne reconnaissent que les symboles HGNC.

Deux sous-étapes, correspondant aux deux temps du pipeline :

  1. :func:`id_type_report` — **diagnostic** : quel identifiant est de quel type,
     et quelle tumeur porte ses comptes sur quel bloc. Ne modifie rien.
  2. :func:`harmonize` — **conversion** vers les symboles approuvés, les lignes
     retombant sur le même symbole étant additionnées.

La table de correspondance est le **HGNC complete gene set** de genenames.org,
c'est-à-dire exactement le jeu de données que le paquet R ``hgnc`` télécharge via
``import_hgnc_dataset()`` et interroge via ``crosswalk()``. :func:`build_crosswalk`
en est l'équivalent Python : mêmes colonnes source, même opération, sans
dépendance à R. Il couvre en une passe ``entrez_id``, ``ensembl_gene_id``,
``refseq_accession``, ainsi que ``alias_symbol`` et ``prev_symbol`` — ces deux
dernières permettant de ramener aussi les symboles périmés à leur forme actuelle.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: jeu de données HGNC complet (celui qu'utilise le paquet R `hgnc`)
HGNC_URL = ("https://storage.googleapis.com/public-download-files/hgnc/"
            "tsv/tsv/hgnc_complete_set.txt")

#: emplacement du cache local, pour ne télécharger qu'une fois
HGNC_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) \
    / "gardenofforks" / "hgnc_complete_set.txt"

# [0-9] et pas \d : \d matcherait aussi les chiffres unicode ("٣", "²").
# L'ordre compte : il fait office de priorité dans classify_ids.
ID_PATTERNS: dict[str, str] = {
    "entrez": r"[0-9]+(\.0+)?",                        # 7157, ou 7157.0 après un float
    "ensembl": r"ENS[A-Z]*[GTP][0-9]{6,}(\.[0-9]+)?",  # ENSG00000141510.17
    "refseq": r"(N|X)(M|R|P)_[0-9]+(\.[0-9]+)?",       # NM_000546.6
    "symbol": r"[A-Za-z][A-Za-z0-9\-\.@_]*",           # TP53, MT-CO1, HLA-DRB1
}

#: types d'identifiants qu'une correspondance peut convertir en symbole
MAPPABLE = ("entrez", "ensembl", "refseq", "symbol")

#: colonnes du jeu HGNC utilisées comme source, et type d'identifiant associé
_CROSSWALK_COLUMNS = {
    "entrez_id": "entrez",
    "ensembl_gene_id": "ensembl",
    "refseq_accession": "refseq",
}


# --------------------------------------------------------------------------
# Sous-étape 1 : diagnostic
# --------------------------------------------------------------------------
def _as_text(ids) -> pd.Series:
    """Index -> Series de chaînes, sans jamais laisser de NaN (NaN/None -> "")."""
    def conv(v):
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v).strip()

    return pd.Series([conv(v) for v in np.asarray(ids, dtype=object)], dtype="string")


def classify_ids(ids) -> pd.Series:
    """Étiquette chaque identifiant : entrez / ensembl / refseq / symbol / vide / autre.

    L'ordre de :data:`ID_PATTERNS` fait la priorité : un identifiant purement
    numérique est un Entrez, pas un symbole, même si le motif « symbole » est
    permissif.
    """
    ids = pd.Index(ids)
    txt = _as_text(ids)
    out = np.full(len(txt), "autre", dtype=object)
    out[(txt.str.len() == 0).to_numpy()] = "vide"
    for name, pattern in ID_PATTERNS.items():
        free = out == "autre"
        if not free.any():
            break
        hit = txt.str.fullmatch(pattern).fillna(False).to_numpy()
        out[free & hit] = name
    return pd.Series(out, index=ids, name="id_type")


def normalize_id(value: str, id_type: str) -> str:
    """Forme canonique d'un identifiant, pour la mise en correspondance.

    Retire le suffixe de version d'Ensembl/RefSeq (``ENSG…​.17`` -> ``ENSG…``) et
    la décimale parasite d'un Entrez passé par un float (``7157.0`` -> ``7157``).
    """
    text = str(value).strip()
    if id_type == "entrez":
        return text.split(".")[0]
    if id_type in ("ensembl", "refseq"):
        return text.split(".")[0].upper()
    if id_type == "symbol":
        return text.upper()
    return text


def id_type_report(counts: pd.DataFrame, min_count: int = 0) -> dict:
    """Diagnostic complet : types d'identifiants, et bloc porteur de chaque tumeur.

    `counts` est orienté **tumeurs × gènes** (la sortie de `preprocessing.load_matrix`).
    Renvoie un dict contenant :

    ``id_types``   Series type par gène
    ``by_type``    DataFrame : nombre et part des identifiants de chaque type
    ``by_sample``  DataFrame : par tumeur, part des comptes portée par chaque type,
                   le type dominant, et le nombre de gènes non nuls par type
    ``mixed``      True si plusieurs types portent effectivement des comptes
    """
    id_types = classify_ids(counts.columns)
    present = sorted(set(id_types) - {"vide"})

    n = len(id_types)
    by_type = pd.DataFrame({
        "n_ids": id_types.value_counts(),
        "pct_ids": (100 * id_types.value_counts() / max(n, 1)).round(2),
    }).reindex(sorted(set(id_types))).fillna(0)

    matrix = counts.to_numpy()
    totals = matrix.sum(axis=1)
    rows: dict[str, pd.Series] = {}
    for t in present:
        mask = (id_types == t).to_numpy()
        block = matrix[:, mask]
        rows[f"somme_{t}"] = pd.Series(block.sum(axis=1), index=counts.index)
        rows[f"n_nonzero_{t}"] = pd.Series((block > min_count).sum(axis=1),
                                           index=counts.index)
    by_sample = pd.DataFrame(rows)
    for t in present:
        by_sample[f"pct_{t}"] = (
            100 * by_sample[f"somme_{t}"] / np.where(totals > 0, totals, 1)
        ).round(2)
    share = by_sample[[f"pct_{t}" for t in present]]
    by_sample["type_dominant"] = (
        share.idxmax(axis=1).str.removeprefix("pct_") if len(present) else "aucun"
    )
    by_sample["total"] = totals

    carrying = [t for t in present
                if by_sample.get(f"somme_{t}", pd.Series(dtype=float)).sum() > 0]
    return {"id_types": id_types, "by_type": by_type, "by_sample": by_sample,
            "types_presents": present, "types_porteurs": carrying,
            "mixed": len(carrying) > 1}


def log_report(report: dict, log: logging.Logger | None = None, top: int = 8) -> None:
    """Journalise le diagnostic sous une forme lisible."""
    log = log or logger
    by_type = report["by_type"]
    log.info("Identifiants de gènes : %s",
             " | ".join(f"{t} {int(r.n_ids)} ({r.pct_ids:.1f} %)"
                        for t, r in by_type.iterrows() if r.n_ids))
    if not report["mixed"]:
        log.info("Un seul espace d'identifiants porte des comptes (%s).",
                 report["types_porteurs"][0] if report["types_porteurs"] else "aucun")
        return

    counts_by_dom = report["by_sample"]["type_dominant"].value_counts()
    log.warning("Matrice MIXTE : %d espaces d'identifiants portent des comptes. "
                "Répartition des tumeurs par espace dominant : %s",
                len(report["types_porteurs"]),
                ", ".join(f"{t} : {n}" for t, n in counts_by_dom.items()))
    minority = counts_by_dom.index[-1]
    names = report["by_sample"].index[report["by_sample"]["type_dominant"] == minority]
    log.warning("Tumeurs sur l'espace minoritaire (%s) : %s%s", minority,
                ", ".join(map(str, names[:top])),
                "" if len(names) <= top else f" … (+{len(names) - top})")


# --------------------------------------------------------------------------
# Sous-étape 2 : jeu HGNC, crosswalk, conversion
# --------------------------------------------------------------------------
def load_hgnc_dataset(path: str | Path | None = None,
                      cache: str | Path | None = None) -> pd.DataFrame:
    """Charge le *HGNC complete gene set*, en le téléchargeant au besoin.

    Équivalent Python de ``hgnc::import_hgnc_dataset()`` : même fichier, même
    source. `path` force un fichier local (utile hors ligne ou pour figer une
    version) ; sinon le fichier est mis en cache sous
    ``~/.cache/gardenofforks/`` et n'est téléchargé qu'une fois.
    """
    if path:
        target = Path(path).expanduser()
        if not target.exists():
            raise FileNotFoundError(
                f"harmonize_hgnc_file : fichier introuvable — {target}")
    else:
        target = Path(cache) if cache else HGNC_CACHE
        target = target.expanduser()
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Téléchargement du jeu HGNC (une seule fois) : %s -> %s",
                        HGNC_URL, target)
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(HGNC_URL, tmp)   # noqa: S310 — URL en dur
            tmp.replace(target)

    table = pd.read_csv(target, sep="\t", dtype=str, low_memory=False)
    missing = {"symbol"} - set(table.columns)
    if missing:
        raise ValueError(f"{target} : colonne(s) HGNC absente(s) — {sorted(missing)}. "
                         "Attendu le 'HGNC complete gene set' de genenames.org.")
    if "status" in table.columns:
        table = table[table["status"].fillna("Approved") == "Approved"]
    logger.info("Jeu HGNC : %d symboles approuvés (%s)", len(table), target.name)
    return table


def _split(value: str) -> list[str]:
    """Découpe un champ HGNC multi-valué (``"NCRNA00181|A1BGAS|A1BG-AS"``)."""
    return [p.strip() for p in str(value).split("|") if p.strip()]


def build_crosswalk(hgnc: pd.DataFrame, map_aliases: bool = True) -> dict[str, str]:
    """Construit ``{identifiant normalisé: symbole approuvé}`` depuis le jeu HGNC.

    Équivalent de ``hgnc::crosswalk()`` appliqué à toutes les colonnes source
    d'un coup. Les symboles approuvés sont enregistrés en premier et ne sont
    jamais écrasés ; viennent ensuite les identifiants de bases externes, puis —
    si `map_aliases` — les symboles précédents et les alias.

    Les alias ambigus (pointant vers plusieurs symboles approuvés) sont écartés :
    renommer sur une correspondance ambiguë fusionnerait des gènes distincts.
    """
    symbols = hgnc["symbol"].dropna().astype(str)
    crosswalk: dict[str, str] = {normalize_id(s, "symbol"): s for s in symbols}
    approved = set(crosswalk)

    def add_column(column: str, kind: str) -> None:
        """Ajoute une colonne source, en refusant toute correspondance ambiguë.

        HGNC contient de vraies ambiguïtés : l'Entrez 125775236 est porté par
        HILPDA-AS1 ET EFCAB3P1. Prendre la première ligne venue renommerait
        silencieusement un gène en un autre ; on préfère ne pas convertir.
        """
        if column not in hgnc.columns:
            return
        targets: dict[str, set[str]] = {}
        for raw, symbol in zip(hgnc[column], hgnc["symbol"]):
            if pd.isna(raw) or pd.isna(symbol):
                continue
            for part in _split(raw):
                key = normalize_id(part, kind)
                if key in approved:       # un symbole approuvé prime toujours
                    continue
                targets.setdefault(key, set()).add(str(symbol))
        ambiguous = 0
        for key, candidates in targets.items():
            if len(candidates) > 1:
                ambiguous += 1
                continue
            crosswalk.setdefault(key, next(iter(candidates)))
        if ambiguous:
            logger.info("crosswalk : %d %s ambigus ignorés (plusieurs symboles "
                        "approuvés candidats).", ambiguous, column)

    for column, kind in _CROSSWALK_COLUMNS.items():
        add_column(column, kind)
    if map_aliases:
        # prev_symbol avant alias_symbol : un symbole retiré est une information
        # plus forte qu'un simple synonyme.
        for column in ("prev_symbol", "alias_symbol"):
            add_column(column, "symbol")

    logger.info("Crosswalk HGNC : %d identifiants reconnus.", len(crosswalk))
    return crosswalk


def harmonize(counts: pd.DataFrame, crosswalk: dict[str, str],
              id_types: pd.Series | None = None,
              drop_unmapped: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ramène les colonnes de `counts` (tumeurs × gènes) aux symboles HGNC approuvés.

    Les identifiants convertis qui retombent sur un symbole déjà présent sont
    **additionnés** : c'est le comportement correct sur une matrice en blocs
    disjoints, où un lot porte ses comptes sur un espace et zéro sur l'autre.

    Renvoie la matrice harmonisée et une table de traçabilité (un identifiant
    d'origine par ligne : son type, sa cible, et s'il a été converti).
    """
    if id_types is None:
        id_types = classify_ids(counts.columns)

    kinds, targets, converted, recognised = [], [], [], []
    for gene in counts.columns:
        kind = id_types.get(gene, "autre")
        symbol = crosswalk.get(normalize_id(gene, kind)) if kind in MAPPABLE else None
        kinds.append(kind)
        targets.append(symbol if symbol else gene)
        recognised.append(symbol is not None)
        # « converti » = le nom a changé ; un symbole déjà à jour est reconnu
        # sans être modifié, et ne compte donc pas comme une conversion.
        converted.append(symbol is not None and str(symbol) != str(gene))

    trace = pd.DataFrame({"id_origine": list(map(str, counts.columns)),
                          "type": kinds, "symbole": targets,
                          "reconnu": recognised, "converti": converted})

    keep = np.ones(len(targets), dtype=bool)
    if drop_unmapped:
        keep = trace["reconnu"].to_numpy()
        n_drop = int((~keep).sum())
        if n_drop:
            logger.warning("Harmonisation : %d identifiant(s) inconnus du jeu HGNC "
                           "retirés (harmonize_drop_unmapped=y).", n_drop)

    out = counts.loc[:, keep]
    out.columns = pd.Index([t for t, k in zip(targets, keep) if k],
                           name=counts.columns.name)

    n_dup = int(out.columns.duplicated().sum())
    if n_dup:
        # Vérifie que la fusion n'additionne pas deux mesures d'une même tumeur :
        # sur des blocs disjoints, l'un des deux termes est toujours nul.
        dup_names = out.columns[out.columns.duplicated()].unique()
        overlap = 0
        for name in dup_names[:200]:                      # borné : simple garde-fou
            block = out.loc[:, out.columns == name]
            overlap += int(((block > 0).sum(axis=1) > 1).sum())
        if overlap:
            logger.warning("Harmonisation : %d couple(s) tumeur×symbole ont des "
                           "comptes dans PLUSIEURS espaces d'identifiants ; leurs "
                           "valeurs sont additionnées. Vérifie la fusion des lots.",
                           overlap)
        out = out.T.groupby(level=0).sum().T
        logger.info("Harmonisation : %d symbole(s) issus de plusieurs identifiants "
                    "fusionnés par somme.", n_dup)

    n_conv = int(trace["converti"].sum())
    n_unknown = int((~trace["reconnu"]).sum())
    logger.info("Harmonisation : %d identifiant(s) renommés en symbole ; "
                "matrice %d tumeurs x %d gènes (avant : %d).",
                n_conv, out.shape[0], out.shape[1], counts.shape[1])
    if n_unknown:
        logger.warning("Harmonisation : %d identifiant(s) inconnus du jeu HGNC — "
                       "conservés tels quels (harmonize_drop_unmapped=n).", n_unknown)
    return out, trace


__all__ = [
    "HGNC_URL", "HGNC_CACHE", "ID_PATTERNS", "MAPPABLE",
    "classify_ids", "normalize_id", "id_type_report", "log_report",
    "load_hgnc_dataset", "build_crosswalk", "harmonize",
]
