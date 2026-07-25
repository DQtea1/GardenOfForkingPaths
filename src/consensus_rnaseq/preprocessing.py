"""Prétraitement d'une matrice bulk RNA-seq avant consensus clustering.

Convention interne : toutes les fonctions renvoient une matrice
`samples x genes` (patients en lignes), qui est le format attendu par
scikit-learn et par le module `consensus`.
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
def load_matrix(path: str | Path, genes_in_rows: bool = True) -> pd.DataFrame:
    """Charge une matrice d'expression (csv/tsv/parquet) -> `samples x genes`."""
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep, index_col=0)

    if genes_in_rows:
        df = df.T
    df = df.astype(np.float64)
    logger.info("Matrice chargée : %d échantillons x %d gènes", *df.shape)
    return df


# --------------------------------------------------------------------------
# Filtrage / normalisation
# --------------------------------------------------------------------------
def drop_zero_count_genes(counts: pd.DataFrame) -> pd.DataFrame:
    """Retire les gènes à count nul dans **toutes** les tumeurs.

    Filtre de base indépendant de `min_cpm`/`min_frac_samples` : un gène sans
    aucun compte n'apporte aucune information et fait planter certaines étapes
    en aval (VST de DESeq2 notamment, dont l'estimation des size factors exige
    des gènes non nuls partout). À appliquer sur des *counts bruts*, avant tout
    filtrage ou normalisation.
    """
    keep = (counts.values > 0).any(axis=0)
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info("Gènes à 0 count dans toutes les tumeurs retirés : %d / %d",
                    n_dropped, len(keep))
    return counts.loc[:, keep]


def filter_low_expression(
    counts: pd.DataFrame, min_cpm: float = 1.0, min_frac_samples: float = 0.2
) -> pd.DataFrame:
    """Garde les gènes exprimés (>= `min_cpm` CPM) dans >= `min_frac_samples`
    des échantillons. À appliquer sur des *counts bruts*."""
    lib = counts.sum(axis=1).values[:, None]
    cpm = counts.values / np.maximum(lib, 1) * 1e6
    keep = (cpm >= min_cpm).mean(axis=0) >= min_frac_samples
    logger.info("Filtrage expression : %d / %d gènes conservés", keep.sum(), len(keep))
    return counts.loc[:, keep]


def drop_technical_genes(
    expr: pd.DataFrame,
    patterns: tuple[str, ...] = ("^RP[LS]", "^MT-", "^MRP[LS]", "^HB[ABDEGMQZ]\\d?$"),
) -> pd.DataFrame:
    """Retire ribosomiques / mitochondriaux / hémoglobines, qui dominent
    souvent la variance et créent des clusters purement techniques."""
    mask = np.zeros(expr.shape[1], dtype=bool)
    for pat in patterns:
        mask |= expr.columns.str.match(pat, case=False, na=False)
    logger.info("Gènes techniques retirés : %d", int(mask.sum()))
    return expr.loc[:, ~mask]


def log_cpm(counts: pd.DataFrame, prior_count: float = 1.0) -> pd.DataFrame:
    """log2(CPM + prior). Alternative rapide au VST de DESeq2.

    Si tu disposes déjà d'une matrice VST/rlog (recommandé pour du clustering),
    passe `--already-normalized` dans le pipeline et saute cette étape.
    """
    lib = counts.sum(axis=1).values[:, None]
    cpm = counts.values / np.maximum(lib, 1) * 1e6
    return pd.DataFrame(
        np.log2(cpm + prior_count), index=counts.index, columns=counts.columns
    )


def vst_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    """Variance Stabilizing Transformation de DESeq2, via PyDESeq2.

    C'est la normalisation **par défaut** pour des counts bruts (préférable au
    logCPM interne pour du clustering) : elle stabilise la variance sur toute la
    gamme d'expression, en particulier les gènes faiblement exprimés dont la
    variance de comptage dominerait sinon la distance entre tumeurs. Attend des
    *counts bruts* (entiers) `samples x genes`.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyDESeq2 absent : `pip install pydeseq2` pour la normalisation VST "
            "(ou choisis norm_method=logcpm)."
        ) from exc

    if (counts.values < 0).any():
        raise ValueError("VST attend des counts positifs ; valeurs négatives détectées "
                         "(la matrice est peut-être déjà normalisée ? -> --already-normalized).")

    int_counts = counts.round().astype(int)
    metadata = pd.DataFrame({"_dummy": "a"}, index=int_counts.index)
    with contextlib.redirect_stdout(io.StringIO()):   # ceinture + bretelles avec quiet=True
        dds = DeseqDataSet(counts=int_counts, metadata=metadata, design="~1", quiet=True)
        dds.vst()
        vst = dds.layers["vst_counts"]
    logger.info("Normalisation VST (DESeq2/PyDESeq2) : %d tumeurs x %d gènes", *vst.shape)
    return pd.DataFrame(vst, index=counts.index, columns=counts.columns)


def select_variable_genes(
    expr: pd.DataFrame, n_top: int = 5000, method: str = "mad"
) -> pd.DataFrame:
    """Sélectionne les `n_top` gènes les plus variables.

    `mad` (median absolute deviation) est plus robuste que la variance en
    présence de quelques échantillons extrêmes — utile sur des tumeurs
    atypiques où un outlier ne doit pas piloter la sélection.
    """
    X = expr.values
    if method == "mad":
        score = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
    elif method == "var":
        score = X.var(axis=0)
    else:
        raise ValueError(f"method inconnue : {method}")

    n_top = min(n_top, expr.shape[1])
    idx = np.argsort(score)[::-1][:n_top]
    idx.sort()
    logger.info("Sélection : %d gènes les plus variables (%s)", n_top, method)
    return expr.iloc[:, idx]


def center_genes(expr: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
    """Centrage (médiane) gène par gène, comme dans Monti et al. 2003.

    Le centrage est indispensable si la distance de base est euclidienne.
    `scale=True` ajoute une réduction (z-score) : à éviter si tu veux garder
    l'amplitude d'expression comme information discriminante.
    """
    X = expr.values.copy()
    X -= np.median(X, axis=0)
    if scale:
        mad = np.median(np.abs(X), axis=0)
        X /= np.maximum(mad * 1.4826, 1e-8)
    return pd.DataFrame(X, index=expr.index, columns=expr.columns)


def pca_outliers(
    expr: pd.DataFrame,
    sd_threshold: float,
    n_pc: int = 10,
    min_explained_var: float = 0.0,
    random_state: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Détecte les tumeurs aberrantes dans l'espace de l'ACP par leur écart-type.

    Une ACP est calculée sur la matrice prétraitée (`samples x genes`). Pour
    chacune des composantes inspectées, les scores sont standardisés (z-score) ;
    une tumeur est déclarée aberrante si `|z|` dépasse `sd_threshold` sur au moins
    une composante. C'est le contrôle qualité classique en transcriptomique
    (« échantillon à plus de N SD sur une CP »).

    Composantes inspectées :
      - par défaut, les `n_pc` premières ;
      - si `min_explained_var > 0`, on prend plutôt **toutes** les composantes
        dont la variance expliquée individuelle dépasse ce seuil (fraction ]0,1[ ;
        une valeur > 1 est interprétée en pourcentage, p. ex. 8 -> 8 %).

    `sd_threshold <= 0` (ou non renseigné) ne retire personne.

    Renvoie
    -------
    keep : np.ndarray booléen aligné sur `expr.index` (True = tumeur conservée).
    diag : DataFrame de diagnostic (scores PC1/PC2, z max, CP responsable, flag).
    """
    n = expr.shape[0]
    diag = pd.DataFrame({"sample": expr.index.to_numpy()})

    if not sd_threshold or sd_threshold <= 0:
        diag["PC1"] = np.nan
        diag["PC2"] = np.nan
        diag["max_abs_z"] = 0.0
        diag["worst_pc"] = 0
        diag["is_outlier"] = False
        return np.ones(n, dtype=bool), diag

    use_var = bool(min_explained_var) and min_explained_var > 0
    if use_var and min_explained_var > 1:       # accepte "8" pour 8 %
        min_explained_var = min_explained_var / 100.0

    n_fit = (n - 1) if use_var else n_pc
    n_fit = max(1, int(min(n_fit, n - 1, expr.shape[1])))
    pca = PCA(n_components=n_fit, random_state=random_state)
    scores = pca.fit_transform(expr.values)

    if use_var:
        sel = np.flatnonzero(pca.explained_variance_ratio_ > min_explained_var)
        if sel.size == 0:
            sel = np.array([0])
            logger.warning("Aucune CP au-dessus de %.1f %% de variance ; "
                           "PC1 utilisée par défaut.", 100 * min_explained_var)
        comp_desc = "%d CP > %.1f %% var" % (sel.size, 100 * min_explained_var)
    else:
        sel = np.arange(scores.shape[1])
        comp_desc = "%d premières CP" % sel.size

    sub = scores[:, sel]
    sd = sub.std(axis=0, ddof=0)
    z = np.abs((sub - sub.mean(axis=0)) / np.where(sd > 0, sd, 1.0))
    max_abs_z = z.max(axis=1)
    is_outlier = max_abs_z > sd_threshold

    diag["PC1"] = scores[:, 0]
    diag["PC2"] = scores[:, 1] if scores.shape[1] > 1 else np.nan
    diag["max_abs_z"] = max_abs_z
    diag["worst_pc"] = (sel[z.argmax(axis=1)] + 1).astype(int)
    diag["is_outlier"] = is_outlier

    logger.info(
        "ACP outliers : %d / %d tumeurs à plus de %.1f SD (%s)",
        int(is_outlier.sum()), n, sd_threshold, comp_desc,
    )
    return ~is_outlier, diag


def preprocess(
    counts: pd.DataFrame,
    already_normalized: bool = False,
    min_cpm: float = 1.0,
    min_frac_samples: float = 0.2,
    remove_technical: bool = True,
    n_top_genes: int = 5000,
    variance_method: str = "mad",
    center: bool = True,
    scale: bool = False,
    norm_method: str = "vst",
) -> pd.DataFrame:
    """Enchaîne le prétraitement complet et renvoie `samples x genes`.

    `norm_method` (counts bruts uniquement) : "vst" (DESeq2/PyDESeq2, défaut) ou
    "logcpm". Ignoré si `already_normalized`.
    """
    expr = counts
    if not already_normalized:
        expr = drop_zero_count_genes(expr)
        expr = filter_low_expression(expr, min_cpm, min_frac_samples)
        if norm_method == "vst":
            expr = vst_normalize(expr)
        elif norm_method == "logcpm":
            expr = log_cpm(expr)
        else:
            raise ValueError(f"norm_method inconnu : {norm_method!r} (vst | logcpm).")
    if remove_technical:
        expr = drop_technical_genes(expr)
    expr = select_variable_genes(expr, n_top_genes, variance_method)
    if center:
        expr = center_genes(expr, scale=scale)
    return expr
