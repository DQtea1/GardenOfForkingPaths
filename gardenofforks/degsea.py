"""Étapes 7 et 13 — DEGSEA : expression différentielle (DESeq2) + GSEA.

Deux consommateurs, un seul moteur : le DEGSEA **clinique** (étape 7,
`run_clinical_degsea_group`) et le DEGSEA **par cluster** (étape 13,
`run_degsea`). Ce qui suit décrit le second.

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
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as cf
from . import preprocessing as pp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Filtrage des gènes en entrée d'un DEGSEA
# --------------------------------------------------------------------------
#: normalisation des options « liste » (motifs, chemins) — définie une seule fois
#: dans :mod:`gardenofforks.config`, qui porte déjà la logique YAML/CLI.
_as_patterns = cf.as_str_tuple


@dataclass(frozen=True)
class DegseaFilters:
    """Filtrage appliqué aux counts **bruts** juste avant DESeq2.

    Le DEGSEA part de la matrice brute, jamais de la matrice prétraitée du
    clustering (DESeq2 modélise des comptages). Ces filtres rejouent donc, pour
    l'analyse différentielle, les mêmes familles de filtres que l'étape 2 — mais
    avec leurs propres seuils, parce que le bon compromis n'est pas le même pour
    du clustering et pour un test gène par gène.

    Tous portent sur les gènes, sauf le dernier — `zero_samples` — qui retire des
    **tumeurs**, et qui doit rester en dernier : il juge chaque tumeur sur le jeu
    de gènes finalement testé.
    """
    low_counts: bool = False
    min_count: int = 15
    min_frac_samples: float = 0.3
    technical: bool = False
    technical_patterns: tuple[str, ...] = pp.TECHNICAL_PATTERNS
    unmapped: bool = False
    unmapped_patterns: tuple[str, ...] = pp.UNMAPPED_PATTERNS
    variable: bool = False
    n_top_genes: int = 5000
    variance_method: str = "mad"
    zero_samples: bool = False
    max_zero_frac: float = 0.8

    @classmethod
    def from_args(cls, args, prefix: str = "clinical_degsea") -> DegseaFilters:
        """Construit les réglages à partir des options `<prefix>_*` du YAML."""
        def opt(name, default=None):
            # `null` dans le YAML doit reprendre le défaut, pas propager None.
            value = getattr(args, f"{prefix}_{name}", default)
            return default if value is None else value

        return cls(
            low_counts=opt("filter_low_counts", "n") == "y",
            min_count=int(opt("min_count_per_sample", 15)),
            min_frac_samples=float(opt("min_frac_samples", 0.3)),
            technical=opt("filter_technical", "n") == "y",
            technical_patterns=_as_patterns(opt("technical_patterns"),
                                            pp.TECHNICAL_PATTERNS),
            unmapped=opt("filter_unmapped", "n") == "y",
            unmapped_patterns=_as_patterns(opt("unmapped_patterns"),
                                           pp.UNMAPPED_PATTERNS),
            variable=opt("select_variable", "n") == "y",
            n_top_genes=int(opt("n_top_genes", 5000)),
            variance_method=str(opt("variance_method", "mad")),
            zero_samples=opt("filter_zero_samples", "n") == "y",
            max_zero_frac=float(opt("max_zero_frac", 0.8)),
        )

    def apply(self, counts: pd.DataFrame) -> pd.DataFrame:
        """Applique les filtres actifs à `counts` (échantillons × gènes bruts).

        L'ordre compte : on retire d'abord ce qui n'a pas à être testé (gènes
        techniques, identifiants non annotés), puis on filtre sur l'expression,
        ensuite on sélectionne les plus variables — sur ce qui reste — et
        **enfin** on retire les tumeurs quasi vides sur ces gènes-là.
        """
        n_samples0, n0 = counts.shape
        if self.technical:
            counts = pp.drop_genes_matching(counts, self.technical_patterns,
                                            "techniques")
        if self.unmapped:
            counts = pp.drop_genes_matching(counts, self.unmapped_patterns,
                                            "non annotés")
        if self.low_counts:
            counts = pp.filter_low_counts(counts, self.min_count,
                                          self.min_frac_samples)
        if self.variable:
            # Le classement se fait sur du logCPM : sur des counts bruts, la
            # variance suit la moyenne et la profondeur de librairie, elle ne
            # mesure rien de biologique. Les counts bruts sont ensuite sous-
            # ensemblés sur les gènes retenus.
            ranked = pp.select_variable_genes(pp.log_cpm(counts),
                                              self.n_top_genes,
                                              self.variance_method)
            counts = counts.loc[:, ranked.columns]
        if self.zero_samples:
            # EN DERNIER, volontairement : une tumeur est jugée sur les gènes
            # réellement testés, pas sur la matrice de départ.
            counts = pp.filter_zero_samples(counts, self.max_zero_frac)
        if counts.shape[1] != n0:
            logger.info("DEGSEA : %d / %d gènes conservés après filtrage.",
                        counts.shape[1], n0)
        if counts.shape[0] != n_samples0:
            logger.info("DEGSEA : %d / %d tumeurs conservées après filtrage.",
                        counts.shape[0], n_samples0)
        if not counts.shape[0]:
            raise ValueError(
                "Filtrage DEGSEA : aucune tumeur ne survit au filtre "
                f"max_zero_frac={self.max_zero_frac} — desserre le seuil.")
        if not counts.shape[1]:
            raise ValueError(
                "Filtrage DEGSEA : aucun gène ne survit aux filtres configurés "
                "— desserre min_count / min_frac_samples.")
        return counts


@dataclass(frozen=True)
class DeseqSettings:
    """Garde-fous numériques de l'ajustement DESeq2.

    Deux familles de protections, contre deux pathologies distinctes.

    **Cook / valeurs aberrantes.** La distance de Cook mesure l'influence d'un
    échantillon sur le coefficient d'un gène. Un seul comptage extrême — une
    tumeur contaminée, un artefact de mapping — suffit à créer un « gène
    différentiel » qui n'existe que par lui.
      - `refit_cooks` (côté ajustement) remplace ces comptages par la moyenne
        tronquée des autres, puis réajuste. N'agit qu'à partir de
        `min_replicates` = 7 échantillons ;
      - `cooks_filter` (côté test) met `padj` à NA pour tout gène dont un
        échantillon dépasse le seuil de Cook, plutôt que de le déclarer
        significatif. C'est l'équivalent du `cooksCutoff` de DESeq2 en R.

    **Dispersion effondrée.** La statistique de Wald vaut `log2FC / lfcSE`, et
    `lfcSE` croît comme la racine de la dispersion. Si l'estimation de dispersion
    échoue et tombe de plusieurs ordres de grandeur, l'erreur-type s'effondre et
    |z| explose sans que l'effet ait bougé — c'est la « barre » de gènes
    infiniment significatifs en haut d'un volcano. Observé en vrai : PMEL passant
    d'une dispersion de 4,5 à 2,5e-05 par le simple ajout d'une covariable.
      - `min_disp` relève le plancher que PyDESeq2 s'autorise ; c'est une
        atténuation, pas un remède ;
      - `flag_low_dispersion` marque les gènes dont la dispersion ajustée est
        sous `min_plausible_disp`. Une tumeur en bulk RNA-seq a typiquement une
        dispersion de 0,1 à 0,5 : en dessous de 1e-3, l'estimation est douteuse,
        pas précise. Ces gènes sont signalés dans la table et écartés de
        l'étiquetage du volcano — mais **jamais supprimés en silence**.
    """
    cooks_filter: bool = True
    refit_cooks: bool = True
    independent_filter: bool = True
    min_disp: float = 1e-8
    flag_low_dispersion: bool = True
    min_plausible_disp: float = 1e-3

    @classmethod
    def from_args(cls, args, prefix: str = "clinical_degsea") -> DeseqSettings:
        def opt(name, default):
            value = getattr(args, f"{prefix}_{name}", default)
            return default if value is None else value

        return cls(
            cooks_filter=opt("cooks_filter", "y") == "y",
            refit_cooks=opt("refit_cooks", "y") == "y",
            independent_filter=opt("independent_filter", "y") == "y",
            min_disp=float(opt("min_disp", 1e-8)),
            flag_low_dispersion=opt("flag_low_dispersion", "y") == "y",
            min_plausible_disp=float(opt("min_plausible_disp", 1e-3)),
        )

    def annotate(self, table: pd.DataFrame, dds) -> pd.DataFrame:
        """Ajoute la dispersion ajustée et le drapeau de dispersion douteuse."""
        table = table.copy()
        disp = pd.Series(dds.var["dispersions"], index=dds.var_names)
        table["dispersion"] = disp.reindex(table.index).to_numpy()
        suspect = table["dispersion"] < self.min_plausible_disp
        table["dispersion_suspecte"] = suspect.fillna(False) if self.flag_low_dispersion \
            else False
        n = int(table["dispersion_suspecte"].sum())
        if n:
            logger.warning(
                "DESeq2 : %d gène(s) à dispersion < %.0e — erreur-type effondrée, "
                "|z| non interprétable. Marqués `dispersion_suspecte` et exclus de "
                "l'étiquetage du volcano. Ex. : %s", n, self.min_plausible_disp,
                ", ".join(map(str, table.index[table["dispersion_suspecte"]][:8])))
        return table


# --------------------------------------------------------------------------
# Briques : DESeq2 et GSEA sur un contraste
# --------------------------------------------------------------------------
def deseq2_model(counts: pd.DataFrame, metadata: pd.DataFrame, design: str,
                 contrast: tuple[str, str, str],
                 n_cpus: int | None = None,
                 settings: DeseqSettings | None = None) -> pd.DataFrame:
    """Ajuste un modèle DESeq2 général et renvoie un contraste catégoriel.

    ``design`` est une formule PyDESeq2, par exemple ``"~ age + response"``.
    ``contrast`` suit le format ``(variable, test, control)`` : le log2FC est
    donc ``test / control``. Les counts et métadonnées doivent être strictement
    alignés et sans valeur manquante pour les variables du design.

    L'ajustement et l'extraction sont délégués à :func:`fit_deseq2` et
    :func:`contrast_from_fit` : les garde-fous de ``settings`` (Cook, plancher de
    dispersion, marquage) s'appliquent donc ici comme au DEGSEA clinique.
    """
    if not isinstance(design, str) or not design.strip().startswith("~"):
        raise ValueError("design DESeq2 invalide : une formule du type '~ age + response' est attendue.")
    if len(contrast) != 3 or not contrast[0]:
        raise ValueError("contrast doit être un triplet (variable, test, control).")
    counts = counts.copy()
    metadata = metadata.copy()
    counts.index = counts.index.astype(str)
    metadata.index = metadata.index.astype(str)
    if counts.index.has_duplicates or metadata.index.has_duplicates:
        raise ValueError("Les identifiants échantillons DESeq2 doivent être uniques.")
    if not counts.index.equals(metadata.index):
        raise ValueError("counts et metadata doivent être strictement alignés avant DESeq2.")
    variable, test, control = map(str, contrast)
    if variable not in metadata.columns:
        raise ValueError(f"Variable de contraste absente des métadonnées : {variable!r}.")

    settings = settings or DeseqSettings()
    dds = fit_deseq2(counts, metadata, design, n_cpus=n_cpus, settings=settings)
    return contrast_from_fit(dds, variable, test, control, n_cpus=n_cpus,
                             settings=settings)


def contrast_from_fit(dds, variable: str, test: str, control: str,
                      n_cpus: int | None = None,
                      settings: DeseqSettings | None = None) -> pd.DataFrame:
    """Tire un contraste d'un modèle **déjà ajusté**.

    L'ajustement (`dds.deseq2()`) domine largement le coût ; extraire un
    contraste supplémentaire du même modèle est presque gratuit. Plusieurs
    contrastes partageant une formule doivent donc partager leur ajustement.
    """
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    settings = settings or DeseqSettings()
    inference = DefaultInference(n_cpus=n_cpus)
    with contextlib.redirect_stdout(io.StringIO()):
        st = DeseqStats(dds, contrast=[str(variable), str(test), str(control)],
                        cooks_filter=settings.cooks_filter,
                        independent_filter=settings.independent_filter,
                        quiet=True, inference=inference)
        st.summary()
    table = st.results_df.sort_values("stat", ascending=False)
    return settings.annotate(table, dds)


def fit_deseq2(counts: pd.DataFrame, metadata: pd.DataFrame, design: str,
               n_cpus: int | None = None,
               settings: DeseqSettings | None = None):
    """Ajuste un `DeseqDataSet` sans extraire de contraste."""
    settings = settings or DeseqSettings()
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference

    int_counts = counts.round().astype(int)
    size_factors_fit_type = "ratio"
    if (int_counts.to_numpy() == 0).any(axis=0).all():
        size_factors_fit_type = "poscounts"
        logger.info(
            "DESeq2 : aucun gène non nul sur les %d échantillons — size factors "
            "estimés par 'poscounts' (médiane des ratios inapplicable).",
            int_counts.shape[0],
        )
    with contextlib.redirect_stdout(io.StringIO()):
        dds = DeseqDataSet(counts=int_counts, metadata=metadata, design=design,
                           size_factors_fit_type=size_factors_fit_type,
                           min_disp=settings.min_disp,
                           refit_cooks=settings.refit_cooks,
                           quiet=True, inference=DefaultInference(n_cpus=n_cpus))
        dds.deseq2()
    return dds


def deseq2_contrast(counts: pd.DataFrame, groups: pd.Series,
                    target: str, ref: str, n_cpus: int | None = None,
                    settings: DeseqSettings | None = None) -> pd.DataFrame:
    """Contraste DESeq2 simple ``~ group`` (un cluster contre une référence).

    La version générique :func:`deseq2_model` porte les covariables cliniques et
    les formules arbitraires.
    """
    metadata = pd.DataFrame({"group": groups.astype(str).to_numpy()}, index=counts.index)
    return deseq2_model(
        counts, metadata, design="~ group",
        contrast=("group", str(target), str(ref)), n_cpus=n_cpus,
        settings=settings,
    )


def gsea_prerank_scores(scores: pd.Series, gene_sets: str | Path,
                        permutations: int = 1000, min_size: int = 15,
                        max_size: int = 500, threads: int = 4,
                        seed: int = 0) -> pd.DataFrame | None:
    """GSEA pré-classé sur un vecteur de scores signés indexé par gène.

    Cette brique est indépendante de DESeq2 : ``scores`` peut contenir une
    statistique de Wald, les poids d'un métagène ICA ou tout autre classement
    continu signé. Les doublons d'identifiants sont moyennés afin de fournir à
    GSEA un rang unique par gène.

    Renvoie ``gseapy.prerank(...).res2d`` (Term, NES, NOM p-val, FDR q-val,
    Lead_genes…) ou ``None`` si le calcul est indisponible.
    """
    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy absent : GSEA sauté (`pip install gseapy`).")
        return None

    if not isinstance(scores, pd.Series):
        scores = pd.Series(scores)
    ranked = pd.to_numeric(scores, errors="coerce").dropna()
    ranked.index = ranked.index.astype(str)
    if ranked.index.has_duplicates:
        ranked = ranked.groupby(level=0, sort=False).mean()
    ranked = ranked.sort_values(ascending=False)
    rnk = ranked.rename_axis("gene").reset_index()
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
        logger.warning("GSEA pré-classé échoué : %s", exc)
        return None


def gsea_prerank(results_df: pd.DataFrame, gene_sets: str | Path,
                 permutations: int = 1000, min_size: int = 15,
                 max_size: int = 500, threads: int = 4,
                 seed: int = 0) -> pd.DataFrame | None:
    """GSEA pré-classé sur la statistique de Wald DESeq2."""
    if "stat" not in results_df:
        raise ValueError("Le résultat DESeq2 doit contenir une colonne 'stat'.")
    return gsea_prerank_scores(
        results_df["stat"], gene_sets, permutations=permutations,
        min_size=min_size, max_size=max_size, threads=threads, seed=seed,
    )


def resolve_gene_sets(gene_sets) -> dict[str, str]:
    """Normalise des collections GMT et ignore les chemins inexistants."""
    if isinstance(gene_sets, (str, Path)):
        gene_sets = {Path(gene_sets).stem: str(gene_sets)}
    resolved = {}
    for name, path in (gene_sets or {}).items():
        p = Path(os.path.expanduser(str(path)))
        if p.exists():
            resolved[str(name)] = str(p)
        else:
            logger.warning("GSEA : gene set introuvable, ignoré : %s (%s)", name, p)
    return resolved


# Alias interne conservé pour compatibilité avec les appels historiques.
_resolve_gene_sets = resolve_gene_sets


def contrast_group_sizes(metadata: pd.DataFrame, design: str, contrast: str,
                         control: str, test: str) -> tuple[int, int, int]:
    """Effectifs (control, test, exclus) d'un contraste, SANS ajuster DESeq2.

    Rejoue la sélection d'échantillons de :func:`run_clinical_degsea_group` —
    modalités du contraste, puis complétude des variables du design — pour permettre
    d'écarter une combinaison inexploitable avant d'y consacrer du calcul. Le
    filtrage des gènes pouvant encore retirer des tumeurs, ces effectifs sont une
    borne supérieure ; la vérification définitive reste dans l'ajustement.
    """
    design_vars = _design_variables(design, metadata.columns)
    values = metadata[contrast].astype("string")
    keep = values.isin([str(control), str(test)]) & \
        metadata[design_vars].notna().all(axis=1)
    kept = values[keep]
    return (int((kept == str(control)).sum()), int((kept == str(test)).sum()),
            int((~keep).sum()))


def _design_variables(design: str, metadata_columns) -> list[str]:
    """Variables de métadonnées explicitement citées dans une formule simple.

    Les noms des colonnes sont recherchés comme identifiants complets, ce qui
    couvre ``~ age + response`` et ``~ C(batch) + response``. Les formules avec
    transformations complexes restent acceptées par PyDESeq2, mais les valeurs
    manquantes doivent alors être traitées par l'appelant avant cette fonction.
    """
    return [str(column) for column in metadata_columns
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(column))}(?![A-Za-z0-9_])", design)]


def reduce_design(design: str, metadata: pd.DataFrame,
                  min_level_count: int = 1) -> tuple[str, list[str], dict[str, str]]:
    """Retire de la formule les termes inexploitables **sur ce sous-ensemble**.

    Une formule figée ne survit pas à la stratification : si une mutation n'a
    aucun porteur dans un tissu, sa colonne y est constante, la matrice de design
    perd son rang et DESeq2 s'arrête sur « the model matrix is not full rank ».
    On construit donc la formule à partir de ce qui est réellement présent.

    Règles, par terme du design :
      - covariable **numérique** à plus de deux valeurs -> continue, conservée ;
      - variable **catégorielle** -> conservée si au moins deux modalités sont
        présentes, chacune portée par >= `min_level_count` tumeurs ;
      - sinon écartée, avec le motif.

    Renvoie la formule réduite, les termes conservés et ``{terme: motif}``.
    """
    kept, dropped = [], {}
    for term in _design_variables(design, metadata.columns):
        values = metadata[term].dropna()
        if values.empty:
            dropped[term] = "aucune valeur renseignée"
            continue
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > 2:
            kept.append(term)                      # covariable continue
            continue
        sizes = values.astype(str).value_counts()
        if len(sizes) < 2:
            dropped[term] = f"constante ({sizes.index[0]!r})"
        elif int(sizes.min()) < min_level_count:
            dropped[term] = (f"modalité {sizes.idxmin()!r} à {int(sizes.min())} "
                             f"tumeur(s) < {min_level_count}")
        else:
            kept.append(term)
    return ("~ " + " + ".join(kept)) if kept else "~ 1", kept, dropped


def _as_model_dtypes(meta: pd.DataFrame, terms, levels: dict) -> pd.DataFrame:
    """Prépare les colonnes du design pour PyDESeq2.

    `_load_metadata` produit des colonnes en ``StringDtype`` — indispensable pour
    préserver les ``<NA>`` lors de la sélection des échantillons, mais PyDESeq2
    ne reconnaît comme catégorielles que ``object`` et ``category``. Une colonne
    laissée en ``string`` traverse jusqu'à la matrice de design, qui finit en
    ``dtype('O')``, et le contrôle de rang échoue sur un obscur
    « Cannot cast ufunc 'svd' input from dtype('O') ».
    """
    meta = meta.copy()
    for term in terms:
        if term in levels:                          # variable de contraste
            control, test = levels[term]
            meta[term] = pd.Categorical(meta[term].astype(str),
                                        categories=[control, test], ordered=True)
        elif not pd.api.types.is_numeric_dtype(meta[term]):
            meta[term] = meta[term].astype(str)     # object, pas StringDtype
    return meta


def run_clinical_degsea_group(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    design: str,
    specs: dict[str, dict],
    min_count: int = 10,
    gene_filters: DegseaFilters | None = None,
    permutations: int = 1000,
    n_jobs: int = 1,
    seed: int = 0,
    min_level_count: int = 1,
    settings: DeseqSettings | None = None,
) -> dict:
    """Ajuste **une seule fois** le modèle d'une strate, puis en tire chaque contraste.

    Plusieurs expériences cliniques partagent souvent la même formule et ne
    diffèrent que par la variable contrastée. Les ajuster séparément refait le
    même calcul autant de fois : l'ajustement domine le coût, l'extraction d'un
    contraste supplémentaire est presque gratuite. Les résultats sont identiques,
    pas seulement équivalents.

    La formule est d'abord **réduite** aux termes exploitables sur cette strate
    (cf. :func:`reduce_design`) : sans ça, une mutation absente du tissu rendrait
    la matrice de design singulière.

    `specs` : ``{nom: {contrast, control, test, gene_sets, outdir, min_group}}``.
    Renvoie ``{design_used, kept_terms, dropped_terms, n_samples, results, skipped}``.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata doit être un DataFrame indexé par échantillon.")
    if not isinstance(design, str) or not design.strip().startswith("~"):
        raise ValueError("design clinique invalide : attendu, par exemple, '~ age + response'.")

    counts = counts.copy()
    counts.index = counts.index.astype(str)
    metadata = metadata.copy()
    metadata.index = metadata.index.astype(str)
    if counts.index.has_duplicates or metadata.index.has_duplicates:
        raise ValueError("Les identifiants échantillons counts/métadonnées doivent être uniques.")
    meta = metadata.reindex(counts.index)

    for name, spec in specs.items():
        if str(spec["control"]) == str(spec["test"]):
            raise ValueError(f"{name} : control et test doivent différer.")
        if str(spec["contrast"]) not in meta.columns:
            raise ValueError(f"{name} : contraste absent des métadonnées — "
                             f"{spec['contrast']!r}.")

    reduced, kept, dropped_terms = reduce_design(design, meta, min_level_count)
    if dropped_terms:
        logger.info("Design réduit sur cette strate : %s  (écartés : %s)", reduced,
                    " ; ".join(f"{k} — {v}" for k, v in dropped_terms.items()))

    skipped = {name: f"terme écarté du design : {dropped_terms.get(str(spec['contrast']), 'absent')}"
               for name, spec in specs.items() if str(spec["contrast"]) not in kept}
    usable = {name: spec for name, spec in specs.items() if name not in skipped}
    empty = {"design_used": reduced, "kept_terms": kept, "dropped_terms": dropped_terms,
             "n_samples": 0, "results": {}, "skipped": skipped}
    if not usable:
        return empty

    # Un contraste doit être binaire : on restreint aux tumeurs portant l'une des
    # deux modalités de CHAQUE contraste tiré de ce modèle, puis à celles dont
    # toutes les variables du design réduit sont renseignées.
    levels = {str(spec["contrast"]): (str(spec["control"]), str(spec["test"]))
              for spec in usable.values()}
    selected = pd.Series(True, index=meta.index)
    for column, (control, test) in levels.items():
        selected &= meta[column].astype("string").isin([control, test])
    keep_samples = selected & meta[kept].notna().all(axis=1)
    dropped_samples = int((~keep_samples).sum())

    cnt = counts.loc[keep_samples].round().astype(int)
    meta = meta.loc[keep_samples]
    # Filtres appliqués APRÈS la sélection : les seuils d'expression portent sur
    # la cohorte réellement testée. Le dernier peut retirer des tumeurs.
    if gene_filters is not None:
        cnt = gene_filters.apply(cnt)
        if len(cnt.index) != len(meta.index):
            meta = meta.loc[cnt.index]
            dropped_samples = int(len(counts.index) - len(cnt.index))

    meta = _as_model_dtypes(meta, kept, levels)
    keep_genes = cnt.columns[cnt.sum(axis=0) >= int(min_count)]
    if not len(keep_genes):
        raise ValueError("Aucun gène ne passe le filtre de counts pour le contraste clinique.")

    threads = ((os.cpu_count() or 1) if n_jobs in (-1, 0, None)
               else max(1, int(n_jobs)))
    logger.info("DESeq2 clinique : ajustement unique sur %d tumeurs, %d gènes, "
                "design=%s ; %d contraste(s) à en tirer.",
                len(meta), len(keep_genes), reduced, len(usable))
    settings = settings or DeseqSettings()
    dds = fit_deseq2(cnt.loc[:, keep_genes], meta[kept], reduced, n_cpus=threads,
                     settings=settings)

    results: dict[str, dict] = {}
    for name, spec in usable.items():
        contrast = str(spec["contrast"])
        control, test = str(spec["control"]), str(spec["test"])
        sizes = meta[contrast].value_counts()
        n_control, n_test = int(sizes.get(control, 0)), int(sizes.get(test, 0))
        min_group = int(spec.get("min_group", 3))
        if n_control < min_group or n_test < min_group:
            skipped[name] = (f"Contraste {contrast}: {test} vs {control} : effectifs "
                             f"insuffisants ({test}={n_test}, {control}={n_control}; "
                             f"minimum={min_group}).")
            continue

        table = contrast_from_fit(dds, contrast, test, control, n_cpus=threads,
                                  settings=settings)
        outdir = Path(spec["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(outdir / "deseq2.csv", index_label="gene")
        meta.loc[:, kept].assign(**{contrast: meta[contrast].astype(str)}).to_csv(
            outdir / "samples_used.csv", index_label="sample")

        gsea = {}
        for collection, path in _resolve_gene_sets(spec.get("gene_sets")).items():
            found = gsea_prerank(table, path, permutations=permutations,
                                 threads=threads, seed=seed)
            if found is not None:
                found.to_csv(outdir / f"gsea_{collection}.csv", index=False)
            gsea[collection] = found

        results[name] = {
            "results": table, "gsea": gsea,
            "n_samples": int(len(meta)), "n_test": n_test, "n_control": n_control,
            "n_dropped": dropped_samples,
            "design_variables": kept, "design_used": reduced,
            "dropped_terms": dropped_terms,
        }
        logger.info("DESeq2 clinique : contraste %s (%d %s vs %d %s) extrait.",
                    contrast, n_test, test, n_control, control)

    return {"design_used": reduced, "kept_terms": kept, "dropped_terms": dropped_terms,
            "n_samples": int(len(meta)), "results": results, "skipped": skipped}


# SUPPRIMÉ — run_clinical_degsea (enveloppe à un seul contraste) et son
# exception InsufficientGroups : plus aucun appelant depuis que l'orchestrateur
# groupe les contrastes par formule pour partager l'ajustement. Un contraste
# unique s'obtient en passant un `specs` d'une seule entrée à
# run_clinical_degsea_group. Code dans l'historique git.


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _run_one(cnt: pd.DataFrame, groups: pd.Series, target: str, ref: str,
             tag: str, scheme: str, de_dir: Path, gs_dir: Path, gene_sets: dict,
             permutations, min_count: int, threads: int, seed: int,
             settings: DeseqSettings | None = None) -> tuple[str, str, dict]:
    """Un contraste : DESeq2 **une fois**, puis GSEA **pour chaque collection**
    de `gene_sets` ({nom: chemin .gmt}). Renvoie (tag, scheme, {nom: res2d|None}).
    Pensé pour un worker joblib indépendant ; `threads` borne PyDESeq2 et GSEA."""
    keep = cnt.columns[cnt.sum(axis=0) >= min_count]
    res = deseq2_contrast(cnt[keep], groups, target, ref, n_cpus=threads,
                          settings=settings)
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
    gene_filters: DegseaFilters | None = None,
    permutations: int = 1000,
    heatmap_pval: float = 0.05,
    subdir: str = "",
    n_jobs: int = -1,
    seed: int = 0,
    settings: DeseqSettings | None = None,
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
    settings : garde-fous numériques DESeq2 (Cook, plancher de dispersion,
        marquage des dispersions invraisemblables). Voir :class:`DeseqSettings`.

    Renvoie `{collection: matrice NES (pathways × clusters)}` (one-vs-all), pour
    les heatmaps de synthèse — dict vide si aucun résultat GSEA.
    """
    # Normalisation gene_sets -> {nom: chemin}, en ne gardant que l'existant.
    gene_sets = _resolve_gene_sets(gene_sets)
    logger.info("DEGSEA : GSEA sur %d collection(s) : %s", len(gene_sets),
                ", ".join(gene_sets) or "aucune (DESeq2 seul)")

    base = Path(outdir) / "tables" / "degsea"
    if subdir:
        base = base / subdir      # une sous-arbo par k quand on balaie tous les k
    cnt = counts.loc[sample_names].round().astype(int)
    lab = pd.Series(np.asarray(labels), index=list(sample_names))
    # Filtrage appliqué UNE fois, sur la partition entière, et non par contraste :
    # tous les contrastes partagent alors le même univers de gènes, sans quoi les
    # NES d'une heatmap pathways × clusters seraient calculés sur des classements
    # de tailles différentes, donc non comparables d'un cluster à l'autre.
    # Le dernier filtre peut retirer des tumeurs : les labels suivent.
    if gene_filters is not None:
        cnt = gene_filters.apply(cnt)
        if len(cnt.index) != len(lab):
            lab = lab.loc[cnt.index]

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
                          permutations, min_count, inner_threads, seed,
                          settings or DeseqSettings())
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
