#!/usr/bin/env python
"""Orchestration du pipeline : enchaîne les étapes et relie leurs dépendances.

Le texte d'aide de `gof-run`, la définition des options et la validation de la
configuration vivent dans :mod:`gardenofforks.config` ; ce module ne s'occupe
que de l'ordre des étapes et du passage des résultats de l'une à l'autre.

Ce module fait partie du paquet `gardenofforks` : il ne se lance pas par chemin
de fichier (`python gardenofforks/run_pipeline.py` casse les imports relatifs),
mais par `gof-run …` ou `python -m gardenofforks.run_pipeline …`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config as cf
from . import deconv as dc
from . import degsea as dg
from . import harmonize_ids as hid
from . import ica as ic
from . import ica_cluster_compare as icc
from . import ica_gsea as ig
from . import metrics as mt
from . import outrider as od
from . import plots as pl
from . import preprocessing as pp
from . import purity as pur
from . import report as rp
from . import sigproj as sp
from .analysis_branch import AnalysisBranch, BranchPaths, BranchSettings
from .config import ConfigError, load_config
from .results import PipelineResults


@dataclass
class _Ctx:
    """État d'orchestration : entrées, branche principale et sorties spécialisées."""
    args: object; log: object; outdir: Path; t_start: float; eff_n_jobs: int
    raw: object = None
    X_df: object = None
    metadata: object = None
    primary: AnalysisBranch | None = None
    sig_scores: object = None
    sig_provenance: object = None
    sig_tests: object = None
    ica_result: object = None
    degsea_by_k: dict = field(default_factory=dict)
    clinical_degsea: dict = field(default_factory=dict)
    outrider: dict = field(default_factory=dict)
    deconv: dict = field(default_factory=dict)
    ica_branches: dict = field(default_factory=dict)
    ica_metagene_gsea: dict = field(default_factory=dict)
    # branches ICA vivantes (dimension -> (branche, métasamples)), conservées
    # pour leur rejouer les analyses cliniques en fin de run.
    ica_branch_objects: dict = field(default_factory=dict)


def _setup(argv) -> _Ctx:
    # ------------------------------- 1. entrées / sorties (configuration, journal)
    args = load_config(argv)

    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    # journalisation : console (heure) + fichier .log horodaté (date complète)
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = outdir / log_path
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    _sh = logging.StreamHandler(sys.stderr)
    _sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                       datefmt="%H:%M:%S"))
    _fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(_sh)
    root.addHandler(_fh)
    log = logging.getLogger("pipeline")

    t_start = time.perf_counter()
    log.info("================ Démarrage du pipeline — %s ================",
             time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Config : %s | sortie : %s | journal : %s",
             args.config or "(ligne de commande)", outdir.resolve(), log_path.resolve())

    # parallel="n" force tout en séquentiel, quel que soit --n-jobs
    eff_n_jobs = args.n_jobs if args.parallel == "y" else 1
    if args.parallel == "n":
        log.info("Parallélisation désactivée (--parallel n) : n_jobs=1 partout.")
    return _Ctx(args=args, log=log, outdir=outdir, t_start=t_start, eff_n_jobs=eff_n_jobs)


def _preprocess_matrix(c: _Ctx, samples=None) -> None:
    """Étape 2 — construit la matrice du clustering à partir de `c.raw`.

    Séparée du chargement parce que l'harmonisation des identifiants (étape 3)
    réécrit l'index de `c.raw` : le prétraitement — dont le VST, de loin le plus
    cher — doit donc avoir lieu APRÈS elle, et une seule fois.

    ``samples`` restreint le calcul à une sous-cohorte : c'est ce qui permet de
    rejouer le prétraitement sur les seules tumeurs conservées (étape 2b).
    """
    args = c.args
    counts = c.raw if samples is None else c.raw.loc[list(samples)]
    c.X_df = pp.preprocess(
        counts,
        already_normalized=args.already_normalized,
        min_cpm=args.min_cpm,
        min_frac_samples=args.min_frac_samples,
        remove_technical=not args.keep_technical,
        n_top_genes=args.n_top_genes,
        variance_method=args.variance_method,
        center=True,
        scale=args.scale_genes,
        norm_method=args.norm_method,
    )
    c.log.info("Matrice prétraitée : %d tumeurs x %d gènes", *c.X_df.shape)


def _refit_matrix(c: _Ctx) -> None:
    """Étape 2b — rejoue le prétraitement sur les tumeurs réellement conservées.

    Le prétraitement (2) précède forcément la détection d'outliers (5), qui a
    besoin d'une matrice normalisée pour faire son ACP. Mais du coup TOUT ce que
    l'étape 2 décide — quels gènes passent le filtre de prévalence, les size
    factors du VST, la médiane de centrage et surtout la liste des `n_top_genes`
    les plus variables — est décidé en tenant compte de tumeurs qui viennent
    d'être écartées. Une tumeur aberrante peut donc choisir une partie du
    panneau de gènes… avant d'être jetée.

    On rejoue donc l'étape 2 sur la cohorte propre. Le coût est un second VST,
    payé uniquement si des tumeurs ont réellement été retirées.
    """
    args, log = c.args, c.log
    kept = list(c.X_df.index)
    dropped = len(c.raw.index) - len(kept)
    if args.refit_after_sample_filters != "y":
        if dropped:
            log.info("2b. Reprise du prétraitement désactivée "
                     "(refit_after_sample_filters = n) : le panneau de gènes reste "
                     "celui calculé avec les %d tumeur(s) écartée(s).", dropped)
        return
    if not dropped:
        return

    genes_before = set(c.X_df.columns)
    log.info("2b. Reprise du prétraitement sur les %d tumeurs conservées "
             "(%d écartée(s) aux étapes 4-5) : normalisation, gènes techniques et "
             "sélection des plus variables rejugés sans elles…", len(kept), dropped)
    _preprocess_matrix(c, samples=kept)
    genes_after = set(c.X_df.columns)
    log.info("2b. Panneau de gènes : %d conservés à l'identique, %d entrés, "
             "%d sortis (sur %d).", len(genes_before & genes_after),
             len(genes_after - genes_before), len(genes_before - genes_after),
             len(genes_before))


def _load_data(c: _Ctx) -> None:
    # ------------------------------------------- 1. chargement de la matrice
    c.raw = pp.load_matrix(c.args.counts, genes_in_rows=not c.args.samples_in_rows)


def _harmonize_gene_ids(c: _Ctx) -> None:
    """Étape 3 — ramène tous les identifiants de gènes aux symboles HGNC.

    Deux sous-étapes : un diagnostic des espaces d'identifiants présents, puis la
    conversion proprement dite. Le diagnostic est journalisé et exporté même
    quand la matrice s'avère homogène — c'est lui qui prouve qu'elle l'est.
    """
    args, log, outdir = c.args, c.log, c.outdir
    if args.harmonize_gene_ids != "y":
        return

    # ---- sous-étape 1 : de quel type est chaque identifiant, et qui les porte ?
    report = hid.id_type_report(c.raw)
    hid.log_report(report, log)
    report["by_type"].to_csv(outdir / "tables" / "gene_id_types.csv",
                             index_label="type_id")
    report["by_sample"].to_csv(outdir / "tables" / "gene_id_types_by_sample.csv",
                               index_label="sample")

    # ---- sous-étape 2 : conversion vers les symboles approuvés
    # Elle a lieu même sur une matrice homogène : un jeu tout en symboles peut
    # encore contenir des symboles périmés, que le crosswalk remet à jour.
    hgnc = hid.load_hgnc_dataset(args.harmonize_hgnc_file)
    crosswalk = hid.build_crosswalk(hgnc,
                                    map_aliases=args.harmonize_map_aliases == "y")
    harmonized, trace = hid.harmonize(
        c.raw, crosswalk, id_types=report["id_types"],
        drop_unmapped=args.harmonize_drop_unmapped == "y")
    trace.to_csv(outdir / "tables" / "gene_id_harmonization.csv", index=False)
    c.raw = harmonized
    log.info("Harmonisation terminée : le prétraitement (étape 2) part des "
             "identifiants harmonisés.")


def _load_metadata(c: _Ctx) -> None:
    """Charge une fois les métadonnées et normalise les identifiants échantillon."""
    if not c.args.metadata:
        c.metadata = None
        return
    # utf-8-sig : absorbe le BOM des exports Excel. Les tables cliniques
    # accentuées sont souvent en cp1252 : on retente plutôt que de planter.
    try:
        metadata = pd.read_csv(c.args.metadata, sep=None, engine="python",
                               index_col=0, encoding="utf-8-sig")
    except UnicodeDecodeError:
        metadata = pd.read_csv(c.args.metadata, sep=None, engine="python",
                               index_col=0, encoding="cp1252")
        c.log.warning("Métadonnées non-UTF-8 : relues en cp1252 — %s",
                      c.args.metadata)
    metadata.index = metadata.index.astype(str)

    # DESeq2 lit toute colonne numérique du design comme une covariable
    # continue : une variable de contraste codée 0/1 doit être en chaînes pour
    # être traitée comme catégorielle. `string` (et non `str`) laisse les NA
    # manquants, que le design clinique doit encore pouvoir exclure.
    # Colonnes déduites de la configuration, pas listées à la main : celles
    # citées comme `contrast` par les designs cliniques, et celles servant à
    # stratifier. Les secondes deviennent des libellés d'interface et des noms
    # de dossier, elles doivent donc être des chaînes elles aussi.
    contrasts = cf.contrast_columns(c.args.clinical_degsea)
    strata = cf.grouping_columns(c.args.group_DESeq2_by)
    columns = tuple(dict.fromkeys(contrasts + strata))
    missing = {"contrast": [x for x in contrasts if x not in metadata.columns],
               "group_DESeq2_by": [x for x in strata if x not in metadata.columns]}
    if any(missing.values()):
        raise ConfigError("colonne(s) absente(s) des métadonnées — " + " ; ".join(
            f"{key} : {', '.join(map(repr, cols))}"
            for key, cols in missing.items() if cols))
    for col in columns:
        values = metadata[col]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            # Un seul NA suffit à faire relire une colonne d'entiers en float :
            # sans ce détour par Int64, 0 deviendrait "0.0" et ne correspondrait
            # plus au `control: "0"` du YAML — contraste vide, sans erreur.
            finite = values.dropna()
            if finite.eq(finite.round()).all():
                values = values.astype("Int64")
        metadata[col] = values.astype("string")
        c.log.info("Contraste DESeq2 %r relu en chaînes : %s", col,
                   ", ".join(map(str, pd.unique(metadata[col].dropna()))))

    c.metadata = _restrict_metadata(c, metadata)


def _restrict_metadata(c: _Ctx, metadata: pd.DataFrame) -> pd.DataFrame:
    """Restreint les métadonnées aux colonnes de `filter_columns`.

    La coupe est faite **ici**, au plus tôt : tout le reste du pipeline — khi²,
    corrélations, associations, DEGSEA clinique, projection de signatures,
    rapport — consomme `c.metadata` directement ou via
    `AnalysisBranch.aligned_metadata()`. Une colonne retirée à cet endroit ne
    peut donc réapparaître nulle part, ni dans un calcul ni dans un menu.

    Les colonnes dont une ÉTAPE dépend sont vérifiées d'abord : variables des
    designs `clinical_degsea`, contrastes, colonnes de stratification et
    `color_by`. En retirer une ne provoquerait pas d'erreur — `_design_variables`
    ne trouverait simplement plus le terme — mais changerait silencieusement le
    modèle ajusté. On préfère refuser la configuration.
    """
    args, log = c.args, c.log
    allowed = cf.as_str_tuple(args.filter_columns)
    if not allowed:
        return metadata

    required: dict[str, set[str]] = {
        "contrast": set(cf.contrast_columns(args.clinical_degsea)),
        "group_DESeq2_by": set(cf.grouping_columns(args.group_DESeq2_by)),
        "color_by": {str(args.color_by)} if args.color_by else set(),
        "subset_by": set(cf.subset_columns(args.subset_by)),
    }
    designs = set()
    for spec in cf.clinical_experiments(args.clinical_degsea).values():
        designs |= set(dg._design_variables(str(spec["design"]), metadata.columns))
    required["design clinical_degsea"] = designs

    blocking = {key: sorted(cols - set(allowed)) for key, cols in required.items()}
    if any(blocking.values()):
        raise ConfigError(
            "filter_columns exclut des colonnes dont une étape dépend — ajoute-les "
            "à la liste, ou retire l'étape qui les utilise : " + " ; ".join(
                f"{key} : {', '.join(map(repr, cols))}"
                for key, cols in blocking.items() if cols))

    unknown = [col for col in allowed if col not in metadata.columns]
    if unknown:
        log.warning("filter_columns : %d colonne(s) demandée(s) mais absente(s) de "
                    "la table clinique — %s", len(unknown), ", ".join(unknown))

    keep = [col for col in metadata.columns if col in set(allowed)]
    log.info("filter_columns : %d / %d colonnes cliniques conservées ; %d écartée(s) "
             "pour tout le run.", len(keep), metadata.shape[1],
             metadata.shape[1] - len(keep))
    return metadata.loc[:, keep]


def _purity_filter(c: _Ctx) -> None:
    args, log, outdir = c.args, c.log, c.outdir
    raw, X_df = c.raw, c.X_df
    # -------------------------------------- 4. pureté tumorale (PUREE) + filtrage
    purity_thr = pur.parse_threshold(args.purity_threshold)
    if purity_thr is not None:
        purity = pur.run_puree(raw, args.puree_dir, args.puree_python,
                               gene_id_type=args.puree_gene_id)
        keep_p = pur.purity_keep_mask(purity, purity_thr, args.purity_direction)
        pd.DataFrame({"sample": purity.index, "purity": purity.to_numpy(),
                      "kept": keep_p.to_numpy()}).to_csv(
            outdir / "tables" / "purity_puree.csv", index=False)
        pl.plot_purity(purity, keep_p, purity_thr, args.purity_direction,
                       outdir / "figures")
        log.info("Pureté PUREE : médiane %.2f (min %.2f, max %.2f)",
                 purity.median(), purity.min(), purity.max())
        n_rm = int((~keep_p.to_numpy()).sum())
        if n_rm:
            keep_aligned = keep_p.reindex(X_df.index).fillna(True).to_numpy().astype(bool)
            log.warning("Filtrage pureté (%s %.2f) : %d / %d tumeurs retirées — %s",
                        args.purity_direction, purity_thr, n_rm, X_df.shape[0],
                        ", ".join(map(str, X_df.index[~keep_aligned])))
            X_df = X_df.loc[keep_aligned]
            log.info("Matrice après filtrage pureté : %d tumeurs x %d gènes", *X_df.shape)
        else:
            log.info("Aucune tumeur retirée au seuil de pureté %.2f.", purity_thr)
    c.X_df = X_df


def _outlier_filter(c: _Ctx) -> None:
    args, log, outdir = c.args, c.log, c.outdir
    X_df = c.X_df
    # -------------------------------------------- 5. filtrage d'outliers (ACP)
    if args.outlier_sd_threshold and args.outlier_sd_threshold > 0:
        keep, pca_diag = pp.pca_outliers(
            X_df, args.outlier_sd_threshold,
            n_pc=args.outlier_n_pc,
            min_explained_var=args.outlier_min_explained_var,
            random_state=args.seed,
        )
        pca_diag.to_csv(outdir / "tables" / "pca_outliers.csv", index=False)
        pl.plot_pca_outliers(pca_diag, outdir / "figures", args.outlier_sd_threshold)
        n_out = int((~keep).sum())
        if n_out:
            log.warning("Outliers ACP retirés (> %.1f SD) : %d / %d tumeurs — %s",
                        args.outlier_sd_threshold, n_out, X_df.shape[0],
                        ", ".join(map(str, X_df.index[~keep])))
            X_df = X_df.loc[keep]
            log.info("Matrice après filtrage : %d tumeurs x %d gènes", *X_df.shape)
        else:
            log.info("Aucun outlier ACP au seuil %.1f SD.", args.outlier_sd_threshold)
    c.X_df = X_df


def _ica(c: _Ctx) -> None:
    """Branche ICA indépendante, démarrée sur la matrice prétraitée.

    Elle ne lit ni n'écrit aucun résultat du consensus clustering principal :
    `run_ica` détermine d'abord la MSTD, conserve ses deux voisines testées et
    la meilleure stabilité moyenne, puis `_run_ica_branches` les analyse avec
    le flux commun.
    """
    args, log, outdir, eff_n_jobs = c.args, c.log, c.outdir, c.eff_n_jobs
    if args.run_ica != "y":
        log.info("8. ICA stabilisée désactivée (run_ica = n).")
        return

    log.info(
        "8. ICA stabilisée : dimensions %d..%d (pas %d), %d runs/dimension…",
        args.ica_n_components_min, args.ica_n_components_max,
        args.ica_n_components_step, args.ica_n_runs,
    )
    c.ica_result = ic.run_ica(
        c.X_df, outdir,
        min_components=args.ica_n_components_min,
        max_components=args.ica_n_components_max,
        step=args.ica_n_components_step,
        n_runs=args.ica_n_runs,
        top_n_dimensions=args.ica_top_dimensions,
        algorithm=args.ica_algorithm,
        fun=args.ica_fun,
        resampling=None if args.ica_resampling == "none" else args.ica_resampling,
        max_iter=args.ica_max_iter,
        n_jobs=eff_n_jobs,
        random_state=args.seed,
        deterministic=(args.ica_deterministic == "y"),
    )
    log.info(
        "ICA stabilisée terminée : MSTD=%d ; dimensions sauvegardées=%s",
        c.ica_result.mstd, list(c.ica_result.persisted_dimensions),
    )


def _run_ica_branches(c: _Ctx) -> None:
    """Lance le flux commun pour chacune des projections ICA persistées."""
    if c.ica_result is None:
        return

    settings = BranchSettings.from_args(c.args, n_jobs=c.eff_n_jobs)
    for dimension in c.ica_result.persisted_dimensions:
        dec = c.ica_result.decompositions[int(dimension)]
        paths = BranchPaths(
            root=c.outdir,
            table_dir=c.outdir / "tables" / "ica" / f"m{dimension}",
            figure_dir=c.outdir / "figures" / "ica" / f"m{dimension}",
            consensus_matrix_dir=c.outdir / "tables" / "ica" / f"m{dimension}",
            output_subdir=f"ica/m{dimension}",
        )
        branch = AnalysisBranch(
            name=f"ICA m={dimension}",
            matrix=dec.metasamples,
            paths=paths,
            settings=settings,
            metadata=c.metadata,
            forced_k=c.args.ica_k_final,
            forced_k_name="ica_k_final",
            input_export_name="ica_projection.csv",
            logger=c.log,
        ).run()
        # Les analyses cliniques attendent la fin du run : à cet instant, ni les
        # signatures ni la déconvolution n'existent encore.
        c.ica_branch_objects[int(dimension)] = (branch, dec.metasamples)
        labels_by_k = {
            int(k): branch.result.labels(int(k), c.args.linkage)
            for k in branch.k_values
        }
        c.log.info(
            "ICA m=%d — comparaisons inter-clusters des métasamples pour %d valeur(s) de k…",
            int(dimension), len(labels_by_k),
        )
        cluster_comparisons = icc.compare_ica_clusters(
            dec.metasamples,
            labels_by_k,
            paths.table_dir,
            min_cluster_size=c.args.min_cluster_size,
            clustering_method=c.args.base,
        )
        c.ica_branches[int(dimension)] = {
            "projection": branch.matrix,
            "stability": dec.stability.copy(),
            "metagenes": dec.metagenes,
            "metagene_gsea": c.ica_metagene_gsea.get(int(dimension), {}),
            "cluster_comparisons": cluster_comparisons,
            "result": branch.result,
            "k_values": branch.k_values,
            "k_final": int(branch.k_final),
            "labels": branch.labels,
            "items": branch.items,
            "coords": branch.coords,
            "coords_by_k": branch.coords_by_k,
            "coords_by_distance": branch.coords_by_distance,
            "meta": branch.aligned_metadata(),
            "color_var": branch.color_var,
            "assoc": branch.assoc,
            "corr": branch.corr,
            "branch_stability_by_k": branch.branch_stability_by_k,
        }


def _ica_metagene_gsea(c: _Ctx) -> None:
    """Annote par GSEA chaque métagène des dimensions ICA sauvegardées."""
    if c.ica_result is None:
        return
    if c.args.run_ica_gsea != "y":
        c.log.info("9. GSEA des métagènes ICA désactivé (run_ica_gsea = n).")
        return

    gene_sets = dg.resolve_gene_sets(
        cf.collections_or_fallback(c.args.gsea_collections, c.args.gsea_gene_sets)
    )
    if not gene_sets:
        c.log.warning(
            "9. GSEA des métagènes ICA demandé, mais aucune collection GMT "
            "existante n'est disponible ; étape sautée."
        )
        # Conserver toutes les dimensions et composantes dans le contrat de
        # résultats permet au rapport d'expliquer l'absence d'enrichissements.
        c.ica_metagene_gsea = {
            int(dimension): {
                str(component): {}
                for component in c.ica_result.decompositions[
                    int(dimension)
                ].metagenes.index
            }
            for dimension in c.ica_result.persisted_dimensions
        }
        return

    c.log.info(
        "9. Annotation GSEA des métagènes ICA : %d dimension(s), "
        "%d collection(s), %d permutations…",
        len(c.ica_result.persisted_dimensions), len(gene_sets),
        c.args.gsea_permutations,
    )
    selected_dimensions = tuple(
        int(dimension) for dimension in c.ica_result.persisted_dimensions
    )
    for dimension in selected_dimensions:
        dec = c.ica_result.decompositions[int(dimension)]
        roles = (getattr(c.ica_result, "dimension_roles", {}) or {}).get(
            int(dimension), ()
        )
        c.log.info(
            "GSEA ICA — m=%d (%s) : %d métagène(s)…",
            int(dimension), ", ".join(map(str, roles)) or "dimension retenue",
            len(dec.metagenes),
        )
        dimension_results = ig.run_ica_metagene_gsea(
            dec.metagenes,
            gene_sets,
            c.outdir / "tables" / "ica" / f"m{int(dimension)}",
            permutations=c.args.gsea_permutations,
            min_size=c.args.ica_gsea_min_size,
            max_size=c.args.ica_gsea_max_size,
            n_jobs=c.eff_n_jobs,
            seed=c.args.seed,
        )
        missing_components = set(map(str, dec.metagenes.index)) - set(
            dimension_results
        )
        if missing_components:
            raise RuntimeError(
                f"GSEA ICA m={dimension} : métagènes non traités : "
                + ", ".join(sorted(missing_components))
            )
        c.ica_metagene_gsea[int(dimension)] = dimension_results
    missing = set(selected_dimensions) - set(c.ica_metagene_gsea)
    if missing:  # garde-fou : aucune des dimensions retenues ne doit être omise
        raise RuntimeError(
            "GSEA ICA absent pour les dimensions sélectionnées : "
            + ", ".join(map(str, sorted(missing)))
        )
    c.log.info(
        "GSEA ICA terminé pour toutes les dimensions sélectionnées : %s.",
        list(selected_dimensions),
    )


def _run_primary_branch(c: _Ctx) -> None:
    """Lance la branche historique via le même flux que les branches ICA."""
    paths = BranchPaths(
        root=c.outdir,
        table_dir=c.outdir / "tables",
        figure_dir=c.outdir / "figures",
        consensus_matrix_dir=c.outdir,
    )
    c.primary = AnalysisBranch(
        name="Consensus Clustering",
        matrix=c.X_df,
        paths=paths,
        settings=BranchSettings.from_args(c.args, n_jobs=c.eff_n_jobs),
        metadata=c.metadata,
        forced_k=c.args.k_final,
        logger=c.log,
    ).run()


def _clinical_strata(c: _Ctx, counts) -> list[tuple[str, str, pd.Index]]:
    """Strates sur lesquelles rejouer toute la batterie de designs cliniques.

    Renvoie toujours en premier la strate non filtrée (`ALL`/`ALL`), puis une
    strate par modalité de chaque colonne de `group_DESeq2_by`. Les tumeurs sans
    valeur pour une colonne de stratification n'appartiennent à aucune de ses
    modalités : elles ne sont présentes que dans la strate non filtrée.
    """
    log, metadata = c.log, c.metadata
    index = counts.index.intersection(metadata.index)
    strata = [(cf.ALL_STRATA, cf.ALL_STRATA, index)]

    for col in cf.grouping_columns(c.args.group_DESeq2_by):
        values = metadata.loc[index, col]
        modalities = sorted(map(str, values.dropna().unique()))
        if not modalities:
            log.warning("group_DESeq2_by : la colonne %r n'a aucune modalité "
                        "exploitable — ignorée.", col)
            continue
        log.info("Stratification %r : %d modalité(s) — %s", col, len(modalities),
                 ", ".join(modalities[:10])
                 + ("" if len(modalities) <= 10 else f" … (+{len(modalities)-10})"))
        for modality in modalities:
            strata.append((col, modality,
                           values.index[values.astype("string") == modality]))
    return strata


def _clinical_degsea(c: _Ctx) -> None:
    """Exécute les expériences cliniques configurées, sans consensus clustering.

    Cette étape ne lit ni labels ni ``AnalysisBranch`` : elle consomme seulement
    les counts bruts, les métadonnées et le dictionnaire ``clinical_degsea``.

    Chaque design est rejoué sur chaque strate définie par `group_DESeq2_by`,
    plus la cohorte entière. Les combinaisons dont les effectifs sont trop
    faibles sont écartées **avant** tout ajustement : sur une stratification fine
    croisée avec des contrastes rares, elles sont la majorité, et les calculer
    pour rien coûterait des heures.
    """
    args, log = c.args, c.log
    c.clinical_degsea = {}
    # Trois états : 'n' coupe l'étape même si des designs sont configurés,
    # 'y' la demande explicitement, non renseigné = la présence d'un bloc
    # `clinical_degsea` décide (comportement historique).
    if args.run_clinical_degsea == "n":
        log.info("7. DEGSEA clinique désactivé (run_clinical_degsea = n) : les "
                 "designs du YAML ne sont pas rejoués.")
        return
    experiments = cf.clinical_experiments(args.clinical_degsea)
    if not experiments:
        if args.run_clinical_degsea == "y":
            log.warning("DEGSEA clinique demandé, mais aucune expérience clinical_degsea n'est configurée.")
        return
    # L'incompatibilité avec --already-normalized est refusée en amont par
    # config.validate ; seules les métadonnées ne sont connues qu'ici.
    if c.metadata is None:
        raise ValueError("DEGSEA clinique configuré, mais aucune table de métadonnées n'est fournie.")

    # Par défaut le DEGSEA clinique porte sur TOUTE la matrice : les tumeurs
    # écartées aux étapes 4 et 5 (pureté PUREE, outliers ACP) y reviennent, car ces
    # filtres sont jugés sur la matrice du clustering et n'engagent pas un
    # contraste clinique. `clinical_degsea_drop_pca_outliers: y` aligne les deux.
    counts = c.raw
    if args.clinical_degsea_drop_pca_outliers == "y":
        if c.X_df is None:
            log.warning("clinical_degsea_drop_pca_outliers=y mais aucune matrice "
                        "filtrée disponible — filtre ignoré.")
        else:
            kept = counts.index.intersection(c.X_df.index)
            n_out = counts.shape[0] - len(kept)
            counts = counts.loc[kept]
            log.info("DEGSEA clinique : aligné sur les tumeurs conservées aux "
                     "étapes 4 et 5 — %d tumeur(s) écartée(s), %d conservée(s).",
                     n_out, len(kept))

    gene_filters = dg.DegseaFilters.from_args(args)
    deseq_settings = dg.DeseqSettings.from_args(args)
    strata = _clinical_strata(c, counts)
    n_planned = len(strata) * len(experiments)
    log.info("7. DEGSEA clinique : %d strate(s) x %d design(s) = %d ajustement(s) "
             "DESeq2 au maximum.", len(strata), len(experiments), n_planned)

    plan: list[dict] = []
    for group_col, modality, index in strata:
        meta_stratum = c.metadata.loc[c.metadata.index.intersection(index)]
        done = {}
        # Les expériences partageant une formule partagent leur ajustement : le
        # coût est dans `dds.deseq2()`, pas dans l'extraction d'un contraste.
        by_design: dict[str, dict] = {}
        for name, spec in experiments.items():
            by_design.setdefault(str(spec["design"]), {})[name] = spec

        for design, group in by_design.items():
            group_specs, rows = {}, {}
            for name, spec in group.items():
                contrast = str(spec["contrast"])
                control, test = str(spec["control"]), str(spec["test"])
                min_group = int(spec.get("min_group", 3))
                rows[name] = {"group_col": group_col, "modalite": modality,
                              "experience": name, "design": design,
                              "contraste": contrast}

                # Pré-contrôle : écarter une combinaison inexploitable AVANT d'y
                # consacrer un ajustement. Sur une stratification fine croisée
                # avec des contrastes rares, c'est la majorité des cas.
                n_ctrl, n_test, _ = dg.contrast_group_sizes(
                    meta_stratum, design, contrast, control, test)
                rows[name] |= {"n_control": n_ctrl, "n_test": n_test}
                if min(n_ctrl, n_test) < min_group:
                    plan.append(rows[name] | {
                        "statut": "écartée", "design_utilise": "",
                        "raison": f"effectifs {test}={n_test}, {control}={n_ctrl} "
                                  f"< min_group={min_group}"})
                    continue
                group_specs[name] = {
                    "contrast": contrast, "control": control, "test": test,
                    "min_group": min_group,
                    "gene_sets": cf.clinical_gene_sets(
                        args.gsea_collections, args.gsea_gene_sets, spec),
                    "outdir": (c.outdir / "tables" / "clinical_degsea"
                               / cf.slug(group_col) / cf.slug(modality) / name),
                }
            if not group_specs:
                continue

            out = dg.run_clinical_degsea_group(
                counts.loc[counts.index.intersection(index)], c.metadata,
                design=design, specs=group_specs,
                min_count=int(next(iter(group.values())).get("min_count", 10)),
                gene_filters=gene_filters,
                permutations=int(next(iter(group.values())).get(
                    "gsea_permutations", args.gsea_permutations)),
                n_jobs=c.eff_n_jobs, seed=args.seed, settings=deseq_settings,
                min_level_count=min(int(s.get("min_group", 3))
                                    for s in group.values()),
            )
            dropped_txt = " ; ".join(f"{k} ({v})"
                                     for k, v in out["dropped_terms"].items())
            for name, reason in out["skipped"].items():
                log.warning("DEGSEA clinique [%s/%s/%s] écarté : %s",
                            group_col, modality, name, reason)
                plan.append(rows[name] | {"statut": "écartée", "raison": reason,
                                          "design_utilise": out["design_used"],
                                          "termes_ecartes": dropped_txt})
            for name, result in out["results"].items():
                spec = group[name]
                done[name] = {
                    "design": design, "design_used": out["design_used"],
                    "dropped_terms": dropped_txt,
                    "contrast": str(spec["contrast"]),
                    "control": str(spec["control"]), "test": str(spec["test"]),
                    "group_col": group_col, "modality": modality,
                    "n_samples": result["n_samples"], "n_test": result["n_test"],
                    "n_control": result["n_control"], "n_dropped": result["n_dropped"],
                    "collections": sorted(result["gsea"]),
                }
                plan.append(rows[name] | {
                    "statut": "calculée", "raison": "",
                    "design_utilise": out["design_used"], "termes_ecartes": dropped_txt,
                    "n_control": result["n_control"], "n_test": result["n_test"]})
                log.info("DEGSEA clinique [%s/%s/%s] terminé : n=%d (%s=%d vs %s=%d).",
                         group_col, modality, name, result["n_samples"],
                         spec["test"], result["n_test"],
                         spec["control"], result["n_control"])
        if done:
            c.clinical_degsea.setdefault(group_col, {})[modality] = done

    table = pd.DataFrame(plan)
    root = c.outdir / "tables" / "clinical_degsea"
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "plan.csv", index=False)
    n_done = int((table["statut"] == "calculée").sum()) if len(table) else 0
    log.info("DEGSEA clinique : %d / %d combinaisons calculées, %d écartées "
             "pour effectifs — détail dans %s.", n_done, len(table),
             len(table) - n_done, root / "plan.csv")


def _outrider(c: _Ctx) -> None:
    """Étape 7b — expression aberrante par tumeur, découpée par `subset_by`.

    Comme le DEGSEA clinique, cette étape ne lit ni labels ni branche : elle ne
    consomme que les counts BRUTS et les métadonnées. Elle est donc placée juste
    à côté de lui, avant tout ce qui dépend du clustering.
    """
    args, log = c.args, c.log
    c.outrider = {}
    if args.run_outrider != "y":
        return
    if c.metadata is None and cf.subset_specs(args.subset_by):
        log.warning("7b. OUTRIDER : `subset_by` est renseigné mais aucune table de "
                    "métadonnées n'est fournie — étape sautée.")
        return

    metadata = c.metadata if c.metadata is not None else pd.DataFrame(
        index=pd.Index([str(s) for s in c.raw.index]))
    settings = od.OutriderSettings.from_args(args, n_jobs=c.eff_n_jobs)
    c.outrider = od.run_outrider(
        c.raw, metadata, c.outdir, subset_by=args.subset_by, settings=settings,
        samples=c.raw.index,
    )


def _degsea(c: _Ctx, k: int, gene_sets: dict[str, str], *,
             output_subdir: str = "") -> dict:
    """Exécute DEGSEA pour une partition consensus ``k`` donnée.

    Le calcul par contraste est fourni par :func:`degsea.run_degsea` : DESeq2
    n'est ajusté qu'une fois par contraste puis son classement est réutilisé
    pour toutes les collections GSEA.
    """
    args, branch = c.args, c.primary
    labels = branch.result.labels(int(k), args.linkage)
    return dg.run_degsea(
        c.raw, labels, branch.result.sample_names, c.outdir,
        gene_sets=gene_sets,
        gene_filters=dg.DegseaFilters.from_args(args, prefix="degsea"),
        settings=dg.DeseqSettings.from_args(args, prefix="degsea"),
        mode=args.degsea_mode,
        permutations=args.gsea_permutations,
        heatmap_pval=args.gsea_heatmap_pval,
        subdir=output_subdir,
        n_jobs=c.eff_n_jobs,
        seed=args.seed,
    )


def _degsea_all_k(c: _Ctx) -> None:
    """Orchestre DEGSEA sur le K recommandé PAC+Δ(K), ou sur tous les K."""
    args, log, outdir = c.args, c.log, c.outdir
    branch = c.primary
    c.degsea_by_k = {}
    if args.run_degsea != "y":
        return

    gene_sets = cf.collections_or_fallback(args.gsea_collections, args.gsea_gene_sets)
    all_k = args.degsea_all_k == "y"
    try:
        recommended_k = int(mt.suggest_k(
            branch.result,
            args.min_cluster_size,
            method="both",
        ))
    except Exception as exc:
        recommended_k = int(branch.k_final)
        log.warning(
            "DEGSEA : calcul du k recommandé PAC+Delta(K) impossible (%s) ; "
            "repli sur k_final=%d.",
            exc, recommended_k,
        )
    ks_degsea = list(branch.k_values) if all_k else [recommended_k]
    if not all_k:
        log.info(
            "DEGSEA ciblé sur le k recommandé PAC+Delta(K) : k=%d%s.",
            recommended_k,
            (
                f" (distinct de k_final={branch.k_final}, choisi par "
                f"k_criterion={args.k_criterion})"
                if recommended_k != branch.k_final else ""
            ),
        )
    log.info("13. DEGSEA par cluster : DESeq2 + GSEA (mode=%s, %d collection(s)) "
             "sur %d valeur(s) de k=%s — étape longue…",
             args.degsea_mode, len(gene_sets), len(ks_degsea), list(ks_degsea))

    for index, k in enumerate(ks_degsea, start=1):
        if all_k:
            log.info("DEGSEA — k=%d (%d/%d)…", k, index, len(ks_degsea))
        result = _degsea(c, k, gene_sets, output_subdir=f"k{k}" if all_k else "")
        c.degsea_by_k[int(k)] = result
        if k == recommended_k:
            for coll, matrix in result.items():
                pl.plot_gsea_ova_heatmap(
                    matrix, outdir / "figures",
                    pval=args.gsea_heatmap_pval, collection=coll,
                )
    log.info("DEGSEA terminé : tables dans %s", outdir / "tables" / "degsea")


def _signatures(c: _Ctx) -> None:
    args, log, outdir, eff_n_jobs = c.args, c.log, c.outdir, c.eff_n_jobs
    branch = c.primary
    raw, result, k_values = c.raw, branch.result, branch.k_values
    # ----------------------------------------- 14. projection de signatures
    sig_scores = None
    sig_tests = None
    prov = None
    if args.compute_signatures == "y":
        # sources harmonisées : signature_sources du YAML, sinon une source .gmt
        # unique (signatures_gmt / load_signatures_select / gsea_gene_sets).
        sources = dict(args.signature_sources)
        if not sources:
            fb = (args.signatures_gmt
                  or args.gsea_collections.get("signatures_select")
                  or args.gsea_gene_sets)
            if fb:
                sources = {"signatures": {"format": "gmt", "path": fb}}
        signatures, prov = sp.load_signature_sources(sources)
        if not signatures:
            log.warning("Projection de signatures : aucune signature chargée "
                        "(sources : %s) — étape sautée.", list(sources) or "aucune")
        else:
            log.info("14. Projection de %d signatures (%d source(s) : %s)",
                     len(signatures), len(sources), ", ".join(sources))
            (outdir / "tables" / "signatures").mkdir(parents=True, exist_ok=True)
            prov.to_csv(outdir / "tables" / "signatures" / "signature_sources.csv",
                        index=False)
            sub = raw.loc[result.sample_names]
            expr_full = sub if args.already_normalized else pp.log_cpm(sub)
            meta_full = branch.aligned_metadata()
            sig_scores = sp.run_signature_projection(
                expr_full, signatures, meta_full, outdir,
                corr_method=args.sig_corr_method, top_n=args.sig_top_n,
                sig_pval=args.sig_pval, max_text_levels=args.sig_max_text_levels,
                n_jobs=eff_n_jobs, seed=args.seed,
            )
            # tests de Wilcoxon (one-vs-rest) score de signature x modalité, pour
            # chaque k (stratif. cluster) et chaque variable clinique catégorielle
            # -> étoiles au-dessus des boxplots du rapport (14.2 bis).
            cluster_labels_by_k = {k: result.labels(k, args.linkage) for k in k_values}
            sig_tests, sig_tests_tidy = sp.stratified_signature_tests(
                sig_scores, cluster_labels_by_k, meta_full,
                max_text_levels=args.sig_max_text_levels)
            if len(sig_tests_tidy):
                sig_tests_tidy.to_csv(
                    outdir / "tables" / "signatures" / "signature_group_tests.csv",
                    index=False)
                n_sig = int((sig_tests_tidy["padj"] < 0.05).sum())
                log.info("Projection 14.2bis : %d tests de Wilcoxon (score x modalité, "
                         "one-vs-rest + pairwise, tous k), %d significatifs "
                         "(FDR < 0.05) -> %s", len(sig_tests_tidy), n_sig,
                         outdir / "tables" / "signatures" / "signature_group_tests.csv")
            log.info("Projection de signatures terminée : tables dans %s",
                     outdir / "tables" / "signatures")
    c.sig_scores, c.sig_tests, c.sig_provenance = sig_scores, sig_tests, prov


def _deconvolution(c: _Ctx) -> None:
    args, log, outdir = c.args, c.log, c.outdir
    branch = c.primary
    raw, result, labels = c.raw, branch.result, branch.labels
    # ------------------------------------------------ 15. déconvolution (R)
    deconv = {}
    if args.run_deconv == "y":
        log.info("15. Déconvolution (omnideconv / immunedeconv) — étape longue…")
        if args.already_normalized:
            log.warning("Déconvolution : --already-normalized est actif, mais la "
                        "déconvolution attend des counts BRUTS (CPM linéaire pour "
                        "immunedeconv, counts pour BayesPrism). Résultats peu fiables.")
        deconv = dc.run_deconvolution(
            raw, result.sample_names, outdir,
            methods=args.deconv_methods or None,       # None -> batterie par défaut
            reference=args.deconv_reference or None,
            rscript=args.deconv_rscript,
        )
        for meth, frac in deconv.items():
            pl.plot_deconvolution(frac, labels, result.sample_names, meth,
                                  outdir / "figures")
        log.info("Déconvolution terminée : tables dans %s",
                 outdir / "tables" / "deconvolution")
    c.deconv = deconv


def _clinical_analyses(c: _Ctx) -> None:
    """Étapes 16-17 — khi² et corrélations, pour TOUTES les branches.

    Ces deux analyses croisent une partition (ou des scores) avec les variables
    cliniques. Elles sont volontairement jouées ici, en fin de pipeline, et non
    dans `AnalysisBranch.run()` : les corrélations consomment les scores de
    signatures (étape 14) et de déconvolution (étape 15), qui n'existaient pas
    encore au moment où les branches sont construites. Lancées trop tôt — ce que
    faisait la branche ICA — elles ne voyaient que la clinique continue et les
    composantes, sans que rien ne le signale.

    Le khi² ne dépend, lui, que des labels et des métadonnées : le déplacer ici
    ne change aucun résultat, mais met les deux familles de tests au même
    endroit, avec le même périmètre, pour la branche historique comme pour
    chaque projection ICA.
    """
    branches = [("Consensus Clustering", c.primary, None)]
    branches += [(f"ICA m={dim}", branch, features)
                 for dim, (branch, features) in sorted(c.ica_branch_objects.items())]

    c.log.info("16-17. Analyses cliniques (khi² + corrélations) sur %d branche(s) : "
               "%s. Les corrélations disposent maintenant des signatures (%s) et de "
               "la déconvolution (%s).", len(branches),
               ", ".join(name for name, _, _ in branches),
               "oui" if c.sig_scores else "non", "oui" if c.deconv else "non")
    for name, branch, features in branches:
        branch.run_associations()
        branch.run_correlations(
            sig_scores=c.sig_scores, deconv=c.deconv or None,
            extra_features=features, extra_prefix="ica",
        )

    # Le rapport lit les branches ICA via un dictionnaire figé plus haut :
    # on y reporte les résultats produits à l'instant.
    for dim, (branch, _) in c.ica_branch_objects.items():
        if dim in c.ica_branches:
            c.ica_branches[dim]["assoc"] = branch.assoc
            c.ica_branches[dim]["corr"] = branch.corr


def _report(c: _Ctx) -> None:
    args, log, outdir = c.args, c.log, c.outdir
    branch = c.primary
    # ------------------------------ 18. rapport d'analyse HTML interactif
    if args.create_report == "y":
        results = PipelineResults(
            result=branch.result, k_final=branch.k_final, linkage_method=args.linkage,
            min_cluster_size=args.min_cluster_size, k_criterion=args.k_criterion,
            coords=branch.coords, coords_by_k=branch.coords_by_k,
            coords_by_distance=branch.coords_by_distance,
            meta=branch.aligned_metadata(), sig_scores=c.sig_scores,
            sig_provenance=c.sig_provenance,
            sig_tests=c.sig_tests, deconv=(c.deconv or None),
            degsea_by_k=(c.degsea_by_k or None),
            clinical_degsea=(c.clinical_degsea or None),
            outrider=(c.outrider or None),
            branch_stability_by_k=(branch.branch_stability_by_k or None),
            assoc=(branch.assoc or None), corr=(branch.corr or None),
            ica={"result": c.ica_result, "branches": c.ica_branches,
                 "enabled": args.run_ica == "y",
                 "gseaEnabled": args.run_ica_gsea == "y"},
            filter_columns=cf.as_str_tuple(args.filter_columns))
        rp.build_report(results, outdir)
        log.info("Rapport d'analyse : %s", outdir / "report.html")


def _save(c: _Ctx) -> None:
    args, log, outdir, t_start = c.args, c.log, c.outdir, c.t_start
    k_final = c.primary.k_final
    # --------------------------------------------- 19. sauvegarde du run
    with open(outdir / "run_params.json", "w") as fh:
        json.dump({**vars(args), "outdir": str(args.outdir), "k_final": k_final},
                  fh, indent=2, default=str)

    dt = time.perf_counter() - t_start
    log.info("================ Terminé en %dh %02dm %02ds — résultats dans %s ================",
             int(dt // 3600), int(dt % 3600 // 60), int(dt % 60), outdir.resolve())


def main(argv=None) -> int:
    """Orchestre les dépendances entre entrées, branches et enrichissements."""
    try:
        c = _setup(argv)
    except ConfigError as exc:
        # Faute de configuration : message net sur stderr, sans pile d'appels —
        # rien n'a encore été calculé, il n'y a rien d'autre à diagnostiquer.
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    # L'ordre ci-dessous EST la définition des étapes : il est repris tel quel
    # par les sections numérotées des fichiers de configuration.
    _load_data(c)              # 1.  chargement de la matrice
    _harmonize_gene_ids(c)     # 3.  identifiants -> symboles HGNC
    _preprocess_matrix(c)      # 2.  filtrage + normalisation + gènes variables
    _purity_filter(c)          # 4.  pureté tumorale (PUREE)
    _outlier_filter(c)         # 5.  outliers ACP
    _refit_matrix(c)           # 2b. prétraitement rejoué sans les tumeurs écartées
    _load_metadata(c)          # 1.  métadonnées cliniques (+ filter_columns)
    _clinical_degsea(c)        # 7.  DEGSEA clinique         (collections : 6)
    _outrider(c)               # 7b. OUTRIDER, un run par sous-groupe subset_by
    _ica(c)                    # 8.  ICA stabilisée
    _ica_metagene_gsea(c)      # 9.  GSEA des métagènes ICA
    _run_ica_branches(c)       # 10-12. consensus, Jaccard, embeddings (ICA)
    _run_primary_branch(c)     # 10-12. idem sur l'expression
    _degsea_all_k(c)           # 13. DEGSEA par cluster
    _signatures(c)             # 14. projection de signatures
    _deconvolution(c)          # 15. déconvolution
    _clinical_analyses(c)      # 16-17. khi² + corrélations, toutes branches
    _report(c)                 # 18. rapport HTML
    _save(c)                   # 19. paramètres du run
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
