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
from . import ica as ic
from . import ica_cluster_compare as icc
from . import ica_gsea as ig
from . import metrics as mt
from . import plots as pl
from . import preprocessing as pp
from . import purity as pur
from . import report as rp
from . import sigproj as sp
from .analysis_branch import AnalysisBranch, BranchPaths, BranchSettings
from .config import ConfigError, build_parser, load_config
from .results import PipelineResults


@dataclass
class _Ctx:
    """État d'orchestration : entrées, branche principale et sorties spécialisées."""
    args: object; log: object; outdir: Path; t_start: float; eff_n_jobs: int
    raw: object = None
    X_df: object = None
    metadata: object = None
    primary: AnalysisBranch | None = None
    nes: object = None
    sig_scores: object = None
    sig_provenance: object = None
    sig_tests: object = None
    ica_result: object = None
    degsea_by_k: dict = field(default_factory=dict)
    clinical_degsea: dict = field(default_factory=dict)
    deconv: dict = field(default_factory=dict)
    ica_branches: dict = field(default_factory=dict)
    ica_metagene_gsea: dict = field(default_factory=dict)


def _setup(argv) -> _Ctx:
    # ------------------------------------------ 0. configuration & démarrage
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


def _load_data(c: _Ctx) -> None:
    args, log = c.args, c.log
    # ---------------------------------------------------------------- 1. data
    raw = pp.load_matrix(args.counts, genes_in_rows=not args.samples_in_rows)
    X_df = pp.preprocess(
        raw,
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
    log.info("Matrice prétraitée : %d tumeurs x %d gènes", *X_df.shape)
    c.raw, c.X_df = raw, X_df


def _load_metadata(c: _Ctx) -> None:
    """Charge une fois les métadonnées et normalise les identifiants échantillon."""
    if not c.args.metadata:
        c.metadata = None
        return
    metadata = pd.read_csv(c.args.metadata, sep=None, engine="python", index_col=0)
    metadata.index = metadata.index.astype(str)
    c.metadata = metadata


def _purity_filter(c: _Ctx) -> None:
    args, log, outdir = c.args, c.log, c.outdir
    raw, X_df = c.raw, c.X_df
    # ------------------------------------- 1a. pureté tumorale (PUREE) + filtrage
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
    # ------------------------------------------- 1b. filtrage d'outliers (ACP)
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
        log.info("1c. ICA stabilisée désactivée (run_ica = n).")
        return

    log.info(
        "1c. ICA stabilisée : dimensions %d..%d (pas %d), %d runs/dimension…",
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
        ).run(
            run_associations=True,
            run_correlations=True,
            correlation_extra_features=dec.metasamples,
            correlation_prefix="ica",
        )
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
        c.log.info("1d. GSEA des métagènes ICA désactivé (run_ica_gsea = n).")
        return

    gene_sets = dg.resolve_gene_sets(
        cf.collections_or_fallback(c.args.gsea_collections, c.args.gsea_gene_sets)
    )
    if not gene_sets:
        c.log.warning(
            "1d. GSEA des métagènes ICA demandé, mais aucune collection GMT "
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
        "1d. Annotation GSEA des métagènes ICA : %d dimension(s), "
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
    ).run(run_associations=True)


def _clinical_degsea(c: _Ctx) -> None:
    """Exécute les expériences cliniques configurées, sans consensus clustering.

    Cette étape ne lit ni labels ni ``AnalysisBranch`` : elle consomme seulement
    les counts bruts, les métadonnées et le dictionnaire ``clinical_degsea``.
    """
    args, log = c.args, c.log
    c.clinical_degsea = {}
    experiments = cf.clinical_experiments(args.clinical_degsea)
    if not experiments:
        if args.run_clinical_degsea == "y":
            log.warning("DEGSEA clinique demandé, mais aucune expérience clinical_degsea n'est configurée.")
        return
    # L'incompatibilité avec --already-normalized est refusée en amont par
    # config.validate ; seules les métadonnées ne sont connues qu'ici.
    if c.metadata is None:
        raise ValueError("DEGSEA clinique configuré, mais aucune table de métadonnées n'est fournie.")

    log.info("DEGSEA clinique : %d expérience(s), indépendante(s) du consensus clustering.",
             len(experiments))
    for name, spec in experiments.items():
        gene_sets = cf.clinical_gene_sets(
            args.gsea_collections, args.gsea_gene_sets, spec
        )
        result = dg.run_clinical_degsea(
            c.raw, c.metadata,
            design=str(spec["design"]),
            contrast=str(spec["contrast"]),
            control=str(spec["control"]),
            test=str(spec["test"]),
            gene_sets=gene_sets,
            outdir=c.outdir / "tables" / "clinical_degsea" / name,
            min_group=int(spec.get("min_group", 3)),
            min_count=int(spec.get("min_count", 10)),
            permutations=int(spec.get("gsea_permutations", args.gsea_permutations)),
            n_jobs=c.eff_n_jobs,
            seed=args.seed,
        )
        c.clinical_degsea[name] = {
            "design": str(spec["design"]),
            "contrast": str(spec["contrast"]),
            "control": str(spec["control"]),
            "test": str(spec["test"]),
            "n_samples": result["n_samples"],
            "n_test": result["n_test"],
            "n_control": result["n_control"],
            "n_dropped": result["n_dropped"],
            "collections": sorted(result["gsea"]),
        }
        log.info(
            "DEGSEA clinique [%s] terminé : n=%d (%s=%d vs %s=%d), %d collection(s).",
            name, result["n_samples"], spec["test"], result["n_test"],
            spec["control"], result["n_control"], len(result["gsea"]),
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
    c.nes, c.degsea_by_k = None, {}
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
    log.info("DEGSEA : DESeq2 + GSEA par cluster (mode=%s, %d collection(s)) "
             "sur %d valeur(s) de k=%s — étape longue…",
             args.degsea_mode, len(gene_sets), len(ks_degsea), list(ks_degsea))

    final_nes_by_coll = {}
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
        if k == branch.k_final:
            # La synthèse historique est construite sur k_final : ne lui
            # transmettre qu'une matrice NES calculée pour cette même partition.
            final_nes_by_coll = result

    if final_nes_by_coll:
        key = next(
            (name for name in ("h", "HALLMARK", "hallmark")
             if name in final_nes_by_coll),
            next(iter(final_nes_by_coll)),
        )
        c.nes = final_nes_by_coll[key]
    log.info("DEGSEA terminé : tables dans %s", outdir / "tables" / "degsea")


def _signatures(c: _Ctx) -> None:
    args, log, outdir, eff_n_jobs = c.args, c.log, c.outdir, c.eff_n_jobs
    branch = c.primary
    raw, result, k_values = c.raw, branch.result, branch.k_values
    # ------------------------------------------ 7. projection de signatures
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
            log.info("7. Projection de %d signatures (%d source(s) : %s)",
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
                sig_pval=args.sig_pval, n_jobs=eff_n_jobs, seed=args.seed,
            )
            # tests de Wilcoxon (one-vs-rest) score de signature x modalité, pour
            # chaque k (stratif. cluster) et chaque variable clinique catégorielle
            # -> étoiles au-dessus des boxplots du rapport (7.2 bis).
            cluster_labels_by_k = {k: result.labels(k, args.linkage) for k in k_values}
            sig_tests, sig_tests_tidy = sp.stratified_signature_tests(
                sig_scores, cluster_labels_by_k, meta_full)
            if len(sig_tests_tidy):
                sig_tests_tidy.to_csv(
                    outdir / "tables" / "signatures" / "signature_group_tests.csv",
                    index=False)
                n_sig = int((sig_tests_tidy["padj"] < 0.05).sum())
                log.info("Projection 7.2bis : %d tests de Wilcoxon (score x modalité, "
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
    # ------------------------------------------------- 8. déconvolution (R)
    deconv = {}
    if args.run_deconv == "y":
        log.info("8. Déconvolution (omnideconv / immunedeconv) — étape longue…")
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


def _correlations(c: _Ctx) -> None:
    """Complète la branche historique une fois signatures/déconv disponibles."""
    c.primary.run_correlations(sig_scores=c.sig_scores, deconv=c.deconv or None)


def _synthesis(c: _Ctx) -> None:
    args, log, outdir = c.args, c.log, c.outdir
    branch = c.primary
    # ------------------------------- 9. figure de synthèse (tout combiné)
    pl.plot_cluster_overview(
        branch.result, branch.k_final, outdir / "figures", linkage_method=args.linkage,
        branch_stability=branch.branch_stability, items=branch.items,
        color_by=branch.color_var, color_label=args.color_by or "color_by",
        nes=c.nes,
    )
    log.info("Figure de synthèse : cluster_overview_k%d.png", branch.k_final)


def _report(c: _Ctx) -> None:
    args, log, outdir = c.args, c.log, c.outdir
    branch = c.primary
    # ------------------------------ 10. rapport d'analyse HTML interactif
    if args.create_report == "y":
        results = PipelineResults(
            result=branch.result, k_final=branch.k_final, linkage_method=args.linkage,
            min_cluster_size=args.min_cluster_size, k_criterion=args.k_criterion,
            coords=branch.coords, coords_by_k=branch.coords_by_k,
            meta=branch.aligned_metadata(), sig_scores=c.sig_scores,
            sig_provenance=c.sig_provenance,
            sig_tests=c.sig_tests, deconv=(c.deconv or None),
            degsea_by_k=(c.degsea_by_k or None),
            clinical_degsea=(c.clinical_degsea or None),
            branch_stability_by_k=(branch.branch_stability_by_k or None),
            assoc=(branch.assoc or None), corr=(branch.corr or None),
            ica={"result": c.ica_result, "branches": c.ica_branches,
                 "enabled": args.run_ica == "y",
                 "gseaEnabled": args.run_ica_gsea == "y"})
        rp.build_report(results, outdir)
        log.info("Rapport d'analyse : %s", outdir / "report.html")


def _save(c: _Ctx) -> None:
    args, log, outdir, t_start = c.args, c.log, c.outdir, c.t_start
    k_final = c.primary.k_final
    # --------------------------------------------- 11. sauvegarde du run
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
    _load_data(c)
    _purity_filter(c)
    _outlier_filter(c)
    _load_metadata(c)
    _clinical_degsea(c)
    _ica(c)
    _ica_metagene_gsea(c)
    _run_ica_branches(c)
    _run_primary_branch(c)
    _degsea_all_k(c)
    _signatures(c)
    _deconvolution(c)
    _correlations(c)
    _synthesis(c)
    _report(c)
    _save(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
