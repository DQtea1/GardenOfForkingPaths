#!/usr/bin/env python
"""Pipeline complet : prétraitement -> consensus clustering -> diagnostics ->
embeddings t-SNE / UMAP -> figures et tables.

Exemples
--------
# jeu de démonstration (500 tumeurs simulées, 4 sous-types)
python make_demo_data.py && python run_pipeline.py --counts data/demo_counts.tsv \
    --outdir results/demo --n-resamples 300

# données réelles, matrice VST déjà normalisée (gènes en lignes)
python run_pipeline.py --counts data/vst.tsv --already-normalized \
    --k-max 10 --n-resamples 1000 --base hierarchical --metric pearson \
    --gene-mode bootstrap --outdir results/run01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import catassoc as ca
from src import consensus as cc
from src import deconv as dc
from src import degsea as dg
from src import embedding as emb
from src import metrics as mt
from src import plots as pl
from src import preprocessing as pp
from src import purity as pur
from src import report as rp
from src import sigproj as sp
from src import stability as st


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, type=Path,
                   help="fichier YAML de paramètres. Tout paramètre passé en "
                        "ligne de commande a la priorité sur le YAML.")
    io = p.add_argument_group("entrées / sorties")
    io.add_argument("--counts", default=None, help="matrice csv/tsv/parquet "
                    "(obligatoire, en ligne de commande ou dans le YAML)")
    io.add_argument("--samples-in-rows", action="store_true",
                    help="par défaut : gènes en lignes, échantillons en colonnes")
    io.add_argument("--metadata", default=None,
                    help="csv/tsv indexé par échantillon (annotations cliniques)")
    io.add_argument("--color-by", default=None,
                    help="colonne des métadonnées à superposer sur les embeddings")
    io.add_argument("--outdir", default="results/run", type=Path)

    pre = p.add_argument_group("prétraitement")
    pre.add_argument("--already-normalized", action="store_true",
                     help="entrée déjà en VST/rlog/logCPM : saute filtrage + normalisation")
    pre.add_argument("--norm_method", choices=["vst", "logcpm"], default="vst",
                     help="normalisation des counts bruts (si --already-normalized "
                          "n'est pas mis) : 'vst' (DESeq2/PyDESeq2, défaut) ou 'logcpm'.")
    pre.add_argument("--min-cpm", type=float, default=1.0)
    pre.add_argument("--min-frac-samples", type=float, default=0.2)
    pre.add_argument("--keep-technical", action="store_true")
    pre.add_argument("--n-top-genes", type=int, default=5000)
    pre.add_argument("--variance-method", choices=["mad", "var"], default="mad")
    pre.add_argument("--scale-genes", action="store_true")
    pre.add_argument("--outlier_sd_threshold", type=float, default=0.0,
                     help="ACP sur la matrice prétraitée puis retrait des tumeurs "
                          "à plus de N écarts-types sur une composante principale. "
                          "0 ou absent = aucun retrait.")
    pre.add_argument("--outlier_n_pc", type=int, default=10,
                     help="nombre de composantes principales inspectées pour la "
                          "détection d'outliers (défaut 10).")
    pre.add_argument("--outlier_min_explained_var", type=float, default=0.0,
                     help="si > 0, inspecte plutôt toutes les composantes dont la "
                          "variance expliquée dépasse ce seuil (fraction ]0,1[ ; "
                          "8 = 8 %%). Prioritaire sur --outlier_n_pc.")

    pur_g = p.add_argument_group("pureté tumorale (PUREE)")
    pur_g.add_argument("--purity_threshold", default="0",
                       help="seuil de pureté dans ]0,1[ pour filtrer les tumeurs. "
                            "0 / null / false = aucun filtrage (PUREE n'est pas lancé).")
    pur_g.add_argument("--purity_direction", choices=["higher", "lower"],
                       default="higher",
                       help="'higher' garde les puretés >= seuil (retire les "
                            "faibles puretés) ; 'lower' garde les puretés <= seuil.")
    pur_g.add_argument("--puree_dir", default="/home/quentin/02_MODELS/PUREE",
                       help="dossier du dépôt PUREE (predict_purity.py, models/, data/).")
    pur_g.add_argument("--puree_python",
                       default="/home/quentin/miniforge3/envs/PUREE/bin/python",
                       help="interpréteur Python de l'environnement PUREE.")
    pur_g.add_argument("--puree_gene_id", choices=["HGNC", "ENSEMBL"], default="HGNC",
                       help="type d'identifiant des gènes de la matrice d'entrée.")

    con = p.add_argument_group("consensus clustering")
    con.add_argument("--k-min", type=int, default=2)
    con.add_argument("--k-max", type=int, default=8)
    con.add_argument("--n-resamples", type=int, default=1000)
    con.add_argument("--prop-samples", type=float, default=0.8)
    con.add_argument("--prop-genes", type=float, default=0.8)
    con.add_argument("--sample-mode", choices=["subsample", "bootstrap"],
                     default="subsample")
    con.add_argument("--gene-mode", choices=["subsample", "bootstrap"],
                     default="subsample")
    con.add_argument("--base", choices=["hierarchical", "kmeans", "kmedoids"],
                     default="hierarchical")
    con.add_argument("--metric", choices=["pearson", "spearman", "euclidean", "cosine"],
                     default="pearson")
    con.add_argument("--linkage", default="average",
                     choices=["average", "complete", "ward", "single"])
    con.add_argument("--k-final", type=int, default=None,
                     help="k retenu ; par défaut choisi automatiquement (voir --k_criterion)")
    con.add_argument("--k_criterion", choices=["pac", "deltak", "both"], default="both",
                     help="critère de choix auto de k (si --k-final absent) : 'pac' "
                          "(minimise le PAC), 'deltak' (coude de Δ(K)), 'both' (défaut).")
    con.add_argument("--min-cluster-size", type=int, default=10)

    stab = p.add_argument_group("stabilité des branches (Jaccard bootstrap)")
    stab.add_argument("--compute_jaccard", choices=["y", "n"], default="y",
                      help="'y' : après la partition finale, calcule la stabilité "
                           "Jaccard de chaque branche de l'arbre consensus par "
                           "bootstrap des gènes (n_resamples arbres). Défaut 'y'.")

    deg = p.add_argument_group("DEGSEA (DESeq2 + GSEA par cluster)")
    deg.add_argument("--run_degsea", choices=["y", "n"], default="n",
                     help="'y' : après les embeddings, DESeq2 + GSEA par cluster "
                          "(one-vs-all et one-vs-one). Étape longue. Défaut 'n'.")
    deg.add_argument("--degsea_mode", choices=["ova", "ovo", "both"], default="both",
                     help="contrastes DESeq2 : ova (one-vs-all), ovo (one-vs-one, "
                          "coûteux : k(k-1)/2), both (défaut).")
    deg.add_argument("--degsea_all_k", choices=["y", "n"], default="n",
                     help="'y' : calcule le DEGSEA pour TOUS les k de la plage "
                          "[k_min..k_max] (une sous-arborescence tables/degsea/k<k>/ "
                          "par k, et un panneau DEGSEA aligné sur n'importe quel k "
                          "dans le rapport). Très coûteux. Défaut 'n' (seul k_final).")
    deg.add_argument("--gsea_gene_sets",
                     default=str(Path.home() / ".cache/gseapy/Enrichr.MSigDB_Hallmark_2020.gmt"),
                     help="fichier .gmt de gene sets pour le GSEA (hallmarks MSigDB par défaut).")
    deg.add_argument("--gsea_permutations", type=int, default=1000,
                     help="nombre de permutations du GSEA pré-classé (défaut 1000).")
    deg.add_argument("--gsea_heatmap_pval", type=float, default=0.05,
                     help="seuil de p-valeur (nominale GSEA) pour inclure un "
                          "pathway dans la heatmap one-vs-all : tous les pathways "
                          "significatifs dans >= 1 cluster (défaut 0.05).")

    sig = p.add_argument_group("projection de signatures (scoring + association clinique)")
    sig.add_argument("--compute_signatures", choices=["y", "n"], default="n",
                     help="'y' : après DEGSEA, score les signatures par tumeur "
                          "(ssGSEA + expression moyenne) et teste leur association "
                          "aux variables cliniques. Défaut 'n'.")
    sig.add_argument("--signatures_gmt", default=None,
                     help="fichier .gmt des signatures à scorer. Défaut : la "
                          "collection load_signatures_select du YAML, sinon "
                          "--gsea_gene_sets.")
    sig.add_argument("--sig_corr_method", choices=["spearman", "pearson"],
                     default="spearman",
                     help="corrélation score↔variable continue (défaut spearman).")
    sig.add_argument("--sig_top_n", type=int, default=8,
                     help="nombre de top signatures affichées par variable (défaut 8).")
    sig.add_argument("--sig_pval", type=float, default=0.05,
                     help="seuil de FDR pour retenir une signature comme "
                          "significativement associée (défaut 0.05).")

    dec = p.add_argument_group("déconvolution (omnideconv / immunedeconv)")
    dec.add_argument("--run_deconv", choices=["y", "n"], default="n",
                     help="'y' : batterie de déconvolution (MCPcounter, xCell, "
                          "quanTIseq, EPIC, et DWLS/BayesPrism si référence "
                          "single-cell). Étape longue. Défaut 'n'. Méthodes et "
                          "paramètres : bloc deconv_methods du YAML ; référence : "
                          "bloc deconv_reference.")
    dec.add_argument("--deconv_rscript", default="Rscript",
                     help="interpréteur Rscript (omnideconv + immunedeconv installés).")

    chi = p.add_argument_group("association catégorielle (khi² d'indépendance)")
    chi.add_argument("--run_chi2", choices=["y", "n"], default="y",
                     help="'y' (défaut) : croise cluster (chaque k) × variables "
                          "cliniques catégorielles et clinique × clinique — khi² "
                          "d'indépendance (Fisher/Monte-Carlo en repli), V de Cramér "
                          "et résidus standardisés ajustés. Sauté sans métadonnées "
                          "catégorielles. Étape légère.")
    chi.add_argument("--chi2_mc_resamples", type=int, default=2000,
                     help="permutations du khi² de Monte-Carlo (repli des tables R×C "
                          "aux conditions de Cochran non remplies ; défaut 2000).")

    rep = p.add_argument_group("rapport d'analyse (HTML interactif)")
    rep.add_argument("--create_report", choices=["y", "n"], default="y",
                     help="'y' (défaut) : génère outdir/report.html — rapport "
                          "interactif autonome de tous les résultats.")

    embg = p.add_argument_group("embeddings")
    embg.add_argument("--t-SNE_dim", dest="tsne_dim", type=int, choices=[2, 3],
                      default=2,
                      help="dimensions des embeddings t-SNE / UMAP : 2 (PNG "
                           "statiques, défaut) ou 3 (HTML interactif rotatable, "
                           "survol = ID de la tumeur ; nécessite plotly)")
    embg.add_argument("--perplexity", type=float, default=30.0)
    embg.add_argument("--n-neighbors", type=int, default=15)
    embg.add_argument("--min-dist", type=float, default=0.1)
    embg.add_argument("--no-umap", action="store_true")

    p.add_argument("--parallel", choices=["y", "n"], default="y",
                   help="'y' (défaut) : parallélise le rééchantillonnage consensus, "
                        "la stabilité Jaccard, DEGSEA et les embeddings sur --n-jobs "
                        "cœurs. 'n' : force tout en séquentiel (n_jobs=1), utile pour "
                        "déboguer ou sur une machine partagée.")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_file", default="run.log",
                   help="fichier journal horodaté (relatif à outdir si non absolu ; "
                        "défaut run.log). Toute la progression y est écrite.")
    return p


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML absent : `pip install pyyaml` pour utiliser "
                          "--config.") from exc
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} : le YAML doit être un dictionnaire clé: valeur.")
    return data


def parse_args(argv=None):
    """Fusionne, par priorité croissante : défauts argparse < YAML < ligne de
    commande. Les clés du YAML sont les noms `dest` (avec des underscores :
    `n_top_genes`, `color_by`, `tsne_dim`, ...)."""
    parser = build_parser()
    args = parser.parse_args(argv)                 # défauts appliqués
    merged = vars(args)
    gsea_collections: dict[str, str] = {}
    signature_sources: dict = {}
    deconv_methods: dict = {}
    deconv_reference: dict = {}

    # 1. YAML : écrase les défauts, uniquement pour des clés connues
    if args.config:
        config = _load_yaml(args.config)
        unknown = set(config) - set(merged) - {"config"}

        # 1a. Collections de gene sets GSEA. Forme recommandée : un dict
        #     `gsea_collections: {nom: chemin.gmt}` (valeur str = chemin activé,
        #     ou {enabled: bool, path: ...}). Forme héritée aussi acceptée :
        #     des clés plates `load_<nom>: chemin.gmt`.
        if isinstance(config.get("gsea_collections"), dict):
            for name, spec in config["gsea_collections"].items():
                if isinstance(spec, str):
                    path, enabled = spec, True
                elif isinstance(spec, dict):
                    path, enabled = spec.get("path"), spec.get("enabled", True)
                else:
                    continue
                if enabled and path:
                    gsea_collections[str(name)] = str(Path(path).expanduser())
            unknown.discard("gsea_collections")
        for key in sorted(k for k in unknown if k.startswith("load_")):
            val = config[key]
            if isinstance(val, str) and val.strip().lower().endswith(".gmt"):
                gsea_collections[key[len("load_"):]] = str(Path(val).expanduser())
                unknown.discard(key)   # consommée ; les load_* non-.gmt restent signalées

        # 1a-bis. dicts imbriqués : sources de signatures (étape 7) et réglages
        #         de déconvolution (étape 8).
        if isinstance(config.get("signature_sources"), dict):
            signature_sources = config["signature_sources"]
            unknown.discard("signature_sources")
        if isinstance(config.get("deconv_methods"), dict):
            deconv_methods = config["deconv_methods"]
            unknown.discard("deconv_methods")
        if isinstance(config.get("deconv_reference"), dict):
            deconv_reference = config["deconv_reference"]
            unknown.discard("deconv_reference")

        # 1b. Autres clés inconnues : avertissement non bloquant (fonctions à venir,
        #     p. ex. human_pathways, IPRES flat — remplacé par signature_sources).
        if unknown:
            print(f"[config] clés ignorées (non gérées par le pipeline) : "
                  f"{', '.join(sorted(unknown))}", file=sys.stderr)

        for key, val in config.items():
            if key != "config" and key in merged:   # seules les clés connues
                merged[key] = val

    # 2. Ligne de commande : réappliquée par-dessus le YAML (priorité maximale).
    #    On repère les arguments réellement saisis en reparsant avec des défauts
    #    supprimés — seules les clés présentes ont été passées explicitement.
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    provided = vars(parser.parse_args(argv))
    provided.pop("config", None)
    merged.update(provided)

    if not merged.get("counts"):
        parser.error("--counts est obligatoire (en ligne de commande ou dans le YAML).")

    merged["gsea_collections"] = gsea_collections     # {nom: chemin .gmt}
    merged["signature_sources"] = signature_sources   # {nom: {format, path, ...}}
    merged["deconv_methods"] = deconv_methods         # {méthode: {enabled, ...}}
    merged["deconv_reference"] = deconv_reference     # {format, path, celltype_col, ...}
    return argparse.Namespace(**merged)


def main(argv=None) -> int:
    # ------------------------------------------ 0. configuration & démarrage
    args = parse_args(argv)

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

    # ------------------------------------------------- 2. consensus clustering
    k_values = tuple(range(args.k_min, args.k_max + 1))
    result = cc.consensus_clustering(
        X_df.values,
        k_values=k_values,
        n_resamples=args.n_resamples,
        prop_samples=args.prop_samples,
        prop_genes=args.prop_genes,
        sample_mode=args.sample_mode,
        gene_mode=args.gene_mode,
        base=args.base,
        metric=args.metric,
        linkage_method=args.linkage,
        sample_names=X_df.index.values,
        random_state=args.seed,
        n_jobs=eff_n_jobs,
    )

    # ------------------------------------------------------- 3. diagnostics k
    tab = mt.summary(result)
    tab.to_csv(outdir / "tables" / "k_selection.csv", index=False)
    log.info("Diagnostics par k :\n%s", tab.to_string(index=False))

    if args.k_final:
        k_final, reason = args.k_final, "(imposé)"
    else:
        k_final = mt.suggest_k(result, args.min_cluster_size, method=args.k_criterion)
        reason = f"(auto, critère : {args.k_criterion})"
    log.info("k retenu : %d %s", k_final, reason)

    pl.plot_cdf(result, outdir / "figures")
    pl.plot_tracking(result, outdir / "figures")
    for k in k_values:
        pl.plot_consensus_heatmap(result, k, outdir / "figures", args.linkage)

    # --------------------------------------------------- 4. partition finale
    labels = result.labels(k_final, args.linkage)
    items = mt.item_consensus(result, k_final)
    sil = mt.silhouette_per_sample(result, k_final)
    assign = (
        items.merge(sil[["sample", "silhouette"]], on="sample")
        .sort_values(["cluster", "item_consensus"], ascending=[True, False])
    )
    assign.to_csv(outdir / "tables" / f"cluster_assignments_k{k_final}.csv", index=False)
    mt.cluster_consensus(result, k_final).to_csv(
        outdir / "tables" / f"cluster_consensus_k{k_final}.csv", index=False)
    pl.plot_item_consensus(items, outdir / "figures", k_final)

    np.save(outdir / f"consensus_matrix_k{k_final}.npy", result.consensus[k_final])
    pd.DataFrame(result.distance(k_final), index=result.sample_names,
                 columns=result.sample_names).to_csv(
        outdir / "tables" / f"consensus_distance_k{k_final}.csv.gz", compression="gzip")

    # ------------------------------------ 4b. stabilité des branches (Jaccard)
    bs = None
    bs_by_k = {}
    if args.compute_jaccard == "y":
        # calculée pour TOUS les k (arbres bootstrap partagés -> coût ~ un seul k),
        # pour que le rapport affiche la stabilité sur l'arbre de n'importe quel k
        bs_by_k = st.branch_stability_multi(
            X_df.values, {k: result.distance(k) for k in k_values},
            n_resamples=args.n_resamples,
            gene_mode="bootstrap", prop_genes=1.0,
            metric=args.metric, linkage_method=args.linkage,
            min_size=2, random_state=args.seed, n_jobs=eff_n_jobs,
        )
        bs = bs_by_k[k_final]
        bs_tab = bs.to_frame(sample_names=result.sample_names, final_labels=labels)
        bs_tab.to_csv(outdir / "tables" / f"branch_stability_k{k_final}.csv", index=False)
        pl.plot_branch_stability(bs, outdir / "figures", k_final)
        n_stable = int((bs_tab["stability"] >= 0.75).sum())
        finals = bs_tab.loc[bs_tab["is_final_cluster"], ["branch_id", "size", "stability"]]
        log.info(
            "Stabilité Jaccard : %d/%d branches stables (>= 0,75). "
            "Clusters finaux (k=%d) :\n%s",
            n_stable, len(bs_tab), k_final,
            finals.to_string(index=False) if len(finals) else "(aucun cluster == branche)",
        )

    # ----------------------------------------------------- 5. t-SNE et UMAP
    D = result.distance(k_final)
    coords = emb.embeddings_table(
        D, result.sample_names, labels,
        run_umap=not args.no_umap,
        n_components=args.tsne_dim,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        n_jobs=eff_n_jobs,
    )
    coords = coords.merge(items[["sample", "item_consensus"]], on="sample")
    coords.to_csv(outdir / "tables" / f"embeddings_k{k_final}.csv", index=False)

    # 2D -> PNG statiques ; 3D -> HTML interactifs (rotation, survol = ID tumeur)
    plot_emb = pl.plot_embeddings_3d if args.tsne_dim == 3 else pl.plot_embeddings
    plot_emb(coords, outdir / "figures", k_final)
    plot_emb(coords, outdir / "figures", k_final,
             color_by=coords["item_consensus"], color_label="item consensus")

    # superposition d'une variable clinique (contrôle des confondants)
    color_var = None
    if args.metadata and args.color_by:
        meta = pd.read_csv(args.metadata, sep=None, engine="python", index_col=0)
        var = meta.reindex(result.sample_names)[args.color_by]
        color_var = var.to_numpy()
        plot_emb(coords, outdir / "figures", k_final,
                 color_by=var.reset_index(drop=True),
                 color_label=args.color_by)
        ct = pd.crosstab(labels, var.values)
        ct.to_csv(outdir / "tables" / f"crosstab_{args.color_by}_k{k_final}.csv")
        log.info("Croisement cluster x %s :\n%s", args.color_by, ct.to_string())

    # ------------------------------------------------ 6. DEGSEA (DESeq2 + GSEA)
    nes = None
    nes_by_coll = {}
    degsea_by_k = {}                    # {k: {collection: matrice NES}} -> rapport
    if args.run_degsea == "y":
        # collections `load_*.gmt` du YAML si présentes, sinon le .gmt unique
        gene_sets = args.gsea_collections or {
            Path(args.gsea_gene_sets).stem: args.gsea_gene_sets}
        all_k = args.degsea_all_k == "y"
        ks_degsea = list(k_values) if all_k else [k_final]
        log.info("DEGSEA : DESeq2 + GSEA par cluster (mode=%s, %d collection(s)) "
                 "sur %d valeur(s) de k=%s — étape longue…",
                 args.degsea_mode, len(gene_sets), len(ks_degsea), list(ks_degsea))
        for kk in ks_degsea:
            if all_k:
                log.info("DEGSEA — k=%d (%d/%d)…", kk,
                         ks_degsea.index(kk) + 1, len(ks_degsea))
            labels_k = result.labels(kk, args.linkage)
            res = dg.run_degsea(
                raw, labels_k, result.sample_names, outdir,
                gene_sets=gene_sets,
                mode=args.degsea_mode,
                permutations=args.gsea_permutations,
                heatmap_pval=args.gsea_heatmap_pval,
                subdir=(f"k{kk}" if all_k else ""),
                n_jobs=eff_n_jobs,
                seed=args.seed,
            )
            degsea_by_k[kk] = res
            if kk == k_final:
                nes_by_coll = res
                # une heatmap NES par collection (pour le k final uniquement)
                for coll, m in res.items():
                    pl.plot_gsea_ova_heatmap(m, outdir / "figures",
                                             pval=args.gsea_heatmap_pval, collection=coll)
        # pour la figure de synthèse : Hallmark de préférence, sinon la 1re dispo
        if nes_by_coll:
            key = next((k for k in ("h", "HALLMARK", "hallmark") if k in nes_by_coll),
                       next(iter(nes_by_coll)))
            nes = nes_by_coll[key]
        log.info("DEGSEA terminé : tables dans %s", outdir / "tables" / "degsea")

    # ------------------------------------------ 7. projection de signatures
    sig_scores = None
    sig_tests = None
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
            meta_full = None
            if args.metadata:
                meta_full = pd.read_csv(args.metadata, sep=None, engine="python",
                                        index_col=0).reindex(result.sample_names)
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
                n_sig = int((sig_tests_tidy["pvalue"] < 0.05).sum())
                log.info("Projection 7.2bis : %d tests de Wilcoxon (score x modalité, "
                         "one-vs-rest + pairwise, tous k), %d significatifs "
                         "(p < 0.05) -> %s", len(sig_tests_tidy), n_sig,
                         outdir / "tables" / "signatures" / "signature_group_tests.csv")
            log.info("Projection de signatures terminée : tables dans %s",
                     outdir / "tables" / "signatures")

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

    # ------------------ 9a. khi² d'indépendance (cluster/clinique catégoriel)
    if args.run_chi2 == "y" and args.metadata:
        meta_chi = pd.read_csv(args.metadata, sep=None, engine="python",
                               index_col=0).reindex(result.sample_names)
        cluster_labels_by_k = {k: result.labels(k, args.linkage) for k in k_values}
        ca.run_categorical_association(
            cluster_labels_by_k, meta_chi, result.sample_names, outdir, k_final,
            mc_resamples=args.chi2_mc_resamples, seed=args.seed)
    elif args.run_chi2 == "y":
        log.info("9a. Khi² : pas de métadonnées (--metadata) — étape sautée.")

    # ------------------------------- 9. figure de synthèse (tout combiné)
    pl.plot_cluster_overview(
        result, k_final, outdir / "figures", linkage_method=args.linkage,
        branch_stability=bs, items=items,
        color_by=color_var, color_label=args.color_by or "color_by",
        nes=nes,
    )
    log.info("Figure de synthèse : cluster_overview_k%d.png", k_final)

    # ------------------------------ 10. rapport d'analyse HTML interactif
    if args.create_report == "y":
        report_meta = None
        if args.metadata:
            report_meta = pd.read_csv(args.metadata, sep=None, engine="python",
                                      index_col=0).reindex(result.sample_names)
        rp.build_report(
            result, k_final, outdir,
            coords=coords, meta=report_meta, sig_scores=sig_scores,
            sig_tests=sig_tests, deconv=(deconv or None),
            degsea_by_k=(degsea_by_k or None),
            branch_stability_by_k=(bs_by_k or None),
            min_cluster_size=args.min_cluster_size, k_criterion=args.k_criterion,
            linkage_method=args.linkage,
        )
        log.info("Rapport d'analyse : %s", outdir / "report.html")

    # --------------------------------------------- 11. sauvegarde du run
    with open(outdir / "run_params.json", "w") as fh:
        json.dump({**vars(args), "outdir": str(args.outdir), "k_final": k_final},
                  fh, indent=2, default=str)

    dt = time.perf_counter() - t_start
    log.info("================ Terminé en %dh %02dm %02ds — résultats dans %s ================",
             int(dt // 3600), int(dt % 3600 // 60), int(dt % 60), outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
