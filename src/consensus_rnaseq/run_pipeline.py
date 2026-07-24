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
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from consensus_rnaseq import consensus as cc
from consensus_rnaseq import embedding as emb
from consensus_rnaseq import metrics as mt
from consensus_rnaseq import plots as pl
from consensus_rnaseq import preprocessing as pp
from consensus_rnaseq import purity as pur
from consensus_rnaseq import stability as st


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
                     help="k retenu ; par défaut choisi automatiquement (PAC min)")
    con.add_argument("--min-cluster-size", type=int, default=10)

    stab = p.add_argument_group("stabilité des branches (Jaccard bootstrap)")
    stab.add_argument("--compute_jaccard", choices=["y", "n"], default="y",
                      help="'y' : après la partition finale, calcule la stabilité "
                           "Jaccard de chaque branche de l'arbre consensus par "
                           "bootstrap des gènes (n_resamples arbres). Défaut 'y'.")

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

    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
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

    # 1. YAML : écrase les défauts, uniquement pour des clés connues
    if args.config:
        config = _load_yaml(args.config)
        unknown = set(config) - set(merged) - {"config"}
        if unknown:
            parser.error(f"clés inconnues dans {args.config} : "
                         f"{', '.join(sorted(unknown))}")
        for key, val in config.items():
            if key != "config":
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

    return argparse.Namespace(**merged)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("pipeline")

    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

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
        n_jobs=args.n_jobs,
    )

    # ------------------------------------------------------- 3. diagnostics k
    tab = mt.summary(result)
    tab.to_csv(outdir / "tables" / "k_selection.csv", index=False)
    log.info("Diagnostics par k :\n%s", tab.to_string(index=False))

    k_final = args.k_final or mt.suggest_k(result, args.min_cluster_size)
    log.info("k retenu : %d %s", k_final, "(imposé)" if args.k_final else "(heuristique PAC + coude Δ(K))")

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
    if args.compute_jaccard == "y":
        bs = st.branch_stability(
            X_df.values, result.distance(k_final),
            n_resamples=args.n_resamples,
            gene_mode="bootstrap", prop_genes=1.0,
            metric=args.metric, linkage_method=args.linkage,
            min_size=2, random_state=args.seed, n_jobs=args.n_jobs,
        )
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
    )
    coords = coords.merge(items[["sample", "item_consensus"]], on="sample")
    coords.to_csv(outdir / "tables" / f"embeddings_k{k_final}.csv", index=False)

    # 2D -> PNG statiques ; 3D -> HTML interactifs (rotation, survol = ID tumeur)
    plot_emb = pl.plot_embeddings_3d if args.tsne_dim == 3 else pl.plot_embeddings
    plot_emb(coords, outdir / "figures", k_final)
    plot_emb(coords, outdir / "figures", k_final,
             color_by=coords["item_consensus"], color_label="item consensus")

    # superposition d'une variable clinique (contrôle des confondants)
    if args.metadata and args.color_by:
        meta = pd.read_csv(args.metadata, sep=None, engine="python", index_col=0)
        var = meta.reindex(result.sample_names)[args.color_by]
        plot_emb(coords, outdir / "figures", k_final,
                 color_by=var.reset_index(drop=True),
                 color_label=args.color_by)
        ct = pd.crosstab(labels, var.values)
        ct.to_csv(outdir / "tables" / f"crosstab_{args.color_by}_k{k_final}.csv")
        log.info("Croisement cluster x %s :\n%s", args.color_by, ct.to_string())

    with open(outdir / "run_params.json", "w") as fh:
        json.dump({**vars(args), "outdir": str(args.outdir), "k_final": k_final},
                  fh, indent=2, default=str)

    log.info("Terminé. Résultats dans %s", outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
