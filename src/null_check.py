#!/usr/bin/env python
"""Contrôle par modèle nul — l'étape que tout le monde saute.

Le consensus clustering produit **toujours** des matrices d'apparence
« blocs » et des PAC bas, même sur des données sans aucune structure de
groupe (Senbabaoglu, Michailidis & Li, Sci Rep 2014). Un PAC de 0,04 n'a
donc aucune valeur dans l'absolu : il n'a de sens que comparé à ce qu'on
obtient sur des données de même dimension, même distribution marginale,
mais sans structure.

Deux nuls sont calculés ici :
  - **permutation par gène** : chaque gène est permuté indépendamment entre
    les tumeurs. Détruit toute covariance entre gènes tout en conservant la
    distribution marginale de chacun. Nul le plus sévère.
  - **normal multivarié apparié** : tirage gaussien reproduisant la matrice
    de covariance *entre gènes* mais pas la structure de groupes — utile car
    des gènes corrélés suffisent à créer des blocs sans sous-types réels.

Usage
-----
python null_check.py --counts data/demo_counts.tsv --outdir results/demo \
    --n-resamples 200 --k-max 7
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from src import consensus as cc
from src import metrics as mt
from src import preprocessing as pp


def permute_genes(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute indépendamment chaque colonne (gène)."""
    Xp = X.copy()
    for j in range(Xp.shape[1]):
        rng.shuffle(Xp[:, j])
    return Xp


def gaussian_null(X: np.ndarray, rng: np.random.Generator, n_pc: int = 50) -> np.ndarray:
    """Tirage gaussien conservant la covariance inter-gènes (via une
    approximation de rang `n_pc`, sinon la covariance est ingérable)."""
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    n_pc = min(n_pc, S.size)
    scores = U[:, :n_pc] * S[:n_pc]
    fake = rng.normal(0, scores.std(axis=0), size=scores.shape)
    return fake @ Vt[:n_pc] + X.mean(0)


def run(X: np.ndarray, k_values, args, seed: int, n_jobs: int) -> pd.DataFrame:
    res = cc.consensus_clustering(
        X, k_values=k_values, n_resamples=args.n_resamples,
        prop_samples=args.prop_samples, prop_genes=args.prop_genes,
        sample_mode=args.sample_mode, gene_mode=args.gene_mode,
        base=args.base, metric=args.metric, random_state=seed, n_jobs=n_jobs,
    )
    return mt.summary(res)[["k", "PAC", "auc_cdf", "delta_k"]]


def _run_labeled(X, k_values, args, seed, n_jobs, model, rep) -> pd.DataFrame:
    """Un run étiqueté (observé ou réplicat nul) — tâche indépendante, pensée
    pour être dispatchée en parallèle sur les réplicats."""
    return run(X, k_values, args, seed, n_jobs).assign(model=model, rep=rep)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", required=True)
    p.add_argument("--samples-in-rows", action="store_true")
    p.add_argument("--already-normalized", action="store_true")
    p.add_argument("--n-top-genes", type=int, default=5000)
    p.add_argument("--outdir", type=Path, default=Path("results/null"))
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--n-resamples", type=int, default=200)
    p.add_argument("--n-null", type=int, default=3, help="réplicats par modèle nul")
    p.add_argument("--prop-samples", type=float, default=0.8)
    p.add_argument("--prop-genes", type=float, default=0.8)
    p.add_argument("--sample-mode", default="subsample")
    p.add_argument("--gene-mode", default="subsample")
    p.add_argument("--base", default="hierarchical")
    p.add_argument("--metric", default="pearson")
    p.add_argument("--parallel", choices=["y", "n"], default="y",
                   help="'y' (défaut) : lance les réplicats (observé + nuls) en "
                        "parallèle sur --n-jobs cœurs. 'n' : séquentiel (n_jobs=1).")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    outdir = Path(args.outdir); (outdir / "figures").mkdir(parents=True, exist_ok=True)
    k_values = tuple(range(args.k_min, args.k_max + 1))
    rng = np.random.default_rng(args.seed)

    raw = pp.load_matrix(args.counts, genes_in_rows=not args.samples_in_rows)
    X = pp.preprocess(raw, already_normalized=args.already_normalized,
                      n_top_genes=args.n_top_genes).values

    # Générer les jeux de données (observé + nuls) est séquentiel — `rng` est
    # partagé et son état d'avancement doit rester déterministe. Seuls les runs
    # de consensus clustering sur ces jeux (indépendants une fois générés) sont
    # candidats à la parallélisation.
    jobs = [("observé", 0, X)]
    for r in range(args.n_null):
        jobs.append(("nul: permutation par gène", r, permute_genes(X, rng)))
        jobs.append(("nul: covariance appariée", r, gaussian_null(X, rng)))

    parallel = args.parallel == "y" and len(jobs) > 1
    inner_n_jobs = 1 if parallel else args.n_jobs
    if parallel:
        print(f"{len(jobs)} runs (observé + nuls) en parallèle sur {args.n_jobs} cœurs")
        frames = Parallel(n_jobs=args.n_jobs)(
            delayed(_run_labeled)(Xj, k_values, args, args.seed + r, inner_n_jobs, model, r)
            for model, r, Xj in jobs
        )
    else:
        frames = []
        for model, r, Xj in jobs:
            print(f"{model} (réplicat {r + 1})")
            frames.append(_run_labeled(Xj, k_values, args, args.seed + r,
                                       inner_n_jobs, model, r))

    tab = pd.concat(frames, ignore_index=True)
    tab.to_csv(outdir / "null_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, ax in zip(["PAC", "delta_k"], axes):
        for model, sub in tab.groupby("model"):
            agg = sub.groupby("k")[metric].agg(["mean", "min", "max"])
            style = "o-" if model == "observé" else "s--"
            ax.plot(agg.index, agg["mean"], style, label=model, lw=2 if model == "observé" else 1.2)
            ax.fill_between(agg.index, agg["min"], agg["max"], alpha=0.15)
        ax.set_xlabel("k"); ax.set_ylabel(metric)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("PAC : observé vs. nuls (l'écart est le signal)")
    axes[1].set_title("Δ(K)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "null_comparison.png", dpi=200, bbox_inches="tight")

    print(tab.groupby(["model", "k"])["PAC"].mean().unstack().round(3).to_string())
    print(f"\nFigure : {outdir / 'figures' / 'null_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
