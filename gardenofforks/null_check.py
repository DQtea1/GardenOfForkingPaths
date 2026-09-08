#!/usr/bin/env python
r"""Contrôle par modèle nul — l'étape que tout le monde saute.

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
gof-nullcheck --counts data/demo_counts.tsv --outdir results/demo \
    --n-resamples 200 --k-max 7

Équivalent sans la commande console (le lancement par chemin de fichier casse
les imports relatifs du paquet) :

    python -m gardenofforks.null_check …
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from . import config as cf
from . import consensus as cc
from . import metrics as mt
from . import preprocessing as pp


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


def build_parser() -> argparse.ArgumentParser:
    """Le parser du pipeline, plus l'option propre au contrôle nul.

    Les options sont **celles de `gof-run`** — mêmes noms, mêmes types, mêmes
    valeurs autorisées : un `--metric` accepté par le pipeline l'est ici aussi,
    et une faute de frappe est refusée des deux côtés. Auparavant ce module
    redéclarait une quinzaine d'options à la main, avec ses propres défauts et
    sans les `choices` : `--base kmedoid` y passait sans broncher.

    Seuls les défauts de cadrage restent locaux, le contrôle nul étant plus
    léger qu'un run complet (moins de rééchantillonnages, plage de k plus large).
    """
    p = cf.build_parser()
    p.description = __doc__
    p.add_argument("--n-null", type=int, default=3,
                   help="réplicats par modèle nul (défaut 3).")
    p.set_defaults(outdir=Path("results/null"), k_min=2, k_max=8, n_resamples=200)
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not args.counts:
        p.error("--counts est obligatoire.")

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
