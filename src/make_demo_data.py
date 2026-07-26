#!/usr/bin/env python
"""Simule 500 tumeurs avec 4 sous-types transcriptionnels + un continuum,
pour vérifier le pipeline de bout en bout avant de brancher les vraies données.

La simulation reproduit trois difficultés réelles :
  - des tailles de groupes déséquilibrées (200 / 140 / 100 / 60)
  - un gradient continu (pureté tumorale) orthogonal aux sous-types, qui
    créera de faux clusters si on ne le surveille pas
  - 12 % de tumeurs "atypiques" à profil intermédiaire entre deux sous-types
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES, N_GENES = 500, 8000
SUBTYPE_SIZES = [200, 140, 100, 60]
FRAC_ATYPICAL = 0.12
SEED = 42


def main(outdir: str | Path = "data") -> None:
    rng = np.random.default_rng(SEED)
    subtype = np.concatenate([[i] * n for i, n in enumerate(SUBTYPE_SIZES)])
    rng.shuffle(subtype)
    n_sub = len(SUBTYPE_SIZES)

    # programmes transcriptionnels : 300 gènes par sous-type
    base = rng.normal(6.0, 1.5, N_GENES)
    loadings = np.zeros((n_sub, N_GENES))
    for s in range(n_sub):
        idx = rng.choice(N_GENES, 300, replace=False)
        loadings[s, idx] = rng.normal(1.6, 0.4, 300)

    W = np.eye(n_sub)[subtype].astype(float)

    # tumeurs atypiques : mélange de deux sous-types
    n_atyp = int(FRAC_ATYPICAL * N_SAMPLES)
    atyp = rng.choice(N_SAMPLES, n_atyp, replace=False)
    for i in atyp:
        other = rng.choice([s for s in range(n_sub) if s != subtype[i]])
        alpha = rng.uniform(0.35, 0.65)
        W[i] = 0
        W[i, subtype[i]] = alpha
        W[i, other] = 1 - alpha

    # gradient continu type "pureté tumorale"
    purity = rng.uniform(0.3, 0.95, N_SAMPLES)
    purity_program = np.zeros(N_GENES)
    purity_program[rng.choice(N_GENES, 500, replace=False)] = rng.normal(1.2, 0.3, 500)

    log_expr = base + W @ loadings + purity[:, None] * purity_program
    log_expr += rng.normal(0, 0.7, log_expr.shape)

    # counts par un modèle binomial négatif approché
    mu = np.exp(log_expr / 1.5)
    lib = rng.uniform(0.7, 1.4, (N_SAMPLES, 1))
    counts = rng.negative_binomial(n=5, p=5 / (5 + mu * lib)).astype(int)

    samples = [f"TUM_{i:03d}" for i in range(N_SAMPLES)]
    genes = [f"GENE{i:05d}" for i in range(N_GENES)]
    # quelques gènes techniques pour tester le filtrage
    for j, name in enumerate(["RPL13A", "RPS6", "MT-CO1", "MT-ND4", "HBB", "HBA1"]):
        genes[j] = name
        counts[:, j] *= 50

    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts.T, index=genes, columns=samples).to_csv(
        out / "demo_counts.tsv", sep="\t")
    pd.DataFrame(
        {"sample": samples, "true_subtype": [f"ST{s+1}" for s in subtype],
         "purity": purity.round(3),
         "atypical": np.isin(np.arange(N_SAMPLES), atyp)}
    ).set_index("sample").to_csv(out / "demo_metadata.tsv", sep="\t")

    print(f"{out}/demo_counts.tsv     {counts.T.shape[0]} gènes x {N_SAMPLES} tumeurs")
    print(f"{out}/demo_metadata.tsv   vérité terrain : 4 sous-types, {n_atyp} atypiques")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data",
                        help="dossier de sortie pour demo_counts.tsv / demo_metadata.tsv")
    main(parser.parse_args().outdir)
