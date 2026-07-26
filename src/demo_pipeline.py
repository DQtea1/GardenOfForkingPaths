#!/usr/bin/env python
"""Bout en bout : génère un jeu de données simulé (make_demo_data.py) puis
lance le pipeline complet dessus, pour valider l'installation en ~30 s.

Données et résultats sont écrits dans demo/ (à la racine du dépôt) :
    demo/demo_counts.tsv
    demo/demo_metadata.tsv
    demo/results/          (figures, tables, report.html, ...)

Usage
-----
python demo_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import make_demo_data as mdd
from src import run_pipeline as rp

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"


def main() -> int:
    data_dir = DEMO_DIR
    outdir = DEMO_DIR / "results"

    mdd.main(data_dir)

    return rp.main([
        "--counts", str(data_dir / "demo_counts.tsv"),
        "--metadata", str(data_dir / "demo_metadata.tsv"),
        "--color-by", "true_subtype",
        "--outdir", str(outdir),
        "--n-resamples", "300",
        "--k-max", "7",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
