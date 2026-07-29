#!/usr/bin/env python
"""Bout en bout : génère un jeu de données simulé (make_demo_data.py) puis
lance le pipeline complet dessus, pour valider l'installation en ~30 s.

Données et résultats sont écrits dans `demo/`, sous le dossier courant :
    demo/demo_counts.tsv
    demo/demo_metadata.tsv
    demo/results/          (figures, tables, report.html, ...)

Usage
-----
gof-demo                        # -> ./demo/
gof-demo --demo-dir /tmp/essai  # -> /tmp/essai/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import make_demo_data as mdd
from . import run_pipeline as rp


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo-dir", type=Path, default=Path.cwd() / "demo",
                   help="dossier des données et des résultats de démonstration "
                        "(défaut : ./demo)")
    args = p.parse_args(argv)

    data_dir = args.demo_dir
    outdir = data_dir / "results"

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
