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
    p.add_argument("--regenerate", action="store_true",
                   help="régénère les données simulées MÊME si elles existent "
                        "déjà, en écrasant demo_counts.tsv / demo_metadata.tsv.")
    args = p.parse_args(argv)

    data_dir = args.demo_dir
    outdir = data_dir / "results"
    counts = data_dir / "demo_counts.tsv"
    metadata = data_dir / "demo_metadata.tsv"

    # Le dossier de démo peut contenir un jeu préparé à la main (p. ex. celui à
    # nomenclatures mixtes de demo/make_mixed_ids_demo.py). Le régénérer sans
    # prévenir le détruirait : on ne simule que s'il n'y a rien, ou sur demande.
    if args.regenerate or not (counts.exists() and metadata.exists()):
        mdd.main(data_dir)
    else:
        print(f"{counts} existe déjà : données conservées "
              f"(--regenerate pour les resimuler).")

    return rp.main([
        "--counts", str(counts),
        "--metadata", str(metadata),
        "--color-by", "true_subtype",
        "--outdir", str(outdir),
        "--n-resamples", "300",
        "--k-max", "7",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
