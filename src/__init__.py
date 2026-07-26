"""Consensus clustering de tumeurs à partir de bulk RNA-seq.

Les modules de `src/` s'importent en **absolu à plat** (`import consensus`,
`from sigproj import ...`) : le dossier `src/` est mis sur `sys.path` quand on
lance un script directement (`python src/run_pipeline.py`) ou via l'insertion
`sys.path` en tête de `run_pipeline.py` / `null_check.py`. On n'importe donc pas
les sous-modules ici — cela forcerait un contexte de package incompatible.
"""

__version__ = "0.1.0"