"""Consensus clustering de tumeurs à partir de bulk RNA-seq.

`src/` est un **package** (`src`). Les scripts d'entrée (`run_pipeline.py`,
`null_check.py`, `demo_pipeline.py`) mettent la **racine du dépôt** sur
`sys.path` puis importent `from src import <module>` ; les modules internes
s'importent en **relatif** (`from .consensus import ...`). Lancer depuis la
racine du dépôt : `python src/run_pipeline.py …` ou `python -m src.run_pipeline`.
"""

__version__ = "0.1.0"