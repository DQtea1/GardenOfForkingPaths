"""Garden of Forks — consensus clustering de tumeurs à partir de bulk RNA-seq.

Le paquet s'installe (`pip install -e ".[full]"`) et s'importe normalement :
`from gardenofforks import consensus`. Aucun bricolage de `sys.path` n'est
nécessaire, depuis n'importe quel dossier de travail.

**Convention d'import** : à l'intérieur du paquet, toujours en **relatif**
(`from . import consensus as cc`, `from .consensus import ConsensusResult`) ;
à l'extérieur (tests, notebooks), en **absolu** (`from gardenofforks import …`).

Points d'entrée console (cf. `[project.scripts]` du `pyproject.toml`) :
    gof-run              pipeline complet          -> run_pipeline:main
    gof-demo             démo bout en bout (~30 s) -> demo_pipeline:main
    gof-nullcheck        contrôle nul              -> null_check:main
    gof-make-demo-data   jeu simulé seul           -> make_demo_data:cli

Équivalents module : `python -m gardenofforks.run_pipeline …`
"""

__version__ = "0.1.0"
