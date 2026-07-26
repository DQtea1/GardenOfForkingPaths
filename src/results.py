"""Conteneur des résultats du pipeline.

Regroupe en **un seul objet** tout ce qu'une étape produit et que le rapport (ou
la figure de synthèse) consomme, au lieu de faire circuler une douzaine de
variables faiblement reliées. `build_report(results, outdir)` remplace ainsi un
appel à ~13 arguments.

C'est aussi la **fondation** d'un export unique type AnnData (`.h5ad`) : tout est
déjà rassemblé et aligné sur `result.sample_names`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineResults:
    """Sorties du run, alignées sur `result.sample_names`.

    Champs obligatoires : la partition consensus et le k retenu. Les autres sont
    remplis au fil des étapes (``None`` / ``{}`` si l'étape n'a pas tourné).
    """

    result: Any                                  # consensus.ConsensusResult
    k_final: int
    linkage_method: str = "average"
    min_cluster_size: int = 10
    k_criterion: str = "both"

    coords: Any = None                           # embeddings t-SNE / UMAP (DataFrame)
    meta: Any = None                             # métadonnées cliniques (DataFrame)
    sig_scores: dict | None = None               # {"ssgsea": df, "mean": df}
    sig_tests: dict | None = None                # tests de Wilcoxon (7.2bis)
    deconv: dict | None = None                   # {méthode: df types × tumeurs}
    degsea_by_k: dict | None = None              # {k: {collection: matrice NES}}
    branch_stability_by_k: dict | None = None    # {k: stability.BranchStability}
    assoc: dict | None = None                    # khi² (9a) prêt pour le rapport
    corr: dict | None = None                     # corrélations (9b) prêtes pour le rapport
