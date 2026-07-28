"""Flux commun des branches d'analyse fondées sur un consensus clustering.

Une :class:`AnalysisBranch` reçoit une matrice ``échantillons × variables`` et
exécute le parcours partagé par le consensus historique et les projections ICA :
consensus multi-k, choix de k, partition, exports, embeddings et tests
cliniques. Les étapes propres à l'expression brute (DEGSEA, signatures,
déconvolution) restent volontairement dans l'orchestrateur du pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import catassoc as ca
from . import consensus as cc
from . import correlate as co
from . import embedding as emb
from . import metrics as mt
from . import plots as pl
from . import stability as st


@dataclass(frozen=True)
class BranchSettings:
    """Paramètres communs à une branche de consensus.

    Cette vue typée isole les calculs de branche de ``argparse.Namespace`` : le
    pipeline conserve la CLI existante, tandis que la branche ne dépend que des
    paramètres qu'elle consomme réellement.
    """

    k_values: tuple[int, ...]
    n_resamples: int
    prop_samples: float
    prop_genes: float
    sample_mode: str
    gene_mode: str
    base: str
    metric: str
    linkage_method: str
    min_cluster_size: int
    k_criterion: str
    tsne_dim: int
    run_umap: bool
    perplexity: float
    n_neighbors: int
    min_dist: float
    random_state: int
    n_jobs: int
    color_by: str | None
    run_chi2: bool
    chi2_mc_resamples: int
    ordinal_variables: dict[str, Any]
    run_correlations: bool
    corr_method: str
    corr_all_pairs: bool
    compute_jaccard: bool

    @classmethod
    def from_args(cls, args, *, n_jobs: int) -> "BranchSettings":
        return cls(
            k_values=tuple(range(args.k_min, args.k_max + 1)),
            n_resamples=int(args.n_resamples),
            prop_samples=float(args.prop_samples),
            prop_genes=float(args.prop_genes),
            sample_mode=str(args.sample_mode),
            gene_mode=str(args.gene_mode),
            base=str(args.base),
            metric=str(args.metric),
            linkage_method=str(args.linkage),
            min_cluster_size=int(args.min_cluster_size),
            k_criterion=str(args.k_criterion),
            tsne_dim=int(args.tsne_dim),
            run_umap=not bool(args.no_umap),
            perplexity=float(args.perplexity),
            n_neighbors=int(args.n_neighbors),
            min_dist=float(args.min_dist),
            random_state=int(args.seed),
            n_jobs=int(n_jobs),
            color_by=str(args.color_by) if args.color_by else None,
            run_chi2=args.run_chi2 == "y",
            chi2_mc_resamples=int(args.chi2_mc_resamples),
            ordinal_variables=dict(args.ordinal_variables or {}),
            run_correlations=args.run_correlations == "y",
            corr_method=str(args.corr_method),
            corr_all_pairs=args.corr_all_pairs == "y",
            compute_jaccard=args.compute_jaccard == "y",
        )


@dataclass(frozen=True)
class BranchPaths:
    """Emplacements de sortie d'une branche, sans collision entre branches."""

    root: Path
    table_dir: Path
    figure_dir: Path
    consensus_matrix_dir: Path
    output_subdir: str | None = None

    def create(self) -> None:
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.consensus_matrix_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AnalysisBranch:
    """Exécute et conserve les résultats d'une analyse par consensus.

    ``matrix`` est la seule entrée analytique : expression pour la branche
    historique, métasamples ICA pour une branche ICA. Les deux suivent ensuite
    exactement le même contrat de résultats.
    """

    name: str
    matrix: pd.DataFrame
    paths: BranchPaths
    settings: BranchSettings
    metadata: pd.DataFrame | None = None
    forced_k: int | None = None
    forced_k_name: str = "k_final"
    input_export_name: str | None = None
    logger: logging.Logger | None = None

    result: Any = None
    k_final: int | None = None
    labels: np.ndarray | None = None
    items: pd.DataFrame | None = None
    coords: pd.DataFrame | None = None
    # Une table d'embeddings par K. Chaque table est calculée sur D_K = 1 - C_K,
    # et non sur la seule distance du K retenu.
    coords_by_k: dict[int, pd.DataFrame] = field(default_factory=dict)
    color_var: np.ndarray | None = None
    assoc: dict | None = None
    corr: dict | None = None
    branch_stability: Any = None
    branch_stability_by_k: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = self.matrix.copy()
        self.matrix.index = self.matrix.index.astype(str)
        self.matrix.columns = self.matrix.columns.astype(str)
        if self.metadata is not None:
            self.metadata = self.metadata.copy()
            self.metadata.index = self.metadata.index.astype(str)
        if self.logger is None:
            self.logger = logging.getLogger("pipeline")

    @property
    def sample_names(self) -> np.ndarray:
        return self.matrix.index.to_numpy()

    @property
    def k_values(self) -> tuple[int, ...]:
        return self.settings.k_values

    def aligned_metadata(self) -> pd.DataFrame | None:
        if self.metadata is None:
            return None
        return self.metadata.reindex(self.sample_names)

    def run(self, *, run_associations: bool = True,
            run_correlations: bool = False,
            correlation_extra_features: pd.DataFrame | None = None,
            correlation_prefix: str = "ica") -> "AnalysisBranch":
        """Exécute le flux commun jusqu'aux sorties cliniques demandées."""
        self.paths.create()
        if self.input_export_name:
            self.matrix.to_csv(
                self.paths.table_dir / self.input_export_name,
                index_label="sample",
            )
        self._run_consensus()
        self._select_k_and_diagnostics()
        self._finalize_partition()
        self._run_branch_stability()
        self._run_embeddings()
        if run_associations:
            self.run_associations()
        if run_correlations:
            self.run_correlations(
                extra_features=correlation_extra_features,
                extra_prefix=correlation_prefix,
            )
        return self

    def _run_consensus(self) -> None:
        self.logger.info(
            "%s — consensus clustering sur %d variables…",
            self.name,
            self.matrix.shape[1],
        )
        s = self.settings
        self.result = cc.consensus_clustering(
            self.matrix.values,
            k_values=s.k_values,
            n_resamples=s.n_resamples,
            prop_samples=s.prop_samples,
            prop_genes=s.prop_genes,
            sample_mode=s.sample_mode,
            gene_mode=s.gene_mode,
            base=s.base,
            metric=s.metric,
            linkage_method=s.linkage_method,
            sample_names=self.sample_names,
            random_state=s.random_state,
            n_jobs=s.n_jobs,
        )

    def _select_k_and_diagnostics(self) -> None:
        tab = mt.summary(self.result)
        tab.to_csv(self.paths.table_dir / "k_selection.csv", index=False)
        self.logger.info("%s — diagnostics par k :\n%s", self.name, tab.to_string(index=False))

        if self.forced_k is not None:
            if self.forced_k not in self.k_values:
                raise ValueError(
                    f"{self.forced_k_name} doit être compris entre k_min et k_max."
                )
            self.k_final, reason = int(self.forced_k), "imposé"
        else:
            self.k_final = mt.suggest_k(
                self.result,
                self.settings.min_cluster_size,
                method=self.settings.k_criterion,
            )
            reason = f"auto ({self.settings.k_criterion})"
        self.logger.info("%s — k retenu : %d (%s)", self.name, self.k_final, reason)

        pl.plot_cdf(self.result, self.paths.figure_dir)
        pl.plot_tracking(self.result, self.paths.figure_dir)
        for k in self.k_values:
            pl.plot_consensus_heatmap(
                self.result,
                k,
                self.paths.figure_dir,
                self.settings.linkage_method,
            )

    def _finalize_partition(self) -> None:
        self.labels = self.result.labels(self.k_final, self.settings.linkage_method)
        self.items = mt.item_consensus(self.result, self.k_final)
        silhouette = mt.silhouette_per_sample(self.result, self.k_final)
        assignments = (
            self.items.merge(silhouette[["sample", "silhouette"]], on="sample")
            .sort_values(["cluster", "item_consensus"], ascending=[True, False])
        )
        assignments.to_csv(
            self.paths.table_dir / f"cluster_assignments_k{self.k_final}.csv",
            index=False,
        )
        mt.cluster_consensus(self.result, self.k_final).to_csv(
            self.paths.table_dir / f"cluster_consensus_k{self.k_final}.csv",
            index=False,
        )
        pl.plot_item_consensus(self.items, self.paths.figure_dir, self.k_final)

        np.save(
            self.paths.consensus_matrix_dir / f"consensus_matrix_k{self.k_final}.npy",
            self.result.consensus[self.k_final],
        )
        pd.DataFrame(
            self.result.distance(self.k_final),
            index=self.result.sample_names,
            columns=self.result.sample_names,
        ).to_csv(
            self.paths.table_dir / f"consensus_distance_k{self.k_final}.csv.gz",
            compression="gzip",
        )

    def _run_branch_stability(self) -> None:
        """Calcule les Jaccard de toutes les branches et les exporte.

        Cette étape est commune à la matrice transcriptomique historique et à
        chaque projection ICA : le rapport peut ainsi afficher la même
        annotation de stabilité sur l'arbre de tout ``k``.
        """
        self.branch_stability = None
        self.branch_stability_by_k = {}
        if not self.settings.compute_jaccard:
            return

        s = self.settings
        self.branch_stability_by_k = st.branch_stability_multi(
            self.matrix.values,
            {k: self.result.distance(k) for k in self.k_values},
            n_resamples=s.n_resamples,
            gene_mode="bootstrap",
            prop_genes=1.0,
            metric=s.metric,
            linkage_method=s.linkage_method,
            min_size=2,
            random_state=s.random_state,
            n_jobs=s.n_jobs,
        )
        self.branch_stability = self.branch_stability_by_k[self.k_final]
        table = self.branch_stability.to_frame(
            sample_names=self.result.sample_names,
            final_labels=self.labels,
        )
        table.to_csv(
            self.paths.table_dir / f"branch_stability_k{self.k_final}.csv",
            index=False,
        )
        pl.plot_branch_stability(
            self.branch_stability,
            self.paths.figure_dir,
            self.k_final,
        )
        n_stable = int((table["stability"] >= 0.75).sum())
        finals = table.loc[
            table["is_final_cluster"], ["branch_id", "size", "stability"]
        ]
        self.logger.info(
            "%s — stabilité Jaccard : %d/%d branches stables (>= 0,75). "
            "Clusters finaux (k=%d):\n%s",
            self.name,
            n_stable,
            len(table),
            self.k_final,
            finals.to_string(index=False) if len(finals) else "(aucun cluster == branche)",
        )

    def _run_embeddings(self) -> None:
        s = self.settings
        self.coords_by_k = {}
        for k in self.k_values:
            # C_K dépend de K : l'embedding doit donc être recalculé sur
            # D_K = 1 - C_K. Le décalage de graine rend chaque K reproductible
            # indépendamment de l'ordre des dimensions testées.
            labels_k = self.result.labels(k, s.linkage_method)
            items_k = mt.item_consensus(self.result, k)
            coords_k = emb.embeddings_table(
                self.result.distance(k),
                self.result.sample_names,
                labels_k,
                run_umap=s.run_umap,
                n_components=s.tsne_dim,
                perplexity=s.perplexity,
                n_neighbors=s.n_neighbors,
                min_dist=s.min_dist,
                random_state=s.random_state + int(k),
                n_jobs=s.n_jobs,
            ).merge(items_k[["sample", "item_consensus"]], on="sample")
            self.coords_by_k[int(k)] = coords_k
            coords_k.to_csv(
                self.paths.table_dir / f"embeddings_k{k}.csv",
                index=False,
            )

        # Compatibilité avec les consommateurs qui représentent la partition
        # finale (figures statiques, synthèse) : ``coords`` reste le K retenu.
        self.coords = self.coords_by_k[int(self.k_final)]

        plot_emb = pl.plot_embeddings_3d if s.tsne_dim == 3 else pl.plot_embeddings
        plot_emb(self.coords, self.paths.figure_dir, self.k_final)
        plot_emb(
            self.coords,
            self.paths.figure_dir,
            self.k_final,
            color_by=self.coords["item_consensus"],
            color_label="item consensus",
        )

        self.color_var = None
        metadata = self.aligned_metadata()
        if metadata is None or not s.color_by:
            return
        if s.color_by not in metadata:
            self.logger.warning(
                "%s — color_by=%s absent des métadonnées : superposition ignorée.",
                self.name,
                s.color_by,
            )
            return
        color = metadata[s.color_by]
        self.color_var = color.to_numpy()
        plot_emb(
            self.coords,
            self.paths.figure_dir,
            self.k_final,
            color_by=color.reset_index(drop=True),
            color_label=s.color_by,
        )
        crosstab = pd.crosstab(self.labels, color.to_numpy())
        crosstab.to_csv(
            self.paths.table_dir / f"crosstab_{s.color_by}_k{self.k_final}.csv"
        )
        self.logger.info(
            "%s — croisement cluster x %s :\n%s",
            self.name,
            s.color_by,
            crosstab.to_string(),
        )

    def run_associations(self) -> dict | None:
        """Exécute les associations catégorielles pour la branche si demandées."""
        self.assoc = None
        if not self.settings.run_chi2:
            return self.assoc
        metadata = self.aligned_metadata()
        if metadata is None:
            self.logger.info("%s — khi² ignoré : pas de métadonnées.", self.name)
            return self.assoc
        self.assoc = ca.run_categorical_association(
            {
                k: self.result.labels(k, self.settings.linkage_method)
                for k in self.k_values
            },
            metadata,
            self.result.sample_names,
            self.paths.root,
            self.k_final,
            ordinal=self.settings.ordinal_variables,
            mc_resamples=self.settings.chi2_mc_resamples,
            seed=self.settings.random_state,
            output_subdir=self.paths.output_subdir,
        )
        return self.assoc

    def run_correlations(self, *, sig_scores: dict | None = None,
                         deconv: dict | None = None,
                         extra_features: pd.DataFrame | None = None,
                         extra_prefix: str = "ica") -> dict | None:
        """Exécute les corrélations continues avec les sorties de la branche."""
        self.corr = None
        if not self.settings.run_correlations:
            return self.corr
        self.corr = co.run_correlations(
            sig_scores,
            deconv,
            self.aligned_metadata(),
            self.result.sample_names,
            self.paths.root,
            method=self.settings.corr_method,
            all_pairs=self.settings.corr_all_pairs,
            extra_features=extra_features,
            extra_prefix=extra_prefix,
            output_subdir=self.paths.output_subdir,
        )
        return self.corr


__all__ = ["AnalysisBranch", "BranchPaths", "BranchSettings"]
