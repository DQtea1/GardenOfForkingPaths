"""Figures : heatmap consensus, CDF, PAC/delta-K, tracking plot, embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

from .consensus import ConsensusResult
from .metrics import consensus_cdf, delta_k, pac
from .stability import BranchStability

CONSENSUS_CMAP = LinearSegmentedColormap.from_list(
    "consensus", ["#ffffff", "#c6dbef", "#4292c6", "#08306b"]
)
CLUSTER_COLORS = plt.get_cmap("tab10").colors


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> Path:
    """Écrit une figure et la referme. Point de passage unique du paquet.

    Refermer est ce qui évite la fuite de mémoire matplotlib sur un run qui
    produit des centaines de figures (une par k, par branche, par variable).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _save(fig: plt.Figure, outdir: Path, name: str, dpi: int = 200) -> Path:
    """Variante « dossier + nom de fichier », employée dans tout ce module."""
    return save_figure(fig, Path(outdir) / name, dpi=dpi)


def plot_consensus_heatmap(result: ConsensusResult, k: int, outdir: Path,
                           linkage_method: str = "average") -> Path:
    """Heatmap de la matrice consensus réordonnée par le dendrogramme,
    avec barre de clusters. La lecture visuelle est le premier critère :
    on veut des blocs nets, pas un dégradé continu."""
    C = result.consensus[k]
    Z = result.linkage_tree(k, linkage_method)   # mémoïsé côté ConsensusResult
    order = result.order(k, linkage_method)
    labels = result.labels(k, linkage_method)[order]

    fig = plt.figure(figsize=(7.5, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 6], width_ratios=[20, 1],
                          hspace=0.02, wspace=0.03)

    ax_d = fig.add_subplot(gs[0, 0])
    dendrogram(Z, ax=ax_d, no_labels=True, color_threshold=0, link_color_func=lambda _: "#555")
    ax_d.set_axis_off()

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(C[np.ix_(order, order)], cmap=CONSENSUS_CMAP, vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{C.shape[0]} tumeurs  —  k = {k}  —  PAC = {pac(C):.3f}")

    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.imshow(labels[:, None], cmap=matplotlib.colors.ListedColormap(
        CLUSTER_COLORS[: labels.max()]), aspect="auto", interpolation="nearest")
    ax_c.set_xticks([]); ax_c.set_yticks([])

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.4])
    fig.colorbar(im, cax=cax, label="indice de consensus")
    fig.suptitle(f"Matrice de consensus — k = {k}", y=0.94)
    return _save(fig, outdir, f"consensus_heatmap_k{k}.png")


def plot_cdf(result: ConsensusResult, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = sorted(result.consensus)
    for i, k in enumerate(ks):
        grid, cdf = consensus_cdf(result.consensus[k])
        axes[0].plot(grid, cdf, label=f"k={k}", color=plt.get_cmap("viridis")(i / max(1, len(ks) - 1)))
    axes[0].set_xlabel("indice de consensus"); axes[0].set_ylabel("CDF")
    axes[0].set_title("CDF du consensus"); axes[0].legend(fontsize=8)

    dk = delta_k(result)
    pacs = [pac(result.consensus[k]) for k in ks]
    axes[1].plot(ks, pacs, "o-", color="#c0392b", label="PAC (à minimiser)")
    ax2 = axes[1].twinx()
    ax2.plot(dk["k"], dk["delta_k"], "s--", color="#2980b9", label="Δ(K)")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("PAC", color="#c0392b")
    ax2.set_ylabel("Δ(K)", color="#2980b9")
    axes[1].set_title("Choix de k")
    fig.tight_layout()
    return _save(fig, outdir, "cdf_pac_deltak.png")


def plot_tracking(result: ConsensusResult, outdir: Path) -> Path:
    """Tracking plot : suivi de l'affectation de chaque tumeur quand k augmente.
    Révèle les clusters qui se scindent proprement vs. ceux qui se réorganisent."""
    ks = sorted(result.consensus)
    order = result.order(max(ks))
    mat = np.vstack([result.labels(k)[order] for k in ks])
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(ks) + 1.5))
    ax.imshow(mat, aspect="auto", interpolation="nearest",
              cmap=matplotlib.colors.ListedColormap(CLUSTER_COLORS[: int(mat.max())]))
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"k={k}" for k in ks])
    ax.set_xticks([]); ax.set_xlabel("tumeurs (ordre du dendrogramme à k max)")
    ax.set_title("Tracking plot")
    return _save(fig, outdir, "tracking_plot.png")


def plot_embeddings(emb: pd.DataFrame, outdir: Path, k: int,
                    color_by: pd.Series | None = None,
                    color_label: str = "cluster") -> Path:
    """Nuages t-SNE et UMAP, colorés par cluster consensus (ou autre variable)."""
    methods = [("tsne1", "tsne2", "t-SNE")]
    if "umap1" in emb.columns:
        methods.append(("umap1", "umap2", "UMAP"))

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5.2),
                             squeeze=False)
    values = emb["cluster"] if color_by is None else pd.Series(color_by).reset_index(drop=True)
    categorical = color_by is None or not pd.api.types.is_numeric_dtype(values)

    for ax, (x, y, name) in zip(axes[0], methods):
        if categorical:
            for i, g in enumerate(pd.unique(values)):
                m = (values == g).values
                ax.scatter(emb.loc[m, x], emb.loc[m, y], s=26, alpha=0.85,
                           color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                           edgecolor="white", linewidth=0.4, label=f"{color_label} {g}")
            ax.legend(fontsize=8, frameon=False)
        else:
            sc = ax.scatter(emb[x], emb[y], c=values, s=26, cmap="viridis",
                            edgecolor="white", linewidth=0.4)
            fig.colorbar(sc, ax=ax, label=color_label)
        ax.set_xlabel(f"{name} 1"); ax.set_ylabel(f"{name} 2")
        ax.set_title(f"{name} sur distance consensus (k={k})")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    suffix = "cluster" if color_by is None else color_label.replace(" ", "_")
    return _save(fig, outdir, f"embeddings_k{k}_{suffix}.png")


def plot_deconvolution(frac: pd.DataFrame, labels, sample_names, method: str,
                       outdir: Path) -> Path:
    """Heatmap de la composition cellulaire **moyenne par cluster** (clusters ×
    types cellulaires) pour une méthode de déconvolution. Valeurs z-scorées par
    type cellulaire pour comparer les clusters quelle que soit l'échelle de la
    méthode (scores MCP/xCell vs fractions quanTIseq/EPIC/DWLS/BayesPrism)."""
    frac = frac.reindex(columns=[str(s) for s in sample_names])
    frac = frac.dropna(axis=1, how="all")
    lab = pd.Series(list(labels), index=[str(s) for s in sample_names])
    lab = lab.reindex(frac.columns)
    clusters = sorted(lab.dropna().unique())
    # moyenne par cluster : cell_type × cluster
    M = pd.DataFrame({c: frac.loc[:, lab[lab == c].index].mean(axis=1) for c in clusters})
    Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1).replace(0, 1), axis=0)

    n_ct, n_cl = Z.shape
    fig, ax = plt.subplots(figsize=(1.0 * n_cl + 3.2, 0.42 * n_ct + 1.6))
    vmax = float(np.nanmax(np.abs(Z.values))) or 1.0
    im = ax.imshow(Z.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(n_cl)); ax.set_xticklabels([f"C{c}" for c in clusters])
    ax.set_yticks(range(n_ct)); ax.set_yticklabels([t[:34] for t in Z.index], fontsize=8)
    ax.set_xlabel("cluster")
    ax.set_title(f"Déconvolution — {method} (moyenne par cluster, z par type)")
    for i in range(n_ct):
        for j in range(n_cl):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=6.5,
                    color="black" if abs(Z.values[i, j]) < 0.6 * vmax else "white")
    fig.colorbar(im, ax=ax, label="z-score (par type cellulaire)", fraction=0.03, pad=0.02)
    return _save(fig, outdir, f"deconv_{method}.png")


# SUPPRIMÉ — plot_signature_boxplots / plot_signature_heatmap : figures
# statiques par variable clinique, dont les appels étaient commentés dans
# sigproj (« Added unnecessary computing time »). Le rapport HTML rend la même
# chose en interactif, onglet « Boxplots ». Code dans l'historique git.


def plot_gsea_ova_heatmap(nes: pd.DataFrame, outdir: Path,
                          pval: float | None = None,
                          collection: str | None = None) -> Path:
    """Heatmap de synthèse GSEA one-vs-all : pathways (lignes) × clusters
    (colonnes), colorés par NES (rouge = enrichi dans le cluster, bleu =
    déplété). Contient tous les pathways significatifs (p < `pval`) dans au
    moins un cluster. `collection` (optionnel) suffixe le titre et le fichier
    quand plusieurs collections de gene sets sont testées."""
    n_terms, n_clusters = nes.shape
    fig, ax = plt.subplots(figsize=(1.1 * n_clusters + 3.4, 0.32 * n_terms + 1.6))
    vmax = float(np.nanmax(np.abs(nes.values))) or 1.0
    im = ax.imshow(nes.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n_clusters)); ax.set_xticklabels(nes.columns)
    ax.set_yticks(range(n_terms))
    ax.set_yticklabels([t[:44] for t in nes.index], fontsize=8)
    ax.set_xlabel("cluster (one-vs-all)")
    sig = f"significatifs (p < {pval:g})" if pval is not None else "significatifs"
    coll = f" — {collection}" if collection else ""
    ax.set_title(f"GSEA{coll} — NES par cluster ({n_terms} pathways {sig})")
    for i in range(n_terms):
        for j in range(n_clusters):
            v = nes.values[i, j]
            if np.isfinite(v) and abs(v) >= 0.6 * vmax:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white")
    fig.colorbar(im, ax=ax, label="NES", fraction=0.025, pad=0.02)
    suffix = f"_{collection}" if collection else ""
    return _save(fig, outdir, f"gsea_ova_heatmap{suffix}.png")


def plot_purity(purity: pd.Series, keep: pd.Series, threshold: float,
                direction: str, outdir: Path) -> Path:
    """Histogramme des puretés PUREE, seuil de filtrage marqué ; la zone retirée
    (puretés `< seuil` si on garde les hautes, `> seuil` sinon) est grisée."""
    vals = pd.Series(purity).dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(vals, bins=30, color="#4292c6", edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, ls="--", lw=1.2, color="#c0392b",
               label=f"seuil = {threshold:g} (garde puretés {'≥' if direction == 'higher' else '≤'} seuil)")
    lo, hi = (0.0, threshold) if direction == "higher" else (threshold, 1.0)
    ax.axvspan(lo, hi, color="grey", alpha=0.12)
    n_rm = int((~keep.astype(bool)).sum())
    ax.set_xlabel("pureté tumorale (PUREE)")
    ax.set_ylabel("nombre de tumeurs")
    ax.set_title(f"Pureté PUREE — {n_rm} tumeur(s) retirée(s)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, "purity_puree.png")


def plot_pca_outliers(diag: pd.DataFrame, outdir: Path, sd_threshold: float) -> Path:
    """PC1 vs PC2 de l'ACP, tumeurs aberrantes marquées et annotées.

    Les pointillés gris = seuil ±`sd_threshold` SD sur PC1 / PC2. Attention : le
    flag utilise aussi les composantes au-delà de PC2 ; une tumeur à l'intérieur
    du cadre peut donc être marquée aberrante (outlier sur une CP supérieure)."""
    keep = ~diag["is_outlier"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(diag.loc[keep, "PC1"], diag.loc[keep, "PC2"], s=26, alpha=0.8,
               color="#4292c6", edgecolor="white", linewidth=0.4, label="conservées")
    out = ~keep
    if out.any():
        ax.scatter(diag.loc[out, "PC1"], diag.loc[out, "PC2"], s=52, alpha=0.9,
                   color="#c0392b", edgecolor="black", linewidth=0.5, marker="X",
                   label="aberrantes")
        for _, r in diag.loc[out].iterrows():
            ax.annotate(str(r["sample"]), (r["PC1"], r["PC2"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")

    for col, line in (("PC1", ax.axvline), ("PC2", ax.axhline)):
        mean, sd = diag[col].mean(), diag[col].std(ddof=0)
        for s in (-1, 1):
            line(mean + s * sd_threshold * sd, ls="--", lw=0.7, color="grey")

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Détection d'outliers — ACP (> {sd_threshold:g} SD)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, "pca_outliers.png")


def plot_branch_stability(bs: BranchStability, outdir: Path, k: int,
                          cmap_name: str = "RdYlGn") -> Path:
    """Dendrogramme de l'arbre consensus dont chaque branche est colorée par son
    score de stabilité Jaccard (rouge = instable, vert = stable). Chaque lien en
    U correspond à un nœud interne, donc à une branche ; les branches grises sont
    exclues du score (racine / singletons)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    stab = dict(zip(bs.node_ids, bs.stability))
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(cmap_name)

    def link_color(node_id: int) -> str:
        s = stab.get(node_id)
        return "#bbbbbb" if s is None else matplotlib.colors.to_hex(cmap(norm(s)))

    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(bs.linkage, ax=ax, no_labels=True, link_color_func=link_color)
    ax.set_xticks([])
    ax.set_ylabel("distance consensus (hauteur de fusion)")
    ax.set_title(f"Stabilité des branches — Jaccard bootstrap des gènes (k={k})")
    ax.spines[["top", "right"]].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="stabilité (moyenne du Jaccard max sur B bootstraps)")
    return _save(fig, outdir, f"branch_stability_dendrogram_k{k}.png")


def plot_embeddings_3d(emb: pd.DataFrame, outdir: Path, k: int,
                       color_by: pd.Series | None = None,
                       color_label: str = "cluster") -> list[Path]:
    """Nuages t-SNE / UMAP **3D interactifs**, un fichier HTML autonome par
    méthode : rotation à la souris, zoom, et survol d'un point pour afficher
    l'ID de la tumeur (+ son item consensus). Ouvre le .html dans un navigateur.

    Nécessite `plotly` ; installe-le avec `pip install plotly` ou reste en 2D
    (`--t-SNE_dim 2`, défaut). Même mise en garde qu'en 2D : l'embedding est
    calculé sur la distance consensus, la position *relative* des nuages n'a
    pas de sens, seule la structure locale est interprétable.
    """
    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotly absent : `pip install plotly` "
            "(nécessaire pour les embeddings 3D, --t-SNE_dim 3)"
        ) from exc

    outdir.mkdir(parents=True, exist_ok=True)
    methods = [("tsne1", "tsne2", "tsne3", "t-SNE")]
    if "umap3" in emb.columns:
        methods.append(("umap1", "umap2", "umap3", "UMAP"))

    if color_by is None:
        values = emb["cluster"].astype(str)
        label, categorical = "cluster", True
    else:
        values = pd.Series(color_by).reset_index(drop=True)
        label = color_label
        categorical = not pd.api.types.is_numeric_dtype(values)
        if categorical:
            values = values.astype(str)

    discrete = [matplotlib.colors.to_hex(c) for c in CLUSTER_COLORS]
    suffix = "cluster" if color_by is None else color_label.replace(" ", "_")
    paths: list[Path] = []
    for x, y, z, name in methods:
        df = emb.copy()
        df[label] = values.values
        hover_data = {x: False, y: False, z: False}
        if "item_consensus" in df.columns:
            hover_data["item_consensus"] = ":.3f"
        fig = px.scatter_3d(
            df, x=x, y=y, z=z, color=label,
            hover_name="sample", hover_data=hover_data,
            color_discrete_sequence=discrete,
            color_continuous_scale="Viridis",
            title=f"{name} 3D sur distance consensus (k={k}) — {label}",
        )
        fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="white")))
        fig.update_layout(
            legend_title_text=label,
            scene=dict(xaxis_title=f"{name} 1", yaxis_title=f"{name} 2",
                       zaxis_title=f"{name} 3"),
        )
        path = outdir / f"embeddings3d_k{k}_{name.replace('-', '').lower()}_{suffix}.html"
        fig.write_html(path, include_plotlyjs=True)
        paths.append(path)
    return paths


# SUPPRIMÉ — plot_cluster_overview (figure de synthèse) : la fonction était
# entièrement commentée depuis longtemps, et `run_pipeline._synthesis` l'appelait
# encore. Le jour où l'appel aurait été décommenté, il aurait levé une
# AttributeError. Code disponible dans l'historique git si la figure revient.


def plot_item_consensus(items: pd.DataFrame, outdir: Path, k: int) -> Path:
    """Distribution du consensus par tumeur, cluster par cluster.
    Les points bas = tumeurs ambiguës, à examiner individuellement."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    clusters = sorted(items["cluster"].unique())
    data = [items.loc[items["cluster"] == c, "item_consensus"].values for c in clusters]
    ax.boxplot(data, tick_labels=[f"C{c}" for c in clusters], showfliers=False)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(rng.normal(i, 0.06, d.size), d, s=14, alpha=0.6,
                   color=CLUSTER_COLORS[(i - 1) % len(CLUSTER_COLORS)])
    ax.axhline(0.8, ls="--", lw=0.8, color="grey")
    ax.set_ylabel("item consensus"); ax.set_title(f"Stabilité par tumeur (k={k})")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, f"item_consensus_k{k}.png")


def plot_chi2_residuals(res: dict, var_name: str, cluster_label: str,
                        outdir: Path, fname: str) -> Path:
    """Heatmap des résidus standardisés ajustés (Haberman) d'un croisement.

    Rouge = sur-représentation, bleu = sous-représentation ; l'annotation porte
    le résidu et une/deux étoiles quand |résidu| > 1.96 / 2.58 (seuils ~5 %/1 %).
    """
    adj = res["adj_residuals"]
    nr, nc = adj.shape
    fig, ax = plt.subplots(figsize=(max(4.0, 0.85 * nc + 2.2),
                                    max(2.8, 0.5 * nr + 1.6)))
    vmax = max(3.0, float(np.abs(adj.values).max()))
    im = ax.imshow(adj.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(nc)); ax.set_xticklabels(adj.columns, rotation=45, ha="right")
    ax.set_yticks(range(nr)); ax.set_yticklabels(adj.index)
    ax.set_xlabel(var_name)
    for i in range(nr):
        for j in range(nc):
            v = adj.values[i, j]
            star = "**" if abs(v) > 2.58 else "*" if abs(v) > 1.96 else ""
            ax.text(j, i, f"{v:.1f}{star}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > vmax * 0.6 else "#222")
    cv = res["cramers_v"]
    ax.set_title(f"Résidus ajustés — {cluster_label} × {var_name}\n"
                 f"{res['test']} · p = {res['pvalue']:.2g} · "
                 f"V de Cramér = {cv:.2f}" + ("" if res["conditions_met"]
                 else " · conditions khi² non remplies"), fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("résidu standardisé ajusté")
    fig.text(0.5, 0.005, "⚠ khi² : suppose 1 tumeur/patient — duplicats patient "
             "→ p-valeurs sous-estimées", ha="center", va="bottom",
             fontsize=7, color="#a12622")
    return _save(fig, outdir, fname, dpi=140)


def plot_correlation_heatmap(rho: pd.DataFrame, padj: pd.DataFrame, method: str,
                             outdir: Path, fname: str) -> Path:
    """Heatmap des corrélations clinique (lignes) × variables dérivées (colonnes).

    Rouge = corrélation positive, bleu = négative ; étoile si FDR significatif
    (* q<0.05, ** q<0.01). `rho`/`padj` : mêmes index/colonnes."""
    nr, nc = rho.shape
    fig, ax = plt.subplots(figsize=(max(5.0, 0.34 * nc + 3.0),
                                    max(2.4, 0.5 * nr + 1.4)))
    im = ax.imshow(rho.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(nc)); ax.set_xticklabels(rho.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(nr)); ax.set_yticklabels(rho.index, fontsize=8)
    for i in range(nr):
        for j in range(nc):
            q = padj.values[i, j]
            if np.isfinite(q) and q <= 0.05:
                ax.text(j, i, "**" if q <= 0.01 else "*", ha="center", va="center",
                        fontsize=7, color="#111")
    ax.set_title(f"Corrélations {method} — clinique × variables dérivées "
                 f"(★ FDR ≤ 0,05)", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85); cbar.set_label("ρ")
    return _save(fig, outdir, fname, dpi=140)
