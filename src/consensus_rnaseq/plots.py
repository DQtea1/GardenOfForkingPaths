"""Figures : heatmap consensus, CDF, PAC/delta-K, tracking plot, embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

from .consensus import ConsensusResult
from .metrics import consensus_cdf, delta_k, pac
from .stability import BranchStability

CONSENSUS_CMAP = LinearSegmentedColormap.from_list(
    "consensus", ["#ffffff", "#c6dbef", "#4292c6", "#08306b"]
)
CLUSTER_COLORS = plt.get_cmap("tab10").colors


def _save(fig: plt.Figure, outdir: Path, name: str, dpi: int = 200) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_consensus_heatmap(result: ConsensusResult, k: int, outdir: Path,
                           linkage_method: str = "average") -> Path:
    """Heatmap de la matrice consensus réordonnée par le dendrogramme,
    avec barre de clusters. La lecture visuelle est le premier critère :
    on veut des blocs nets, pas un dégradé continu."""
    C = result.consensus[k]
    D = result.distance(k)
    Z = linkage(squareform(D, checks=False), method=linkage_method)
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


def _sanitize(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


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


def plot_signature_boxplots(scores: pd.DataFrame, var: pd.Series,
                            signatures: list[str], var_name: str, method: str,
                            assoc: pd.DataFrame, outdir: Path) -> Path:
    """Grille de boxplots : pour une variable clinique catégorielle, la
    distribution du score (méthode `method`) de chaque top signature par
    modalité. Un fichier par (variable, méthode)."""
    var = var.reindex(scores.columns)
    levels = [m for m in pd.Series(var.dropna().unique())]
    palette = {m: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, m in enumerate(levels)}

    n = len(signatures)
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for idx, sig in enumerate(signatures):
        ax = axes.flat[idx]; ax.set_visible(True)
        data, labs, colors = [], [], []
        for m in levels:
            vals = scores.loc[sig, var.index[var == m]].dropna().values
            if len(vals):
                data.append(vals); labs.append(f"{m}\n(n={len(vals)})"); colors.append(palette[m])
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        tick_labels=labs, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black")
        rng = np.random.default_rng(0)
        for i, d in enumerate(data, start=1):
            ax.scatter(rng.normal(i, 0.06, len(d)), d, s=9, alpha=0.5, color="#333", zorder=3)
        best = assoc[(assoc["variable"] == var_name) & (assoc["signature"] == sig)]
        p = best["pvalue"].min() if len(best) else np.nan
        q = best["padj"].min() if len(best) else np.nan
        ax.set_title(f"{sig[:32]}\np={p:.1e}  q={q:.1e}", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylabel(f"score ({method})", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Signatures ~ {var_name}  ·  {method}  (top {n})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, outdir, f"sig_boxplots_{method}_{_sanitize(var_name)}.png")


def plot_signature_heatmap(scores: pd.DataFrame, var: pd.Series,
                           signatures: list[str], var_name: str, method: str,
                           outdir: Path) -> Path:
    """Heatmap top signatures × tumeurs (colonnes ordonnées et annotées par la
    variable clinique). Scores z-normalisés par signature. Un fichier par
    (variable, méthode)."""
    var = var.reindex(scores.columns)
    keep = var.dropna().index
    levels = list(pd.Series(var[keep].unique()))
    order = [s for m in levels for s in keep[var[keep] == m]]     # groupé par modalité
    palette = {m: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, m in enumerate(levels)}

    M = scores.loc[signatures, order]
    Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1).replace(0, 1), axis=0)  # z par ligne

    fig = plt.figure(figsize=(max(7, 0.045 * len(order) + 3), 0.4 * len(signatures) + 2.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.4, 10], width_ratios=[40, 1],
                          hspace=0.02, wspace=0.03)
    # bandeau de modalités
    ax_a = fig.add_subplot(gs[0, 0])
    ann = np.array([matplotlib.colors.to_rgb(palette[var[s]]) for s in order])[None]
    ax_a.imshow(ann, aspect="auto"); ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title(f"Activation des signatures par tumeur — {var_name} · {method}",
                   fontsize=11)
    # heatmap
    ax = fig.add_subplot(gs[1, 0])
    vmax = float(np.nanmax(np.abs(Z.values))) or 1.0
    im = ax.imshow(Z.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(signatures)))
    ax.set_yticklabels([s[:40] for s in signatures], fontsize=8)
    ax.set_xticks([]); ax.set_xlabel(f"{len(order)} tumeurs (groupées par {var_name})")
    fig.colorbar(im, cax=fig.add_subplot(gs[1, 1]), label="score (z par signature)")
    # légende modalités
    ax_a.legend(handles=[Patch(color=palette[m], label=str(m)) for m in levels],
                loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
                frameon=False, title=var_name)
    return _save(fig, outdir, f"sig_heatmap_{method}_{_sanitize(var_name)}.png")


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


def plot_cluster_overview(
    result: ConsensusResult, k: int, outdir: Path,
    linkage_method: str = "average",
    branch_stability: BranchStability | None = None,
    items: pd.DataFrame | None = None,
    color_by=None, color_label: str = "color_by",
    nes: pd.DataFrame | None = None,
    n_top_pathways: int = 5,
) -> Path:
    """Figure de synthèse combinant, autour d'un axe commun (l'ordre du
    dendrogramme) :

      - **haut** : arbre consensus, branches colorées par leur stabilité Jaccard ;
      - **centre** : heatmap de la matrice consensus réordonnée + barre de clusters ;
      - **gauche** : proportion de chaque modalité de `color_by` par cluster ;
      - **droite** : enrichissement GSEA (NES, one-vs-all) des top voies par cluster ;
      - **bas** : item consensus (stabilité des tumeurs) par cluster.

    Les panneaux latéraux sont alignés sur les blocs de clusters de la heatmap.
    """
    import matplotlib.colors as mcolors

    C = result.consensus[k]
    D = result.distance(k)
    Z = linkage(squareform(D, checks=False), method=linkage_method)
    order = leaves_list(Z)
    n = C.shape[0]
    labels_ord = result.labels(k, linkage_method)[order]
    Cord = C[np.ix_(order, order)]
    X = 10 * n

    # blocs de clusters contigus dans l'ordre des feuilles
    blocks = []
    i = 0
    while i < n:
        lab = int(labels_ord[i]); j = i
        while j < n and labels_ord[j] == lab:
            j += 1
        blocks.append((lab, i, j)); i = j
    block_labels = [b[0] for b in blocks]
    color_for = {lab: CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
                 for idx, lab in enumerate(block_labels)}

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.35, 0.16, 4.5, 1.7],
                          width_ratios=[1.75, 4.5, 2.5], hspace=0.09, wspace=0.08)

    # ---- arbre consensus (haut) ----
    ax_d = fig.add_subplot(gs[0, 1])
    norm_s = Normalize(0.0, 1.0); cmap_s = plt.get_cmap("RdYlGn")
    stab = (dict(zip(branch_stability.node_ids, branch_stability.stability))
            if branch_stability is not None else {})

    def _link_color(nid):
        s = stab.get(nid)
        return "#8a8a8a" if s is None else mcolors.to_hex(cmap_s(norm_s(s)))

    dendrogram(Z, ax=ax_d, no_labels=True,
               link_color_func=_link_color if branch_stability is not None
               else (lambda _: "#555555"))
    ax_d.set_xlim(0, X); ax_d.set_axis_off()

    # ---- barre d'annotation des clusters ----
    ax_a = fig.add_subplot(gs[1, 1])
    ann = np.zeros(n, int)
    for idx, (lab, a, b) in enumerate(blocks):
        ann[a:b] = idx
    ann_rgb = np.array([mcolors.to_rgb(color_for[block_labels[a]]) for a in ann])[None]
    ax_a.imshow(ann_rgb, aspect="auto", extent=[0, X, 0, 1])
    ax_a.set_xlim(0, X); ax_a.set_xticks([]); ax_a.set_yticks([])

    # ---- heatmap consensus (centre) ----
    ax_h = fig.add_subplot(gs[2, 1])
    im_h = ax_h.imshow(Cord, cmap=CONSENSUS_CMAP, vmin=0, vmax=1, aspect="auto",
                       extent=[0, X, X, 0], interpolation="nearest")
    ax_h.set_xlim(0, X); ax_h.set_ylim(X, 0)
    ax_h.set_xticks([]); ax_h.set_yticks([])

    # ---- boxplot item consensus (bas) ----
    ax_b = fig.add_subplot(gs[3, 1])
    positions = [10 * (a + b) / 2 for (_, a, b) in blocks]
    if items is not None:
        data = [items.loc[items["cluster"] == lab, "item_consensus"].dropna().values
                for lab in block_labels]
        widths = [max(0.8 * 10 * (b - a), 6) for (_, a, b) in blocks]
        bp = ax_b.boxplot(data, positions=positions, widths=widths, showfliers=False,
                          patch_artist=True, manage_ticks=False)
        for patch, lab in zip(bp["boxes"], block_labels):
            patch.set_facecolor(color_for[lab]); patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black")
        ax_b.axhline(0.8, ls="--", lw=0.8, color="grey")
        allv = np.concatenate([d for d in data if len(d)]) if any(len(d) for d in data) \
            else np.array([1.0])
        lo = min(0.75, float(np.nanmin(allv)) - 0.03)   # garde la ligne 0,8 visible
        ax_b.set_ylim(max(0.0, lo), 1.02)
    ax_b.set_xlim(0, X)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([f"C{lab}\n(n={b - a})" for (lab, a, b) in blocks], fontsize=8)
    ax_b.set_ylabel("item consensus")
    ax_b.spines[["top", "right"]].set_visible(False)

    # ---- proportions des modalités (gauche) ----
    ax_l = fig.add_subplot(gs[2, 0])
    mod_handles = []
    if color_by is not None:
        carr = np.array(["NA" if (v is None or (isinstance(v, float) and np.isnan(v)))
                         else str(v) for v in np.asarray(color_by, dtype=object)])
        modalities = sorted(set(carr))
        palette = plt.get_cmap("tab20").colors
        mod_color = {m: palette[i % len(palette)] for i, m in enumerate(modalities)}
        for (lab, a, b) in blocks:
            vals = carr[order[a:b]]
            center, height, left = 10 * (a + b) / 2, 0.85 * 10 * (b - a), 0.0
            for m in modalities:
                p = float(np.mean(vals == m))
                if p > 0:
                    ax_l.barh(center, p, height=height, left=left, color=mod_color[m],
                              edgecolor="white", linewidth=0.3)
                    left += p
        ax_l.set_xlim(0, 1); ax_l.set_ylim(X, 0); ax_l.set_yticks([])
        ax_l.set_xlabel("proportion")
        ax_l.spines[["top", "right", "left"]].set_visible(False)
        mod_handles = [Patch(color=mod_color[m], label=m) for m in modalities]
    else:
        ax_l.axis("off")

    # ---- enrichissement GSEA (droite) ----
    ax_r = fig.add_subplot(gs[2, 2])
    im_r = None
    if nes is not None and not nes.empty:
        sel = []
        for lab in block_labels:
            col = f"c{lab}"
            if col in nes.columns:
                for t in nes[col].abs().sort_values(ascending=False).index[:n_top_pathways]:
                    if t not in sel:
                        sel.append(t)
        y_edges = [10 * b[1] for b in blocks] + [X]
        Cr = np.array([[nes.loc[p, f"c{lab}"]
                        if (f"c{lab}" in nes.columns and p in nes.index) else np.nan
                        for p in sel] for lab in block_labels])
        vmax = float(np.nanmax(np.abs(Cr))) or 1.0
        im_r = ax_r.pcolormesh(np.arange(len(sel) + 1), y_edges, Cr,
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_r.set_ylim(X, 0); ax_r.set_yticks([])
        ax_r.set_xticks(np.arange(len(sel)) + 0.5)
        ax_r.set_xticklabels([p[:38] for p in sel], rotation=90, fontsize=7)
        ax_r.xaxis.set_ticks_position("bottom")
    else:
        ax_r.axis("off")
        ax_r.text(0.5, 0.5, "GSEA non calculé\n(run_degsea = n)", ha="center",
                  va="center", fontsize=9, color="grey")

    # ---- légendes & barres de couleur (une paire par coin, pour aérer) ----
    # haut-gauche : légende clusters + barre consensus
    ax_tl = fig.add_subplot(gs[0, 0]); ax_tl.axis("off")
    ax_tl.legend(handles=[Patch(color=color_for[lab], label=f"C{lab}")
                          for lab in block_labels], title="cluster",
                 loc="upper center", ncol=len(block_labels) if len(block_labels) <= 4 else 2,
                 fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.colorbar(im_h, cax=ax_tl.inset_axes([0.12, 0.12, 0.8, 0.09]),
                 orientation="horizontal", label="indice de consensus")

    # bas-gauche : légende des modalités + barre de stabilité
    ax_bl = fig.add_subplot(gs[3, 0]); ax_bl.axis("off")
    if mod_handles:
        ax_bl.legend(handles=mod_handles, title=color_label, loc="upper center",
                     ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.86))
    if branch_stability is not None:
        sm = plt.cm.ScalarMappable(norm=norm_s, cmap=cmap_s); sm.set_array([])
        fig.colorbar(sm, cax=ax_bl.inset_axes([0.12, 0.06, 0.8, 0.09]),
                     orientation="horizontal", label="stabilité de branche (Jaccard)")

    # haut-droite : titre + barre NES
    ax_tr = fig.add_subplot(gs[0, 2]); ax_tr.axis("off")
    ax_tr.text(0.5, 0.9, "Enrichissement GSEA (one-vs-all)", ha="center",
               fontsize=10, transform=ax_tr.transAxes)
    if im_r is not None:
        fig.colorbar(im_r, cax=ax_tr.inset_axes([0.15, 0.5, 0.7, 0.11]),
                     orientation="horizontal", label="NES")
    # bas-droite laissé libre : les noms de voies débordent du panneau de droite

    fig.suptitle(f"Synthèse du consensus clustering — k = {k}  ·  {n} tumeurs",
                 fontsize=15, y=0.999)
    return _save(fig, outdir, f"cluster_overview_k{k}.png")


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
