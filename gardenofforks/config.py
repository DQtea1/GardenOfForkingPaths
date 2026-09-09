"""Configuration du pipeline : une seule source de vérité pour la CLI et le YAML.

Historiquement, les valeurs venues du YAML étaient recopiées telles quelles dans
le ``Namespace`` : elles échappaient donc aux ``choices`` et aux ``type`` de
argparse, qui ne s'appliquent qu'aux chaînes de la ligne de commande. Une faute
de frappe (``metric: peason``, ``run_ica: maybe``) passait sans bruit et ne
cassait qu'au fond des workers joblib — après le prétraitement et le scan ICA —
voire pas du tout (``run_ica: maybe`` désactivait l'ICA en silence).

Ce module applique **les mêmes règles aux deux sources**, puis vérifie la
cohérence d'ensemble **avant** tout calcul :

    défauts argparse  <  fichier --config  <  ligne de commande

`load_config(argv)` renvoie le même ``argparse.Namespace`` qu'avant : le format
du YAML, le nom des clés et celui des options sont inchangés.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Texte d'aide de `gof-run` (ex-docstring de run_pipeline). Il vit ici, avec le
# parser qu'il décrit ; la docstring de run_pipeline documente l'orchestration.
_CLI_DESCRIPTION = r"""Pipeline complet : prétraitement -> consensus clustering -> diagnostics ->
embeddings t-SNE / UMAP -> figures et tables.

Ce module fait partie du paquet `gardenofforks` : il ne se lance pas par chemin
de fichier (`python gardenofforks/run_pipeline.py` casse les imports relatifs),
mais par la commande console ou la forme `-m` :

    gof-run …                        # après `pip install -e ".[full]"`
    python -m gardenofforks.run_pipeline …

Exemples
--------
# jeu de démonstration (500 tumeurs simulées, 4 sous-types)
gof-make-demo-data --outdir data && gof-run --counts data/demo_counts.tsv \
    --outdir results/demo --n-resamples 300

# données réelles, matrice VST déjà normalisée (gènes en lignes)
gof-run --counts data/vst.tsv --already-normalized \
    --k-max 10 --n-resamples 1000 --base hierarchical --metric pearson \
    --gene-mode bootstrap --outdir results/run01
"""


class ConfigError(ValueError):
    """Configuration invalide, détectée avant le moindre calcul."""


# Sentinelle de provenance : distingue « absent de la ligne de commande » de
# « passé explicitement, avec une valeur qui vaut le défaut ».
_UNSET = object()

# Blocs YAML structurés : ils n'ont pas d'option en ligne de commande et sont
# normalisés à part (cf. _extract_structured).
_STRUCTURED_KEYS = (
    "gsea_collections",
    "signature_sources",
    "deconv_methods",
    "deconv_reference",
    "ordinal_variables",
    "clinical_degsea",
    "subset_by",
)


def as_str_tuple(value, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Normalise une option « liste de chaînes » venue du YAML ou du terminal.

    Le YAML donne une liste, la ligne de commande une chaîne séparée par des
    virgules, et `null` doit reprendre le défaut. Sert aux motifs de filtrage
    comme aux chemins de fichiers de correspondance.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


def _add_deseq2_options(group, prefix: str) -> None:
    """Garde-fous numériques de l'ajustement DESeq2, pour un préfixe donné.

    Voir :class:`gardenofforks.degsea.DeseqSettings` pour le détail des deux
    pathologies visées : valeurs aberrantes (Cook) et dispersion effondrée.
    """
    add = group.add_argument
    add(f"--{prefix}_cooks_filter", choices=["y", "n"], default="y",
        help="'y' (défaut) : met padj à NA pour un gène dont un échantillon a une "
             "distance de Cook excessive, au lieu de le déclarer significatif. "
             "Équivalent du cooksCutoff de DESeq2 en R. 'n' désactive ce garde-fou.")
    add(f"--{prefix}_refit_cooks", choices=["y", "n"], default="y",
        help="'y' (défaut) : remplace les comptages aberrants (Cook) par la moyenne "
             "tronquée des autres échantillons, puis réajuste. Sans effet en dessous "
             "de 7 échantillons, seuil interne de PyDESeq2.")
    add(f"--{prefix}_independent_filter", choices=["y", "n"], default="y",
        help="'y' (défaut) : filtrage indépendant de DESeq2 sur l'expression moyenne, "
             "qui augmente la puissance après correction FDR.")
    add(f"--{prefix}_min_disp", type=float, default=1e-8,
        help="plancher de dispersion de PyDESeq2 (défaut 1e-8). Le relever (1e-3) "
             "atténue l'effondrement des erreurs-types, sans le supprimer.")
    add(f"--{prefix}_flag_low_dispersion", choices=["y", "n"], default="y",
        help="'y' (défaut) : marque d'une colonne `dispersion_suspecte` les gènes "
             "dont la dispersion ajustée est invraisemblablement basse — leur |z| "
             "est ininterprétable. Ils restent dans les tables mais sont écartés de "
             "l'étiquetage du volcano.")
    add(f"--{prefix}_min_plausible_disp", type=float, default=1e-3,
        help="seuil du marquage ci-dessus (défaut 1e-3). Une tumeur en bulk RNA-seq "
             "a une dispersion de 0,1 à 0,5 ; en dessous de 1e-3, l'estimation est "
             "douteuse plutôt que précise.")


def _add_degsea_filter_options(group, prefix: str, scope: str) -> None:
    """Déclare le jeu d'options `<prefix>_*` de filtrage des gènes d'un DEGSEA.

    Les deux DEGSEA — clinique (onglet « Expression Analysis ») et par
    cluster (onglet « Consensus-Clustering ») — partagent la même mécanique de
    filtrage mais des **seuils indépendants** : une seule définition ici, deux
    préfixes à l'appel. Côté exécution, le pendant est
    :meth:`gardenofforks.degsea.DegseaFilters.from_args`, qui relit ces mêmes
    options par préfixe.

    Rappel : un DEGSEA part des counts BRUTS (DESeq2 modélise des comptages),
    donc aucun filtre de l'étape 2 — min_cpm, keep_technical, n_top_genes — ne
    s'y applique ; ces options rejouent ces familles de filtres pour lui.
    """
    add = group.add_argument
    add(f"--{prefix}_filter_low_counts", choices=["y", "n"], default="n",
        help=f"'y' : {scope} — ne garde que les gènes atteignant "
             f"--{prefix}_min_count_per_sample counts bruts chez au moins "
             f"--{prefix}_min_frac_samples des tumeurs testées.")
    add(f"--{prefix}_min_count_per_sample", type=int, default=15,
        help="counts bruts minimum par tumeur pour le filtre ci-dessus (défaut 15).")
    add(f"--{prefix}_min_frac_samples", type=float, default=0.3,
        help="fraction minimale de tumeurs atteignant ce seuil (0.3 = 30 %%, défaut).")
    add(f"--{prefix}_filter_technical", choices=["y", "n"], default="n",
        help="'y' : retire les gènes techniques (ribosomiques, mitochondriaux, "
             "hémoglobines) avant DESeq2.")
    add(f"--{prefix}_technical_patterns", default=None,
        help="expressions régulières des gènes techniques (liste YAML, ou liste "
             "séparée par des virgules en ligne de commande). Défaut : ^RP[LS], "
             "^MT-, ^MRP[LS], ^HB[ABDEGMQZ]\\d?$.")
    add(f"--{prefix}_filter_unmapped", choices=["y", "n"], default="n",
        help="'y' : retire les identifiants qui ne sont pas des gènes annotés — "
             "clones génomiques (AC243964.4) et identifiants Ensembl non convertis.")
    add(f"--{prefix}_unmapped_patterns", default=None,
        help="expressions régulières des identifiants non annotés (liste YAML, ou "
             "séparée par des virgules). Défaut : ^[A-Z]{1,2}\\d{5,6}\\.\\d+$, ^ENSG\\d+.")
    add(f"--{prefix}_select_variable", choices=["y", "n"], default="n",
        help="'y' : ne teste que les N gènes les plus variables. ATTENTION : "
             "présélectionner sur la variance biaise l'estimation de la dispersion "
             "DESeq2 et invalide le filtrage indépendant, donc les FDR. Défaut 'n'.")
    add(f"--{prefix}_n_top_genes", type=int, default=5000,
        help="nombre de gènes conservés par le filtre de variance ci-dessus (défaut 5000).")
    add(f"--{prefix}_variance_method", choices=["mad", "var"], default="mad",
        help="mesure de variabilité pour ce filtre : mad (robuste, défaut) | var.")
    # Dernier de la série, et le seul à porter sur les TUMEURS : il juge chaque
    # tumeur sur le jeu de gènes finalement retenu.
    add(f"--{prefix}_filter_zero_samples", choices=["y", "n"], default="n",
        help="'y' : retire les tumeurs dont plus de "
             f"--{prefix}_max_zero_frac des gènes RESTANTS sont à 0. Appliqué en "
             "dernier. Une telle tumeur est ininterprétable et met en échec "
             "l'estimation 'poscounts' des size factors (médiane sur un ensemble "
             "vide -> NaN, qui contamine ensuite toutes les tumeurs).")
    add(f"--{prefix}_max_zero_frac", type=float, default=0.8,
        help="fraction maximale de gènes à 0 tolérée par tumeur (0.8 = 80 %%, défaut).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=_CLI_DESCRIPTION,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, type=Path,
                   help="fichier YAML de paramètres. Tout paramètre passé en "
                        "ligne de commande a la priorité sur le YAML.")
    io = p.add_argument_group("entrées / sorties")
    io.add_argument("--counts", default=None, help="matrice csv/tsv/parquet "
                    "(obligatoire, en ligne de commande ou dans le YAML)")
    io.add_argument("--samples-in-rows", action="store_true",
                    help="par défaut : gènes en lignes, échantillons en colonnes")
    io.add_argument("--metadata", default=None,
                    help="csv/tsv indexé par échantillon (annotations cliniques)")
    io.add_argument("--color-by", default=None,
                    help="colonne des métadonnées à superposer sur les embeddings")
    io.add_argument("--outdir", default="results/run", type=Path)
    io.add_argument("--filter_columns", default=None, metavar="COL[,COL…]",
                    help="RESTREINT les métadonnées cliniques à ces colonnes, pour "
                         "TOUT le run (liste YAML, ou noms séparés par des virgules). "
                         "La coupe a lieu dès le chargement : khi², corrélations, "
                         "associations, DEGSEA clinique, projection de signatures et "
                         "toutes les vues du rapport ne verront jamais les autres "
                         "colonnes. Vide = aucune restriction. Une colonne dont une "
                         "étape dépend (contraste, variable de design, group_DESeq2_by, "
                         "color_by) ne peut pas être exclue : le run s'arrête avec un "
                         "message plutôt que de changer un modèle en silence.")

    harm = p.add_argument_group("harmonisation des identifiants de gènes (étape 3)")
    harm.add_argument("--harmonize_gene_ids", choices=["y", "n"], default="n",
                      help="'y' : ramène tous les identifiants de gènes aux symboles "
                           "HGNC avant toute analyse. Nécessaire quand la matrice "
                           "fusionne des lots quantifiés dans des espaces différents "
                           "(Entrez, Ensembl, symboles) : la jointure les rend alors "
                           "disjoints, et tout filtre de prévalence élimine l'espace "
                           "minoritaire. Défaut 'n'.")
    harm.add_argument("--harmonize_hgnc_file", default=None,
                      help="jeu 'HGNC complete gene set' local (tsv de genenames.org). "
                           "Défaut : téléchargé une fois puis mis en cache sous "
                           "~/.cache/gardenofforks/. Renseigner un chemin fige la "
                           "version d'annotation et rend le run hors ligne.")
    harm.add_argument("--harmonize_map_aliases", choices=["y", "n"], default="y",
                      help="'y' (défaut) : ramène aussi les symboles périmés et les "
                           "alias à leur symbole approuvé (colonnes prev_symbol et "
                           "alias_symbol). Les alias ambigus sont toujours ignorés.")
    harm.add_argument("--harmonize_drop_unmapped", choices=["y", "n"], default="n",
                      help="'y' : retire les identifiants inconnus du jeu HGNC. "
                           "Défaut 'n' : ils sont conservés sous leur nom d'origine.")

    pre = p.add_argument_group("prétraitement")
    pre.add_argument("--already-normalized", action="store_true",
                     help="entrée déjà en VST/rlog/logCPM : saute filtrage + normalisation")
    pre.add_argument("--norm_method", choices=["vst", "logcpm"], default="vst",
                     help="normalisation des counts bruts (si --already-normalized "
                          "n'est pas mis) : 'vst' (DESeq2/PyDESeq2, défaut) ou 'logcpm'.")
    pre.add_argument("--min-cpm", type=float, default=1.0)
    pre.add_argument("--min-frac-samples", type=float, default=0.2)
    pre.add_argument("--keep-technical", action="store_true")
    pre.add_argument("--n-top-genes", type=int, default=5000)
    pre.add_argument("--variance-method", choices=["mad", "var"], default="mad")
    pre.add_argument("--scale-genes", action="store_true")
    pre.add_argument("--refit_after_sample_filters", choices=["y", "n"], default="y",
                     help="'y' (défaut) : une fois les tumeurs écartées par la "
                          "pureté PUREE et l'ACP, REJOUE le prétraitement sur la "
                          "seule cohorte conservée. Sans cela, le filtre de "
                          "prévalence, les size factors, la médiane de centrage et "
                          "surtout la sélection des n_top_genes les plus variables "
                          "restent décidés avec des tumeurs qui ne sont plus dans "
                          "l'analyse. Coût : un second VST, payé uniquement si des "
                          "tumeurs ont réellement été retirées. 'n' conserve le "
                          "panneau de gènes de la première passe.")
    pre.add_argument("--outlier_sd_threshold", type=float, default=0.0,
                     help="ACP sur la matrice prétraitée puis retrait des tumeurs "
                          "à plus de N écarts-types sur une composante principale. "
                          "0 ou absent = aucun retrait.")
    pre.add_argument("--outlier_n_pc", type=int, default=10,
                     help="nombre de composantes principales inspectées pour la "
                          "détection d'outliers (défaut 10).")
    pre.add_argument("--outlier_min_explained_var", type=float, default=0.0,
                     help="si > 0, inspecte plutôt toutes les composantes dont la "
                          "variance expliquée dépasse ce seuil (fraction ]0,1[ ; "
                          "8 = 8 %%). Prioritaire sur --outlier_n_pc.")

    ica_g = p.add_argument_group("ICA stabilisée (branche parallèle)")
    ica_g.add_argument("--run_ica", choices=["y", "n"], default="y",
                       help="'y' (défaut) : exécute la branche ICA stabilisée indépendante "
                            "après le prétraitement ; 'n' la désactive. Nécessite "
                            "`pip install stabilized-ica`.")
    ica_g.add_argument("--ica_n_components_min", type=int, default=6,
                       help="nombre minimal de composantes ICA à évaluer (MSTD).")
    ica_g.add_argument("--ica_n_components_max", type=int, default=8,
                       help="nombre maximal de composantes ICA à évaluer (MSTD).")
    ica_g.add_argument("--ica_n_components_step", type=int, default=1,
                       help="pas entre deux dimensions ICA testées (défaut 2).")
    ica_g.add_argument("--ica_n_runs", type=int, default=100,
                       help="nombre d'exécutions FastICA par dimension pour estimer la stabilité.")
    ica_g.add_argument("--ica_top_dimensions", type=int, default=4,
                       help="nombre maximal de branches ICA sauvegardées (1–4 ; défaut 4) : "
                            "MSTD, voisin inférieur, voisin supérieur, puis meilleure "
                            "stabilité moyenne.")
    ica_g.add_argument("--ica_algorithm", default="fastica_par",
                       choices=["fastica_par", "fastica_def", "picard_fastica", "picard",
                                "picard_ext", "picard_orth"],
                       help="solveur de stabilized-ica (défaut FastICA parallèle).")
    ica_g.add_argument("--ica_fun", default="logcosh", choices=["logcosh", "exp", "cube", "tanh"],
                       help="non-linéarité de l'ICA stabilisée (défaut logcosh).")
    ica_g.add_argument("--ica_resampling", default="none",
                       choices=["none", "bootstrap", "fast_bootstrap"],
                       help="rééchantillonnage interne stabilized-ica ; 'none' est le défaut.")
    ica_g.add_argument("--ica_max_iter", type=int, default=2000,
                       help="itérations maximales du solveur ICA (défaut 2000).")
    ica_g.add_argument("--ica_deterministic", choices=["y", "n"], default="y",
                       help="'y' force les fits ICA en série pour une graine reproductible ; "
                            "'n' autorise le parallélisme interne, moins déterministe.")
    ica_g.add_argument("--ica_k_final", type=int, default=None,
                       help="k imposé pour le consensus sur les projections ICA ; sinon sélection auto indépendante.")
    ica_g.add_argument("--run_ica_gsea", choices=["y", "n"], default="y",
                       help="'y' (défaut) : annote chaque métagène des dimensions "
                            "ICA conservées par GSEA pré-classé sur ses poids signés.")
    ica_g.add_argument("--ica_gsea_min_size", type=int, default=15,
                       help="taille minimale d'un gene set pour le GSEA des métagènes.")
    ica_g.add_argument("--ica_gsea_max_size", type=int, default=500,
                       help="taille maximale d'un gene set pour le GSEA des métagènes.")

    pur_g = p.add_argument_group("pureté tumorale (PUREE)")
    pur_g.add_argument("--purity_threshold", default="0",
                       help="seuil de pureté dans ]0,1[ pour filtrer les tumeurs. "
                            "0 / null / false = aucun filtrage (PUREE n'est pas lancé).")
    pur_g.add_argument("--purity_direction", choices=["higher", "lower"],
                       default="higher",
                       help="'higher' garde les puretés >= seuil (retire les "
                            "faibles puretés) ; 'lower' garde les puretés <= seuil.")
    pur_g.add_argument("--puree_dir", default="/home/quentin/02_MODELS/PUREE",
                       help="dossier du dépôt PUREE (predict_purity.py, models/, data/).")
    pur_g.add_argument("--puree_python",
                       default="/home/quentin/miniforge3/envs/PUREE/bin/python",
                       help="interpréteur Python de l'environnement PUREE.")
    pur_g.add_argument("--puree_gene_id", choices=["HGNC", "ENSEMBL"], default="HGNC",
                       help="type d'identifiant des gènes de la matrice d'entrée.")

    con = p.add_argument_group("consensus clustering")
    con.add_argument("--k-min", type=int, default=3)
    con.add_argument("--k-max", type=int, default=6)
    con.add_argument("--n-resamples", type=int, default=1000)
    con.add_argument("--prop-samples", type=float, default=0.8)
    con.add_argument("--prop-genes", type=float, default=0.8)
    con.add_argument("--sample-mode", choices=["subsample", "bootstrap"],
                     default="subsample")
    con.add_argument("--gene-mode", choices=["subsample", "bootstrap"],
                     default="subsample")
    con.add_argument("--base", choices=["hierarchical", "kmeans", "kmedoids"],
                     default="hierarchical")
    con.add_argument("--metric", choices=["pearson", "spearman", "euclidean", "cosine"],
                     default="pearson")
    con.add_argument("--linkage", default="average",
                     choices=["average", "complete", "ward", "single"])
    con.add_argument("--k-final", type=int, default=None,
                     help="k retenu ; par défaut choisi automatiquement (voir --k_criterion)")
    con.add_argument("--k_criterion", choices=["pac", "deltak", "both"], default="both",
                     help="critère de choix auto de k (si --k-final absent) : 'pac' "
                          "(minimise le PAC), 'deltak' (coude de Δ(K)), 'both' (défaut).")
    con.add_argument("--min-cluster-size", type=int, default=10)

    stab = p.add_argument_group("stabilité des branches (Jaccard bootstrap)")
    stab.add_argument("--compute_jaccard", choices=["y", "n"], default="y",
                      help="'y' : après la partition finale, calcule la stabilité "
                           "Jaccard de chaque branche de l'arbre consensus par "
                           "bootstrap des gènes (n_resamples arbres). Défaut 'y'.")

    deg = p.add_argument_group("DEGSEA (DESeq2 + GSEA par cluster)")
    deg.add_argument("--run_degsea", choices=["y", "n"], default="n",
                     help="'y' : après les embeddings, DESeq2 + GSEA par cluster "
                          "(one-vs-all et one-vs-one). Étape longue. Défaut 'n'.")
    deg.add_argument("--degsea_mode", choices=["ova", "ovo", "both"], default="both",
                     help="contrastes DESeq2 : ova (one-vs-all), ovo (one-vs-one, "
                          "coûteux : k(k-1)/2), both (défaut).")
    deg.add_argument("--degsea_all_k", choices=["y", "n"], default="n",
                     help="'y' : calcule le DEGSEA pour TOUS les k de la plage "
                          "[k_min..k_max] (une sous-arborescence tables/degsea/k<k>/ "
                          "par k, et un panneau DEGSEA aligné sur n'importe quel k "
                          "dans le rapport). Très coûteux. Défaut 'n' : uniquement "
                          "le k recommandé par le critère combiné PAC+Delta(K).")
    # Filtrage des gènes en entrée du DEGSEA par cluster : mêmes filtres que le
    # DEGSEA clinique, seuils indépendants (cf. _add_degsea_filter_options).
    _add_degsea_filter_options(deg, "degsea", "DEGSEA par cluster")
    _add_deseq2_options(deg, "degsea")
    deg.add_argument("--gsea_gene_sets",
                     default=str(Path.home() / ".cache/gseapy/Enrichr.MSigDB_Hallmark_2020.gmt"),
                     help="fichier .gmt de gene sets pour le GSEA (hallmarks MSigDB par défaut).")
    deg.add_argument("--gsea_permutations", type=int, default=1000,
                     help="nombre de permutations du GSEA pré-classé (défaut 1000).")
    deg.add_argument("--gsea_heatmap_pval", type=float, default=0.05,
                     help="seuil de **FDR q-valeur GSEA** (permutations) pour inclure "
                          "un pathway dans la heatmap one-vs-all : tous ceux "
                          "significatifs après correction dans >= 1 cluster (défaut 0.05). "
                          "Convention GSEA usuelle : 0.25.")

    clinical_deg = p.add_argument_group("DEGSEA clinique (DESeq2 ajusté + GSEA)")
    clinical_deg.add_argument("--run_clinical_degsea", choices=["y", "n"], default=None,
                             help="interrupteur du DEGSEA clinique (étape 7). NON "
                                  "RENSEIGNÉ (défaut) : l'étape tourne si — et "
                                  "seulement si — le YAML contient un bloc "
                                  "clinical_degsea. 'n' la désactive même quand ce "
                                  "bloc existe, ce qui permet de garder les designs "
                                  "sous la main sans les rejouer ; 'y' est explicite "
                                  "et prévient si aucune expérience n'est configurée.")
    # Filtrage des gènes en entrée du DEGSEA clinique : mêmes filtres que le
    # DEGSEA par cluster, seuils indépendants (cf. _add_degsea_filter_options).
    clinical_deg.add_argument("--clinical_degsea_drop_pca_outliers",
                             choices=["y", "n"], default="n",
                             help="'y' : restreint le DEGSEA clinique aux tumeurs "
                                  "conservées par les filtres des étapes 4 et 5 (pureté "
                                  "PUREE + outliers ACP). Défaut 'n' : le DEGSEA "
                                  "clinique porte sur TOUTES les tumeurs de la "
                                  "matrice, y compris les outliers ACP. Sans objet "
                                  "pour le DEGSEA par cluster, qui part déjà de la "
                                  "partition, donc des tumeurs filtrées.")
    _add_degsea_filter_options(clinical_deg, "clinical_degsea", "DEGSEA clinique")
    _add_deseq2_options(clinical_deg, "clinical_degsea")
    clinical_deg.add_argument("--group_DESeq2_by", default=None, metavar="COL[,COL…]",
                             help="colonne(s) des métadonnées servant à STRATIFIER le "
                                  "DEGSEA clinique : pour chaque modalité de chaque "
                                  "colonne, toute la batterie de designs clinical_degsea "
                                  "est rejouée sur les seules tumeurs de cette modalité "
                                  "(liste YAML, ou noms séparés par des virgules). "
                                  "Les valeurs 'ALL', 'None' et 'NA' désignent l'analyse "
                                  "NON filtrée, sur toute la cohorte ; c'est le défaut. "
                                  "ATTENTION au coût : le nombre d'ajustements DESeq2 "
                                  "vaut n_designs x (1 + somme des modalités).")

    out_g = p.add_argument_group("OUTRIDER — expression aberrante (étape 7b)")
    out_g.add_argument("--run_outrider", choices=["y", "n"], default="n",
                       help="'y' : cherche les gènes anormalement exprimés CHEZ UNE "
                            "TUMEUR (py_outrider, autoencodeur + binomiale négative), "
                            "indépendamment du consensus clustering. Le découpage de "
                            "la cohorte est déclaré par le bloc YAML `subset_by` : un "
                            "run par combinaison de modalités. Étape longue. Défaut 'n'.")
    out_g.add_argument("--outrider_python", default=None,
                       help="interpréteur de l'environnement où py_outrider est "
                            "installé (TensorFlow). Défaut : l'interpréteur courant, "
                            "ce qui suppose `pip install py_outrider` dans le même "
                            "environnement que le pipeline.")
    out_g.add_argument("--outrider_profile", default="outrider",
                       choices=["outrider", "protrider", "pca"],
                       help="modèle py_outrider : 'outrider' (counts RNA-seq, "
                            "binomiale négative, défaut), 'protrider' (intensités "
                            "protéiques, gaussienne + covariables), 'pca'.")
    out_g.add_argument("--outrider_encod_dim", type=int, default=None,
                       help="dimension du goulot de l'autoencodeur. NON RENSEIGNÉE "
                            "(défaut) : py_outrider cherche l'optimum, ce qui est "
                            "correct mais coûte plusieurs ajustements complets PAR "
                            "GROUPE. Règle du pouce pour figer : n_tumeurs / 4.")
    out_g.add_argument("--outrider_iterations", type=int, default=None,
                       help="itérations maximales d'ajustement (défaut du profil).")
    out_g.add_argument("--outrider_min_samples", type=int, default=30,
                       help="taille minimale d'un sous-groupe pour être analysé "
                            "(défaut 30). Sous ce seuil l'autoencodeur n'a pas de "
                            "quoi apprendre une norme : le groupe est écarté AVANT "
                            "calcul et le motif est tracé dans plan.csv.")
    out_g.add_argument("--outrider_min_count", type=int, default=10,
                       help="counts bruts minimum par tumeur pour qu'un gène soit "
                            "testé (défaut 10), jugé DANS le sous-groupe.")
    out_g.add_argument("--outrider_min_frac_samples", type=float, default=0.25,
                       help="fraction des tumeurs du sous-groupe devant atteindre ce "
                            "seuil (0.25 = 25 %%, défaut).")
    out_g.add_argument("--outrider_min_genes", type=int, default=100,
                       help="nombre minimal de gènes survivant au filtrage pour "
                            "lancer un run (défaut 100).")
    out_g.add_argument("--outrider_alpha", type=float, default=0.05,
                       help="seuil de FDR au-delà duquel un couple (tumeur, gène) "
                            "est déclaré aberrant (défaut 0.05).")
    out_g.add_argument("--outrider_max_events", type=int, default=5000,
                       help="nombre maximal d'événements aberrants embarqués par run "
                            "dans le rapport, les plus significatifs d'abord (défaut "
                            "5000). Les tables complètes restent sur disque.")
    out_g.add_argument("--outrider_figures", choices=["y", "n"], default="y",
                       help="'y' (défaut) : embarque dans le rapport de quoi rejouer "
                            "les figures d'OUTRIDER (volcano par tumeur, rang "
                            "d'expression, QQ, heatmaps, dispersions…). 'n' ne garde "
                            "que les tables.")
    out_g.add_argument("--outrider_figure_cells", type=int, default=20000,
                       help="budget, en CELLULES (gènes × tumeurs), des matrices "
                            "embarquées pour les figures (défaut 20 000, soit environ "
                            "1 Mo de rapport PAR RUN — sept matrices y sont stockées). "
                            "Le panneau de gènes rétrécit donc automatiquement quand la "
                            "cohorte grandit : c'est ce qui borne le poids du rapport, "
                            "au prix d'un nuage de volcano plus clairsemé.")
    out_g.add_argument("--outrider_heatmap_genes", type=int, default=100,
                       help="gènes les plus variables retenus pour la heatmap "
                            "gènes × tumeurs (défaut 100 ; R en prend 500).")
    out_g.add_argument("--outrider_covariates", default=None, metavar="COL[,COL…]",
                       help="colonnes cliniques passées à py_outrider comme "
                            "covariables connues (profil protrider surtout).")
    out_g.add_argument("--outrider_extra_args", default=None,
                       help="options supplémentaires passées telles quelles à "
                            "py_outrider (ex. '--distribution NB --convergence 1e-4').")
    out_g.add_argument("--outrider_threads", type=int, default=0,
                       help="threads TensorFlow PAR run OUTRIDER. 0 (défaut) : ~4, "
                            "valeur où le gain sature sur un autoencodeur de cette "
                            "taille (mesuré : 1 thread 168 s, 4 -> 78 s, 8 -> 70 s, "
                            "16 -> 63 s sur 150 tumeurs × 4000 gènes). Le nombre de "
                            "runs menés de front s'en déduit (n_jobs / threads) : "
                            "2 -> 8 runs de front sur 16 cœurs, 8 -> 2 runs. Moins de "
                            "threads = plus de runs simultanés = plus de mémoire.")
    out_g.add_argument("--outrider_jobs", type=int, default=0,
                       help="sous-groupes OUTRIDER menés DE FRONT. 0 (défaut) : "
                            "choisi automatiquement, ~4 threads TensorFlow par run "
                            "— au-delà, un autoencodeur de cette taille ne "
                            "s'accélère plus, alors que les sous-groupes sont "
                            "indépendants. 1 = séquentiel. Le produit "
                            "(runs × threads) reste borné par n_jobs. ATTENTION : "
                            "chaque run de front, c'est un TensorFlow de plus en "
                            "mémoire.")
    out_g.add_argument("--outrider_keep_h5ad", choices=["y", "n"], default="y",
                       help="'y' (défaut) : conserve l'AnnData complet de chaque run "
                            "(toutes les matrices : p-valeurs, z-scores, prédictions). "
                            "C'est lui qui portera les figures.")

    sig = p.add_argument_group("projection de signatures (scoring + association clinique)")
    sig.add_argument("--compute_signatures", choices=["y", "n"], default="n",
                     help="'y' : après DEGSEA, score les signatures par tumeur "
                          "(ssGSEA + expression moyenne) et teste leur association "
                          "aux variables cliniques. Défaut 'n'.")
    sig.add_argument("--signatures_gmt", default=None,
                     help="fichier .gmt des signatures à scorer. Défaut : la "
                          "collection load_signatures_select du YAML, sinon "
                          "--gsea_gene_sets.")
    sig.add_argument("--sig_corr_method", choices=["spearman", "pearson"],
                     default="spearman",
                     help="corrélation score↔variable continue (défaut spearman).")
    sig.add_argument("--sig_top_n", type=int, default=8,
                     help="nombre de top signatures affichées par variable (défaut 8).")
    sig.add_argument("--sig_pval", type=float, default=0.05,
                     help="seuil de FDR pour retenir une signature comme "
                          "significativement associée (défaut 0.05).")
    sig.add_argument("--sig_max_text_levels", type=int, default=20,
                     help="nombre maximum de modalités d'une variable clinique "
                          "TEXTE encore testée contre les signatures (défaut 20). "
                          "Au-delà, la colonne est traitée comme un identifiant "
                          "(centre, lot, date) et écartée : elle serait sinon "
                          "testée par paire de modalités, ce qui fait exploser le "
                          "nombre de tests et dilue la correction FDR.")

    dec = p.add_argument_group("déconvolution (omnideconv / immunedeconv)")
    dec.add_argument("--run_deconv", choices=["y", "n"], default="n",
                     help="'y' : batterie de déconvolution (MCPcounter, xCell, "
                          "quanTIseq, EPIC, et DWLS/BayesPrism si référence "
                          "single-cell). Étape longue. Défaut 'n'. Méthodes et "
                          "paramètres : bloc deconv_methods du YAML ; référence : "
                          "bloc deconv_reference.")
    dec.add_argument("--deconv_rscript", default="Rscript",
                     help="interpréteur Rscript (omnideconv + immunedeconv installés).")

    chi = p.add_argument_group("association catégorielle (khi² d'indépendance)")
    chi.add_argument("--run_chi2", choices=["y", "n"], default="y",
                     help="'y' (défaut) : croise cluster (chaque k) × variables "
                          "cliniques catégorielles et clinique × clinique — khi² "
                          "d'indépendance (Fisher/Monte-Carlo en repli), V de Cramér "
                          "et résidus standardisés ajustés. Sauté sans métadonnées "
                          "catégorielles. Étape légère.")
    chi.add_argument("--chi2_mc_resamples", type=int, default=2000,
                     help="permutations du khi² de Monte-Carlo (repli des tables R×C "
                          "aux conditions de Cochran non remplies ; défaut 2000).")

    cor = p.add_argument_group("corrélations continues (étape 17)")
    cor.add_argument("--run_correlations", choices=["y", "n"], default="y",
                     help="'y' (défaut) : corrèle deux à deux les variables CONTINUES "
                          "par patient — clinique continue × signatures/déconvolution "
                          "et signatures × déconvolution (Spearman + FDR). Léger. "
                          "Sauté sans variable continue exploitable.")
    cor.add_argument("--corr_method", choices=["spearman", "pearson"], default="spearman",
                     help="méthode de corrélation (défaut spearman, robuste).")
    cor.add_argument("--corr_all_pairs", choices=["y", "n"], default="n",
                     help="'y' : calcule AUSSI signature×signature et déconv×déconv "
                          "(redondant / compositionnel — à interpréter avec prudence). "
                          "Défaut 'n'.")

    rep = p.add_argument_group("rapport d'analyse (HTML interactif)")
    rep.add_argument("--create_report", choices=["y", "n"], default="y",
                     help="'y' (défaut) : génère outdir/report.html — rapport "
                          "interactif autonome de tous les résultats.")

    embg = p.add_argument_group("embeddings")
    embg.add_argument("--t-SNE_dim", dest="tsne_dim", type=int, choices=[2, 3],
                      default=2,
                      help="dimensions des embeddings t-SNE / UMAP : 2 (PNG "
                           "statiques, défaut) ou 3 (HTML interactif rotatable, "
                           "survol = ID de la tumeur ; nécessite plotly)")
    embg.add_argument("--embedding_distances", default="consensus",
                      help="espaces de distance sur lesquels calculer t-SNE / UMAP, "
                           "séparés par des virgules (liste en YAML). 'consensus' "
                           "(défaut) = D_K = 1 - C_K, recalculé pour chaque k ; "
                           "'euclidean', 'correlation' et 'manhattan' sont calculées "
                           "directement sur la matrice d'entrée de la branche (scores "
                           "des composantes pour l'ICA) et ne dépendent pas de k. Le "
                           "rapport propose un menu pour passer de l'une à l'autre.")
    embg.add_argument("--perplexity", type=float, default=30.0)
    embg.add_argument("--n-neighbors", type=int, default=15)
    embg.add_argument("--min-dist", type=float, default=0.1)
    embg.add_argument("--no-umap", action="store_true")

    p.add_argument("--parallel", choices=["y", "n"], default="y",
                   help="'y' (défaut) : parallélise le rééchantillonnage consensus, "
                        "la stabilité Jaccard, DEGSEA et les embeddings sur --n-jobs "
                        "cœurs. 'n' : force tout en séquentiel (n_jobs=1), utile pour "
                        "déboguer ou sur une machine partagée.")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_file", default="run.log",
                   help="fichier journal horodaté (relatif à outdir si non absolu ; "
                        "défaut run.log). Toute la progression y est écrite.")
    return p



# --------------------------------------------------------------------------
# Introspection du parser : les règles de validation viennent du parser
# lui-même, pour qu'il n'existe jamais deux listes de `choices` à synchroniser.
# --------------------------------------------------------------------------
def _field_rules(parser: argparse.ArgumentParser) -> dict[str, dict]:
    """Extrait `{dest: {"choices", "type"}}` des options déclarées.

    argparse n'expose pas d'API publique d'introspection ; on lit `_actions`
    **sans le modifier** (l'ancien code y écrasait les défauts, ce qui corrompait
    le parser pour tout appel ultérieur).
    """
    rules: dict[str, dict] = {}
    for action in parser._actions:
        if action.dest in (argparse.SUPPRESS, "help"):
            continue
        rules[action.dest] = {
            "choices": tuple(action.choices) if action.choices else None,
            "type": action.type,
            "is_flag": isinstance(
                action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
            ),
        }
    return rules


def _coerce(key: str, value, rule: dict):
    """Applique à une valeur YAML le `type` et les `choices` de son option."""
    if value is None:
        return None

    choices = rule.get("choices")

    # PyYAML lit `yes` / `no` / `true` / `false` comme des booléens : pour les
    # options y/n, on les retraduit au lieu de rejeter une écriture naturelle.
    if isinstance(value, bool):
        if choices and set(choices) == {"y", "n"}:
            return "y" if value else "n"
        if rule.get("is_flag"):
            return bool(value)

    caster = rule.get("type")
    if caster is not None and not isinstance(value, (dict, list, tuple)):
        try:
            value = caster(value)
        except (TypeError, ValueError) as exc:
            name = getattr(caster, "__name__", str(caster))
            raise ConfigError(
                f"{key} : valeur invalide {value!r} (attendu : {name})."
            ) from exc

    if choices and value not in choices:
        options = ", ".join(map(str, choices))
        suggestion = difflib.get_close_matches(str(value), list(map(str, choices)),
                                               n=1, cutoff=0.6)
        hint = f" — voulez-vous dire {suggestion[0]!r} ?" if suggestion else ""
        raise ConfigError(
            f"{key} : {value!r} n'est pas une valeur acceptée{hint} "
            f"(attendu : {options})."
        )
    return value


# --------------------------------------------------------------------------
# Blocs YAML structurés
# --------------------------------------------------------------------------
def _normalize_collections(spec) -> dict[str, str]:
    """`gsea_collections: {nom: chemin.gmt}` ou `{nom: {enabled, path}}`."""
    out: dict[str, str] = {}
    for name, value in (spec or {}).items():
        if isinstance(value, str):
            path, enabled = value, True
        elif isinstance(value, dict):
            path, enabled = value.get("path"), value.get("enabled", True)
        else:
            continue
        if enabled and path:
            out[str(name)] = str(Path(path).expanduser())
    return out


def _extract_structured(config: dict, consumed: set[str]) -> dict:
    """Normalise les blocs imbriqués et note les clés consommées."""
    out = {key: {} for key in _STRUCTURED_KEYS}

    if isinstance(config.get("gsea_collections"), dict):
        out["gsea_collections"] = _normalize_collections(config["gsea_collections"])
        consumed.add("gsea_collections")

    # Forme héritée : des clés plates `load_<nom>: chemin.gmt`.
    for key in sorted(k for k in config if k.startswith("load_")):
        value = config[key]
        if isinstance(value, str) and value.strip().lower().endswith(".gmt"):
            out["gsea_collections"][key[len("load_"):]] = str(Path(value).expanduser())
            consumed.add(key)

    for key in ("signature_sources", "deconv_methods", "deconv_reference"):
        if isinstance(config.get(key), dict):
            out[key] = config[key]
            consumed.add(key)

    # Variables ORDINALES (test de tendance, étape 16). Deux écritures :
    #   {stade: [I, II, III]}  ordre explicite (recommandé)
    #   [stade, grade]         ordre = tri des modalités
    if "ordinal_variables" in config:
        value = config["ordinal_variables"]
        if isinstance(value, dict):
            out["ordinal_variables"] = {
                str(k): (list(v) if v else None) for k, v in value.items()
            }
        elif isinstance(value, (list, tuple)):
            out["ordinal_variables"] = {str(k): None for k in value}
        consumed.add("ordinal_variables")

    if "clinical_degsea" in config:
        value = config["clinical_degsea"]
        if not isinstance(value, dict):
            raise ConfigError("clinical_degsea doit être un dictionnaire d'expériences.")
        out["clinical_degsea"] = value
        consumed.add("clinical_degsea")

    # Découpages OUTRIDER (7b) : {nom: [colonnes]}. Une liste vide — ou `null` —
    # désigne la cohorte entière, un seul run sans stratification.
    if "subset_by" in config:
        value = config["subset_by"]
        if isinstance(value, dict):
            out["subset_by"] = {
                str(k): ([] if v is None else
                         [v] if isinstance(v, str) else list(v))
                for k, v in value.items()
            }
        elif isinstance(value, (list, tuple)):      # forme courte : [col1, col2]
            out["subset_by"] = {str(v): [str(v)] for v in value}
        elif value is None:
            out["subset_by"] = {}
        else:
            raise ConfigError(
                "subset_by doit être un dictionnaire {nom: [colonnes]} (ou une "
                "liste de colonnes, chacune devenant son propre découpage).")
        consumed.add("subset_by")

    return out


def _check_unknown_keys(config: dict, known: set[str], consumed: set[str]) -> list[str]:
    """Signale les clés YAML non gérées ; une faute de frappe est bloquante.

    Une clé qui ressemble fortement à une option connue (`k_maxx` -> `k_max`)
    est une erreur : l'ignorer en silence donnerait une analyse menée avec le
    défaut, sans le moindre signal. Une clé sans voisin plausible reste un
    simple avertissement — le YAML sert aussi de brouillon pour des options à
    venir.
    """
    unknown = sorted(set(config) - known - consumed - {"config"})
    if not unknown:
        return []
    typos, ignored = [], []
    for key in unknown:
        match = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.8)
        (typos if match else ignored).append((key, match[0] if match else None))
    if ignored:
        logger.warning(
            "[config] clés ignorées (non gérées par le pipeline) : %s",
            ", ".join(key for key, _ in ignored),
        )
    return [
        f"{key} : clé inconnue — voulez-vous dire {near!r} ?" for key, near in typos
    ]


# --------------------------------------------------------------------------
# Cohérence d'ensemble — tout ce qui se vérifie sans lire les données
# --------------------------------------------------------------------------
def _check_ranges(cfg: dict, errors: list[str]) -> None:
    def bounded(key, low=None, high=None, *, low_open=False, high_open=False):
        value = cfg.get(key)
        if value is None:
            return
        if low is not None and (value <= low if low_open else value < low):
            errors.append(f"{key} = {value} : attendu {'>' if low_open else '>='} {low}.")
        if high is not None and (value >= high if high_open else value > high):
            errors.append(f"{key} = {value} : attendu {'<' if high_open else '<='} {high}.")

    bounded("k_min", 2)
    bounded("n_resamples", 1)
    bounded("min_cluster_size", 1)
    bounded("prop_samples", 0, 1, low_open=True)
    bounded("prop_genes", 0, 1, low_open=True)
    bounded("ica_n_components_min", 2)
    bounded("ica_n_components_step", 1)
    bounded("ica_n_runs", 2)
    bounded("ica_top_dimensions", 1, 4)
    bounded("ica_max_iter", 1)
    bounded("ica_gsea_min_size", 1)
    bounded("outlier_sd_threshold", 0)
    bounded("outlier_n_pc", 1)
    bounded("outlier_min_explained_var", 0, 1, high_open=True)
    bounded("perplexity", 0, low_open=True)
    bounded("n_neighbors", 2)
    bounded("min_dist", 0)
    bounded("gsea_permutations", 1)
    bounded("gsea_heatmap_pval", 0, 1, low_open=True)
    bounded("sig_pval", 0, 1, low_open=True)
    bounded("sig_top_n", 1)
    bounded("chi2_mc_resamples", 1)
    bounded("outrider_min_samples", 2)
    bounded("outrider_min_count", 0)
    bounded("outrider_min_frac_samples", 0, 1)
    bounded("outrider_min_genes", 1)
    bounded("outrider_alpha", 0, 1, low_open=True)
    bounded("outrider_max_events", 1)
    bounded("outrider_encod_dim", 1)
    bounded("outrider_iterations", 1)
    bounded("outrider_jobs", 0)
    bounded("outrider_threads", 0)

    k_min, k_max = cfg.get("k_min"), cfg.get("k_max")
    if k_min is not None and k_max is not None and k_max < k_min:
        errors.append(
            f"k_max = {k_max} < k_min = {k_min} : la plage de k serait vide."
        )
    for key in ("k_final", "ica_k_final"):
        value = cfg.get(key)
        if value is not None and k_min is not None and k_max is not None:
            if not k_min <= value <= k_max:
                errors.append(
                    f"{key} = {value} : hors de la plage k_min..k_max "
                    f"({k_min}..{k_max})."
                )

    ica_min, ica_max = cfg.get("ica_n_components_min"), cfg.get("ica_n_components_max")
    if ica_min is not None and ica_max is not None and ica_max < ica_min:
        errors.append(
            f"ica_n_components_max = {ica_max} < ica_n_components_min = {ica_min}."
        )
    lo, hi = cfg.get("ica_gsea_min_size"), cfg.get("ica_gsea_max_size")
    if lo is not None and hi is not None and hi < lo:
        errors.append(f"ica_gsea_max_size = {hi} < ica_gsea_min_size = {lo}.")

    if cfg.get("n_jobs") == 0:
        errors.append("n_jobs = 0 : utiliser -1 (tous les cœurs) ou un entier > 0.")

    try:
        from .purity import parse_threshold

        parse_threshold(cfg.get("purity_threshold"))
    except ValueError as exc:
        errors.append(str(exc))


def _check_inputs(cfg: dict, errors: list[str]) -> None:
    """Existence des fichiers d'entrée — la panne la plus fréquente."""
    counts = cfg.get("counts")
    if not counts:
        errors.append("--counts est obligatoire (en ligne de commande ou dans le YAML).")
    elif not Path(counts).expanduser().exists():
        errors.append(f"counts : fichier introuvable — {counts}")

    metadata = cfg.get("metadata")
    if metadata and not Path(metadata).expanduser().exists():
        errors.append(f"metadata : fichier introuvable — {metadata}")

    # Les GMT manquants sont filtrés plus tard par degsea.resolve_gene_sets :
    # on prévient sans bloquer, un run peut viser une machine où ils existent.
    for name, path in (cfg.get("gsea_collections") or {}).items():
        if not Path(path).expanduser().exists():
            logger.warning("[config] collection GSEA introuvable, ignorée : %s (%s)",
                           name, path)


def _check_steps(cfg: dict, errors: list[str]) -> None:
    """Combinaisons d'étapes impossibles, détectées avant le premier calcul."""
    from .embedding import DISTANCE_CHOICES

    unknown = [
        name for name in as_str_tuple(cfg.get("embedding_distances"))
        if name.strip().lower() not in DISTANCE_CHOICES
    ]
    if unknown:
        errors.append(
            "embedding_distances : valeur(s) inconnue(s) "
            f"{', '.join(unknown)} — attendu {', '.join(DISTANCE_CHOICES)}."
        )
    if cfg.get("already_normalized") and clinical_experiments(cfg.get("clinical_degsea")):
        errors.append(
            "clinical_degsea requiert des counts BRUTS : incompatible avec "
            "--already-normalized."
        )
    if cfg.get("already_normalized") and cfg.get("run_degsea") == "y":
        errors.append(
            "run_degsea requiert des counts BRUTS : incompatible avec "
            "--already-normalized."
        )
    if cfg.get("already_normalized") and cfg.get("run_outrider") == "y":
        errors.append(
            "run_outrider requiert des counts BRUTS (OUTRIDER modélise une "
            "binomiale négative) : incompatible avec --already-normalized."
        )
    try:
        specs = subset_specs(cfg.get("subset_by"))
    except ConfigError as exc:
        errors.append(str(exc))
    else:
        if cfg.get("run_outrider") == "y" and not specs:
            logger.warning(
                "[config] run_outrider = y sans bloc subset_by : OUTRIDER sera "
                "lancé une seule fois, sur toute la cohorte. Sur une cohorte "
                "hétérogène, le modèle apprend la moyenne des populations et "
                "chaque programme spécifique devient une « aberration ».")
    if cfg.get("tsne_dim") == 3:
        try:
            import plotly  # noqa: F401
        except ImportError:
            errors.append(
                "t-SNE_dim = 3 exige plotly : `pip install -e \".[plotly]\"`."
            )


def validate(cfg: dict, *, extra: list[str] | None = None) -> None:
    """Vérifie la configuration fusionnée ; lève une `ConfigError` groupée.

    Toutes les fautes sont rapportées d'un coup : corriger un fichier de
    configuration ne doit pas demander autant d'allers-retours qu'il contient
    d'erreurs. ``extra`` reçoit celles déjà relevées champ par champ.
    """
    errors: list[str] = list(extra or [])
    _check_ranges(cfg, errors)
    _check_inputs(cfg, errors)
    _check_steps(cfg, errors)
    if errors:
        raise ConfigError(
            "configuration invalide :\n  - " + "\n  - ".join(errors)
        )


# --------------------------------------------------------------------------
# Collections GMT : un seul point de résolution (auparavant réécrit 4 fois,
# avec des variantes — `expanduser` tantôt appliqué, tantôt oublié).
# --------------------------------------------------------------------------
def collections_or_fallback(collections, gsea_gene_sets) -> dict[str, str]:
    """Collections déclarées, sinon repli sur l'unique `--gsea_gene_sets`."""
    resolved = {
        str(name): str(Path(path).expanduser())
        for name, path in (collections or {}).items()
    }
    if resolved:
        return resolved
    if gsea_gene_sets:
        path = Path(gsea_gene_sets).expanduser()
        return {path.stem: str(path)}
    return {}


def enabled(value, default: bool = True) -> bool:
    """Interprète les booléens YAML, y compris les formes ``y`` / ``n``."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"y", "yes", "true", "1", "on"}


def clinical_experiments(config: dict | None) -> dict[str, dict]:
    """Normalise le dictionnaire YAML des expériences DEGSEA cliniques."""
    if not config or not enabled(config.get("enabled"), True):
        return {}
    raw = config.get("experiments", config)
    if not isinstance(raw, dict):
        raise ConfigError("clinical_degsea.experiments doit être un dictionnaire.")
    experiments = {}
    for name, spec in raw.items():
        if name == "enabled":
            continue
        if not isinstance(spec, dict):
            raise ConfigError(f"clinical_degsea.{name} doit être un dictionnaire.")
        if not enabled(spec.get("enabled"), True):
            continue
        experiment = dict(spec)
        if "contrast" not in experiment and "contraste" in experiment:
            experiment["contrast"] = experiment["contraste"]
        missing = [f for f in ("design", "contrast", "control", "test")
                   if not experiment.get(f)]
        if missing:
            raise ConfigError(
                f"clinical_degsea.{name} incomplet : clé(s) requise(s) "
                f"{', '.join(missing)}."
            )
        safe_name = str(name)
        if Path(safe_name).name != safe_name or safe_name in {"", ".", ".."}:
            raise ConfigError(f"Nom d'expérience clinique invalide : {name!r}.")
        experiments[safe_name] = experiment
    return experiments


#: clé et libellé de la strate « toute la cohorte, sans filtre »
ALL_STRATA = "ALL"

#: valeurs de `group_DESeq2_by` qui désignent l'absence de filtre
_NO_GROUPING = {"all", "none", "na", "null", "aucun", ""}


def slug(value) -> str:
    """Nom de dossier sûr pour une colonne ou une modalité clinique.

    Défini **ici**, et non des deux côtés : `run_pipeline` écrit les dossiers,
    `report` les relit. Deux implémentations qui divergeraient d'un caractère
    donneraient un rapport vide sans le moindre message.
    """
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return text or "vide"


def grouping_columns(value) -> tuple[str, ...]:
    """Normalise `group_DESeq2_by` -> colonnes de stratification, sans doublon.

    Les sentinelles ('ALL', 'None', 'NA'…) sont retirées : la strate non filtrée
    est **toujours** produite, elle n'a donc pas à être demandée explicitement.
    """
    return tuple(dict.fromkeys(
        str(col) for col in as_str_tuple(value)
        if str(col).strip().lower() not in _NO_GROUPING))


def subset_specs(config) -> dict[str, tuple[str, ...]]:
    """Normalise `subset_by` -> ``{nom: (colonnes…)}``, ordre du YAML conservé.

    Chaque entrée décrit UN découpage de la cohorte : OUTRIDER (étape 7b) est
    relancé indépendamment sur chaque combinaison de modalités de ces colonnes.
    Une liste vide signifie « pas de découpage » : un seul run, toute la cohorte.
    """
    if not config:
        return {}
    if isinstance(config, (list, tuple)):
        return {str(col): (str(col),) for col in config}
    if not isinstance(config, dict):
        raise ConfigError("subset_by doit être un dictionnaire {nom: [colonnes]}.")
    out: dict[str, tuple[str, ...]] = {}
    for name, columns in config.items():
        if columns is None:
            out[str(name)] = ()
        elif isinstance(columns, str):
            out[str(name)] = (columns,)
        elif isinstance(columns, (list, tuple)):
            out[str(name)] = tuple(str(c) for c in columns)
        else:
            raise ConfigError(
                f"subset_by.{name} : attendu une liste de colonnes, reçu "
                f"{type(columns).__name__}.")
    return out


def subset_columns(config) -> tuple[str, ...]:
    """Toutes les colonnes citées par `subset_by`, sans doublon.

    Sert au même usage que :func:`grouping_columns` : ces colonnes ne peuvent
    pas être écartées par `filter_columns`, sinon le découpage disparaîtrait
    en silence.
    """
    return tuple(dict.fromkeys(
        col for columns in subset_specs(config).values() for col in columns))


def contrast_columns(config: dict | None) -> tuple[str, ...]:
    """Colonnes citées comme `contrast` par les expériences cliniques.

    Sert à relire ces colonnes en chaînes dès le chargement des métadonnées :
    une variable 0/1 laissée en numérique serait traitée par DESeq2 comme une
    covariable continue, et un seul NA la ferait lire en float — ses modalités
    deviendraient '0.0'/'1.0' et ne correspondraient plus aux `control`/`test`.
    Les dériver ici évite d'avoir à maintenir la liste en double dans le YAML.
    """
    return tuple(dict.fromkeys(
        str(spec["contrast"]) for spec in clinical_experiments(config).values()))


def clinical_gene_sets(collections, gsea_gene_sets, experiment: dict) -> dict[str, str]:
    """Résout les collections GMT demandées par une expérience clinique."""
    available = collections_or_fallback(collections, gsea_gene_sets)
    selected = experiment.get("collections", experiment.get("gsea_collections"))
    if selected is None or selected == "all":
        return available
    if isinstance(selected, dict):
        return _normalize_collections(selected)
    if isinstance(selected, str):
        if selected in available:
            selected = [selected]
        else:  # chemin GMT unique renseigné directement dans l'expérience
            path = Path(selected).expanduser()
            return {path.stem: str(path)}
    if not isinstance(selected, (list, tuple)):
        raise ConfigError(
            "collections clinique doit être 'all', un nom, une liste ou un dictionnaire."
        )
    unknown = [str(name) for name in selected if str(name) not in available]
    if unknown:
        raise ConfigError(
            "Collection(s) clinique(s) absente(s) de gsea_collections : "
            + ", ".join(unknown)
        )
    return {str(name): available[str(name)] for name in selected}


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML absent : `pip install pyyaml` pour utiliser "
                          "--config.") from exc
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} : le YAML doit être un dictionnaire clé: valeur.")
    return data


def _cli_provided(parser: argparse.ArgumentParser, argv) -> dict:
    """Options réellement saisies au terminal, défauts exclus.

    argparse n'applique le défaut d'une option que si le namespace ne porte pas
    déjà l'attribut : en le pré-remplissant d'une sentinelle, seules les valeurs
    venues de `argv` subsistent. Aucune option n'a besoin d'être reparsée, et le
    parser n'est pas modifié.
    """
    dests = vars(parser.parse_args([]))
    namespace = argparse.Namespace(**{dest: _UNSET for dest in dests})
    parser.parse_args(argv, namespace=namespace)
    return {k: v for k, v in vars(namespace).items() if v is not _UNSET}


def load_config(argv=None) -> argparse.Namespace:
    """Fusionne défauts < YAML < ligne de commande, puis valide l'ensemble.

    Les clés du YAML sont les noms `dest` des options, avec des underscores
    (`n_top_genes`, `color_by`, `tsne_dim`, ...).
    """
    parser = build_parser()
    merged = vars(parser.parse_args([]))          # défauts seuls
    rules = _field_rules(parser)

    cli = _cli_provided(parser, argv)
    config_path = cli.get("config")

    structured = {key: {} for key in _STRUCTURED_KEYS}
    errors: list[str] = []
    if config_path:
        raw = load_yaml(Path(config_path))
        consumed: set[str] = set()
        structured = _extract_structured(raw, consumed)
        errors += _check_unknown_keys(raw, set(merged), consumed)
        for key, value in raw.items():
            if key == "config" or key in consumed or key not in merged:
                continue
            # Le YAML passe exactement par les mêmes `type` et `choices` que la
            # ligne de commande : c'est tout l'objet de ce module.
            try:
                merged[key] = _coerce(key, value, rules.get(key, {}))
            except ConfigError as exc:
                # On garde le défaut et on continue : l'utilisateur verra toutes
                # ses fautes en une fois plutôt qu'une par exécution.
                errors.append(str(exc))

    cli.pop("config", None)
    merged.update(cli)                            # priorité maximale
    merged["config"] = config_path
    merged.update(structured)

    validate(merged, extra=errors)
    return argparse.Namespace(**merged)


__all__ = [
    "ConfigError",
    "build_parser",
    "load_config",
    "load_yaml",
    "validate",
    "collections_or_fallback",
    "clinical_experiments",
    "clinical_gene_sets",
    "contrast_columns",
    "grouping_columns",
    "subset_specs",
    "subset_columns",
    "as_str_tuple",
    "slug",
    "ALL_STRATA",
    "enabled",
]
