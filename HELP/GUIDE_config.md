# Guide de choix des paramètres (`config.yaml`)

Un guide pratique pour régler chaque paramètre du pipeline. Rappel de priorité :
**défauts internes < `config.yaml` < ligne de commande**. Tu peux donc figer un
run dans le YAML et surcharger un seul réglage au terminal :

```bash
python run_pipeline.py --config config.yaml --base kmeans --metric spearman
```

> **Règle d'or.** Aucun paramètre ne « trouve » les bons clusters tout seul. Le
> pipeline te donne des diagnostics (heatmaps, PAC, Δ(K), stabilité Jaccard,
> modèle nul). La bonne démarche = lancer, **regarder les figures**, ajuster.

Les sections ci-dessous sont numérotées comme les **étapes du pipeline**
(`run_pipeline.py`, `pipeline_map.html` et `config_SARAH.yaml` partagent la même
numérotation) : `1` chargement/prétraitement, `1a` PUREE, `1b` outliers, `2`
consensus, `4b` stabilité Jaccard, `5` embeddings, `6` DEGSEA, `7` signatures,
`8` déconvolution, `10` rapport HTML. Les étapes sans paramètre utilisateur
(3 diagnostics de k, 4 partition, 9 synthèse, 11 sauvegarde) n'ont pas de section
ici.

---

## Avant tout : la robustesse (le réglage le plus important)

Un vrai sous-typage **survit au changement de réglages**. Avant de conclure quoi
que ce soit, relance en variant :

- `n_top_genes` : 2000 / 5000 / 10000
- `base` : `hierarchical` / `kmeans` / `kmedoids`
- `metric` : `pearson` / `spearman`
- `gene_mode` : `subsample` / `bootstrap`

Si la partition change du tout au tout, c'est qu'elle n'est pas réelle. Compare
les partitions avec un indice de Rand ajusté, et surtout lance `null_check.py`.

---

## 1 · Entrées / sorties — `_setup()` + `_load_metadata()`

> **Numérotation.** Les numéros de ce guide, ceux des sections du YAML et ceux
> des messages du journal désignent tous la même chose : le **rang d'exécution
> réel** de l'étape dans `main()` (`run_pipeline.py`). Le YAML suit cet ordre du
> début à la fin ; ce guide, lui, garde un regroupement thématique, donc ses
> sections ne se suivent pas toujours (16 · khi² est décrite entre 15 et 17,
> mais avec le DEGSEA). Les étapes 3 (harmonisation), 6 (collections de
> gene sets), 7 (DEGSEA clinique), 8 (ICA) et 9 (GSEA des métagènes ICA) n'ont
> pas encore de section ici : voir les commentaires du YAML.

| Paramètre | Comment le régler |
|---|---|
| `counts` | Chemin de la matrice. **Counts bruts** (gènes en lignes par défaut) → laisse `already_normalized: false`. Déjà en VST/rlog/logCPM → `already_normalized: true`. |
| `samples_in_rows` | `false` si gènes en lignes / échantillons en colonnes (cas RNA-seq classique). `true` sinon. |
| `metadata` | CSV/TSV indexé par échantillon (batch, histo, RIN, pureté…). Indispensable pour vérifier les confondants. |
| `color_by` | Colonne des métadonnées à superposer sur les embeddings + croiser avec les clusters (ex. `histo_classe`). |
| `outdir` | Dossier de sortie. Utilise un nom **par run** (`results/run_vst_5000g`) pour comparer les variantes. |

---

## 2 · Prétraitement (normalisation, gènes variables) — `_load_data()`

### Normalisation

| Paramètre | Choix | Recommandation |
|---|---|---|
| `already_normalized` | `false` / `true` | `false` sur des counts bruts. `true` si la matrice est déjà transformée (VST, rlog, logCPM, TPM logué). |
| `norm_method` | `vst` / `logcpm` | **`vst` (défaut)** : meilleur pour le clustering, stabilise la variance des gènes peu exprimés. `logcpm` = plus rapide, sans PyDESeq2, un peu moins bon. |

### Filtrage des gènes

Sur des counts bruts, un premier filtre **automatique** (pas de paramètre) retire
les gènes à 0 count dans **toutes** les tumeurs, avant tout autre traitement —
nécessaire pour la VST, dont l'estimation des size factors est indéfinie sur un
gène tout-zéro.

| Paramètre | Recommandation |
|---|---|
| `min_cpm` / `min_frac_samples` | Défauts `1.0` / `0.2` = « au moins 1 CPM dans ≥ 20 % des tumeurs ». Bien pour la plupart des cohortes. Cohorte très hétérogène → baisse `min_frac_samples` à `0.1`. Appliqué **uniquement** sur counts bruts. |
| `keep_technical` | `false` = retire ribosomiques / mitochondriaux / hémoglobines (RP*, MT-*, HB*), qui créent des clusters purement techniques. Ne mets `true` que si tu étudies spécifiquement ces programmes. |
| `n_top_genes` | **Le levier le plus influent.** `5000` par défaut. Moins (2000) = focalise sur le signal fort ; plus (10000) = capte des programmes fins mais aussi du bruit. **Teste 2000/5000/10000.** |
| `variance_method` | `mad` (défaut, robuste aux tumeurs extrêmes) ou `var`. Garde `mad` sur des tumeurs atypiques. |
| `scale_genes` | `false` par défaut (garde l'amplitude d'expression comme info). Mets `true` **seulement** si tu veux que chaque gène pèse pareil — utile avec `metric: euclidean`. |

---

## 4 · Pureté tumorale (PUREE) — optionnel — `_purity_filter()`

| Paramètre | Recommandation |
|---|---|
| `purity_threshold` | `0`/`null`/`false` = pas de filtrage (PUREE non lancé). Sinon un seuil dans `]0,1[`. Typique : `0.5`–`0.6` pour écarter les tumeurs à faible contenu tumoral. |
| `purity_direction` | `higher` = garde puretés ≥ seuil (retire les tumeurs stromales/immunes). `lower` = l'inverse (rare). |
| `puree_gene_id` | `HGNC` (symboles) ou `ENSEMBL`, **selon les identifiants de ta matrice**. Mauvais choix → puretés qui s'écrasent sur une constante. Vérifie l'étalement dans `tables/purity_puree.csv`. |
| `puree_dir` / `puree_python` | Chemins du dépôt PUREE et de son interpréteur. À adapter si tu changes de machine. |

---

## 5 · Filtrage d'outliers (ACP) — optionnel — `_outlier_filter()`

| Paramètre | Recommandation |
|---|---|
| `outlier_sd_threshold` | `0` = désactivé. Sinon `3`–`4` SD est classique. **Attention** : sur des tumeurs atypiques, un seuil bas (< 3) élimine les cas intermédiaires *intéressants* plutôt que des artefacts. Regarde `figures/pca_outliers.png` avant de figer. |
| `outlier_n_pc` | Nombre de CP inspectées (défaut `10`). |
| `outlier_min_explained_var` | Alternative à `outlier_n_pc` : `0.08` = inspecte les CP expliquant > 8 % de variance (les axes de vraie structure). Prioritaire sur `outlier_n_pc` si > 0. |

---

## 10 · Consensus clustering — le cœur — `AnalysisBranch._run_consensus()`

### Combien de clusters (`k`)

| Paramètre | Recommandation |
|---|---|
| `k_min` / `k_max` | `2` → borne haute selon le nombre de sous-types plausible. Mets `k_max` large (15–30) pour voir où le PAC sature ; ça ne coûte pas cher. |
| `k_final` | `null` = choix auto (voir `k_criterion`). **À valider par les heatmaps.** Mets un entier une fois que tu as tranché (ex. `7`). |
| `k_criterion` | Critère du choix auto (si `k_final: null`) : `pac` (minimise le PAC, fiable mais sous-estime parfois k), `deltak` (coude de Δ(K), sur-estime souvent k), `both` (défaut, croise les deux). Sans effet si `k_final` est fixé. |
| `min_cluster_size` | `10` par défaut. Sert à l'heuristique `suggest_k` : un k qui produit un cluster de 3 tumeurs sur-partitionne. Monte-le sur une grosse cohorte. |

**Comment choisir k concrètement :** regarde dans l'ordre
`figures/consensus_heatmap_k*.png` (blocs nets ?), puis `PAC` (bas = bon) dans
`tables/k_selection.csv`, puis le coude de `Δ(K)`, puis la stabilité Jaccard.

### Rééchantillonnage

| Paramètre | Recommandation |
|---|---|
| `n_resamples` | `1000` (défaut) est un bon compromis. `≥ 500` minimum. Plus haut = consensus plus lisse mais plus lent (coût **linéaire**). Pour un test rapide : `200`. |
| `prop_samples` / `prop_genes` | `0.8` = standard Monti. Rarement à toucher. |
| `sample_mode` / `gene_mode` | `subsample` (sans remise, défaut) ou `bootstrap` (avec remise, **test plus sévère**). Lance les deux et compare les PAC : un gros écart = clusters fragiles. |

### Algorithme de base (`base`) — arbre de décision

```
Tumeurs transcriptomiques, sous-types corrélationnels  → hierarchical (défaut)
Tu veux des clusters compacts / sphériques, gros n     → kmeans
Robustesse aux outliers, sur distance précalculée      → kmedoids
```

| `base` | Ce qu'il fait | Quand |
|---|---|---|
| `hierarchical` | CAH sur la distance choisie, coupée à k. **Défaut, recommandé** en transcriptomique. | Cas général. Seul mode qui utilise `linkage`. |
| `kmeans` | k-means sur l'expression. **Ignore `metric` et `linkage`** (travaille en euclidien sur X). | Vérifier la robustesse ; grandes cohortes. Pense à `scale_genes: true`. |
| `kmedoids` | PAM sur la distance choisie. Plus robuste aux extrêmes que k-means. | Alternative robuste ; utilise `metric`, ignore `linkage`. |

> ⚠️ **Ce qui utilise quoi.** `metric` → `hierarchical` et `kmedoids` (pas
> `kmeans`). `linkage` → `hierarchical` pour le clustering de base **et**, dans
> tous les cas, pour la partition finale sur la matrice consensus. Donc
> `linkage` compte toujours au moins pour l'étape finale.

### Distance de base (`metric`)

| `metric` | Caractère | Quand |
|---|---|---|
| `pearson` | Corrélation linéaire, insensible à l'échelle par échantillon. **Défaut, standard RNA-seq.** | Presque toujours. |
| `spearman` | Corrélation de rangs. Robuste aux outliers et aux non-linéarités, plus conservateur. | Données bruitées, batch résiduel, tumeurs extrêmes. |
| `euclidean` | Distance brute. Sensible à l'amplitude. | Seulement avec centrage **et** `scale_genes: true`. Requis si `linkage: ward`. |
| `cosine` | Angle entre profils. | Alternative à pearson sans centrage. |

Règle simple : **reste sur `pearson`**, teste `spearman` pour la robustesse.

### Mode de linkage (`linkage`) — CAH uniquement

| `linkage` | Effet | Quand |
|---|---|---|
| `average` | Fusionne selon la distance moyenne. Équilibré. **Défaut, recommandé** (Monti). | Cas général. |
| `complete` | Distance max → clusters compacts, mais sensible aux outliers. | Si `average` donne des clusters trop « filandreux ». |
| `ward` | Minimise la variance intra → clusters de taille/forme homogènes. | **Uniquement avec `metric: euclidean`** (+ `scale_genes: true`). Incohérent sur une distance de corrélation. |
| `single` | Distance min → « chaînage ». Généralement mauvais ici. | À éviter, sauf structures en filaments. |

Combo recommandé par défaut : **`base: hierarchical` + `metric: pearson` +
`linkage: average`**. Combo « variance/euclidien » cohérent : **`base:
hierarchical` + `metric: euclidean` + `linkage: ward` + `scale_genes: true`**.

---

## 11 · Stabilité des branches (Jaccard) — `_run_branch_stability()`

| Paramètre | Recommandation |
|---|---|
| `compute_jaccard` | `y` = calcule la stabilité de chaque branche par bootstrap des gènes (`n_resamples` arbres). Coûte ~1× le temps du run. Mets `n` pour un test rapide, `y` pour l'analyse finale. Lecture : branche > 0,75 = stable, > 0,85 = très stable. |

---

## 12 · Embeddings (t-SNE / UMAP) — `_run_embeddings()`

Rappel : sur la distance consensus, c'est une **visualisation**, pas une
validation (calculée sur la même distance que le clustering). Les positions
*entre* nuages n'y ont pas de sens.

| Paramètre | Recommandation |
|---|---|
| `tsne_dim` | `2` = PNG statiques. `3` = HTML interactif (rotation, survol = ID tumeur ; nécessite `plotly`). |
| `embedding_distances` | Espaces de distance sur lesquels t-SNE / UMAP sont appris ; le menu **Distance** du rapport (onglets Consensus et ICA) bascule de l'un à l'autre. `consensus` = D_K = 1 − C_K, recalculé pour chaque k (défaut historique). `euclidean`, `correlation`, `manhattan` sont calculées **directement sur la matrice d'entrée de la branche** (scores des composantes côté ICA, expression sélectionnée côté consensus) : elles ne dépendent pas de k, qui n'y pilote plus que la coloration par cluster. Une structure visible sur la seule distance consensus est une structure produite par le clustering, pas par les données — d'où l'intérêt de comparer. |
| `perplexity` | t-SNE. `30` par défaut. Grossièrement entre `n/100` et `50`. Petite cohorte → baisse (`10`–`20`). |
| `n_neighbors` | UMAP. `15` par défaut. Plus haut (`30`–`50`) = structure plus globale ; plus bas = structure locale. |
| `min_dist` | UMAP. `0.1` par défaut. Plus bas = nuages plus compacts. |
| `no_umap` | `true` si `umap-learn` absent ou pour aller plus vite (t-SNE seul). |

---

## 7b · OUTRIDER — expression aberrante par tumeur — `_outrider()`

Question **orthogonale** au reste du pipeline : le consensus clustering cherche
des groupes, le DEGSEA des différences entre groupes, OUTRIDER des événements
**individuels** — un gène anormalement exprimé chez UNE tumeur. Un autoencodeur
débruiteur apprend la covariance du jeu (effets techniques et programmes
partagés compris), puis chaque comptage est testé contre sa valeur attendue
(binomiale négative, FDR). Le calcul est délégué à `py_outrider`.

**`subset_by` est le réglage qui décide de la validité du résultat.** OUTRIDER
n'a de sens que sur une population *comparable* : mélanger deux types tumoraux
fait apprendre au modèle la moyenne des deux, et les programmes propres à chaque
type deviennent des « aberrations » chez tout le monde. Chaque entrée déclare un
découpage, et chaque combinaison de modalités **réellement observée** donne un
run indépendant — autoencodeur, size factors et correction FDR propres :

```yaml
subset_by:
  histo_sexe: [histo_classe, sexe]   # melanome×M, melanome×F, sarcome×M, …
  histo:      [histo_classe]         # un run par classe histologique
  cohorte:    []                     # un seul run, toute la cohorte
```

Les combinaisons sans tumeur n'existent pas ; celles sous
`outrider_min_samples` sont écartées **avant** calcul. Une tumeur dont une
colonne de découpage est vide n'entre dans aucun groupe de ce découpage : elle
n'a pas de comparable déclaré. Le plan complet — retenus, écartés, motifs —
part dans `tables/outrider/plan.csv`, et le rapport propose un **menu déroulant**
(un `optgroup` par bloc `subset_by`) pour choisir le run affiché.

| Paramètre | Recommandation |
|---|---|
| `run_outrider` | `n` (défaut) / `y`. Étape longue : un autoencodeur **par sous-groupe**. |
| `subset_by` | Le réglage central, ci-dessus. Sans lui, un seul run sur toute la cohorte — rarement ce qu'on veut sur une cohorte hétérogène (le pipeline le signale). |
| `outrider_python` | Interpréteur de l'environnement où `py_outrider` est installé (il tire TensorFlow). `null` = celui du pipeline. Même logique que `puree_python`. |
| `outrider_profile` | `outrider` (counts RNA-seq, binomiale négative), `protrider` (intensités protéiques, gaussienne + covariables), `pca`. |
| `outrider_encod_dim` | Dimension du goulot de l'autoencodeur. `null` (défaut) = py_outrider cherche l'optimum : c'est correct, et **de loin le plus long** (plusieurs ajustements complets par sous-groupe). Règle du pouce pour figer : `n_tumeurs / 4`. |
| `outrider_min_samples` | `30` par défaut. En dessous, l'autoencodeur n'a pas de quoi apprendre une norme ; le groupe est écarté avec son motif. |
| `outrider_min_count` / `outrider_min_frac_samples` | Filtre d'expression, jugé **dans le sous-groupe** (`10` counts chez `25 %` des tumeurs par défaut) : un gène quasi nul n'a pas de loi estimable et dilue la FDR. |
| `outrider_alpha` | Seuil de FDR d'un couple (tumeur, gène) aberrant. `0.05` par défaut. |
| `outrider_max_events` | Événements embarqués par run dans le rapport, les plus significatifs d'abord (`5000`). Les tables complètes restent sur disque. |
| `outrider_keep_h5ad` | `y` (défaut) : garde l'AnnData complet de chaque run (p-valeurs, z-scores, prédictions), pour rejouer un run ou aller plus loin que le rapport. |
| `outrider_jobs` | Sous-groupes menés **de front**. `0` (défaut) = automatique : `n_jobs` réparti à raison de ~4 threads TensorFlow par run. `1` = séquentiel. Mesuré sur 150 tumeurs × 4000 gènes : un run seul prend 168 s à 1 thread, 78 s à 4, 63 s à 16 — le gain sature vite, alors que les sous-groupes sont indépendants. Quatre runs de front à 4 threads : **106 s au total contre 256 s** en série à 16 threads. Contrepartie : chaque run de front est un TensorFlow de plus en mémoire (la préparation des matrices, elle, reste sérialisée). |
| `outrider_figures` | `y` (défaut) : embarque de quoi tracer les 14 figures d'OUTRIDER dans le rapport. `n` ne garde que les tables. |
| `outrider_figure_cells` | Budget en **cellules** (gènes × tumeurs) des matrices embarquées, ≈ 1 Mo de rapport **par run** (`20000` par défaut). Le panneau de gènes rétrécit donc quand la cohorte grandit. |
| `outrider_heatmap_genes` | Gènes les plus variables de la heatmap gènes × tumeurs (`100` ; R en prend 500). |

### Les 14 figures, dans le rapport

Le sous-onglet OUTRIDER porte un troisième niveau d'onglets, un par figure du
paquet R. Partout : **clic sur un gène** → ses valeurs et un bouton GeneCards
(comme les volcanos DESeq2) ; **clic sur une tumeur** → ses données cliniques
(comme les boxplots).

| Figure R | Ici | Note |
|---|---|---|
| `plotAberrantPerSample` | Aberrations / tumeur | tumeurs triées, ligne de repère `outlierRatio` réglable |
| `plotVolcano` | Volcano | par tumeur ; x au choix z-score ou log2FC |
| `plotExpressionRank` | Rang d'expression | par gène ; comptages normalisés ou bruts |
| `plotExpectedVsObservedCounts` | Observé vs attendu | par gène, avec la diagonale |
| `plotQQ` | QQ | global (toutes les p-valeurs) ou par gène |
| `plotCountCorHeatmap` | Corrélation tumeurs | avant/après normalisation |
| `plotCountGeneSampleHeatmap` | Heatmap gènes × tumeurs | gènes les plus variables, centrés |
| `plotDispEsts` | Dispersions | dispersion ajustée vs expression moyenne |
| `plotEncDimSearch` | Dimension d'encodage | vide si `outrider_encod_dim` est fixé |
| `plotSizeFactors` | Size factors | ligne de repère à 1 |
| `plotExpressedGenes` | Gènes exprimés | par tumeur, union et intersection cumulées |
| `plotPowerAnalysis` | Puissance | **dérivée** des paramètres ajustés (binomiale négative), là où R simule des outliers injectés |
| `plotFPKM` | Filtrage | **équivalent** : faute de longueurs de gènes, on trace la statistique de filtrage réellement appliquée, pas des FPKM |
| `plotManhattan` | Manhattan | **non calculable** : il faut la position génomique de chaque gène, que le pipeline ne charge pas |

Les figures par gène et par tumeur lisent toutes le même **panneau** de gènes,
borné par `outrider_figure_cells`. Sa composition n'est pas neutre : tous les
gènes aberrants d'abord, les plus significatifs ensuite, puis **un tirage
aléatoire du reste** — sans ce tirage, le nuage d'un volcano ne contiendrait que
des points significatifs et donnerait l'illusion d'un jeu entièrement aberrant.
Le nombre de gènes réellement testés reste affiché à côté.

Sorties, par sous-groupe, sous `tables/outrider/<découpage>/<modalités>/` :
`aberrant_events.csv` (un événement par ligne), `aberrant_per_sample.csv`,
`aberrant_per_gene.csv`, `summary.json`, `counts.csv` (l'entrée exacte envoyée à
py_outrider, pour rejouer un run à la main) et `outrider.h5ad`.

> Installation : `py_outrider` n'est pas une dépendance du paquet — il tire
> TensorFlow. `pip install py_outrider` dans un environnement dédié, puis
> pointer `outrider_python` dessus.

> Compatibilité : `py_outrider` 0.1.0 (2022) n'est plus maintenu et échoue sur
> un environnement récent, à deux endroits sans rapport avec son modèle —
> `anndata >= 0.9` remet en colonne les vecteurs 1-D d'`obsm` (les size factors),
> ce qui fait diffuser une multiplication en `(n, n, g)` ; et `pandas >= 2.0` a
> retiré `DataFrame.append`, utilisé pour l'historique de perte, donc **après**
> l'entraînement. `gardenofforks/outrider_runner.py` corrige les deux au
> lancement, dans son seul processus : rien à installer ni à épingler dans
> l'environnement py_outrider. Le détail de ce qui a été appliqué est journalisé
> une fois par run et repris dans `summary.json` (`anndata_compat`).

---

## 13 · DEGSEA par cluster (DESeq2 + GSEA) — `_degsea_all_k()`

Caractérise chaque cluster après le clustering. **Étape longue**, désactivée par défaut.

| Paramètre | Recommandation |
|---|---|
| `run_degsea` | `n` (défaut) / `y`. Ne l'active qu'une fois la partition figée (k choisi). |
| `degsea_mode` | `ova` (one-vs-all, rapide, K contrastes), `ovo` (one-vs-one, **K(K-1)/2** contrastes, coûteux), `both`. Sur un grand K, reste sur `ova`. |
| `degsea_all_k` | `n` (défaut) / `y`. Avec `n`, le DEGSEA est calculé uniquement pour le **k recommandé** par le critère combiné PAC+Δ(K), celui marqué « ★ recommandé » dans le rapport. `y` le calcule pour **tous** les k de `[k_min..k_max]` : une sous-arborescence `tables/degsea/k<k>/` par k, et le panneau *DEGSEA OVA* s'aligne sur le k choisi. **Très coûteux** (≈ somme des K contrastes DESeq2 sur toute la plage) — à réserver à l'exploration comparative. |
| `gsea_gene_sets` | Chemin d'un `.gmt` **unique** (défaut = hallmarks MSigDB). Gènes en **symboles HGNC**. |
| `gsea_collections` | Pour tester **plusieurs collections** : un **dictionnaire** `{nom: chemin.gmt}` (ex. `c2: …/c2.gmt`, `h: …/h.gmt`). Chaque `.gmt` donne son propre GSEA (DESeq2 mutualisé). Désactive une collection en commentant sa ligne ou avec `{enabled: false, path: …}`. Remplace `gsea_gene_sets` s'il y en a au moins une. Fichiers absents ignorés (avertissement). *(La forme héritée `load_<nom>: chemin.gmt` reste acceptée.)* |
| `gsea_permutations` | `1000` (défaut). Baisse à `200`–`500` pour aller plus vite pendant la mise au point. |
| `gsea_heatmap_pval` | Seuil de **FDR q-valeur GSEA** (permutations, défaut `0.05`) pour les heatmaps `gsea_ova_heatmap_<collection>.png` : elles contiennent **tous** les pathways significatifs **après correction** (q < seuil) dans au moins un cluster. Convention GSEA usuelle : `0.25`. |

⚠️ Double-dipping : p-valeurs anticonservatives (mêmes données pour définir les clusters et les tester). Lecture descriptive, pas inférentielle.

## 14 · Projection de signatures (scoring + association clinique) — `_signatures()`

Indépendant du clustering ; score des signatures par tumeur puis test d'association aux variables cliniques. Désactivé par défaut.

| Paramètre | Recommandation |
|---|---|
| `compute_signatures` | `n` (défaut) / `y`. Score chaque signature par tumeur (ssGSEA **et** expression moyenne) et teste l'association à chaque variable des métadonnées. |
| `signature_sources` | Bloc **harmonisé** de sources hétérogènes (une entrée par fichier) : `format: gmt` (path) ou `format: csv` (path + `name_col`/`genes_col`/`detail_col`/`genes_sep`). Ajoute une entrée pour une nouvelle source, aucun code à toucher. Provenance tracée dans `signature_sources.csv`. Garde des jeux **curés** (dizaines–centaines) : le ssGSEA sur une grosse collection (c5…) est très lent. |
| `signatures_gmt` | Raccourci **source unique** `.gmt` (si `signature_sources` absent). Sinon `load_signatures_select`, sinon `gsea_gene_sets`. |
| `sig_corr_method` | `spearman` (défaut, robuste/monotone) ou `pearson` pour les variables cliniques **continues**. Les catégorielles utilisent Wilcoxon (Mann-Whitney) par paire de modalités. |
| `sig_top_n` | Nombre de top signatures affichées par variable (défaut `8`). |
| `sig_pval` | Seuil de FDR (BH) pour retenir une signature comme significative dans les figures (défaut `0.05`). |

Sorties : `tables/signatures/{scores,association}_{ssgsea,mean}.csv` et `figures/sig_{boxplots,heatmap}_{méthode}_{variable}.png`.

## 15 · Déconvolution (omnideconv / immunedeconv) — `_deconvolution()`

Composition cellulaire par tumeur, batterie de méthodes (calcul en R). **Longue**, désactivée par défaut.

| Paramètre | Recommandation |
|---|---|
| `run_deconv` | `n` (défaut) / `y`. Nécessite `omnideconv` + `immunedeconv` installés (env `lab`). Attend des **counts bruts** (symboles HGNC). |
| `deconv_methods` | Un bloc par méthode : `{enabled: true/false, <paramètres>}`. Tout activé par défaut. Sans référence : `mcp_counter`, `xcell`, `quantiseq` (`tumor: true`), `epic` (`tumor: true`). Avec référence : `dwls` (`dwls_method: mast_optimized`), `bayesprism`. Les paramètres inconnus d'une méthode sont ignorés. |
| `deconv_reference` | Référence single-cell pour DWLS/BayesPrism : `{format: rds, path, celltype_col, batch_col, max_cells_per_type}`. Le `.rds` est un Seurat/SCE ; `max_cells_per_type` (~200) sous-échantillonne pour la tractabilité. |

⚠️ **BayesPrism** = heures sur ~500 tumeurs (Gibbs). Mets `bayesprism: {enabled: false}` pour une passe rapide. Sorties : `tables/deconvolution/deconv_<méthode>.csv` + `figures/deconv_<méthode>.png`.

---

## 16 · Khi² d'indépendance (variables catégorielles) — `_clinical_analyses()`

Croise la **partition en clusters** (pour chaque k) avec chaque **variable clinique catégorielle**, et les variables cliniques entre elles — pour savoir *quelles* variables distinguent les groupes et *quelles modalités* y sont sur/sous-représentées. **Léger**, activé par défaut (sauté sans métadonnées catégorielles).

| Paramètre | Recommandation |
|---|---|
| `run_chi2` | `y` (défaut) / `n`. Pour chaque paire : tableau de contingence **avec marges**, **khi² de Pearson** d'indépendance, **V de Cramér** (taille d'effet) et **résidus standardisés ajustés** (Haberman : \|r\| > 1.96 / 2.58 = sur/sous-représentation à 5 %/1 %). **FDR (BH)** sur toutes les paires. |
| `chi2_mc_resamples` | `2000` (défaut). Nombre de permutations du **khi² de Monte-Carlo**, utilisé en **repli** quand les conditions de Cochran ne sont pas remplies (> 20 % de cellules à effectif attendu < 5, ou un attendu < 1) sur une table **R×C** ; les tables **2×2** basculent alors sur le **test exact de Fisher**. |
| `ordinal_variables` | Déclare les variables **ordinales** (stade, grade…). Un croisement impliquant une ordinale **et** un axe scorable (autre ordinale, ou variable **binaire**) utilise alors un **test de tendance** (Cochran-Armitage / linéaire-par-linéaire, ~χ²(1)) plutôt que le khi² nominal : il teste une évolution **monotone** de la proportion, plus puissant et plus juste pour de l'ordinal. Forme recommandée (ordre explicite) : `ordinal_variables: {stade: [I, II, III, IV], grade: [G1, G2, G3]}`. Forme courte `ordinal_variables: [stade, grade]` → l'ordre est le **tri** des modalités (à vérifier : `low/mid/high` se trie mal). Un croisement ordinale × nominale à > 2 modalités (p. ex. **cluster** à k>2) reste en khi². |

Sorties (`tables/chi2/`) : `chi2_summary.csv` (toutes paires, tous k, test utilisé, p, V de Cramér, FDR, diagnostic des conditions) ; `<paire>__counts.csv` (contingence + marges) et `<paire>__adjresiduals.csv` (résidus ajustés) pour k_final et les paires cliniques. Figures : `figures/chi2_residuals_clusterK<k>_<var>.png` (heatmap des résidus, rouge = sur-représenté, bleu = sous-représenté, étoilé si significatif). *Note : pour des variables **ordinales** (stade, grade), un test de tendance serait plus puissant — le khi² nominal est un choix conservateur.*

---

## 17 · Corrélations (variables continues par patient) — `_clinical_analyses()`

Corrèle deux à deux les variables **continues** à l'échelle du patient. **Léger**, activé par défaut.

| Paramètre | Recommandation |
|---|---|
| `run_correlations` | `y` (défaut) / `n`. Assemble les variables continues — **clinique continue**, scores de **signatures** (ssGSEA + moyenne), scores de **déconvolution** — et corrèle : clinique × (signatures, déconv), signatures × déconv, clinique × clinique. **Spearman** + **FDR (BH)**, observations complètes par paire. Sauté si < 2 variables continues. |
| `corr_method` | `spearman` (défaut, robuste — recommandé pour des scores bornés/asymétriques) ou `pearson`. |
| `corr_all_pairs` | `n` (défaut) / `y`. Par défaut on **saute** signature × signature (ssGSEA/moyenne redondants) et **déconv × déconv** (fractions **compositionnelles** → corrélations négatives artéfactuelles). `y` les calcule quand même (à interpréter avec prudence : CLR / corrélation partielle). |

Sorties : `tables/correlations/correlations.csv` (table longue : var1/bloc, var2/bloc, méthode, ρ, p, FDR, n), `correlations_clinic_matrix.csv` (matrice clinique × dérivés) et `figures/correlations_clinic.png` (heatmap ρ, ★ si FDR ≤ 0,05). ⚠️ La **pureté tumorale** confond expression et déconvolution : une corrélation forte peut la refléter.

---

## 18 · Rapport d'analyse (HTML interactif) — `_report()`

| Paramètre | Recommandation |
|---|---|
| `create_report` | `y` (défaut) / `n`. Génère `outdir/report.html` : un rapport **autonome** (ouvrable dans un navigateur, sans dépendance) qui agrège tout. Onglet *Résultats* — sous-onglet **Non-supervisé** : heatmap de consensus (k trié par PAC), + panneaux **alignés sur les patients** (arbre, clusters, item consensus, clinique, signatures, déconvolution, DEGSEA) empilables ; **Signatures détaillé** (boxplots de score par groupe) ; **t-SNE/UMAP** (coloration par variable/signature/déconv). Onglet *Tableaux* (tri/filtre). Onglet *Pré-analyse* (figures ACP/PUREE/CDF). |

Le fichier embarque toutes les données en JSON → il peut peser plusieurs Mo (matrices de consensus par k). Rien à installer pour le lire.

---

## 19 · Divers (transversal : parallélisation, graine)

| Paramètre | Recommandation |
|---|---|
| `parallel` | `y` (défaut) = parallélise le rééchantillonnage consensus, la stabilité Jaccard, DEGSEA (un contraste = une tâche) et les embeddings (t-SNE ∥ UMAP) sur `n_jobs` cœurs. `n` = force tout en séquentiel (`n_jobs=1`), utile pour déboguer ou sur une machine partagée. |
| `n_jobs` | `-1` = tous les cœurs (défaut). Sans effet si `parallel: n`. Réduis (ex. `4`) si la machine est partagée mais que tu veux garder du parallélisme partiel. |
| `seed` | `0`. Fixe la reproductibilité. Change-le pour vérifier la stabilité aux graines (embeddings surtout). |
| `log_file` | Journal horodaté de la progression (défaut `run.log`, dans `outdir`). Toute la sortie console y est aussi écrite, avec date complète et temps total en fin de run. Mets un chemin absolu pour le placer ailleurs. |

---

## Trois profils de départ

**Analyse standard (recommandée), counts bruts :**
```yaml
already_normalized: false
norm_method: vst
n_top_genes: 5000
variance_method: mad
base: hierarchical
metric: pearson
linkage: average
n_resamples: 1000
k_min: 2
k_max: 15
k_final: null
compute_jaccard: y
```

**Test de robustesse (à comparer au précédent, indice de Rand) :**
```yaml
base: kmedoids        # ou kmeans
metric: spearman
gene_mode: bootstrap
n_top_genes: 2000
```

**Passe rapide (mise au point) :**
```yaml
norm_method: logcpm
n_resamples: 200
k_max: 8
compute_jaccard: n
no_umap: true
```

---

## Après le run : que regarder, dans l'ordre

1. `figures/consensus_heatmap_k*.png` — des blocs nets ? (critère n°1)
2. `tables/k_selection.csv` — PAC bas, coude de Δ(K).
3. `figures/branch_stability_dendrogram_k*.png` — branches vertes = stables.
4. `null_check.py` — l'écart observé vs nul est-il réel ? (obligatoire)
5. `tables/crosstab_*.csv` — les clusters s'alignent-ils sur un confondant
   (batch, pureté, histo) plutôt que sur de la biologie ?
