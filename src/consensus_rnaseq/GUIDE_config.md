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

---

## 0. Le réglage le plus important : la robustesse

Un vrai sous-typage **survit au changement de réglages**. Avant de conclure quoi
que ce soit, relance en variant :

- `n_top_genes` : 2000 / 5000 / 10000
- `base` : `hierarchical` / `kmeans` / `kmedoids`
- `metric` : `pearson` / `spearman`
- `gene_mode` : `subsample` / `bootstrap`

Si la partition change du tout au tout, c'est qu'elle n'est pas réelle. Compare
les partitions avec un indice de Rand ajusté, et surtout lance `null_check.py`.

---

## 1. Entrées / sorties

| Paramètre | Comment le régler |
|---|---|
| `counts` | Chemin de la matrice. **Counts bruts** (gènes en lignes par défaut) → laisse `already_normalized: false`. Déjà en VST/rlog/logCPM → `already_normalized: true`. |
| `samples_in_rows` | `false` si gènes en lignes / échantillons en colonnes (cas RNA-seq classique). `true` sinon. |
| `metadata` | CSV/TSV indexé par échantillon (batch, histo, RIN, pureté…). Indispensable pour vérifier les confondants. |
| `color_by` | Colonne des métadonnées à superposer sur les embeddings + croiser avec les clusters (ex. `histo_classe`). |
| `outdir` | Dossier de sortie. Utilise un nom **par run** (`results/run_vst_5000g`) pour comparer les variantes. |

---

## 2. Prétraitement

### Normalisation

| Paramètre | Choix | Recommandation |
|---|---|---|
| `already_normalized` | `false` / `true` | `false` sur des counts bruts. `true` si la matrice est déjà transformée (VST, rlog, logCPM, TPM logué). |
| `norm_method` | `vst` / `logcpm` | **`vst` (défaut)** : meilleur pour le clustering, stabilise la variance des gènes peu exprimés. `logcpm` = plus rapide, sans PyDESeq2, un peu moins bon. |

### Filtrage des gènes

| Paramètre | Recommandation |
|---|---|
| `min_cpm` / `min_frac_samples` | Défauts `1.0` / `0.2` = « au moins 1 CPM dans ≥ 20 % des tumeurs ». Bien pour la plupart des cohortes. Cohorte très hétérogène → baisse `min_frac_samples` à `0.1`. Appliqué **uniquement** sur counts bruts. |
| `keep_technical` | `false` = retire ribosomiques / mitochondriaux / hémoglobines (RP*, MT-*, HB*), qui créent des clusters purement techniques. Ne mets `true` que si tu étudies spécifiquement ces programmes. |
| `n_top_genes` | **Le levier le plus influent.** `5000` par défaut. Moins (2000) = focalise sur le signal fort ; plus (10000) = capte des programmes fins mais aussi du bruit. **Teste 2000/5000/10000.** |
| `variance_method` | `mad` (défaut, robuste aux tumeurs extrêmes) ou `var`. Garde `mad` sur des tumeurs atypiques. |
| `scale_genes` | `false` par défaut (garde l'amplitude d'expression comme info). Mets `true` **seulement** si tu veux que chaque gène pèse pareil — utile avec `metric: euclidean`. |

### Filtrage d'outliers (ACP) — optionnel

| Paramètre | Recommandation |
|---|---|
| `outlier_sd_threshold` | `0` = désactivé. Sinon `3`–`4` SD est classique. **Attention** : sur des tumeurs atypiques, un seuil bas (< 3) élimine les cas intermédiaires *intéressants* plutôt que des artefacts. Regarde `figures/pca_outliers.png` avant de figer. |
| `outlier_n_pc` | Nombre de CP inspectées (défaut `10`). |
| `outlier_min_explained_var` | Alternative à `outlier_n_pc` : `0.08` = inspecte les CP expliquant > 8 % de variance (les axes de vraie structure). Prioritaire sur `outlier_n_pc` si > 0. |

---

## 3. Pureté tumorale (PUREE) — optionnel

| Paramètre | Recommandation |
|---|---|
| `purity_threshold` | `0`/`null`/`false` = pas de filtrage (PUREE non lancé). Sinon un seuil dans `]0,1[`. Typique : `0.5`–`0.6` pour écarter les tumeurs à faible contenu tumoral. |
| `purity_direction` | `higher` = garde puretés ≥ seuil (retire les tumeurs stromales/immunes). `lower` = l'inverse (rare). |
| `puree_gene_id` | `HGNC` (symboles) ou `ENSEMBL`, **selon les identifiants de ta matrice**. Mauvais choix → puretés qui s'écrasent sur une constante. Vérifie l'étalement dans `tables/purity_puree.csv`. |
| `puree_dir` / `puree_python` | Chemins du dépôt PUREE et de son interpréteur. À adapter si tu changes de machine. |

---

## 4. Consensus clustering — le cœur

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

## 5. Stabilité des branches (Jaccard)

| Paramètre | Recommandation |
|---|---|
| `compute_jaccard` | `y` = calcule la stabilité de chaque branche par bootstrap des gènes (`n_resamples` arbres). Coûte ~1× le temps du run. Mets `n` pour un test rapide, `y` pour l'analyse finale. Lecture : branche > 0,75 = stable, > 0,85 = très stable. |

---

## 5 bis. DEGSEA (DESeq2 + GSEA par cluster)

Caractérise chaque cluster après le clustering. **Étape longue**, désactivée par défaut.

| Paramètre | Recommandation |
|---|---|
| `run_degsea` | `n` (défaut) / `y`. Ne l'active qu'une fois la partition figée (k choisi). |
| `degsea_mode` | `ova` (one-vs-all, rapide, K contrastes), `ovo` (one-vs-one, **K(K-1)/2** contrastes, coûteux), `both`. Sur un grand K, reste sur `ova`. |
| `gsea_gene_sets` | Chemin d'un `.gmt`. Défaut = hallmarks MSigDB. Autres dispos en cache : Reactome, KEGG, GO_BP. Les gènes doivent être des **symboles HGNC**. |
| `gsea_permutations` | `1000` (défaut). Baisse à `200`–`500` pour aller plus vite pendant la mise au point. |

⚠️ Double-dipping : p-valeurs anticonservatives (mêmes données pour définir les clusters et les tester). Lecture descriptive, pas inférentielle.

## 6. Embeddings (t-SNE / UMAP)

Rappel : c'est une **visualisation**, pas une validation (calculée sur la même
distance consensus). Les positions *entre* nuages n'ont pas de sens.

| Paramètre | Recommandation |
|---|---|
| `tsne_dim` | `2` = PNG statiques. `3` = HTML interactif (rotation, survol = ID tumeur ; nécessite `plotly`). |
| `perplexity` | t-SNE. `30` par défaut. Grossièrement entre `n/100` et `50`. Petite cohorte → baisse (`10`–`20`). |
| `n_neighbors` | UMAP. `15` par défaut. Plus haut (`30`–`50`) = structure plus globale ; plus bas = structure locale. |
| `min_dist` | UMAP. `0.1` par défaut. Plus bas = nuages plus compacts. |
| `no_umap` | `true` si `umap-learn` absent ou pour aller plus vite (t-SNE seul). |

---

## 7. Divers

| Paramètre | Recommandation |
|---|---|
| `n_jobs` | `-1` = tous les cœurs (défaut). Réduis si la machine est partagée. |
| `seed` | `0`. Fixe la reproductibilité. Change-le pour vérifier la stabilité aux graines (embeddings surtout). |

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
