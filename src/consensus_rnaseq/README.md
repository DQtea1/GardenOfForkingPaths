# Consensus clustering de tumeurs atypiques — bulk RNA-seq

Pipeline de classification non supervisée pour ~500 tumeurs, avec
**rééchantillonnage double (patients ET gènes)** et visualisation t-SNE / UMAP
calculée **à partir de la distance consensus**, pas de l'expression brute.

```
.
├── run_pipeline.py                  # pipeline complet (CLI)
├── config.yaml                      # tous les paramètres du pipeline (--config)
├── GUIDE_config.md                  # comment régler chaque paramètre (algo, distance, linkage…)
├── null_check.py                    # contrôle par modèle nul — à ne pas sauter
├── make_demo_data.py                # 500 tumeurs simulées, 4 sous-types + atypiques
├── requirements.txt
└── src/consensus_rnaseq/
    ├── preprocessing.py             # filtrage, VST/logCPM, gènes variables, centrage, outliers ACP
    ├── purity.py                    # pureté tumorale (PUREE) + filtrage
    ├── consensus.py                 # cœur : rééchantillonnage + matrice consensus
    ├── metrics.py                   # CDF, Δ(K), PAC, item/cluster consensus, silhouette
    ├── stability.py                 # stabilité Jaccard des branches (bootstrap gènes)
    ├── embedding.py                 # t-SNE / UMAP 2D/3D sur distance précalculée
    └── plots.py                     # heatmaps, tracking plot, nuages, dendro stabilité
```

## Démarrage rapide

```bash
pip install -r requirements.txt

# 1. jeu de démonstration, pour valider l'installation (~30 s)
python make_demo_data.py
python run_pipeline.py --counts data/demo_counts.tsv \
    --metadata data/demo_metadata.tsv --color-by true_subtype \
    --outdir results/demo --n-resamples 300 --k-max 7

python /home/quentin/01_PROJETS/14_ConsensusClusterBulk/src/consensus_rnaseq/run_pipeline.py \
    --counts /mnt/d/03_SARAH_projet/00_DATA/01_BULK/01_MERGED_BULKS/00_Unfiltered/RNAseq_counts.csv \
    --metadata /mnt/d/03_SARAH_projet/00_DATA/02_CLINIC/clinique_itd.csv \
    --color-by histo_classe \
    --outdir results/SARAH \
    --n-top-genes 1000 \
    --n-resamples 1000 \
    --k-max 30 \
    --k-final 7 \
    --compute_jaccard y \
    --t-SNE_dim 3 

# 2. tes données (counts bruts, gènes en lignes)
python run_pipeline.py --counts data/counts.tsv --outdir results/run01 \
    --n-resamples 1000 --k-max 10 --n-top-genes 5000

# 3. contrôle nul, obligatoire avant toute interprétation
python null_check.py --counts data/counts.tsv --outdir results/run01/null
```

## Fichier de configuration

Plutôt que de tout passer en ligne de commande, on peut remplir `config.yaml`
(un modèle commenté listant *tous* les paramètres) et lancer :

```bash
python run_pipeline.py --config config.yaml
```

Priorité : défauts internes < `config.yaml` < ligne de commande. Un paramètre
saisi au terminal l'emporte donc toujours sur le YAML — pratique pour rejouer un
run en changeant un seul réglage :

```bash
python run_pipeline.py --config config.yaml --k-max 12 --compute_jaccard n
```

Les clés du YAML utilisent des underscores (`n_top_genes`, `color_by`,
`tsne_dim`, ...), pas des tirets ; `null` = défaut interne. `--counts` reste
obligatoire mais peut être fourni indifféremment dans le YAML ou au terminal.

Embeddings en 3D interactifs : `--t-SNE_dim 3` (défaut `2`). En 2D on sort les
PNG habituels ; en 3D, un fichier `.html` autonome par méthode (t-SNE, UMAP)
qu'on ouvre dans un navigateur — rotation à la souris, zoom, et survol d'un
point pour lire l'ID de la tumeur et son item consensus. Nécessite `plotly`
(`pip install plotly`). La mise en garde du piège n°3 vaut aussi en 3D : la
position relative des nuages n'a pas de sens, seule la structure locale compte.

**Normalisation.** Sur des counts bruts, le pipeline applique **par défaut le
VST de DESeq2** (via PyDESeq2), préférable au logCPM pour du clustering car il
stabilise la variance sur toute la gamme d'expression (les gènes faiblement
exprimés ne pilotent plus la distance). `norm_method: logcpm` rebascule sur le
logCPM interne (plus rapide, sans PyDESeq2). Si ta matrice est **déjà**
transformée (VST / rlog / logCPM), ajoute `--already-normalized` : filtrage et
normalisation sont alors sautés.

Temps indicatif : 500 tumeurs × 5 000 gènes, B = 1 000, k = 2..10, 8 cœurs →
quelques minutes. Le coût est linéaire en B et quadratique en nombre de tumeurs.

## Filtrage par pureté tumorale (PUREE)

Juste après le chargement des données, `purity_threshold` (dans le YAML ou
`--purity_threshold`) déclenche une estimation de la pureté tumorale par
**PUREE** (Revkov et al., *Commun Biol* 2023) sur la matrice d'expression brute,
puis retire les tumeurs selon le seuil. `purity_direction: higher` garde les
puretés `>= seuil` (retire les tumeurs à faible contenu tumoral, dominées par le
stroma / l'infiltrat immunitaire) ; `lower` garde les puretés `<= seuil`. Un
seuil `0`, `null` ou `false` désactive tout (PUREE n'est même pas lancé).

PUREE tourne dans **son propre environnement** : renseigne `puree_dir` (dépôt
contenant `predict_purity.py`) et `puree_python` (interpréteur de l'env PUREE) ;
`puree_gene_id` vaut `HGNC` (symboles) ou `ENSEMBL` selon les identifiants de ta
matrice. Le pipeline l'appelle en sous-processus. Sorties :
`tables/purity_puree.csv` (pureté + `kept` par tumeur) et
`figures/purity_puree.png` (histogramme, seuil et zone retirée).

## Filtrage d'outliers (ACP)

`--outlier_sd_threshold N` calcule, entre le prétraitement et le consensus
clustering, une ACP sur la matrice prétraitée et retire les tumeurs dont le
score s'écarte de plus de `N` écarts-types de la moyenne sur au moins une des
`--outlier_n_pc` premières composantes (défaut 10). Non renseigné ou `0` : aucun
retrait. Variante : `--outlier_min_explained_var 0.08` inspecte plutôt **toutes**
les composantes dont la variance expliquée dépasse le seuil (ici 8 %) au lieu
d'un nombre fixe — pratique pour ne garder que les axes de structure réelle.
Sorties : `tables/pca_outliers.csv` (z max et CP responsable par tumeur)
et `figures/pca_outliers.png` (PC1 vs PC2, tumeurs retirées annotées). Un seuil
trop bas (< 3) sur des tumeurs atypiques risque d'éliminer les cas
intermédiaires les plus intéressants plutôt que de vrais artefacts techniques —
regarde la figure avant de figer le seuil.

## Ce que fait le rééchantillonnage double

À chaque itération *b* on tire 80 % des patients **et** 80 % des gènes, on
clusterise la sous-matrice, et on incrémente :

- `M[i,j]` si *i* et *j* tombent dans le même cluster ;
- `I[i,j]` si *i* et *j* ont été tirés ensemble.

Consensus `C = M / I`, distance `D = 1 − C`.

Le rééchantillonnage des gènes est l'ajout par rapport à ConsensusClusterPlus
(qui, par défaut, ne rééchantillonne que les patients — l'option `pFeature`
existe mais est rarement utilisée). Il répond à une question différente :
*est-ce que mes groupes tiennent si je change de jeu de features ?* Sur des
tumeurs atypiques, un cluster qui ne survit qu'à un sous-ensemble de gènes est
souvent un cluster porté par un seul programme (cycle cellulaire, infiltrat
immunitaire, contamination stromale) plutôt qu'un vrai sous-type.

**Subsample vs bootstrap.** Le défaut est le sous-échantillonnage sans remise
(Monti et al. 2003). `--gene-mode bootstrap` tire avec remise : les gènes
dupliqués sont surpondérés dans la distance, ce qui augmente la variance entre
itérations et donne un test de stabilité plus sévère. Sur les patients, le
bootstrap avec remise crée des lignes identiques, dont on ne garde qu'une
occurrence pour la connectivité — l'effet est alors proche d'un
sous-échantillonnage à ~63 %. Lance les deux et compare les PAC : un écart
important signale des clusters fragiles.

## Choix de k

Trois critères, à croiser — aucun ne tranche seul :

| Sortie | Lecture |
|---|---|
| `figures/consensus_heatmap_k*.png` | **critère principal** : blocs nets, sans dégradé aux bordures |
| `PAC` | proportion de paires au consensus ambigu (]0,1 ; 0,9[) — à minimiser |
| `Δ(K)` | gain d'aire sous la CDF ; on cherche le coude, pas le maximum |
| `silhouette_consensus` | cohérence sur la distance consensus |
| `min_cluster_size` | un k qui produit un cluster de 3 tumeurs sur-partitionne |

`--k-final N` force k ; sinon `suggest_k` applique une heuristique
(k le plus grand parmi ceux à PAC quasi minimal qui apporte encore un Δ(K)
substantiel). C'est une aide au tri, pas une décision — regarde les heatmaps.

Δ(K) a un biais monotone documenté et sur-estime souvent k ; le PAC est plus
fiable, mais sature sur des données bien structurées (d'où l'heuristique mixte).

## Stabilité des branches (Jaccard bootstrap)

`--compute_jaccard y` ajoute, après la partition finale, une évaluation de la
reproductibilité de **chaque branche** de l'arbre consensus (façon pvclust /
Hennig 2007). On construit `--n-resamples` arbres bootstrap en rééchantillonnant
les **gènes avec remise** (toutes les tumeurs restent présentes, ce qui garde un
univers commun et rend le Jaccard entre branches exact). Pour chaque branche de
l'arbre consensus, le score = moyenne sur les B arbres bootstrap du Jaccard
maximal avec une branche de l'arbre bootstrap.

Lecture (seuils de Hennig) : `< 0,5` branche instable (souvent un artefact) ;
`0,6–0,75` motif présent mais incertain ; `> 0,75` stable ; `> 0,85` très stable.
Sorties : `tables/branch_stability_k*.csv` (une ligne par branche : taille,
score, membres, et `is_final_cluster` pour les branches égales à un cluster
retenu) et `figures/branch_stability_dendrogram_k*.png` (dendrogramme dont les
branches sont colorées du rouge = instable au vert = stable). C'est le
complément « par branche » du PAC, qui est lui global.

## Pièges à connaître

**1. Le consensus clustering trouve toujours des clusters.** Sur des données
sans structure de groupe, les matrices consensus paraissent quand même
blocardes et le PAC reste bas. C'est le résultat central de Şenbabaoğlu,
Michailidis & Li (Sci Rep 2014). D'où `null_check.py` : il recalcule le PAC
sur des données permutées gène par gène et sur un nul gaussien à covariance
appariée. **Le signal, c'est l'écart entre observé et nul**, pas la valeur
absolue du PAC. Sur le jeu de démonstration : PAC observé ≈ 0,05 contre ≈ 0,65
(nul apparié) et ≈ 1,0 (permutation).

**2. L'embedding n'est pas une validation.** t-SNE et UMAP sont calculés sur la
même distance consensus qui a servi à définir les clusters : ils *ne peuvent
pas* les contredire. Des blobs bien séparés sont attendus mécaniquement. Pour
une lecture indépendante, refais un UMAP sur l'expression (`metric="correlation"`
sur la matrice de gènes) et vérifie que la structure y est encore visible.

**3. La distance consensus est très non euclidienne.** Elle est bornée dans
[0, 1], avec beaucoup d'ex aequo à 0 et 1. Les distances entre clusters bien
séparés valent toutes 1, donc **la position relative des nuages sur le plan
n'a aucun sens** — seule la structure locale (compacité, tumeurs en position
intermédiaire) est interprétable. Ne raconte pas d'histoire sur « le cluster 2
est entre le 1 et le 3 ». Vérifie aussi la robustesse aux graines
(`embedding.stability_of_embedding`).

**4. Confondants techniques.** Avant de conclure à des sous-types biologiques,
croise les clusters avec : batch/série de séquençage, profondeur de librairie,
taux de duplicats, % mitochondrial, RIN, pureté tumorale estimée (ESTIMATE),
type histologique, site de prélèvement. `--metadata / --color-by` produit le
tableau croisé et la superposition sur l'embedding. Un « sous-type » qui
s'aligne sur le batch n'en est pas un.

**5. Sélection de gènes.** Le nombre de gènes variables retenus (5 000 par
défaut) influence fortement le résultat. Vérifie la stabilité sur 2 000 /
5 000 / 10 000 gènes avant de figer une partition.

**6. Les tumeurs à `item_consensus` bas** (< 0,7-0,8) sont celles qui n'ont pas
de place stable — sur un corpus de tumeurs atypiques, ce sont souvent les cas
les plus intéressants cliniquement, pas du bruit à écarter.

## Sorties

```
results/run01/
├── run_params.json
├── consensus_matrix_k4.npy
├── figures/
│   ├── consensus_heatmap_k*.png     # matrices consensus réordonnées
│   ├── cdf_pac_deltak.png           # choix de k
│   ├── tracking_plot.png            # suivi des affectations quand k augmente
│   ├── embeddings_k4_cluster.png    # t-SNE + UMAP colorés par cluster
│   └── item_consensus_k4.png        # stabilité par tumeur
└── tables/
    ├── k_selection.csv
    ├── cluster_assignments_k4.csv   # cluster, item consensus, silhouette
    ├── cluster_consensus_k4.csv
    ├── consensus_distance_k4.csv.gz
    └── embeddings_k4.csv
```

## Suites naturelles

- **Caractérisation** : expression différentielle par cluster (limma-voom ou
  DESeq2 avec le cluster en facteur), puis GSEA / ssGSEA sur les hallmarks.
  Attention au double-dipping : les p-valeurs issues de tests sur des groupes
  définis par les mêmes données sont anticonservatives (voir l'inférence
  post-clustering, Gao/Bien/Witten 2022, ou une approche par data-splitting).
- **Validation externe** : projeter une cohorte indépendante sur les centroïdes
  des clusters (corrélation aux centroïdes / classifieur type PAM ou NTP), et
  vérifier que la structure se reproduit.
- **Robustesse méthodologique** : relancer avec `--base kmeans` et
  `--base kmedoids`, `--metric spearman`, et comparer les partitions
  (indice de Rand ajusté). Un vrai sous-typage survit au changement
  d'algorithme de base.

## Références

- Monti et al., *Machine Learning* 52:91-118 (2003) — consensus clustering.
- Şenbabaoğlu, Michailidis & Li, *Sci Rep* 4:6207 (2014) — critique du Δ(K),
  introduction du PAC, comportement sur données nulles.
- Wilkerson & Hayes, *Bioinformatics* 26:1572 (2010) — ConsensusClusterPlus.
