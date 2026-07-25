# `human_pathways.rds` — structure des données

Fichier référencé dans `config_SARAH.yaml` (`human_pathways:`), chargé si
`load_human_pathways: y`. C'est un objet **R sérialisé** (`.rds`) contenant
**MSigDB regroupé par collection**, plus une collection maison.

## Structure (liste R imbriquée à 3 niveaux)

```
human_pathways                       (named list — 10 collections)
└── <collection>  ex. "h", "c2"…     (named list — N gene sets)
    └── <gene set>  ex. "HALLMARK_HYPOXIA"   (character vector)
        └── "HIF1A", "VEGFA", "LDHA", …      (symboles HGNC)
```

- Niveau 1 : **10 collections** (`c1 c2 c3 c4 c5 c6 c7 c8 h sigGeNeHetX`).
- Niveau 2 : chaque collection est une **liste nommée de gene sets**.
- Niveau 3 : chaque gene set est un **vecteur de symboles de gènes** (HGNC ;
  ~0,04 % d'IDs ENSEMBL résiduels non mappés).

**Total : 46 157 gene sets.** Voir `human_pathways_structure.png` pour le
nombre de gene sets et la distribution de taille par collection.

## Les collections

| Clé | Gene sets | Taille méd. | Contenu (MSigDB) |
|---|--:|--:|---|
| `c5` | 28 339 | 8 | **Ontology** : Gene Ontology (BP/CC/MF) + Human Phenotype Ontology |
| `c2` | 6 495 | 34 | **Curated** : pathways canoniques (Reactome, KEGG, WikiPathways, BioCarta, PID) + perturbations |
| `c7` | 5 219 | 199 | **Immunologic** : ImmuneSigDB + réponses vaccinales |
| `c3` | 3 713 | 127 | **Regulatory targets** : cibles de facteurs de transcription (TFT) et de miARN (MIR) |
| `c4` | 858 | 62 | **Computational cancer** : voisinages de gènes / modules cancer |
| `c8` | 830 | 115 | **Cell type** : signatures de types cellulaires (single-cell) |
| `c1` | 300 | 114 | **Positional** : bandes cytogénétiques (ex. `chr1p11`) |
| `c6` | 189 | 172 | **Oncogenic** : dérégulations oncogéniques |
| `sigGeNeHetX` | 164 | 93 | **CUSTOM (non-MSigDB)** : signatures immunes maison (ex. `IMMU_Neutroatlas_ARG1+`) |
| `h` | 50 | 180 | **Hallmark** : 50 signatures raffinées d'états biologiques |

## Notes d'usage

- **Symboles HGNC** → directement compatible avec le GSEA du pipeline (même
  convention que `gsea_gene_sets`, hallmarks MSigDB).
- **`sigGeNeHetX` n'est pas du MSigDB standard** : c'est une collection ajoutée
  (signatures immunes / Neutro-atlas), probablement le cœur d'intérêt de la
  config IMMUNO.
- **`c5` domine en nombre** (28 k gene sets, souvent très petits — médiane 8
  gènes) : à filtrer par taille (`min_size`/`max_size`) avant un GSEA pour
  éviter le bruit et les temps de calcul.
- Pour brancher ces gene sets sur `gseapy`, il faudra les **convertir en `.gmt`**
  (une ligne `nom <tab> description <tab> gène1 <tab> gène2 …` par gene set),
  ou les passer sous forme de dict `{nom: [gènes]}`.
