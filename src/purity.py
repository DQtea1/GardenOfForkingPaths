"""Estimation de la pureté tumorale par PUREE, puis filtrage des tumeurs.

PUREE (Revkov et al., *Commun Biol* 2023) prédit la fraction de cellules
tumorales à partir de l'expression bulk. Il embarque son propre jeu de gènes
prédictifs et ses propres dépendances (versions figées de scikit-learn) : on
l'exécute donc **en sous-processus**, dans son environnement dédié, plutôt que
de l'importer dans le pipeline. On lui passe la matrice d'expression brute
(`samples x genes`, identifiants de gènes HGNC ou ENSEMBL), on récupère une
pureté par tumeur, puis on retire les tumeurs au-dessus ou au-dessous d'un seuil.

Usage typique : écarter les tumeurs à faible pureté (fort contenu stromal /
immunitaire), dont le profil bulk reflète surtout le micro-environnement et crée
des faux sous-types (cf. piège n°4 du README).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_FALSY = {"", "0", "0.0", "null", "none", "false", "no"}


def parse_threshold(value) -> float | None:
    """Interprète `purity_threshold`. Renvoie le seuil dans ]0, 1[ si le filtrage
    est actif, sinon `None` (valeurs 0 / null / false / '' -> pas de filtrage)."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        if value.strip().lower() in _FALSY:
            return None
        value = float(value)
    value = float(value)
    if value <= 0:
        return None
    if not 0.0 < value < 1.0:
        raise ValueError(f"purity_threshold doit être dans ]0, 1[ (reçu {value}).")
    return value


def run_puree(
    expr: pd.DataFrame,
    puree_dir: str | Path,
    python_exe: str | Path,
    gene_id_type: str = "HGNC",
) -> pd.Series:
    """Lance PUREE sur `expr` (`samples x genes`) et renvoie une Series de pureté
    indexée par échantillon (alignée sur `expr.index`).

    `puree_dir` : dossier du dépôt PUREE (contenant `predict_purity.py`, `models/`
    et `data/`) ; le script est exécuté depuis ce dossier car il charge ses
    modèles par chemin relatif. `python_exe` : interpréteur de l'environnement
    PUREE. `gene_id_type` : "HGNC" (symboles) ou "ENSEMBL".
    """
    puree_dir = Path(puree_dir)
    script = puree_dir / "predict_purity.py"
    if not script.exists():
        raise FileNotFoundError(
            f"predict_purity.py introuvable dans {puree_dir} — vérifie `puree_dir`.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_path, out_path = tmp / "expr.tsv", tmp / "purity.tsv"
        expr.to_csv(in_path, sep="\t")
        cmd = [str(python_exe), str(script),
               "--data_path", str(in_path), "--output", str(out_path),
               "--gene_identifier_type", gene_id_type]
        logger.info("PUREE : %s (cwd=%s)", " ".join(cmd), puree_dir)
        proc = subprocess.run(cmd, cwd=str(puree_dir),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "PUREE a échoué (code %d).\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:]))
        purity = pd.read_csv(out_path, sep="\t", index_col=0)["purity"]

    purity = purity.reindex(expr.index)
    n_missing = int(purity.isna().sum())
    if n_missing:
        logger.warning("PUREE : %d tumeur(s) sans pureté prédite (conservées, "
                       "non filtrées).", n_missing)
    return purity


def purity_keep_mask(
    purity: pd.Series, threshold: float, direction: str
) -> pd.Series:
    """Masque booléen (True = tumeur conservée).

    `direction="higher"` garde les puretés `>= threshold` (retire les faibles
    puretés) ; `direction="lower"` garde les puretés `<= threshold`. Les tumeurs
    sans pureté (NaN) sont conservées par prudence.
    """
    if direction == "higher":
        mask = purity >= threshold
    elif direction == "lower":
        mask = purity <= threshold
    else:
        raise ValueError(f"purity_direction inconnu : {direction!r} (higher | lower).")
    return mask.where(purity.notna(), True)
