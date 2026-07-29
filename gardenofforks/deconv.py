"""Section 8 — Déconvolution : batterie de méthodes via omnideconv / immunedeconv.

Estime la composition cellulaire de chaque tumeur avec plusieurs algorithmes :

  - **sans référence** (immunedeconv) : MCPcounter, xCell, quanTIseq, EPIC —
    signatures intégrées, tournent directement sur l'expression bulk ;
  - **avec référence single-cell** (omnideconv) : DWLS, BayesPrism — construisent
    une signature à partir d'un scRNA-seq annoté (types cellulaires) fourni par
    l'utilisateur, puis déconvoluent le bulk.

Le calcul lourd est délégué à R (`deconvolve.R`) en sous-processus (omnideconv et
immunedeconv sont dans le même environnement que le pipeline). Chaque méthode est
activable et paramétrable indépendamment via le YAML ; les paramètres inconnus
d'une méthode sont ignorés côté R (filtrés sur les vrais arguments).

⚠️ BayesPrism (échantillonnage de Gibbs) est **très lent** sur ~500 tumeurs ;
la référence single-cell est sous-échantillonnée par type cellulaire
(`max_cells_per_type`) pour rester tractable.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

R_SCRIPT = Path(__file__).with_name("deconvolve.R")

# Batterie par défaut : toutes les méthodes, paramètres aux valeurs conseillées.
DEFAULT_METHODS: dict = {
    "mcp_counter": {"enabled": True},
    "xcell": {"enabled": True},
    "quantiseq": {"enabled": True, "tumor": True},
    "epic": {"enabled": True, "tumor": True},
    "dwls": {"enabled": True, "dwls_method": "mast_optimized"},
    "bayesprism": {"enabled": True},
}

_TRUTHY = {True, 1, "y", "yes", "true", "1", "on"}


def _enabled(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return value in _TRUTHY


def _normalize_methods(methods: dict) -> dict:
    """Uniformise `enabled` (y/true/1 -> bool) en gardant les autres paramètres."""
    out = {}
    for name, spec in methods.items():
        spec = dict(spec or {})
        spec["enabled"] = _enabled(spec.get("enabled", True))
        out[name] = spec
    return out


def run_deconvolution(
    counts: pd.DataFrame,
    sample_names,
    outdir: Path,
    methods: dict | None = None,
    reference: dict | None = None,
    rscript: str = "Rscript",
) -> dict[str, pd.DataFrame]:
    """Lance la batterie de déconvolution. `counts` : tumeurs × gènes (counts
    bruts, symboles HGNC). Renvoie {méthode: DataFrame types cellulaires × tumeurs}."""
    methods = _normalize_methods(methods or DEFAULT_METHODS)
    base = Path(outdir) / "tables" / "deconvolution"
    base.mkdir(parents=True, exist_ok=True)

    active = [m for m, s in methods.items() if s["enabled"]]
    if not active:
        logger.warning("Déconvolution : aucune méthode activée — étape sautée.")
        return {}
    needs_ref = [m for m in ("dwls", "bayesprism") if methods.get(m, {}).get("enabled")]
    if needs_ref and not (reference and reference.get("path")):
        logger.warning("Déconvolution : %s nécessite(nt) une référence single-cell "
                       "(deconv_reference) — ces méthodes seront sautées.", needs_ref)

    bulk = counts.loc[sample_names].round().astype(int).T          # gènes × tumeurs
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bulk_path, cfg_path = tmp / "bulk.tsv", tmp / "config.json"
        bulk.to_csv(bulk_path, sep="\t")
        cfg_path.write_text(json.dumps({"methods": methods, "reference": reference}))
        cmd = [rscript, str(R_SCRIPT), "--bulk", str(bulk_path),
               "--config", str(cfg_path), "--outdir", str(base)]
        logger.info("Déconvolution : %d méthode(s) active(s) (%s) — appel de R…",
                    len(active), ", ".join(active))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            logger.info("deconvolve.R :\n%s", proc.stdout.strip())
        if proc.returncode != 0:
            logger.warning("deconvolve.R a renvoyé le code %d\nstderr:\n%s",
                           proc.returncode, proc.stderr[-3000:])

    results = {}
    for meth in methods:
        f = base / f"deconv_{meth}.csv"
        if f.exists():
            results[meth] = pd.read_csv(f, index_col=0)           # types × tumeurs
    logger.info("Déconvolution : %d/%d méthode(s) ont produit un résultat.",
                len(results), len(active))
    return results
