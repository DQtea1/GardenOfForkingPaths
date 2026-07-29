"""Utilitaires statistiques partagés (source unique de vérité).

Regroupe des fonctions auparavant dupliquées à l'identique dans plusieurs modules :
correction de tests multiples (Benjamini-Hochberg), détection du type d'une
variable (catégorielle / continue) et notation étoilée d'une p-valeur.

Note sur `max_levels` : le seuil « nombre de modalités au-delà duquel une variable
numérique est jugée *continue* » dépend **du contexte** et reste donc un
**argument explicite** à chaque appel (il n'y a pas de bonne valeur universelle) :
    - projection de signatures (association clinique)   -> 6
    - khi² d'indépendance (variables cliniques)          -> 12
    - affichage des annotations cliniques du rapport     -> 8
Centraliser la *logique* sans imposer un seuil unique évite les incohérences de
code tout en respectant ces choix par étape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(pvals) -> np.ndarray:
    """Correction Benjamini-Hochberg (FDR) -> q-valeurs, alignées sur l'entrée."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def is_categorical(series: pd.Series, max_levels: int = 6) -> bool | None:
    """True = catégorielle, False = continue, None = inexploitable (vide).

    Non numérique -> catégorielle ; numérique -> catégorielle si <= `max_levels`
    modalités distinctes, continue sinon.
    """
    s = series.dropna()
    if s.empty:
        return None
    if not pd.api.types.is_numeric_dtype(s):
        return True
    return s.nunique() <= max_levels


def is_continuous(series: pd.Series, max_levels: int = 6) -> bool:
    """Numérique **et** à plus de `max_levels` modalités distinctes."""
    return is_categorical(series, max_levels) is False


def stars(p) -> str:
    """Notation étoilée d'une p-/q-valeur : *** < 0.001, ** < 0.01, * < 0.05."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
