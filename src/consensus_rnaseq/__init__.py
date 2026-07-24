"""Consensus clustering de tumeurs à partir de bulk RNA-seq."""

from . import preprocessing
from . import consensus, embedding, metrics, plots, purity, stability  # noqa: F401

__version__ = "0.1.0"