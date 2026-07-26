"""Consensus clustering de tumeurs à partir de bulk RNA-seq."""

from . import preprocessing
from . import consensus, deconv, degsea, embedding, metrics, plots  # noqa: F401
from . import purity, report, sigproj, stability  # noqa: F401

__version__ = "0.1.0"