"""Public topic-model API.

This package keeps the stable import path `ads_bib.topic_model` while
organizing implementation by concern.
"""

from __future__ import annotations

from ads_bib.topic_model.backends import (
    fit_bertopic,
    fit_toponymy,
    reduce_outliers,
    reduce_toponymy_outliers,
)
from ads_bib.topic_model.embeddings import OpenRouterEmbedder, compute_embeddings
from ads_bib.topic_model.output import build_topic_dataframe
from ads_bib.topic_model.reduction import reduce_dimensions

__all__ = [
    "OpenRouterEmbedder",
    "build_topic_dataframe",
    "compute_embeddings",
    "fit_bertopic",
    "fit_toponymy",
    "reduce_dimensions",
    "reduce_outliers",
    "reduce_toponymy_outliers",
]
