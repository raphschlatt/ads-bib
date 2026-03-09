"""NASA ADS bibliometric analysis pipeline."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ads_bib._utils.logging import suppress_noisy_third_party_logs

suppress_noisy_third_party_logs()

from ads_bib.author_disambiguation import apply_author_disambiguation
from ads_bib.citations import build_all_nodes, process_all_citations
from ads_bib.config import init_paths, load_env
from ads_bib.pipeline import (
    PipelineConfig,
    PipelineContext,
    run_pipeline,
    run_author_disambiguation_stage,
    run_citations_stage,
    run_curate_stage,
    run_embeddings_stage,
    run_export_stage,
    run_reduction_stage,
    run_search_stage,
    run_tokenize_stage,
    run_topic_dataframe_stage,
    run_topic_fit_stage,
    run_translate_stage,
    run_visualize_stage,
)
from ads_bib.curate import get_cluster_summary, remove_clusters
from ads_bib.export import resolve_dataset
from ads_bib.run_manager import RunManager
from ads_bib.search import search_ads
from ads_bib.tokenize import tokenize_texts
from ads_bib.topic_model import (
    build_topic_dataframe,
    compute_embeddings,
    fit_bertopic,
    fit_toponymy,
    reduce_dimensions,
    reduce_outliers,
)
from ads_bib.translate import detect_languages, translate_dataframe

try:
    __version__ = version("ads_bib")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "apply_author_disambiguation",
    "PipelineConfig",
    "PipelineContext",
    "RunManager",
    "build_all_nodes",
    "build_topic_dataframe",
    "compute_embeddings",
    "detect_languages",
    "fit_bertopic",
    "fit_toponymy",
    "get_cluster_summary",
    "init_paths",
    "load_env",
    "process_all_citations",
    "reduce_dimensions",
    "reduce_outliers",
    "remove_clusters",
    "resolve_dataset",
    "search_ads",
    "run_pipeline",
    "run_author_disambiguation_stage",
    "run_citations_stage",
    "run_curate_stage",
    "run_embeddings_stage",
    "run_export_stage",
    "run_reduction_stage",
    "run_search_stage",
    "run_tokenize_stage",
    "run_topic_dataframe_stage",
    "run_topic_fit_stage",
    "run_translate_stage",
    "run_visualize_stage",
    "tokenize_texts",
    "translate_dataframe",
]
