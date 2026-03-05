from __future__ import annotations

import inspect

import numpy as np

import ads_bib.topic_model as tm


EXPECTED_PUBLIC_NAMES = {
    "OpenRouterEmbedder",
    "build_topic_dataframe",
    "compute_embeddings",
    "fit_bertopic",
    "fit_toponymy",
    "reduce_dimensions",
    "reduce_outliers",
}


def _assert_keyword_only(fn, names: list[str]) -> None:
    signature = inspect.signature(fn)
    for name in names:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_topic_model_public_api_names_are_stable():
    assert EXPECTED_PUBLIC_NAMES.issubset(set(dir(tm)))


def test_compute_embeddings_signature_contract():
    signature = inspect.signature(tm.compute_embeddings)
    assert list(signature.parameters) == [
        "documents",
        "provider",
        "model",
        "cache_dir",
        "batch_size",
        "max_workers",
        "dtype",
        "api_key",
        "openrouter_cost_mode",
        "cost_tracker",
    ]
    _assert_keyword_only(
        tm.compute_embeddings,
        [
            "provider",
            "model",
            "cache_dir",
            "batch_size",
            "max_workers",
            "dtype",
            "api_key",
            "openrouter_cost_mode",
            "cost_tracker",
        ],
    )
    assert signature.parameters["dtype"].default == np.float16


def test_reduce_dimensions_signature_contract():
    signature = inspect.signature(tm.reduce_dimensions)
    assert list(signature.parameters) == [
        "embeddings",
        "method",
        "params_5d",
        "params_2d",
        "random_state",
        "cache_dir",
        "cache_suffix",
        "embedding_id",
        "show_progress",
    ]
    _assert_keyword_only(
        tm.reduce_dimensions,
        [
            "method",
            "params_5d",
            "params_2d",
            "random_state",
            "cache_dir",
            "cache_suffix",
            "embedding_id",
            "show_progress",
        ],
    )


def test_fit_bertopic_signature_contract():
    signature = inspect.signature(tm.fit_bertopic)
    assert list(signature.parameters) == [
        "documents",
        "reduced_5d",
        "llm_provider",
        "llm_model",
        "llm_prompt",
        "pipeline_models",
        "parallel_models",
        "mmr_diversity",
        "llm_nr_docs",
        "llm_diversity",
        "llm_delay",
        "llm_max_new_tokens",
        "embedding_model_name",
        "keybert_model",
        "min_df",
        "clustering_method",
        "clustering_params",
        "top_n_words",
        "pos_spacy_model",
        "show_progress",
        "api_key",
        "openrouter_cost_mode",
        "cost_tracker",
    ]
    _assert_keyword_only(
        tm.fit_bertopic,
        [
            "llm_provider",
            "llm_model",
            "llm_prompt",
            "pipeline_models",
            "parallel_models",
            "mmr_diversity",
            "llm_nr_docs",
            "llm_diversity",
            "llm_delay",
            "llm_max_new_tokens",
            "embedding_model_name",
            "keybert_model",
            "min_df",
            "clustering_method",
            "clustering_params",
            "top_n_words",
            "pos_spacy_model",
            "show_progress",
            "api_key",
            "openrouter_cost_mode",
            "cost_tracker",
        ],
    )


def test_fit_toponymy_signature_contract():
    signature = inspect.signature(tm.fit_toponymy)
    assert list(signature.parameters) == [
        "documents",
        "embeddings",
        "clusterable_vectors",
        "backend",
        "layer_index",
        "llm_provider",
        "llm_model",
        "embedding_model",
        "api_key",
        "openrouter_api_base",
        "openrouter_cost_mode",
        "max_workers",
        "local_llm_max_new_tokens",
        "clusterer_params",
        "object_description",
        "corpus_description",
        "verbose",
        "cost_tracker",
    ]
    _assert_keyword_only(
        tm.fit_toponymy,
        [
            "backend",
            "layer_index",
            "llm_provider",
            "llm_model",
            "embedding_model",
            "api_key",
            "openrouter_api_base",
            "openrouter_cost_mode",
            "max_workers",
            "local_llm_max_new_tokens",
            "clusterer_params",
            "object_description",
            "corpus_description",
            "verbose",
            "cost_tracker",
        ],
    )


def test_reduce_outliers_signature_contract():
    signature = inspect.signature(tm.reduce_outliers)
    assert list(signature.parameters) == [
        "topic_model",
        "documents",
        "topics",
        "reduced_5d",
        "threshold",
        "llm_provider",
        "llm_model",
        "show_progress",
        "api_key",
        "openrouter_cost_mode",
        "cost_tracker",
    ]
    _assert_keyword_only(
        tm.reduce_outliers,
        [
            "threshold",
            "llm_provider",
            "llm_model",
            "show_progress",
            "api_key",
            "openrouter_cost_mode",
            "cost_tracker",
        ],
    )


def test_build_topic_dataframe_signature_contract():
    signature = inspect.signature(tm.build_topic_dataframe)
    assert list(signature.parameters) == [
        "df",
        "topic_model",
        "topics",
        "reduced_2d",
        "embeddings",
        "topic_info",
    ]
