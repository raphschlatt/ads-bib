from __future__ import annotations

import contextlib
from contextlib import contextmanager
import logging
import sys
import time
import types
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pytest

import ads_bib.topic_model as tm
from ads_bib._utils import logging as logging_utils
from ads_bib.topic_model import backends as tm_backends
from ads_bib.topic_model import embeddings as tm_embeddings
from ads_bib.topic_model import reduction as tm_reduction


class _FakeTopicModel:
    def __init__(self):
        self.vectorizer_model = object()
        self.ctfidf_model = object()
        self.representation_model = object()
        self.topic_representations_ = {
            -1: [("outlier", 1.0)],
            0: [("topic zero", 1.0)],
            1: [("topic one", 1.0)],
        }
        self.updated_topics = None
        self.assigned_labels = None

    def get_topic_info(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Topic": [-1, 0, 1],
                "Name": ["Outlier", "Topic Zero", "Topic One"],
                "Main": [["outlier"], ["zero"], ["one"]],
            }
        )

    def set_topic_labels(self, labels):
        self.assigned_labels = labels

    def reduce_outliers(self, documents, topics, strategy, embeddings, threshold):
        assert strategy == "embeddings"
        assert threshold == 0.8
        return [0 if t == -1 else t for t in topics]

    def update_topics(self, documents, topics, vectorizer_model, ctfidf_model, representation_model):
        self.updated_topics = np.array(topics)
        assert vectorizer_model is self.vectorizer_model
        assert ctfidf_model is self.ctfidf_model
        assert representation_model is self.representation_model


def test_build_topic_dataframe_uses_new_generic_columns():
    model = _FakeTopicModel()
    df = pd.DataFrame({"Bibcode": ["a", "b", "c"]})
    reduced_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    topics = np.array([0, 1, 0])
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)

    out = tm.build_topic_dataframe(df, model, topics, reduced_2d, embeddings)

    assert "embedding_2d_x" in out.columns
    assert "embedding_2d_y" in out.columns
    assert "topic_id" in out.columns
    assert "UMAP-1" not in out.columns
    assert "UMAP-2" not in out.columns
    assert "Cluster" not in out.columns
    assert out["topic_id"].tolist() == [0, 1, 0]
    assert out["Name"].tolist() == ["Topic Zero", "Topic One", "Topic Zero"]
    assert out["Main"].tolist() == ["zero", "one", "zero"]
    assert "full_embeddings" in out.columns
    assert model.assigned_labels[-1] == "Outlier Topic"


def test_build_topic_dataframe_accepts_external_topic_info():
    df = pd.DataFrame({"Bibcode": ["a", "b", "c"]})
    reduced_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    topics = np.array([0, 1, -1])
    topic_info = pd.DataFrame(
        {
            "Topic": [-1, 0, 1],
            "Name": ["Outlier Topic", "Alpha", "Beta"],
            "Main": ["noise", "alpha", "beta"],
        }
    )

    out = tm.build_topic_dataframe(
        df,
        topic_model=object(),
        topics=topics,
        reduced_2d=reduced_2d,
        topic_info=topic_info,
    )

    assert out["topic_id"].tolist() == [0, 1, -1]
    assert out["Name"].tolist() == ["Alpha", "Beta", "Outlier Topic"]
    assert out["Main"].tolist() == ["alpha", "beta", "noise"]


def test_reduce_outliers_refreshes_representations_and_logs(caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.topic_model")
    model = _FakeTopicModel()
    topics = np.array([-1, 1, -1], dtype=int)
    reduced_5d = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]], dtype=np.float32)

    new_topics = tm.reduce_outliers(
        model,
        documents=["a", "b", "c"],
        topics=topics,
        reduced_5d=reduced_5d,
        threshold=0.8,
    )

    assert "Refreshing topic representations after outlier reassignment" in caplog.text
    assert np.array_equal(new_topics, np.array([0, 1, 0]))
    assert np.array_equal(model.updated_topics, np.array([0, 1, 0]))


def test_reduce_outliers_tracks_post_outlier_llm_usage(monkeypatch):
    model = _FakeTopicModel()
    calls: dict = {}

    @contextmanager
    def _fake_track_litellm_usage(*, enabled: bool):
        assert enabled is True
        yield {"prompt_tokens": 12, "completion_tokens": 3, "call_records": []}

    def _fake_record_llm_usage(usage, **kwargs):
        calls["usage"] = usage
        calls["step"] = kwargs["step"]
        calls["llm_provider"] = kwargs["llm_provider"]

    monkeypatch.setattr(tm_backends, "_track_litellm_usage", _fake_track_litellm_usage)
    monkeypatch.setattr(tm_backends, "_record_llm_usage", _fake_record_llm_usage)

    tm.reduce_outliers(
        model,
        documents=["a", "b", "c"],
        topics=np.array([-1, 1, -1]),
        reduced_5d=np.ones((3, 5), dtype=np.float32),
        threshold=0.8,
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
        api_key="key",
        openrouter_cost_mode="hybrid",
        cost_tracker=object(),
    )

    assert calls["step"] == "llm_labeling_post_outliers"
    assert calls["llm_provider"] == "openrouter"
    assert calls["usage"]["prompt_tokens"] == 12


def test_create_llm_local_uses_transformers_text_generation_pipeline(monkeypatch):
    calls: dict = {}

    class _FakeTextGeneration:
        def __init__(self, generator, *, prompt, pipeline_kwargs):
            calls["generator"] = generator
            calls["prompt"] = prompt
            calls["pipeline_kwargs"] = pipeline_kwargs

    fake_representation = types.ModuleType("bertopic.representation")
    fake_representation.TextGeneration = _FakeTextGeneration
    monkeypatch.setitem(sys.modules, "bertopic.representation", fake_representation)

    fake_transformers = types.ModuleType("transformers")

    def _fake_pipeline(task, *, model, device_map, dtype=None, torch_dtype=None):
        calls["task"] = task
        calls["model"] = model
        calls["device_map"] = device_map
        calls["dtype"] = dtype
        calls["torch_dtype"] = torch_dtype
        return "fake-generator"

    fake_transformers.pipeline = _fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    llm = tm_backends._create_llm(
        provider="local",
        model="Qwen/Qwen3-0.6B",
        prompt="topic: <label>",
        nr_docs=8,
        diversity=0.2,
        delay=0.3,
        llm_max_new_tokens=64,
        api_key=None,
    )

    assert isinstance(llm, _FakeTextGeneration)
    assert calls["task"] == "text-generation"
    assert calls["model"] == "Qwen/Qwen3-0.6B"
    assert calls["device_map"] == "auto"
    assert calls["dtype"] == "auto"
    assert calls["torch_dtype"] is None
    assert calls["generator"] == "fake-generator"
    assert calls["prompt"] == "topic: <label>"
    assert calls["pipeline_kwargs"] == {"do_sample": False, "max_new_tokens": 64, "num_return_sequences": 1}


def test_create_llm_local_raises_actionable_error_for_unknown_arch(monkeypatch):
    fake_representation = types.ModuleType("bertopic.representation")
    fake_representation.TextGeneration = object
    monkeypatch.setitem(sys.modules, "bertopic.representation", fake_representation)

    fake_transformers = types.ModuleType("transformers")

    def _fake_pipeline(*args, **kwargs):
        del args, kwargs
        raise ValueError(
            "The checkpoint you are trying to load has model type `qwen3` "
            "but Transformers does not recognize this architecture."
        )

    fake_transformers.pipeline = _fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="Local topic labeling model 'Qwen/Qwen3-0.6B'"):
        tm_backends._create_llm(
            provider="local",
            model="Qwen/Qwen3-0.6B",
            prompt="topic: <label>",
            nr_docs=8,
            diversity=0.2,
            delay=0.3,
            llm_max_new_tokens=64,
            api_key=None,
        )


def test_create_llm_huggingface_api_normalizes_model_and_passes_api_key(monkeypatch):
    calls: dict = {}

    class _FakeLiteLLM:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

    fake_representation = types.ModuleType("bertopic.representation")
    fake_representation.LiteLLM = _FakeLiteLLM
    monkeypatch.setitem(sys.modules, "bertopic.representation", fake_representation)

    llm = tm_backends._create_llm(
        provider="huggingface_api",
        model="unsloth/Qwen2.5-72B-Instruct:featherless-ai",
        prompt="topic: <label>",
        nr_docs=8,
        diversity=0.2,
        delay=0.3,
        llm_max_new_tokens=64,
        api_key="hf-token",
    )

    assert isinstance(llm, _FakeLiteLLM)
    assert calls["kwargs"]["model"] == "huggingface/featherless-ai/unsloth/Qwen2.5-72B-Instruct"
    assert calls["kwargs"]["generator_kwargs"]["api_key"] == "hf-token"
    assert calls["kwargs"]["generator_kwargs"]["max_tokens"] == 64


class _FakeLayer:
    def __init__(self, labels):
        self.cluster_labels = np.asarray(labels, dtype=int)


class _FakeToponymyClusterer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeEVoCClusterer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeOpenAIEmbedder:
    def __init__(self, api_key, model, base_url):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def encode(self, texts, show_progress_bar=None, **kwargs):
        del show_progress_bar, kwargs
        return np.ones((len(texts), 3), dtype=np.float32)


class _FakeHuggingFaceNamer:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


class _FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, show_progress_bar=None, **kwargs):
        del show_progress_bar, kwargs
        return np.ones((len(texts), 3), dtype=np.float32)


class _FakeToponymyModel:
    def __init__(
        self,
        llm_wrapper,
        text_embedding_model,
        clusterer,
        object_description,
        corpus_description,
        verbose,
    ):
        self.llm_wrapper = llm_wrapper
        self.text_embedding_model = text_embedding_model
        self.clusterer = clusterer
        self.object_description = object_description
        self.corpus_description = corpus_description
        self.verbose = verbose
        self.cluster_layers_ = [_FakeLayer([-1, 0, 1]), _FakeLayer([0, 0, 1])]
        self.topic_names_ = [["Alpha", "Beta"], ["Macro Alpha", "Macro Beta"]]
        self.fitted = False

    def fit(self, objects, embedding_vectors, clusterable_vectors, **kwargs):
        del kwargs
        self.fitted = True
        self.objects = objects
        self.embedding_vectors = embedding_vectors
        self.clusterable_vectors = clusterable_vectors
        return self


def _install_fake_toponymy_modules(monkeypatch, *, include_hf_namer: bool = True):
    fake_toponymy = types.ModuleType("toponymy")
    fake_toponymy.Toponymy = _FakeToponymyModel
    fake_toponymy.ToponymyClusterer = _FakeToponymyClusterer

    fake_embedding_wrappers = types.ModuleType("toponymy.embedding_wrappers")
    fake_embedding_wrappers.OpenAIEmbedder = _FakeOpenAIEmbedder

    fake_clustering = types.ModuleType("toponymy.clustering")
    fake_clustering.EVoCClusterer = _FakeEVoCClusterer

    fake_llm_wrappers = types.ModuleType("toponymy.llm_wrappers")
    if include_hf_namer:
        fake_llm_wrappers.HuggingFaceNamer = _FakeHuggingFaceNamer

    monkeypatch.setitem(sys.modules, "toponymy", fake_toponymy)
    monkeypatch.setitem(sys.modules, "toponymy.embedding_wrappers", fake_embedding_wrappers)
    monkeypatch.setitem(sys.modules, "toponymy.clustering", fake_clustering)
    monkeypatch.setitem(sys.modules, "toponymy.llm_wrappers", fake_llm_wrappers)


def test_fit_toponymy_uses_backend_clusterer_and_tracks_step(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)
    calls: dict = {}

    def _fake_create_tracked_namer(*, model, api_key, base_url, max_workers):
        calls["namer"] = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "max_workers": max_workers,
        }
        usage = {"prompt_tokens": 22, "completion_tokens": 7, "call_records": []}
        return object(), usage

    def _fake_record_llm_usage(usage, **kwargs):
        calls["record"] = {"usage": usage, **kwargs}

    monkeypatch.setattr(tm_backends, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)
    monkeypatch.setattr(tm_backends, "_record_llm_usage", _fake_record_llm_usage)

    model, topics, topic_info = tm.fit_toponymy(
        documents=["d1", "d2", "d3"],
        embeddings=np.ones((3, 3), dtype=np.float32),
        clusterable_vectors=np.ones((3, 2), dtype=np.float32),
        backend="toponymy_evoc",
        layer_index=0,
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
        embedding_provider="openrouter",
        embedding_model="google/gemini-embedding-001",
        api_key="key",
        clusterer_params={"min_clusters": 4},
        cost_tracker=object(),
    )

    assert isinstance(model.clusterer, _FakeEVoCClusterer)
    assert model.clusterer.kwargs["min_clusters"] == 4
    assert topics.tolist() == [-1, 0, 1]
    assert topic_info["Topic"].tolist() == [-1, 0, 1]
    assert topic_info["Name"].tolist() == ["Outlier Topic", "Alpha", "Beta"]
    assert calls["record"]["step"] == "llm_labeling_toponymy_evoc"
    assert calls["record"]["llm_provider"] == "openrouter"
    assert calls["record"]["usage"]["prompt_tokens"] == 22
    assert calls["namer"]["max_workers"] == 5


def test_fit_toponymy_records_toponymy_embedding_costs_for_openrouter(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)
    calls: dict = {}

    class _Tracker:
        def __init__(self):
            self.entries = []

        def add(self, **kwargs):
            self.entries.append(kwargs)

    class _FakeOpenRouterEmbedder:
        def __init__(self, *, api_key, model, batch_size=64, max_workers=5, dtype=np.float32, api_base=None):
            del api_key, model, batch_size, dtype, api_base
            self.max_workers = max_workers
            self.usage = {"prompt_tokens": 0, "total_tokens": 0, "call_records": []}

        def encode(self, texts, verbose=None, show_progress_bar=None, **kwargs):
            del texts, verbose, show_progress_bar, kwargs
            return np.ones((0, 0), dtype=np.float32)

    def _fake_fit_outputs(**kwargs):
        embedder = kwargs["text_embedding_model"]
        embedder.usage = {
            "prompt_tokens": 17,
            "total_tokens": 17,
            "call_records": [{"generation_id": "gid-embed"}],
        }
        topic_info = pd.DataFrame({"Topic": [0], "Name": ["Alpha"], "Main": ["Alpha"]})
        return object(), np.array([0], dtype=int), topic_info

    def _fake_resolve_openrouter_costs(
        call_records,
        *,
        mode,
        api_key=None,
        max_workers=5,
        **kwargs,
    ):
        del kwargs
        calls["call_records"] = list(call_records)
        calls["api_key"] = api_key
        calls["mode"] = mode
        calls["max_workers"] = max_workers
        return 0.0042, {"total_cost_usd": 0.0042}

    monkeypatch.setattr(tm_backends, "OpenRouterEmbedder", _FakeOpenRouterEmbedder)
    monkeypatch.setattr(tm_backends, "_fit_and_extract_toponymy_outputs", _fake_fit_outputs)
    monkeypatch.setattr(tm_backends, "_record_llm_usage", lambda usage, **kwargs: None)
    monkeypatch.setattr(tm_backends, "resolve_openrouter_costs", _fake_resolve_openrouter_costs)

    tracker = _Tracker()
    tm.fit_toponymy(
        documents=["d1", "d2"],
        embeddings=np.ones((2, 3), dtype=np.float32),
        clusterable_vectors=np.ones((2, 2), dtype=np.float32),
        backend="toponymy_evoc",
        llm_provider="local",
        llm_model="local-llm",
        embedding_provider="openrouter",
        embedding_model="google/gemini-embedding-001",
        api_key="key",
        openrouter_cost_mode="hybrid",
        cost_tracker=tracker,
    )

    assert calls["api_key"] == "key"
    assert calls["mode"] == "hybrid"
    assert calls["call_records"] == [{"generation_id": "gid-embed"}]
    assert tracker.entries[0]["step"] == "toponymy_embeddings"
    assert tracker.entries[0]["provider"] == "openrouter"
    assert tracker.entries[0]["model"] == "google/gemini-embedding-001"
    assert tracker.entries[0]["prompt_tokens"] == 17
    assert tracker.entries[0]["total_tokens"] == 17
    assert tracker.entries[0]["cost_usd"] == 0.0042


def test_fit_toponymy_passes_max_workers_to_openrouter_models(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)
    calls: dict = {}

    class _FakeOpenRouterEmbedder:
        def __init__(self, *, api_key, model, batch_size=64, max_workers=5, dtype=np.float32, api_base=None):
            del api_key, model, batch_size, dtype, api_base
            calls["embedder_workers"] = max_workers
            self.max_workers = max_workers
            self.usage = {"prompt_tokens": 0, "total_tokens": 0, "call_records": []}

        def encode(self, texts, verbose=None, show_progress_bar=None, **kwargs):
            del texts, verbose, show_progress_bar, kwargs
            return np.ones((0, 0), dtype=np.float32)

    def _fake_create_tracked_namer(*, model, api_key, base_url, max_workers):
        del model, api_key, base_url
        calls["namer_workers"] = max_workers
        return object(), {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    monkeypatch.setattr(tm_backends, "OpenRouterEmbedder", _FakeOpenRouterEmbedder)
    monkeypatch.setattr(tm_backends, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)
    monkeypatch.setattr(tm_backends, "_record_llm_usage", lambda usage, **kwargs: None)

    tm.fit_toponymy(
        documents=["d1", "d2"],
        embeddings=np.ones((2, 3), dtype=np.float32),
        clusterable_vectors=np.ones((2, 2), dtype=np.float32),
        backend="toponymy",
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
        embedding_provider="openrouter",
        embedding_model="google/gemini-embedding-001",
        api_key="key",
        max_workers=9,
    )

    assert calls["namer_workers"] == 9
    assert calls["embedder_workers"] == 9


def test_fit_toponymy_does_not_record_toponymy_embedding_costs_for_local(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)

    class _Tracker:
        def __init__(self):
            self.entries = []

        def add(self, **kwargs):
            self.entries.append(kwargs)

    class _EmbedderWithUsage:
        def __init__(self):
            self.max_workers = 9
            self.usage = {
                "prompt_tokens": 33,
                "total_tokens": 33,
                "call_records": [{"generation_id": "gid-local"}],
            }

        def encode(self, texts, show_progress_bar=None, **kwargs):
            del show_progress_bar, kwargs
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr(tm_backends, "_build_toponymy_models", lambda **kwargs: (object(), None, _EmbedderWithUsage()))
    monkeypatch.setattr(
        tm_backends,
        "_fit_and_extract_toponymy_outputs",
        lambda **kwargs: (
            object(),
            np.array([0], dtype=int),
            pd.DataFrame({"Topic": [0], "Name": ["Alpha"], "Main": ["Alpha"]}),
        ),
    )
    monkeypatch.setattr(tm_backends, "_record_llm_usage", lambda usage, **kwargs: None)
    monkeypatch.setattr(tm_backends, "resolve_openrouter_costs", lambda call_records, **kwargs: (99.0, {"total_cost_usd": 99.0}))

    tracker = _Tracker()
    tm.fit_toponymy(
        documents=["d1"],
        embeddings=np.ones((1, 3), dtype=np.float32),
        clusterable_vectors=np.ones((1, 2), dtype=np.float32),
        backend="toponymy",
        llm_provider="local",
        llm_model="local-llm",
        embedding_provider="local",
        embedding_model="local-embedder",
        cost_tracker=tracker,
    )

    assert tracker.entries == []


def test_fit_toponymy_filters_unsupported_evoc_init_params(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)

    class _StrictEVoCClusterer:
        def __init__(self, min_clusters=5, min_samples=3):
            self.kwargs = {"min_clusters": min_clusters, "min_samples": min_samples}

    sys.modules["toponymy.clustering"].EVoCClusterer = _StrictEVoCClusterer

    def _fake_create_tracked_namer(*, model, api_key, base_url, max_workers):
        del model, api_key, base_url, max_workers
        return object(), {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    monkeypatch.setattr(tm_backends, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, topics, _ = tm.fit_toponymy(
            documents=["d1", "d2", "d3"],
            embeddings=np.ones((3, 3), dtype=np.float32),
            clusterable_vectors=np.ones((3, 2), dtype=np.float32),
            backend="toponymy_evoc",
            layer_index=0,
            llm_provider="openrouter",
            llm_model="google/gemini-3-flash-preview",
            embedding_provider="openrouter",
            api_key="key",
            clusterer_params={"min_clusters": 4, "cluster_levels": 2},
        )

    assert any(
        "Dropping unsupported EVoCClusterer parameter(s): cluster_levels" in str(w.message)
        for w in caught
    )
    assert isinstance(model.clusterer, _StrictEVoCClusterer)
    assert model.clusterer.kwargs == {"min_clusters": 4, "min_samples": 3}
    assert topics.tolist() == [-1, 0, 1]


def test_fit_toponymy_filters_unsupported_toponymy_init_params(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)

    class _StrictToponymyClusterer:
        def __init__(self, min_clusters=5, base_min_cluster_size=10):
            self.kwargs = {
                "min_clusters": min_clusters,
                "base_min_cluster_size": base_min_cluster_size,
            }

    sys.modules["toponymy"].ToponymyClusterer = _StrictToponymyClusterer

    def _fake_create_tracked_namer(*, model, api_key, base_url, max_workers):
        del model, api_key, base_url, max_workers
        return object(), {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    monkeypatch.setattr(tm_backends, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, topics, _ = tm.fit_toponymy(
            documents=["d1", "d2", "d3"],
            embeddings=np.ones((3, 3), dtype=np.float32),
            clusterable_vectors=np.ones((3, 2), dtype=np.float32),
            backend="toponymy",
            layer_index=0,
            llm_provider="openrouter",
            llm_model="google/gemini-3-flash-preview",
            embedding_provider="openrouter",
            api_key="key",
            clusterer_params={
                "min_clusters": 7,
                "base_min_cluster_size": 11,
                "cluster_levels": 3,
            },
        )

    assert any(
        "Dropping unsupported ToponymyClusterer parameter(s): cluster_levels" in str(w.message)
        for w in caught
    )
    assert isinstance(model.clusterer, _StrictToponymyClusterer)
    assert model.clusterer.kwargs == {"min_clusters": 7, "base_min_cluster_size": 11}
    assert topics.tolist() == [-1, 0, 1]


def test_fit_toponymy_supports_local_llm_provider(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    calls: dict = {}

    def _fake_record_llm_usage(usage, **kwargs):
        calls["usage"] = usage
        calls["kwargs"] = kwargs

    monkeypatch.setattr(tm_backends, "_record_llm_usage", _fake_record_llm_usage)

    model, topics, _ = tm.fit_toponymy(
        documents=["d1", "d2", "d3"],
        embeddings=np.ones((3, 3), dtype=np.float32),
        clusterable_vectors=np.ones((3, 2), dtype=np.float32),
        backend="toponymy",
        layer_index=0,
        llm_provider="local",
        llm_model="local-llm",
        embedding_provider="local",
        embedding_model="local-embedder",
        local_llm_max_new_tokens=77,
    )

    assert isinstance(model.llm_wrapper, _FakeHuggingFaceNamer)
    assert model.llm_wrapper.model == "local-llm"
    assert model.llm_wrapper._local_max_new_tokens == 77
    assert model.llm_wrapper._max_tokens(128) == 77
    assert isinstance(model.text_embedding_model, _FakeSentenceTransformer)
    assert model.text_embedding_model.model_name == "local-embedder"
    assert topics.tolist() == [-1, 0, 1]
    assert calls["usage"] is None
    assert calls["kwargs"]["llm_provider"] == "local"


def test_fit_toponymy_supports_gguf_embedding_provider_independent_of_local_llm(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)
    calls: dict = {}

    class _FakeGGUFEmbedder:
        def __init__(self, *, model, batch_size=64, max_workers=5, dtype=np.float32, n_ctx=4096, pooling="cls"):
            del batch_size, dtype, n_ctx, pooling
            self.model = model
            self.max_workers = max_workers
            calls["embedding_model"] = model
            calls["embedding_workers"] = max_workers

        def encode(self, texts, verbose=None, show_progress_bar=None, **kwargs):
            del texts, verbose, show_progress_bar, kwargs
            return np.ones((0, 0), dtype=np.float32)

    def _fake_record_llm_usage(usage, **kwargs):
        calls["usage"] = usage
        calls["kwargs"] = kwargs

    monkeypatch.setattr(tm_backends, "GGUFEmbedder", _FakeGGUFEmbedder)
    monkeypatch.setattr(tm_backends, "_record_llm_usage", _fake_record_llm_usage)

    model, topics, _ = tm.fit_toponymy(
        documents=["d1", "d2", "d3"],
        embeddings=np.ones((3, 3), dtype=np.float32),
        clusterable_vectors=np.ones((3, 2), dtype=np.float32),
        backend="toponymy",
        layer_index=0,
        llm_provider="local",
        llm_model="local-llm",
        embedding_provider="gguf",
        embedding_model="Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf",
        max_workers=7,
    )

    assert isinstance(model.llm_wrapper, _FakeHuggingFaceNamer)
    assert isinstance(model.text_embedding_model, _FakeGGUFEmbedder)
    assert model.text_embedding_model.model == "Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf"
    assert calls["embedding_workers"] == 7
    assert topics.tolist() == [-1, 0, 1]
    assert calls["usage"] is None
    assert calls["kwargs"]["llm_provider"] == "local"


@pytest.mark.slow
def test_fit_toponymy_local_requires_hf_dependencies(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch, include_hf_namer=False)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    try:
        tm.fit_toponymy(
            documents=["d1"],
            embeddings=np.ones((1, 3), dtype=np.float32),
            clusterable_vectors=np.ones((1, 2), dtype=np.float32),
            backend="toponymy",
            llm_provider="local",
            llm_model="local-llm",
            embedding_provider="local",
            embedding_model="local-embedder",
        )
        assert False, "Expected ImportError for missing local Toponymy dependencies"
    except ImportError as exc:
        assert "llm_provider='local' requires optional dependencies" in str(exc)


def test_fit_toponymy_validates_layer_index(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)

    def _fake_create_tracked_namer(*, model, api_key, base_url, max_workers):
        del model, api_key, base_url, max_workers
        return object(), {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    monkeypatch.setattr(tm_backends, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)

    try:
        tm.fit_toponymy(
            documents=["d1", "d2", "d3"],
            embeddings=np.ones((3, 3), dtype=np.float32),
            clusterable_vectors=np.ones((3, 2), dtype=np.float32),
            backend="toponymy",
            layer_index=9,
            llm_provider="openrouter",
            llm_model="google/gemini-3-flash-preview",
            embedding_provider="openrouter",
            api_key="key",
        )
        assert False, "Expected ValueError for invalid layer_index"
    except ValueError as exc:
        assert "layer_index" in str(exc)


def test_fit_toponymy_rejects_invalid_backend():
    try:
        tm.fit_toponymy(
            documents=["d1"],
            embeddings=np.ones((1, 3), dtype=np.float32),
            clusterable_vectors=np.ones((1, 2), dtype=np.float32),
            backend="invalid",
            llm_provider="openrouter",
            api_key="key",
        )
        assert False, "Expected ValueError for invalid backend"
    except ValueError as exc:
        assert "Expected 'toponymy' or 'toponymy_evoc'" in str(exc)


def test_fit_toponymy_rejects_huggingface_api_embedding_provider():
    with pytest.raises(ValueError, match="Invalid embedding_provider 'huggingface_api'"):
        tm.fit_toponymy(
            documents=["d1"],
            embeddings=np.ones((1, 3), dtype=np.float32),
            clusterable_vectors=np.ones((1, 2), dtype=np.float32),
            backend="toponymy",
            llm_provider="local",
            llm_model="local-llm",
            embedding_provider="huggingface_api",
            embedding_model="hf-api-embedder",
        )


def test_record_llm_usage_fetches_openrouter_cost(monkeypatch):
    calls: dict = {}

    def _fake_resolve_openrouter_costs(call_records, *, mode, api_key=None, max_workers=5, **kwargs):
        del max_workers, kwargs
        calls["call_records"] = call_records
        calls["api_key"] = api_key
        calls["mode"] = mode
        return 0.0123, {"total_cost_usd": 0.0123}

    class _Tracker:
        def __init__(self):
            self.entries = []

        def add(self, **kwargs):
            self.entries.append(kwargs)

    monkeypatch.setattr(tm_backends, "resolve_openrouter_costs", _fake_resolve_openrouter_costs)
    tracker = _Tracker()

    tm_backends._record_llm_usage(
        {"prompt_tokens": 9, "completion_tokens": 3, "call_records": [{"generation_id": "gen_1"}]},
        step="llm_labeling_toponymy",
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
        api_key="key",
        openrouter_cost_mode="hybrid",
        cost_tracker=tracker,
    )

    assert calls["api_key"] == "key"
    assert calls["mode"] == "hybrid"
    assert calls["call_records"][0]["generation_id"] == "gen_1"
    assert tracker.entries[0]["cost_usd"] == 0.0123


def test_patch_clusterer_for_toponymy_kwargs_drops_unsupported_kwargs():
    class _StrictClusterer:
        def __init__(self):
            self.seen = None

        def fit_predict(
            self,
            clusterable_vectors,
            embedding_vectors,
            layer_class,
            verbose=None,
            show_progress_bar=None,
        ):
            del clusterable_vectors, embedding_vectors, layer_class
            self.seen = (verbose, show_progress_bar)
            return "ok"

    clusterer = _StrictClusterer()
    tm_backends._patch_clusterer_for_toponymy_kwargs(clusterer)

    out = clusterer.fit_predict(
        np.ones((3, 2), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32),
        object,
        verbose=True,
        show_progress_bar=False,
        exemplar_delimiters=["x"],
        prompt_format="system_user",
    )

    assert out == "ok"
    assert clusterer.seen == (True, False)


def test_compute_embeddings_passes_max_workers_to_openrouter(monkeypatch):
    calls: dict = {}

    class _FakeOpenRouterEmbedder:
        def __init__(self, *, api_key, model, batch_size=64, max_workers=5, dtype=np.float32, api_base=None):
            del api_key, model, batch_size, dtype, api_base
            calls["max_workers"] = max_workers
            self.usage = {"prompt_tokens": 0, "total_tokens": 0, "call_records": []}

        def encode(self, texts, verbose=None, show_progress_bar=None, **kwargs):
            del verbose, kwargs
            calls["show_progress_bar"] = show_progress_bar
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(tm_embeddings, "OpenRouterEmbedder", _FakeOpenRouterEmbedder)

    emb = tm.compute_embeddings(
        ["d1", "d2", "d3"],
        provider="openrouter",
        model="google/gemini-embedding-001",
        max_workers=7,
        api_key="key",
    )

    assert emb.shape == (3, 2)
    assert calls["max_workers"] == 7
    assert calls["show_progress_bar"] is True


def test_compute_embeddings_passes_progress_callback_to_openrouter(monkeypatch):
    calls: dict[str, object] = {}
    progress_updates: list[int] = []

    class _FakeOpenRouterEmbedder:
        def __init__(self, *, api_key, model, batch_size=64, max_workers=5, dtype=np.float32, api_base=None):
            del api_key, model, batch_size, max_workers, dtype, api_base
            self.usage = {"prompt_tokens": 0, "total_tokens": 0, "call_records": []}

        def encode(self, texts, verbose=None, show_progress_bar=None, progress_callback=None, **kwargs):
            del texts, verbose, kwargs
            calls["show_progress_bar"] = show_progress_bar
            calls["progress_callback"] = progress_callback
            progress_callback(2)
            progress_callback(1)
            return np.ones((3, 2), dtype=np.float32)

    monkeypatch.setattr(tm_embeddings, "OpenRouterEmbedder", _FakeOpenRouterEmbedder)

    emb = tm.compute_embeddings(
        ["d1", "d2", "d3"],
        provider="openrouter",
        model="google/gemini-embedding-001",
        api_key="key",
        progress_callback=progress_updates.append,
    )

    assert emb.shape == (3, 2)
    assert calls["show_progress_bar"] is False
    assert callable(calls["progress_callback"])
    assert progress_updates == [2, 1]


def test_compute_embeddings_validates_gguf_provider_import_path(monkeypatch):
    calls: dict[str, Any] = {}

    def _fake_validate_provider(provider, **kwargs):
        calls["provider"] = provider
        calls["valid"] = kwargs["valid"]
        calls["requires_import"] = kwargs["requires_import"]

    class _FakeGGUFEmbedder:
        def __init__(self, *, model, batch_size=64, max_workers=5, dtype=np.float32, n_ctx=4096, pooling="cls"):
            del model, batch_size, max_workers, dtype, n_ctx, pooling

        def encode(self, texts, verbose=None, show_progress_bar=None, progress_callback=None, **kwargs):
            del verbose, show_progress_bar, progress_callback, kwargs
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(tm_embeddings, "validate_provider", _fake_validate_provider)
    monkeypatch.setattr(tm_embeddings, "GGUFEmbedder", _FakeGGUFEmbedder)

    emb = tm.compute_embeddings(
        ["d1", "d2"],
        provider="gguf",
        model="Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf",
    )

    assert emb.shape == (2, 2)
    assert calls["provider"] == "gguf"
    assert "gguf" in calls["valid"]
    assert calls["requires_import"]["gguf"] == "llama_cpp"


def test_gguf_embedder_batches_keep_order_and_progress(monkeypatch):
    calls: list[dict[str, Any]] = []
    progress_updates: list[int] = []

    monkeypatch.setattr(tm_embeddings, "resolve_gguf_model", lambda model: f"/fake/{model.split(':')[-1]}")
    monkeypatch.setattr(tm_embeddings, "recommended_gguf_thread_settings", lambda cpu_total=None: (4, 8))

    def _fake_embed_gguf(
        texts,
        *,
        model_path,
        n_ctx,
        n_threads,
        n_threads_batch,
        normalize,
        truncate,
        pooling="cls",
    ):
        calls.append(
            {
                "texts": list(texts),
                "model_path": model_path,
                "n_ctx": n_ctx,
                "n_threads": n_threads,
                "n_threads_batch": n_threads_batch,
                "normalize": normalize,
                "truncate": truncate,
                "pooling": pooling,
            }
        )
        return np.array(
            [[float(int(text.split("_")[1])), 1.0] for text in texts],
            dtype=np.float32,
        )

    monkeypatch.setattr(tm_embeddings, "embed_gguf", _fake_embed_gguf)

    embedder = tm_embeddings.GGUFEmbedder(
        model="Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf",
        batch_size=2,
        max_workers=9,
        dtype=np.float16,
    )
    emb = embedder.encode(
        [f"doc_{i}" for i in range(5)],
        show_progress_bar=False,
        progress_callback=progress_updates.append,
    )

    assert emb.shape == (5, 2)
    assert emb.dtype == np.float16
    assert emb[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert progress_updates == [2, 2, 1]
    assert calls[0]["model_path"] == "/fake/Qwen3-Embedding-0.6B-Q8_0.gguf"
    assert calls[0]["n_threads"] == 4
    assert calls[0]["n_threads_batch"] == 8
    assert calls[0]["normalize"] is True
    assert calls[0]["truncate"] is True


def test_ensure_embedding_model_loads_llama_with_cls_pooling(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.LLAMA_POOLING_TYPE_CLS = 2
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    monkeypatch.setattr(gguf_mod, "_embedding_model", None)
    monkeypatch.setattr(gguf_mod, "_embedding_model_path", None)
    monkeypatch.setattr(gguf_mod, "_embedding_model_config", None)

    calls: dict[str, Any] = {}
    fake_model = object()

    def _fake_load_llama(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return fake_model

    safe_calls: list[Any] = []
    monkeypatch.setattr(gguf_mod, "_load_llama", _fake_load_llama)
    monkeypatch.setattr(gguf_mod, "_make_llama_jupyter_safe", lambda llm: safe_calls.append(llm))

    model = gguf_mod._ensure_embedding_model(
        model_path="/fake/qwen-embed.gguf",
        n_ctx=4096,
        n_threads=4,
        n_threads_batch=8,
        pooling="cls",
    )

    assert model is fake_model
    assert calls["path"] == "/fake/qwen-embed.gguf"
    assert calls["kwargs"]["embedding"] is True
    assert calls["kwargs"]["pooling_type"] == 2
    assert safe_calls == [fake_model]


def test_load_llama_omits_optional_none_kwargs(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    calls: dict[str, Any] = {}

    class _FakeLlama:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    monkeypatch.setattr(gguf_mod, "safe_stdio", contextlib.nullcontext)

    model = gguf_mod._load_llama("/fake/model.gguf", n_ctx=512)

    assert isinstance(model, _FakeLlama)
    assert "n_batch" not in calls["kwargs"]
    assert "n_threads" not in calls["kwargs"]
    assert "n_threads_batch" not in calls["kwargs"]
    assert "pooling_type" not in calls["kwargs"]


def test_temporarily_raise_logger_level_restores_previous_level():
    logger = logging.getLogger("transformers.utils.loading_report")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        with logging_utils.temporarily_raise_logger_level(
            "transformers.utils.loading_report",
            level=logging.ERROR,
        ):
            assert logger.level == logging.ERROR
        assert logger.level == logging.INFO
    finally:
        logger.setLevel(previous_level)


def test_load_llama_wraps_unknown_qwen35_architecture_with_actionable_error(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    class _FakeLlama:
        def __init__(self, **kwargs):
            del kwargs
            raise ValueError("Failed to load model from file: /fake/model.gguf (unknown model architecture: 'qwen35')")

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.Llama = _FakeLlama
    fake_llama_cpp.__version__ = "0.3.16"
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    monkeypatch.setattr(gguf_mod, "safe_stdio", contextlib.nullcontext)

    with pytest.raises(RuntimeError, match="supports its architecture"):
        gguf_mod._load_llama("/fake/model.gguf", n_ctx=512)


def test_ensure_embedding_model_loads_llama_with_last_pooling(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.LLAMA_POOLING_TYPE_CLS = 2
    fake_llama_cpp.LLAMA_POOLING_TYPE_LAST = 3
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    monkeypatch.setattr(gguf_mod, "_embedding_model", None)
    monkeypatch.setattr(gguf_mod, "_embedding_model_path", None)
    monkeypatch.setattr(gguf_mod, "_embedding_model_config", None)

    calls: dict[str, Any] = {}
    fake_model = object()

    def _fake_load_llama(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(gguf_mod, "_load_llama", _fake_load_llama)
    monkeypatch.setattr(gguf_mod, "_make_llama_jupyter_safe", lambda llm: None)

    model = gguf_mod._ensure_embedding_model(
        model_path="/fake/qwen-embed.gguf",
        n_ctx=4096,
        n_threads=4,
        n_threads_batch=8,
        pooling="last",
    )

    assert model is fake_model
    assert calls["kwargs"]["pooling_type"] == 3


def test_ensure_embedding_model_cache_invalidates_on_pooling_change(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.LLAMA_POOLING_TYPE_CLS = 2
    fake_llama_cpp.LLAMA_POOLING_TYPE_LAST = 3
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    monkeypatch.setattr(gguf_mod, "_embedding_model", None)
    monkeypatch.setattr(gguf_mod, "_embedding_model_path", None)
    monkeypatch.setattr(gguf_mod, "_embedding_model_config", None)

    load_calls: list[dict] = []

    def _fake_load_llama(path, **kwargs):
        load_calls.append({"path": path, **kwargs})
        return object()

    monkeypatch.setattr(gguf_mod, "_load_llama", _fake_load_llama)
    monkeypatch.setattr(gguf_mod, "_make_llama_jupyter_safe", lambda llm: None)

    gguf_mod._ensure_embedding_model(
        model_path="/fake/model.gguf", n_ctx=4096, n_threads=4, n_threads_batch=8, pooling="cls",
    )
    gguf_mod._ensure_embedding_model(
        model_path="/fake/model.gguf", n_ctx=4096, n_threads=4, n_threads_batch=8, pooling="last",
    )
    assert len(load_calls) == 2
    assert load_calls[0]["pooling_type"] == 2
    assert load_calls[1]["pooling_type"] == 3


def test_ensure_embedding_model_rejects_invalid_pooling(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    monkeypatch.setattr(gguf_mod, "_embedding_model", None)
    monkeypatch.setattr(gguf_mod, "_embedding_model_path", None)
    monkeypatch.setattr(gguf_mod, "_embedding_model_config", None)

    with pytest.raises(ValueError, match="Unknown GGUF pooling type"):
        gguf_mod._ensure_embedding_model(
            model_path="/fake/model.gguf", n_ctx=4096, n_threads=4, n_threads_batch=8, pooling="bogus",
        )


def test_embed_gguf_normalizes_before_return(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    calls: dict[str, Any] = {}

    _vectors = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    embed_calls: list[str] = []

    class _FakeEmbeddingModel:
        def embed(self, text, normalize=False, truncate=True):
            embed_calls.append(text)
            calls["normalize"] = normalize
            calls["truncate"] = truncate
            return _vectors[text]

    monkeypatch.setattr(gguf_mod, "_ensure_embedding_model", lambda **kwargs: _FakeEmbeddingModel())

    emb = gguf_mod.embed_gguf(
        ["a", "b"],
        model_path="/fake/qwen-embed.gguf",
    )

    assert emb.shape == (2, 2)
    assert embed_calls == ["a", "b"]
    assert calls["normalize"] is True
    assert calls["truncate"] is True


def test_embed_local_raises_actionable_error_for_unknown_arch(monkeypatch):
    fake_sentence_transformers = types.ModuleType("sentence_transformers")

    class _BrokenSentenceTransformer:
        def __init__(self, model):
            del model
            raise ValueError(
                "The checkpoint you are trying to load has model type `gemma3_text` "
                "but Transformers does not recognize this architecture."
            )

    fake_sentence_transformers.SentenceTransformer = _BrokenSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    with pytest.raises(RuntimeError, match="Local embeddings model 'google/embeddinggemma-300m'"):
        tm_embeddings._embed_local(
            ["doc-a"],
            model="google/embeddinggemma-300m",
            batch_size=8,
            dtype=np.float32,
        )


def test_embed_local_raises_actionable_error_for_torch_runtime_requirement(monkeypatch):
    fake_sentence_transformers = types.ModuleType("sentence_transformers")

    class _BrokenSentenceTransformer:
        def __init__(self, model):
            del model

        def encode(self, documents, show_progress_bar=None, batch_size=64):
            del documents, show_progress_bar, batch_size
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")

    fake_sentence_transformers.SentenceTransformer = _BrokenSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    with pytest.raises(RuntimeError, match="requires torch>=2.6"):
        tm_embeddings._embed_local(
            ["doc-a"],
            model="google/embeddinggemma-300m",
            batch_size=8,
            dtype=np.float32,
        )


def test_openrouter_embedder_parallel_keeps_document_order(monkeypatch):
    fake_litellm = types.ModuleType("litellm")

    def _fake_embedding(model, input, api_key):
        del model, api_key
        first_idx = int(input[0].split("_")[1])
        # Force out-of-order completion to validate deterministic reconstruction.
        if first_idx == 0:
            time.sleep(0.05)
        else:
            time.sleep(0.01)
        return {
            "id": f"gen_{first_idx}",
            "usage": {"prompt_tokens": len(input), "total_tokens": len(input)},
            "data": [{"embedding": [float(int(text.split('_')[1]))]} for text in input],
        }

    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    fake_litellm.embedding = _fake_embedding
    monkeypatch.setattr(tm_embeddings, "extract_usage_stats", lambda resp: resp["usage"])
    monkeypatch.setattr(tm_embeddings, "extract_generation_id", lambda resp: resp["id"])
    monkeypatch.setattr(tm_embeddings, "extract_response_cost", lambda **kwargs: None)

    embedder = tm_embeddings.OpenRouterEmbedder(
        api_key="key",
        model="google/gemini-embedding-001",
        batch_size=2,
        dtype=np.float32,
        max_workers=3,
    )
    emb = embedder.encode([f"doc_{i}" for i in range(6)], show_progress_bar=False)
    usage = embedder.usage

    assert emb.shape == (6, 1)
    assert emb[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert usage["prompt_tokens"] == 6
    assert usage["total_tokens"] == 6
    assert [r["generation_id"] for r in usage["call_records"]] == ["gen_0", "gen_2", "gen_4"]


def test_openrouter_embedder_retries_when_data_is_none(monkeypatch):
    fake_litellm = types.ModuleType("litellm")
    calls = {"attempts": 0}

    def _fake_embedding(model, input, api_key):
        del model, api_key
        calls["attempts"] += 1
        if calls["attempts"] == 1:
            return {"id": "gen_fail", "usage": {"prompt_tokens": 0, "total_tokens": 0}, "data": None}
        return {
            "id": "gen_ok",
            "usage": {"prompt_tokens": len(input), "total_tokens": len(input)},
            "data": [{"embedding": [1.0, 2.0]} for _ in input],
        }

    def _fake_retry_call(func, *, max_retries, delay, backoff, on_retry=None):
        del delay, backoff
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                if on_retry is not None:
                    on_retry(attempt + 1, max_retries, 0.0, exc)
        raise RuntimeError("retry_call exhausted unexpectedly.")

    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    fake_litellm.embedding = _fake_embedding
    monkeypatch.setattr(tm_embeddings, "retry_call", _fake_retry_call)
    monkeypatch.setattr(tm_embeddings, "extract_usage_stats", lambda resp: resp["usage"])
    monkeypatch.setattr(tm_embeddings, "extract_generation_id", lambda resp: resp["id"])
    monkeypatch.setattr(tm_embeddings, "extract_response_cost", lambda **kwargs: None)

    embedder = tm_embeddings.OpenRouterEmbedder(
        api_key="key",
        model="google/gemini-embedding-001",
        batch_size=2,
        max_workers=1,
        dtype=np.float32,
    )
    emb = embedder.encode(["doc_a", "doc_b"], show_progress_bar=False)

    assert calls["attempts"] == 2
    assert emb.shape == (2, 2)
    assert embedder.usage["total_tokens"] == 2
    assert [r["generation_id"] for r in embedder.usage["call_records"]] == ["gen_ok"]


def test_openrouter_embedder_prealloc_keeps_dtype(monkeypatch):
    fake_litellm = types.ModuleType("litellm")

    def _fake_embedding(model, input, api_key):
        del model, api_key
        return {
            "id": f"gen_{input[0]}",
            "usage": {"prompt_tokens": len(input), "total_tokens": len(input)},
            "data": [{"embedding": [float(int(text.split('_')[1])), 10.0]} for text in input],
        }

    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    fake_litellm.embedding = _fake_embedding
    monkeypatch.setattr(tm_embeddings, "extract_usage_stats", lambda resp: resp["usage"])
    monkeypatch.setattr(tm_embeddings, "extract_generation_id", lambda resp: resp["id"])
    monkeypatch.setattr(tm_embeddings, "extract_response_cost", lambda **kwargs: None)

    embedder = tm_embeddings.OpenRouterEmbedder(
        api_key="key",
        model="google/gemini-embedding-001",
        batch_size=2,
        max_workers=2,
        dtype=np.float16,
    )
    emb = embedder.encode([f"doc_{i}" for i in range(5)], show_progress_bar=False)

    assert emb.shape == (5, 2)
    assert emb.dtype == np.float16
    assert emb[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_embed_huggingface_api_prealloc_keeps_order(monkeypatch):
    calls: dict[str, Any] = {}

    class _Client:
        async def feature_extraction(self, input, *, model):
            calls.setdefault("batches", []).append((model, list(input)))
            return np.asarray(
                [[float(int(text.split("_")[1])), 1.0] for text in input],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        tm_embeddings,
        "_create_huggingface_async_client",
        lambda *, model, api_key: (_Client(), "test-model"),
    )

    emb = tm_embeddings._embed_huggingface_api(
        [f"doc_{i}" for i in range(5)],
        model="test-model",
        batch_size=2,
        dtype=np.float32,
        max_workers=2,
        show_progress=False,
        api_key="hf-token",
    )

    assert emb.shape == (5, 2)
    assert emb.dtype == np.float32
    assert emb[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert calls["batches"] == [
        ("test-model", ["doc_0", "doc_1"]),
        ("test-model", ["doc_2", "doc_3"]),
        ("test-model", ["doc_4"]),
    ]


def test_compute_embeddings_defaults_to_float16(monkeypatch):
    calls: dict[str, Any] = {}

    def _fake_embed_local(documents, model_name, batch_size, dtype):
        del model_name, batch_size
        calls["dtype"] = np.dtype(dtype)
        return np.ones((len(documents), 2), dtype=np.dtype(dtype))

    monkeypatch.setattr(tm_embeddings, "_embed_local", _fake_embed_local)

    emb = tm.compute_embeddings(
        ["doc-a", "doc-b"],
        provider="local",
        model="local/test-model",
    )

    assert calls["dtype"] == np.dtype(np.float16)
    assert emb.dtype == np.float16


def test_compute_embeddings_recomputes_on_cache_n_docs_mismatch(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.topic_model")
    model = "local/test-model"
    cache_file = tmp_path / "embeddings_local_local_test-model.npz"
    np.savez_compressed(
        cache_file,
        embeddings=np.zeros((1, 3), dtype=np.float32),
        n_docs=1,
        doc_fingerprint="stale",
        provider="local",
        model=model,
    )
    calls = {"n": 0}

    def _fake_embed_local(documents, model_name, batch_size, dtype):
        del model_name, batch_size, dtype
        calls["n"] += 1
        return np.ones((len(documents), 3), dtype=np.float32)

    monkeypatch.setattr(tm_embeddings, "_embed_local", _fake_embed_local)
    out = tm.compute_embeddings(
        ["doc-a", "doc-b"],
        provider="local",
        model=model,
        cache_dir=tmp_path,
    )

    assert out.shape == (2, 3)
    assert calls["n"] == 1
    assert "Embedding cache mismatch" in caplog.text


def test_compute_embeddings_recomputes_on_cache_fingerprint_mismatch(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.topic_model")
    model = "local/test-model"
    cache_file = tmp_path / "embeddings_local_local_test-model.npz"
    np.savez_compressed(
        cache_file,
        embeddings=np.zeros((2, 2), dtype=np.float32),
        n_docs=2,
        doc_fingerprint="invalid-fingerprint",
        provider="local",
        model=model,
    )
    calls = {"n": 0}

    def _fake_embed_local(documents, model_name, batch_size, dtype):
        del model_name, batch_size, dtype
        calls["n"] += 1
        return np.full((len(documents), 2), fill_value=7.0, dtype=np.float32)

    monkeypatch.setattr(tm_embeddings, "_embed_local", _fake_embed_local)
    out = tm.compute_embeddings(
        ["doc-a", "doc-b"],
        provider="local",
        model=model,
        cache_dir=tmp_path,
    )

    assert out.shape == (2, 2)
    assert float(out[0, 0]) == 7.0
    assert calls["n"] == 1
    assert "Embedding cache mismatch" in caplog.text


def test_compute_embeddings_casts_valid_cache_to_requested_dtype(monkeypatch, tmp_path):
    docs = ["doc-a", "doc-b"]
    model = "local/test-model"
    cache_file = tmp_path / "embeddings_local_local_test-model.npz"
    np.savez_compressed(
        cache_file,
        embeddings=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        n_docs=len(docs),
        doc_fingerprint=tm_embeddings._documents_fingerprint(docs),
        provider="local",
        model=model,
    )

    def _fail_embed_local(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Expected valid cache reuse without recompute.")

    monkeypatch.setattr(tm_embeddings, "_embed_local", _fail_embed_local)

    out = tm.compute_embeddings(
        docs,
        provider="local",
        model=model,
        cache_dir=tmp_path,
        dtype=np.float16,
    )

    assert out.dtype == np.float16
    assert np.allclose(out, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16))


def test_compute_embeddings_paid_provider_memory_preflight_fails_fast(monkeypatch):
    def _fake_validate_provider(*args, **kwargs):
        del args, kwargs

    class _FakeOpenRouterEmbedder:
        def __init__(self, **kwargs):
            del kwargs
            raise AssertionError("OpenRouter embedder must not be instantiated when preflight fails.")

    monkeypatch.setattr(tm_embeddings, "validate_provider", _fake_validate_provider)
    monkeypatch.setattr(tm_embeddings, "_available_memory_bytes", lambda: 1024 * 1024)
    monkeypatch.setattr(tm_embeddings, "OpenRouterEmbedder", _FakeOpenRouterEmbedder)

    docs = [f"doc_{i}" for i in range(2000)]
    with pytest.raises(MemoryError, match="SAMPLE_SIZE"):
        tm.compute_embeddings(
            docs,
            provider="openrouter",
            model="google/gemini-embedding-001",
            api_key="key",
        )


def test_reduce_recomputes_on_params_hash_mismatch(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.topic_model")
    embeddings = np.ones((4, 3), dtype=np.float32)
    cache_path = tmp_path / "reduced_unit.npz"
    np.savez_compressed(
        cache_path,
        reduced=np.zeros((4, 2), dtype=np.float32),
        n_docs=4,
        embedding_fingerprint=tm_reduction._array_fingerprint(embeddings),
        method="pacmap",
        n_components=2,
        random_state=42,
        params_hash="stale",
    )

    calls = {"n": 0}

    class _FakePaCMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, values):
            del values
            calls["n"] += 1
            return np.full((4, 2), fill_value=9.0, dtype=np.float32)

    fake_pacmap = types.ModuleType("pacmap")
    fake_pacmap.PaCMAP = _FakePaCMAP
    monkeypatch.setitem(sys.modules, "pacmap", fake_pacmap)

    reduced = tm_reduction._reduce_with_cache(
        embeddings=embeddings,
        n_components=2,
        method="pacmap",
        params={"n_neighbors": 15, "metric": "euclidean"},
        random_state=42,
        cache_dir=tmp_path,
        name="unit",
    )

    assert calls["n"] == 1
    assert np.allclose(reduced, 9.0)
    assert "Reduction cache mismatch" in caplog.text


def test_reduce_recomputes_on_embedding_fingerprint_mismatch(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.topic_model")
    embeddings = np.ones((3, 2), dtype=np.float32)
    params = {"n_neighbors": 12}
    params_hash = tm_reduction._stable_hash(
        {
            "method": "pacmap",
            "n_components": 2,
            "random_state": 42,
            "params": params,
        }
    )
    cache_path = tmp_path / "reduced_fp.npz"
    np.savez_compressed(
        cache_path,
        reduced=np.zeros((3, 2), dtype=np.float32),
        n_docs=3,
        embedding_fingerprint="wrong",
        method="pacmap",
        n_components=2,
        random_state=42,
        params_hash=params_hash,
    )
    calls = {"n": 0}

    class _FakePaCMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, values):
            del values
            calls["n"] += 1
            return np.full((3, 2), fill_value=4.0, dtype=np.float32)

    fake_pacmap = types.ModuleType("pacmap")
    fake_pacmap.PaCMAP = _FakePaCMAP
    monkeypatch.setitem(sys.modules, "pacmap", fake_pacmap)

    reduced = tm_reduction._reduce_with_cache(
        embeddings=embeddings,
        n_components=2,
        method="pacmap",
        params=params,
        random_state=42,
        cache_dir=tmp_path,
        name="fp",
    )

    assert calls["n"] == 1
    assert np.allclose(reduced, 4.0)
    assert "Reduction cache mismatch" in caplog.text


def test_compute_embeddings_uses_cache_on_second_call(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.topic_model")
    calls = {"n": 0}

    def _fake_embed_local(documents, model_name, batch_size, dtype):
        del model_name, batch_size, dtype
        calls["n"] += 1
        return np.arange(len(documents) * 2, dtype=np.float32).reshape(len(documents), 2)

    monkeypatch.setattr(tm_embeddings, "_embed_local", _fake_embed_local)
    docs = ["doc-a", "doc-b", "doc-c"]

    first = tm.compute_embeddings(
        docs,
        provider="local",
        model="local/test-model",
        cache_dir=tmp_path,
    )
    second = tm.compute_embeddings(
        docs,
        provider="local",
        model="local/test-model",
        cache_dir=tmp_path,
    )

    assert calls["n"] == 1
    assert np.array_equal(first, second)
    assert "Loaded embeddings from cache" in caplog.text


def test_compute_embeddings_gguf_uses_cache_then_recomputes_on_mismatch(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.topic_model")
    calls = {"n": 0}

    class _FakeGGUFEmbedder:
        def __init__(self, *, model, batch_size=64, max_workers=5, dtype=np.float32, n_ctx=4096, pooling="cls"):
            del model, batch_size, max_workers, dtype, n_ctx, pooling

        def encode(self, texts, verbose=None, show_progress_bar=None, progress_callback=None, **kwargs):
            del verbose, show_progress_bar, progress_callback, kwargs
            calls["n"] += 1
            base = float(calls["n"])
            return np.full((len(texts), 2), fill_value=base, dtype=np.float32)

    monkeypatch.setattr(tm_embeddings, "GGUFEmbedder", _FakeGGUFEmbedder)
    model = "Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf"
    docs = ["doc-a", "doc-b"]

    first = tm.compute_embeddings(
        docs,
        provider="gguf",
        model=model,
        cache_dir=tmp_path,
        dtype=np.float32,
    )
    second = tm.compute_embeddings(
        docs,
        provider="gguf",
        model=model,
        cache_dir=tmp_path,
        dtype=np.float32,
    )
    third = tm.compute_embeddings(
        docs + ["doc-c"],
        provider="gguf",
        model=model,
        cache_dir=tmp_path,
        dtype=np.float32,
    )

    assert calls["n"] == 2
    assert np.array_equal(first, second)
    assert np.allclose(first, 1.0)
    assert np.allclose(third, 2.0)
    assert "Loaded embeddings from cache" in caplog.text
    assert "Embedding cache mismatch" in caplog.text


def test_compute_embeddings_gguf_pooling_change_invalidates_cache(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.topic_model")
    init_poolings: list[str] = []

    class _FakeGGUFEmbedder:
        def __init__(
            self,
            *,
            model,
            batch_size=64,
            max_workers=5,
            dtype=np.float32,
            n_ctx=4096,
            pooling="cls",
        ):
            del model, batch_size, max_workers, dtype, n_ctx
            init_poolings.append(pooling)
            self.pooling = pooling

        def encode(self, texts, verbose=None, show_progress_bar=None, progress_callback=None, **kwargs):
            del verbose, show_progress_bar, progress_callback, kwargs
            fill = 1.0 if self.pooling == "cls" else 2.0
            return np.full((len(texts), 2), fill_value=fill, dtype=np.float32)

    monkeypatch.setattr(tm_embeddings, "GGUFEmbedder", _FakeGGUFEmbedder)
    model = "Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf"
    docs = ["doc-a", "doc-b"]

    first = tm.compute_embeddings(
        docs,
        provider="gguf",
        model=model,
        cache_dir=tmp_path,
        dtype=np.float32,
        gguf_pooling="cls",
    )
    second = tm.compute_embeddings(
        docs,
        provider="gguf",
        model=model,
        cache_dir=tmp_path,
        dtype=np.float32,
        gguf_pooling="last",
    )
    cache_file = tmp_path / "embeddings_gguf_Qwen_Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf.npz"
    cache_data = np.load(cache_file, allow_pickle=True)

    assert init_poolings == ["cls", "last"]
    assert np.allclose(first, 1.0)
    assert np.allclose(second, 2.0)
    assert str(cache_data["gguf_pooling"]) == "last"
    assert "GGUF pooling changed" in caplog.text


def test_compute_embeddings_cache_hit_reports_document_progress(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _fake_embed_local(documents, model_name, batch_size, dtype):
        del model_name, batch_size, dtype
        calls["n"] += 1
        return np.arange(len(documents) * 2, dtype=np.float32).reshape(len(documents), 2)

    monkeypatch.setattr(tm_embeddings, "_embed_local", _fake_embed_local)
    docs = ["doc-a", "doc-b", "doc-c"]
    progress_updates: list[int] = []

    tm.compute_embeddings(
        docs,
        provider="local",
        model="local/test-model",
        cache_dir=tmp_path,
    )
    tm.compute_embeddings(
        docs,
        provider="local",
        model="local/test-model",
        cache_dir=tmp_path,
        progress_callback=progress_updates.append,
    )

    assert calls["n"] == 1
    assert progress_updates == [3]


def test_reduce_dimensions_uses_cache_then_recomputes_on_param_change(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.topic_model")
    calls = {"fit_transform": 0}

    class _FakePaCMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, embeddings):
            calls["fit_transform"] += 1
            n_components = int(self.kwargs["n_components"])
            marker = float(self.kwargs.get("n_neighbors", 0) + n_components)
            return np.full((len(embeddings), n_components), fill_value=marker, dtype=np.float32)

    fake_pacmap = types.ModuleType("pacmap")
    fake_pacmap.PaCMAP = _FakePaCMAP
    monkeypatch.setitem(sys.modules, "pacmap", fake_pacmap)

    embeddings = np.ones((6, 4), dtype=np.float32)

    r5_first, r2_first = tm.reduce_dimensions(
        embeddings,
        method="pacmap",
        params_5d={"n_neighbors": 15},
        params_2d={"n_neighbors": 15},
        random_state=42,
        cache_dir=tmp_path,
        cache_suffix="cache_hit",
        show_progress=False,
    )
    r5_cached, r2_cached = tm.reduce_dimensions(
        embeddings,
        method="pacmap",
        params_5d={"n_neighbors": 15},
        params_2d={"n_neighbors": 15},
        random_state=42,
        cache_dir=tmp_path,
        cache_suffix="cache_hit",
        show_progress=False,
    )
    r5_changed, r2_changed = tm.reduce_dimensions(
        embeddings,
        method="pacmap",
        params_5d={"n_neighbors": 16},
        params_2d={"n_neighbors": 15},
        random_state=42,
        cache_dir=tmp_path,
        cache_suffix="cache_hit",
        show_progress=False,
    )

    # First run computes both projections (5D + 2D), second run reuses cache, third run
    # recomputes only 5D because only params_5d changed.
    assert calls["fit_transform"] == 3
    assert np.array_equal(r5_first, r5_cached)
    assert np.array_equal(r2_first, r2_cached)
    assert not np.array_equal(r5_first, r5_changed)
    assert np.array_equal(r2_first, r2_changed)
    assert "Loaded 5d_cache_hit from cache" in caplog.text
    assert "Loaded 2d_cache_hit from cache" in caplog.text


# ---------------------------------------------------------------------------
# GGUF provider tests
# ---------------------------------------------------------------------------


def test_create_llm_gguf_uses_native_bertopic_llamacpp(monkeypatch):
    """GGUF branch should use BERTopic's native LlamaCPP representation model."""
    calls: dict = {}

    class _FakeBERTopicLlamaCPP:
        def __init__(self, model, *, prompt=None, pipeline_kwargs=None, nr_docs=4, diversity=None, **kw):
            calls["model"] = model
            calls["prompt"] = prompt
            calls["pipeline_kwargs"] = pipeline_kwargs
            calls["nr_docs"] = nr_docs
            calls["diversity"] = diversity

    fake_representation = types.ModuleType("bertopic.representation")
    fake_representation.LlamaCPP = _FakeBERTopicLlamaCPP
    monkeypatch.setitem(sys.modules, "bertopic.representation", fake_representation)

    import ads_bib._utils.gguf_backend as gguf_mod

    class _FakeLlama:
        def __init__(self, **kwargs):
            if "pooling_type" in kwargs and kwargs["pooling_type"] is None:
                raise TypeError("pooling_type must be omitted, not None")
            calls["llama_kwargs"] = kwargs

    monkeypatch.setattr(gguf_mod, "resolve_gguf_model", lambda model: "/fake/gguf.gguf")
    monkeypatch.setattr(gguf_mod, "safe_stdio", contextlib.nullcontext)
    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    safe_calls: list = []
    monkeypatch.setattr(gguf_mod, "_make_llama_jupyter_safe", lambda llm: safe_calls.append(llm))

    llm = tm_backends._create_llm(
        provider="gguf",
        model="mradermacher/Qwen3-0.6B-GGUF",
        prompt="topic: <label>",
        nr_docs=8,
        diversity=0.2,
        delay=0.3,
        llm_max_new_tokens=64,
        api_key=None,
    )

    assert isinstance(llm, _FakeBERTopicLlamaCPP)
    assert isinstance(calls["model"], _FakeLlama)
    assert calls["pipeline_kwargs"] == {"max_tokens": 64}
    assert calls["nr_docs"] == 8
    assert calls["diversity"] == 0.2
    assert calls["llama_kwargs"]["model_path"] == "/fake/gguf.gguf"
    assert calls["llama_kwargs"]["n_ctx"] == 4096
    assert "pooling_type" not in calls["llama_kwargs"]
    assert safe_calls == [calls["model"]]


def test_fit_toponymy_supports_gguf_llm_provider(monkeypatch):
    """GGUF branch should use Toponymy's native LlamaCppNamer."""
    _install_fake_toponymy_modules(monkeypatch)

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    import ads_bib._utils.gguf_backend as gguf_mod

    monkeypatch.setattr(gguf_mod, "resolve_gguf_model", lambda model: "/fake/gguf.gguf")
    monkeypatch.setattr(gguf_mod, "safe_stdio", contextlib.nullcontext)
    safe_calls: list = []
    monkeypatch.setattr(gguf_mod, "_make_llama_jupyter_safe", lambda llm: safe_calls.append(llm))

    class _FakeNamer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.llm = object()  # stand-in for the internal Llama instance

    fake_llm_wrappers = sys.modules["toponymy.llm_wrappers"]
    fake_llm_wrappers.LlamaCppNamer = _FakeNamer

    calls: dict = {}

    def _fake_record_llm_usage(usage, **kwargs):
        calls["usage"] = usage
        calls["kwargs"] = kwargs

    monkeypatch.setattr(tm_backends, "_record_llm_usage", _fake_record_llm_usage)

    model, topics, _ = tm.fit_toponymy(
        documents=["d1", "d2", "d3"],
        embeddings=np.ones((3, 3), dtype=np.float32),
        clusterable_vectors=np.ones((3, 2), dtype=np.float32),
        backend="toponymy",
        layer_index=0,
        llm_provider="gguf",
        llm_model="mradermacher/Qwen3-0.6B-GGUF",
        embedding_provider="local",
        embedding_model="local-embedder",
        local_llm_max_new_tokens=77,
    )

    assert isinstance(model.llm_wrapper, _FakeNamer)
    assert model.llm_wrapper.kwargs["model_path"] == "/fake/gguf.gguf"
    assert model.llm_wrapper.kwargs["n_ctx"] == 4096
    assert model.llm_wrapper.kwargs["n_gpu_layers"] == -1
    assert len(safe_calls) == 1  # _make_llama_jupyter_safe was called
    assert safe_calls[0] is model.llm_wrapper.llm
    assert isinstance(model.text_embedding_model, _FakeSentenceTransformer)
    assert topics.tolist() == [-1, 0, 1]
    assert calls["usage"] is None
    assert calls["kwargs"]["llm_provider"] == "gguf"
