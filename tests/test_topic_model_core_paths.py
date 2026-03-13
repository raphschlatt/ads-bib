from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
import types

import numpy as np
import pytest

import ads_bib.topic_model as tm
from ads_bib.topic_model import backends as tm_backends
from ads_bib.topic_model import reduction as tm_reduction


def test_reduce_dimensions_calls_reduce_for_5d_and_2d(monkeypatch):
    calls: list[tuple[int, str, dict, int, str]] = []

    def _fake_reduce(embeddings, n_components, method, params, random_state, cache_dir, name):
        del cache_dir
        calls.append((n_components, method, dict(params), random_state, name))
        return np.full((len(embeddings), n_components), fill_value=n_components, dtype=np.float32)

    monkeypatch.setattr(tm_reduction, "_reduce_with_cache", _fake_reduce)

    r5, r2 = tm.reduce_dimensions(
        np.ones((4, 3), dtype=np.float32),
        method="umap",
        params_5d={"a": 1},
        params_2d={"b": 2},
        random_state=123,
        cache_suffix="unit",
    )

    assert r5.shape == (4, 5)
    assert r2.shape == (4, 2)
    assert calls == [
        (5, "umap", {"a": 1}, 123, "5d_unit"),
        (2, "umap", {"b": 2}, 123, "2d_unit"),
    ]


def test_reduce_pacmap_normalizes_metric_and_ignores_min_dist(monkeypatch):
    calls: dict = {}

    class _FakePaCMAP:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def fit_transform(self, embeddings):
            return np.ones((len(embeddings), calls["kwargs"]["n_components"]), dtype=np.float32)

    fake_pacmap = types.ModuleType("pacmap")
    fake_pacmap.PaCMAP = _FakePaCMAP
    monkeypatch.setitem(sys.modules, "pacmap", fake_pacmap)

    out = tm_reduction._reduce_with_cache(
        embeddings=np.ones((3, 4), dtype=np.float32),
        n_components=5,
        method="pacmap",
        params={"metric": "cosine", "min_dist": 0.1, "n_neighbors": 30},
        random_state=42,
        cache_dir=None,
        name="unit",
    )

    assert out.shape == (3, 5)
    assert calls["kwargs"]["distance"] == "angular"
    assert "metric" not in calls["kwargs"]
    assert "min_dist" not in calls["kwargs"]
    assert calls["kwargs"]["n_neighbors"] == 30


def test_cluster_documents_uses_selected_cluster_model(monkeypatch):
    calls: dict = {}

    class _FakeClusterModel:
        def fit_predict(self, reduced_5d):
            calls["shape"] = reduced_5d.shape
            return np.array([0, -1, 1], dtype=int)

    def _fake_create_cluster_model(method: str, params: dict | None = None):
        calls["method"] = method
        calls["params"] = params
        return _FakeClusterModel()

    monkeypatch.setattr(tm_backends, "_create_cluster_model", _fake_create_cluster_model)

    out = tm_backends.cluster_documents(
        np.ones((3, 5), dtype=np.float32),
        method="hdbscan",
        params={"min_cluster_size": 10},
    )

    assert out.tolist() == [0, -1, 1]
    assert calls["method"] == "hdbscan"
    assert calls["params"] == {"min_cluster_size": 10}
    assert calls["shape"] == (3, 5)


def test_fit_bertopic_constructs_model_and_records_llm_usage(monkeypatch):
    calls: dict = {}

    class _FakeBERTopic:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def fit_transform(self, documents, reduced_5d):
            calls["fit_documents"] = list(documents)
            calls["fit_shape"] = reduced_5d.shape
            return np.zeros(len(documents), dtype=int), None

    class _FakeBaseDimensionalityReduction:
        pass

    class _FakeClassTfidfTransformer:
        pass

    class _FakeCountVectorizer:
        def __init__(self, **kwargs):
            calls["vectorizer_kwargs"] = kwargs

    fake_bertopic = types.ModuleType("bertopic")
    fake_bertopic.BERTopic = _FakeBERTopic
    fake_dim = types.ModuleType("bertopic.dimensionality")
    fake_dim.BaseDimensionalityReduction = _FakeBaseDimensionalityReduction
    fake_vec = types.ModuleType("bertopic.vectorizers")
    fake_vec.ClassTfidfTransformer = _FakeClassTfidfTransformer

    fake_sklearn = types.ModuleType("sklearn")
    fake_feature_extraction = types.ModuleType("sklearn.feature_extraction")
    fake_text = types.ModuleType("sklearn.feature_extraction.text")
    fake_text.CountVectorizer = _FakeCountVectorizer

    monkeypatch.setitem(sys.modules, "bertopic", fake_bertopic)
    monkeypatch.setitem(sys.modules, "bertopic.dimensionality", fake_dim)
    monkeypatch.setitem(sys.modules, "bertopic.vectorizers", fake_vec)
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction", fake_feature_extraction)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", fake_text)

    cluster_model = object()
    def _fake_build_representation_model(**kwargs):
        calls["rep_kwargs"] = kwargs
        return {"rep": kwargs}

    monkeypatch.setattr(tm_backends, "_build_representation_model", _fake_build_representation_model)
    monkeypatch.setattr(tm_backends, "_create_cluster_model", lambda method, params: cluster_model)

    @contextmanager
    def _fake_track_litellm_usage(*, enabled: bool):
        calls["track_enabled"] = enabled
        yield {"prompt_tokens": 9, "completion_tokens": 3, "call_records": []}

    def _fake_record_llm_usage(usage, **kwargs):
        calls["usage"] = usage
        calls["record_kwargs"] = kwargs

    monkeypatch.setattr(tm_backends, "_track_litellm_usage", _fake_track_litellm_usage)
    monkeypatch.setattr(tm_backends, "_record_llm_usage", _fake_record_llm_usage)

    model = tm.fit_bertopic(
        documents=["d1", "d2"],
        reduced_5d=np.ones((2, 5), dtype=np.float32),
        llm_provider="openrouter",
        llm_model="openrouter/model",
        pipeline_models=["POS"],
        parallel_models=["MMR"],
        clustering_method="hdbscan",
        clustering_params={"min_cluster_size": 10},
        top_n_words=33,
        pos_spacy_model="en_core_web_md",
        min_df=1,
        api_key="key",
        cost_tracker=object(),
    )

    assert isinstance(model, _FakeBERTopic)
    assert calls["fit_documents"] == ["d1", "d2"]
    assert calls["fit_shape"] == (2, 5)
    assert calls["track_enabled"] is True
    assert calls["init_kwargs"]["hdbscan_model"] is cluster_model
    assert calls["init_kwargs"]["top_n_words"] == 33
    assert calls["vectorizer_kwargs"]["min_df"] == 1
    assert calls["rep_kwargs"]["pos_spacy_model"] == "en_core_web_md"
    assert calls["record_kwargs"]["step"] == "llm_labeling"
    assert calls["record_kwargs"]["llm_provider"] == "openrouter"
    assert calls["usage"]["prompt_tokens"] == 9


def test_fit_bertopic_suppresses_minilm_load_report_for_keybert(monkeypatch, caplog):
    calls: dict = {}

    class _FakeBERTopic:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def fit_transform(self, documents, reduced_5d):
            calls["fit_documents"] = list(documents)
            calls["fit_shape"] = reduced_5d.shape
            return np.zeros(len(documents), dtype=int), None

    class _FakeBaseDimensionalityReduction:
        pass

    class _FakeClassTfidfTransformer:
        pass

    class _FakeCountVectorizer:
        def __init__(self, **kwargs):
            calls["vectorizer_kwargs"] = kwargs

    class _FakeSentenceTransformer:
        def __init__(self, model_name):
            calls["sentence_transformer_model"] = model_name
            logging.getLogger("transformers.modeling_utils").warning(
                "BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2"
            )
            logging.getLogger("transformers.integrations.tensor_parallel").warning(
                "The following layers were not sharded: embeddings.word_embeddings.weight"
            )

    fake_bertopic = types.ModuleType("bertopic")
    fake_bertopic.BERTopic = _FakeBERTopic
    fake_dim = types.ModuleType("bertopic.dimensionality")
    fake_dim.BaseDimensionalityReduction = _FakeBaseDimensionalityReduction
    fake_vec = types.ModuleType("bertopic.vectorizers")
    fake_vec.ClassTfidfTransformer = _FakeClassTfidfTransformer

    fake_sklearn = types.ModuleType("sklearn")
    fake_feature_extraction = types.ModuleType("sklearn.feature_extraction")
    fake_text = types.ModuleType("sklearn.feature_extraction.text")
    fake_text.CountVectorizer = _FakeCountVectorizer

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = _FakeSentenceTransformer

    monkeypatch.setitem(sys.modules, "bertopic", fake_bertopic)
    monkeypatch.setitem(sys.modules, "bertopic.dimensionality", fake_dim)
    monkeypatch.setitem(sys.modules, "bertopic.vectorizers", fake_vec)
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction", fake_feature_extraction)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", fake_text)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    cluster_model = object()
    monkeypatch.setattr(tm_backends, "_build_representation_model", lambda **kwargs: {"rep": kwargs})
    monkeypatch.setattr(tm_backends, "_create_cluster_model", lambda method, params: cluster_model)

    @contextmanager
    def _fake_capture_external_output(*args, **kwargs):
        del args, kwargs
        yield

    monkeypatch.setattr(tm_backends, "capture_external_output", _fake_capture_external_output)
    monkeypatch.setattr(tm_backends, "get_runtime_log_path", lambda: None)
    with caplog.at_level(logging.WARNING):
        model = tm.fit_bertopic(
            documents=["d1", "d2"],
            reduced_5d=np.ones((2, 5), dtype=np.float32),
            llm_provider="openrouter",
            llm_model="openrouter/model",
            pipeline_models=["KeyBERT"],
            parallel_models=[],
            clustering_method="hdbscan",
            clustering_params={"min_cluster_size": 10},
            top_n_words=10,
            min_df=1,
            api_key="key",
        )

    assert isinstance(model, _FakeBERTopic)
    assert calls["sentence_transformer_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert calls["init_kwargs"]["embedding_model"].__class__ is _FakeSentenceTransformer
    assert "LOAD REPORT" not in caplog.text
    assert "not sharded" not in caplog.text


def test_fit_bertopic_skips_keybert_helper_model_when_keybert_disabled(monkeypatch):
    calls: dict = {}

    class _FakeBERTopic:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def fit_transform(self, documents, reduced_5d):
            return np.zeros(len(documents), dtype=int), None

    class _FakeBaseDimensionalityReduction:
        pass

    class _FakeClassTfidfTransformer:
        pass

    class _FakeCountVectorizer:
        def __init__(self, **kwargs):
            calls["vectorizer_kwargs"] = kwargs

    class _BrokenSentenceTransformer:
        def __init__(self, model_name):
            del model_name
            raise AssertionError("KeyBERT helper model should not load when KeyBERT is disabled")

    fake_bertopic = types.ModuleType("bertopic")
    fake_bertopic.BERTopic = _FakeBERTopic
    fake_dim = types.ModuleType("bertopic.dimensionality")
    fake_dim.BaseDimensionalityReduction = _FakeBaseDimensionalityReduction
    fake_vec = types.ModuleType("bertopic.vectorizers")
    fake_vec.ClassTfidfTransformer = _FakeClassTfidfTransformer

    fake_sklearn = types.ModuleType("sklearn")
    fake_feature_extraction = types.ModuleType("sklearn.feature_extraction")
    fake_text = types.ModuleType("sklearn.feature_extraction.text")
    fake_text.CountVectorizer = _FakeCountVectorizer

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = _BrokenSentenceTransformer

    monkeypatch.setitem(sys.modules, "bertopic", fake_bertopic)
    monkeypatch.setitem(sys.modules, "bertopic.dimensionality", fake_dim)
    monkeypatch.setitem(sys.modules, "bertopic.vectorizers", fake_vec)
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction", fake_feature_extraction)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", fake_text)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    cluster_model = object()
    monkeypatch.setattr(tm_backends, "_build_representation_model", lambda **kwargs: {"rep": kwargs})
    monkeypatch.setattr(tm_backends, "_create_cluster_model", lambda method, params: cluster_model)

    model = tm.fit_bertopic(
        documents=["d1", "d2"],
        reduced_5d=np.ones((2, 5), dtype=np.float32),
        llm_provider="openrouter",
        llm_model="openrouter/model",
        pipeline_models=["POS"],
        parallel_models=["MMR"],
        clustering_method="hdbscan",
        clustering_params={"min_cluster_size": 10},
        top_n_words=10,
        min_df=1,
        api_key="key",
    )

    assert isinstance(model, _FakeBERTopic)
    assert calls["init_kwargs"]["embedding_model"] is None


def test_compute_embeddings_validates_provider():
    with pytest.raises(ValueError, match="Invalid provider 'bad_provider'"):
        tm.compute_embeddings(["doc"], provider="bad_provider", model="m")


def test_fit_bertopic_validates_provider():
    with pytest.raises(ValueError, match="Invalid provider 'bad_provider'"):
        tm.fit_bertopic(
            ["doc"], np.ones((1, 5), dtype=np.float32),
            llm_provider="bad_provider", llm_model="m",
        )


def test_reduce_dimensions_auto_builds_suffix_from_embedding_id(monkeypatch):
    calls: list[tuple[int, str, dict, int, str]] = []

    def _fake_reduce(embeddings, n_components, method, params, random_state, cache_dir, name):
        del cache_dir
        calls.append((n_components, method, dict(params), random_state, name))
        return np.full((len(embeddings), n_components), fill_value=n_components, dtype=np.float32)

    monkeypatch.setattr(tm_reduction, "_reduce_with_cache", _fake_reduce)

    tm.reduce_dimensions(
        np.ones((4, 3), dtype=np.float32),
        method="pacmap",
        params_5d={"n_neighbors": 60, "metric": "angular"},
        random_state=42,
        embedding_id="openrouter/google/gemini-embedding-001",
    )

    expected_suffix = "openrouter_google_gemini-embedding-001_pacmap_nn60_minddef_metricangular_rs42"
    assert calls[0][4] == f"5d_{expected_suffix}"
    assert calls[1][4] == f"2d_{expected_suffix}"


def test_reduce_dimensions_explicit_suffix_takes_precedence(monkeypatch):
    calls: list[str] = []

    def _fake_reduce(embeddings, n_components, method, params, random_state, cache_dir, name):
        del cache_dir
        calls.append(name)
        return np.full((len(embeddings), n_components), fill_value=n_components, dtype=np.float32)

    monkeypatch.setattr(tm_reduction, "_reduce_with_cache", _fake_reduce)

    tm.reduce_dimensions(
        np.ones((4, 3), dtype=np.float32),
        cache_suffix="explicit",
        embedding_id="openrouter/model",
    )

    assert calls[0] == "5d_explicit"
    assert calls[1] == "2d_explicit"
