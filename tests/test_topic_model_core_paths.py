from __future__ import annotations

from contextlib import contextmanager
import sys
import types

import numpy as np

import ads_bib.topic_model as tm


def test_reduce_dimensions_calls_reduce_for_5d_and_2d(monkeypatch):
    calls: list[tuple[int, str, dict, int, str]] = []

    def _fake_reduce(embeddings, n_components, method, params, random_state, cache_dir, name):
        del cache_dir
        calls.append((n_components, method, dict(params), random_state, name))
        return np.full((len(embeddings), n_components), fill_value=n_components, dtype=np.float32)

    monkeypatch.setattr(tm, "_reduce", _fake_reduce)

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

    out = tm._reduce(
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

    monkeypatch.setattr(tm, "_create_cluster_model", _fake_create_cluster_model)

    out = tm.cluster_documents(
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

    monkeypatch.setattr(tm, "_build_representation_model", _fake_build_representation_model)
    monkeypatch.setattr(tm, "_create_cluster_model", lambda method, params: cluster_model)

    @contextmanager
    def _fake_track_litellm_usage(*, enabled: bool):
        calls["track_enabled"] = enabled
        yield {"prompt_tokens": 9, "completion_tokens": 3, "call_records": []}

    def _fake_record_llm_usage(usage, **kwargs):
        calls["usage"] = usage
        calls["record_kwargs"] = kwargs

    monkeypatch.setattr(tm, "_track_litellm_usage", _fake_track_litellm_usage)
    monkeypatch.setattr(tm, "_record_llm_usage", _fake_record_llm_usage)

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
        pos_spacy_model="en_core_web_lg",
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
    assert calls["rep_kwargs"]["pos_spacy_model"] == "en_core_web_lg"
    assert calls["record_kwargs"]["step"] == "llm_labeling"
    assert calls["record_kwargs"]["llm_provider"] == "openrouter"
    assert calls["usage"]["prompt_tokens"] == 9
