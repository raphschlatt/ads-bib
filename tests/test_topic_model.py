from __future__ import annotations

from contextlib import contextmanager
import sys
import time
import types

import numpy as np
import pandas as pd

import ads_bib.topic_model as tm


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


def test_reduce_outliers_refreshes_representations_and_logs(capsys):
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

    output = capsys.readouterr().out
    assert "Refreshing topic representations after outlier reassignment" in output
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

    monkeypatch.setattr(tm, "_track_litellm_usage", _fake_track_litellm_usage)
    monkeypatch.setattr(tm, "_record_llm_usage", _fake_record_llm_usage)

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

    def _fake_create_tracked_namer(*, model, api_key, base_url):
        calls["namer"] = {"model": model, "api_key": api_key, "base_url": base_url}
        usage = {"prompt_tokens": 22, "completion_tokens": 7, "call_records": []}
        return object(), usage

    def _fake_record_llm_usage(usage, **kwargs):
        calls["record"] = {"usage": usage, **kwargs}

    monkeypatch.setattr(tm, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)
    monkeypatch.setattr(tm, "_record_llm_usage", _fake_record_llm_usage)

    model, topics, topic_info = tm.fit_toponymy(
        documents=["d1", "d2", "d3"],
        embeddings=np.ones((3, 3), dtype=np.float32),
        clusterable_vectors=np.ones((3, 2), dtype=np.float32),
        backend="toponymy_evoc",
        layer_index=0,
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
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


def test_fit_toponymy_filters_unsupported_evoc_init_params(monkeypatch, capsys):
    _install_fake_toponymy_modules(monkeypatch)

    class _StrictEVoCClusterer:
        def __init__(self, min_clusters=5, min_samples=3):
            self.kwargs = {"min_clusters": min_clusters, "min_samples": min_samples}

    sys.modules["toponymy.clustering"].EVoCClusterer = _StrictEVoCClusterer

    def _fake_create_tracked_namer(*, model, api_key, base_url):
        del model, api_key, base_url
        return object(), {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    monkeypatch.setattr(tm, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)

    model, topics, _ = tm.fit_toponymy(
        documents=["d1", "d2", "d3"],
        embeddings=np.ones((3, 3), dtype=np.float32),
        clusterable_vectors=np.ones((3, 2), dtype=np.float32),
        backend="toponymy_evoc",
        layer_index=0,
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
        api_key="key",
        clusterer_params={"min_clusters": 4, "cluster_levels": 2},
    )

    output = capsys.readouterr().out
    assert "Dropping unsupported EVoCClusterer parameter(s): cluster_levels" in output
    assert isinstance(model.clusterer, _StrictEVoCClusterer)
    assert model.clusterer.kwargs == {"min_clusters": 4, "min_samples": 3}
    assert topics.tolist() == [-1, 0, 1]


def test_fit_toponymy_filters_unsupported_toponymy_init_params(monkeypatch, capsys):
    _install_fake_toponymy_modules(monkeypatch)

    class _StrictToponymyClusterer:
        def __init__(self, min_clusters=5, base_min_cluster_size=10):
            self.kwargs = {
                "min_clusters": min_clusters,
                "base_min_cluster_size": base_min_cluster_size,
            }

    sys.modules["toponymy"].ToponymyClusterer = _StrictToponymyClusterer

    def _fake_create_tracked_namer(*, model, api_key, base_url):
        del model, api_key, base_url
        return object(), {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    monkeypatch.setattr(tm, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)

    model, topics, _ = tm.fit_toponymy(
        documents=["d1", "d2", "d3"],
        embeddings=np.ones((3, 3), dtype=np.float32),
        clusterable_vectors=np.ones((3, 2), dtype=np.float32),
        backend="toponymy",
        layer_index=0,
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
        api_key="key",
        clusterer_params={
            "min_clusters": 7,
            "base_min_cluster_size": 11,
            "cluster_levels": 3,
        },
    )

    output = capsys.readouterr().out
    assert "Dropping unsupported ToponymyClusterer parameter(s): cluster_levels" in output
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

    monkeypatch.setattr(tm, "_record_llm_usage", _fake_record_llm_usage)

    model, topics, _ = tm.fit_toponymy(
        documents=["d1", "d2", "d3"],
        embeddings=np.ones((3, 3), dtype=np.float32),
        clusterable_vectors=np.ones((3, 2), dtype=np.float32),
        backend="toponymy",
        layer_index=0,
        llm_provider="local",
        llm_model="local-llm",
        embedding_model="local-embedder",
    )

    assert isinstance(model.llm_wrapper, _FakeHuggingFaceNamer)
    assert model.llm_wrapper.model == "local-llm"
    assert isinstance(model.text_embedding_model, _FakeSentenceTransformer)
    assert model.text_embedding_model.model_name == "local-embedder"
    assert topics.tolist() == [-1, 0, 1]
    assert calls["usage"] is None
    assert calls["kwargs"]["llm_provider"] == "local"


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
            embedding_model="local-embedder",
        )
        assert False, "Expected ImportError for missing local Toponymy dependencies"
    except ImportError as exc:
        assert "llm_provider='local' requires optional dependencies" in str(exc)


def test_fit_toponymy_validates_layer_index(monkeypatch):
    _install_fake_toponymy_modules(monkeypatch)

    def _fake_create_tracked_namer(*, model, api_key, base_url):
        del model, api_key, base_url
        return object(), {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    monkeypatch.setattr(tm, "_create_tracked_toponymy_namer", _fake_create_tracked_namer)

    try:
        tm.fit_toponymy(
            documents=["d1", "d2", "d3"],
            embeddings=np.ones((3, 3), dtype=np.float32),
            clusterable_vectors=np.ones((3, 2), dtype=np.float32),
            backend="toponymy",
            layer_index=9,
            llm_provider="openrouter",
            llm_model="google/gemini-3-flash-preview",
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


def test_record_llm_usage_fetches_openrouter_cost(monkeypatch):
    calls: dict = {}

    def _fake_fetch_openrouter_costs(call_records, api_key, *, openrouter_cost_mode, max_workers=5):
        del max_workers
        calls["call_records"] = call_records
        calls["api_key"] = api_key
        calls["mode"] = openrouter_cost_mode
        return 0.0123

    class _Tracker:
        def __init__(self):
            self.entries = []

        def add(self, **kwargs):
            self.entries.append(kwargs)

    monkeypatch.setattr(tm, "_fetch_openrouter_costs", _fake_fetch_openrouter_costs)
    tracker = _Tracker()

    tm._record_llm_usage(
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
    tm._patch_clusterer_for_toponymy_kwargs(clusterer)

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

    def _fake_embed_openrouter(documents, model, batch_size, dtype, api_key, *, max_workers=5):
        del model, batch_size, api_key
        calls["max_workers"] = max_workers
        return np.ones((len(documents), 2), dtype=dtype), {
            "prompt_tokens": 0,
            "total_tokens": 0,
            "call_records": [],
        }

    monkeypatch.setattr(tm, "_embed_openrouter", _fake_embed_openrouter)

    emb = tm.compute_embeddings(
        ["d1", "d2", "d3"],
        provider="openrouter",
        model="google/gemini-embedding-001",
        max_workers=7,
        api_key="key",
    )

    assert emb.shape == (3, 2)
    assert calls["max_workers"] == 7


def test_embed_openrouter_parallel_keeps_document_order(monkeypatch):
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
    monkeypatch.setattr(tm, "extract_usage_stats", lambda resp: resp["usage"])
    monkeypatch.setattr(tm, "extract_generation_id", lambda resp: resp["id"])
    monkeypatch.setattr(tm, "extract_response_cost", lambda **kwargs: None)

    emb, usage = tm._embed_openrouter(
        [f"doc_{i}" for i in range(6)],
        model="google/gemini-embedding-001",
        batch_size=2,
        dtype=np.float32,
        api_key="key",
        max_workers=3,
    )

    assert emb.shape == (6, 1)
    assert emb[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert usage["prompt_tokens"] == 6
    assert usage["total_tokens"] == 6
    assert [r["generation_id"] for r in usage["call_records"]] == ["gen_0", "gen_2", "gen_4"]
