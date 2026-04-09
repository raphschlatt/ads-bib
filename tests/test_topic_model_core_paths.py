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

pytestmark = pytest.mark.requires_topic_stack


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
    monkeypatch.setattr(tm_backends, "validate_provider", lambda *a, **k: None)

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
    monkeypatch.setattr(
        tm_backends,
        "_consume_openrouter_representation_usage",
        lambda representation_model: {"prompt_tokens": 9, "completion_tokens": 3, "call_records": []},
    )
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
    assert calls["track_enabled"] is False
    assert calls["init_kwargs"]["hdbscan_model"] is cluster_model
    assert calls["init_kwargs"]["top_n_words"] == 33
    assert calls["vectorizer_kwargs"]["min_df"] == 1
    assert calls["rep_kwargs"]["pos_spacy_model"] == "en_core_web_md"
    assert calls["record_kwargs"]["step"] == "llm_labeling"
    assert calls["record_kwargs"]["llm_provider"] == "openrouter"
    assert calls["usage"]["prompt_tokens"] == 9


def test_fit_bertopic_uses_defaults_only_when_model_lists_are_none(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(tm_backends, "validate_provider", lambda *a, **k: None)

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

    class _FakeSentenceTransformer:
        def __init__(self, model_name):
            calls["sentence_transformer_model"] = model_name

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

    def _fake_build_representation_model(**kwargs):
        calls["rep_kwargs"] = kwargs
        return {"rep": kwargs}

    monkeypatch.setattr(tm_backends, "_build_representation_model", _fake_build_representation_model)
    monkeypatch.setattr(tm_backends, "_create_cluster_model", lambda method, params: object())

    tm.fit_bertopic(
        documents=["d1", "d2"],
        reduced_5d=np.ones((2, 5), dtype=np.float32),
        llm_provider="openrouter",
        llm_model="openrouter/model",
        pipeline_models=None,
        parallel_models=None,
        clustering_method="hdbscan",
        clustering_params={"min_cluster_size": 10},
        top_n_words=10,
        min_df=1,
        api_key="key",
    )

    assert calls["rep_kwargs"]["pipeline_models"] == ["POS", "KeyBERT", "MMR"]
    assert calls["rep_kwargs"]["parallel_models"] == ["MMR", "POS", "KeyBERT"]
    assert calls["sentence_transformer_model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_fit_bertopic_preserves_explicitly_empty_model_lists(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(tm_backends, "validate_provider", lambda *a, **k: None)

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

    def _fake_build_representation_model(**kwargs):
        calls["rep_kwargs"] = kwargs
        return {"rep": kwargs}

    monkeypatch.setattr(tm_backends, "_build_representation_model", _fake_build_representation_model)
    monkeypatch.setattr(tm_backends, "_create_cluster_model", lambda method, params: object())

    tm.fit_bertopic(
        documents=["d1", "d2"],
        reduced_5d=np.ones((2, 5), dtype=np.float32),
        llm_provider="openrouter",
        llm_model="openrouter/model",
        pipeline_models=[],
        parallel_models=[],
        clustering_method="hdbscan",
        clustering_params={"min_cluster_size": 10},
        top_n_words=10,
        min_df=1,
        api_key="key",
    )

    assert calls["rep_kwargs"]["pipeline_models"] == []
    assert calls["rep_kwargs"]["parallel_models"] == []


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
    monkeypatch.setattr(tm_backends, "validate_provider", lambda *a, **k: None)

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


@pytest.mark.parametrize(
    "module_name",
    [
        "bertopic.representation._textgeneration",
        "bertopic.representation._litellm",
        "bertopic.representation._openai",
        "bertopic.representation._llamacpp",
    ],
)
def test_bridge_bertopic_label_progress_updates_reporter_incrementally(monkeypatch, module_name):
    fake_module = types.ModuleType(module_name)
    original_tqdm = object()
    fake_module.tqdm = original_tqdm
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    class _FakeProgressBar:
        def __init__(self, total: int | None) -> None:
            self.total = total
            self.updates: list[int] = []
            self.refreshed_totals: list[int | None] = []

        def update(self, amount: int = 1) -> None:
            self.updates.append(int(amount))

        def refresh(self) -> None:
            self.refreshed_totals.append(self.total)

    class _FakeProgressContext:
        def __init__(self, progress_bar: _FakeProgressBar) -> None:
            self._progress_bar = progress_bar

        def __enter__(self) -> _FakeProgressBar:
            return self._progress_bar

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    class _FakeReporter:
        def __init__(self) -> None:
            self.calls: list[tuple[int | None, str]] = []
            self.progress_bar: _FakeProgressBar | None = None

        def progress(self, *, total: int | None, desc: str):
            self.calls.append((total, desc))
            self.progress_bar = _FakeProgressBar(total)
            return _FakeProgressContext(self.progress_bar)

    reporter = _FakeReporter()

    with tm_backends._bridge_bertopic_label_progress(reporter=reporter, desc="fit"):
        for _ in fake_module.tqdm(["topic-a", "topic-b", "topic-c"], disable=True):
            pass
        for _ in fake_module.tqdm(["topic-d"], disable=True):
            pass

    assert reporter.calls == [(3, "fit")]
    assert reporter.progress_bar is not None
    assert reporter.progress_bar.updates == [1, 1, 1, 1]
    assert reporter.progress_bar.total == 4
    assert reporter.progress_bar.refreshed_totals == [4]
    assert fake_module.tqdm is original_tqdm


def test_bridge_bertopic_label_progress_patches_local_backends_tqdm(monkeypatch):
    original_tqdm = tm_backends.tqdm

    class _FakeProgressBar:
        def __init__(self, total: int | None) -> None:
            self.total = total
            self.updates: list[int] = []

        def update(self, amount: int = 1) -> None:
            self.updates.append(int(amount))

        def refresh(self) -> None:
            return None

    class _FakeProgressContext:
        def __init__(self, progress_bar: _FakeProgressBar) -> None:
            self._progress_bar = progress_bar

        def __enter__(self) -> _FakeProgressBar:
            return self._progress_bar

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    class _FakeReporter:
        def __init__(self) -> None:
            self.progress_bar: _FakeProgressBar | None = None

        def progress(self, *, total: int | None, desc: str):
            assert desc == "fit"
            self.progress_bar = _FakeProgressBar(total)
            return _FakeProgressContext(self.progress_bar)

    reporter = _FakeReporter()

    with tm_backends._bridge_bertopic_label_progress(reporter=reporter, desc="fit"):
        for _ in tm_backends.tqdm(["topic-a", "topic-b"], disable=True):
            pass

    assert reporter.progress_bar is not None
    assert reporter.progress_bar.updates == [1, 1]
    assert tm_backends.tqdm is original_tqdm


def test_bridge_bertopic_label_progress_keeps_modules_unchanged_without_reporter(monkeypatch):
    module_name = "bertopic.representation._llamacpp"
    fake_module = types.ModuleType(module_name)
    original_tqdm = object()
    fake_module.tqdm = original_tqdm
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    with tm_backends._bridge_bertopic_label_progress(reporter=None, desc="fit"):
        assert fake_module.tqdm is original_tqdm

    assert fake_module.tqdm is original_tqdm


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
