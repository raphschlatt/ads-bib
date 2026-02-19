from __future__ import annotations

from contextlib import contextmanager

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
