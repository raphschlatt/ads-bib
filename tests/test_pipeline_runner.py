from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ads_bib.pipeline as pipeline


class _DummyTopicModel:
    def __init__(self) -> None:
        self.topics_ = [0, 1]

    def get_topic_info(self) -> pd.DataFrame:
        return pd.DataFrame({"Topic": [0, 1], "Name": ["Topic 0", "Topic 1"]})


def test_pipeline_config_yaml_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "run:\n"
            "  run_name: test\n"
            "  start_stage: translate\n"
            "search:\n"
            "  query: author:test\n"
            "translate:\n"
            "  fasttext_model: data/models/lid.176.bin\n"
        ),
        encoding="utf-8",
    )

    config = pipeline.PipelineConfig.from_yaml(config_path)
    data = config.to_dict()

    assert data["run"]["run_name"] == "test"
    assert data["run"]["start_stage"] == "translate"
    assert data["search"]["query"] == "author:test"
    assert data["translate"]["fasttext_model"] == "data/models/lid.176.bin"


def test_run_pipeline_respects_stage_slice(monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"fasttext_model": "data/models/lid.176.bin"},
        }
    )
    events: list[str] = []
    fake_ctx = SimpleNamespace(config=config, run=SimpleNamespace(save_config=lambda cfg: events.append("save_config")))

    monkeypatch.setattr(
        pipeline.PipelineContext,
        "create",
        classmethod(lambda cls, config, **kwargs: fake_ctx),
    )
    monkeypatch.setattr(
        pipeline,
        "_STAGE_FUNCS",
        {name: (lambda ctx, stage=name: events.append(stage) or ctx) for name in pipeline.STAGE_ORDER},
    )

    pipeline.run_pipeline(config, start_stage="translate", stop_stage="topic_fit", load_environment=False)

    assert events == [
        "save_config",
        "translate",
        "tokenize",
        "author_disambiguation",
        "embeddings",
        "reduction",
        "topic_fit",
    ]


def test_run_topic_fit_stage_uses_tokenized_snapshot_and_caches(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": "data/models/lid.176.bin"},
            "author_disambiguation": {"enabled": False},
            "topic_model": {
                "backend": "bertopic",
                "embedding_provider": "local",
                "embedding_model": "mini",
                "llm_provider": "local",
                "llm_model": "tiny",
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    pubs = pd.DataFrame(
        [
            {"Bibcode": "b1", "Author": ["Doe, A."], "full_text": "alpha beta", "Title_en": "A", "Abstract_en": "alpha", "Year": 2020},
            {"Bibcode": "b2", "Author": ["Roe, B."], "full_text": "gamma delta", "Title_en": "B", "Abstract_en": "beta", "Year": 2021},
        ]
    )
    refs = pd.DataFrame(
        [
            {"Bibcode": "r1", "Author": ["Ref, A."], "Title_en": "R", "Abstract_en": "ref", "Year": 2019}
        ]
    )
    events: list[str] = []

    monkeypatch.setattr(
        pipeline,
        "load_disambiguated_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(pipeline, "load_tokenized_snapshot", lambda **kwargs: (pubs.copy(), refs.copy()))
    monkeypatch.setattr(pipeline, "save_disambiguated_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "search_ads",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("search should not run")),
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("export should not run")),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_languages",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("language detection should not run")),
    )
    monkeypatch.setattr(
        pipeline,
        "translate_dataframe",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("translate should not run")),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_embeddings",
        lambda documents, **kwargs: events.append("embeddings") or np.ones((len(documents), 4)),
    )
    monkeypatch.setattr(
        pipeline,
        "reduce_dimensions",
        lambda embeddings, **kwargs: (events.append("reduction") or np.ones((len(embeddings), 5)), np.ones((len(embeddings), 2))),
    )
    monkeypatch.setattr(
        pipeline,
        "fit_bertopic",
        lambda documents, reduced_5d, **kwargs: events.append("fit") or _DummyTopicModel(),
    )
    monkeypatch.setattr(
        pipeline,
        "reduce_outliers",
        lambda topic_model, documents, topics, reduced_5d, **kwargs: np.asarray(topics),
    )

    pipeline.run_topic_fit_stage(ctx)

    assert events == ["embeddings", "reduction", "fit"]
    assert ctx.documents == ["alpha beta", "gamma delta"]
    assert ctx.publications is not None
    assert ctx.refs is not None
    assert ctx.topics.tolist() == [0, 1]
    assert list(ctx.topic_info["Name"]) == ["Topic 0", "Topic 1"]


def test_validate_stage_name_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Invalid stage"):
        pipeline.validate_stage_name("unknown")
