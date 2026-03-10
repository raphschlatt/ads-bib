from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ads_bib.pipeline as pipeline
from ads_bib.prompts import BERTOPIC_LABELING_PHYSICS


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


def test_default_pipeline_config_template_loads():
    config = pipeline.PipelineConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "pipeline" / "default.yaml"
    )
    data = config.to_dict()

    assert data["run"]["start_stage"] == "search"
    assert data["run"]["stop_stage"] is None
    assert data["topic_model"]["llm_prompt_name"] == "physics"
    assert data["author_disambiguation"]["enabled"] is False
    assert data["tokenize"]["spacy_model"] == "en_core_web_md"
    assert data["tokenize"]["fallback_model"] == "en_core_web_md"


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
    completed: set[str] = set()

    def _runner(stage_name):
        def _run(_ctx):
            if stage_name == "search":
                completed.add("search")
                events.append("search")
                return _ctx
            if stage_name == "export":
                if "search" not in completed:
                    raise pipeline.StagePrerequisiteError("export", "search", "need search")
                completed.add("export")
                events.append("export")
                return _ctx
            if stage_name == "translate":
                if "export" not in completed:
                    raise pipeline.StagePrerequisiteError("translate", "export", "need export")
                completed.add("translate")
                events.append("translate")
                return _ctx
            if stage_name == "tokenize":
                if "translate" not in completed:
                    raise pipeline.StagePrerequisiteError("tokenize", "translate", "need translate")
                completed.add("tokenize")
                events.append("tokenize")
                return _ctx
            if stage_name == "author_disambiguation":
                if "tokenize" not in completed:
                    raise pipeline.StagePrerequisiteError(
                        "author_disambiguation",
                        "tokenize",
                        "need tokenize",
                    )
                completed.add("author_disambiguation")
                events.append("author_disambiguation")
                return _ctx
            if stage_name == "embeddings":
                if "author_disambiguation" not in completed:
                    raise pipeline.StagePrerequisiteError("embeddings", "author_disambiguation", "need and")
                completed.add("embeddings")
                events.append("embeddings")
                return _ctx
            if stage_name == "reduction":
                if "embeddings" not in completed:
                    raise pipeline.StagePrerequisiteError("reduction", "embeddings", "need embeddings")
                completed.add("reduction")
                events.append("reduction")
                return _ctx
            if stage_name == "topic_fit":
                if "reduction" not in completed:
                    raise pipeline.StagePrerequisiteError("topic_fit", "reduction", "need reduction")
                completed.add("topic_fit")
                events.append("topic_fit")
                return _ctx
            events.append(stage_name)
            return _ctx

        return _run

    monkeypatch.setattr(pipeline, "_STAGE_FUNCS", {name: _runner(name) for name in pipeline.STAGE_ORDER})

    pipeline.run_pipeline(config, start_stage="translate", stop_stage="topic_fit", load_environment=False)

    assert events == [
        "save_config",
        "search",
        "export",
        "translate",
        "tokenize",
        "author_disambiguation",
        "embeddings",
        "reduction",
        "topic_fit",
    ]


def test_run_translate_stage_prefers_current_export_results_over_snapshot(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {
                "enabled": True,
                "provider": "nllb",
                "model": "stub",
                "fasttext_model": str(tmp_path / "lid.176.bin"),
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.publications = pd.DataFrame([{"Bibcode": "fresh-pub", "Title": "T", "Abstract": "A"}])
    ctx.refs = pd.DataFrame([{"Bibcode": "fresh-ref", "Title": "RT", "Abstract": "RA"}])

    monkeypatch.setattr(
        pipeline,
        "load_translated_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("stale translated snapshot should not load")),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_languages",
        lambda df, columns, model_path: df.assign(
            **{f"{col}_lang": "en" for col in columns}
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "translate_dataframe",
        lambda df, columns, **kwargs: (
            df.assign(**{f"{col}_en": df[col] for col in columns}),
            {},
        ),
    )
    monkeypatch.setattr(pipeline, "save_translated_snapshot", lambda *args, **kwargs: None)

    pipeline.run_translate_stage(ctx)

    assert ctx.publications["Bibcode"].tolist() == ["fresh-pub"]
    assert ctx.refs["Bibcode"].tolist() == ["fresh-ref"]
    assert "Title_en" in ctx.publications.columns
    assert "Title_en" in ctx.refs.columns


def test_run_translate_stage_requires_export_when_no_inputs(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)

    monkeypatch.setattr(
        pipeline,
        "load_translated_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(pipeline.StagePrerequisiteError) as excinfo:
        pipeline.run_translate_stage(ctx)

    assert excinfo.value.required_stage == "export"


def test_run_tokenize_stage_prefers_current_translated_results_over_snapshot(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.publications = pd.DataFrame([{"Bibcode": "fresh-pub", "Title_en": "T", "Abstract_en": "A"}])
    ctx.refs = pd.DataFrame([{"Bibcode": "fresh-ref", "Title_en": "RT", "Abstract_en": "RA"}])

    monkeypatch.setattr(
        pipeline,
        "load_tokenized_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("stale tokenized snapshot should not load")),
    )
    monkeypatch.setattr(
        pipeline,
        "ensure_spacy_model",
        lambda **kwargs: ("en_core_web_md", object()),
    )
    monkeypatch.setattr(
        pipeline,
        "tokenize_texts",
        lambda df, **kwargs: df.assign(
            full_text=[f"{row.Title_en}. {row.Abstract_en}" for row in df.itertuples()],
            tokens=[["tok"] for _ in range(len(df))],
        ),
    )
    monkeypatch.setattr(pipeline, "save_tokenized_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "save_parquet", lambda *args, **kwargs: None)

    pipeline.run_tokenize_stage(ctx)

    assert ctx.publications["Bibcode"].tolist() == ["fresh-pub"]
    assert ctx.publications["tokens"].tolist() == [["tok"]]


def test_run_tokenize_stage_requires_translate_when_no_inputs(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)

    monkeypatch.setattr(
        pipeline,
        "load_tokenized_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(pipeline.StagePrerequisiteError) as excinfo:
        pipeline.run_tokenize_stage(ctx)

    assert excinfo.value.required_stage == "translate"


def test_run_author_disambiguation_stage_prefers_current_tokenized_results_over_snapshot(
    tmp_path,
    monkeypatch,
):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
            "author_disambiguation": {"enabled": False},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.publications = pd.DataFrame(
        [{"Bibcode": "fresh-pub", "Title_en": "T", "Abstract_en": "A", "full_text": "T. A", "tokens": [["tok"]]}]
    )
    ctx.refs = pd.DataFrame([{"Bibcode": "fresh-ref", "Title_en": "RT", "Abstract_en": "RA"}])
    saved: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(
        pipeline,
        "load_disambiguated_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("stale disambiguated snapshot should not load")),
    )
    monkeypatch.setattr(
        pipeline,
        "save_disambiguated_snapshot",
        lambda pubs, refs, **kwargs: saved.update({"pubs": pubs.copy(), "refs": refs.copy()}),
    )

    pipeline.run_author_disambiguation_stage(ctx)

    assert saved["pubs"]["Bibcode"].tolist() == ["fresh-pub"]
    assert saved["refs"]["Bibcode"].tolist() == ["fresh-ref"]


def test_run_author_disambiguation_stage_requires_tokenize_when_no_inputs(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
            "author_disambiguation": {"enabled": False},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)

    monkeypatch.setattr(
        pipeline,
        "load_disambiguated_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(pipeline.StagePrerequisiteError) as excinfo:
        pipeline.run_author_disambiguation_stage(ctx)

    assert excinfo.value.required_stage == "tokenize"


def test_run_embeddings_stage_uses_reporter_progress(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
            "topic_model": {
                "embedding_provider": "openrouter",
                "embedding_model": "google/gemini-embedding-001",
                "embedding_api_key": "key",
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.publications = pd.DataFrame(
        [
            {"Bibcode": "b1", "full_text": "alpha"},
            {"Bibcode": "b2", "full_text": "beta"},
            {"Bibcode": "b3", "full_text": "gamma"},
        ]
    )

    calls: dict[str, object] = {}

    class _FakeProgress:
        def __init__(self) -> None:
            self.updates: list[int] = []

        def update(self, amount: int = 1) -> None:
            self.updates.append(int(amount))

    class _FakeProgressContext:
        def __init__(self, progress: _FakeProgress) -> None:
            self._progress = progress

        def __enter__(self) -> _FakeProgress:
            return self._progress

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    class _FakeReporter:
        def __init__(self) -> None:
            self.progress_bar = _FakeProgress()

        def progress(self, *, total: int | None, desc: str):
            calls["total"] = total
            calls["desc"] = desc
            return _FakeProgressContext(self.progress_bar)

    ctx.reporter = _FakeReporter()

    def _fake_compute_embeddings(documents, **kwargs):
        calls["documents"] = list(documents)
        calls["show_progress"] = kwargs["show_progress"]
        kwargs["progress_callback"](2)
        kwargs["progress_callback"](1)
        return np.ones((len(documents), 4), dtype=np.float32)

    monkeypatch.setattr(pipeline, "compute_embeddings", _fake_compute_embeddings)

    pipeline.run_embeddings_stage(ctx)

    assert calls["desc"] == "embeddings"
    assert calls["total"] == 3
    assert calls["documents"] == ["alpha", "beta", "gamma"]
    assert calls["show_progress"] is False
    assert ctx.reporter.progress_bar.updates == [2, 1]
    assert ctx.embeddings is not None
    assert ctx.embeddings.shape == (3, 4)


def test_run_pipeline_topic_fit_uses_tokenized_snapshot_and_caches(tmp_path, monkeypatch):
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
            {"Bibcode": "b1", "Author": ["Doe, A."], "full_text": "alpha beta", "tokens": [["alpha", "beta"]], "Title_en": "A", "Abstract_en": "alpha", "Year": 2020},
            {"Bibcode": "b2", "Author": ["Roe, B."], "full_text": "gamma delta", "tokens": [["gamma", "delta"]], "Title_en": "B", "Abstract_en": "beta", "Year": 2021},
        ]
    )
    refs = pd.DataFrame(
        [
            {"Bibcode": "r1", "Author": ["Ref, A."], "Title_en": "R", "Abstract_en": "ref", "Year": 2019}
        ]
    )
    events: list[str] = []
    seen_prompts: list[str] = []

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
        lambda documents, reduced_5d, **kwargs: (
            seen_prompts.append(kwargs["llm_prompt"]),
            events.append("fit"),
            _DummyTopicModel(),
        )[-1],
    )
    monkeypatch.setattr(
        pipeline,
        "reduce_outliers",
        lambda topic_model, documents, topics, reduced_5d, **kwargs: np.asarray(topics),
    )

    pipeline._run_stage_for_pipeline(ctx, "topic_fit")

    assert events == ["embeddings", "reduction", "fit"]
    assert ctx.documents == ["alpha beta", "gamma delta"]
    assert ctx.publications is not None
    assert ctx.refs is not None
    assert ctx.topics.tolist() == [0, 1]
    assert list(ctx.topic_info["Name"]) == ["Topic 0", "Topic 1"]
    assert seen_prompts == [BERTOPIC_LABELING_PHYSICS]


def test_validate_stage_name_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Invalid stage"):
        pipeline.validate_stage_name("unknown")
