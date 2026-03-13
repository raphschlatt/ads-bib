from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import yaml

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
    assert "keybert_model" not in data["topic_model"]


def test_official_pipeline_config_directory_contains_four_presets():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "pipeline"
    assert sorted(path.name for path in config_dir.glob("*.yaml")) == [
        "hf_api.yaml",
        "local_cpu.yaml",
        "local_gpu.yaml",
        "openrouter.yaml",
    ]


def test_openrouter_pipeline_config_template_loads():
    config = pipeline.PipelineConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "pipeline" / "openrouter.yaml"
    )
    data = config.to_dict()

    assert data["run"]["start_stage"] == "search"
    assert data["run"]["stop_stage"] is None
    assert data["search"]["query"] == 'author:"Hawking, S*"'
    assert data["topic_model"]["llm_prompt_name"] == "physics"
    assert data["topic_model"]["gguf_embedding_pooling"] == "cls"
    assert data["author_disambiguation"]["enabled"] is False
    assert data["tokenize"]["spacy_model"] == "en_core_web_md"
    assert data["tokenize"]["fallback_model"] == "en_core_web_md"
    assert data["translate"]["model"] == "google/gemini-3.1-flash-lite-preview"
    assert data["topic_model"]["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert data["topic_model"]["llm_model"] == "google/gemini-3.1-flash-lite-preview"
    assert data["translate"]["fasttext_model"] == "data/models/lid.176.bin"


@pytest.mark.parametrize(
    (
        "config_name",
        "translate_provider",
        "translate_model",
        "embedding_provider",
        "embedding_model",
        "llm_provider",
        "llm_model",
        "gguf_pooling",
    ),
    [
        (
            "hf_api.yaml",
            "huggingface_api",
            "unsloth/Qwen2.5-72B-Instruct:featherless-ai",
            "huggingface_api",
            "Qwen/Qwen3-Embedding-8B",
            "huggingface_api",
            "unsloth/Qwen2.5-72B-Instruct:featherless-ai",
            "cls",
        ),
        (
            "local_cpu.yaml",
            "nllb",
            "data/models/nllb-200-distilled-600M-ct2-int8",
            "local",
            "google/embeddinggemma-300m",
            "gguf",
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "cls",
        ),
        (
            "local_gpu.yaml",
            "gguf",
            "mradermacher/translategemma-4b-it-GGUF:translategemma-4b-it.Q4_K_M.gguf",
            "local",
            "google/embeddinggemma-300m",
            "gguf",
            "unsloth/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q4_K_M.gguf",
            "cls",
        ),
    ],
)
def test_official_pipeline_config_templates_load(
    config_name,
    translate_provider,
    translate_model,
    embedding_provider,
    embedding_model,
    llm_provider,
    llm_model,
    gguf_pooling,
):
    config = pipeline.PipelineConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "pipeline" / config_name
    )

    assert config.translate.provider == translate_provider
    assert config.translate.model == translate_model
    assert config.search.query == 'author:"Hawking, S*"'
    assert config.translate.fasttext_model == "data/models/lid.176.bin"
    assert config.translate.max_workers == 8
    assert config.topic_model.embedding_provider == embedding_provider
    assert config.topic_model.embedding_model == embedding_model
    assert config.topic_model.embedding_batch_size == 32
    assert config.topic_model.embedding_max_workers == 8
    assert config.topic_model.llm_provider == llm_provider
    assert config.topic_model.llm_model == llm_model
    assert config.topic_model.gguf_embedding_pooling == gguf_pooling
    assert config.topic_model.params_5d == {
        "n_neighbors": 30,
        "metric": "angular",
        "random_state": 42,
    }
    assert config.topic_model.params_2d == {
        "n_neighbors": 30,
        "metric": "angular",
        "random_state": 42,
    }
    assert config.topic_model.cluster_params == {
        "min_cluster_size": 15,
        "min_samples": 3,
        "cluster_selection_method": "eom",
        "cluster_selection_epsilon": 0.05,
    }
    assert config.topic_model.min_df == 3
    assert config.topic_model.bertopic_label_max_tokens == 64
    assert config.citations.min_counts == {
        "direct": 3,
        "co_citation": 6,
        "bibliographic_coupling": 3,
        "author_co_citation": 5,
    }


def test_run_topic_fit_stage_uses_implicit_keybert_default(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"fasttext_model": str(tmp_path / "lid.176.bin")},
            "topic_model": {
                "backend": "bertopic",
                "llm_provider": "openrouter",
                "llm_model": "google/gemini-3.1-flash-lite-preview",
                "embedding_provider": "local",
                "embedding_model": "google/embeddinggemma-300m",
                "cluster_params": {"min_cluster_size": 2, "min_samples": 1},
                "min_df": 1,
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.documents = ["doc-a", "doc-b"]
    ctx.embeddings = np.ones((2, 4), dtype=np.float32)
    ctx.reduced_5d = np.ones((2, 5), dtype=np.float32)

    calls: dict[str, object] = {}

    class _FakeTopicModel:
        topics_ = [0, 1]

        def get_topic_info(self):
            return pd.DataFrame({"Topic": [0, 1], "Name": ["Topic 0", "Topic 1"]})

    def _fake_fit_bertopic(documents, reduced_5d, **kwargs):
        calls["documents"] = list(documents)
        calls["shape"] = reduced_5d.shape
        calls["kwargs"] = kwargs
        return _FakeTopicModel()

    monkeypatch.setattr(pipeline, "fit_bertopic", _fake_fit_bertopic)
    monkeypatch.setattr(
        pipeline,
        "reduce_outliers",
        lambda topic_model, documents, topics, reduced_5d, **kwargs: np.asarray(topics),
    )

    pipeline.run_topic_fit_stage(ctx)

    assert calls["documents"] == ["doc-a", "doc-b"]
    assert calls["shape"] == (2, 5)
    assert "keybert_model" not in calls["kwargs"]


def test_pipeline_config_allows_huggingface_api_for_bertopic():
    config = pipeline.PipelineConfig.from_dict(
        {
            "topic_model": {
                "backend": "bertopic",
                "embedding_provider": "huggingface_api",
                "llm_provider": "huggingface_api",
            }
        }
    )

    assert config.topic_model.embedding_provider == "huggingface_api"
    assert config.topic_model.llm_provider == "huggingface_api"


def test_pipeline_config_rejects_huggingface_api_for_toponymy():
    with pytest.raises(ValueError, match="Invalid provider 'huggingface_api'"):
        pipeline.PipelineConfig.from_dict(
            {
                "topic_model": {
                    "backend": "toponymy",
                    "embedding_provider": "huggingface_api",
                    "llm_provider": "local",
                }
            }
        )


def test_prepare_pipeline_config_injects_hf_keys(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    config = pipeline.PipelineConfig.from_dict(
        {
            "translate": {"provider": "huggingface_api", "api_key": None},
            "topic_model": {
                "backend": "bertopic",
                "embedding_provider": "huggingface_api",
                "embedding_api_key": None,
                "llm_provider": "huggingface_api",
                "llm_api_key": None,
            },
        }
    )

    prepared = pipeline.prepare_pipeline_config(config)

    assert prepared.translate.api_key == "hf-token"
    assert prepared.topic_model.embedding_api_key == "hf-token"
    assert prepared.topic_model.llm_api_key == "hf-token"


def test_pipeline_config_rejects_unknown_gguf_pooling():
    with pytest.raises(ValueError, match="Unknown GGUF pooling type"):
        pipeline.PipelineConfig.from_dict(
            {
                "topic_model": {
                    "embedding_provider": "gguf",
                    "gguf_embedding_pooling": "bogus",
                }
            }
        )


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
    monkeypatch.setattr(pipeline, "_finalize_run_summary", lambda *args, **kwargs: None)
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


def test_run_pipeline_writes_summary_for_partial_cli_run(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"fasttext_model": str(tmp_path / "lid.176.bin")},
        }
    )
    run = pipeline.RunManager(run_name="cli_partial", project_root=tmp_path)

    monkeypatch.setattr(
        pipeline,
        "_STAGE_FUNCS",
        {name: (lambda ctx, _name=name: ctx) for name in pipeline.STAGE_ORDER},
    )

    pipeline.run_pipeline(
        config,
        stop_stage="translate",
        run=run,
        load_environment=False,
    )

    summary = yaml.safe_load((run.paths["root"] / "run_summary.yaml").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 2
    assert summary["run"]["status"] == "completed"
    assert summary["stages"]["requested_start_stage"] == "search"
    assert summary["stages"]["requested_stop_stage"] == "translate"
    assert summary["stages"]["completed_stages"] == ["search", "export", "translate"]
    assert summary["stages"]["failed_stage"] is None


def test_run_pipeline_writes_failed_summary_on_stage_error(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"fasttext_model": str(tmp_path / "lid.176.bin")},
        }
    )
    run = pipeline.RunManager(run_name="cli_failed", project_root=tmp_path)

    def _runner(stage_name):
        def _run(ctx):
            if stage_name == "tokenize":
                raise RuntimeError("boom")
            return ctx

        return _run

    monkeypatch.setattr(pipeline, "_STAGE_FUNCS", {name: _runner(name) for name in pipeline.STAGE_ORDER})

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run_pipeline(
            config,
            stop_stage="tokenize",
            run=run,
            load_environment=False,
        )

    summary = yaml.safe_load((run.paths["root"] / "run_summary.yaml").read_text(encoding="utf-8"))
    assert summary["run"]["status"] == "failed"
    assert summary["run"]["error"] == "RuntimeError: boom"
    assert summary["stages"]["requested_start_stage"] == "search"
    assert summary["stages"]["requested_stop_stage"] == "tokenize"
    assert summary["stages"]["completed_stages"] == ["search", "export", "translate"]
    assert summary["stages"]["failed_stage"] == "tokenize"


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
