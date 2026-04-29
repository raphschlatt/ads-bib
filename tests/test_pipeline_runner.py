from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import yaml

import numpy as np
import pandas as pd
import pytest

import ads_bib._dataset_bundle as dataset_bundle
import ads_bib.pipeline as pipeline
from ads_bib.presets import get_preset_names, get_preset_summary, load_preset_config
from ads_bib.prompts import BERTOPIC_LABELING_PHYSICS, TOPONYMY_LABELING_PHYSICS


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


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"backend": "remote"}, "author_disambiguation.backend"),
        ({"runtime": "cuda"}, "author_disambiguation.runtime"),
        ({"modal_gpu": "a100"}, "author_disambiguation.modal_gpu"),
    ],
)
def test_pipeline_config_validates_author_disambiguation_options(section, message):
    with pytest.raises(ValueError, match=message):
        pipeline.PipelineConfig.from_dict({"author_disambiguation": section})


def test_package_preset_registry_contains_four_presets():
    assert get_preset_names() == (
        "openrouter",
        "hf_api",
        "local_cpu",
        "local_gpu",
    )
    assert "Package-managed local CPU road" in get_preset_summary("local_cpu")
    assert "Package-managed local GPU road" in get_preset_summary("local_gpu")



def test_openrouter_package_preset_loads():
    config = load_preset_config("openrouter")
    data = config.to_dict()

    assert data["run"]["start_stage"] == "search"
    assert data["run"]["stop_stage"] is None
    assert data["run"]["run_name"] == "ads_bib_openrouter"
    assert data["search"]["query"] == ""
    assert data["topic_model"]["llm_prompt_name"] == "physics"
    assert data["author_disambiguation"]["enabled"] is False
    assert data["tokenize"]["spacy_model"] == "en_core_web_md"
    assert data["tokenize"]["fallback_model"] == "en_core_web_md"
    assert data["translate"]["model"] == "google/gemini-3-flash-preview"
    assert data["translate"]["model_repo"] is None
    assert data["translate"]["model_file"] is None
    assert data["translate"]["model_path"] is None
    assert data["llama_server"]["command"] == "llama-server"
    assert data["llama_server"]["reasoning"] == "off"
    assert data["topic_model"]["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert data["topic_model"]["llm_model"] == "google/gemini-3-flash-preview"
    assert data["topic_model"]["llm_model_repo"] is None
    assert data["topic_model"]["llm_model_file"] is None
    assert data["topic_model"]["llm_model_path"] is None
    assert data["topic_model"]["toponymy_layer_index"] == "auto"
    assert data["topic_model"]["toponymy_local_label_max_tokens"] == 128
    assert data["visualization"]["font_family"] == "Cinzel"
    assert data["visualization"]["title"] == "ADS Topic Map"
    assert data["visualization"]["topic_tree"] is False
    assert data["curation"]["cluster_targets"] == []
    assert data["citations"]["cited_authors_exclude"] is None
    assert data["translate"]["fasttext_model"] == "data/models/lid.176.bin"


@pytest.mark.parametrize(
    (
        "config_name",
        "translate_provider",
        "translate_model",
        "translate_model_repo",
        "translate_model_file",
        "embedding_provider",
        "embedding_model",
        "llm_provider",
        "llm_model",
        "llm_model_repo",
        "llm_model_file",
        "llm_model_path",
    ),
    [
        (
            "hf_api.yaml",
            "huggingface_api",
            "unsloth/Qwen2.5-72B-Instruct:featherless-ai",
            None,
            None,
            "huggingface_api",
            "Qwen/Qwen3-Embedding-8B",
            "huggingface_api",
            "unsloth/Qwen2.5-72B-Instruct:featherless-ai",
            None,
            None,
            None,
        ),
        (
            "local_cpu.yaml",
            "nllb",
            "JustFrederik/nllb-200-distilled-600M-ct2-int8",
            None,
            None,
            "local",
            "google/embeddinggemma-300m",
            "llama_server",
            None,
            "mradermacher/Qwen3.5-0.8B-GGUF",
            "Qwen3.5-0.8B.Q4_K_M.gguf",
            None,
        ),
        (
            "local_gpu.yaml",
            "transformers",
            "google/translategemma-4b-it",
            None,
            None,
            "local",
            "google/embeddinggemma-300m",
            "local",
            "google/gemma-3-1b-it",
            None,
            None,
            None,
        ),
    ],
)
def test_official_pipeline_config_templates_load(
    config_name,
    translate_provider,
    translate_model,
    translate_model_repo,
    translate_model_file,
    embedding_provider,
    embedding_model,
    llm_provider,
    llm_model,
    llm_model_repo,
    llm_model_file,
    llm_model_path,
):
    config = load_preset_config(config_name.removesuffix(".yaml"))

    assert config.translate.provider == translate_provider
    assert config.translate.model == translate_model
    assert config.translate.model_repo == translate_model_repo
    assert config.translate.model_file == translate_model_file
    assert config.translate.model_path is None
    assert config.search.query == ""
    assert config.translate.fasttext_model == "data/models/lid.176.bin"
    assert config.translate.max_workers == 8
    assert config.llama_server.command == "llama-server"
    expected_gpu_layers = 0 if config_name == "local_cpu.yaml" else -1
    if config_name in {"local_cpu.yaml", "local_gpu.yaml"}:
        assert config.llama_server.gpu_layers == expected_gpu_layers
    assert config.topic_model.embedding_provider == embedding_provider
    assert config.topic_model.embedding_model == embedding_model
    assert config.topic_model.embedding_batch_size == 32
    assert config.topic_model.embedding_max_workers == 8
    assert config.topic_model.llm_provider == llm_provider
    assert config.topic_model.llm_model == llm_model
    assert config.topic_model.llm_model_repo == llm_model_repo
    assert config.topic_model.llm_model_file == llm_model_file
    assert config.topic_model.llm_model_path == llm_model_path
    assert config.topic_model.toponymy_layer_index == "auto"
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
    assert config.topic_model.toponymy_local_label_max_tokens == 128
    assert config.visualization.font_family == "Cinzel"
    assert config.visualization.title == "ADS Topic Map"
    assert config.visualization.topic_tree is False
    assert config.curation.cluster_targets == []
    assert config.citations.min_counts == {
        "direct": 2,
        "co_citation": 3,
        "bibliographic_coupling": 2,
        "author_co_citation": 3,
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


def test_run_topic_fit_stage_uses_bertopic_progress_bridge(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"fasttext_model": str(tmp_path / "lid.176.bin")},
            "topic_model": {
                "backend": "bertopic",
                "llm_provider": "llama_server",
                "llm_model_path": "data/models/qwen35_gguf/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf",
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

    calls: dict[str, object] = {"descs": [], "details": []}

    class _FakeTopicModel:
        topics_ = [0, 1]

        def get_topic_info(self):
            return pd.DataFrame({"Topic": [0, 1], "Name": ["Topic 0", "Topic 1"]})

    @contextmanager
    def _fake_bridge(*, reporter, desc: str):
        calls["descs"].append(desc)
        assert reporter is ctx.reporter
        yield

    class _FakeReporter:
        output_mode = "cli"

        def detail(self, message: str, *args: object) -> None:
            calls["details"].append(message % args if args else message)

    ctx.reporter = _FakeReporter()

    monkeypatch.setattr(pipeline.topic_model_backends, "_bridge_bertopic_label_progress", _fake_bridge)
    monkeypatch.setattr(pipeline, "fit_bertopic", lambda documents, reduced_5d, **kwargs: _FakeTopicModel())
    monkeypatch.setattr(
        pipeline,
        "reduce_outliers",
        lambda topic_model, documents, topics, reduced_5d, **kwargs: np.asarray(topics),
    )

    pipeline.run_topic_fit_stage(ctx)

    assert calls["descs"] == ["fit", "outlier refresh"]
    assert calls["details"] == [
        "preparing BERTopic clustering and label generation",
        "reassigning outliers before topic-label refresh",
    ]


def test_resolve_topic_defaults_scales_toponymy_min_clusters_for_small_corpus(tmp_path):
    config = pipeline.PipelineConfig.from_dict({"run": {"project_root": str(tmp_path)}})
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.documents = [f"doc-{idx}" for idx in range(360)]

    resolved = pipeline._resolve_topic_defaults(ctx)

    assert resolved["toponymy_cluster_params"]["min_clusters"] == 3
    assert resolved["toponymy_cluster_params"]["base_min_cluster_size"] == 10


def test_resolve_topic_defaults_keeps_toponymy_overrides_authoritative(tmp_path):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "topic_model": {
                "toponymy_cluster_params": {"min_clusters": 7, "base_min_cluster_size": 12},
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.documents = [f"doc-{idx}" for idx in range(360)]

    resolved = pipeline._resolve_topic_defaults(ctx)

    assert resolved["toponymy_cluster_params"]["min_clusters"] == 7
    assert resolved["toponymy_cluster_params"]["base_min_cluster_size"] == 12


def test_warn_if_aggressive_toponymy_config_logs_warning(monkeypatch):
    calls: dict[str, str] = {}

    def _fake_warning(message, *args):
        calls["message"] = message % args

    monkeypatch.setattr(pipeline.logger, "warning", _fake_warning)

    pipeline._warn_if_aggressive_toponymy_config(
        backend="toponymy",
        n_docs=120,
        clusterer_params={"min_clusters": 10, "base_min_cluster_size": 200},
    )

    assert "Toponymy config may be too aggressive" in calls["message"]


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


def test_pipeline_config_allows_huggingface_api_for_toponymy():
    config = pipeline.PipelineConfig.from_dict(
        {
            "topic_model": {
                "backend": "toponymy",
                "embedding_provider": "huggingface_api",
                "llm_provider": "huggingface_api",
            }
        }
    )

    assert config.topic_model.embedding_provider == "huggingface_api"
    assert config.topic_model.llm_provider == "huggingface_api"


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


def test_pipeline_config_rejects_legacy_llama_server_model_string():
    with pytest.raises(ValueError, match="Legacy topic_model.llm_model value"):
        pipeline.PipelineConfig.from_dict(
            {
                "topic_model": {
                    "llm_provider": "llama_server",
                    "llm_model": "unsloth/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q4_K_M.gguf",
                }
            }
        )


def test_pipeline_config_defaults_toponymy_layer_index_to_auto():
    config = pipeline.PipelineConfig.from_dict({})
    assert config.topic_model.toponymy_layer_index == "auto"


def test_pipeline_config_normalizes_null_toponymy_layer_index_to_auto():
    config = pipeline.PipelineConfig.from_dict({"topic_model": {"toponymy_layer_index": None}})
    assert config.topic_model.toponymy_layer_index == "auto"


def test_pipeline_config_accepts_string_toponymy_layer_index_auto():
    config = pipeline.PipelineConfig.from_dict({"topic_model": {"toponymy_layer_index": "auto"}})
    assert config.topic_model.toponymy_layer_index == "auto"


def test_pipeline_config_accepts_string_toponymy_layer_index_int():
    config = pipeline.PipelineConfig.from_dict({"topic_model": {"toponymy_layer_index": "2"}})
    assert config.topic_model.toponymy_layer_index == 2


def test_pipeline_config_defaults_visualization_topic_tree_to_false():
    config = pipeline.PipelineConfig.from_dict({})
    assert config.visualization.topic_tree is False


def test_pipeline_config_normalizes_visualization_topic_tree_auto_to_false():
    config = pipeline.PipelineConfig.from_dict({"visualization": {"topic_tree": "auto"}})
    assert config.visualization.topic_tree is False


def test_pipeline_config_normalizes_curation_cluster_targets():
    config = pipeline.PipelineConfig.from_dict(
        {
            "curation": {
                "cluster_targets": [
                    {"layer": "1", "cluster_id": "3"},
                    {"layer": 0, "cluster_id": -1},
                ]
            }
        }
    )

    assert config.curation.cluster_targets == [
        {"layer": 1, "cluster_id": 3},
        {"layer": 0, "cluster_id": -1},
    ]


def test_summary_lines_for_topic_fit_include_toponymy_hierarchy():
    ctx = SimpleNamespace(
        topics=np.asarray([0, 0, 1, -1]),
        topic_info=pd.DataFrame(
            {
                "Topic": [-1, 0, 1],
                "Name": ["Outlier Topic", "Macro Alpha", "Macro Beta"],
            }
        ),
        topic_hierarchy={
            "topic_layer_count": 2,
            "topic_primary_layer_index": 1,
            "topic_clusters_per_layer": [2, 2],
            "topic_primary_layer_selection": "auto",
        },
        config=SimpleNamespace(topic_model=SimpleNamespace(backend="toponymy")),
    )

    lines = pipeline._summary_lines_for_stage(ctx, "topic_fit")

    assert lines == [
        "backend: toponymy | layers: 2 | primary_layer: 1 (auto) | "
        "clusters/layer: [2, 2] | topics: 2 | outliers: 1"
    ]


def test_run_curate_stage_uses_layer_aware_cluster_targets_for_toponymy(tmp_path):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "topic_model": {"backend": "toponymy"},
            "curation": {"cluster_targets": [{"layer": 1, "cluster_id": 20}]},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.topic_hierarchy = {"topic_primary_layer_index": 1}
    ctx.topic_df = pd.DataFrame(
        {
            "topic_id": [20, 20, 30, -1],
            "Name": ["Macro A", "Macro A", "Macro B", "Outlier Topic"],
            "topic_primary_layer_index": [1, 1, 1, 1],
            "topic_layer_0_id": [100, 101, 200, -1],
            "topic_layer_0_label": ["Alpha", "Beta", "Gamma", "Unlabelled"],
            "topic_layer_1_id": [20, 20, 30, -1],
            "topic_layer_1_label": ["Macro A", "Macro A", "Macro B", "Unlabelled"],
        }
    )

    pipeline.run_curate_stage(ctx)

    assert ctx.curated_df is not None
    assert ctx.curated_df["topic_layer_1_id"].tolist() == [30, -1]


def test_run_curate_stage_maps_legacy_clusters_to_remove_to_working_layer_for_toponymy(tmp_path):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "topic_model": {"backend": "toponymy"},
            "curation": {"clusters_to_remove": [20]},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.topic_hierarchy = {"topic_primary_layer_index": 1}
    ctx.topic_df = pd.DataFrame(
        {
            "topic_id": [20, 20, 30, -1],
            "Name": ["Macro A", "Macro A", "Macro B", "Outlier Topic"],
            "topic_primary_layer_index": [1, 1, 1, 1],
            "topic_layer_0_id": [100, 101, 200, -1],
            "topic_layer_0_label": ["Alpha", "Beta", "Gamma", "Unlabelled"],
            "topic_layer_1_id": [20, 20, 30, -1],
            "topic_layer_1_label": ["Macro A", "Macro A", "Macro B", "Unlabelled"],
        }
    )

    pipeline.run_curate_stage(ctx)

    assert ctx.curated_df is not None
    assert ctx.curated_df["topic_layer_1_id"].tolist() == [30, -1]


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
    assert summary["run"]["started_at_utc"] is not None
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
    monkeypatch.setattr(dataset_bundle, "save_parquet", lambda *args, **kwargs: None)

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


def test_run_author_disambiguation_stage_uses_default_ads_and_bundle(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
            "author_disambiguation": {"enabled": True},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.publications = pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "Author": ["Hawking, S."],
                "Year": 1974,
                "Title_en": "Black holes",
                "Abstract_en": "Evaporation",
                "tokens": [["black", "hole"]],
            }
        ]
    )
    ctx.refs = pd.DataFrame(
        [
            {
                "Bibcode": "r1",
                "Author": ["Hawking, S."],
                "Year": 1973,
                "Title_en": "Gravity",
                "Abstract_en": "Classical",
            }
        ]
    )
    calls: dict[str, object] = {}

    def _fake_apply(publications, references, **kwargs):
        calls.update(kwargs)
        return (
            publications.assign(author_uids=[["uid:hawking"]], author_display_names=[["Hawking, S."]]),
            references.assign(author_uids=[["uid:hawking"]], author_display_names=[["Hawking, S."]]),
        )

    monkeypatch.setattr(pipeline, "apply_author_disambiguation", _fake_apply)

    pipeline.run_author_disambiguation_stage(ctx)

    assert calls["backend"] == "local"
    assert calls["runtime"] == "auto"
    assert calls["modal_gpu"] == "l4"
    assert calls["model_bundle"] is None
    assert calls["dataset_id"] == ctx.run.run_id


def test_run_author_disambiguation_stage_rejects_missing_author_uid_outputs(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
            "author_disambiguation": {"enabled": True},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.publications = pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "Author": ["Hawking, S."],
                "Year": 1974,
                "Title_en": "Black holes",
                "Abstract_en": "Evaporation",
                "tokens": [["black", "hole"]],
            }
        ]
    )
    ctx.refs = pd.DataFrame(
        [{"Bibcode": "r1", "Author": ["Hawking, S."], "Year": 1973, "Title_en": "Gravity", "Abstract_en": "Classical"}]
    )
    monkeypatch.setattr(
        pipeline,
        "apply_author_disambiguation",
        lambda publications, references, **kwargs: (publications.copy(), references.copy()),
    )

    with pytest.raises(ValueError, match="did not produce author_uids"):
        pipeline.run_author_disambiguation_stage(ctx)


def test_run_author_disambiguation_stage_bridges_local_progress(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
            "author_disambiguation": {"enabled": True},
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.publications = pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "Author": ["Hawking, S."],
                "Year": 1974,
                "Title_en": "Black holes",
                "Abstract_en": "Evaporation",
                "tokens": [["black", "hole"]],
            }
        ]
    )
    ctx.refs = pd.DataFrame(columns=["Bibcode", "Author", "Year", "Title_en", "Abstract_en"])

    class _FakeProgress:
        def __init__(self) -> None:
            self.total = 0
            self.n = 0
            self.updates: list[int] = []

        def update(self, amount: int = 1) -> None:
            self.updates.append(int(amount))
            self.n += int(amount)

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
            self.calls: list[tuple[int | None, str]] = []

        def progress(self, *, total: int | None, desc: str):
            self.progress_bar.total = int(total or 0)
            self.calls.append((total, desc))
            return _FakeProgressContext(self.progress_bar)

    ctx.reporter = _FakeReporter()

    def _fake_apply(publications, references, **kwargs):
        assert kwargs["progress"] is True
        handler = kwargs["progress_handler"]
        assert callable(handler)
        handler(SimpleNamespace(kind="stage_done", stage_key="load"))
        handler(SimpleNamespace(kind="run_done"))
        return (
            publications.assign(author_uids=[["uid:hawking"]], author_display_names=[["Hawking, S."]]),
            references.assign(author_uids=[], author_display_names=[]),
        )

    monkeypatch.setattr(pipeline, "apply_author_disambiguation", _fake_apply)

    pipeline.run_author_disambiguation_stage(ctx)

    assert ctx.reporter.calls == [(8, "disambiguate")]
    assert ctx.reporter.progress_bar.updates == [1, 7]


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


def test_run_embeddings_stage_projects_topic_workframe_columns(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
            "topic_model": {
                "embedding_provider": "local",
                "embedding_model": "mini",
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.publications = pd.DataFrame(
        [
            {
                "Bibcode": "b1",
                "References": ["r1"],
                "Author": ["Doe, A."],
                "Year": 2020,
                "Journal": "Journal A",
                "Title_lang": "en",
                "Title_en": "Alpha",
                "Abstract_lang": "en",
                "Abstract_en": "Alpha abstract",
                "full_text": "alpha",
                "tokens": [["alpha"]],
                "Citation Count": 3,
                "Internal Note": "drop-me",
            }
        ]
    )

    monkeypatch.setattr(
        pipeline,
        "compute_embeddings",
        lambda documents, **kwargs: np.ones((len(documents), 4), dtype=np.float32),
    )

    pipeline.run_embeddings_stage(ctx)

    assert ctx.topic_input_df is not None
    assert list(ctx.topic_input_df.columns) == [
        "Bibcode",
        "References",
        "Author",
        "Year",
        "Journal",
        "Title_lang",
        "Title_en",
        "Abstract_lang",
        "Abstract_en",
        "Citation Count",
        "full_text",
        "tokens",
    ]
    assert "Internal Note" not in ctx.topic_input_df.columns
    assert ctx.documents == ["alpha"]


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


def test_run_topic_fit_stage_passes_toponymy_prompt_instructions(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "token"},
            "translate": {"fasttext_model": str(tmp_path / "lid.176.bin")},
            "topic_model": {
                "backend": "toponymy",
                "llm_provider": "openrouter",
                "llm_model": "google/gemini-3-flash-preview",
                "embedding_provider": "local",
                "embedding_model": "google/embeddinggemma-300m",
                "outlier_threshold": 0,
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.documents = ["doc-a", "doc-b"]
    ctx.embeddings = np.ones((2, 4), dtype=np.float32)
    ctx.reduced_5d = np.ones((2, 5), dtype=np.float32)

    calls: dict[str, object] = {}
    fake_model = SimpleNamespace(topic_names_=[["Topic 0", "Topic 1"]], topic_primary_layer_index_=0)

    def _fake_fit_toponymy(documents, embeddings, clusterable_vectors, **kwargs):
        calls["documents"] = list(documents)
        calls["embedding_shape"] = embeddings.shape
        calls["clusterable_shape"] = clusterable_vectors.shape
        calls["kwargs"] = kwargs
        return fake_model, np.array([0, 1]), pd.DataFrame(
            {"Topic": [0, 1], "Name": ["Topic 0", "Topic 1"]}
        )

    monkeypatch.setattr(pipeline, "fit_toponymy", _fake_fit_toponymy)

    pipeline.run_topic_fit_stage(ctx)

    assert calls["documents"] == ["doc-a", "doc-b"]
    assert calls["embedding_shape"] == (2, 4)
    assert calls["clusterable_shape"] == (2, 5)
    assert calls["kwargs"]["llm_prompt"] == TOPONYMY_LABELING_PHYSICS


def test_visualization_templates_use_corpus_counts(tmp_path):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": 'author:"Hawking, S*"'},
            "translate": {"fasttext_model": str(tmp_path / "lid.176.bin")},
            "topic_model": {
                "llm_provider": "openrouter",
                "llm_model": "google/gemini-3-flash-preview",
            },
        }
    )
    ctx = SimpleNamespace(
        config=config,
        topic_df=pd.DataFrame({"topic_id": [0, 0, 1, -1]}),
        topic_info=pd.DataFrame({"Topic": [0, 1, -1], "Name": ["A", "B", "Outlier Topic"]}),
    )

    assert pipeline._topic_title(ctx) == "ADS Topic Map"
    assert pipeline._topic_subtitle(ctx) == "2 topics from 4 ADS records"


def test_run_citations_stage_uses_explicit_cited_author_excludes(tmp_path, monkeypatch):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": '(author:"Hawking, S*") OR author:"Penrose, R*"'},
            "translate": {"fasttext_model": str(tmp_path / "lid.176.bin")},
            "citations": {
                "cited_authors_exclude": ["Ellis, G"],
                "cited_author_uids_exclude": ["uid:hawking"],
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    ctx.curated_df = pd.DataFrame({"Bibcode": ["PUB-1"]})
    ctx.refs = pd.DataFrame({"Bibcode": ["PUB-1"], "RefBibcode": ["REF-1"]})
    ctx.author_entities = pd.DataFrame([{"author_uid": "uid:hawking"}])

    calls: dict[str, object] = {}

    def _fake_prepare_citation_publications(publications, refs, **kwargs):
        calls["prepare_kwargs"] = kwargs
        return publications

    def _fake_build_citation_inputs_from_publications(publications):
        calls["filtered_publications"] = publications
        return ["PUB-1"], [["REF-1"]]

    def _fake_build_all_nodes(publications, refs):
        calls["all_nodes_args"] = (publications, refs)
        return pd.DataFrame({"node": []})

    def _fake_process_all_citations(**kwargs):
        calls["process_kwargs"] = kwargs
        return {"direct": pd.DataFrame()}

    def _fake_export_wos_format(publications, refs, *, output_path):
        calls["wos_output_path"] = output_path

    monkeypatch.setattr(pipeline, "prepare_citation_publications", _fake_prepare_citation_publications)
    monkeypatch.setattr(
        pipeline,
        "build_citation_inputs_from_publications",
        _fake_build_citation_inputs_from_publications,
    )
    monkeypatch.setattr(pipeline, "build_all_nodes", _fake_build_all_nodes)
    monkeypatch.setattr(pipeline, "process_all_citations", _fake_process_all_citations)
    monkeypatch.setattr(pipeline, "export_wos_format", _fake_export_wos_format)

    pipeline.run_citations_stage(ctx)

    assert calls["prepare_kwargs"] == {
        "authors_filter": None,
        "authors_filter_uids": None,
        "cited_authors_exclude": ["Ellis, G"],
        "cited_author_uids_exclude": ["uid:hawking"],
    }
    assert calls["process_kwargs"]["cited_authors_exclude"] == ["Ellis, G"]
    assert calls["process_kwargs"]["cited_author_uids_exclude"] == ["uid:hawking"]
    assert calls["process_kwargs"]["author_entities"] is ctx.author_entities
    assert Path(calls["wos_output_path"]).name == "download_wos_export_filtered.txt"


def test_validate_stage_name_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Invalid stage"):
        pipeline.validate_stage_name("unknown")
