"""Runtime preflight checks for packaged CLI runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import spacy.util

from ads_bib._utils.huggingface_api import HUGGINGFACE_API_KEY_ENV_VARS
from ads_bib._utils.llama_server import inspect_llama_server_command, prepare_llama_server_command
from ads_bib._utils.model_specs import ModelSpec
from ads_bib.bootstrap import DEFAULT_FASTTEXT_MODEL_RELATIVE_PATH
from ads_bib.config import _module_is_available
from ads_bib.pipeline import PipelineConfig, STAGE_ORDER, StageName, prepare_pipeline_config, validate_stage_name
from ads_bib.topic_model._runtime import (
    BERTOPIC_LLM_PROVIDER_IMPORTS,
    EMBEDDING_PROVIDER_IMPORTS,
    TOPONYMY_EMBEDDING_PROVIDER_IMPORTS,
    TOPONYMY_LLM_PROVIDER_IMPORTS,
)

DoctorStatus = Literal["ok", "warn", "fail"]
_TOPIC_STAGES = frozenset({"embeddings", "reduction", "topic_fit", "topic_dataframe", "visualize", "curate"})
_PARQUET_STAGES = frozenset(
    {
        "export",
        "translate",
        "tokenize",
        "author_disambiguation",
        "embeddings",
        "reduction",
        "topic_fit",
        "topic_dataframe",
        "visualize",
        "curate",
        "citations",
    }
)


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: DoctorStatus
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]
    active_stages: tuple[StageName, ...]

    def has_failures(self) -> bool:
        return any(check.status == "fail" for check in self.checks)

    @property
    def ok_count(self) -> int:
        return sum(check.status == "ok" for check in self.checks)

    @property
    def warn_count(self) -> int:
        return sum(check.status == "warn" for check in self.checks)

    @property
    def fail_count(self) -> int:
        return sum(check.status == "fail" for check in self.checks)

    def failing_checks(self) -> tuple[DoctorCheck, ...]:
        return tuple(check for check in self.checks if check.status == "fail")


def _ok(name: str, detail: str) -> DoctorCheck:
    return DoctorCheck(name=name, status="ok", detail=detail)


def _warn(name: str, detail: str) -> DoctorCheck:
    return DoctorCheck(name=name, status="warn", detail=detail)


def _fail(name: str, detail: str) -> DoctorCheck:
    return DoctorCheck(name=name, status="fail", detail=detail)


def _resolve_project_root(config: PipelineConfig) -> Path:
    root = Path(config.run.project_root) if config.run.project_root else Path.cwd()
    return root.expanduser().resolve()


def _resolve_config_path(project_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _resolve_stage_slice(
    config: PipelineConfig,
    *,
    start_stage: StageName | str | None = None,
    stop_stage: StageName | str | None = None,
) -> tuple[StageName, StageName | None, tuple[StageName, ...]]:
    start = validate_stage_name(start_stage or config.run.start_stage)
    stop = validate_stage_name(stop_stage) if stop_stage is not None else config.run.stop_stage
    start_idx = STAGE_ORDER.index(start)
    if stop is None:
        return start, None, STAGE_ORDER[start_idx:]
    stop_idx = STAGE_ORDER.index(stop)
    if stop_idx < start_idx:
        raise ValueError("stop_stage must be after or equal to start_stage.")
    return start, stop, STAGE_ORDER[start_idx : stop_idx + 1]


def _spacy_model_installed(model_name: str) -> bool:
    return _module_is_available(model_name) or spacy.util.is_package(model_name)


def _module_check(label: str, module_name: str) -> DoctorCheck:
    if _module_is_available(module_name):
        return _ok(label, f"optional dependency '{module_name}' is available")
    return _fail(label, f"missing optional dependency '{module_name}'")


def _api_key_check(label: str, value: str | None, *, hint: str) -> DoctorCheck:
    if value:
        return _ok(label, "configured")
    return _fail(label, hint)


def _llama_server_runtime_check(
    name: str,
    *,
    config: PipelineConfig,
    project_root: Path,
    prepare_managed_runtime: bool,
) -> DoctorCheck:
    try:
        resolution = (
            prepare_llama_server_command(
                config.llama_server.command,
                project_root=project_root,
                gpu_layers=config.llama_server.gpu_layers,
            )
            if prepare_managed_runtime
            else inspect_llama_server_command(
                config.llama_server.command,
                project_root=project_root,
                gpu_layers=config.llama_server.gpu_layers,
            )
        )
    except Exception as exc:
        return _fail(name, str(exc))

    if resolution.command is not None:
        return _ok(name, resolution.detail)
    if resolution.source == "managed_pending":
        return _warn(name, resolution.detail)
    return _fail(name, resolution.detail)


def _collect_translate_checks(
    config: PipelineConfig,
    project_root: Path,
    *,
    prepare_managed_runtime: bool,
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    cfg = config.translate
    model_path = _resolve_config_path(project_root, cfg.fasttext_model)
    if model_path is None:
        checks.append(_fail("translate.fasttext_model", "translate.fasttext_model is required"))
    elif model_path.is_file():
        checks.append(_ok("translate.fasttext_model", f"found at {model_path}"))
    elif model_path == _resolve_config_path(project_root, DEFAULT_FASTTEXT_MODEL_RELATIVE_PATH):
        checks.append(
            _warn(
                "translate.fasttext_model",
                f"missing at {model_path}; ads-bib run will download the default model automatically",
            )
        )
    else:
        checks.append(
            _fail(
                "translate.fasttext_model",
                f"missing at {model_path}; download lid.176.bin there or set translate.fasttext_model to an existing file",
            )
        )

    provider = cfg.provider
    if provider == "openrouter":
        checks.append(
            _api_key_check(
                "translate.api_key",
                cfg.api_key,
                hint="missing OpenRouter key; set OPENROUTER_API_KEY or translate.api_key",
            )
        )
        checks.append(_module_check("translate.provider", "openai"))
    elif provider == "huggingface_api":
        checks.append(
            _api_key_check(
                "translate.api_key",
                cfg.api_key,
                hint=(
                    "missing Hugging Face key; set HF_TOKEN "
                    f"(aliases: {', '.join(HUGGINGFACE_API_KEY_ENV_VARS[1:])}) or translate.api_key"
                ),
            )
        )
        checks.append(_module_check("translate.provider", "huggingface_hub"))
    elif provider == "llama_server":
        checks.append(_module_check("translate.provider", "openai"))
        checks.append(
            _llama_server_runtime_check(
                "translate.llama_server.command",
                config=config,
                project_root=project_root,
                prepare_managed_runtime=prepare_managed_runtime,
            )
        )

        try:
            model_spec = ModelSpec.from_fields(
                model_repo=cfg.model_repo,
                model_file=cfg.model_file,
                model_path=cfg.model_path,
                legacy_value=cfg.model,
                field_label="translate.model",
            )
        except Exception as exc:
            checks.append(_fail("translate.model", f"{type(exc).__name__}: {exc}"))
        else:
            if model_spec.model_path is not None:
                local_model = _resolve_config_path(project_root, model_spec.model_path)
                if local_model is not None and local_model.is_file():
                    checks.append(_ok("translate.model", f"local GGUF found at {local_model}"))
                else:
                    checks.append(
                        _fail(
                            "translate.model",
                            f"local GGUF not found at {local_model}",
                        )
                    )
            else:
                checks.append(_module_check("translate.model", "huggingface_hub"))
    elif provider == "nllb":
        for module_name in ("ctranslate2", "transformers", "sentencepiece", "huggingface_hub"):
            checks.append(_module_check("translate.provider", module_name))

    if provider in {"openrouter", "huggingface_api", "nllb"} and not str(cfg.model or "").strip():
        checks.append(_fail("translate.model", f"provider '{provider}' requires translate.model"))
    elif str(cfg.model or "").strip():
        checks.append(_ok("translate.model", f"configured as {cfg.model}"))

    return checks


def _collect_tokenize_checks(config: PipelineConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    cfg = config.tokenize
    preferred_available = _spacy_model_installed(cfg.spacy_model)
    fallback_available = _spacy_model_installed(cfg.fallback_model)

    if preferred_available:
        checks.append(_ok("tokenize.spacy_model", f"spaCy model '{cfg.spacy_model}' is installed"))
        return checks

    if fallback_available:
        checks.append(
            _warn(
                "tokenize.spacy_model",
                f"'{cfg.spacy_model}' is missing; the run will fall back to '{cfg.fallback_model}'",
            )
        )
        return checks

    if cfg.auto_download:
        checks.append(
            _warn(
                "tokenize.spacy_model",
                f"'{cfg.spacy_model}' is missing; tokenize.auto_download=true will try to install it at runtime",
            )
        )
        return checks

    checks.append(
        _fail(
            "tokenize.spacy_model",
            f"neither '{cfg.spacy_model}' nor fallback '{cfg.fallback_model}' is installed",
        )
    )
    return checks


def _collect_author_disambiguation_checks() -> list[DoctorCheck]:
    return [_module_check("author_disambiguation", "author_name_disambiguation")]


def _collect_topic_checks(
    config: PipelineConfig,
    project_root: Path,
    active_stages: tuple[StageName, ...],
    *,
    prepare_managed_runtime: bool,
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    cfg = config.topic_model

    if cfg.backend == "bertopic":
        checks.append(_module_check("topic_model.backend", "bertopic"))
        embedding_imports = EMBEDDING_PROVIDER_IMPORTS
        llm_imports = BERTOPIC_LLM_PROVIDER_IMPORTS
    else:
        checks.append(_module_check("topic_model.backend", "toponymy"))
        checks.append(_module_check("topic_model.backend", "dask"))
        embedding_imports = TOPONYMY_EMBEDDING_PROVIDER_IMPORTS
        llm_imports = TOPONYMY_LLM_PROVIDER_IMPORTS

    if "reduction" in active_stages or any(stage in active_stages for stage in {"topic_fit", "topic_dataframe", "visualize", "curate"}):
        if cfg.reduction_method not in {"pacmap", "umap"}:
            checks.append(
                _fail(
                    "topic_model.reduction_method",
                    f"unsupported reduction method '{cfg.reduction_method}'",
                )
            )
        else:
            reduction_module = "pacmap" if cfg.reduction_method == "pacmap" else "umap"
            checks.append(_module_check("topic_model.reduction_method", reduction_module))

    if "topic_fit" in active_stages or any(stage in active_stages for stage in {"topic_dataframe", "visualize", "curate"}):
        if cfg.clustering_method not in {"fast_hdbscan", "hdbscan"}:
            checks.append(
                _fail(
                    "topic_model.clustering_method",
                    f"unsupported clustering method '{cfg.clustering_method}'",
                )
            )
        else:
            cluster_module = "fast_hdbscan" if cfg.clustering_method == "fast_hdbscan" else "hdbscan"
            checks.append(_module_check("topic_model.clustering_method", cluster_module))

    embedding_module = embedding_imports.get(cfg.embedding_provider)
    if embedding_module is not None:
        checks.append(_module_check("topic_model.embedding_provider", embedding_module))
    if cfg.embedding_provider == "openrouter":
        checks.append(
            _api_key_check(
                "topic_model.embedding_api_key",
                cfg.embedding_api_key,
                hint="missing OpenRouter embedding key; set OPENROUTER_API_KEY or topic_model.embedding_api_key",
            )
        )
    elif cfg.embedding_provider == "huggingface_api":
        checks.append(
            _api_key_check(
                "topic_model.embedding_api_key",
                cfg.embedding_api_key,
                hint=(
                    "missing Hugging Face embedding key; set HF_TOKEN "
                    f"(aliases: {', '.join(HUGGINGFACE_API_KEY_ENV_VARS[1:])}) or topic_model.embedding_api_key"
                ),
            )
        )

    llm_module = llm_imports.get(cfg.llm_provider)
    if llm_module is not None:
        checks.append(_module_check("topic_model.llm_provider", llm_module))
    if cfg.llm_provider == "openrouter":
        if cfg.backend == "toponymy":
            api_key = cfg.llm_api_key or cfg.embedding_api_key
            checks.append(
                _api_key_check(
                    "topic_model.llm_api_key",
                    api_key,
                    hint="missing OpenRouter key for Toponymy; set OPENROUTER_API_KEY or topic_model.*_api_key",
                )
            )
        else:
            checks.append(
                _api_key_check(
                    "topic_model.llm_api_key",
                    cfg.llm_api_key,
                    hint="missing OpenRouter labeling key; set OPENROUTER_API_KEY or topic_model.llm_api_key",
                )
            )
    elif cfg.llm_provider == "huggingface_api":
        checks.append(
            _api_key_check(
                "topic_model.llm_api_key",
                cfg.llm_api_key,
                hint=(
                    "missing Hugging Face labeling key; set HF_TOKEN "
                    f"(aliases: {', '.join(HUGGINGFACE_API_KEY_ENV_VARS[1:])}) or topic_model.llm_api_key"
                ),
            )
        )
    elif cfg.llm_provider == "llama_server":
        checks.append(
            _llama_server_runtime_check(
                "topic_model.llama_server.command",
                config=config,
                project_root=project_root,
                prepare_managed_runtime=prepare_managed_runtime,
            )
        )
        try:
            model_spec = ModelSpec.from_fields(
                model_repo=cfg.llm_model_repo,
                model_file=cfg.llm_model_file,
                model_path=cfg.llm_model_path,
                legacy_value=cfg.llm_model,
                field_label="topic_model.llm_model",
            )
        except Exception as exc:
            checks.append(_fail("topic_model.llm_model", f"{type(exc).__name__}: {exc}"))
        else:
            if model_spec.model_path is not None:
                local_model = _resolve_config_path(project_root, model_spec.model_path)
                if local_model is not None and local_model.is_file():
                    checks.append(_ok("topic_model.llm_model", f"local GGUF found at {local_model}"))
                else:
                    checks.append(_fail("topic_model.llm_model", f"local GGUF not found at {local_model}"))
            else:
                checks.append(_module_check("topic_model.llm_model", "huggingface_hub"))

    if "visualize" in active_stages and config.visualization.enabled:
        for module_name in ("datamapplot", "seaborn", "matplotlib"):
            checks.append(_module_check("visualization", module_name))

    return checks


def _collect_citations_checks(config: PipelineConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    if config.citations.output_format in {"gexf", "all"}:
        checks.append(_module_check("citations.output_format", "networkx"))
    return checks


def collect_doctor_report(
    config: PipelineConfig,
    *,
    start_stage: StageName | str | None = None,
    stop_stage: StageName | str | None = None,
    prepare_managed_runtime: bool = False,
) -> DoctorReport:
    """Collect stage-aware runtime checks for one effective pipeline config."""
    prepared = prepare_pipeline_config(config)
    project_root = _resolve_project_root(prepared)
    _, _, active_stages = _resolve_stage_slice(
        prepared,
        start_stage=start_stage,
        stop_stage=stop_stage,
    )

    checks: list[DoctorCheck] = []
    if "search" in active_stages:
        if str(prepared.search.query).strip():
            checks.append(_ok("search.query", "configured"))
        else:
            checks.append(
                _fail(
                    "search.query",
                    "missing; set search.query in YAML or pass --set search.query='...'",
                )
            )
        checks.append(
            _api_key_check(
                "search.ads_token",
                prepared.search.ads_token,
                hint="missing ADS token; set ADS_TOKEN/ADS_API_KEY or search.ads_token",
            )
        )

    if any(stage in _PARQUET_STAGES for stage in active_stages):
        checks.append(_module_check("pyarrow", "pyarrow"))

    if "translate" in active_stages and prepared.translate.enabled:
        checks.extend(
            _collect_translate_checks(
                prepared,
                project_root,
                prepare_managed_runtime=prepare_managed_runtime,
            )
        )

    if "tokenize" in active_stages and prepared.tokenize.enabled:
        checks.extend(_collect_tokenize_checks(prepared))

    if "author_disambiguation" in active_stages and prepared.author_disambiguation.enabled:
        checks.extend(_collect_author_disambiguation_checks())

    if any(stage in _TOPIC_STAGES for stage in active_stages):
        checks.extend(
            _collect_topic_checks(
                prepared,
                project_root,
                active_stages,
                prepare_managed_runtime=prepare_managed_runtime,
            )
        )

    if "citations" in active_stages:
        checks.extend(_collect_citations_checks(prepared))

    return DoctorReport(checks=tuple(checks), active_stages=active_stages)


def format_doctor_report(report: DoctorReport) -> str:
    """Render a compact text report for CLI output."""
    start_stage = report.active_stages[0]
    end_stage = report.active_stages[-1]
    lines = [f"Doctor report for stages {start_stage} -> {end_stage}"]
    for check in report.checks:
        lines.append(f"[{check.status.upper()}] {check.name}: {check.detail}")
    lines.append(
        "Summary: "
        f"{report.ok_count} ok, {report.warn_count} warn, {report.fail_count} fail"
    )
    return "\n".join(lines) + "\n"
