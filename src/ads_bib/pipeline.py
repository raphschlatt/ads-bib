"""Shared pipeline orchestration for notebook and CLI frontends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import logging
import os
from pathlib import Path
import random
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from . import prompts
from ads_bib._utils.huggingface_api import resolve_huggingface_api_key
from ads_bib._utils.llama_server import LlamaServerConfig
from ads_bib._utils.model_specs import ModelSpec
from ads_bib._utils.checkpoints import (
    load_disambiguated_snapshot,
    load_tokenized_snapshot,
    load_translated_snapshot,
    save_disambiguated_snapshot,
    save_tokenized_snapshot,
    save_translated_snapshot,
)
from ads_bib._utils.costs import CostTracker
from ads_bib._utils.io import load_parquet, save_parquet
from ads_bib._utils.logging import (
    OutputMode,
    StageReporter,
    capture_external_output,
    configure_runtime_logging,
)
from ads_bib.author_disambiguation import apply_author_disambiguation
from ads_bib.citations import (
    build_all_nodes,
    build_citation_inputs_from_publications,
    export_wos_format,
    process_all_citations,
)
from ads_bib.config import init_paths, load_env, validate_provider
from ads_bib.curate import (
    get_cluster_summary,
    get_hierarchy_cluster_summary,
    normalize_cluster_targets,
    remove_cluster_targets,
    remove_clusters,
)
from ads_bib.export import resolve_dataset
from ads_bib.run_manager import RunManager
from ads_bib.search import search_ads
from ads_bib.tokenize import ensure_spacy_model, tokenize_texts
from ads_bib.topic_model import (
    build_topic_dataframe,
    compute_embeddings,
    fit_bertopic,
    fit_toponymy,
    reduce_dimensions,
    reduce_outliers,
)
from ads_bib.topic_model import backends as topic_model_backends
from ads_bib.topic_model._runtime import (
    BERTOPIC_LLM_PROVIDERS,
    EMBEDDING_PROVIDERS,
    TOPONYMY_EMBEDDING_PROVIDERS,
    TOPONYMY_LLM_PROVIDERS,
)
from ads_bib.translate import detect_languages, translate_dataframe

logger = logging.getLogger(__name__)

StageName = Literal[
    "search",
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
]

STAGE_ORDER: tuple[StageName, ...] = (
    "search",
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
)


class StagePrerequisiteError(RuntimeError):
    """Raised when a strict stage is missing its required upstream state."""

    def __init__(
        self,
        stage: StageName,
        required_stage: StageName | None,
        message: str,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.required_stage = required_stage


def _prompt_value(name: str) -> str:
    return str(getattr(prompts, name))


_PROMPT_MAP: dict[str, str] = {
    "physics": _prompt_value("BERTOPIC_LABELING_PHYSICS"),
    "generic": _prompt_value("BERTOPIC_LABELING_GENERIC"),
}


@dataclass
class RunConfig:
    run_name: str = "ADS_Curation_Run"
    start_stage: StageName = "search"
    stop_stage: StageName | None = None
    random_seed: int = 42
    openrouter_cost_mode: str = "hybrid"
    project_root: str | None = None


@dataclass
class SearchConfig:
    query: str = ""
    ads_token: str | None = None
    refresh_search: bool = True
    refresh_export: bool = True


@dataclass
class TranslateConfig:
    enabled: bool = True
    provider: str = "openrouter"
    model: str | None = "google/gemini-3-flash-preview"
    model_repo: str | None = None
    model_file: str | None = None
    model_path: str | None = None
    api_key: str | None = None
    max_workers: int = 10
    max_tokens: int = 2048
    fasttext_model: str | None = None


@dataclass
class TokenizeConfig:
    enabled: bool = True
    spacy_model: str = "en_core_web_md"
    batch_size: int = 512
    n_process: int = 1
    disable: tuple[str, ...] = ("ner", "parser", "textcat")
    fallback_model: str = "en_core_web_md"
    auto_download: bool = True


@dataclass
class AuthorDisambiguationConfig:
    enabled: bool = False
    model_bundle: str | None = None
    dataset_id: str | None = None
    force_refresh: bool = False
    infer_stage: str = "full"


@dataclass
class TopicModelConfig:
    sample_size: int | None = None
    embedding_provider: str = "openrouter"
    embedding_model: str = "google/gemini-embedding-001"
    embedding_api_key: str | None = None
    embedding_batch_size: int = 64
    embedding_max_workers: int = 20
    reduction_method: str = "pacmap"
    params_5d: dict[str, Any] = field(default_factory=dict)
    params_2d: dict[str, Any] = field(default_factory=dict)
    backend: str = "bertopic"
    clustering_method: str = "fast_hdbscan"
    cluster_params: dict[str, Any] = field(default_factory=dict)
    toponymy_cluster_params: dict[str, Any] = field(default_factory=dict)
    toponymy_evoc_cluster_params: dict[str, Any] = field(default_factory=dict)
    toponymy_layer_index: int | Literal["auto"] | None = "auto"
    llm_prompt_name: str = "physics"
    llm_prompt: str | None = None
    llm_provider: str = "openrouter"
    llm_model: str | None = "google/gemini-3-flash-preview"
    llm_model_repo: str | None = None
    llm_model_file: str | None = None
    llm_model_path: str | None = None
    llm_api_key: str | None = None
    bertopic_label_max_tokens: int = 128
    toponymy_local_label_max_tokens: int = 256
    pipeline_models: list[str] = field(default_factory=lambda: ["POS", "KeyBERT", "MMR"])
    parallel_models: list[str] = field(default_factory=lambda: ["MMR", "POS", "KeyBERT"])
    toponymy_embedding_model: str | None = None
    toponymy_max_workers: int = 10
    min_df: int | None = None
    outlier_threshold: float = 0.5


def _normalize_topic_tree_setting(value: bool | str | None) -> bool | Literal["auto"]:
    """Normalize visualization.topic_tree to ``True``, ``False``, or ``"auto"``."""
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto", "none", "null"}:
            return "auto"
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(
            f"Invalid visualization.topic_tree value '{value}'. Expected true, false, or 'auto'."
        )
    raise TypeError(
        f"Invalid visualization.topic_tree type {type(value).__name__}. "
        "Expected bool, 'auto', or null."
    )


@dataclass
class VisualizationConfig:
    enabled: bool = True
    title: str = "ADS Bibliometric Map"
    subtitle_template: str = "Topics labeled with {provider}/{model}"
    dark_mode: bool = True
    topic_tree: bool | Literal["auto"] = "auto"


@dataclass
class CurationConfig:
    cluster_targets: list[dict[str, int]] = field(default_factory=list)
    clusters_to_remove: list[int] = field(default_factory=list)


@dataclass
class CitationsConfig:
    metrics: list[str] = field(
        default_factory=lambda: [
            "direct",
            "co_citation",
            "bibliographic_coupling",
            "author_co_citation",
        ]
    )
    min_counts: dict[str, int] = field(
        default_factory=lambda: {
            "direct": 1,
            "co_citation": 1,
            "bibliographic_coupling": 1,
            "author_co_citation": 1,
        }
    )
    authors_filter: list[str] | None = None
    output_format: str = "gexf"


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    llama_server: LlamaServerConfig = field(default_factory=LlamaServerConfig)
    tokenize: TokenizeConfig = field(default_factory=TokenizeConfig)
    author_disambiguation: AuthorDisambiguationConfig = field(default_factory=AuthorDisambiguationConfig)
    topic_model: TopicModelConfig = field(default_factory=TopicModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    citations: CitationsConfig = field(default_factory=CitationsConfig)

    def __post_init__(self) -> None:
        self.run.start_stage = validate_stage_name(self.run.start_stage)
        if self.run.stop_stage is not None:
            self.run.stop_stage = validate_stage_name(self.run.stop_stage)
        self.llama_server = self.llama_server.normalized()
        self.topic_model.toponymy_layer_index = topic_model_backends.normalize_toponymy_layer_index(
            self.topic_model.toponymy_layer_index
        )
        self.visualization.topic_tree = _normalize_topic_tree_setting(
            self.visualization.topic_tree
        )
        self.curation.cluster_targets = normalize_cluster_targets(
            self.curation.cluster_targets
        )
        validate_provider(
            self.translate.provider,
            valid={"openrouter", "huggingface_api", "llama_server", "nllb"},
        )

        backend = self.topic_model.backend.strip().lower()
        if backend not in {"bertopic", "toponymy", "toponymy_evoc"}:
            raise ValueError(
                f"Invalid topic_model.backend '{self.topic_model.backend}'. "
                "Expected one of: bertopic, toponymy, toponymy_evoc."
            )

        if backend == "bertopic":
            validate_provider(
                self.topic_model.embedding_provider,
                valid=set(EMBEDDING_PROVIDERS),
            )
            validate_provider(
                self.topic_model.llm_provider,
                valid=set(BERTOPIC_LLM_PROVIDERS),
            )
        else:
            validate_provider(
                self.topic_model.embedding_provider,
                valid=set(TOPONYMY_EMBEDDING_PROVIDERS),
            )
            validate_provider(
                self.topic_model.llm_provider,
                valid=set(TOPONYMY_LLM_PROVIDERS),
            )

        if self.author_disambiguation.enabled and not self.author_disambiguation.model_bundle:
            raise ValueError(
                "author_disambiguation.model_bundle is required when author_disambiguation.enabled=true."
            )
        if self.topic_model.llm_prompt is None and self.topic_model.llm_prompt_name not in _PROMPT_MAP:
            allowed = ", ".join(sorted(_PROMPT_MAP))
            raise ValueError(
                f"Invalid topic_model.llm_prompt_name '{self.topic_model.llm_prompt_name}'. "
                f"Expected one of: {allowed}."
            )
        if self.translate.provider == "llama_server":
            ModelSpec.from_fields(
                model_repo=self.translate.model_repo,
                model_file=self.translate.model_file,
                model_path=self.translate.model_path,
                legacy_value=self.translate.model,
                field_label="translate.model",
            )
        if self.topic_model.llm_provider == "llama_server":
            ModelSpec.from_fields(
                model_repo=self.topic_model.llm_model_repo,
                model_file=self.topic_model.llm_model_file,
                model_path=self.topic_model.llm_model_path,
                legacy_value=self.topic_model.llm_model,
                field_label="topic_model.llm_model",
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return cls(
            run=RunConfig(**data.get("run", {})),
            search=SearchConfig(**data.get("search", {})),
            translate=TranslateConfig(**data.get("translate", {})),
            llama_server=LlamaServerConfig(**data.get("llama_server", {})),
            tokenize=TokenizeConfig(
                **{
                    **data.get("tokenize", {}),
                    "disable": tuple(data.get("tokenize", {}).get("disable", ("ner", "parser", "textcat"))),
                }
            ),
            author_disambiguation=AuthorDisambiguationConfig(
                **data.get("author_disambiguation", {})
            ),
            topic_model=TopicModelConfig(**data.get("topic_model", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
            curation=CurationConfig(**data.get("curation", {})),
            citations=CitationsConfig(**data.get("citations", {})),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        return cls.from_dict(payload)


@dataclass
class PipelineContext:
    config: PipelineConfig
    paths: dict[str, Path]
    run: RunManager
    tracker: CostTracker
    project_root: Path
    output_mode: OutputMode = "cli"
    reporter: StageReporter | None = None
    runtime_log_path: Path | None = None
    start_time: float | None = None
    bibcodes: list[str] | None = None
    references: list[list[str]] | None = None
    esources: list[list[str]] | None = None
    fulltext_urls: list[str | None] | None = None
    publications: pd.DataFrame | None = None
    refs: pd.DataFrame | None = None
    topic_input_df: pd.DataFrame | None = None
    documents: list[str] | None = None
    embeddings: np.ndarray | None = None
    reduced_5d: np.ndarray | None = None
    reduced_2d: np.ndarray | None = None
    topic_model: Any | None = None
    topics: np.ndarray | None = None
    topic_info: pd.DataFrame | None = None
    topic_hierarchy: dict[str, Any] | None = None
    topic_df: pd.DataFrame | None = None
    curated_df: pd.DataFrame | None = None
    citation_results: dict[str, pd.DataFrame] | None = None
    resume_blocked_from: StageName | None = None

    @classmethod
    def create(
        cls,
        config: PipelineConfig,
        *,
        project_root: Path | str | None = None,
        run_name: str | None = None,
        paths: dict[str, Path] | None = None,
        run: RunManager | None = None,
        tracker: CostTracker | None = None,
        start_time: float | None = None,
        load_environment: bool = True,
        output_mode: OutputMode = "cli",
    ) -> PipelineContext:
        root = Path(project_root or config.run.project_root or Path.cwd())
        if load_environment:
            load_env(project_root=root)
        resolved_paths = paths or init_paths(project_root=root)
        resolved_run = run or RunManager(run_name=run_name or config.run.run_name, project_root=root)
        resolved_tracker = tracker or CostTracker()
        runtime_log_path = configure_runtime_logging(
            output_mode=output_mode,
            log_file=resolved_run.paths["logs"] / "runtime.log",
        )
        return cls(
            config=config,
            paths=resolved_paths,
            run=resolved_run,
            tracker=resolved_tracker,
            project_root=root,
            output_mode=output_mode,
            reporter=StageReporter(output_mode=output_mode),
            runtime_log_path=runtime_log_path,
            start_time=start_time,
        )


def validate_stage_name(stage: StageName | str) -> StageName:
    value = str(stage)
    if value not in STAGE_ORDER:
        allowed = ", ".join(STAGE_ORDER)
        raise ValueError(f"Invalid stage '{stage}'. Expected one of: {allowed}.")
    return value  # type: ignore[return-value]


def snapshot_block_from_invalidation(stage: StageName | str) -> StageName | None:
    stage_name = validate_stage_name(stage)
    if stage_name in {"search", "translate"}:
        return "translate"
    if stage_name == "tokenize":
        return "tokenize"
    if stage_name == "author_disambiguation":
        return "author_disambiguation"
    return None


def _set_resume_block(ctx: PipelineContext, candidate: StageName | None) -> None:
    if candidate is None:
        return
    if (
        ctx.resume_blocked_from is None
        or STAGE_ORDER.index(candidate) < STAGE_ORDER.index(ctx.resume_blocked_from)
    ):
        ctx.resume_blocked_from = candidate


def _snapshot_allowed(ctx: PipelineContext, snapshot_stage: StageName) -> bool:
    blocked_from = ctx.resume_blocked_from
    if blocked_from is None:
        return True
    return STAGE_ORDER.index(snapshot_stage) < STAGE_ORDER.index(blocked_from)


def _advance_resume_block(ctx: PipelineContext, completed_stage: StageName) -> None:
    blocked_from = ctx.resume_blocked_from
    if blocked_from is None:
        return
    if completed_stage == "translate" and blocked_from == "translate":
        ctx.resume_blocked_from = "tokenize"
        return
    if completed_stage == "tokenize" and blocked_from in {"translate", "tokenize"}:
        ctx.resume_blocked_from = "author_disambiguation"
        return
    if completed_stage == "author_disambiguation" and blocked_from in {
        "translate",
        "tokenize",
        "author_disambiguation",
    }:
        ctx.resume_blocked_from = None


def _has_source_frames(ctx: PipelineContext) -> bool:
    return ctx.publications is not None and ctx.refs is not None


def _has_translated_frames(ctx: PipelineContext) -> bool:
    if not _has_source_frames(ctx):
        return False
    assert ctx.publications is not None
    assert ctx.refs is not None
    required = {"Title_en", "Abstract_en"}
    return required.issubset(ctx.publications.columns) and required.issubset(ctx.refs.columns)


def _try_load_snapshot(
    ctx: PipelineContext,
    stage: StageName,
    load_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
) -> bool:
    """Try loading a snapshot for *stage*; return True on success."""
    if not _has_source_frames(ctx) and _snapshot_allowed(ctx, stage):
        try:
            ctx.publications, ctx.refs = load_fn(
                cache_dir=ctx.paths["cache"],
                run_data_dir=ctx.run.paths["data"],
            )
            return True
        except FileNotFoundError:
            pass
    return False


def _stage_slice(start_stage: StageName, stop_stage: StageName | None) -> tuple[StageName, ...]:
    start_idx = STAGE_ORDER.index(validate_stage_name(start_stage))
    if stop_stage is None:
        return STAGE_ORDER[start_idx:]
    stop_idx = STAGE_ORDER.index(validate_stage_name(stop_stage))
    if stop_idx < start_idx:
        raise ValueError("stop_stage must be after or equal to start_stage.")
    return STAGE_ORDER[start_idx : stop_idx + 1]


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_topic_prompt(cfg: TopicModelConfig) -> str:
    if cfg.llm_prompt:
        return cfg.llm_prompt
    return _PROMPT_MAP[cfg.llm_prompt_name]


def prepare_pipeline_config(config: PipelineConfig) -> PipelineConfig:
    prepared = PipelineConfig.from_dict(config.to_dict())
    load_env(project_root=prepared.run.project_root)

    if not prepared.search.ads_token:
        prepared.search.ads_token = os.getenv("ADS_TOKEN") or os.getenv("ADS_API_KEY")

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    huggingface_api_key = resolve_huggingface_api_key()
    if prepared.translate.provider == "openrouter" and not prepared.translate.api_key:
        prepared.translate.api_key = openrouter_api_key
    if prepared.translate.provider == "huggingface_api" and not prepared.translate.api_key:
        prepared.translate.api_key = huggingface_api_key
    if (
        prepared.topic_model.embedding_provider == "openrouter"
        and not prepared.topic_model.embedding_api_key
    ):
        prepared.topic_model.embedding_api_key = openrouter_api_key
    if (
        prepared.topic_model.embedding_provider == "huggingface_api"
        and not prepared.topic_model.embedding_api_key
    ):
        prepared.topic_model.embedding_api_key = huggingface_api_key
    if prepared.topic_model.llm_provider == "openrouter" and not prepared.topic_model.llm_api_key:
        prepared.topic_model.llm_api_key = openrouter_api_key
    if prepared.topic_model.llm_provider == "huggingface_api" and not prepared.topic_model.llm_api_key:
        prepared.topic_model.llm_api_key = huggingface_api_key

    return prepared


def _topic_subtitle(config: PipelineConfig) -> str:
    model_label = config.topic_model.llm_model
    if config.topic_model.llm_provider == "llama_server":
        model_label = ModelSpec.from_fields(
            model_repo=config.topic_model.llm_model_repo,
            model_file=config.topic_model.llm_model_file,
            model_path=config.topic_model.llm_model_path,
            legacy_value=config.topic_model.llm_model,
            field_label="topic_model.llm_model",
        ).display_name()
    return config.visualization.subtitle_template.format(
        provider=config.topic_model.llm_provider,
        model=model_label,
    )


def _summary_lines_for_stage(ctx: PipelineContext, stage: StageName) -> list[str]:
    if stage == "search" and ctx.bibcodes is not None and ctx.references is not None:
        total_refs = sum(len(refs) for refs in ctx.references)
        unique_refs = len({ref for refs in ctx.references for ref in refs if ref})
        return [
            "result: "
            f"{len(ctx.bibcodes):,} publications | "
            f"{unique_refs:,} unique refs | "
            f"{total_refs:,} total refs"
        ]

    if stage == "export" and ctx.publications is not None and ctx.refs is not None:
        return [f"publications: {len(ctx.publications):,} | references: {len(ctx.refs):,}"]

    if stage == "translate" and ctx.publications is not None and ctx.refs is not None:
        if "Title_lang" in ctx.publications.columns:
            pub_title = int((ctx.publications["Title_lang"] != "en").sum())
            pub_abs = int((ctx.publications["Abstract_lang"] != "en").sum())
            ref_title = int((ctx.refs["Title_lang"] != "en").sum())
            ref_abs = int((ctx.refs["Abstract_lang"] != "en").sum())
            return [
                f"publications: {len(ctx.publications):,} (Title={pub_title:,} Abstract={pub_abs:,} translated) | "
                f"references: {len(ctx.refs):,} (Title={ref_title:,} Abstract={ref_abs:,} translated)"
            ]
        return [f"publications: {len(ctx.publications):,} | references: {len(ctx.refs):,}"]

    if stage == "tokenize" and ctx.publications is not None:
        return [f"documents: {len(ctx.publications):,} | model: {ctx.config.tokenize.spacy_model}"]

    if stage == "author_disambiguation" and ctx.publications is not None and ctx.refs is not None:
        if ctx.config.author_disambiguation.enabled:
            return ["author_uids attached to publications and references"]
        return ["disabled — skipped"]

    if stage == "embeddings" and ctx.embeddings is not None:
        return [f"embeddings: {ctx.embeddings.shape[0]:,} x {ctx.embeddings.shape[1]:,}"]

    if stage == "reduction" and ctx.reduced_5d is not None and ctx.reduced_2d is not None:
        return [
            "reduced: "
            f"5D {ctx.reduced_5d.shape[0]:,}x{ctx.reduced_5d.shape[1]} | "
            f"2D {ctx.reduced_2d.shape[0]:,}x{ctx.reduced_2d.shape[1]}"
        ]

    if stage == "topic_fit" and ctx.topics is not None:
        outliers = int((ctx.topics == -1).sum())
        topic_count = 0
        if ctx.topic_info is not None:
            topic_count = int((ctx.topic_info["Topic"] != -1).sum()) if "Topic" in ctx.topic_info.columns else len(ctx.topic_info)
        if ctx.topic_hierarchy is not None:
            layer_count = int(ctx.topic_hierarchy.get("topic_layer_count", 0))
            primary_layer = int(ctx.topic_hierarchy.get("topic_primary_layer_index", 0))
            selection = str(ctx.topic_hierarchy.get("topic_primary_layer_selection", "manual"))
            clusters_per_layer = ", ".join(
                str(int(value))
                for value in ctx.topic_hierarchy.get("topic_clusters_per_layer", [])
            )
            return [
                f"backend: {ctx.config.topic_model.backend} | "
                f"layers: {layer_count:,} | "
                f"primary_layer: {primary_layer} ({selection}) | "
                f"clusters/layer: [{clusters_per_layer}] | "
                f"topics: {topic_count:,} | outliers: {outliers:,}"
            ]
        return [
            f"backend: {ctx.config.topic_model.backend} | topics: {topic_count:,} | outliers: {outliers:,}"
        ]

    if stage == "topic_dataframe" and ctx.topic_df is not None:
        return [f"topic dataframe: {len(ctx.topic_df):,} rows"]

    if stage == "visualize":
        return [f"saved: {ctx.run.paths['plots'] / 'topic_map.html'}"]

    if stage == "curate" and ctx.curated_df is not None:
        topic_count = ctx.curated_df["topic_id"].nunique() if "topic_id" in ctx.curated_df.columns else 0
        base = f"curated dataset: {len(ctx.curated_df):,} | topics: {topic_count:,}"
        if ctx.topic_df is not None and len(ctx.topic_df) > len(ctx.curated_df):
            removed = len(ctx.topic_df) - len(ctx.curated_df)
            n_targets = len(ctx.config.curation.cluster_targets)
            n_legacy = len(ctx.config.curation.clusters_to_remove)
            total_targets = n_targets + n_legacy
            if total_targets > 0:
                target_label = "targets" if n_targets > 0 else "clusters"
                base += f" ({removed:,} rows removed from {total_targets} {target_label})"
        return [base]

    if stage == "citations" and ctx.citation_results is not None:
        parts = []
        for name in sorted(ctx.citation_results):
            edges_df = ctx.citation_results[name]
            parts.append(f"{name}: {len(edges_df):,} edges")
        return [" | ".join(parts)]

    return []


def _report_stage_end(ctx: PipelineContext, stage: StageName) -> None:
    reporter = getattr(ctx, "reporter", None)
    if reporter is None:
        return
    for line in _summary_lines_for_stage(ctx, stage):
        reporter.detail(line)


def _infer_completed_stages(ctx: PipelineContext) -> list[StageName]:
    """Infer completed stages from the current in-memory/run artifact state."""
    completed: list[StageName] = []

    if (
        ctx.bibcodes is not None
        and ctx.references is not None
        and ctx.esources is not None
        and ctx.fulltext_urls is not None
    ):
        completed.append("search")
    if ctx.publications is not None and ctx.refs is not None:
        completed.append("export")
    if "export" in completed and _has_translated_frames(ctx):
        completed.append("translate")
    if (
        "translate" in completed
        and ctx.publications is not None
        and ctx.refs is not None
        and "tokens" in ctx.publications.columns
    ):
        completed.append("tokenize")
    if "tokenize" in completed and ctx.publications is not None and ctx.refs is not None and (
        not ctx.config.author_disambiguation.enabled or "author_uids" in ctx.publications.columns
    ):
        completed.append("author_disambiguation")
    if (
        "author_disambiguation" in completed
        and ctx.embeddings is not None
        and ctx.documents is not None
        and ctx.topic_input_df is not None
    ):
        completed.append("embeddings")
    if "embeddings" in completed and ctx.reduced_5d is not None and ctx.reduced_2d is not None:
        completed.append("reduction")
    if (
        "reduction" in completed
        and ctx.topic_model is not None
        and ctx.topics is not None
        and ctx.topic_info is not None
    ):
        completed.append("topic_fit")
    if "topic_fit" in completed and ctx.topic_df is not None:
        completed.append("topic_dataframe")
    if "topic_dataframe" in completed and (ctx.run.paths["plots"] / "topic_map.html").exists():
        completed.append("visualize")
    if "topic_dataframe" in completed and ctx.curated_df is not None:
        completed.append("curate")
    if "curate" in completed and ctx.citation_results is not None:
        completed.append("citations")
    return completed


def _finalize_run_summary(
    ctx: PipelineContext,
    *,
    status: str,
    requested_start_stage: StageName | None,
    requested_stop_stage: StageName | None,
    completed_stages: list[StageName] | None = None,
    failed_stage: StageName | None = None,
    error: str | None = None,
) -> Path:
    """Write the run summary and emit one compact frontend-visible status line."""
    summary_path = ctx.run.save_summary(
        cost_tracker=ctx.tracker,
        publications=ctx.publications,
        refs=ctx.refs,
        curated=ctx.curated_df,
        topic_hierarchy=ctx.topic_hierarchy,
        start_time=ctx.start_time,
        status=status,
        requested_start_stage=requested_start_stage,
        requested_stop_stage=requested_stop_stage,
        completed_stages=completed_stages or _infer_completed_stages(ctx),
        failed_stage=failed_stage,
        error=error,
    )
    try:
        display_path = summary_path.relative_to(ctx.project_root)
    except ValueError:
        display_path = summary_path

    reporter = getattr(ctx, "reporter", None)
    message = f"run {status} | summary: {display_path}"
    if failed_stage is not None:
        message = f"run {status} at {failed_stage} | summary: {display_path}"
    if reporter is not None:
        reporter.detail(message)
    else:
        logger.info(message)
    return summary_path


def _default_toponymy_min_clusters(n_docs: int) -> int:
    """Scale Toponymy ``min_clusters`` with corpus size for small datasets."""
    return max(3, min(10, int(n_docs * 0.006)))


def _warn_if_aggressive_toponymy_config(
    *,
    backend: str,
    n_docs: int,
    clusterer_params: dict[str, Any],
) -> None:
    """Warn when Toponymy cluster defaults are likely too strict for corpus size."""
    try:
        min_clusters = int(clusterer_params.get("min_clusters"))
        base_min_cluster_size = int(clusterer_params.get("base_min_cluster_size"))
    except (TypeError, ValueError):
        return

    if min_clusters <= 0 or base_min_cluster_size <= 0:
        return

    estimated_max_clusters = n_docs // base_min_cluster_size
    if estimated_max_clusters >= min_clusters:
        return

    logger.warning(
        "Toponymy config may be too aggressive | backend=%s | docs=%s | "
        "min_clusters=%s | base_min_cluster_size=%s | estimated_max_clusters~%s. "
        "Consider lowering min_clusters and/or base_min_cluster_size.",
        backend,
        f"{n_docs:,}",
        min_clusters,
        base_min_cluster_size,
        estimated_max_clusters,
    )


def _resolve_topic_defaults(ctx: PipelineContext) -> dict[str, Any]:
    if ctx.documents is None:
        raise ValueError("documents are not available.")
    cfg = ctx.config.topic_model
    n_docs = len(ctx.documents)
    min_cluster_size = max(15, int(n_docs * 0.001))
    base_min_cluster_size = max(10, min(55, int(n_docs * 0.001)))
    toponymy_min_clusters = _default_toponymy_min_clusters(n_docs)
    min_df = cfg.min_df if cfg.min_df is not None else max(1, min(5, n_docs // 100))

    cluster_params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": 3,
        "cluster_selection_method": "eom",
        "cluster_selection_epsilon": 0.05,
        **cfg.cluster_params,
    }
    toponymy_cluster_params = {
        "min_clusters": toponymy_min_clusters,
        "min_samples": 3,
        "base_min_cluster_size": base_min_cluster_size,
        **cfg.toponymy_cluster_params,
    }
    toponymy_evoc_cluster_params = {
        "min_clusters": toponymy_min_clusters,
        "min_samples": 3,
        "base_min_cluster_size": base_min_cluster_size,
        "noise_level": 0.35,
        "n_neighbors": 15,
        "n_epochs": 35,
        **cfg.toponymy_evoc_cluster_params,
    }

    logger.info("Topic defaults | docs=%s | min_df=%s | min_cluster_size=%s | base_min_cluster_size=%s",
                f"{n_docs:,}", min_df, cluster_params["min_cluster_size"], base_min_cluster_size)
    return {
        "min_df": min_df,
        "cluster_params": cluster_params,
        "toponymy_cluster_params": toponymy_cluster_params,
        "toponymy_evoc_cluster_params": toponymy_evoc_cluster_params,
        "toponymy_embedding_model": cfg.toponymy_embedding_model or cfg.embedding_model,
    }


def _save_curated_dataset(ctx: PipelineContext) -> Path:
    if ctx.curated_df is None:
        raise ValueError("curated_df is not available.")
    curated_path = ctx.run.paths["data"] / "publications.parquet"
    save_parquet(ctx.curated_df, curated_path)
    logger.info("Curated dataset saved: %s records", f"{len(ctx.curated_df):,}")
    return curated_path


def _load_curated_dataset(ctx: PipelineContext) -> pd.DataFrame:
    curated_path = ctx.run.paths["data"] / "publications.parquet"
    if curated_path.exists():
        return load_parquet(curated_path)
    raise FileNotFoundError(f"Curated dataset not found at {curated_path}")


def _require_stage(stage: StageName, required_stage: StageName, message: str) -> StagePrerequisiteError:
    return StagePrerequisiteError(stage, required_stage, message)


def run_search_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.bibcodes is not None:
        return ctx

    cfg = ctx.config.search
    if not cfg.ads_token:
        raise ValueError("search.ads_token is required.")
    reporter = ctx.reporter
    if reporter is None or not cfg.refresh_search:
        ctx.bibcodes, ctx.references, ctx.esources, ctx.fulltext_urls = search_ads(
            cfg.query,
            cfg.ads_token,
            raw_dir=ctx.paths["raw"],
            force_refresh=cfg.refresh_search,
        )
        return ctx

    with reporter.progress(total=None, desc="fetch") as pbar:
        progress_callback = None if pbar is None else pbar.update
        ctx.bibcodes, ctx.references, ctx.esources, ctx.fulltext_urls = search_ads(
            cfg.query,
            cfg.ads_token,
            raw_dir=ctx.paths["raw"],
            force_refresh=cfg.refresh_search,
            progress_callback=progress_callback,
        )
    return ctx


def run_export_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.publications is not None and ctx.refs is not None:
        return ctx

    if (
        ctx.bibcodes is None
        or ctx.references is None
        or ctx.esources is None
        or ctx.fulltext_urls is None
    ):
        raise _require_stage(
            "export",
            "search",
            "Export stage requires search results in memory. Run the search stage first.",
        )
    cfg = ctx.config.search
    flat_refs = sorted({ref for refs in ctx.references for ref in refs if ref})
    reporter = ctx.reporter
    if reporter is None:
        ctx.publications, ctx.refs = resolve_dataset(
            ctx.bibcodes,
            ctx.references,
            ctx.esources,
            ctx.fulltext_urls,
            cfg.ads_token or "",
            cache_dir=ctx.paths["raw"],
            force_refresh=cfg.refresh_export,
        )
        return ctx

    with reporter.progress(total=len(ctx.bibcodes) + len(flat_refs), desc="export") as pbar:
        progress_callback = None if pbar is None else pbar.update
        ctx.publications, ctx.refs = resolve_dataset(
            ctx.bibcodes,
            ctx.references,
            ctx.esources,
            ctx.fulltext_urls,
            cfg.ads_token or "",
            cache_dir=ctx.paths["raw"],
            force_refresh=cfg.refresh_export,
            show_progress=False,
            progress_callback=progress_callback,
        )
    return ctx


def run_translate_stage(ctx: PipelineContext) -> PipelineContext:
    if _has_translated_frames(ctx):
        return ctx

    if _try_load_snapshot(ctx, "translate", load_translated_snapshot):
        return ctx

    if not _has_source_frames(ctx):
        raise _require_stage(
            "translate",
            "export",
            "Translate stage requires exported publications and references in memory, "
            "or a valid translated snapshot for the same stage.",
        )

    assert ctx.publications is not None
    assert ctx.refs is not None
    cfg = ctx.config.translate
    if not cfg.enabled:
        save_translated_snapshot(
            ctx.publications,
            ctx.refs,
            cache_dir=ctx.paths["cache"],
            run_data_dir=ctx.run.paths["data"],
        )
        _advance_resume_block(ctx, "translate")
        return ctx

    if not cfg.fasttext_model:
        raise ValueError("translate.fasttext_model is required.")

    ctx.publications = detect_languages(
        ctx.publications,
        ["Title", "Abstract"],
        model_path=cfg.fasttext_model,
    )
    ctx.refs = detect_languages(
        ctx.refs,
        ["Title", "Abstract"],
        model_path=cfg.fasttext_model,
    )

    reporter = ctx.reporter
    if reporter is None:
        ctx.publications, _ = translate_dataframe(
            ctx.publications,
            ["Title", "Abstract"],
            provider=cfg.provider,
            model=cfg.model,
            model_repo=cfg.model_repo,
            model_file=cfg.model_file,
            model_path=cfg.model_path,
            api_key=cfg.api_key,
            max_workers=cfg.max_workers,
            max_translation_tokens=cfg.max_tokens,
            llama_server_config=ctx.config.llama_server,
            runtime_log_path=ctx.runtime_log_path,
            openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
            cost_tracker=ctx.tracker,
        )
        ctx.refs, _ = translate_dataframe(
            ctx.refs,
            ["Title", "Abstract"],
            provider=cfg.provider,
            model=cfg.model,
            model_repo=cfg.model_repo,
            model_file=cfg.model_file,
            model_path=cfg.model_path,
            api_key=cfg.api_key,
            max_workers=cfg.max_workers,
            max_translation_tokens=cfg.max_tokens,
            llama_server_config=ctx.config.llama_server,
            runtime_log_path=ctx.runtime_log_path,
            openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
            cost_tracker=ctx.tracker,
        )
    else:
        pubs_title = int((ctx.publications["Title_lang"] != "en").sum())
        pubs_abs = int((ctx.publications["Abstract_lang"] != "en").sum())
        refs_title = int((ctx.refs["Title_lang"] != "en").sum())
        refs_abs = int((ctx.refs["Abstract_lang"] != "en").sum())
        reporter.detail(
            "publications non-English: Title=%s | Abstract=%s",
            f"{pubs_title:,}",
            f"{pubs_abs:,}",
        )
        reporter.detail(
            "references non-English: Title=%s | Abstract=%s",
            f"{refs_title:,}",
            f"{refs_abs:,}",
        )

        translate_total = pubs_title + pubs_abs + refs_title + refs_abs
        progress_callback = None
        progress_cm = reporter.progress(total=translate_total, desc="translate")
        with progress_cm as pbar:
            progress_callback = None if pbar is None else pbar.update
            ctx.publications, pub_cost_info = translate_dataframe(
                ctx.publications,
                ["Title", "Abstract"],
                provider=cfg.provider,
                model=cfg.model,
                model_repo=cfg.model_repo,
                model_file=cfg.model_file,
                model_path=cfg.model_path,
                api_key=cfg.api_key,
                max_workers=cfg.max_workers,
                max_translation_tokens=cfg.max_tokens,
                llama_server_config=ctx.config.llama_server,
                runtime_log_path=ctx.runtime_log_path,
                openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
                cost_tracker=ctx.tracker,
                show_progress=False,
                progress_callback=progress_callback,
            )

            ctx.refs, ref_cost_info = translate_dataframe(
                ctx.refs,
                ["Title", "Abstract"],
                provider=cfg.provider,
                model=cfg.model,
                model_repo=cfg.model_repo,
                model_file=cfg.model_file,
                model_path=cfg.model_path,
                api_key=cfg.api_key,
                max_workers=cfg.max_workers,
                max_translation_tokens=cfg.max_tokens,
                llama_server_config=ctx.config.llama_server,
                runtime_log_path=ctx.runtime_log_path,
                openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
                cost_tracker=ctx.tracker,
                show_progress=False,
                progress_callback=progress_callback,
            )
        total_cost = sum(
            cost or 0.0
            for cost in (pub_cost_info.get("cost_usd"), ref_cost_info.get("cost_usd"))
            if cost is not None
        )
        if total_cost > 0:
            reporter.detail("cost: $%.4f", total_cost)
    save_translated_snapshot(
        ctx.publications,
        ctx.refs,
        cache_dir=ctx.paths["cache"],
        run_data_dir=ctx.run.paths["data"],
    )
    _advance_resume_block(ctx, "translate")
    return ctx


def run_tokenize_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.publications is not None and ctx.refs is not None and "tokens" in ctx.publications.columns:
        return ctx

    if _try_load_snapshot(ctx, "tokenize", load_tokenized_snapshot):
        return ctx

    if not _has_translated_frames(ctx):
        raise _require_stage(
            "tokenize",
            "translate",
            "Tokenize stage requires translated publications and references in memory, "
            "or a valid tokenized snapshot for the same stage.",
        )

    assert ctx.publications is not None
    assert ctx.refs is not None
    cfg = ctx.config.tokenize
    if not cfg.enabled:
        save_tokenized_snapshot(
            ctx.publications,
            ctx.refs,
            cache_dir=ctx.paths["cache"],
            run_data_dir=ctx.run.paths["data"],
        )
        _advance_resume_block(ctx, "tokenize")
        return ctx

    reporter = ctx.reporter
    if reporter is None:
        model_to_use, preloaded_nlp = ensure_spacy_model(
            spacy_model=cfg.spacy_model,
            fallback_model=cfg.fallback_model,
            spacy_disable=cfg.disable,
            auto_download=cfg.auto_download,
        )
        ctx.publications = tokenize_texts(
            ctx.publications,
            spacy_model=model_to_use,
            nlp=preloaded_nlp,
            batch_size=cfg.batch_size,
            n_process=cfg.n_process,
            spacy_disable=cfg.disable,
        )
    else:
        model_to_use, preloaded_nlp = ensure_spacy_model(
            spacy_model=cfg.spacy_model,
            fallback_model=cfg.fallback_model,
            spacy_disable=cfg.disable,
            auto_download=cfg.auto_download,
        )

        with reporter.progress(total=len(ctx.publications), desc="tokenize docs") as tokenize_pbar:
            progress_callback = None if tokenize_pbar is None else tokenize_pbar.update
            ctx.publications = tokenize_texts(
                ctx.publications,
                spacy_model=model_to_use,
                nlp=preloaded_nlp,
                batch_size=cfg.batch_size,
                n_process=cfg.n_process,
                spacy_disable=cfg.disable,
                show_progress=False,
                progress_callback=progress_callback,
            )
    save_parquet(ctx.refs, ctx.run.paths["data"] / "references.parquet")
    save_tokenized_snapshot(
        ctx.publications,
        ctx.refs,
        cache_dir=ctx.paths["cache"],
        run_data_dir=ctx.run.paths["data"],
    )
    _advance_resume_block(ctx, "tokenize")
    return ctx


def run_author_disambiguation_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.publications is not None and ctx.refs is not None and "author_uids" in ctx.publications.columns:
        return ctx

    if _try_load_snapshot(ctx, "author_disambiguation", load_disambiguated_snapshot):
        return ctx

    if ctx.publications is None or "tokens" not in ctx.publications.columns or ctx.refs is None:
        raise _require_stage(
            "author_disambiguation",
            "tokenize",
            "Author disambiguation stage requires tokenized publications and references in memory, "
            "or a valid disambiguated snapshot for the same stage.",
        )

    assert ctx.publications is not None
    assert ctx.refs is not None
    cfg = ctx.config.author_disambiguation
    if cfg.enabled:
        reporter = ctx.reporter
        if reporter is None:
            ctx.publications, ctx.refs = apply_author_disambiguation(
                ctx.publications,
                ctx.refs,
                model_bundle=cfg.model_bundle or "",
                dataset_id=cfg.dataset_id or ctx.run.run_id,
                cache_dir=ctx.paths["cache"],
                run_data_dir=ctx.run.paths["data"],
                force_refresh=cfg.force_refresh,
                infer_stage=cfg.infer_stage,
            )
        else:
            with reporter.progress(total=1, desc="disambiguate") as pbar:
                ctx.publications, ctx.refs = apply_author_disambiguation(
                    ctx.publications,
                    ctx.refs,
                    model_bundle=cfg.model_bundle or "",
                    dataset_id=cfg.dataset_id or ctx.run.run_id,
                    cache_dir=ctx.paths["cache"],
                    run_data_dir=ctx.run.paths["data"],
                    force_refresh=cfg.force_refresh,
                    infer_stage=cfg.infer_stage,
                )
                if pbar is not None:
                    pbar.update(1)
    else:
        save_disambiguated_snapshot(
            ctx.publications,
            ctx.refs,
            cache_dir=ctx.paths["cache"],
            run_data_dir=ctx.run.paths["data"],
        )
    _advance_resume_block(ctx, "author_disambiguation")
    return ctx


def run_embeddings_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.embeddings is not None and ctx.documents is not None and ctx.topic_input_df is not None:
        return ctx

    if ctx.publications is None or "full_text" not in ctx.publications.columns:
        raise _require_stage(
            "embeddings",
            "author_disambiguation",
            "Embeddings stage requires tokenized publications in memory. "
            "Run the author_disambiguation stage first.",
        )
    cfg = ctx.config.topic_model
    df = ctx.publications.copy()
    if cfg.sample_size is not None:
        df = df.sample(
            n=min(cfg.sample_size, len(df)),
            random_state=ctx.config.run.random_seed,
        ).reset_index(drop=True)
        logger.info("Sampling topic input: %s documents", f"{len(df):,}")
    ctx.topic_input_df = df
    ctx.documents = df["full_text"].tolist()
    reporter = ctx.reporter
    if reporter is None:
        ctx.embeddings = compute_embeddings(
            ctx.documents,
            provider=cfg.embedding_provider,
            model=cfg.embedding_model,
            cache_dir=ctx.paths["embeddings_cache"],
            batch_size=cfg.embedding_batch_size,
            max_workers=cfg.embedding_max_workers,
            api_key=cfg.embedding_api_key,
            openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
            cost_tracker=ctx.tracker,
        )
        return ctx

    with reporter.progress(total=len(ctx.documents), desc="embeddings") as pbar:
        progress_callback = None if pbar is None else pbar.update
        ctx.embeddings = compute_embeddings(
            ctx.documents,
            provider=cfg.embedding_provider,
            model=cfg.embedding_model,
            cache_dir=ctx.paths["embeddings_cache"],
            batch_size=cfg.embedding_batch_size,
            max_workers=cfg.embedding_max_workers,
            api_key=cfg.embedding_api_key,
            openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
            cost_tracker=ctx.tracker,
            show_progress=False,
            progress_callback=progress_callback,
        )
    return ctx


def run_reduction_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.reduced_5d is not None and ctx.reduced_2d is not None:
        return ctx

    if ctx.embeddings is None:
        raise _require_stage(
            "reduction",
            "embeddings",
            "Reduction stage requires embeddings in memory. Run the embeddings stage first.",
        )
    cfg = ctx.config.topic_model
    with capture_external_output(ctx.runtime_log_path):
        ctx.reduced_5d, ctx.reduced_2d = reduce_dimensions(
            ctx.embeddings,
            method=cfg.reduction_method,
            params_5d=cfg.params_5d,
            params_2d=cfg.params_2d,
            random_state=ctx.config.run.random_seed,
            cache_dir=ctx.paths["dim_reduction_cache"],
            embedding_id=f"{cfg.embedding_provider}/{cfg.embedding_model}",
        )
    return ctx


def run_topic_fit_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.topic_model is not None and ctx.topics is not None and ctx.topic_info is not None:
        return ctx

    if ctx.documents is None or ctx.reduced_5d is None or ctx.embeddings is None:
        raise _require_stage(
            "topic_fit",
            "reduction",
            "Topic-fit stage requires reduced embeddings and documents in memory. "
            "Run the reduction stage first.",
        )
    cfg = ctx.config.topic_model
    resolved = _resolve_topic_defaults(ctx)
    reporter = ctx.reporter
    if reporter is not None and reporter.output_mode == "notebook":
        reporter.detail(
            "Topic defaults | docs=%s | min_df=%s | min_cluster_size=%s | base_min_cluster_size=%s",
            f"{len(ctx.documents):,}",
            resolved["min_df"],
            resolved["cluster_params"]["min_cluster_size"],
            resolved["toponymy_cluster_params"]["base_min_cluster_size"],
        )

    if cfg.backend == "bertopic":
        if reporter is None:
            topic_model = fit_bertopic(
                ctx.documents,
                ctx.reduced_5d,
                llm_provider=cfg.llm_provider,
                llm_model=cfg.llm_model,
                llm_model_repo=cfg.llm_model_repo,
                llm_model_file=cfg.llm_model_file,
                llm_model_path=cfg.llm_model_path,
                llm_prompt=_resolve_topic_prompt(cfg),
                llm_max_new_tokens=cfg.bertopic_label_max_tokens,
                pipeline_models=cfg.pipeline_models,
                parallel_models=cfg.parallel_models,
                min_df=resolved["min_df"],
                clustering_method=cfg.clustering_method,
                clustering_params=resolved["cluster_params"],
                api_key=cfg.llm_api_key,
                openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
                cost_tracker=ctx.tracker,
                llama_server_config=ctx.config.llama_server,
                runtime_log_path=ctx.runtime_log_path,
            )
            topics = np.array(topic_model.topics_)
            topics = reduce_outliers(
                topic_model,
                ctx.documents,
                topics,
                ctx.reduced_5d,
                threshold=cfg.outlier_threshold,
                llm_provider=cfg.llm_provider,
                llm_model=cfg.llm_model,
                api_key=cfg.llm_api_key,
                openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
                cost_tracker=ctx.tracker,
            )
        else:
            reporter.detail("preparing BERTopic clustering and label generation")
            with topic_model_backends._bridge_bertopic_label_progress(reporter=reporter, desc="fit"):
                with capture_external_output(ctx.runtime_log_path):
                    topic_model = fit_bertopic(
                        ctx.documents,
                        ctx.reduced_5d,
                        llm_provider=cfg.llm_provider,
                        llm_model=cfg.llm_model,
                        llm_model_repo=cfg.llm_model_repo,
                        llm_model_file=cfg.llm_model_file,
                        llm_model_path=cfg.llm_model_path,
                        llm_prompt=_resolve_topic_prompt(cfg),
                        llm_max_new_tokens=cfg.bertopic_label_max_tokens,
                        pipeline_models=cfg.pipeline_models,
                        parallel_models=cfg.parallel_models,
                        min_df=resolved["min_df"],
                        clustering_method=cfg.clustering_method,
                        clustering_params=resolved["cluster_params"],
                        api_key=cfg.llm_api_key,
                        openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
                        cost_tracker=ctx.tracker,
                        llama_server_config=ctx.config.llama_server,
                        runtime_log_path=ctx.runtime_log_path,
                        show_progress=False,
                    )
            topics = np.array(topic_model.topics_)
            reporter.detail("reassigning outliers before topic-label refresh")
            with topic_model_backends._bridge_bertopic_label_progress(
                reporter=reporter,
                desc="outlier refresh",
            ):
                with capture_external_output(ctx.runtime_log_path):
                    topics = reduce_outliers(
                        topic_model,
                        ctx.documents,
                        topics,
                        ctx.reduced_5d,
                        threshold=cfg.outlier_threshold,
                        llm_provider=cfg.llm_provider,
                        llm_model=cfg.llm_model,
                        api_key=cfg.llm_api_key,
                        openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
                        cost_tracker=ctx.tracker,
                        show_progress=False,
                    )
        topic_info = topic_model.get_topic_info()
    elif cfg.backend in {"toponymy", "toponymy_evoc"}:
        clusterer_params = (
            resolved["toponymy_evoc_cluster_params"]
            if cfg.backend == "toponymy_evoc"
            else resolved["toponymy_cluster_params"]
        )
        _warn_if_aggressive_toponymy_config(
            backend=cfg.backend,
            n_docs=len(ctx.documents),
            clusterer_params=clusterer_params,
        )
        toponymy_api_key = cfg.llm_api_key or cfg.embedding_api_key
        if reporter is None:
            with capture_external_output(ctx.runtime_log_path):
                topic_model, topics, topic_info = fit_toponymy(
                    ctx.documents,
                    ctx.embeddings,
                    ctx.reduced_5d,
                    backend=cfg.backend,
                    layer_index=cfg.toponymy_layer_index,
                    llm_provider=cfg.llm_provider,
                    llm_model=cfg.llm_model,
                    llm_model_repo=cfg.llm_model_repo,
                    llm_model_file=cfg.llm_model_file,
                    llm_model_path=cfg.llm_model_path,
                    embedding_provider=cfg.embedding_provider,
                    embedding_model=resolved["toponymy_embedding_model"],
                    local_llm_max_new_tokens=cfg.toponymy_local_label_max_tokens,
                    api_key=toponymy_api_key,
                    openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
                    max_workers=cfg.toponymy_max_workers,
                    clusterer_params=clusterer_params,
                    cost_tracker=ctx.tracker,
                    llama_server_config=ctx.config.llama_server,
                    runtime_log_path=ctx.runtime_log_path,
                )
        else:
            with reporter.progress(total=1, desc="fit") as fit_pbar:
                with capture_external_output(ctx.runtime_log_path):
                    topic_model, topics, topic_info = fit_toponymy(
                        ctx.documents,
                        ctx.embeddings,
                        ctx.reduced_5d,
                        backend=cfg.backend,
                        layer_index=cfg.toponymy_layer_index,
                        llm_provider=cfg.llm_provider,
                        llm_model=cfg.llm_model,
                        llm_model_repo=cfg.llm_model_repo,
                        llm_model_file=cfg.llm_model_file,
                        llm_model_path=cfg.llm_model_path,
                        embedding_provider=cfg.embedding_provider,
                        embedding_model=resolved["toponymy_embedding_model"],
                        local_llm_max_new_tokens=cfg.toponymy_local_label_max_tokens,
                        api_key=toponymy_api_key,
                        openrouter_cost_mode=ctx.config.run.openrouter_cost_mode,
                        max_workers=cfg.toponymy_max_workers,
                        clusterer_params=clusterer_params,
                        cost_tracker=ctx.tracker,
                        llama_server_config=ctx.config.llama_server,
                        runtime_log_path=ctx.runtime_log_path,
                    )
                if fit_pbar is not None:
                    fit_pbar.update(1)
    else:
        raise ValueError(f"Invalid topic_model.backend '{cfg.backend}'.")

    ctx.topic_model = topic_model
    ctx.topics = np.asarray(topics)
    ctx.topic_info = topic_info
    ctx.topic_hierarchy = (
        topic_model_backends.get_toponymy_hierarchy_metadata(topic_model)
        if cfg.backend in {"toponymy", "toponymy_evoc"}
        else None
    )
    ctx.tracker.log_steps_summary(
        [
            "llm_labeling",
            "llm_labeling_post_outliers",
            "llm_labeling_toponymy",
            "llm_labeling_toponymy_evoc",
            "toponymy_embeddings",
        ]
    )
    return ctx


def run_topic_dataframe_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.topic_df is not None:
        return ctx

    if (
        ctx.topic_input_df is None
        or ctx.topic_model is None
        or ctx.topics is None
        or ctx.reduced_2d is None
    ):
        raise _require_stage(
            "topic_dataframe",
            "topic_fit",
            "Topic-dataframe stage requires a fitted topic model and reduced 2D embeddings in memory. "
            "Run the topic_fit stage first.",
        )
    ctx.topic_df = build_topic_dataframe(
        ctx.topic_input_df,
        ctx.topic_model,
        ctx.topics,
        ctx.reduced_2d,
        embeddings=None,
        topic_info=ctx.topic_info,
    )
    return ctx


def run_visualize_stage(ctx: PipelineContext) -> PipelineContext:
    if not ctx.config.visualization.enabled:
        logger.info("Visualization disabled.")
        return ctx

    from ads_bib.visualize import create_topic_map

    if ctx.topic_df is None:
        raise _require_stage(
            "visualize",
            "topic_dataframe",
            "Visualize stage requires topic_df in memory. Run the topic_dataframe stage first.",
        )
    create_topic_map(
        ctx.topic_df,
        title=ctx.config.visualization.title,
        subtitle=_topic_subtitle(ctx.config),
        dark_mode=ctx.config.visualization.dark_mode,
        topic_tree=ctx.config.visualization.topic_tree,
        output_path=ctx.run.paths["plots"] / "topic_map.html",
    )
    return ctx


def _resolve_working_layer_index(ctx: PipelineContext) -> int | None:
    """Resolve the configured Toponymy working-layer index from context metadata."""
    if ctx.topic_hierarchy is not None:
        value = ctx.topic_hierarchy.get("topic_primary_layer_index")
        if value is not None:
            return int(value)
    if ctx.topic_df is not None and "topic_primary_layer_index" in ctx.topic_df.columns:
        value = ctx.topic_df["topic_primary_layer_index"].iloc[0]
        if pd.notna(value):
            return int(value)
    return None


def run_curate_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.curated_df is not None:
        return ctx

    if ctx.topic_df is None:
        raise _require_stage(
            "curate",
            "topic_dataframe",
            "Curate stage requires topic_df in memory. Run the topic_dataframe stage first.",
        )
    ctx.curated_df = ctx.topic_df.copy()
    hierarchical_backend = (
        ctx.config.topic_model.backend in {"toponymy", "toponymy_evoc"}
        and any(
            column.startswith("topic_layer_") and column.endswith("_id")
            for column in ctx.curated_df.columns
        )
    )
    working_layer_index = _resolve_working_layer_index(ctx)
    display_summary = (
        get_hierarchy_cluster_summary(
            ctx.curated_df,
            working_layer_index=working_layer_index,
        )
        if hierarchical_backend
        else get_cluster_summary(ctx.curated_df)
    )
    logger.info("Cluster summary rows: %s", f"{len(display_summary):,}")
    if hierarchical_backend:
        cluster_targets = list(ctx.config.curation.cluster_targets)
        if ctx.config.curation.clusters_to_remove:
            if working_layer_index is None:
                raise ValueError(
                    "Legacy curation.clusters_to_remove requires a resolved Toponymy working layer."
                )
            logger.info(
                "Applying legacy curation.clusters_to_remove against working layer %s.",
                working_layer_index,
            )
            cluster_targets.extend(
                {
                    "layer": working_layer_index,
                    "cluster_id": int(cluster_id),
                }
                for cluster_id in ctx.config.curation.clusters_to_remove
            )
        if cluster_targets:
            ctx.curated_df = remove_cluster_targets(
                ctx.curated_df,
                cluster_targets,
            )
    elif ctx.config.curation.cluster_targets:
        raise ValueError(
            "curation.cluster_targets is supported only for toponymy and toponymy_evoc backends."
        )
    elif ctx.config.curation.clusters_to_remove:
        ctx.curated_df = remove_clusters(
            ctx.curated_df,
            ctx.config.curation.clusters_to_remove,
        )
    _save_curated_dataset(ctx)
    return ctx


def run_citations_stage(ctx: PipelineContext) -> PipelineContext:
    if ctx.citation_results is not None:
        return ctx

    if ctx.curated_df is None or ctx.refs is None:
        raise _require_stage(
            "citations",
            "curate",
            "Citations stage requires a curated dataset and reference table in memory. "
            "Run the curate stage first.",
        )

    bibcodes, references = build_citation_inputs_from_publications(ctx.curated_df)
    all_nodes = build_all_nodes(ctx.curated_df, ctx.refs)
    cfg = ctx.config.citations
    ctx.citation_results = process_all_citations(
        bibcodes=bibcodes,
        references=references,
        publications=ctx.curated_df,
        ref_df=ctx.refs,
        all_nodes=all_nodes,
        metrics=cfg.metrics,
        min_counts=cfg.min_counts,
        authors_filter=cfg.authors_filter,
        output_format=cfg.output_format,
        output_dir=ctx.run.paths["data"],
        show_progress=False if ctx.reporter is not None else True,
    )
    suffix = "_filtered" if cfg.authors_filter else ""
    export_wos_format(
        ctx.curated_df,
        ctx.refs,
        output_path=ctx.run.paths["data"] / f"download_wos_export{suffix}.txt",
    )
    return ctx


_STAGE_FUNCS: dict[StageName, Any] = {
    "search": run_search_stage,
    "export": run_export_stage,
    "translate": run_translate_stage,
    "tokenize": run_tokenize_stage,
    "author_disambiguation": run_author_disambiguation_stage,
    "embeddings": run_embeddings_stage,
    "reduction": run_reduction_stage,
    "topic_fit": run_topic_fit_stage,
    "topic_dataframe": run_topic_dataframe_stage,
    "visualize": run_visualize_stage,
    "curate": run_curate_stage,
    "citations": run_citations_stage,
}


def _execute_stage(ctx: PipelineContext, stage: StageName) -> PipelineContext:
    reporter = getattr(ctx, "reporter", None)
    if reporter is not None:
        ctx.reporter.stage_start(stage)
    result = _STAGE_FUNCS[stage](ctx)
    _report_stage_end(ctx, stage)
    return result


def _run_stage_for_pipeline(
    ctx: PipelineContext,
    stage: StageName,
    *,
    executed: set[StageName] | None = None,
    completed: list[StageName] | None = None,
    current_stage: dict[str, StageName] | None = None,
) -> PipelineContext:
    if executed is None:
        executed = set()
    if stage in executed:
        return ctx
    try:
        if current_stage is not None:
            current_stage["stage"] = stage
        result = _execute_stage(ctx, stage)
        executed.add(stage)
        if completed is not None:
            completed.append(stage)
        return result
    except StagePrerequisiteError as exc:
        required_stage = exc.required_stage
        if required_stage is None:
            raise
        _run_stage_for_pipeline(
            ctx,
            required_stage,
            executed=executed,
            completed=completed,
            current_stage=current_stage,
        )
        if current_stage is not None:
            current_stage["stage"] = stage
        result = _execute_stage(ctx, stage)
        executed.add(stage)
        if completed is not None:
            completed.append(stage)
        return result


def run_pipeline(
    config: PipelineConfig,
    *,
    start_stage: StageName | None = None,
    stop_stage: StageName | None = None,
    project_root: Path | str | None = None,
    run_name: str | None = None,
    paths: dict[str, Path] | None = None,
    run: RunManager | None = None,
    tracker: CostTracker | None = None,
    start_time: float | None = None,
    load_environment: bool = True,
) -> PipelineContext:
    prepared_config = prepare_pipeline_config(config)
    resolved_start = validate_stage_name(start_stage or prepared_config.run.start_stage)
    resolved_stop = (
        validate_stage_name(stop_stage) if stop_stage is not None else prepared_config.run.stop_stage
    )
    _set_random_seed(prepared_config.run.random_seed)
    ctx = PipelineContext.create(
        prepared_config,
        project_root=project_root,
        run_name=run_name,
        paths=paths,
        run=run,
        tracker=tracker,
        start_time=start_time,
        load_environment=load_environment,
        output_mode="cli",
    )
    ctx.run.save_config(prepared_config)
    reporter = getattr(ctx, "reporter", None)
    if reporter is not None:
        reporter.set_stage_plan(_stage_slice("search", resolved_stop))
    executed: set[StageName] = set()
    completed: list[StageName] = []
    current_stage: dict[str, StageName] = {}
    failed_stage: StageName | None = None
    error_message: str | None = None
    try:
        for stage in _stage_slice(resolved_start, resolved_stop):
            _run_stage_for_pipeline(
                ctx,
                stage,
                executed=executed,
                completed=completed,
                current_stage=current_stage,
            )
        return ctx
    except Exception as exc:
        failed_stage = current_stage.get("stage")
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _finalize_run_summary(
            ctx,
            status="failed" if error_message is not None else "completed",
            requested_start_stage=resolved_start,
            requested_stop_stage=resolved_stop,
            completed_stages=completed,
            failed_stage=failed_stage,
            error=error_message,
        )


def pipeline_config_to_dict(config: PipelineConfig | dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if isinstance(config, PipelineConfig):
        return config.to_dict()
    if is_dataclass(config):
        return asdict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if not isinstance(result, dict):
            raise TypeError("to_dict() must return a dict.")
        return result
    raise TypeError("Expected PipelineConfig, dataclass, dict, or object with to_dict().")


__all__ = [
    "AuthorDisambiguationConfig",
    "CitationsConfig",
    "CurationConfig",
    "PipelineConfig",
    "PipelineContext",
    "RunConfig",
    "SearchConfig",
    "StagePrerequisiteError",
    "STAGE_ORDER",
    "StageName",
    "TokenizeConfig",
    "TopicModelConfig",
    "TranslateConfig",
    "VisualizationConfig",
    "pipeline_config_to_dict",
    "prepare_pipeline_config",
    "run_pipeline",
    "validate_stage_name",
]
