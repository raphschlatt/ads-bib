"""Shared stage ordering, resume, and notebook invalidation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import pandas as pd

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


class _StageStateContext(Protocol):
    bibcodes: list[str] | None
    references: list[list[str]] | None
    esources: list[list[str]] | None
    fulltext_urls: list[str | None] | None
    publications: pd.DataFrame | None
    refs: pd.DataFrame | None
    topic_input_df: pd.DataFrame | None
    documents: list[str] | None
    embeddings: Any | None
    reduced_5d: Any | None
    reduced_2d: Any | None
    topic_model: Any | None
    topics: Any | None
    topic_info: pd.DataFrame | None
    topic_hierarchy: dict[str, Any] | None
    topic_df: pd.DataFrame | None
    curated_df: pd.DataFrame | None
    citation_results: dict[str, pd.DataFrame] | None
    resume_blocked_from: StageName | None


@dataclass(frozen=True)
class _InvalidationRule:
    config_paths: tuple[tuple[str, ...], ...] = ()
    clear_fields: tuple[str, ...] = ()
    drop_publications: tuple[str, ...] = ()
    drop_refs: tuple[str, ...] = ()


_INVALIDATION_RULES: dict[StageName, _InvalidationRule] = {
    "search": _InvalidationRule(
        config_paths=(("search",),),
        clear_fields=(
            "bibcodes",
            "references",
            "esources",
            "fulltext_urls",
            "publications",
            "refs",
            "topic_input_df",
            "documents",
            "embeddings",
            "reduced_5d",
            "reduced_2d",
            "topic_model",
            "topics",
            "topic_info",
            "topic_hierarchy",
            "topic_df",
            "curated_df",
            "citation_results",
        ),
    ),
    "translate": _InvalidationRule(
        config_paths=(
            ("translate",),
            ("translate", "model_repo"),
            ("translate", "model_file"),
            ("translate", "model_path"),
            ("run", "openrouter_cost_mode"),
        ),
        drop_publications=(
            "Title_lang",
            "Abstract_lang",
            "Title_en",
            "Abstract_en",
            "full_text",
            "tokens",
            "author_uids",
            "author_display_names",
        ),
        drop_refs=(
            "Title_lang",
            "Abstract_lang",
            "Title_en",
            "Abstract_en",
            "author_uids",
            "author_display_names",
        ),
        clear_fields=(
            "topic_input_df",
            "documents",
            "embeddings",
            "reduced_5d",
            "reduced_2d",
            "topic_model",
            "topics",
            "topic_info",
            "topic_hierarchy",
            "topic_df",
            "curated_df",
            "citation_results",
        ),
    ),
    "tokenize": _InvalidationRule(
        config_paths=(("tokenize",),),
        drop_publications=("full_text", "tokens", "author_uids", "author_display_names"),
        drop_refs=("author_uids", "author_display_names"),
        clear_fields=(
            "topic_input_df",
            "documents",
            "embeddings",
            "reduced_5d",
            "reduced_2d",
            "topic_model",
            "topics",
            "topic_info",
            "topic_hierarchy",
            "topic_df",
            "curated_df",
            "citation_results",
        ),
    ),
    "author_disambiguation": _InvalidationRule(
        config_paths=(("author_disambiguation",),),
        drop_publications=("author_uids", "author_display_names"),
        drop_refs=("author_uids", "author_display_names"),
        clear_fields=(
            "topic_input_df",
            "documents",
            "embeddings",
            "reduced_5d",
            "reduced_2d",
            "topic_model",
            "topics",
            "topic_info",
            "topic_hierarchy",
            "topic_df",
            "curated_df",
            "citation_results",
        ),
    ),
    "embeddings": _InvalidationRule(
        config_paths=(
            ("run", "random_seed"),
            ("topic_model", "sample_size"),
            ("topic_model", "embedding_provider"),
            ("topic_model", "embedding_model"),
            ("topic_model", "embedding_api_key"),
            ("topic_model", "embedding_batch_size"),
            ("topic_model", "embedding_max_workers"),
        ),
        clear_fields=(
            "topic_input_df",
            "documents",
            "embeddings",
            "reduced_5d",
            "reduced_2d",
            "topic_model",
            "topics",
            "topic_info",
            "topic_hierarchy",
            "topic_df",
            "curated_df",
            "citation_results",
        ),
    ),
    "reduction": _InvalidationRule(
        config_paths=(
            ("topic_model", "reduction_method"),
            ("topic_model", "params_5d"),
            ("topic_model", "params_2d"),
        ),
        clear_fields=(
            "reduced_5d",
            "reduced_2d",
            "topic_model",
            "topics",
            "topic_info",
            "topic_hierarchy",
            "topic_df",
            "curated_df",
            "citation_results",
        ),
    ),
    "topic_fit": _InvalidationRule(
        config_paths=(
            ("topic_model", "backend"),
            ("topic_model", "clustering_method"),
            ("topic_model", "cluster_params"),
            ("topic_model", "toponymy_cluster_params"),
            ("topic_model", "toponymy_layer_index"),
            ("topic_model", "llm_prompt_name"),
            ("topic_model", "llm_prompt"),
            ("topic_model", "llm_provider"),
            ("topic_model", "llm_model"),
            ("topic_model", "llm_model_repo"),
            ("topic_model", "llm_model_file"),
            ("topic_model", "llm_model_path"),
            ("topic_model", "llm_api_key"),
            ("topic_model", "bertopic_label_max_tokens"),
            ("topic_model", "toponymy_local_label_max_tokens"),
            ("topic_model", "pipeline_models"),
            ("topic_model", "parallel_models"),
            ("topic_model", "toponymy_embedding_model"),
            ("topic_model", "toponymy_max_workers"),
            ("topic_model", "min_df"),
            ("topic_model", "outlier_threshold"),
            ("llama_server",),
        ),
        clear_fields=(
            "topic_model",
            "topics",
            "topic_info",
            "topic_hierarchy",
            "topic_df",
            "curated_df",
            "citation_results",
        ),
    ),
    "visualize": _InvalidationRule(config_paths=(("visualization",),)),
    "curate": _InvalidationRule(
        config_paths=(("curation",),),
        clear_fields=("curated_df", "citation_results"),
    ),
    "citations": _InvalidationRule(
        config_paths=(("citations",),),
        clear_fields=("citation_results",),
    ),
}


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


def _set_resume_block(ctx: _StageStateContext, candidate: StageName | None) -> None:
    if candidate is None:
        return
    if (
        ctx.resume_blocked_from is None
        or STAGE_ORDER.index(candidate) < STAGE_ORDER.index(ctx.resume_blocked_from)
    ):
        ctx.resume_blocked_from = candidate


def _snapshot_allowed(ctx: _StageStateContext, snapshot_stage: StageName) -> bool:
    blocked_from = ctx.resume_blocked_from
    if blocked_from is None:
        return True
    return STAGE_ORDER.index(snapshot_stage) < STAGE_ORDER.index(blocked_from)


def _advance_resume_block(ctx: _StageStateContext, completed_stage: StageName) -> None:
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


def _nested_get_mapping(data: Mapping[str, Any] | None, path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _drop_columns(
    frame: pd.DataFrame | None,
    columns: tuple[str, ...],
) -> pd.DataFrame | None:
    if frame is None or not columns:
        return frame
    return frame.drop(columns=list(columns), errors="ignore")


def _earliest_invalidation_stage(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> StageName | None:
    if previous is None:
        return None

    for stage, rule in _INVALIDATION_RULES.items():
        if any(
            _nested_get_mapping(previous, path) != _nested_get_mapping(current, path)
            for path in rule.config_paths
        ):
            return stage
    return None


def _invalidate_context_from(context: _StageStateContext, stage: StageName) -> None:
    stage_name = validate_stage_name(stage)
    rule = _INVALIDATION_RULES.get(stage_name)
    _set_resume_block(context, snapshot_block_from_invalidation(stage_name))
    if rule is None:
        return

    context.publications = _drop_columns(context.publications, rule.drop_publications)
    context.refs = _drop_columns(context.refs, rule.drop_refs)
    for field_name in rule.clear_fields:
        setattr(context, field_name, None)


__all__ = [
    "STAGE_ORDER",
    "StageName",
    "_advance_resume_block",
    "_earliest_invalidation_stage",
    "_invalidate_context_from",
    "_set_resume_block",
    "_snapshot_allowed",
    "snapshot_block_from_invalidation",
    "validate_stage_name",
]
