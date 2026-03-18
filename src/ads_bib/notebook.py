"""Notebook-facing session adapter over the shared pipeline runner."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ads_bib._utils.costs import CostTracker
from ads_bib.pipeline import (
    _execute_stage,
    _finalize_run_summary,
    _set_resume_block,
    PipelineConfig,
    PipelineContext,
    STAGE_ORDER,
    StageName,
    prepare_pipeline_config,
    snapshot_block_from_invalidation,
    validate_stage_name,
)
from ads_bib.run_manager import RunManager

logger = logging.getLogger(__name__)

SECTION_NAMES: tuple[str, ...] = (
    "run",
    "search",
    "translate",
    "llama_server",
    "tokenize",
    "author_disambiguation",
    "topic_model",
    "visualization",
    "curation",
    "citations",
)

_ACTIVE_SESSION: NotebookSession | None = None


def _default_sections(project_root: Path, run_name: str) -> dict[str, dict[str, Any]]:
    data = PipelineConfig().to_dict()
    data["run"]["run_name"] = run_name
    data["run"]["project_root"] = str(project_root)
    return data


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _earliest_invalidation_stage(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> StageName | None:
    if previous is None:
        return None

    stage_checks: list[tuple[StageName, list[tuple[str, ...]]]] = [
        ("search", [("search",)]),
        (
            "translate",
            [
                ("translate",),
                ("translate", "model_repo"),
                ("translate", "model_file"),
                ("translate", "model_path"),
                ("run", "openrouter_cost_mode"),
            ],
        ),
        ("tokenize", [("tokenize",)]),
        ("author_disambiguation", [("author_disambiguation",)]),
        (
            "embeddings",
            [
                ("run", "random_seed"),
                ("topic_model", "sample_size"),
                ("topic_model", "embedding_provider"),
                ("topic_model", "embedding_model"),
                ("topic_model", "embedding_api_key"),
                ("topic_model", "embedding_batch_size"),
                ("topic_model", "embedding_max_workers"),
            ],
        ),
        (
            "reduction",
            [
                ("topic_model", "reduction_method"),
                ("topic_model", "params_5d"),
                ("topic_model", "params_2d"),
            ],
        ),
        (
            "topic_fit",
            [
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
            ],
        ),
        ("topic_fit", [("llama_server",)]),
        ("visualize", [("visualization",)]),
        ("curate", [("curation",)]),
        ("citations", [("citations",)]),
    ]

    for stage, paths_to_check in stage_checks:
        if any(_nested_get(previous, path) != _nested_get(current, path) for path in paths_to_check):
            return stage
    return None


def _invalidate_context_from(context: PipelineContext, stage: StageName) -> None:
    stage_name = validate_stage_name(stage)
    stage_index = STAGE_ORDER.index(stage_name)
    _set_resume_block(context, snapshot_block_from_invalidation(stage_name))

    if stage_index <= STAGE_ORDER.index("search"):
        context.bibcodes = None
        context.references = None
        context.esources = None
        context.fulltext_urls = None
        context.publications = None
        context.refs = None
    elif stage_name == "translate":
        context.publications = _drop_columns(
            context.publications,
            (
                "Title_lang",
                "Abstract_lang",
                "Title_en",
                "Abstract_en",
                "full_text",
                "tokens",
                "author_uids",
                "author_display_names",
            ),
        )
        context.refs = _drop_columns(
            context.refs,
            (
                "Title_lang",
                "Abstract_lang",
                "Title_en",
                "Abstract_en",
                "author_uids",
                "author_display_names",
            ),
        )
    elif stage_name == "tokenize":
        context.publications = _drop_columns(
            context.publications,
            ("full_text", "tokens", "author_uids", "author_display_names"),
        )
        context.refs = _drop_columns(
            context.refs,
            ("author_uids", "author_display_names"),
        )
    elif stage_name == "author_disambiguation":
        context.publications = _drop_columns(
            context.publications,
            ("author_uids", "author_display_names"),
        )
        context.refs = _drop_columns(
            context.refs,
            ("author_uids", "author_display_names"),
        )

    if stage_index <= STAGE_ORDER.index("embeddings"):
        context.topic_input_df = None
        context.documents = None
        context.embeddings = None
        context.reduced_5d = None
        context.reduced_2d = None
        context.topic_model = None
        context.topics = None
        context.topic_info = None
        context.topic_df = None
        context.curated_df = None
        context.citation_results = None
        return

    if stage_index <= STAGE_ORDER.index("reduction"):
        context.reduced_5d = None
        context.reduced_2d = None
        context.topic_model = None
        context.topics = None
        context.topic_info = None
        context.topic_df = None
        context.curated_df = None
        context.citation_results = None
        return

    if stage_index <= STAGE_ORDER.index("topic_fit"):
        context.topic_model = None
        context.topics = None
        context.topic_info = None
        context.topic_df = None
        context.curated_df = None
        context.citation_results = None
        return

    if stage_index <= STAGE_ORDER.index("topic_dataframe"):
        context.topic_df = None
        context.curated_df = None
        context.citation_results = None
        return

    if stage_index <= STAGE_ORDER.index("curate"):
        context.curated_df = None
        context.citation_results = None
        return

    if stage_index <= STAGE_ORDER.index("citations"):
        context.citation_results = None



def _drop_columns(
    frame: pd.DataFrame | None,
    columns: tuple[str, ...],
) -> pd.DataFrame | None:
    if frame is None:
        return None
    return frame.drop(columns=list(columns), errors="ignore")


class NotebookSession:
    """Interactive notebook session over the shared package pipeline."""

    def __init__(
        self,
        *,
        project_root: Path | str | None = None,
        run_name: str = "ADS_Curation_Run",
        start_time: float | None = None,
    ) -> None:
        self._project_root = Path(project_root or Path.cwd())
        self._run_name = run_name
        self._start_time = start_time
        self._section_defaults = _default_sections(self._project_root, run_name)
        self._sections = deepcopy(self._section_defaults)
        self._last_config_data: dict[str, Any] | None = None
        self._context: PipelineContext | None = None
        self._ensure_context(initial=True)

    def _config_from_sections(self) -> PipelineConfig:
        return PipelineConfig.from_dict(deepcopy(self._sections))

    def _prepared_config(self) -> PipelineConfig:
        return prepare_pipeline_config(self._config_from_sections())

    def _ensure_context(self, *, initial: bool = False) -> None:
        config_data = deepcopy(self._sections)
        prepared = self._prepared_config()
        random.seed(prepared.run.random_seed)
        np.random.seed(prepared.run.random_seed)

        if self._context is None or initial:
            self._context = PipelineContext.create(
                prepared,
                project_root=self._project_root,
                run_name=self._run_name,
                start_time=self._start_time,
                output_mode="notebook",
            )
        else:
            invalidation_stage = _earliest_invalidation_stage(self._last_config_data, config_data)
            if invalidation_stage is not None:
                _invalidate_context_from(self._context, invalidation_stage)
                logger.info(
                    "Config changed; invalidated in-memory state from stage '%s'.",
                    invalidation_stage,
                )
            self._context.config = prepared

        self._last_config_data = config_data
        self._context.run.save_config(prepared)

    @property
    def config(self) -> PipelineConfig:
        assert self._context is not None
        return self._context.config

    @property
    def publications(self) -> pd.DataFrame | None:
        assert self._context is not None
        return self._context.publications

    @property
    def refs(self) -> pd.DataFrame | None:
        assert self._context is not None
        return self._context.refs

    @property
    def documents(self) -> list[str] | None:
        assert self._context is not None
        return self._context.documents

    @property
    def embeddings(self) -> np.ndarray | None:
        assert self._context is not None
        return self._context.embeddings

    @property
    def reduced_5d(self) -> np.ndarray | None:
        assert self._context is not None
        return self._context.reduced_5d

    @property
    def reduced_2d(self) -> np.ndarray | None:
        assert self._context is not None
        return self._context.reduced_2d

    @property
    def topic_model(self) -> Any | None:
        assert self._context is not None
        return self._context.topic_model

    @property
    def topic_info(self) -> pd.DataFrame | None:
        assert self._context is not None
        return self._context.topic_info

    @property
    def topic_df(self) -> pd.DataFrame | None:
        assert self._context is not None
        return self._context.topic_df

    @property
    def curated_df(self) -> pd.DataFrame | None:
        assert self._context is not None
        return self._context.curated_df

    @property
    def citation_results(self) -> dict[str, pd.DataFrame] | None:
        assert self._context is not None
        return self._context.citation_results

    @property
    def run(self) -> RunManager:
        assert self._context is not None
        return self._context.run

    @property
    def paths(self) -> dict[str, Path]:
        assert self._context is not None
        return self._context.paths

    @property
    def tracker(self) -> CostTracker:
        assert self._context is not None
        return self._context.tracker

    def set_section(self, name: str, values: Mapping[str, Any]) -> None:
        if name not in SECTION_NAMES:
            allowed = ", ".join(SECTION_NAMES)
            raise ValueError(f"Unknown config section '{name}'. Expected one of: {allowed}.")
        if not isinstance(values, Mapping):
            raise TypeError("Section values must be a mapping.")

        section = deepcopy(self._section_defaults[name])
        section.update(dict(values))

        if name == "run":
            new_run_name = str(section.get("run_name", self._run_name))
            if new_run_name != self._run_name:
                raise ValueError(
                    "run.run_name cannot change within an existing notebook session. "
                    "Set RESET_SESSION=True and recreate the session to start a new run."
                )
            section["project_root"] = str(self._project_root)

        self._sections[name] = section
        self._ensure_context()

    def run_stage(self, stage: StageName | str) -> None:
        assert self._context is not None
        stage_name = validate_stage_name(stage)
        random.seed(self._context.config.run.random_seed)
        np.random.seed(self._context.config.run.random_seed)
        _execute_stage(self._context, stage_name)

    def save_summary(self) -> None:
        assert self._context is not None
        _finalize_run_summary(
            self._context,
            status="completed",
            requested_start_stage=None,
            requested_stop_stage=None,
        )


def get_notebook_session(
    *,
    project_root: Path | str | None = None,
    run_name: str = "ADS_Curation_Run",
    reset: bool = False,
    start_time: float | None = None,
) -> NotebookSession:
    global _ACTIVE_SESSION

    resolved_root = Path(project_root or Path.cwd())
    if reset or _ACTIVE_SESSION is None:
        _ACTIVE_SESSION = NotebookSession(
            project_root=resolved_root,
            run_name=run_name,
            start_time=start_time,
        )
        return _ACTIVE_SESSION

    if _ACTIVE_SESSION._run_name != run_name:
        raise ValueError(
            "Notebook session already exists with a different run_name. "
            "Set RESET_SESSION=True to create a new run."
        )
    if _ACTIVE_SESSION._project_root != resolved_root:
        raise ValueError(
            "Notebook session already exists with a different project_root. "
            "Set RESET_SESSION=True to recreate it."
        )
    return _ACTIVE_SESSION


__all__ = ["NotebookSession", "get_notebook_session"]
