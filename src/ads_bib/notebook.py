"""Notebook-facing session adapter over the shared pipeline runner."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ads_bib._stage_state import (
    StageName,
    _earliest_invalidation_stage,
    _invalidate_context_from,
    validate_stage_name,
)
from ads_bib._utils.costs import CostTracker
from ads_bib.pipeline import (
    _execute_stage,
    _finalize_run_summary,
    PipelineConfig,
    PipelineContext,
    prepare_pipeline_config,
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
