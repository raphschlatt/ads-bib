"""NASA ADS bibliometric analysis pipeline."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ads_bib._utils.logging import suppress_noisy_third_party_logs

suppress_noisy_third_party_logs()

from ads_bib.notebook import NotebookSession, get_notebook_session
from ads_bib.pipeline import (
    PipelineConfig,
    PipelineContext,
    run_pipeline,
)
from ads_bib.run_manager import RunManager
from ads_bib.runner import RunBlockedError, run

try:
    __version__ = version("ads_bib")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "PipelineConfig",
    "PipelineContext",
    "NotebookSession",
    "RunManager",
    "RunBlockedError",
    "get_notebook_session",
    "run",
    "run_pipeline",
]
