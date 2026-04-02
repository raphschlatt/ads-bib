"""Logging helpers for consistent, low-noise runtime output."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import logging
import os
from pathlib import Path
from typing import Literal

from tqdm.auto import tqdm

# Suppress TensorFlow/oneDNN C++ warnings that bypass Python logging.
# These env vars must be set before any transitive TF import (e.g. via
# BERTopic → UMAP → parametric_umap, or thinc, or transformers).
# TF's C++ runtime reads them once at initialization; setting them later
# or restoring them per-scope has no effect.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

OutputMode = Literal["cli", "notebook"]

_CONSOLE_HANDLER_NAME = "ads_bib_console"
_FILE_HANDLER_NAME = "ads_bib_runtime_file"
_CONSOLE_LOGGER_NAME = "ads_bib.console"
_PACKAGE_LOGGER_NAME = "ads_bib"
_RUNTIME_LOG_PATH: Path | None = None


def _remove_named_handler(logger: logging.Logger, handler_name: str) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_ads_bib_handler_name", None) == handler_name:
            logger.removeHandler(handler)
            handler.close()


def _get_or_create_named_handler(
    logger: logging.Logger,
    *,
    handler_name: str,
    factory: type[logging.Handler],
    factory_arg: Path | None = None,
) -> logging.Handler:
    for handler in logger.handlers:
        if getattr(handler, "_ads_bib_handler_name", None) == handler_name:
            return handler

    if factory_arg is None:
        handler = factory()
    else:
        handler = factory(factory_arg, encoding="utf-8")
    setattr(handler, "_ads_bib_handler_name", handler_name)
    logger.addHandler(handler)
    return handler


def get_console_logger() -> logging.Logger:
    """Return the dedicated frontend console logger."""
    return logging.getLogger(_CONSOLE_LOGGER_NAME)


def get_console_stream():
    """Return the active console handler stream when runtime logging is configured."""
    console_logger = get_console_logger()
    for handler in console_logger.handlers:
        if getattr(handler, "_ads_bib_handler_name", None) == _CONSOLE_HANDLER_NAME:
            return getattr(handler, "stream", None)
    return None


def configure_runtime_logging(
    *,
    output_mode: OutputMode,
    log_file: Path | None = None,
) -> Path | None:
    """Configure console/file logging for notebook or CLI runtime."""
    global _RUNTIME_LOG_PATH

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    _remove_named_handler(root_logger, _CONSOLE_HANDLER_NAME)
    _remove_named_handler(root_logger, _FILE_HANDLER_NAME)

    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False

    console_logger = get_console_logger()
    console_logger.setLevel(logging.INFO)
    console_logger.propagate = False

    console_handler = _get_or_create_named_handler(
        console_logger,
        handler_name=_CONSOLE_HANDLER_NAME,
        factory=logging.StreamHandler,
    )
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    if output_mode == "cli":
        console_handler.terminator = "\n"

    file_handler_name = _FILE_HANDLER_NAME

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        resolved = log_file.resolve()
        _RUNTIME_LOG_PATH = resolved
        for logger in (package_logger, console_logger):
            current_path = None
            for handler in logger.handlers:
                if getattr(handler, "_ads_bib_handler_name", None) == file_handler_name:
                    current_path = Path(getattr(handler, "baseFilename", "")).resolve()
                    if current_path == resolved:
                        break
            else:
                current_path = None

            if current_path != resolved:
                _remove_named_handler(logger, file_handler_name)
                file_handler = _get_or_create_named_handler(
                    logger,
                    handler_name=file_handler_name,
                    factory=logging.FileHandler,
                    factory_arg=resolved,
                )
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
                )
    else:
        _remove_named_handler(package_logger, file_handler_name)
        _remove_named_handler(console_logger, file_handler_name)
        _RUNTIME_LOG_PATH = None

    return _RUNTIME_LOG_PATH


def get_runtime_log_path() -> Path | None:
    """Return the active runtime log file path, when configured."""
    return _RUNTIME_LOG_PATH


@contextmanager
def capture_external_output(log_file: Path | None = None):
    """Redirect raw stdout/stderr from third-party libraries into the run log."""
    target = log_file or _RUNTIME_LOG_PATH
    if target is None:
        yield
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            yield


@contextmanager
def temporarily_raise_logger_level(logger_name: str, *, level: int) -> None:
    """Temporarily raise one logger to at least ``level`` and then restore it."""
    logger = logging.getLogger(logger_name)
    previous_level = logger.level
    logger.setLevel(max(level, logger.getEffectiveLevel()))
    try:
        yield
    finally:
        logger.setLevel(previous_level)


class StageReporter:
    """Curated stage-first console output for CLI and notebook runs."""

    def __init__(self, *, output_mode: OutputMode) -> None:
        self.output_mode = output_mode
        self._logger = get_console_logger()
        self._started_any = False
        self._stage_positions: dict[str, int] = {}
        self._stage_total: int | None = None

    def set_stage_plan(self, stages: list[str] | tuple[str, ...]) -> None:
        self._stage_positions = {stage: idx + 1 for idx, stage in enumerate(stages)}
        self._stage_total = len(stages)

    def stage_start(self, stage: str) -> None:
        if self._started_any:
            self._logger.info("")
        self._started_any = True
        title = stage.replace("_", " ").title()
        if self.output_mode == "cli" and self._stage_total and stage in self._stage_positions:
            self._logger.info("[%s/%s] %s", self._stage_positions[stage], self._stage_total, title)
            return
        self._logger.info(title)

    def detail(self, message: str, *args: object) -> None:
        self._logger.info("  " + message, *args)

    def warning(self, message: str, *args: object) -> None:
        self._logger.warning("  warning: " + message, *args)

    @contextmanager
    def progress(self, *, total: int | None, desc: str):
        if total is not None and total <= 0:
            yield None
            return

        kwargs: dict[str, object] = {"total": total, "desc": f"  {desc}", "leave": True}
        console_stream = get_console_stream()
        if console_stream is not None:
            kwargs["file"] = console_stream
        if self.output_mode == "cli":
            kwargs["dynamic_ncols"] = False
            kwargs["ncols"] = 78
            if total is None:
                kwargs["bar_format"] = "{desc:<18}{n_fmt} [{elapsed}]"
            else:
                kwargs["bar_format"] = (
                    "{desc:<18}{percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )
        else:
            if total is None:
                kwargs["bar_format"] = "{desc}: {n_fmt} [{elapsed}]"
        with tqdm(**kwargs) as pbar:
            yield pbar


def suppress_noisy_third_party_logs() -> None:
    """Suppress repetitive third-party transport logs while keeping pipeline logs."""
    try:
        for noisy_logger_name in (
            "httpx",
            "httpcore",
            "LiteLLM",
            "litellm",
            "transformers",
            "sentence_transformers",
            "BERTopic",
            "bertopic",
        ):
            logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

        # TF/absl emit deprecation warnings at WARNING level during import;
        # suppress them more aggressively since they are never actionable.
        for tf_logger_name in ("tensorflow", "absl"):
            logging.getLogger(tf_logger_name).setLevel(logging.ERROR)

        os.environ.setdefault("LITELLM_LOG", "WARNING")
        os.environ.setdefault("LITELLM_VERBOSE", "False")

        try:
            import litellm
        except Exception:
            return

        set_verbose_attr = getattr(litellm, "set_verbose", None)
        if callable(set_verbose_attr):
            try:
                set_verbose_attr(False)
            except Exception:
                pass
        elif set_verbose_attr is not None:
            try:
                setattr(litellm, "set_verbose", False)
            except Exception:
                pass

        if hasattr(litellm, "suppress_debug_info"):
            try:
                litellm.suppress_debug_info = True
            except Exception:
                pass
    except Exception:
        return
