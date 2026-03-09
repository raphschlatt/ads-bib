"""Logging helpers for consistent, low-noise runtime output."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import logging
import os
from pathlib import Path
from typing import Literal

from tqdm.auto import tqdm

OutputMode = Literal["cli", "notebook"]

_CONSOLE_HANDLER_NAME = "ads_bib_console"
_FILE_HANDLER_NAME = "ads_bib_runtime_file"
_CONSOLE_LOGGER_NAME = "ads_bib.console"
_RUNTIME_LOG_PATH: Path | None = None


class _ConsoleAllowList(logging.Filter):
    """Allow only curated console records for the active frontend."""

    def __init__(self, *, output_mode: OutputMode) -> None:
        super().__init__()
        self.output_mode = output_mode

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == _CONSOLE_LOGGER_NAME:
            return True
        if self.output_mode == "notebook" and record.name in {"pipeline", "ads_bib.notebook"}:
            return True
        return False


def _get_or_create_console_handler(root_logger: logging.Logger) -> logging.Handler:
    for handler in root_logger.handlers:
        if getattr(handler, "_ads_bib_handler_name", None) == _CONSOLE_HANDLER_NAME:
            return handler

    handler = logging.StreamHandler()
    setattr(handler, "_ads_bib_handler_name", _CONSOLE_HANDLER_NAME)
    root_logger.addHandler(handler)
    return handler


def _replace_handler_filter(handler: logging.Handler, new_filter: logging.Filter) -> None:
    for existing in list(handler.filters):
        if isinstance(existing, _ConsoleAllowList):
            handler.removeFilter(existing)
    handler.addFilter(new_filter)


def configure_runtime_logging(
    *,
    output_mode: OutputMode,
    log_file: Path | None = None,
) -> Path | None:
    """Configure console/file logging for notebook or CLI runtime."""
    global _RUNTIME_LOG_PATH

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = _get_or_create_console_handler(root_logger)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    _replace_handler_filter(console_handler, _ConsoleAllowList(output_mode=output_mode))

    file_handler: logging.Handler | None = None
    for handler in list(root_logger.handlers):
        if getattr(handler, "_ads_bib_handler_name", None) == _FILE_HANDLER_NAME:
            file_handler = handler
            break

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        resolved = log_file.resolve()
        _RUNTIME_LOG_PATH = resolved

        current_path = None
        if file_handler is not None:
            current_path = Path(getattr(file_handler, "baseFilename", "")).resolve()

        if file_handler is None or current_path != resolved:
            if file_handler is not None:
                root_logger.removeHandler(file_handler)
                file_handler.close()
            file_handler = logging.FileHandler(resolved, encoding="utf-8")
            setattr(file_handler, "_ads_bib_handler_name", _FILE_HANDLER_NAME)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            root_logger.addHandler(file_handler)
    elif file_handler is not None:
        root_logger.removeHandler(file_handler)
        file_handler.close()
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


class StageReporter:
    """Curated stage-first console output for CLI and notebook runs."""

    def __init__(self, *, output_mode: OutputMode) -> None:
        self.output_mode = output_mode
        self._logger = logging.getLogger(_CONSOLE_LOGGER_NAME)
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
    def progress(self, *, total: int, desc: str):
        if total <= 0:
            yield None
            return

        with tqdm(
            total=total,
            desc=f"  {desc}",
            leave=True,
            dynamic_ncols=False,
            ncols=78,
            bar_format="{desc:<18}{percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            yield pbar


def suppress_noisy_third_party_logs() -> None:
    """Suppress repetitive third-party transport logs while keeping pipeline logs."""
    try:
        for noisy_logger_name in ("httpx", "httpcore", "LiteLLM", "litellm"):
            logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

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
