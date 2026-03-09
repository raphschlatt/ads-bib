"""Pytest config for local package imports."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ads_bib._utils import logging as logging_utils


@pytest.fixture(autouse=True)
def _reset_runtime_logging_state():
    yield

    handler_names = {
        getattr(logging_utils, "_CONSOLE_HANDLER_NAME", "ads_bib_console"),
        getattr(logging_utils, "_FILE_HANDLER_NAME", "ads_bib_runtime_file"),
    }
    for logger_name in ("", "ads_bib", "ads_bib.console"):
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            if getattr(handler, "_ads_bib_handler_name", None) in handler_names:
                logger.removeHandler(handler)
                handler.close()
        logger.propagate = True
    logging_utils._RUNTIME_LOG_PATH = None
