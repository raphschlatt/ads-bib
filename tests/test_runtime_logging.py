from __future__ import annotations

import logging
import io

from ads_bib._utils import logging as logging_utils


def test_configure_runtime_logging_cli_filters_console_and_writes_run_log(tmp_path):
    log_path = tmp_path / "runtime.log"
    logging_utils.configure_runtime_logging(output_mode="cli", log_file=log_path)

    console_logger = logging_utils.get_console_logger()
    console_handler = next(
        handler for handler in console_logger.handlers
        if getattr(handler, "_ads_bib_handler_name", None) == "ads_bib_console"
    )

    visible = logging.LogRecord("ads_bib.console", logging.INFO, __file__, 1, "visible", (), None)

    assert bool(console_handler.filter(visible)) is True
    assert console_logger.propagate is False

    with logging_utils.capture_external_output(log_path):
        print("hidden third-party line")
    assert "hidden third-party line" in log_path.read_text(encoding="utf-8")


def test_stage_reporter_cli_uses_stage_numbering():
    reporter = logging_utils.StageReporter(output_mode="cli")
    reporter.set_stage_plan(("search", "export", "translate"))

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    console_logger = logging_utils.get_console_logger()
    console_logger.addHandler(handler)
    try:
        reporter.stage_start("search")
        reporter.detail("result: 382 publications")
        reporter.warning("embedding cache invalidated because document set changed")
    finally:
        console_logger.removeHandler(handler)

    text = stream.getvalue()
    assert "[1/3] Search" in text
    assert "result: 382 publications" in text
    assert "warning: embedding cache invalidated because document set changed" in text


def test_stage_reporter_notebook_uses_plain_stage_titles():
    reporter = logging_utils.StageReporter(output_mode="notebook")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    console_logger = logging_utils.get_console_logger()
    console_logger.addHandler(handler)
    try:
        reporter.stage_start("topic_fit")
    finally:
        console_logger.removeHandler(handler)

    text = stream.getvalue()
    assert "Topic Fit" in text
    assert "[1/" not in text
    assert "===" not in text


def test_stage_reporter_progress_stays_visible_inside_capture_external_output(tmp_path):
    log_path = tmp_path / "runtime.log"
    logging_utils.configure_runtime_logging(output_mode="cli", log_file=log_path)

    console_logger = logging_utils.get_console_logger()
    console_handler = next(
        handler for handler in console_logger.handlers
        if getattr(handler, "_ads_bib_handler_name", None) == "ads_bib_console"
    )

    stream = io.StringIO()
    previous_stream = console_handler.setStream(stream)
    reporter = logging_utils.StageReporter(output_mode="cli")
    try:
        with logging_utils.capture_external_output(log_path):
            print("hidden third-party line")
            with reporter.progress(total=2, desc="fit") as pbar:
                assert pbar is not None
                pbar.update(1)
                pbar.update(1)
    finally:
        console_handler.setStream(previous_stream)

    console_text = stream.getvalue()
    log_text = log_path.read_text(encoding="utf-8")

    assert "fit" in console_text
    assert "2/2" in console_text
    assert "hidden third-party line" in log_text
    assert "2/2" not in log_text
