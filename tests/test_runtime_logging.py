from __future__ import annotations

import logging

from ads_bib._utils import logging as logging_utils


def test_configure_runtime_logging_cli_filters_console_and_writes_run_log(tmp_path):
    log_path = tmp_path / "runtime.log"
    logging_utils.configure_runtime_logging(output_mode="cli", log_file=log_path)

    root = logging.getLogger()
    console_handler = next(
        handler
        for handler in root.handlers
        if getattr(handler, "_ads_bib_handler_name", None) == "ads_bib_console"
    )

    visible = logging.LogRecord("ads_bib.console", logging.INFO, __file__, 1, "visible", (), None)
    hidden = logging.LogRecord("transformers", logging.INFO, __file__, 1, "hidden", (), None)

    assert bool(console_handler.filter(visible)) is True
    assert bool(console_handler.filter(hidden)) is False

    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("transformers").info("hidden third-party line")
    assert "hidden third-party line" in log_path.read_text(encoding="utf-8")


def test_stage_reporter_cli_uses_stage_numbering(caplog):
    reporter = logging_utils.StageReporter(output_mode="cli")
    reporter.set_stage_plan(("search", "export", "translate"))

    with caplog.at_level(logging.INFO):
        reporter.stage_start("search")
        reporter.detail("result: 382 publications")
        reporter.warning("embedding cache invalidated because document set changed")

    assert "[1/3] Search" in caplog.text
    assert "result: 382 publications" in caplog.text
    assert "warning: embedding cache invalidated because document set changed" in caplog.text


def test_stage_reporter_notebook_uses_plain_stage_titles(caplog):
    reporter = logging_utils.StageReporter(output_mode="notebook")

    with caplog.at_level(logging.INFO):
        reporter.stage_start("topic_fit")

    assert "Topic Fit" in caplog.text
    assert "[1/" not in caplog.text
    assert "===" not in caplog.text
