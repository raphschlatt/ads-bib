from __future__ import annotations

from pathlib import Path
import sys

import ads_bib.cli as cli


def test_run_quality_checks_runs_ruff_then_pytest_with_pythonpath():
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command, env):
        calls.append((list(command), env))
        return 0

    rc = cli.run_quality_checks(run_command=fake_runner)

    assert rc == 0
    assert calls[0][0] == [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]
    assert calls[0][1] is None
    assert calls[1][0] == [sys.executable, "-m", "pytest", "-q"]
    assert calls[1][1] is not None
    assert calls[1][1]["PYTHONPATH"] == "src"


def test_run_quality_checks_stops_after_first_failure():
    calls: list[list[str]] = []

    def fake_runner(command, env):
        _ = env
        calls.append(list(command))
        return 1

    rc = cli.run_quality_checks(run_command=fake_runner)

    assert rc == 1
    assert calls == [[sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]]


def test_main_dispatches_check(monkeypatch):
    monkeypatch.setattr(cli, "run_quality_checks", lambda **_: 7)
    assert cli.main(["check"]) == 7


def test_main_dispatches_run(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  run_name: test\nsearch:\n  query: q\n", encoding="utf-8")
    calls: dict[str, object] = {}

    def _fake_run_pipeline(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_pipeline", _fake_run_pipeline)
    rc = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--from",
            "translate",
            "--to",
            "citations",
            "--run-name",
            "cli-test",
            "--set",
            "search.query=author:test",
        ]
    )

    assert rc == 0
    assert calls["config"].search.query == "author:test"
    assert calls["kwargs"] == {
        "start_stage": "translate",
        "stop_stage": "citations",
        "run_name": "cli-test",
    }


def test_parse_override_requires_equals():
    try:
        cli._parse_override("invalid")
    except ValueError as exc:
        assert "key=value" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
