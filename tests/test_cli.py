from __future__ import annotations

from pathlib import Path
import sys

import ads_bib.cli as cli
import ads_bib.doctor as doctor
import pytest


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


def test_main_dispatches_run_with_preset(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_run_pipeline(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_pipeline", _fake_run_pipeline)
    rc = cli.main(
        [
            "run",
            "--preset",
            "openrouter",
            "--set",
            "search.query=author:test",
        ]
    )

    assert rc == 0
    assert calls["config"].search.query == "author:test"
    assert calls["config"].run.run_name == "ads_bib_openrouter"
    assert calls["kwargs"] == {
        "start_stage": None,
        "stop_stage": None,
        "run_name": None,
    }


def test_main_run_requires_search_query(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  run_name: test\nsearch:\n  query: ''\n", encoding="utf-8")
    monkeypatch.setattr(cli, "run_pipeline", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="search.query"):
        cli.main(["run", "--config", str(config_path)])


def test_main_run_missing_legacy_config_path_shows_preset_migration_hint():
    with pytest.raises(FileNotFoundError, match="Legacy preset file") as exc_info:
        cli.main(["run", "--config", "configs/pipeline/local_cpu.yaml"])

    assert "--preset local_cpu" in str(exc_info.value)


def test_main_run_missing_arbitrary_config_path_keeps_normal_file_not_found():
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        cli.main(["run", "--config", "missing.yaml"])


def test_preset_list_prints_all_names(capsys):
    rc = cli.main(["preset", "list"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "openrouter" in output
    assert "hf_api" in output
    assert "local_cpu" in output
    assert "local_gpu" in output


def test_preset_write_writes_yaml(tmp_path, capsys):
    output_path = tmp_path / "openrouter.yaml"

    rc = cli.main(["preset", "write", "openrouter", "--output", str(output_path)])

    assert rc == 0
    assert output_path.exists()
    assert "ads_bib_openrouter" in output_path.read_text(encoding="utf-8")
    assert "Wrote preset 'openrouter'" in capsys.readouterr().out


def test_main_dispatches_bootstrap(monkeypatch, capsys):
    monkeypatch.setattr(cli, "bootstrap_workspace", lambda **_: ["line one", "line two"])

    rc = cli.main(["bootstrap"])

    assert rc == 0
    assert capsys.readouterr().out == "line one\nline two\n"


def test_main_dispatches_doctor(monkeypatch, capsys):
    report = doctor.DoctorReport(
        checks=(doctor.DoctorCheck(name="search.query", status="ok", detail="configured"),),
        active_stages=("search",),
    )
    monkeypatch.setattr(cli, "collect_doctor_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(cli, "format_doctor_report", lambda report: "doctor ok\n")

    rc = cli.main([
        "doctor",
        "--preset",
        "openrouter",
        "--set",
        "search.query=author:test",
    ])

    assert rc == 0
    assert capsys.readouterr().out == "doctor ok\n"


def test_main_doctor_returns_nonzero_on_failures(monkeypatch, capsys):
    report = doctor.DoctorReport(
        checks=(doctor.DoctorCheck(name="search.ads_token", status="fail", detail="missing"),),
        active_stages=("search",),
    )
    monkeypatch.setattr(cli, "collect_doctor_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(cli, "format_doctor_report", lambda report: "doctor failed\n")

    rc = cli.main([
        "doctor",
        "--preset",
        "openrouter",
        "--set",
        "search.query=author:test",
    ])

    assert rc == 1
    assert capsys.readouterr().out == "doctor failed\n"


def test_parse_override_requires_equals():
    try:
        cli._parse_override("invalid")
    except ValueError as exc:
        assert "key=value" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
