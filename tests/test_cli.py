from __future__ import annotations

import ads_bib.cli as cli


def test_run_quality_checks_runs_ruff_then_pytest_with_pythonpath():
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command, env):
        calls.append((list(command), env))
        return 0

    rc = cli.run_quality_checks(run_command=fake_runner)

    assert rc == 0
    assert calls[0][0] == ["ruff", "check", "src", "tests", "scripts"]
    assert calls[0][1] is None
    assert calls[1][0] == ["pytest", "-q"]
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
    assert calls == [["ruff", "check", "src", "tests", "scripts"]]


def test_main_dispatches_check(monkeypatch):
    monkeypatch.setattr(cli, "run_quality_checks", lambda **_: 7)
    assert cli.main(["check"]) == 7
