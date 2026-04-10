from __future__ import annotations

from pathlib import Path

import ads_bib
import ads_bib.doctor as doctor
import ads_bib.runner as runner
import pytest


def _passing_report(*, stages: tuple[str, ...] = ("search",)) -> doctor.DoctorReport:
    return doctor.DoctorReport(
        checks=(doctor.DoctorCheck(name="search.query", status="ok", detail="configured"),),
        active_stages=stages,
    )


def test_run_with_preset_query_uses_preflight_and_pipeline(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_run_pipeline(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs
        return "ctx"

    monkeypatch.setattr(runner, "ensure_default_fasttext_model", lambda **kwargs: None)
    monkeypatch.setattr(runner, "collect_doctor_report", lambda *args, **kwargs: _passing_report())
    monkeypatch.setattr(runner, "run_pipeline", _fake_run_pipeline)

    result = ads_bib.run(
        preset="openrouter",
        query="author:test",
        start_stage="search",
        stop_stage="search",
        run_name="api-test",
    )

    assert result == "ctx"
    assert calls["config"].search.query == "author:test"
    assert calls["config"].run.run_name == "ads_bib_openrouter"
    assert calls["kwargs"] == {
        "start_stage": "search",
        "stop_stage": "search",
        "project_root": None,
        "run_name": "api-test",
    }


def test_run_with_yaml_config_applies_dotted_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "ads-bib.yaml"
    config_path.write_text(
        "run:\n  run_name: yaml-test\nsearch:\n  query: old\n",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    def _fake_run_pipeline(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs
        return "ctx"

    monkeypatch.setattr(runner, "run_pipeline", _fake_run_pipeline)

    result = ads_bib.run(
        config=config_path,
        overrides={
            "search.query": "author:new",
            "topic_model.backend": "toponymy",
            "topic_model.llm_provider": "openrouter",
        },
        preflight=False,
    )

    assert result == "ctx"
    assert calls["config"].search.query == "author:new"
    assert calls["config"].topic_model.backend == "toponymy"
    assert calls["config"].topic_model.llm_provider == "openrouter"


def test_run_requires_exactly_one_config_source():
    with pytest.raises(ValueError, match="Exactly one"):
        ads_bib.run()

    with pytest.raises(ValueError, match="Exactly one"):
        ads_bib.run(preset="openrouter", config={})


def test_run_rejects_query_override_conflict():
    with pytest.raises(ValueError, match="search.query"):
        ads_bib.run(
            preset="openrouter",
            query="author:test",
            overrides={"search.query": "author:other"},
        )


def test_run_preflight_failure_raises_run_blocked_error(monkeypatch):
    report = doctor.DoctorReport(
        checks=(
            doctor.DoctorCheck(name="search.query", status="ok", detail="configured"),
            doctor.DoctorCheck(name="search.ads_token", status="fail", detail="missing ADS token"),
        ),
        active_stages=("search",),
    )

    monkeypatch.setattr(runner, "ensure_default_fasttext_model", lambda **kwargs: None)
    monkeypatch.setattr(runner, "collect_doctor_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(
        runner,
        "run_pipeline",
        lambda *args, **kwargs: pytest.fail("run_pipeline should not be called"),
    )

    with pytest.raises(runner.RunBlockedError) as exc_info:
        ads_bib.run(preset="openrouter", query="author:test")

    assert "Run blocked by preflight checks" in str(exc_info.value)
    assert "search.ads_token" in str(exc_info.value)
    assert exc_info.value.report is report


def test_load_run_config_keeps_legacy_preset_migration_hint():
    with pytest.raises(FileNotFoundError, match="Legacy preset file") as exc_info:
        runner.load_run_config(config=Path("configs/pipeline/local_cpu.yaml"))

    assert "--preset local_cpu" in str(exc_info.value)
