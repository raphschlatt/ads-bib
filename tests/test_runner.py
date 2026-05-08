from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ads_bib
import ads_bib.doctor as doctor
import ads_bib.pipeline as pipeline
import ads_bib.runner as runner
import pandas as pd
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
    monkeypatch.setattr(runner, "_detect_output_mode", lambda: "notebook")
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
        "output_mode": "notebook",
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

    monkeypatch.setattr(runner, "_detect_output_mode", lambda: "cli")
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
    assert calls["kwargs"]["output_mode"] == "cli"


def test_run_with_from_run_delegates_to_variant_plan(monkeypatch, tmp_path):
    calls: dict[str, object] = {}
    base_run = tmp_path / "runs" / "run_base"
    plan = SimpleNamespace(
        config=pipeline.PipelineConfig.from_dict({"search": {"query": "author:base"}}),
        effective_start_stage="topic_fit",
        stop_stage="citations",
        run_name="variant-name",
        initial_state=pipeline.PipelineInitialState(),
        variant={"base_run_id": "run_base"},
    )

    def _fake_plan_run_variant(**kwargs):
        calls["plan_kwargs"] = kwargs
        return plan

    def _fake_run_resolved_config(config, **kwargs):
        calls["config"] = config
        calls["run_kwargs"] = kwargs
        return "ctx"

    monkeypatch.setattr(runner, "plan_run_variant", _fake_plan_run_variant)
    monkeypatch.setattr(runner, "run_resolved_config", _fake_run_resolved_config)

    result = ads_bib.run(
        from_run=base_run,
        query="author:new",
        overrides={"curation.clusters_to_remove": [7, 12]},
        start_stage="curate",
        stop_stage="citations",
        run_name="variant-name",
        project_root=tmp_path,
        output_mode="notebook",
    )

    assert result == "ctx"
    assert calls["plan_kwargs"] == {
        "from_run": base_run,
        "overrides": {
            "curation.clusters_to_remove": [7, 12],
            "search.query": "author:new",
        },
        "start_stage": "curate",
        "stop_stage": "citations",
        "run_name": "variant-name",
        "project_root": tmp_path,
    }
    assert calls["config"] is plan.config
    assert calls["run_kwargs"] == {
        "start_stage": "topic_fit",
        "stop_stage": "citations",
        "run_name": "variant-name",
        "project_root": tmp_path,
        "preflight": True,
        "output_mode": "notebook",
        "initial_state": plan.initial_state,
        "variant": plan.variant,
    }


def test_run_accepts_explicit_output_mode(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_run_pipeline(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs
        return "ctx"

    monkeypatch.setattr(runner, "run_pipeline", _fake_run_pipeline)

    result = ads_bib.run(
        config={"search": {"query": "author:test"}},
        preflight=False,
        output_mode="notebook",
    )

    assert result == "ctx"
    assert calls["kwargs"]["output_mode"] == "notebook"


def test_run_requires_exactly_one_config_source():
    with pytest.raises(ValueError, match="Exactly one"):
        ads_bib.run()

    with pytest.raises(ValueError, match="Exactly one"):
        ads_bib.run(preset="openrouter", config={})

    with pytest.raises(ValueError, match="Exactly one"):
        ads_bib.run(from_run="run_base", preset="openrouter")

    with pytest.raises(ValueError, match="Exactly one"):
        ads_bib.run(from_run="run_base", config={})


def test_run_rejects_query_override_conflict():
    with pytest.raises(ValueError, match="search.query"):
        ads_bib.run(
            preset="openrouter",
            query="author:test",
            overrides={"search.query": "author:other"},
        )

    with pytest.raises(ValueError, match="search.query"):
        ads_bib.run(
            from_run="run_base",
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


def test_detect_output_mode_uses_notebook_for_zmq_shell(monkeypatch):
    class ZMQInteractiveShell:
        pass

    monkeypatch.setattr(runner, "_safe_get_ipython", lambda: ZMQInteractiveShell())

    assert runner._detect_output_mode() == "notebook"


def test_detect_output_mode_uses_cli_for_terminal_shell(monkeypatch):
    class TerminalInteractiveShell:
        pass

    monkeypatch.setattr(runner, "_safe_get_ipython", lambda: TerminalInteractiveShell())

    assert runner._detect_output_mode() == "cli"


def test_pipeline_context_repr_is_compact_and_hides_secrets(tmp_path):
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "search": {"query": "q", "ads_token": "ads-secret"},
            "translate": {"api_key": "translate-secret"},
            "topic_model": {
                "embedding_api_key": "embedding-secret",
                "llm_api_key": "llm-secret",
            },
        }
    )
    ctx = pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)
    text = repr(ctx)
    config_text = repr(config)

    assert "PipelineContext(" in text
    assert "config=" not in text
    for secret in ("ads-secret", "translate-secret", "embedding-secret", "llm-secret"):
        assert secret not in text
        assert secret not in config_text


def test_load_run_config_missing_path_raises_plain_file_not_found():
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        runner.load_run_config(config=Path("missing.yaml"))


def test_resolve_runtime_path_uses_project_root(tmp_path):
    relative = pipeline._resolve_runtime_path(
        "data/models/lid.176.bin",
        project_root=tmp_path,
    )
    assert relative == str((tmp_path / "data" / "models" / "lid.176.bin").resolve())

    absolute_path = tmp_path / "custom" / "lid.176.bin"
    absolute = pipeline._resolve_runtime_path(absolute_path, project_root=tmp_path / "ignored")
    assert absolute == str(absolute_path.resolve())


def test_run_resolved_config_loads_source_input_as_initial_state(monkeypatch, tmp_path):
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "Year": 2020,
                "Author": ["Author A"],
                "Title": "Title",
                "Abstract": "Abstract",
                "References": ["r1", "missing"],
                "tokens": ["stale"],
                "embedding_2d_x": 1.0,
            }
        ]
    ).to_parquet(publications_path, index=False)
    pd.DataFrame(
        [{"Bibcode": "r1", "Year": 2019, "Author": ["Author R"], "Title": "Ref"}]
    ).to_parquet(references_path, index=False)
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path)},
            "source_input": {
                "publications_path": "publications.parquet",
                "references_path": "references.parquet",
            },
        }
    )
    calls: dict[str, object] = {}

    def _fake_run_pipeline(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs
        return "ctx"

    monkeypatch.setattr(runner, "run_pipeline", _fake_run_pipeline)

    result = runner.run_resolved_config(config, preflight=False)

    assert result == "ctx"
    state = calls["kwargs"]["initial_state"]
    assert state.publications.loc[0, "References"] == ["r1"]
    assert state.refs.loc[0, "Abstract"] == ""
    assert "tokens" not in state.publications.columns
    assert "embedding_2d_x" not in state.publications.columns
