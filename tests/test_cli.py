from __future__ import annotations

import sys

import ads_bib.cli as cli
import ads_bib.doctor as doctor
import ads_bib.runner as runner
import pandas as pd
import pytest
import yaml


def _passing_report(*, stages: tuple[str, ...] = ("search",)) -> doctor.DoctorReport:
    return doctor.DoctorReport(
        checks=(doctor.DoctorCheck(name="search.query", status="ok", detail="configured"),),
        active_stages=stages,
    )


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

    def _fake_run_resolved_config(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_resolved_config", _fake_run_resolved_config)
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
    assert callable(calls["kwargs"].pop("notify"))
    assert calls["kwargs"] == {
        "start_stage": "translate",
        "stop_stage": "citations",
        "run_name": "cli-test",
        "output_mode": "cli",
    }


def test_main_dispatches_run_with_preset(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_run_resolved_config(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_resolved_config", _fake_run_resolved_config)
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
    assert callable(calls["kwargs"].pop("notify"))
    assert calls["kwargs"] == {
        "start_stage": None,
        "stop_stage": None,
        "run_name": None,
        "output_mode": "cli",
    }


def _write_base_variant_run(tmp_path):
    run_dir = tmp_path / "runs" / "run_20260101_010101_base"
    run_dir.mkdir(parents=True)
    data_dir = run_dir / "data" / "dataset"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "full_text": "alpha beta",
                "tokens": [["alpha", "beta"]],
                "author_uids": [["u1"]],
            }
        ]
    ).to_parquet(data_dir / "publications.parquet")
    pd.DataFrame([{"Bibcode": "r1", "author_uids": [["u2"]]}]).to_parquet(
        data_dir / "references.parquet"
    )
    config = {
        "run": {"run_name": "base"},
        "search": {"query": "author:test", "ads_token": "token"},
        "translate": {"enabled": False, "fasttext_model": "data/models/lid.176.bin"},
        "topic_model": {
            "backend": "bertopic",
            "embedding_provider": "openrouter",
            "embedding_model": "qwen/qwen3-embedding-8b",
            "llm_provider": "openrouter",
            "llm_model": "google/gemini-3-flash-preview",
        },
    }
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "run_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "artifact_layout_version": 2,
                "run": {"run_id": "run_20260101_010101_base"},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_main_dispatches_run_from_run(monkeypatch, tmp_path):
    run_dir = _write_base_variant_run(tmp_path)
    calls: dict[str, object] = {}

    def _fake_run_resolved_config(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_resolved_config", _fake_run_resolved_config)

    rc = cli.main(
        [
            "run",
            "--from-run",
            str(run_dir),
            "--set",
            "topic_model.embedding_model=google/gemini-embedding-001",
            "--run-name",
            "embedding-swap",
        ]
    )

    assert rc == 0
    assert calls["config"].topic_model.embedding_model == "google/gemini-embedding-001"
    assert callable(calls["kwargs"].pop("notify"))
    assert calls["kwargs"]["start_stage"] == "embeddings"
    assert calls["kwargs"]["stop_stage"] is None
    assert calls["kwargs"]["run_name"] == "embedding-swap"
    assert calls["kwargs"]["output_mode"] == "cli"
    assert calls["kwargs"]["variant"]["base_run_id"] == "run_20260101_010101_base"
    assert calls["kwargs"]["variant"]["changed_keys"] == ["topic_model.embedding_model"]


def test_main_run_from_run_dry_run_prints_plan_and_creates_no_run(tmp_path, monkeypatch, capsys):
    run_dir = _write_base_variant_run(tmp_path)
    monkeypatch.setattr(
        cli,
        "run_resolved_config",
        lambda *args, **kwargs: pytest.fail("dry-run must not execute the pipeline"),
    )

    rc = cli.main(
        [
            "run",
            "--from-run",
            str(run_dir),
            "--set",
            "topic_model.embedding_model=google/gemini-embedding-001",
            "--dry-run",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "Run variant dry run" in output
    assert "Changed keys: topic_model.embedding_model" in output
    assert "Recomputed from: embeddings" in output
    assert sorted(path.name for path in (tmp_path / "runs").iterdir()) == [run_dir.name]


def test_main_run_from_run_explicit_from_overrides_auto_start(monkeypatch, tmp_path):
    run_dir = _write_base_variant_run(tmp_path)
    calls: dict[str, object] = {}

    def _fake_run_resolved_config(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_resolved_config", _fake_run_resolved_config)

    rc = cli.main(
        [
            "run",
            "--from-run",
            str(run_dir),
            "--from",
            "topic_fit",
            "--set",
            "topic_model.embedding_model=google/gemini-embedding-001",
        ]
    )

    assert rc == 0
    assert calls["kwargs"]["start_stage"] == "topic_fit"


def test_main_run_requires_search_query(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  run_name: test\nsearch:\n  query: ''\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ensure_default_fasttext_model", lambda **kwargs: None)
    monkeypatch.setattr(
        runner,
        "collect_doctor_report",
        lambda *args, **kwargs: doctor.DoctorReport(
            checks=(doctor.DoctorCheck(name="search.query", status="fail", detail="missing"),),
            active_stages=("search",),
        ),
    )

    assert cli.main(["run", "--config", str(config_path)]) == 1


def test_main_run_returns_nonzero_with_preflight_report(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  run_name: test\nsearch:\n  query: q\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_pipeline",
        lambda *args, **kwargs: pytest.fail("run_pipeline should not be called"),
    )
    monkeypatch.setattr(runner, "ensure_default_fasttext_model", lambda **kwargs: None)
    monkeypatch.setattr(
        runner,
        "collect_doctor_report",
        lambda *args, **kwargs: doctor.DoctorReport(
            checks=(
                doctor.DoctorCheck(name="search.query", status="ok", detail="configured"),
                doctor.DoctorCheck(
                    name="search.ads_token",
                    status="fail",
                    detail="missing ADS token",
                ),
            ),
            active_stages=("search", "translate"),
        ),
    )

    rc = cli.main(["run", "--config", str(config_path)])

    assert rc == 1
    assert "Run blocked by preflight checks" in capsys.readouterr().err


def test_main_run_reports_prepared_fasttext(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "run:\n"
            "  run_name: test\n"
            "search:\n"
            "  query: q\n"
            "translate:\n"
            "  enabled: true\n"
            "  fasttext_model: data/models/lid.176.bin\n"
        ),
        encoding="utf-8",
    )
    prepared_path = tmp_path / "data" / "models" / "lid.176.bin"
    monkeypatch.setattr(runner, "ensure_default_fasttext_model", lambda **kwargs: prepared_path)
    monkeypatch.setattr(runner, "collect_doctor_report", lambda *args, **kwargs: _passing_report())
    monkeypatch.setattr(runner, "run_pipeline", lambda *args, **kwargs: None)

    rc = cli.main(["run", "--config", str(config_path)])

    assert rc == 0
    assert f"Prepared translate.fasttext_model at {prepared_path}" in capsys.readouterr().out


def test_main_run_reports_prepared_managed_llama_server(monkeypatch, capsys):
    monkeypatch.setattr(runner, "ensure_default_fasttext_model", lambda **kwargs: None)
    monkeypatch.setattr(
        runner,
        "collect_doctor_report",
        lambda *args, **kwargs: _passing_report(stages=("topic_fit",)),
    )
    monkeypatch.setattr(
        runner,
        "prepare_llama_server_runtime",
        lambda **kwargs: type(
            "_Runtime",
            (),
            {
                "source": "managed_downloaded",
                "command": "/tmp/managed/llama-server",
                "detail": "downloaded managed runtime",
            },
        )(),
    )
    monkeypatch.setattr(runner, "run_pipeline", lambda *args, **kwargs: None)

    rc = cli.main(
        [
            "run",
            "--preset",
            "local_cpu",
            "--set",
            "search.query=author:test",
        ]
    )

    assert rc == 0
    assert "Prepared llama_server.command at /tmp/managed/llama-server" in capsys.readouterr().out


def test_main_run_blocks_when_managed_llama_server_prepare_fails(monkeypatch, capsys):
    monkeypatch.setattr(runner, "ensure_default_fasttext_model", lambda **kwargs: None)
    monkeypatch.setattr(
        runner,
        "collect_doctor_report",
        lambda *args, **kwargs: _passing_report(stages=("topic_fit",)),
    )
    monkeypatch.setattr(
        runner,
        "prepare_llama_server_runtime",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("managed runtime failed")),
    )
    monkeypatch.setattr(
        runner,
        "run_pipeline",
        lambda *args, **kwargs: pytest.fail("run_pipeline should not be called"),
    )

    rc = cli.main(
        [
            "run",
            "--preset",
            "local_cpu",
            "--set",
            "search.query=author:test",
        ]
    )

    assert rc == 1
    assert "Run blocked while preparing llama_server.command" in capsys.readouterr().err


def test_main_run_missing_config_path_keeps_normal_file_not_found():
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        cli.main(["run", "--config", "missing.yaml"])


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


def test_main_doctor_reports_config_validation_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "collect_doctor_report",
        lambda *args, **kwargs: pytest.fail("doctor should not run with invalid config"),
    )

    rc = cli.main(
        [
            "doctor",
            "--preset",
            "openrouter",
            "--set",
            "search.query=author:test",
            "--set",
            "curation.clusters_to_remove=7",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "Config error: curation.clusters_to_remove must be a list of integer cluster IDs; "
        "use [7] for one cluster."
    ) in captured.err


def test_parse_override_requires_equals():
    try:
        cli._parse_override("invalid")
    except ValueError as exc:
        assert "key=value" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_parse_override_reads_clusters_to_remove_as_list():
    key, value = cli._parse_override("curation.clusters_to_remove=[7, 12]")

    assert key == "curation.clusters_to_remove"
    assert value == [7, 12]


def test_parse_override_reads_layered_clusters_to_remove_as_list_of_mappings():
    key, value = cli._parse_override(
        "curation.layered_clusters_to_remove=[{layer: 0, cluster_id: 12}, {layer: 1, cluster_id: 20}]"
    )

    assert key == "curation.layered_clusters_to_remove"
    assert value == [{"layer": 0, "cluster_id": 12}, {"layer": 1, "cluster_id": 20}]


def test_parse_override_still_reads_legacy_cluster_targets():
    key, value = cli._parse_override(
        "curation.cluster_targets=[{layer: 0, cluster_id: 12}]"
    )

    assert key == "curation.cluster_targets"
    assert value == [{"layer": 0, "cluster_id": 12}]
