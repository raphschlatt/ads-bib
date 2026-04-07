from __future__ import annotations

from pathlib import Path

import ads_bib.bootstrap as bootstrap
import ads_bib.doctor as doctor
from ads_bib.pipeline import PipelineConfig
from ads_bib.presets import load_preset_config


def test_bootstrap_workspace_creates_env_config_and_fasttext(monkeypatch, tmp_path):
    def _fake_download(url: str, destination: Path) -> None:
        assert url == bootstrap.FASTTEXT_MODEL_URL
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"model")

    monkeypatch.setattr(bootstrap, "_download_file", _fake_download)

    lines = bootstrap.bootstrap_workspace(
        project_root=tmp_path,
        preset_name="openrouter",
        config_output="ads-bib.yaml",
        env_file=".env",
        download_fasttext=True,
    )

    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "cache").is_dir()
    assert (tmp_path / "data" / "models" / "lid.176.bin").read_bytes() == b"model"
    assert (tmp_path / "runs").is_dir()
    assert "ADS_TOKEN=" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ads_bib_openrouter" in (tmp_path / "ads-bib.yaml").read_text(encoding="utf-8")
    assert any("Downloaded fastText model" in line for line in lines)


def test_bootstrap_workspace_keeps_existing_files_without_force(tmp_path):
    env_path = tmp_path / ".env"
    config_path = tmp_path / "ads-bib.yaml"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("CUSTOM=1\n", encoding="utf-8")
    config_path.write_text("custom: true\n", encoding="utf-8")

    lines = bootstrap.bootstrap_workspace(
        project_root=tmp_path,
        preset_name="openrouter",
        config_output="ads-bib.yaml",
        env_file=".env",
    )

    assert env_path.read_text(encoding="utf-8") == "CUSTOM=1\n"
    assert config_path.read_text(encoding="utf-8") == "custom: true\n"
    assert any("Kept existing" in line for line in lines)


def test_collect_doctor_report_is_stage_aware(monkeypatch, tmp_path):
    config_data = load_preset_config("openrouter").to_dict()
    config_data["run"]["project_root"] = str(tmp_path)
    config_data["search"]["query"] = "author:test"
    config = PipelineConfig.from_dict(config_data)

    monkeypatch.setenv("ADS_TOKEN", "token")
    monkeypatch.setattr(doctor, "_module_is_available", lambda module: False)

    report = doctor.collect_doctor_report(config, stop_stage="search")

    assert report.active_stages == ("search",)
    assert report.fail_count == 0


def test_collect_doctor_report_flags_translate_blockers(monkeypatch, tmp_path):
    config_data = load_preset_config("openrouter").to_dict()
    config_data["run"]["project_root"] = str(tmp_path)
    config_data["search"]["query"] = "author:test"
    config = PipelineConfig.from_dict(config_data)

    monkeypatch.setenv("ADS_TOKEN", "token")
    monkeypatch.setattr(doctor, "_module_is_available", lambda module: module == "openai")

    report = doctor.collect_doctor_report(config, stop_stage="translate")
    failures = {check.name: check.detail for check in report.failing_checks()}

    assert "translate.fasttext_model" in failures
    assert "translate.api_key" in failures


def test_collect_doctor_report_warns_for_missing_spacy_with_auto_download(monkeypatch, tmp_path):
    config = PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path), "start_stage": "tokenize"},
            "tokenize": {
                "enabled": True,
                "spacy_model": "en_core_web_md",
                "fallback_model": "en_core_web_md",
                "auto_download": True,
            },
        }
    )

    monkeypatch.setattr(doctor, "_module_is_available", lambda module: True)
    monkeypatch.setattr(doctor, "_spacy_model_installed", lambda model_name: False)

    report = doctor.collect_doctor_report(
        config,
        start_stage="tokenize",
        stop_stage="tokenize",
    )

    assert report.fail_count == 0
    assert report.warn_count == 1
    assert any(
        check.name == "tokenize.spacy_model" and check.status == "warn"
        for check in report.checks
    )


def test_collect_doctor_report_checks_llama_server_runtime(monkeypatch, tmp_path):
    fasttext_path = tmp_path / "data" / "models" / "lid.176.bin"
    gguf_path = tmp_path / "models" / "model.gguf"
    fasttext_path.parent.mkdir(parents=True, exist_ok=True)
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    fasttext_path.write_bytes(b"model")
    gguf_path.write_bytes(b"gguf")

    config = PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path), "start_stage": "translate"},
            "translate": {
                "provider": "llama_server",
                "model_path": str(gguf_path),
                "fasttext_model": "data/models/lid.176.bin",
            },
        }
    )

    monkeypatch.setattr(doctor, "_module_is_available", lambda module: True)

    def _raise_missing(command: str) -> str:
        raise FileNotFoundError(f"llama-server command not found: {command!r}")

    monkeypatch.setattr(doctor, "resolve_llama_server_command", _raise_missing)

    report = doctor.collect_doctor_report(
        config,
        start_stage="translate",
        stop_stage="translate",
    )
    failures = {check.name: check.detail for check in report.failing_checks()}

    assert "translate.llama_server.command" in failures
    assert "translate.model" not in failures