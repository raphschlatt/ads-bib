from __future__ import annotations

from pathlib import Path

import ads_bib.bootstrap as bootstrap
import ads_bib.doctor as doctor
from ads_bib._utils.llama_server import LlamaServerRuntimeResolution
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
    assert "ads_bib_openrouter" in (tmp_path / "ads-bib.yaml").read_text(
        encoding="utf-8"
    )
    assert any("Downloaded fastText model" in line for line in lines)


def test_bootstrap_workspace_writes_dotenv_by_default(tmp_path):
    lines = bootstrap.bootstrap_workspace(project_root=tmp_path)

    assert (tmp_path / ".env").exists()
    assert any(str(tmp_path / ".env") in line for line in lines)


def test_ensure_default_fasttext_model_downloads_only_default_path(monkeypatch, tmp_path):
    calls: list[tuple[str, Path]] = []

    def _fake_download(url: str, destination: Path) -> None:
        calls.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"model")

    monkeypatch.setattr(bootstrap, "_download_file", _fake_download)

    resolved = bootstrap.ensure_default_fasttext_model(
        project_root=tmp_path,
        configured_path="data/models/lid.176.bin",
    )

    assert resolved == tmp_path / "data" / "models" / "lid.176.bin"
    assert calls == [
        (bootstrap.FASTTEXT_MODEL_URL, tmp_path / "data" / "models" / "lid.176.bin")
    ]


def test_ensure_default_fasttext_model_skips_custom_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap,
        "_download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not download")),
    )

    resolved = bootstrap.ensure_default_fasttext_model(
        project_root=tmp_path,
        configured_path="custom/lid.176.bin",
    )

    assert resolved is None


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
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        doctor,
        "_module_is_available",
        lambda module: module in {"openai", "pyarrow"},
    )

    report = doctor.collect_doctor_report(config, stop_stage="translate")
    failures = {check.name: check.detail for check in report.failing_checks()}
    warnings = {check.name: check.detail for check in report.checks if check.status == "warn"}

    assert "translate.fasttext_model" in warnings
    assert (
        "download the default model automatically"
        in warnings["translate.fasttext_model"]
    )
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
    monkeypatch.setattr(
        doctor,
        "inspect_llama_server_command",
        lambda *args, **kwargs: LlamaServerRuntimeResolution(
            command=None,
            source="managed_pending",
            detail="managed llama-server for windows-x86_64-cpu is not cached yet and will be auto-downloaded on ads-bib run",
            managed_root=tmp_path / "data" / "models" / "llama_cpp",
            platform_variant="windows-x86_64-cpu",
        ),
    )

    report = doctor.collect_doctor_report(
        config,
        start_stage="translate",
        stop_stage="translate",
    )
    failures = {check.name: check.detail for check in report.failing_checks()}
    warnings = {check.name: check.detail for check in report.checks if check.status == "warn"}

    assert "translate.llama_server.command" not in failures
    assert "translate.llama_server.command" in warnings
    assert "auto-downloaded on ads-bib run" in warnings["translate.llama_server.command"]
    assert "translate.model" not in failures


def test_collect_doctor_report_fails_for_missing_custom_llama_server_command(monkeypatch, tmp_path):
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
            "llama_server": {"command": "custom-llama-server"},
        }
    )

    monkeypatch.setattr(doctor, "_module_is_available", lambda module: True)
    monkeypatch.setattr(
        doctor,
        "inspect_llama_server_command",
        lambda *args, **kwargs: LlamaServerRuntimeResolution(
            command=None,
            source="missing",
            detail="llama-server command not found on PATH: 'custom-llama-server'",
        ),
    )

    report = doctor.collect_doctor_report(
        config,
        start_stage="translate",
        stop_stage="translate",
    )
    failures = {check.name: check.detail for check in report.failing_checks()}

    assert "translate.llama_server.command" in failures


def test_collect_doctor_report_flags_cpu_only_torch_for_official_local_gpu(monkeypatch, tmp_path):
    config_data = load_preset_config("local_gpu").to_dict()
    config_data["run"]["project_root"] = str(tmp_path)
    config_data["search"]["query"] = "author:test"
    config = PipelineConfig.from_dict(config_data)

    monkeypatch.setenv("ADS_TOKEN", "token")
    monkeypatch.setattr(doctor, "_module_is_available", lambda module: True)
    monkeypatch.setattr(
        doctor,
        "_inspect_torch_runtime",
        lambda: doctor.TorchRuntimeInfo(version="2.6.0+cpu", build="cpu", cuda_available=False),
    )

    report = doctor.collect_doctor_report(config, stop_stage="translate")
    failures = {check.name: check.detail for check in report.failing_checks()}
    ok_checks = {check.name: check.detail for check in report.checks if check.status == "ok"}

    assert "translate.provider.torch_runtime" in ok_checks
    assert "build=cpu" in ok_checks["translate.provider.torch_runtime"]
    assert "translate.provider.cuda_support" in failures


def test_collect_doctor_report_reports_expected_embedding_device(monkeypatch, tmp_path):
    config_data = load_preset_config("local_cpu").to_dict()
    config_data["run"]["project_root"] = str(tmp_path)
    config_data["search"]["query"] = "author:test"
    config = PipelineConfig.from_dict(config_data)

    monkeypatch.setenv("ADS_TOKEN", "token")
    monkeypatch.setattr(doctor, "_module_is_available", lambda module: True)
    monkeypatch.setattr(
        doctor,
        "_inspect_torch_runtime",
        lambda: doctor.TorchRuntimeInfo(version="2.6.0+cpu", build="cpu", cuda_available=False),
    )
    monkeypatch.setattr(
        doctor,
        "inspect_llama_server_command",
        lambda *args, **kwargs: LlamaServerRuntimeResolution(
            command="/tmp/managed/llama-server",
            source="managed_cached",
            detail="using cached managed llama-server runtime",
            managed_root=tmp_path / "data" / "models" / "llama_cpp",
            platform_variant="windows-x86_64-cpu",
        ),
    )

    report = doctor.collect_doctor_report(
        config,
        start_stage="embeddings",
        stop_stage="embeddings",
    )
    ok_checks = {check.name: check.detail for check in report.checks if check.status == "ok"}

    assert "topic_model.embedding_provider.torch_runtime" in ok_checks
    assert (
        ok_checks["topic_model.embedding_provider.expected_device"]
        == "local HF/Torch work is expected to run on cpu"
    )
