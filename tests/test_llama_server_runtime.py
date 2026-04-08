from __future__ import annotations

from io import StringIO
from pathlib import Path
import platform

import pytest

from ads_bib._utils import llama_server as runtime
from ads_bib._utils.model_specs import ModelSpec


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None) -> int:
        del timeout
        self.wait_calls += 1
        return 0


@pytest.fixture(autouse=True)
def _reset_server_registry():
    runtime.stop_all_llama_servers()
    yield
    runtime.stop_all_llama_servers()


def test_request_json_accepts_plain_text(monkeypatch):
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def read(self) -> bytes:
            return b"ok"

    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda request, timeout: _FakeResponse())

    assert runtime._request_json("http://localhost/health", payload=None, timeout=1.0) == {"text": "ok"}


def test_resolve_llama_server_command_accepts_existing_path(tmp_path):
    command = tmp_path / "llama-server"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    assert runtime.resolve_llama_server_command(str(command)) == str(command.resolve())


def test_ensure_llama_server_reuses_process_and_auto_port(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")
    spec = ModelSpec(model_path=str(model_file))
    config = runtime.LlamaServerConfig(port=None)
    log_path = tmp_path / "runtime.log"

    calls: dict[str, object] = {"spawned": 0}
    fake_process = _FakeProcess()
    fake_log_handle = StringIO()

    monkeypatch.setattr(
        runtime,
        "prepare_llama_server_runtime",
        lambda **kwargs: runtime.LlamaServerCommandResolution(
            command="/usr/bin/llama-server",
            source="path",
            detail="resolved llama-server from PATH",
        ),
    )
    monkeypatch.setattr(runtime, "_find_free_port", lambda host: 18080)
    monkeypatch.setattr(
        runtime,
        "_spawn_llama_server",
        lambda **kwargs: (
            calls.__setitem__("spawned", int(calls["spawned"]) + 1) or fake_process,
            fake_log_handle,
        ),
    )
    monkeypatch.setattr(runtime, "_wait_for_server_ready", lambda **kwargs: None)

    first = runtime.ensure_llama_server(
        model_spec=spec,
        config=config,
        runtime_log_path=log_path,
    )
    second = runtime.ensure_llama_server(
        model_spec=spec,
        config=config,
        runtime_log_path=log_path,
    )

    assert first is second
    assert first.base_url == "http://127.0.0.1:18080/v1"
    assert calls["spawned"] == 1


def test_ensure_llama_server_cleans_up_on_readiness_failure(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")
    spec = ModelSpec(model_path=str(model_file))
    config = runtime.LlamaServerConfig()
    fake_process = _FakeProcess()
    fake_log_handle = StringIO()

    monkeypatch.setattr(
        runtime,
        "prepare_llama_server_runtime",
        lambda **kwargs: runtime.LlamaServerCommandResolution(
            command="/usr/bin/llama-server",
            source="path",
            detail="resolved llama-server from PATH",
        ),
    )
    monkeypatch.setattr(runtime, "_find_free_port", lambda host: 18081)
    monkeypatch.setattr(runtime, "_spawn_llama_server", lambda **kwargs: (fake_process, fake_log_handle))
    monkeypatch.setattr(
        runtime,
        "_wait_for_server_ready",
        lambda **kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
    )

    with pytest.raises(TimeoutError, match="not ready"):
        runtime.ensure_llama_server(
            model_spec=spec,
            config=config,
            runtime_log_path=tmp_path / "runtime.log",
        )

    assert fake_process.terminated is True
    assert fake_log_handle.closed is True
    assert runtime._SERVER_REGISTRY == {}


def test_ensure_llama_server_prefers_path_offload_and_falls_back_to_cpu(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")
    spec = ModelSpec(model_path=str(model_file))
    config = runtime.LlamaServerConfig(gpu_layers=0)
    spawned_layers: list[int] = []
    fake_processes: list[_FakeProcess] = []
    fake_logs: list[StringIO] = []

    monkeypatch.setattr(
        runtime,
        "prepare_llama_server_runtime",
        lambda **kwargs: runtime.LlamaServerCommandResolution(
            command="/usr/bin/llama-server",
            source="path",
            detail="resolved llama-server from PATH",
        ),
    )
    monkeypatch.setattr(runtime, "_find_free_port", lambda host: 18082)

    def _fake_spawn(**kwargs):
        spawned_layers.append(kwargs["config"].gpu_layers)
        process = _FakeProcess()
        log_handle = StringIO()
        fake_processes.append(process)
        fake_logs.append(log_handle)
        return process, log_handle

    def _fake_wait(**kwargs):
        if kwargs["process"] is fake_processes[0]:
            raise RuntimeError("offload probe failed")
        return None

    monkeypatch.setattr(runtime, "_spawn_llama_server", _fake_spawn)
    monkeypatch.setattr(runtime, "_wait_for_server_ready", _fake_wait)

    handle = runtime.ensure_llama_server(
        model_spec=spec,
        config=config,
        runtime_log_path=tmp_path / "runtime.log",
    )

    assert handle.base_url == "http://127.0.0.1:18082/v1"
    assert spawned_layers == [-1, 0]
    assert fake_processes[0].terminated is True
    assert fake_logs[0].closed is True
    assert fake_processes[1].poll() is None
    assert fake_logs[1].closed is False


def test_ensure_llama_server_keeps_managed_cpu_runtime_without_path_probe(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")
    spec = ModelSpec(model_path=str(model_file))
    config = runtime.LlamaServerConfig(gpu_layers=0)
    spawned_layers: list[int] = []

    monkeypatch.setattr(
        runtime,
        "prepare_llama_server_runtime",
        lambda **kwargs: runtime.LlamaServerCommandResolution(
            command=str(tmp_path / "managed" / "llama-server"),
            source="managed_downloaded",
            detail="downloaded managed llama-server runtime",
        ),
    )
    monkeypatch.setattr(runtime, "_find_free_port", lambda host: 18083)
    monkeypatch.setattr(
        runtime,
        "_spawn_llama_server",
        lambda **kwargs: (spawned_layers.append(kwargs["config"].gpu_layers) or _FakeProcess(), StringIO()),
    )
    monkeypatch.setattr(runtime, "_wait_for_server_ready", lambda **kwargs: None)

    handle = runtime.ensure_llama_server(
        model_spec=spec,
        config=config,
        runtime_log_path=tmp_path / "runtime.log",
    )

    assert handle.base_url == "http://127.0.0.1:18083/v1"
    assert spawned_layers == [0]


def test_wait_for_server_ready_reports_qwen35_architecture_mismatch(tmp_path):
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "llama_model_load: error loading model architecture: unknown model architecture: 'qwen35'\n",
        encoding="utf-8",
    )
    process = _FakeProcess()
    process.returncode = 1
    handle = runtime.LlamaServerHandle(host="127.0.0.1", port=8080, model_label="qwen35")

    with pytest.raises(RuntimeError, match="too old or incompatible for Qwen3.5"):
        runtime._wait_for_server_ready(
            handle=handle,
            timeout_s=1.0,
            process=process,
            command="/usr/bin/llama-server",
            runtime_log_path=log_path,
        )


def test_wait_for_server_ready_includes_command_and_runtime_log_on_generic_failure(tmp_path):
    log_path = tmp_path / "runtime.log"
    log_path.write_text("generic failure\n", encoding="utf-8")
    process = _FakeProcess()
    process.returncode = 1
    handle = runtime.LlamaServerHandle(host="127.0.0.1", port=8080, model_label="generic")

    with pytest.raises(RuntimeError, match="/usr/bin/llama-server"):
        runtime._wait_for_server_ready(
            handle=handle,
            timeout_s=1.0,
            process=process,
            command="/usr/bin/llama-server",
            runtime_log_path=log_path,
        )


def test_spawn_llama_server_passes_reasoning_off_by_default(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")
    log_path = tmp_path / "runtime.log"
    calls: dict[str, object] = {}

    class _FakePopen:
        def __init__(self, args, **kwargs):
            calls["args"] = list(args)
            calls["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr(runtime.subprocess, "Popen", _FakePopen)

    process, log_handle = runtime._spawn_llama_server(
        command="/usr/bin/llama-server",
        model_path=str(model_file),
        config=runtime.LlamaServerConfig(),
        port=18080,
        runtime_log_path=log_path,
    )

    assert "--reasoning" in calls["args"]
    idx = calls["args"].index("--reasoning")
    assert calls["args"][idx + 1] == "off"
    assert str(model_file) in calls["args"]
    assert log_handle is not None
    log_handle.close()


def test_inspect_llama_server_runtime_uses_managed_cache(tmp_path, monkeypatch):
    asset = runtime._select_managed_llama_asset(gpu_layers=0)
    assert asset is not None
    binary_name = "llama-server.exe" if platform.system().strip().lower() == "windows" else "llama-server"
    command = (
        tmp_path
        / "data"
        / "models"
        / "llama_cpp"
        / runtime.MANAGED_LLAMA_CPP_RELEASE_TAG
        / asset.platform_variant
        / "bin"
        / binary_name
    )
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)

    resolution = runtime.inspect_llama_server_runtime(
        config=runtime.LlamaServerConfig(gpu_layers=0),
        project_root=tmp_path,
        allow_download=False,
    )

    assert resolution.source == "managed_cached"
    assert resolution.command == str(command.resolve())
    assert resolution.managed_root == command.parents[1]


def test_inspect_llama_server_runtime_reports_planned_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)

    resolution = runtime.inspect_llama_server_runtime(
        config=runtime.LlamaServerConfig(gpu_layers=0),
        project_root=tmp_path,
        allow_download=False,
    )

    assert resolution.source == "managed_pending"
    assert resolution.platform_variant is not None
    assert "auto-downloaded on ads-bib run" in str(resolution.detail)


def test_inspect_llama_server_runtime_downloads_managed_binary(tmp_path, monkeypatch):
    calls: dict[str, object] = {}
    downloaded = tmp_path / "runtime" / "llama-server"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)

    def _fake_ensure(*, asset, managed_root):
        calls["asset"] = asset.platform_variant
        calls["managed_root"] = managed_root
        return downloaded

    monkeypatch.setattr(runtime, "_ensure_managed_llama_server_binary", _fake_ensure)

    resolution = runtime.inspect_llama_server_runtime(
        config=runtime.LlamaServerConfig(gpu_layers=0),
        project_root=tmp_path,
        allow_download=True,
    )

    assert resolution.source == "managed_downloaded"
    assert resolution.command == str(downloaded)
    assert calls["managed_root"] == (
        tmp_path
        / "data"
        / "models"
        / "llama_cpp"
        / runtime.MANAGED_LLAMA_CPP_RELEASE_TAG
        / str(calls["asset"])
    )


def test_inspect_llama_server_runtime_reports_unsupported_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(runtime, "_select_managed_llama_asset", lambda **kwargs: None)

    resolution = runtime.inspect_llama_server_runtime(
        config=runtime.LlamaServerConfig(gpu_layers=-1),
        project_root=tmp_path,
        allow_download=False,
    )

    assert resolution.source == "unsupported"
    assert resolution.command is None
    assert "package-managed llama-server runtime is unavailable" in str(resolution.detail)
