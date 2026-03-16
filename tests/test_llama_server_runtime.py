from __future__ import annotations

from io import StringIO
from pathlib import Path

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

    monkeypatch.setattr(runtime, "resolve_llama_server_command", lambda command: "/usr/bin/llama-server")
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

    monkeypatch.setattr(runtime, "resolve_llama_server_command", lambda command: "/usr/bin/llama-server")
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


def test_wait_for_server_ready_reports_qwen35_architecture_mismatch(tmp_path):
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "llama_model_load: error loading model architecture: unknown model architecture: 'qwen35'\n",
        encoding="utf-8",
    )
    process = _FakeProcess()
    process.returncode = 1
    handle = runtime.LlamaServerHandle(host="127.0.0.1", port=8080, model_label="qwen35")

    with pytest.raises(RuntimeError, match="too old for Qwen3.5"):
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
