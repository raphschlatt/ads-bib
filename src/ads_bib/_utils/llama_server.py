"""Managed local llama-server runtime for GGUF chat generation."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from io import TextIOBase
import json
import logging
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any
import urllib.request

from ads_bib._utils.model_specs import ModelSpec

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_SERVER_HOST = "127.0.0.1"
DEFAULT_LLAMA_SERVER_COMMAND = "llama-server"
DEFAULT_LLAMA_SERVER_CTX_SIZE = 4096
DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class LlamaServerConfig:
    """Runtime settings shared by local llama-server call sites."""

    command: str = DEFAULT_LLAMA_SERVER_COMMAND
    host: str = DEFAULT_LLAMA_SERVER_HOST
    port: int | None = None
    threads: int | None = None
    ctx_size: int = DEFAULT_LLAMA_SERVER_CTX_SIZE
    gpu_layers: int = -1
    startup_timeout_s: float = DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_S
    reasoning: str = "off"

    def normalized(self) -> LlamaServerConfig:
        if self.port is not None and int(self.port) <= 0:
            raise ValueError("llama_server.port must be > 0 when provided.")
        if self.threads is not None and int(self.threads) <= 0:
            raise ValueError("llama_server.threads must be > 0 when provided.")
        if int(self.ctx_size) <= 0:
            raise ValueError("llama_server.ctx_size must be > 0.")
        if float(self.startup_timeout_s) <= 0:
            raise ValueError("llama_server.startup_timeout_s must be > 0.")
        reasoning_norm = str(self.reasoning).strip().lower() or "off"
        if reasoning_norm not in ("off", "on"):
            raise ValueError(f"llama_server.reasoning must be 'off' or 'on', got {self.reasoning!r}.")
        return LlamaServerConfig(
            command=str(self.command).strip() or DEFAULT_LLAMA_SERVER_COMMAND,
            host=str(self.host).strip() or DEFAULT_LLAMA_SERVER_HOST,
            port=None if self.port is None else int(self.port),
            threads=None if self.threads is None else int(self.threads),
            ctx_size=int(self.ctx_size),
            gpu_layers=int(self.gpu_layers),
            startup_timeout_s=float(self.startup_timeout_s),
            reasoning=reasoning_norm,
        )


@dataclass(frozen=True)
class LlamaServerHandle:
    """Resolved local server endpoint and display metadata."""

    host: str
    port: int
    model_label: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class _ManagedServer:
    process: subprocess.Popen[str]
    handle: LlamaServerHandle
    runtime_log_path: Path | None
    log_handle: TextIOBase | None = None


_SERVER_REGISTRY: dict[tuple[str, str, str | None], _ManagedServer] = {}


def ensure_llama_server(
    *,
    model_spec: ModelSpec,
    config: LlamaServerConfig,
    runtime_log_path: Path | None,
) -> LlamaServerHandle:
    """Start or reuse one llama-server process for the given model/runtime."""
    normalized = config.normalized()
    model_path = model_spec.resolve()
    registry_key = (model_path, _runtime_key(normalized), _log_key(runtime_log_path))
    existing = _SERVER_REGISTRY.get(registry_key)
    if existing is not None and existing.process.poll() is None:
        return existing.handle
    if existing is not None:
        _SERVER_REGISTRY.pop(registry_key, None)

    resolved_command = resolve_llama_server_command(normalized.command)
    port = normalized.port or _find_free_port(normalized.host)
    handle = LlamaServerHandle(
        host=normalized.host,
        port=port,
        model_label=model_spec.display_name(),
    )
    logger.info(
        "Starting llama-server for %s on %s (command: %s)",
        handle.model_label,
        handle.base_url,
        resolved_command,
    )
    process, log_handle = _spawn_llama_server(
        command=resolved_command,
        model_path=model_path,
        config=normalized,
        port=port,
        runtime_log_path=runtime_log_path,
    )
    managed = _ManagedServer(
        process=process,
        handle=handle,
        runtime_log_path=runtime_log_path,
        log_handle=log_handle,
    )
    _SERVER_REGISTRY[registry_key] = managed
    try:
        _wait_for_server_ready(
            handle=handle,
            timeout_s=normalized.startup_timeout_s,
            process=process,
            command=resolved_command,
            runtime_log_path=runtime_log_path,
        )
    except Exception:
        stop_llama_server(handle=handle, model_path=model_path, config=normalized, runtime_log_path=runtime_log_path)
        raise
    return handle


def stop_llama_server(
    *,
    handle: LlamaServerHandle,
    model_path: str,
    config: LlamaServerConfig,
    runtime_log_path: Path | None,
) -> None:
    """Stop one managed server if present."""
    registry_key = (model_path, _runtime_key(config.normalized()), _log_key(runtime_log_path))
    managed = _SERVER_REGISTRY.pop(registry_key, None)
    if managed is None:
        return
    process = managed.process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
    if managed.log_handle is not None:
        managed.log_handle.close()


def stop_all_llama_servers() -> None:
    """Terminate all managed server processes."""
    for managed in list(_SERVER_REGISTRY.values()):
        process = managed.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        if managed.log_handle is not None:
            managed.log_handle.close()
    _SERVER_REGISTRY.clear()


def resolve_llama_server_command(command: str) -> str:
    """Resolve a llama-server executable from PATH or direct path."""
    raw = str(command).strip()
    if not raw:
        raw = DEFAULT_LLAMA_SERVER_COMMAND
    direct = Path(raw).expanduser()
    if direct.exists():
        return str(direct.resolve())
    resolved = shutil.which(raw)
    if resolved:
        return resolved
    raise FileNotFoundError(f"llama-server command not found: {command!r}")


def build_openai_client(*, handle: LlamaServerHandle):
    """Create an OpenAI-compatible client for one managed server."""
    from openai import OpenAI

    return OpenAI(api_key="local", base_url=handle.base_url)


def probe_server_ready(handle: LlamaServerHandle) -> bool:
    """Return whether the local server responds to readiness probes."""
    urls = [
        (f"http://{handle.host}:{handle.port}/health", None),
        (
            f"http://{handle.host}:{handle.port}/completion",
            {"prompt": "ping", "n_predict": 1, "temperature": 0.0},
        ),
    ]
    for url, payload in urls:
        try:
            _request_json(url, payload=payload, timeout=5.0)
            return True
        except Exception:
            continue
    return False


def _wait_for_server_ready(
    *,
    handle: LlamaServerHandle,
    timeout_s: float,
    process: subprocess.Popen[str],
    command: str,
    runtime_log_path: Path | None,
) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                _format_startup_failure(
                    command=command,
                    runtime_log_path=runtime_log_path,
                    timed_out=False,
                )
            )
        if probe_server_ready(handle):
            return
        time.sleep(1.0)
    raise TimeoutError(
        _format_startup_failure(
            command=command,
            runtime_log_path=runtime_log_path,
            timed_out=True,
        )
    )


def _spawn_llama_server(
    *,
    command: str,
    model_path: str,
    config: LlamaServerConfig,
    port: int,
    runtime_log_path: Path | None,
) -> tuple[subprocess.Popen[str], TextIOBase | None]:
    args = [
        command,
        "-m",
        model_path,
        "--host",
        config.host,
        "--port",
        str(port),
        "--ctx-size",
        str(config.ctx_size),
        "-ngl",
        str(config.gpu_layers),
    ]
    if config.threads is not None:
        args.extend(["--threads", str(config.threads)])
    if config.reasoning:
        args.extend(["--reasoning", config.reasoning])

    stdout_target: Any
    stderr_target: Any
    log_handle = None
    if runtime_log_path is not None:
        runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = runtime_log_path.open("a", encoding="utf-8")
        stdout_target = log_handle
        stderr_target = log_handle
    else:
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL

    try:
        process = subprocess.Popen(
            args,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        return process, log_handle
    except Exception:
        if log_handle is not None:
            log_handle.close()
        raise


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _runtime_key(config: LlamaServerConfig) -> str:
    return json.dumps(
        {
            "command": config.command,
            "host": config.host,
            "port": config.port,
            "threads": config.threads,
            "ctx_size": config.ctx_size,
            "gpu_layers": config.gpu_layers,
            "reasoning": config.reasoning,
        },
        sort_keys=True,
    )


def _log_key(runtime_log_path: Path | None) -> str | None:
    if runtime_log_path is None:
        return None
    return str(runtime_log_path.resolve())


def _format_startup_failure(
    *,
    command: str,
    runtime_log_path: Path | None,
    timed_out: bool,
) -> str:
    log_tail = _read_runtime_log_tail(runtime_log_path)
    if "unknown model architecture: 'qwen35'" in log_tail:
        return (
            f"llama-server '{command}' is too old for Qwen3.5 "
            "(unknown model architecture: 'qwen35'). "
            f"{_runtime_log_hint(runtime_log_path)}"
        )
    if timed_out:
        return (
            f"llama-server '{command}' did not become ready in time. "
            f"{_runtime_log_hint(runtime_log_path)}"
        )
    return (
        f"llama-server '{command}' exited before becoming ready. "
        f"{_runtime_log_hint(runtime_log_path)}"
    )


def _read_runtime_log_tail(runtime_log_path: Path | None, max_chars: int = 8192) -> str:
    if runtime_log_path is None:
        return ""
    try:
        text = runtime_log_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _runtime_log_hint(runtime_log_path: Path | None) -> str:
    if runtime_log_path is None:
        return "See runtime.log for details."
    return f"See runtime log: {runtime_log_path.resolve()}."


def _request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw}


atexit.register(stop_all_llama_servers)
