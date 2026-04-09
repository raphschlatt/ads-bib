"""Managed local llama-server runtime for GGUF chat generation."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, replace
from io import TextIOBase
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
from typing import Any, Literal
import urllib.request
import zipfile

from ads_bib._utils.model_specs import ModelSpec

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_SERVER_HOST = "127.0.0.1"
DEFAULT_LLAMA_SERVER_COMMAND = "llama-server"
DEFAULT_LLAMA_SERVER_CTX_SIZE = 4096
DEFAULT_LLAMA_SERVER_PARALLEL = 8
DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_S = 120.0
MANAGED_LLAMA_CPP_RELEASE_TAG = "b8705"
MANAGED_LLAMA_CPP_RELEASE_BASE_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{MANAGED_LLAMA_CPP_RELEASE_TAG}"
)
MANAGED_LLAMA_CPP_RELATIVE_ROOT = Path("data/models/llama_cpp")


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


@dataclass(frozen=True)
class ManagedLlamaAsset:
    platform_variant: str
    asset_name: str
    archive_format: Literal["zip", "tar.gz"]
    note: str | None = None

    @property
    def download_url(self) -> str:
        return f"{MANAGED_LLAMA_CPP_RELEASE_BASE_URL}/{self.asset_name}"


@dataclass(frozen=True)
class LlamaServerCommandResolution:
    """Inspection result for one effective llama-server command selection."""

    command: str | None
    source: Literal[
        "explicit_path",
        "path",
        "managed_cached",
        "managed_downloaded",
        "managed_pending",
        "missing",
        "unsupported",
    ]
    detail: str
    requested_command: str | None = None
    variant: str | None = None
    managed_root: Path | None = None
    platform_variant: str | None = None


LlamaServerRuntimeResolution = LlamaServerCommandResolution


@dataclass
class _ManagedServer:
    process: subprocess.Popen[str]
    handle: LlamaServerHandle
    runtime_log_path: Path | None
    log_handle: TextIOBase | None = None


class LlamaServerRuntimeError(RuntimeError):
    """Base class for managed llama-server runtime failures."""


class ManagedLlamaServerUnsupportedError(LlamaServerRuntimeError):
    """Raised when no managed runtime is available for the current platform/runtime."""


class ManagedLlamaServerDownloadError(LlamaServerRuntimeError):
    """Raised when the managed runtime download or extraction fails."""


@dataclass(frozen=True)
class _LlamaServerStartAttempt:
    config: LlamaServerConfig
    resolved_command: str
    auto_fallback: bool = False


_SERVER_REGISTRY: dict[tuple[str, str, str | None], _ManagedServer] = {}
_MANAGED_LLAMA_ASSETS: dict[tuple[str, str, str], ManagedLlamaAsset] = {
    (
        "windows",
        "x86_64",
        "cpu",
    ): ManagedLlamaAsset(
        platform_variant="windows-x86_64-cpu",
        asset_name=f"llama-{MANAGED_LLAMA_CPP_RELEASE_TAG}-bin-win-cpu-x64.zip",
        archive_format="zip",
    ),
    (
        "windows",
        "x86_64",
        "cuda12.4",
    ): ManagedLlamaAsset(
        platform_variant="windows-x86_64-cuda12.4",
        asset_name=f"llama-{MANAGED_LLAMA_CPP_RELEASE_TAG}-bin-win-cuda-12.4-x64.zip",
        archive_format="zip",
    ),
    (
        "linux",
        "x86_64",
        "cpu",
    ): ManagedLlamaAsset(
        platform_variant="linux-x86_64-cpu",
        asset_name=f"llama-{MANAGED_LLAMA_CPP_RELEASE_TAG}-bin-ubuntu-x64.tar.gz",
        archive_format="tar.gz",
    ),
    (
        "linux",
        "x86_64",
        "cuda12.4",
    ): ManagedLlamaAsset(
        platform_variant="linux-x86_64-cuda12.4",
        asset_name=f"llama-{MANAGED_LLAMA_CPP_RELEASE_TAG}-bin-ubuntu-vulkan-x64.tar.gz",
        archive_format="tar.gz",
        note=(
            "Linux GPU runtime uses the official llama.cpp Vulkan build while the documented "
            "PyTorch stack stays pinned to CUDA 12.4."
        ),
    ),
}


def ensure_llama_server(
    *,
    model_spec: ModelSpec,
    config: LlamaServerConfig,
    runtime_log_path: Path | None,
    project_root: Path | str | None = None,
) -> LlamaServerHandle:
    """Start or reuse one llama-server process for the given model/runtime."""
    normalized = config.normalized()
    attempts = _build_llama_server_start_attempts(
        config=normalized,
        project_root=project_root,
        runtime_log_path=runtime_log_path,
    )
    try:
        model_path = model_spec.resolve()
    except Exception as exc:
        raise LlamaServerRuntimeError(
            f"GGUF model resolution failed for {model_spec.display_name()}: {exc}"
        ) from exc
    for attempt in attempts:
        registry_key = (model_path, _runtime_key(attempt.config), _log_key(runtime_log_path))
        existing = _SERVER_REGISTRY.get(registry_key)
        if existing is not None and existing.process.poll() is None:
            return existing.handle
        if existing is not None:
            _SERVER_REGISTRY.pop(registry_key, None)

    for index, attempt in enumerate(attempts):
        effective_config = attempt.config
        registry_key = (model_path, _runtime_key(effective_config), _log_key(runtime_log_path))
        port = effective_config.port or _find_free_port(effective_config.host)
        handle = LlamaServerHandle(
            host=effective_config.host,
            port=port,
            model_label=model_spec.display_name(),
        )
        logger.info(
            "Starting llama-server for %s on %s (command: %s, gpu_layers=%s)",
            handle.model_label,
            handle.base_url,
            attempt.resolved_command,
            effective_config.gpu_layers,
        )
        process, log_handle = _spawn_llama_server(
            command=attempt.resolved_command,
            model_path=model_path,
            config=effective_config,
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
                timeout_s=effective_config.startup_timeout_s,
                process=process,
                command=attempt.resolved_command,
                runtime_log_path=runtime_log_path,
            )
            return handle
        except Exception as exc:
            stop_llama_server(
                handle=handle,
                model_path=model_path,
                config=effective_config,
                runtime_log_path=runtime_log_path,
            )
            if attempt.auto_fallback and index + 1 < len(attempts):
                logger.warning(
                    "PATH llama-server startup with gpu_layers=%s failed; retrying with gpu_layers=%s. %s",
                    effective_config.gpu_layers,
                    attempts[index + 1].config.gpu_layers,
                    exc,
                )
                continue
            raise
    raise AssertionError("llama-server start attempts exhausted without returning or raising")


def _build_llama_server_start_attempts(
    *,
    config: LlamaServerConfig,
    project_root: Path | str | None,
    runtime_log_path: Path | None,
) -> list[_LlamaServerStartAttempt]:
    resolution = prepare_llama_server_runtime(
        config=config,
        project_root=project_root,
        runtime_log_path=runtime_log_path,
    )
    resolved_command = str(resolution.command)
    attempts = [
        _LlamaServerStartAttempt(
            config=config,
            resolved_command=resolved_command,
        )
    ]
    if _should_try_path_offload_probe(config=config, resolution=resolution):
        attempts.insert(
            0,
            _LlamaServerStartAttempt(
                config=replace(config, gpu_layers=-1),
                resolved_command=resolved_command,
                auto_fallback=True,
            ),
        )
    return attempts


def _should_try_path_offload_probe(
    *,
    config: LlamaServerConfig,
    resolution: LlamaServerCommandResolution,
) -> bool:
    return (
        _uses_default_llama_server_command(config.command)
        and config.gpu_layers == 0
        and resolution.source == "path"
    )


def _uses_default_llama_server_command(command: str) -> bool:
    return (str(command).strip() or DEFAULT_LLAMA_SERVER_COMMAND) == DEFAULT_LLAMA_SERVER_COMMAND


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


def inspect_llama_server_command(
    command: str,
    *,
    project_root: Path | str | None = None,
    runtime_log_path: Path | None = None,
    gpu_layers: int = -1,
) -> LlamaServerCommandResolution:
    """Inspect how the effective llama-server command would resolve."""
    raw = str(command).strip() or DEFAULT_LLAMA_SERVER_COMMAND
    if _command_looks_like_path(raw):
        direct = Path(raw).expanduser()
        if direct.exists():
            return LlamaServerCommandResolution(
                command=str(direct.resolve()),
                source="explicit_path",
                detail=f"resolved explicit llama-server path: {direct.resolve()}",
                requested_command=raw,
            )
        return LlamaServerCommandResolution(
            command=None,
            source="missing",
            detail=f"llama-server command path not found: {raw!r}",
            requested_command=raw,
        )

    if raw != DEFAULT_LLAMA_SERVER_COMMAND:
        resolved = shutil.which(raw)
        if resolved:
            return LlamaServerCommandResolution(
                command=resolved,
                source="path",
                detail=f"resolved custom llama-server command from PATH: {resolved}",
                requested_command=raw,
            )
        return LlamaServerCommandResolution(
            command=None,
            source="missing",
            detail=f"llama-server command not found on PATH: {raw!r}",
            requested_command=raw,
        )

    resolved = shutil.which(DEFAULT_LLAMA_SERVER_COMMAND)
    if resolved:
        return LlamaServerCommandResolution(
            command=resolved,
            source="path",
            detail=f"resolved llama-server from PATH: {resolved}",
            requested_command=raw,
        )

    asset = _select_managed_llama_asset(gpu_layers=gpu_layers)
    if asset is None:
        return LlamaServerCommandResolution(
            command=None,
            source="unsupported",
            detail=_unsupported_runtime_detail(gpu_layers=gpu_layers),
            requested_command=raw,
        )

    managed_root = _managed_llama_root(
        project_root=project_root,
        runtime_log_path=runtime_log_path,
        asset=asset,
    )
    cached = _find_managed_llama_server_binary(managed_root)
    if cached is not None:
        return LlamaServerCommandResolution(
            command=str(cached),
            source="managed_cached",
            detail=f"managed llama-server runtime cached at {cached}",
            requested_command=raw,
            variant=asset.platform_variant,
            managed_root=managed_root,
            platform_variant=asset.platform_variant,
        )
    return LlamaServerCommandResolution(
        command=None,
        source="managed_pending",
        detail=_pending_runtime_detail(asset=asset, managed_root=managed_root),
        requested_command=raw,
        variant=asset.platform_variant,
        managed_root=managed_root,
        platform_variant=asset.platform_variant,
    )


def inspect_llama_server_runtime(
    *,
    config: LlamaServerConfig,
    project_root: Path | str | None = None,
    runtime_log_path: Path | None = None,
    allow_download: bool = False,
) -> LlamaServerCommandResolution:
    """Inspect or prepare the effective llama-server runtime for one config."""
    if allow_download:
        return prepare_llama_server_command(
            config.command,
            project_root=project_root,
            runtime_log_path=runtime_log_path,
            gpu_layers=config.gpu_layers,
        )
    return inspect_llama_server_command(
        config.command,
        project_root=project_root,
        runtime_log_path=runtime_log_path,
        gpu_layers=config.gpu_layers,
    )


def prepare_llama_server_runtime(
    *,
    config: LlamaServerConfig,
    project_root: Path | str | None = None,
    runtime_log_path: Path | None = None,
) -> LlamaServerCommandResolution:
    """Resolve or download the effective llama-server runtime for one config."""
    return prepare_llama_server_command(
        config.command,
        project_root=project_root,
        runtime_log_path=runtime_log_path,
        gpu_layers=config.gpu_layers,
    )


def prepare_llama_server_command(
    command: str,
    *,
    project_root: Path | str | None = None,
    runtime_log_path: Path | None = None,
    gpu_layers: int = -1,
) -> LlamaServerCommandResolution:
    """Resolve one effective llama-server command, downloading the managed binary when needed."""
    inspected = inspect_llama_server_command(
        command,
        project_root=project_root,
        runtime_log_path=runtime_log_path,
        gpu_layers=gpu_layers,
    )
    if inspected.command is not None:
        return inspected
    if inspected.source == "unsupported":
        raise ManagedLlamaServerUnsupportedError(inspected.detail)
    if inspected.source != "managed_pending":
        raise FileNotFoundError(inspected.detail)

    asset = _select_managed_llama_asset(gpu_layers=gpu_layers)
    if asset is None or inspected.managed_root is None:
        raise ManagedLlamaServerUnsupportedError(_unsupported_runtime_detail(gpu_layers=gpu_layers))
    binary = _ensure_managed_llama_server_binary(asset=asset, managed_root=inspected.managed_root)
    return LlamaServerCommandResolution(
        command=str(binary),
        source="managed_downloaded",
        detail=f"downloaded managed llama-server runtime to {binary}",
        requested_command=str(command).strip() or DEFAULT_LLAMA_SERVER_COMMAND,
        variant=asset.platform_variant,
        managed_root=inspected.managed_root,
        platform_variant=asset.platform_variant,
    )

def resolve_llama_server_command(
    command: str,
    *,
    project_root: Path | str | None = None,
    runtime_log_path: Path | None = None,
    gpu_layers: int = -1,
    allow_managed_download: bool = False,
) -> str:
    """Resolve a llama-server executable from an override, PATH, or managed runtime."""
    if allow_managed_download:
        return str(
            prepare_llama_server_command(
                command,
                project_root=project_root,
                runtime_log_path=runtime_log_path,
                gpu_layers=gpu_layers,
            ).command
        )
    inspected = inspect_llama_server_command(
        command,
        project_root=project_root,
        runtime_log_path=runtime_log_path,
        gpu_layers=gpu_layers,
    )
    if inspected.command is not None:
        return str(inspected.command)
    if inspected.source == "unsupported":
        raise ManagedLlamaServerUnsupportedError(inspected.detail)
    raise FileNotFoundError(inspected.detail)


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
        "--parallel",
        str(DEFAULT_LLAMA_SERVER_PARALLEL),
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
def _command_looks_like_path(raw: str) -> bool:
    path = Path(raw).expanduser()
    return path.is_absolute() or path.parent != Path(".") or any(sep in raw for sep in ("/", "\\"))


def _resolve_project_root(
    *,
    project_root: Path | str | None,
    runtime_log_path: Path | None,
) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    if runtime_log_path is not None:
        log_path = Path(runtime_log_path).expanduser().resolve()
        for parent in log_path.parents:
            if parent.name == "runs":
                return parent.parent.resolve()
        return log_path.parent.resolve()
    return Path.cwd().resolve()


def _managed_llama_root(
    *,
    project_root: Path | str | None,
    runtime_log_path: Path | None,
    asset: ManagedLlamaAsset,
) -> Path:
    root = _resolve_project_root(project_root=project_root, runtime_log_path=runtime_log_path)
    return (root / MANAGED_LLAMA_CPP_RELATIVE_ROOT / MANAGED_LLAMA_CPP_RELEASE_TAG / asset.platform_variant).resolve()


def _select_managed_llama_asset(*, gpu_layers: int) -> ManagedLlamaAsset | None:
    system = platform.system().strip().lower()
    machine = platform.machine().strip().lower()
    machine_norm = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "x64": "x86_64",
    }.get(machine, machine)
    runtime_kind = "cpu" if int(gpu_layers) == 0 else "cuda12.4"
    return _MANAGED_LLAMA_ASSETS.get((system, machine_norm, runtime_kind))


def _unsupported_runtime_detail(*, gpu_layers: int) -> str:
    system = platform.system() or "unknown-os"
    machine = platform.machine() or "unknown-arch"
    runtime_kind = "cpu" if int(gpu_layers) == 0 else "gpu"
    return (
        "package-managed llama-server runtime is unavailable for "
        f"{system}/{machine} ({runtime_kind}). "
        "Supported managed runtime targets are Windows x86_64 CPU/CUDA 12.4 and Linux x86_64 CPU/GPU."
    )


def _pending_runtime_detail(*, asset: ManagedLlamaAsset, managed_root: Path) -> str:
    detail = (
        f"managed llama-server runtime will be auto-downloaded on ads-bib run "
        f"to {managed_root} ({asset.platform_variant})"
    )
    if asset.note:
        detail = f"{detail}; {asset.note}"
    return detail


def _find_managed_llama_server_binary(managed_root: Path) -> Path | None:
    if not managed_root.exists():
        return None
    binary_name = "llama-server.exe" if platform.system().strip().lower() == "windows" else "llama-server"
    candidates = sorted(
        (path for path in managed_root.rglob(binary_name) if path.is_file()),
        key=lambda path: (len(path.parts), str(path)),
    )
    if not candidates:
        return None
    binary = candidates[0].resolve()
    if platform.system().strip().lower() != "windows":
        try:
            binary.chmod(binary.stat().st_mode | 0o111)
        except OSError:
            pass
    return binary


def _ensure_managed_llama_server_binary(*, asset: ManagedLlamaAsset, managed_root: Path) -> Path:
    cached = _find_managed_llama_server_binary(managed_root)
    if cached is not None:
        return cached

    managed_root.mkdir(parents=True, exist_ok=True)
    archive_path = managed_root / asset.asset_name
    logger.info(
        "Downloading managed llama-server runtime %s from %s",
        asset.platform_variant,
        asset.download_url,
    )
    try:
        _download_file(asset.download_url, archive_path)
        _extract_archive(archive_path=archive_path, archive_format=asset.archive_format, destination=managed_root)
    except Exception as exc:
        raise ManagedLlamaServerDownloadError(
            f"managed llama-server runtime download failed for {asset.platform_variant}: {exc}"
        ) from exc
    finally:
        if archive_path.exists():
            archive_path.unlink()

    binary = _find_managed_llama_server_binary(managed_root)
    if binary is None:
        raise ManagedLlamaServerDownloadError(
            f"managed llama-server archive for {asset.platform_variant} did not contain a llama-server binary"
        )
    return binary


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{destination.name}.", suffix=".download", dir=destination.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ads-bib/managed-llama-server"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                shutil.copyfileobj(response, handle)
        temp_path.replace(destination)
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _extract_archive(*, archive_path: Path, archive_format: Literal["zip", "tar.gz"], destination: Path) -> None:
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if os.path.commonpath([str(root), str(target)]) != str(root):
                raise ManagedLlamaServerDownloadError(f"unsafe tar member path: {member.name}")
        archive.extractall(destination)


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
            f"llama-server '{command}' is too old or incompatible for Qwen3.5 "
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
