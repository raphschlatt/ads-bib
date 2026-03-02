"""Shared local runtime helpers for GGUF-backed inference paths."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, TypeAlias


GGUFParallelPolicy: TypeAlias = Literal[
    "auto_calibrated",
    "balanced_auto",
    "max_throughput",
    "stability_first",
]
GGUFTokenBudgetMode: TypeAlias = Literal["column_aware", "global"]


@dataclass(frozen=True)
class GGUFRuntimePlan:
    """Resolved GGUF runtime parameters for local inference."""

    workers: int
    threads: int
    threads_batch: int
    n_ctx: int
    policy: GGUFParallelPolicy
    gpu_offload_supported: bool
    calibrated: bool
    token_budget_mode: GGUFTokenBudgetMode


def cpu_count() -> int:
    """Return CPU count with safe lower-bound fallback."""
    return max(1, int(os.cpu_count() or 1))


def available_memory_bytes() -> int | None:
    """Return available physical memory in bytes (Linux + Windows)."""
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            if ok:
                return int(status.ullAvailPhys)
        except Exception:
            return None
    return None


def estimate_gguf_ram_cap(*, model_path: str, max_workers: int) -> int:
    """Conservative max worker estimate based on free RAM and model size."""
    free_mem = available_memory_bytes()
    if free_mem is None:
        return max(1, int(max_workers))
    try:
        model_size = max(1, int(Path(model_path).stat().st_size))
    except OSError:
        return max(1, int(max_workers))

    # Headroom includes model weights, KV cache, allocator overhead.
    per_worker_bytes = max(int(model_size * 1.8), 2_500_000_000)
    cap = int(free_mem // per_worker_bytes)
    return max(1, min(int(max_workers), cap))


def gguf_provider_build_tag() -> str:
    """Return a stable runtime tag used for auto-calibration cache keys."""
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as llama_lib
    except Exception:
        return "missing_llama_cpp"

    version = str(getattr(llama_cpp, "__version__", "unknown"))
    try:
        gpu = bool(llama_lib.llama_supports_gpu_offload())
    except Exception:
        gpu = False
    return f"{version}|gpu_offload={int(gpu)}"


def resolve_gguf_runtime_plan(
    *,
    max_workers: int,
    policy: GGUFParallelPolicy,
    model_path: str,
    n_ctx: int,
    n_threads: int | None,
    n_threads_batch: int | None,
    token_budget_mode: GGUFTokenBudgetMode,
    gpu_offload_supported: bool,
    calibrated: bool = False,
) -> GGUFRuntimePlan:
    """Resolve a practical local GGUF runtime plan from policy + host resources."""
    worker_request = max(1, int(max_workers))
    cpu_total = cpu_count()
    ram_cap = estimate_gguf_ram_cap(model_path=model_path, max_workers=worker_request)

    if policy == "stability_first":
        workers = 1
    elif policy == "max_throughput":
        workers = min(worker_request, ram_cap)
    else:
        if gpu_offload_supported:
            workers = 1
        else:
            thread_hint = int(n_threads) if n_threads is not None else min(8, cpu_total)
            thread_hint = max(1, thread_hint)
            cpu_cap = max(1, cpu_total // thread_hint)
            workers = min(worker_request, cpu_cap, ram_cap)
    workers = max(1, int(workers))

    if n_threads is None:
        if workers == 1:
            threads = min(8, cpu_total)
        else:
            threads = max(2, min(8, cpu_total // workers))
    else:
        threads = max(1, int(n_threads))

    if n_threads_batch is None:
        threads_batch = max(1, min(cpu_total, max(threads, threads * 2)))
    else:
        threads_batch = max(1, int(n_threads_batch))

    return GGUFRuntimePlan(
        workers=workers,
        threads=threads,
        threads_batch=threads_batch,
        n_ctx=max(1, int(n_ctx)),
        policy=policy,
        gpu_offload_supported=bool(gpu_offload_supported),
        calibrated=bool(calibrated),
        token_budget_mode=token_budget_mode,
    )


def build_gguf_calibration_candidates(
    *,
    max_workers: int,
    cpu_total: int | None = None,
) -> list[tuple[int, int]]:
    """Build ordered `(workers, threads)` candidates for CPU auto-calibration."""
    cpu_total = max(1, int(cpu_total or cpu_count()))
    worker_request = max(1, int(max_workers))
    raw: list[tuple[int, int]] = [
        (1, min(8, cpu_total)),
        (min(2, worker_request), max(2, cpu_total // max(1, min(2, worker_request)))),
    ]
    if cpu_total >= 12 and worker_request >= 3:
        w3 = min(3, worker_request)
        raw.append((w3, max(2, cpu_total // w3)))

    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for workers, threads in raw:
        candidate = (max(1, int(workers)), max(1, int(threads)))
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates

