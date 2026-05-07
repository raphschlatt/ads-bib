"""Shared helpers for local model runtime and memory hygiene."""

from __future__ import annotations

from collections import deque
import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)

_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "out of memory",
    "defaultcpuallocator",
    "cannot allocate memory",
    "not enough memory",
)

_SAMPLING_ONLY_GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
)


def iter_exception_chain(exc: BaseException):
    """Yield *exc* and nested causes/contexts once each."""
    seen: set[int] = set()
    queue = deque([exc])
    while queue:
        current = queue.popleft()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        yield current
        if current.__cause__ is not None:
            queue.append(current.__cause__)
        if current.__context__ is not None:
            queue.append(current.__context__)


def is_memory_oom_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a local CPU/CUDA memory exhaustion error."""
    if isinstance(exc, MemoryError):
        return True
    for nested in iter_exception_chain(exc):
        if isinstance(nested, MemoryError):
            return True
        message = str(nested).lower()
        if any(marker in message for marker in _OOM_MARKERS):
            return True
    return False


def clear_local_memory() -> None:
    """Release Python objects and free cached CUDA allocator blocks when available."""
    gc.collect()
    try:
        import torch
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("Ignoring CUDA cache cleanup failure: %s", exc)


def choose_local_torch_dtype(torch: Any) -> Any | None:
    """Choose a compact local inference dtype for the active runtime."""
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None

    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _resolve_pad_token_id(model: Any, tokenizer: Any | None = None) -> Any | None:
    for source in (
        tokenizer,
        getattr(model, "generation_config", None),
        getattr(model, "config", None),
    ):
        if source is None:
            continue
        pad_token_id = getattr(source, "pad_token_id", None)
        if pad_token_id is not None:
            return pad_token_id
    for source in (
        tokenizer,
        getattr(model, "generation_config", None),
        getattr(model, "config", None),
    ):
        if source is None:
            continue
        eos_token_id = getattr(source, "eos_token_id", None)
        if eos_token_id is not None:
            return eos_token_id
    return None


def configure_deterministic_generation(model: Any, *, tokenizer: Any | None = None) -> Any | None:
    """Normalize local HF generation defaults and return the resolved pad token id."""
    generation_config = getattr(model, "generation_config", None)
    pad_token_id = _resolve_pad_token_id(model, tokenizer)

    if generation_config is not None:
        generation_config.do_sample = False
        for name in _SAMPLING_ONLY_GENERATION_FIELDS:
            if hasattr(generation_config, name):
                setattr(generation_config, name, None)
        if pad_token_id is not None and hasattr(generation_config, "pad_token_id"):
            generation_config.pad_token_id = pad_token_id

    model_config = getattr(model, "config", None)
    if pad_token_id is not None and model_config is not None and hasattr(model_config, "pad_token_id"):
        model_config.pad_token_id = pad_token_id

    return pad_token_id
