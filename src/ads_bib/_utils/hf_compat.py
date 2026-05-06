"""Helpers for actionable HuggingFace local-model compatibility errors."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
import importlib.abc
import importlib.util
from importlib import metadata
import os
import re
import sys
from typing import Any

os.environ.setdefault("USE_TF", "0")

_MODEL_TYPE_PATTERN = re.compile(r"model type [`']([^`']+)[`']")
_KNOWN_NEW_MODEL_TYPES = {"gemma3", "gemma3_text", "qwen3"}
_ARCH_MISMATCH_MARKERS = (
    "does not recognize this architecture",
    "unknown configuration class",
    "unrecognized configuration class",
    "unknown task image-text-to-text",
)
_TORCH_VERSION_PATTERN = re.compile(r"torch\s*>=\s*([0-9]+(?:\.[0-9]+){1,2})", re.IGNORECASE)
_TORCH_RUNTIME_MARKERS = (
    "require torch>=",
    "requires torch>=",
)
_TORCHVISION_RUNTIME_MARKERS = (
    "operator torchvision::nms does not exist",
)
_TORCHCODEC_RUNTIME_MARKERS = (
    "could not load libtorchcodec",
)


class _OptionalTorchcodecBlocker(importlib.abc.MetaPathFinder):
    """Make optional torchcodec imports look absent during text-only ST imports."""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        del path, target
        if fullname == "torchcodec" or fullname.startswith("torchcodec."):
            raise ImportError("torchcodec is disabled for ads-bib text-only SentenceTransformer use")
        return None


def _iter_exception_chain(exc: BaseException):
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


def _message_matches_arch_mismatch(message: str) -> bool:
    msg = message.lower()
    if any(marker in msg for marker in _ARCH_MISMATCH_MARKERS):
        return True
    model_type_match = _MODEL_TYPE_PATTERN.search(message)
    if model_type_match and model_type_match.group(1).lower() in _KNOWN_NEW_MODEL_TYPES:
        return True
    if message.startswith("KeyError:"):
        key_name = message.removeprefix("KeyError:").strip().strip("'\"`").lower()
        if key_name in _KNOWN_NEW_MODEL_TYPES:
            return True
    return False


def is_transformers_architecture_mismatch(exc: BaseException) -> bool:
    """Return True when *exc* indicates unsupported model architecture."""
    for nested in _iter_exception_chain(exc):
        if _message_matches_arch_mismatch(str(nested)):
            return True
    return False


def is_torch_runtime_requirement_error(exc: BaseException) -> bool:
    """Return True when *exc* indicates an unmet torch runtime requirement."""
    for nested in _iter_exception_chain(exc):
        msg = str(nested).lower()
        if any(marker in msg for marker in _TORCH_RUNTIME_MARKERS):
            return True
    return False


def is_torchvision_runtime_mismatch_error(exc: BaseException) -> bool:
    """Return True when *exc* indicates an incompatible torch/torchvision pair."""
    for nested in _iter_exception_chain(exc):
        msg = str(nested).lower()
        if any(marker in msg for marker in _TORCHVISION_RUNTIME_MARKERS):
            return True
    return False


def is_torchcodec_runtime_error(exc: BaseException) -> bool:
    """Return True when *exc* indicates a broken optional TorchCodec install."""
    for nested in _iter_exception_chain(exc):
        msg = str(nested).lower()
        if any(marker in msg for marker in _TORCHCODEC_RUNTIME_MARKERS):
            return True
    return False


def _extract_model_type(exc: BaseException) -> str | None:
    for nested in _iter_exception_chain(exc):
        match = _MODEL_TYPE_PATTERN.search(str(nested))
        if match:
            return match.group(1)
    return None


def _extract_required_torch_version(exc: BaseException) -> str | None:
    for nested in _iter_exception_chain(exc):
        match = _TORCH_VERSION_PATTERN.search(str(nested))
        if match:
            return match.group(1)
    return None


def _installed_distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


@contextmanager
def _temporarily_block_optional_torchcodec() -> Iterator[None]:
    """Hide TorchCodec while importing SentenceTransformers for text embeddings."""
    blocker = _OptionalTorchcodecBlocker()
    original_find_spec = importlib.util.find_spec

    def _find_spec_without_torchcodec(name: str, package: str | None = None):
        if name == "torchcodec" or name.startswith("torchcodec."):
            return None
        return original_find_spec(name, package)

    torchcodec_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "torchcodec" or name.startswith("torchcodec.")
    }
    for name in torchcodec_modules:
        sys.modules.pop(name, None)
    importlib.util.find_spec = _find_spec_without_torchcodec
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass
        for name in list(sys.modules):
            if (name == "torchcodec" or name.startswith("torchcodec.")) and name not in torchcodec_modules:
                sys.modules.pop(name, None)
        sys.modules.update(torchcodec_modules)


def import_sentence_transformer_class() -> Any:
    """Import SentenceTransformer while treating optional TorchCodec as unavailable."""
    with _temporarily_block_optional_torchcodec():
        from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def build_local_hf_compat_message(
    *,
    model: str,
    use_case: str,
    exc: BaseException,
) -> str:
    """Build a concise upgrade hint for unsupported local HF architectures."""
    try:
        import transformers

        transformers_version = getattr(transformers, "__version__", "unknown")
    except Exception:
        transformers_version = "unknown"

    model_type = _extract_model_type(exc)
    details = f"Unsupported model architecture '{model_type}'." if model_type else "Unsupported model architecture."
    return (
        f"Local {use_case} model '{model}' could not be loaded. {details} "
        f"Installed transformers version: {transformers_version}. "
        "Upgrade the local HF stack in your active Python environment, then restart the kernel:\n"
        'uv pip install -U "transformers>=4.56,<4.57" "sentence-transformers>=5.1" "accelerate>=0.31"'
    )


def build_torch_runtime_message(
    *,
    model: str,
    use_case: str,
    exc: BaseException,
) -> str:
    """Build actionable hint for unmet torch runtime requirements."""
    try:
        import torch

        torch_version = getattr(torch, "__version__", "unknown")
    except Exception:
        torch_version = "unknown"

    required = _extract_required_torch_version(exc) or "2.6"
    return (
        f"Local {use_case} model '{model}' requires torch>={required}, but installed torch is {torch_version}. "
        "Upgrade torch in your active Python environment, then restart the kernel:\n"
        'uv pip install -U "torch==2.6.0" "torchvision==0.21.0" '
        "--extra-index-url https://download.pytorch.org/whl/cpu"
    )


def build_torchvision_runtime_message(
    *,
    model: str,
    use_case: str,
    exc: BaseException,
) -> str:
    """Build actionable hint for incompatible torch/torchvision binaries."""
    del exc
    try:
        import torch

        torch_version = getattr(torch, "__version__", "unknown")
    except Exception:
        torch_version = _installed_distribution_version("torch")

    torchvision_version = _installed_distribution_version("torchvision")
    return (
        f"Local {use_case} model '{model}' could not be loaded because torch and torchvision "
        "are incompatible in the active Python environment. "
        f"Installed versions: torch {torch_version}, torchvision {torchvision_version}. "
        "This is common in Colab after installing a pinned torch wheel while keeping "
        "Colab's preinstalled torchvision. Reinstall the matching pair, then restart the kernel:\n"
        'uv pip install -U "torch==2.6.0" "torchvision==0.21.0" '
        "--extra-index-url https://download.pytorch.org/whl/cu124"
    )


def build_torchcodec_runtime_message(
    *,
    model: str,
    use_case: str,
    exc: BaseException,
) -> str:
    """Build actionable hint for broken optional TorchCodec installs."""
    del exc
    torchcodec_version = _installed_distribution_version("torchcodec")
    return (
        f"Local {use_case} model '{model}' could not be loaded because optional "
        f"torchcodec is installed but unusable (torchcodec {torchcodec_version}). "
        "ads-bib uses text-only SentenceTransformers here and does not need torchcodec. "
        "Uninstall torchcodec in the active environment, then restart the kernel:\n"
        "uv pip uninstall torchcodec\n"
        "In Colab use: uv pip uninstall --system torchcodec"
    )


def raise_with_local_hf_compat_hint(
    *,
    model: str,
    use_case: str,
    exc: BaseException,
) -> None:
    """Raise RuntimeError with upgrade hint for architecture mismatches."""
    if is_transformers_architecture_mismatch(exc):
        raise RuntimeError(
            build_local_hf_compat_message(model=model, use_case=use_case, exc=exc)
        ) from exc
    if is_torch_runtime_requirement_error(exc):
        raise RuntimeError(
            build_torch_runtime_message(model=model, use_case=use_case, exc=exc)
        ) from exc
    if is_torchvision_runtime_mismatch_error(exc):
        raise RuntimeError(
            build_torchvision_runtime_message(model=model, use_case=use_case, exc=exc)
        ) from exc
    if is_torchcodec_runtime_error(exc):
        raise RuntimeError(
            build_torchcodec_runtime_message(model=model, use_case=use_case, exc=exc)
        ) from exc
    raise exc
