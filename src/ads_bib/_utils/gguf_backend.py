"""GGUF model loading and inference via llama-cpp-python.

Provides model resolution (local path or HuggingFace Hub download),
Jupyter ``fileno()`` safety patching, and local translation — all
backed by quantised GGUF models for fast CPU inference.

Topic-labeling integration uses the native library classes directly:
* BERTopic  → ``bertopic.representation.LlamaCPP``
* Toponymy  → ``toponymy.llm_wrappers.LlamaCppNamer``
"""

from __future__ import annotations

from collections import deque
import functools
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Local GGUF inference requires 'llama-cpp-python'. Install with:\n"
    "  conda install -n ADS_env -c conda-forge llama-cpp-python=0.3.16\n"
    "or (uv in active ADS_env):\n"
    "  uv pip install -U llama-cpp-python "
    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
    "or (pip, same interpreter as your notebook kernel):\n"
    "  python -m pip install -U llama-cpp-python "
    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
    "For GPU-accelerated builds see "
    "https://github.com/abetlen/llama-cpp-python#installation"
)
_MIN_LLAMA_CPP_FOR_GEMMA3 = (0, 3, 8)
_GGUF_ARCH_MISMATCH_MARKERS = (
    "unknown model architecture",
    "failed to load model",
    "unknown architecture",
    "unsupported architecture",
)
_VERSION_TOKEN_RE = re.compile(r"(\d+)")


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


def _parse_version_triplet(version: str) -> tuple[int, int, int] | None:
    """Parse semantic version prefix from a version string."""
    parts: list[int] = []
    for token in version.split("."):
        match = _VERSION_TOKEN_RE.match(token)
        if not match:
            break
        parts.append(int(match.group(1)))
        if len(parts) == 3:
            break
    if not parts:
        return None
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _is_version_lt(version: str, floor: tuple[int, int, int]) -> bool:
    parsed = _parse_version_triplet(version)
    return parsed is not None and parsed < floor


def _looks_like_gemma3_model(model_path: str) -> bool:
    name = Path(model_path).name.lower()
    return "gemma-3" in name or "gemma3" in name


def _is_architecture_mismatch_error(exc: BaseException) -> bool:
    for nested in _iter_exception_chain(exc):
        msg = str(nested).lower()
        if any(marker in msg for marker in _GGUF_ARCH_MISMATCH_MARKERS):
            return True
    return False


def _get_llama_cpp_version() -> str:
    llama_mod = sys.modules.get("llama_cpp")
    version = getattr(llama_mod, "__version__", None)
    return str(version) if version else "unknown"


def _build_llama_load_error_message(*, model_path: str, exc: BaseException) -> str:
    """Build actionable runtime hint for GGUF model load failures."""
    version = _get_llama_cpp_version()
    is_old_for_gemma3 = _looks_like_gemma3_model(model_path) and _is_version_lt(
        version, _MIN_LLAMA_CPP_FOR_GEMMA3
    )
    if is_old_for_gemma3:
        return (
            f"GGUF model '{model_path}' could not be loaded. This Gemma 3 GGUF requires a newer "
            f"llama-cpp-python runtime than the installed version ({version}). "
            "Upgrade llama-cpp-python in ADS_env, then restart the kernel:\n"
            "conda install -n ADS_env -c conda-forge llama-cpp-python=0.3.16\n"
            "or\n"
            "uv pip install -U llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
            "or\n"
            "python -m pip install -U llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
        )

    details = str(exc).strip()
    details_suffix = f" Details: {details}" if details else ""
    return (
        f"GGUF model '{model_path}' could not be loaded via llama-cpp-python "
        f"(installed version: {version}). Ensure the GGUF file is valid and this runtime supports "
        f"its architecture.{details_suffix}"
    )


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def resolve_gguf_model(model: str) -> str:
    """Resolve a GGUF model specifier to a local file path.

    Accepted formats:

    * **Local path** — returned as-is if the file exists.
    * **``repo_id:filename``** — downloads *filename* from HuggingFace
      *repo_id* (e.g. ``"mradermacher/translategemma-4b-it-GGUF:translategemma-4b-it.Q4_K_M.gguf"``).
    * **``repo_id``** (no colon) — derives a default filename
      ``{basename}.Q4_K_M.gguf`` from the last path component and
      downloads it.

    Downloaded files are cached by *huggingface_hub* and not
    re-downloaded on subsequent calls.
    """
    if Path(model).is_file():
        return str(Path(model).resolve())

    from huggingface_hub import hf_hub_download

    if ":" in model:
        repo_id, filename = model.rsplit(":", 1)
    else:
        repo_id = model
        base = repo_id.split("/")[-1].removesuffix("-GGUF").removesuffix("-gguf")
        filename = f"{base}.Q4_K_M.gguf"

    logger.info("Downloading GGUF model %s/%s …", repo_id, filename)
    return hf_hub_download(repo_id=repo_id, filename=filename)


# ---------------------------------------------------------------------------
# Low-level loading
# ---------------------------------------------------------------------------

@contextmanager
def _safe_stdio():
    """Temporarily ensure sys.stdout/stderr support fileno().

    llama-cpp-python's ``Llama()`` constructor tries to redirect C-level
    stdout/stderr via ``fileno()`` to suppress llama.cpp log output.
    In Jupyter notebook kernels on Windows, ``sys.stdout`` is a custom
    ``OutStream`` that does not support ``fileno()``, causing
    ``UnsupportedOperation`` errors.

    This context manager replaces non-seekable streams with ``os.devnull``
    wrappers for the duration of the Llama constructor call and restores
    the originals afterwards.
    """
    orig_out, orig_err = sys.stdout, sys.stderr
    needs_patch = False
    try:
        sys.stdout.fileno()
    except Exception:
        needs_patch = True

    if not needs_patch:
        yield
        return

    devnull = open(os.devnull, "w")  # noqa: SIM115
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        devnull.close()


def _load_llama(
    model_path: str,
    *,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    n_threads: int | None = None,
    n_threads_batch: int | None = None,
    vocab_only: bool = False,
    verbose: bool = False,
) -> Any:
    """Load a ``llama_cpp.Llama`` instance with actionable compatibility errors."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    try:
        with _safe_stdio():
            return Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                vocab_only=vocab_only,
                verbose=verbose,
            )
    except AssertionError as exc:
        raise RuntimeError(_build_llama_load_error_message(model_path=model_path, exc=exc)) from exc
    except Exception as exc:
        is_gemma3_old_runtime = _looks_like_gemma3_model(model_path) and _is_version_lt(
            _get_llama_cpp_version(), _MIN_LLAMA_CPP_FOR_GEMMA3
        )
        if _is_architecture_mismatch_error(exc) or is_gemma3_old_runtime:
            raise RuntimeError(_build_llama_load_error_message(model_path=model_path, exc=exc)) from exc
        raise


def _make_llama_jupyter_safe(llm: Any) -> None:
    """Monkeypatch a Llama instance so inference calls survive Jupyter's stdout.

    llama-cpp-python redirects C-level stdout/stderr during ``__call__``
    and ``create_chat_completion``.  In Jupyter on Windows the kernel's
    ``sys.stdout`` lacks ``fileno()``, raising ``UnsupportedOperation``.

    This function wraps both methods with :func:`_safe_stdio` so they
    work transparently inside notebook kernels.  If ``fileno()`` already
    works, nothing is patched.
    """
    try:
        sys.stdout.fileno()
        return  # regular terminal — no patch needed
    except Exception:
        pass

    original_call = llm.__call__
    original_chat = llm.create_chat_completion

    @functools.wraps(original_call)
    def _safe_call(*args: Any, **kwargs: Any) -> Any:
        with _safe_stdio():
            return original_call(*args, **kwargs)

    @functools.wraps(original_chat)
    def _safe_chat(*args: Any, **kwargs: Any) -> Any:
        with _safe_stdio():
            return original_chat(*args, **kwargs)

    llm.__call__ = _safe_call
    llm.create_chat_completion = _safe_chat


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

_translation_model: Any | None = None
_translation_model_path: str | None = None
_translation_model_config: tuple[int, int | None, int | None] | None = None

_translation_tokenizer: Any | None = None
_translation_tokenizer_path: str | None = None


def _ensure_translation_model(
    *,
    model_path: str,
    n_ctx: int,
    n_threads: int | None,
    n_threads_batch: int | None,
) -> Any:
    """Load and cache the GGUF translation model for the current process."""
    global _translation_model, _translation_model_path, _translation_model_config
    model_config = (
        int(n_ctx),
        int(n_threads) if n_threads is not None else None,
        int(n_threads_batch) if n_threads_batch is not None else None,
    )
    if (
        _translation_model is None
        or _translation_model_path != model_path
        or _translation_model_config != model_config
    ):
        logger.info("Loading GGUF translation model: %s", model_path)
        _translation_model = _load_llama(
            model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
        )
        _make_llama_jupyter_safe(_translation_model)
        _translation_model_path = model_path
        _translation_model_config = model_config
    return _translation_model


def _ensure_translation_tokenizer(*, model_path: str) -> Any:
    """Load and cache vocab-only tokenizer model for current process."""
    global _translation_tokenizer, _translation_tokenizer_path
    if _translation_tokenizer is None or _translation_tokenizer_path != model_path:
        logger.info("Loading GGUF tokenizer model: %s", model_path)
        _translation_tokenizer = _load_llama(
            model_path,
            n_ctx=512,
            n_gpu_layers=0,
            vocab_only=True,
            verbose=False,
        )
        _translation_tokenizer_path = model_path
    return _translation_tokenizer


def prime_gguf_translation_runtime(
    *,
    model_path: str,
    n_ctx: int,
    n_threads: int | None,
    n_threads_batch: int | None,
    preload_tokenizer: bool,
) -> None:
    """Prime worker-local model state for translation process pools."""
    _ensure_translation_model(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
    )
    if preload_tokenizer:
        _ensure_translation_tokenizer(model_path=model_path)


def translate_gguf(
    text: str,
    target_lang: str,
    *,
    source_lang: str,
    model_path: str,
    n_ctx: int = 4096,
    n_threads: int | None = None,
    n_threads_batch: int | None = None,
    max_tokens: int = 2048,
) -> str:
    """Translate *text* into *target_lang* using a local GGUF model.

    The translation model is lazy-loaded on first call and cached at
    module level (same lifecycle pattern as the fasttext model in
    ``translate.py``).
    """
    model_obj = _ensure_translation_model(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
    )

    with _safe_stdio():
        result = model_obj.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "text": text,
                        },
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
    return result["choices"][0]["message"]["content"].strip()


def gguf_supports_gpu_offload() -> bool:
    """Return whether the installed llama.cpp runtime supports GPU offload."""
    try:
        from llama_cpp import llama_cpp as llama_lib
    except Exception:
        return False
    try:
        return bool(llama_lib.llama_supports_gpu_offload())
    except Exception:
        return False


def _get_translation_tokenizer(model_path: str) -> Any:
    """Return a cached vocab-only GGUF tokenizer model for translation chunking."""
    return _ensure_translation_tokenizer(model_path=model_path)


def count_gguf_tokens(text: str, *, model_path: str) -> int:
    """Count GGUF tokenizer tokens for *text* using a vocab-only model."""
    tokenizer = _get_translation_tokenizer(model_path)
    tokens = tokenizer.tokenize(str(text).encode("utf-8"), add_bos=False, special=False)
    return len(tokens)


def split_text_by_gguf_tokens(
    text: str,
    *,
    model_path: str,
    max_input_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into GGUF token chunks with overlap for robust long-text translation."""
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be > 0.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0.")
    if overlap_tokens >= max_input_tokens:
        raise ValueError("overlap_tokens must be < max_input_tokens.")

    tokenizer = _get_translation_tokenizer(model_path)
    token_ids = tokenizer.tokenize(str(text).encode("utf-8"), add_bos=False, special=False)
    if len(token_ids) <= max_input_tokens:
        return [str(text)]

    chunks: list[str] = []
    start = 0
    n_tokens = len(token_ids)
    while start < n_tokens:
        end = min(start + max_input_tokens, n_tokens)
        chunk_bytes = tokenizer.detokenize(token_ids[start:end], special=False)
        chunk_text = chunk_bytes.decode("utf-8", errors="replace").strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= n_tokens:
            break
        start = max(0, end - overlap_tokens)
    return chunks or [str(text)]
