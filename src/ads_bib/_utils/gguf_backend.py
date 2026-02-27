"""GGUF model loading and inference via llama-cpp-python.

Provides local translation, BERTopic-compatible topic labeling, and
Toponymy-compatible LLM wrappers — all backed by quantised GGUF models
for fast CPU inference.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Local GGUF inference requires 'llama-cpp-python'. Install with:\n"
    "  pip install llama-cpp-python "
    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
    "For GPU-accelerated builds see "
    "https://github.com/abetlen/llama-cpp-python#installation"
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
    n_gpu_layers: int = 0,
    verbose: bool = False,
) -> Any:
    """Load a ``llama_cpp.Llama`` instance with actionable import errors."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    with _safe_stdio():
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

_translation_model: Any | None = None
_translation_model_path: str | None = None


def translate_gguf(
    text: str,
    target_lang: str,
    *,
    model_path: str,
    n_ctx: int = 2048,
    max_tokens: int = 2048,
) -> str:
    """Translate *text* into *target_lang* using a local GGUF model.

    The translation model is lazy-loaded on first call and cached at
    module level (same lifecycle pattern as the fasttext model in
    ``translate.py``).
    """
    global _translation_model, _translation_model_path
    if _translation_model is None or _translation_model_path != model_path:
        logger.info("Loading GGUF translation model: %s", model_path)
        _translation_model = _load_llama(model_path, n_ctx=n_ctx)
        _translation_model_path = model_path

    with _safe_stdio():
        result = _translation_model(
            f"Translate the following scientific text to {target_lang}.\n"
            "Return only the translation and no explanation.\n\n"
            f"{text}",
            max_tokens=max_tokens,
            temperature=0,
            echo=False,
        )
    return result["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# BERTopic labeling wrapper
# ---------------------------------------------------------------------------

class LlamaCppTextGeneration:
    """BERTopic-compatible callable wrapping a GGUF model for topic labeling.

    BERTopic's ``TextGeneration`` calls ``model(prompt, **pipeline_kwargs)``
    and extracts the completion via ``generated_text.replace(prompt, "")``.
    This class satisfies that protocol.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.max_new_tokens = max_new_tokens
        self._llm: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._llm is None:
            logger.info("Loading GGUF labeling model: %s", self.model_path)
            self._llm = _load_llama(self.model_path, n_ctx=self.n_ctx)

    def __call__(self, prompt: str, **kwargs: Any) -> list[dict[str, str]]:
        self._ensure_loaded()
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        with _safe_stdio():
            result = self._llm(prompt, max_tokens=max_tokens, temperature=0, echo=False)
        completion = result["choices"][0]["text"].strip()
        # BERTopic strips prompt via .replace(prompt, "")
        return [{"generated_text": prompt + completion}]


# ---------------------------------------------------------------------------
# Toponymy labeling wrapper
# ---------------------------------------------------------------------------

def _build_llama_cpp_namer(
    model_path: str,
    *,
    n_ctx: int = 4096,
    max_new_tokens: int = 256,
) -> Any:
    """Build a Toponymy-compatible sync LLM wrapper backed by a GGUF model.

    The returned object inherits from ``toponymy.llm_wrappers.LLMWrapper``
    so that Toponymy's ``isinstance`` / ABC checks pass.  The import is
    deferred so that ``toponymy`` stays an optional dependency.
    """
    from toponymy.llm_wrappers import LLMWrapper

    class _LlamaCppNamer(LLMWrapper):
        """Toponymy LLMWrapper backed by llama-cpp-python."""

        def __init__(self) -> None:
            self._model_path = model_path
            self._max_new_tokens = max(1, int(max_new_tokens))
            self._n_ctx = n_ctx
            self._llm: Any | None = None

        def _ensure_loaded(self) -> None:
            if self._llm is None:
                logger.info("Loading GGUF Toponymy labeling model: %s", self._model_path)
                self._llm = _load_llama(self._model_path, n_ctx=self._n_ctx)

        def _cap(self, requested: int) -> int:
            return min(max(1, int(requested)), self._max_new_tokens)

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            self._ensure_loaded()
            with _safe_stdio():
                result = self._llm(
                    prompt, max_tokens=self._cap(max_tokens), temperature=0, echo=False,
                )
            return result["choices"][0]["text"].strip()

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            return self._call_llm(
                f"{system_prompt}\n\n{user_prompt}", temperature, max_tokens,
            )

    return _LlamaCppNamer()
