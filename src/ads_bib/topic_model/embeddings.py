"""Embedding providers and caching for topic modeling."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ads_bib._utils.ads_api import retry_call
from ads_bib._utils.hf_compat import raise_with_local_hf_compat_hint
from ads_bib._utils.openrouter_costs import (
    extract_generation_id,
    extract_response_cost,
    extract_usage_stats,
    normalize_openrouter_cost_mode,
    resolve_openrouter_costs,
)
from ads_bib.config import validate_provider

logger = logging.getLogger("ads_bib.topic_model")

_DOC_FINGERPRINT_SEPARATOR = "\x1f"
_DOC_FINGERPRINT_ENCODING = "utf-8"
_PAID_PROVIDER_FALLBACK_DIM = 3072
_PAID_PROVIDER_DIM_HINTS = {
    "google/gemini-embedding-001": 3072,
}
_EMBEDDING_MEMORY_OVERHEAD_FACTOR = 1.5
_EMBEDDING_MEMORY_BUDGET_FRACTION = 0.70
_EMBEDDING_MEMORY_FALLBACK_BUDGET_BYTES = 2 * 1024**3


def _documents_fingerprint(documents: list[str]) -> str:
    """Return deterministic SHA-256 over document sequence."""
    hasher = hashlib.sha256()
    for doc in documents:
        payload = str(doc).encode(_DOC_FINGERPRINT_ENCODING, errors="replace")
        hasher.update(payload)
        hasher.update(_DOC_FINGERPRINT_SEPARATOR.encode(_DOC_FINGERPRINT_ENCODING))
    return hasher.hexdigest()


def _available_memory_bytes() -> int | None:
    """Return available system memory bytes when discoverable."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or available_pages <= 0:
        return None
    return page_size * available_pages


def _embedding_dim_hint(provider: str, model: str) -> int | None:
    """Return a conservative embedding dimension estimate for paid providers."""
    if provider not in {"openrouter", "huggingface_api"}:
        return None
    model_key = str(model)
    if model_key.startswith("openrouter/"):
        model_key = model_key[len("openrouter/") :]
    return _PAID_PROVIDER_DIM_HINTS.get(model_key, _PAID_PROVIDER_FALLBACK_DIM)


def _bytes_to_gib(value: int) -> float:
    return float(value) / float(1024**3)


def _assert_memory_budget_for_paid_embeddings(
    *,
    provider: str,
    model: str,
    n_docs: int,
    dtype: Any,
) -> None:
    """Fail fast for paid providers if estimated in-memory assembly is unsafe."""
    if n_docs <= 0:
        return
    dim_hint = _embedding_dim_hint(provider, model)
    if dim_hint is None:
        return

    dtype_size = int(np.dtype(dtype).itemsize)
    required_bytes = int(n_docs * dim_hint * dtype_size)
    estimated_peak_bytes = int(required_bytes * _EMBEDDING_MEMORY_OVERHEAD_FACTOR)
    available = _available_memory_bytes()
    if available is None:
        budget_bytes = _EMBEDDING_MEMORY_FALLBACK_BUDGET_BYTES
        budget_label = "fallback 2.00 GiB"
    else:
        budget_bytes = int(available * _EMBEDDING_MEMORY_BUDGET_FRACTION)
        budget_label = f"{int(_EMBEDDING_MEMORY_BUDGET_FRACTION * 100)}% of available RAM"

    if estimated_peak_bytes <= budget_bytes:
        return

    raise MemoryError(
        "Embedding preflight aborted before API calls: estimated peak memory "
        f"{_bytes_to_gib(estimated_peak_bytes):.2f} GiB exceeds budget "
        f"{_bytes_to_gib(budget_bytes):.2f} GiB ({budget_label}) for "
        f"{provider}/{model}. Set SAMPLE_SIZE, keep dtype=float16, or switch provider/model."
    )


def compute_embeddings(
    documents: list[str],
    *,
    provider: str,
    model: str,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    max_workers: int = 5,
    dtype=np.float16,
    api_key: str | None = None,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
) -> np.ndarray:
    """Compute document embeddings with optional cache reuse.

    Parameters
    ----------
    documents : list[str]
        Ordered document texts to embed.
    provider : str
        Embedding backend: ``"local"``, ``"huggingface_api"``, or
        ``"openrouter"``.
    model : str
        Provider-specific embedding model identifier.
    cache_dir : Path, optional
        Cache directory. When set, embeddings are loaded from/saved to
        ``embeddings_{provider}_{model}.npz`` with fingerprint validation.
    batch_size : int
        Per-call batch size for embedding requests.
    max_workers : int
        Worker count used by concurrent providers (OpenRouter).
    dtype : Any
        Target numpy dtype for returned array.
    api_key : str, optional
        Required for ``provider="openrouter"``.
    openrouter_cost_mode : str
        Cost resolution mode for OpenRouter (``"hybrid"``, ``"strict"``,
        ``"fast"``).
    cost_tracker : CostTracker, optional
        When provided, records embedding token/cost summaries.

    Returns
    -------
    np.ndarray
        Embedding matrix with shape ``(n_documents, embedding_dim)``.
    """
    validate_provider(
        provider,
        valid={"local", "huggingface_api", "openrouter"},
        api_key=api_key,
        requires_key={"openrouter"},
        requires_import={
            "local": "sentence_transformers",
            "openrouter": "litellm",
            "huggingface_api": "litellm",
        },
    )
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    target_dtype = np.dtype(dtype)

    model_safe = model.replace("/", "_")
    cache_file = (cache_dir / f"embeddings_{provider}_{model_safe}.npz") if cache_dir else None
    doc_fingerprint = _documents_fingerprint(documents)

    if cache_file and cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        cached = data["embeddings"]
        cached_n_docs = int(data["n_docs"]) if "n_docs" in data.files else None
        cached_fingerprint = str(data["doc_fingerprint"]) if "doc_fingerprint" in data.files else None
        cached_provider = str(data["provider"]) if "provider" in data.files else None
        cached_model = str(data["model"]) if "model" in data.files else None
        is_valid = (
            cached_n_docs == len(documents)
            and cached_fingerprint == doc_fingerprint
            and cached_provider == provider
            and cached_model == model
        )
        if is_valid:
            logger.info("  Loaded embeddings from cache: %s", cache_file.name)
            return cached.astype(target_dtype, copy=False)
        logger.warning(
            "  Embedding cache mismatch for %s. Recomputing "
            "(cached n_docs=%s, current n_docs=%s; cached provider/model=%s/%s, current=%s/%s).",
            cache_file.name,
            cached_n_docs,
            len(documents),
            cached_provider,
            cached_model,
            provider,
            model,
        )

    logger.info("  Computing embeddings with %s/%s ...", provider, model)

    if provider in {"openrouter", "huggingface_api"}:
        _assert_memory_budget_for_paid_embeddings(
            provider=provider,
            model=model,
            n_docs=len(documents),
            dtype=target_dtype,
        )

    if provider == "local":
        logger.info("  Local embedding runtime hint | cpu_count=%s", max(1, int(os.cpu_count() or 1)))
        emb = _embed_local(documents, model, batch_size, target_dtype)
    elif provider == "huggingface_api":
        emb = _embed_huggingface_api(
            documents,
            model,
            batch_size,
            target_dtype,
            cost_tracker=cost_tracker,
        )
    elif provider == "openrouter":
        embedder = OpenRouterEmbedder(
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_workers=max_workers,
            dtype=target_dtype,
        )
        emb = embedder.encode(documents, show_progress_bar=True)
        usage = embedder.usage
        if cost_tracker is not None and usage["total_tokens"] > 0:
            cost_usd, _ = resolve_openrouter_costs(
                usage.get("call_records", []),
                mode=openrouter_cost_mode,
                api_key=api_key,
                max_workers=max_workers,
                logger_obj=logger,
                total_label="OpenRouter cost",
            )
            cost_tracker.add(
                step="embeddings",
                provider="openrouter",
                model=model,
                prompt_tokens=usage["prompt_tokens"],
                total_tokens=usage["total_tokens"],
                cost_usd=cost_usd,
            )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            embeddings=emb,
            model=model,
            provider=provider,
            n_docs=len(documents),
            doc_fingerprint=doc_fingerprint,
            dtype=str(target_dtype),
        )
        logger.info("  Saved: %s", cache_file.name)

    logger.info("  Embeddings: %s", emb.shape)
    if cost_tracker is not None:
        cost_tracker.log_step_summary("embeddings")
    return emb


def _embed_local(
    documents: list[str],
    model: str,
    batch_size: int,
    dtype: Any,
) -> np.ndarray:
    """Embed documents with a local SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    logger.info("  Loading local model: %s", model)
    try:
        st = SentenceTransformer(model)
        emb = st.encode(documents, show_progress_bar=True, batch_size=batch_size)
    except Exception as exc:
        raise_with_local_hf_compat_hint(model=model, use_case="embeddings", exc=exc)
    return emb.astype(dtype)


def _embed_huggingface_api(
    documents: list[str],
    model: str,
    batch_size: int,
    dtype: Any,
    *,
    cost_tracker: "CostTracker | None" = None,
) -> np.ndarray:
    """Embed documents via LiteLLM against a HuggingFace API model."""
    import litellm

    out: np.ndarray | None = None
    total_prompt_tokens = 0
    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]
    for batch_index, batch in tqdm(
        enumerate(batches),
        total=len(batches),
        desc="Embedding (HF API)",
    ):

        def _on_retry(retry_index: int, max_retries: int, wait: float, exc: Exception) -> None:
            logger.warning(
                "  HF API embedding batch %s failed (%s: %s). Retry %s/%s in %.0fs ...",
                batch_index,
                type(exc).__name__,
                exc,
                retry_index,
                max_retries,
                wait,
            )

        def _request_batch() -> Any:
            return litellm.embedding(model=model, input=batch)

        try:
            resp = retry_call(
                _request_batch,
                max_retries=2,
                delay=1.0,
                backoff="linear",
                on_retry=_on_retry,
            )
        except Exception as exc:
            logger.warning(
                "  HF API embedding batch %s failed after 3 attempts: %s: %s",
                batch_index,
                type(exc).__name__,
                exc,
            )
            raise

        batch_embeddings = np.asarray([d["embedding"] for d in resp["data"]], dtype=dtype)
        if batch_embeddings.ndim != 2:
            raise RuntimeError("HF API embedding response must be a 2D array.")
        if batch_embeddings.shape[0] != len(batch):
            raise RuntimeError(
                f"HF API embedding batch size mismatch: expected {len(batch)}, got {batch_embeddings.shape[0]}."
            )
        if out is None:
            out = np.empty((len(documents), batch_embeddings.shape[1]), dtype=dtype)
        elif batch_embeddings.shape[1] != out.shape[1]:
            raise RuntimeError(
                "HF API embedding dimension mismatch across batches: "
                f"expected {out.shape[1]}, got {batch_embeddings.shape[1]}."
            )

        start = batch_index * batch_size
        end = start + len(batch)
        out[start:end] = batch_embeddings

        usage = getattr(resp, "usage", None) or resp.get("usage", {})
        total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or usage.get("prompt_tokens", 0)

    if cost_tracker is not None and total_prompt_tokens > 0:
        cost_tracker.add(
            step="embeddings",
            provider="huggingface_api",
            model=model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=0,
            cost_usd=None,
        )

    if out is None:
        return np.array([], dtype=dtype)
    return out


class OpenRouterEmbedder:
    """Unified OpenRouter embedding client with retry, concurrency, and usage tracking."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        batch_size: int = 64,
        max_workers: int = 5,
        dtype: Any = np.float32,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.batch_size = int(batch_size)
        self.max_workers = int(max_workers)
        self.dtype = dtype
        self.api_base = api_base
        self.reset_usage()

    def reset_usage(self) -> None:
        """Reset tracked usage counters."""
        self.usage = {"prompt_tokens": 0, "total_tokens": 0, "call_records": []}

    @staticmethod
    def _resolve_show_progress(*, verbose: bool | None, show_progress_bar: bool | None) -> bool:
        """Resolve progress-bar visibility from Toponymy-compatible flags."""
        if show_progress_bar is not None:
            return bool(show_progress_bar)
        if verbose is not None:
            return bool(verbose)
        return False

    @staticmethod
    def _extract_response_data(response: Any) -> Any:
        """Return embedding payload for dict- or object-like responses."""
        if isinstance(response, dict):
            return response.get("data")
        return getattr(response, "data", None)

    @staticmethod
    def _extract_embedding(item: Any) -> Any:
        """Extract one embedding vector from dict- or object-like payloads."""
        if isinstance(item, dict):
            return item.get("embedding")
        return getattr(item, "embedding", None)

    def encode(
        self,
        texts: list[str],
        verbose: bool | None = None,
        show_progress_bar: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Embed texts via OpenRouter using LiteLLM with retries and parallel workers."""
        del kwargs
        import litellm

        texts = list(texts)
        if not texts:
            return np.array([], dtype=self.dtype)

        model_name = self.model
        if not model_name.startswith("openrouter/"):
            model_name = f"openrouter/{model_name}"

        batch_size = max(1, self.batch_size)
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        batch_offsets: list[tuple[int, int]] = []
        offset = 0
        for batch in batches:
            start = offset
            offset += len(batch)
            batch_offsets.append((start, offset))
        worker_count = max(1, min(self.max_workers, len(batches)))
        show_progress = self._resolve_show_progress(verbose=verbose, show_progress_bar=show_progress_bar)

        def _embed_batch(batch_index: int, batch: list[str]) -> dict[str, Any]:
            """Embed one batch and return validated embeddings plus usage metadata."""

            def _on_retry(retry_index: int, max_retries: int, wait: float, exc: Exception) -> None:
                logger.warning(
                    "  OpenRouter embedding batch %s failed (%s: %s). Retry %s/%s in %.0fs ...",
                    batch_index,
                    type(exc).__name__,
                    exc,
                    retry_index,
                    max_retries,
                    wait,
                )

            def _request_batch() -> tuple[Any, list[Any]]:
                request_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "input": batch,
                    "api_key": self.api_key,
                }
                if self.api_base:
                    request_kwargs["api_base"] = self.api_base
                response = litellm.embedding(**request_kwargs)

                data = self._extract_response_data(response)
                if data is None:
                    raise RuntimeError("OpenRouter embedding response had data=None.")

                embeddings = [self._extract_embedding(item) for item in data]
                if any(embedding is None for embedding in embeddings):
                    raise RuntimeError("OpenRouter embedding response contained missing embedding vectors.")
                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"OpenRouter embedding batch size mismatch: expected {len(batch)}, got {len(embeddings)}."
                    )
                return response, embeddings

            try:
                response, embeddings = retry_call(
                    _request_batch,
                    max_retries=2,
                    delay=1.0,
                    backoff="linear",
                    on_retry=_on_retry,
                )
            except Exception as exc:
                logger.warning(
                    "  OpenRouter embedding batch %s failed after 3 attempts: %s: %s",
                    batch_index,
                    type(exc).__name__,
                    exc,
                )
                raise

            usage = extract_usage_stats(response)
            return {
                "batch_index": batch_index,
                "embeddings": embeddings,
                "prompt_tokens": usage["prompt_tokens"],
                "total_tokens": usage["total_tokens"],
                "call_record": {
                    "generation_id": extract_generation_id(response),
                    "direct_cost": extract_response_cost(response=response),
                },
            }

        out: np.ndarray | None = None
        usage_by_batch: dict[int, dict[str, Any]] = {}

        def _store_result(result: dict[str, Any]) -> None:
            nonlocal out
            batch_index = int(result["batch_index"])
            start, end = batch_offsets[batch_index]
            batch_embeddings = np.asarray(result["embeddings"], dtype=self.dtype)
            if batch_embeddings.ndim != 2:
                raise RuntimeError("OpenRouter embedding batch must be a 2D array.")
            expected_rows = end - start
            if batch_embeddings.shape[0] != expected_rows:
                raise RuntimeError(
                    "OpenRouter embedding batch size mismatch: "
                    f"expected {expected_rows}, got {batch_embeddings.shape[0]}."
                )
            if out is None:
                out = np.empty((len(texts), batch_embeddings.shape[1]), dtype=self.dtype)
            elif batch_embeddings.shape[1] != out.shape[1]:
                raise RuntimeError(
                    "OpenRouter embedding dimension mismatch across batches: "
                    f"expected {out.shape[1]}, got {batch_embeddings.shape[1]}."
                )
            out[start:end] = batch_embeddings
            usage_by_batch[batch_index] = {
                "prompt_tokens": int(result["prompt_tokens"]),
                "total_tokens": int(result["total_tokens"]),
                "call_record": result["call_record"],
            }

        desc = "Embedding (OpenRouter)"
        if worker_count == 1:
            for batch_index, batch in tqdm(
                enumerate(batches),
                total=len(batches),
                desc=desc,
                disable=not show_progress,
            ):
                result = _embed_batch(batch_index, batch)
                _store_result(result)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(_embed_batch, batch_index, batch) for batch_index, batch in enumerate(batches)]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=desc,
                    disable=not show_progress,
                ):
                    result = future.result()
                    _store_result(result)

        prompt_tokens = 0
        total_tokens = 0
        call_records: list[dict[str, Any]] = []
        for batch_index in range(len(batches)):
            usage = usage_by_batch[batch_index]
            prompt_tokens += usage["prompt_tokens"]
            total_tokens += usage["total_tokens"]
            call_records.append(usage["call_record"])

        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["total_tokens"] += total_tokens
        self.usage["call_records"].extend(call_records)
        if out is None:
            return np.array([], dtype=self.dtype)
        return out


__all__ = ["compute_embeddings", "OpenRouterEmbedder"]
