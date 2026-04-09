"""Embedding providers and caching for topic modeling."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ads_bib._utils.ads_api import retry_call
from ads_bib._utils.huggingface_api import (
    normalize_huggingface_model,
    resolve_huggingface_api_key,
)
from ads_bib._utils.hf_compat import raise_with_local_hf_compat_hint
from ads_bib._utils.logging import (
    capture_external_output,
    get_console_stream,
    get_runtime_log_path,
)
from ads_bib._utils.openrouter_costs import (
    extract_generation_id,
    extract_response_cost,
    extract_usage_stats,
    normalize_openrouter_cost_mode,
    resolve_openrouter_costs,
)
from ads_bib.config import validate_provider
from ads_bib.topic_model._runtime import EMBEDDING_PROVIDER_IMPORTS, EMBEDDING_PROVIDERS

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
_MAX_OPENROUTER_WORKERS = 20


def _resolve_sentence_transformer_device_label(model: Any) -> str:
    """Return a compact runtime-device label for one SentenceTransformer."""
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    module = getattr(model, "_first_module", lambda: None)()
    if module is not None:
        auto_model = getattr(module, "auto_model", None)
        if auto_model is not None:
            try:
                return str(next(auto_model.parameters()).device)
            except (AttributeError, StopIteration, TypeError):
                pass
    return "unknown"



def _local_progress_bar(*, total: int, show_progress: bool):
    """Return a repo-owned local embedding progress bar or a no-op context."""
    if not show_progress or total <= 0:
        return nullcontext(None)

    console_stream = get_console_stream()
    if console_stream is None:
        return nullcontext(None)
    return tqdm(
        total=total,
        desc="Embedding (local)",
        leave=True,
        file=console_stream,
        dynamic_ncols=False,
        ncols=78,
        bar_format="{desc:<18}{percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


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


def _resolve_show_progress(*, verbose: bool | None, show_progress_bar: bool | None) -> bool:
    """Resolve progress-bar visibility from Toponymy-compatible flags."""
    if show_progress_bar is not None:
        return bool(show_progress_bar)
    if verbose is not None:
        return bool(verbose)
    return False


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
    show_progress: bool = True,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Compute document embeddings with optional cache reuse.

    Parameters
    ----------
    documents : list[str]
        Ordered document texts to embed.
    provider : str
        Embedding backend: ``"local"`` (HF / sentence-transformers on CPU or
        GPU), ``"huggingface_api"``, or ``"openrouter"``.
    model : str
        Provider-specific embedding model identifier.
    cache_dir : Path, optional
        Cache directory. When set, embeddings are loaded from/saved to
        ``embeddings_{provider}_{model}.npz`` with fingerprint validation.
    batch_size : int
        Per-call batch size for embedding requests.
    max_workers : int
        Worker count used by concurrent remote providers.
    dtype : Any
        Target numpy dtype for returned array.
    api_key : str, optional
        Required for ``provider="openrouter"`` and ``provider="huggingface_api"``.
    openrouter_cost_mode : str
        Cost resolution mode for OpenRouter (``"hybrid"``, ``"strict"``,
        ``"fast"``).
    cost_tracker : CostTracker, optional
        When provided, records embedding token/cost summaries.
    show_progress : bool
        Show provider-native progress bars when no *progress_callback* is set.
    progress_callback : callable, optional
        Callback receiving completed document counts. When provided, internal
        provider progress bars stay hidden so the frontend can render a single
        stage-level bar.

    Returns
    -------
    np.ndarray
        Embedding matrix with shape ``(n_documents, embedding_dim)``.
    """
    if provider == "huggingface_api":
        model = normalize_huggingface_model(model)
        api_key = resolve_huggingface_api_key(api_key)

    validate_provider(
        provider,
        valid=set(EMBEDDING_PROVIDERS),
        api_key=api_key,
        requires_key={"openrouter", "huggingface_api"},
        requires_import=EMBEDDING_PROVIDER_IMPORTS,
    )
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    target_dtype = np.dtype(dtype)
    internal_show_progress = bool(show_progress) and progress_callback is None

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
            if progress_callback is not None and len(documents) > 0:
                progress_callback(len(documents))
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
        emb = _embed_local(
            documents,
            model,
            batch_size,
            target_dtype,
            show_progress=internal_show_progress,
            progress_callback=progress_callback,
        )
    elif provider == "huggingface_api":
        emb = _embed_huggingface_api(
            documents,
            model,
            batch_size,
            target_dtype,
            max_workers=max_workers,
            show_progress=internal_show_progress,
            progress_callback=progress_callback,
            api_key=api_key,
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
        emb = embedder.encode(
            documents,
            show_progress_bar=internal_show_progress,
            progress_callback=progress_callback,
        )
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
        payload: dict[str, Any] = {
            "embeddings": emb,
            "model": model,
            "provider": provider,
            "n_docs": len(documents),
            "doc_fingerprint": doc_fingerprint,
            "dtype": str(target_dtype),
        }
        np.savez_compressed(cache_file, **payload)
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
    *,
    show_progress: bool = True,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Embed documents with a local SentenceTransformer model."""
    logger.info("  Loading local model: %s", model)
    chunk_size = max(1, int(batch_size))
    total_documents = len(documents)
    if total_documents == 0:
        return np.array([], dtype=dtype)

    try:
        with _local_progress_bar(
            total=total_documents,
            show_progress=bool(show_progress) and progress_callback is None,
        ) as pbar:
            with capture_external_output(get_runtime_log_path()):
                from sentence_transformers import SentenceTransformer

                st = SentenceTransformer(model)
                logger.info(
                    "  Local embedding device | model=%s | device=%s",
                    model,
                    _resolve_sentence_transformer_device_label(st),
                )
                batches: list[np.ndarray] = []
                for start in range(0, total_documents, chunk_size):
                    batch = documents[start : start + chunk_size]
                    batch_embeddings = np.asarray(
                        st.encode(
                            batch,
                            show_progress_bar=False,
                            batch_size=chunk_size,
                        ),
                        dtype=dtype,
                    )
                    if batch_embeddings.ndim != 2:
                        raise RuntimeError("Local embedding batch must be a 2D array.")
                    if batch_embeddings.shape[0] != len(batch):
                        raise RuntimeError(
                            "Local embedding batch size mismatch: "
                            f"expected {len(batch)}, got {batch_embeddings.shape[0]}."
                        )
                    batches.append(batch_embeddings)
                    if progress_callback is not None:
                        progress_callback(len(batch))
                    elif pbar is not None:
                        pbar.update(len(batch))
        emb = np.concatenate(batches, axis=0)
    except Exception as exc:
        raise_with_local_hf_compat_hint(model=model, use_case="embeddings", exc=exc)
    return emb.astype(dtype)


def _create_huggingface_async_client(
    *,
    model: str,
    api_key: str | None,
):
    """Create a native HF async client from one normalized public model id."""
    from huggingface_hub import AsyncInferenceClient

    normalized_model = normalize_huggingface_model(model)
    model_id, provider = (
        normalized_model.rsplit(":", 1) if ":" in normalized_model else (normalized_model, None)
    )
    return (
        AsyncInferenceClient(
            provider=provider,
            api_key=resolve_huggingface_api_key(api_key),
        ),
        model_id,
    )


def _run_huggingface_async(awaitable_factory):
    """Run one HF async workload from sync code, including notebook cells."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable_factory())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(awaitable_factory())).result()


def _embed_huggingface_api(
    documents: list[str],
    model: str,
    batch_size: int,
    dtype: Any,
    *,
    max_workers: int = 5,
    show_progress: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    api_key: str | None = None,
    cost_tracker: "CostTracker | None" = None,
) -> np.ndarray:
    """Embed documents via the native HF async inference client."""
    out: np.ndarray | None = None
    total_prompt_tokens = 0
    client, model_id = _create_huggingface_async_client(model=model, api_key=api_key)
    batches = [
        (batch_index, documents[start : start + batch_size])
        for batch_index, start in enumerate(range(0, len(documents), batch_size))
    ]
    progress = tqdm(
        total=len(documents),
        desc="Embedding (HF API)",
        disable=(not show_progress) or (progress_callback is not None),
    )

    async def _embed_all() -> list[tuple[int, np.ndarray, int]]:
        semaphore = asyncio.Semaphore(max(1, int(max_workers)))
        results: list[tuple[int, np.ndarray, int] | None] = [None] * len(batches)

        async def _embed_one_batch(result_index: int, item: tuple[int, list[str]]) -> None:
            batch_index, batch = item
            async with semaphore:
                try:
                    for attempt in range(3):
                        try:
                            response = await client.feature_extraction(batch, model=model_id)
                            break
                        except Exception as exc:
                            if attempt >= 2:
                                logger.warning(
                                    "  HF API embedding batch %s failed after 3 attempts: %s: %s",
                                    batch_index,
                                    type(exc).__name__,
                                    exc,
                                )
                                raise
                            wait = float(attempt + 1)
                            logger.warning(
                                "  HF API embedding batch %s failed (%s: %s). Retry %s/2 in %.0fs ...",
                                batch_index,
                                type(exc).__name__,
                                exc,
                                attempt + 1,
                                wait,
                            )
                            await asyncio.sleep(wait)

                    batch_embeddings = np.asarray(response, dtype=dtype)
                    if batch_embeddings.ndim != 2:
                        raise RuntimeError("HF API embedding response must be a 2D array.")
                    if batch_embeddings.shape[0] != len(batch):
                        raise RuntimeError(
                            "HF API embedding batch size mismatch: "
                            f"expected {len(batch)}, got {batch_embeddings.shape[0]}."
                        )
                    results[result_index] = (batch_index, batch_embeddings, len(batch))
                finally:
                    if progress_callback is not None:
                        progress_callback(len(batch))
                    else:
                        progress.update(len(batch))

        await asyncio.gather(
            *(_embed_one_batch(result_index, item) for result_index, item in enumerate(batches))
        )
        return [result for result in results if result is not None]

    try:
        results = _run_huggingface_async(_embed_all)
    finally:
        progress.close()

    for batch_index, batch_embeddings, batch_len in results:
        if out is None:
            out = np.empty((len(documents), batch_embeddings.shape[1]), dtype=dtype)
        elif batch_embeddings.shape[1] != out.shape[1]:
            raise RuntimeError(
                "HF API embedding dimension mismatch across batches: "
                f"expected {out.shape[1]}, got {batch_embeddings.shape[1]}."
            )
        start = batch_index * batch_size
        end = start + batch_len
        out[start:end] = batch_embeddings
        total_prompt_tokens += batch_len

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
        progress_callback: Callable[[int], None] | None = None,
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
        effective_max = min(self.max_workers, _MAX_OPENROUTER_WORKERS)
        if self.max_workers > _MAX_OPENROUTER_WORKERS:
            logger.warning(
                "  max_workers=%s exceeds cap of %s for OpenRouter; using %s to avoid socket exhaustion.",
                self.max_workers,
                _MAX_OPENROUTER_WORKERS,
                effective_max,
            )
        worker_count = max(1, min(effective_max, len(batches)))
        show_progress = (
            _resolve_show_progress(verbose=verbose, show_progress_bar=show_progress_bar)
            and progress_callback is None
        )

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
                    max_retries=3,
                    delay=2.0,
                    backoff="exponential",
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
            if progress_callback is not None:
                progress_callback(expected_rows)
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


class HuggingFaceAPIEmbedder:
    """Unified HF API embedder with a Toponymy-compatible ``encode`` method."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        batch_size: int = 64,
        max_workers: int = 5,
        dtype: Any = np.float32,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.batch_size = int(batch_size)
        self.max_workers = int(max_workers)
        self.dtype = dtype
        self.usage = {"prompt_tokens": 0, "total_tokens": 0}

    def encode(
        self,
        texts: list[str],
        verbose: bool | None = None,
        show_progress_bar: bool | None = None,
        progress_callback: Callable[[int], None] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        embeddings = _embed_huggingface_api(
            texts,
            self.model,
            self.batch_size,
            self.dtype,
            max_workers=self.max_workers,
            show_progress=_resolve_show_progress(verbose=verbose, show_progress_bar=show_progress_bar),
            progress_callback=progress_callback,
            api_key=self.api_key,
            cost_tracker=None,
        )
        self.usage["prompt_tokens"] += len(texts)
        self.usage["total_tokens"] += len(texts)
        return embeddings


__all__ = ["compute_embeddings", "HuggingFaceAPIEmbedder", "OpenRouterEmbedder"]
