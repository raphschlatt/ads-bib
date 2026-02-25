"""Embedding providers and caching for topic modeling."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ads_bib._utils.ads_api import retry_call
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


def _documents_fingerprint(documents: list[str]) -> str:
    """Return deterministic SHA-256 over document sequence."""
    hasher = hashlib.sha256()
    for doc in documents:
        payload = str(doc).encode(_DOC_FINGERPRINT_ENCODING, errors="replace")
        hasher.update(payload)
        hasher.update(_DOC_FINGERPRINT_SEPARATOR.encode(_DOC_FINGERPRINT_ENCODING))
    return hasher.hexdigest()


def compute_embeddings(
    documents: list[str],
    *,
    provider: str,
    model: str,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    max_workers: int = 5,
    dtype=np.float32,
    api_key: str | None = None,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
) -> np.ndarray:
    """Compute or load cached document embeddings."""
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
            return cached
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

    if provider == "local":
        emb = _embed_local(documents, model, batch_size, dtype)
    elif provider == "huggingface_api":
        emb = _embed_huggingface_api(
            documents,
            model,
            batch_size,
            dtype,
            cost_tracker=cost_tracker,
        )
    elif provider == "openrouter":
        embedder = OpenRouterEmbedder(
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_workers=max_workers,
            dtype=dtype,
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
    st = SentenceTransformer(model)
    emb = st.encode(documents, show_progress_bar=True, batch_size=batch_size)
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

    all_emb: list[Any] = []
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

        all_emb.extend(d["embedding"] for d in resp["data"])
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

    return np.array(all_emb, dtype=dtype)


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

        batch_results: dict[int, dict[str, Any]] = {}
        desc = "Embedding (OpenRouter)"
        if worker_count == 1:
            for batch_index, batch in tqdm(
                enumerate(batches),
                total=len(batches),
                desc=desc,
                disable=not show_progress,
            ):
                result = _embed_batch(batch_index, batch)
                batch_results[result["batch_index"]] = result
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
                    batch_results[result["batch_index"]] = result

        all_embeddings: list[Any] = []
        prompt_tokens = 0
        total_tokens = 0
        call_records: list[dict[str, Any]] = []
        for batch_index in range(len(batches)):
            result = batch_results[batch_index]
            all_embeddings.extend(result["embeddings"])
            prompt_tokens += int(result["prompt_tokens"])
            total_tokens += int(result["total_tokens"])
            call_records.append(result["call_record"])

        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["total_tokens"] += total_tokens
        self.usage["call_records"].extend(call_records)
        return np.array(all_embeddings, dtype=self.dtype)


__all__ = ["compute_embeddings", "OpenRouterEmbedder"]
