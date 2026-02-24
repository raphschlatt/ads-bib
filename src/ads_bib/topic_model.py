"""Step 5 – Topic modeling backends (BERTopic and Toponymy).

Covers embedding, dimensionality reduction, clustering, and LLM-based
topic labeling with three interchangeable providers each for embeddings
and LLM labeling.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import inspect
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ads_bib._utils.ads_api import retry_call
from ads_bib._utils.openrouter_client import (
    openrouter_chat_completion,
    openrouter_usage_from_response,
)
from ads_bib._utils.openrouter_costs import (
    DEFAULT_OPENROUTER_API_BASE,
    extract_generation_id,
    extract_response_cost,
    extract_usage_stats,
    normalize_openrouter_api_base,
    normalize_openrouter_cost_mode,
    summarize_openrouter_costs,
)

logger = logging.getLogger(__name__)


def _suppress_noisy_third_party_logs() -> None:
    """Suppress repetitive third-party transport logs while keeping pipeline logs."""
    for noisy_logger_name in ("httpx", "httpcore", "LiteLLM", "litellm"):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    os.environ.setdefault("LITELLM_LOG", "WARNING")
    os.environ.setdefault("LITELLM_VERBOSE", "False")

    try:
        import litellm
    except Exception:
        return

    set_verbose_attr = getattr(litellm, "set_verbose", None)
    if callable(set_verbose_attr):
        try:
            set_verbose_attr(False)
        except Exception:
            pass
    elif set_verbose_attr is not None:
        try:
            setattr(litellm, "set_verbose", False)
        except Exception:
            pass

    if hasattr(litellm, "suppress_debug_info"):
        try:
            litellm.suppress_debug_info = True
        except Exception:
            pass


_suppress_noisy_third_party_logs()

# ---------------------------------------------------------------------------
# Module defaults
# ---------------------------------------------------------------------------

DEFAULT_DIM_REDUCTION_RANDOM_STATE = 42
# PaCMAP in this pipeline uses denser neighborhood sampling than many defaults
# to stabilize cluster geometry on large corpora.
DEFAULT_PACMAP_N_NEIGHBORS = 60
# UMAP uses a broader neighborhood for smoother global topic separation.
DEFAULT_UMAP_N_NEIGHBORS = 80
DEFAULT_CLUSTER_MIN_SIZE = 180
DEFAULT_BERTOPIC_TOP_N_WORDS = 20
DEFAULT_POS_SPACY_MODEL = "en_core_web_sm"
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


def _stable_hash(payload: dict[str, Any]) -> str:
    """Return deterministic SHA-256 hash for JSON-serializable payloads."""
    packed = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _array_fingerprint(values: np.ndarray) -> str:
    """Return deterministic SHA-256 fingerprint for a numpy array."""
    arr = np.ascontiguousarray(values)
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(memoryview(arr).cast("B"))
    return hasher.hexdigest()

# ---------------------------------------------------------------------------
# OpenRouter cost lookup
# ---------------------------------------------------------------------------

def _fetch_openrouter_costs(
    call_records: list[dict],
    api_key: str | None,
    *,
    openrouter_cost_mode: str = "hybrid",
    max_workers: int = 5,
) -> float | None:
    """Aggregate OpenRouter costs using direct usage data with optional generation fallback."""
    if not call_records:
        return None

    summary = summarize_openrouter_costs(
        call_records,
        mode=openrouter_cost_mode,
        api_key=api_key,
        max_workers=max_workers,
        retries=2,
        delay=0.5,
        wait_before_fetch=2.0,
    )
    total = summary["total_cost_usd"]
    if total is not None:
        logger.info(
            "  OpenRouter cost: $%.4f (%s/%s priced; direct=%s, fetched=%s, mode=%s)",
            total,
            summary["priced_calls"],
            summary["total_calls"],
            summary["direct_priced_calls"],
            summary["fetched_priced_calls"],
            summary["mode"],
        )
    return total


# ---------------------------------------------------------------------------
# Embeddings (3 providers)
# ---------------------------------------------------------------------------

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
    """Compute or load cached document embeddings.

    Parameters
    ----------
    provider : str
        ``"local"``, ``"huggingface_api"``, or ``"openrouter"``.
    model : str
        Model identifier (provider-specific).
    cache_dir : Path, optional
        If given, embeddings are cached to ``{cache_dir}/embeddings_{provider}_{model}.npz``.
    openrouter_cost_mode : str
        ``"hybrid"`` (default), ``"strict"``, or ``"fast"``.
    """
    from .config import validate_provider
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

    # Check legacy cache (without provider prefix)
    if cache_file and not cache_file.exists():
        legacy = cache_dir / f"embeddings_{model_safe}.npz"
        if legacy.exists():
            logger.info("  Using legacy cache: %s", legacy.name)
            cache_file = legacy

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
            documents, model, batch_size, dtype, cost_tracker=cost_tracker,
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
            cost_usd = _fetch_openrouter_costs(
                usage.get("call_records", []),
                api_key,
                openrouter_cost_mode=openrouter_cost_mode,
                max_workers=max_workers,
            )
            cost_tracker.add(
                step="embeddings", provider="openrouter", model=model,
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

    all_emb = []
    total_prompt_tokens = 0
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
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
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        worker_count = max(1, min(self.max_workers, len(batches)))
        show_progress = self._resolve_show_progress(
            verbose=verbose,
            show_progress_bar=show_progress_bar,
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
                futures = [
                    pool.submit(_embed_batch, batch_index, batch)
                    for batch_index, batch in enumerate(batches)
                ]
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


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    *,
    method: str = "pacmap",
    params_5d: dict | None = None,
    params_2d: dict | None = None,
    random_state: int = DEFAULT_DIM_REDUCTION_RANDOM_STATE,
    cache_dir: Path | None = None,
    cache_suffix: str = "",
    embedding_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 5-D (for clustering) and 2-D (for visualisation) projections.

    Parameters
    ----------
    method : str
        ``"pacmap"`` or ``"umap"``.
    params_5d, params_2d : dict, optional
        Override default parameters for each projection.
    random_state : int
        Shared RNG seed used by PaCMAP/UMAP unless explicitly overridden in
        ``params_5d`` or ``params_2d``.
    cache_dir : Path, optional
        Directory for ``.npz`` caches.
    cache_suffix : str
        Appended to cache filenames for uniqueness.
    embedding_id : str, optional
        Provider/model identifier (e.g. ``"openrouter/model-name"``).
        When *cache_suffix* is empty and *embedding_id* is given, a suffix is
        built automatically from the embedding id, method, and 5-D params.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(reduced_5d, reduced_2d)``.
    """
    if not cache_suffix and embedding_id:
        p5 = params_5d or {}
        cache_suffix = (
            f"{embedding_id.replace('/', '_')}_{method}"
            f"_nn{p5.get('n_neighbors', 'def')}"
            f"_mind{p5.get('min_dist', 'def')}"
            f"_metric{p5.get('metric', 'def')}"
            f"_rs{random_state}"
        )
    r5 = _reduce(
        embeddings,
        5,
        method,
        params_5d or {},
        random_state,
        cache_dir,
        f"5d_{cache_suffix}",
    )
    r2 = _reduce(
        embeddings,
        2,
        method,
        params_2d or {},
        random_state,
        cache_dir,
        f"2d_{cache_suffix}",
    )
    logger.info("  5D shape: %s, 2D shape: %s", r5.shape, r2.shape)
    return r5, r2


def _reduce(
    embeddings: np.ndarray,
    n_components: int,
    method: str,
    params: dict[str, Any],
    random_state: int,
    cache_dir: Path | None,
    name: str,
) -> np.ndarray:
    """Reduce embedding dimensionality with optional on-disk caching."""
    meta_payload = {
        "method": method,
        "n_components": int(n_components),
        "random_state": int(random_state),
        "params": params,
    }
    params_hash = _stable_hash(meta_payload)
    n_docs = int(len(embeddings))
    embedding_fingerprint = _array_fingerprint(embeddings)

    if cache_dir:
        path = cache_dir / f"reduced_{name}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            reduced = data["reduced"] if "reduced" in data.files else None
            cached_n_docs = int(data["n_docs"]) if "n_docs" in data.files else None
            cached_embedding_fingerprint = (
                str(data["embedding_fingerprint"])
                if "embedding_fingerprint" in data.files
                else None
            )
            cached_method = str(data["method"]) if "method" in data.files else None
            cached_n_components = int(data["n_components"]) if "n_components" in data.files else None
            cached_random_state = int(data["random_state"]) if "random_state" in data.files else None
            cached_params_hash = str(data["params_hash"]) if "params_hash" in data.files else None
            is_valid = (
                reduced is not None
                and cached_n_docs == n_docs
                and cached_embedding_fingerprint == embedding_fingerprint
                and cached_method == method
                and cached_n_components == int(n_components)
                and cached_random_state == int(random_state)
                and cached_params_hash == params_hash
            )
            if is_valid:
                logger.info("  Loaded %s from cache", name)
                return reduced
            logger.warning(
                "  Reduction cache mismatch for %s. Recomputing.",
                path.name,
            )

    logger.info("  Computing %s with %s ...", name, method.upper())
    if method == "pacmap":
        import pacmap

        normalized_params = dict(params)
        metric = normalized_params.pop("metric", None)
        if metric is not None and "distance" not in normalized_params:
            normalized_params["distance"] = metric

        if normalized_params.get("distance") == "cosine":
            normalized_params["distance"] = "angular"
            warnings.warn(
                "PaCMAP uses 'distance' and does not support 'cosine'; using 'angular' instead.",
                UserWarning,
                stacklevel=2,
            )

        if "min_dist" in normalized_params:
            normalized_params.pop("min_dist", None)
            warnings.warn(
                "PaCMAP does not support 'min_dist'; parameter ignored.",
                UserWarning,
                stacklevel=2,
            )

        defaults = dict(
            n_components=n_components,
            n_neighbors=DEFAULT_PACMAP_N_NEIGHBORS,
            MN_ratio=0.5,
            FP_ratio=1.0,
            random_state=random_state,
            verbose=False,
        )
        defaults.update(normalized_params)
        defaults["n_components"] = n_components
        model = pacmap.PaCMAP(**defaults)
    elif method == "umap":
        from umap import UMAP
        defaults = dict(
            n_components=n_components,
            n_neighbors=DEFAULT_UMAP_N_NEIGHBORS,
            min_dist=0.05,
            metric="cosine",
            random_state=random_state,
            verbose=False,
        )
        defaults.update(params)
        defaults["n_components"] = n_components
        model = UMAP(**defaults)
    else:
        raise ValueError(f"Unknown dim reduction method: {method}")

    reduced = model.fit_transform(embeddings)

    if cache_dir:
        path = cache_dir / f"reduced_{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            reduced=reduced,
            n_docs=n_docs,
            embedding_fingerprint=embedding_fingerprint,
            method=method,
            n_components=int(n_components),
            random_state=int(random_state),
            params_hash=params_hash,
        )
        logger.info("  Saved: %s", path.name)

    return reduced


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _create_cluster_model(method: str, params: dict | None = None):
    """Build the configured clustering model once for all topic workflows.

    Notes
    -----
    Default ``min_cluster_size`` is ``DEFAULT_CLUSTER_MIN_SIZE`` (180) and can
    be overridden via ``params``.
    """
    params = params or {}

    if method == "fast_hdbscan":
        import fast_hdbscan

        defaults = dict(
            min_cluster_size=DEFAULT_CLUSTER_MIN_SIZE,
            min_samples=3,
            cluster_selection_method="eom",
            cluster_selection_epsilon=0.02,
            allow_single_cluster=False,
        )
        defaults.update(params)
        return fast_hdbscan.HDBSCAN(**defaults)
    if method == "hdbscan":
        from hdbscan import HDBSCAN

        defaults = dict(
            min_cluster_size=DEFAULT_CLUSTER_MIN_SIZE,
            min_samples=3,
            metric="euclidean",
            cluster_selection_method="eom",
            cluster_selection_epsilon=0.02,
            prediction_data=True,
            gen_min_span_tree=True,
        )
        defaults.update(params)
        return HDBSCAN(**defaults)

    raise ValueError(f"Unknown clustering method: {method}")


def cluster_documents(
    reduced_5d: np.ndarray,
    *,
    method: str = "fast_hdbscan",
    params: dict | None = None,
) -> np.ndarray:
    """Cluster documents in the 5-D embedding space.

    Parameters
    ----------
    method : str
        ``"fast_hdbscan"`` or ``"hdbscan"``.
    params : dict, optional
        HDBSCAN parameters (``min_cluster_size``, ``min_samples``, etc.).

    Returns
    -------
    np.ndarray
        Cluster labels (``-1`` = outlier).
    """
    model = _create_cluster_model(method, params)
    clusters = model.fit_predict(reduced_5d)
    n_topics = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_outliers = (clusters == -1).sum()
    logger.info(
        "  %s: %s topics, %s outliers (%.1f%%)",
        method.upper(),
        n_topics,
        f"{n_outliers:,}",
        n_outliers / len(clusters) * 100,
    )
    return clusters


# ---------------------------------------------------------------------------
# BERTopic fitting
# ---------------------------------------------------------------------------

def fit_bertopic(
    documents: list[str],
    reduced_5d: np.ndarray,
    *,
    llm_provider: str = "local",
    llm_model: str = "google/gemma-3-1b-it",
    llm_prompt: str | None = None,
    pipeline_models: list[str] | None = None,
    parallel_models: list[str] | None = None,
    mmr_diversity: float = 0.3,
    llm_nr_docs: int = 8,
    llm_diversity: float = 0.2,
    llm_delay: float = 0.3,
    embedding_model_name: str | None = None,
    keybert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_df: int = 2,
    clustering_method: str = "fast_hdbscan",
    clustering_params: dict | None = None,
    top_n_words: int = DEFAULT_BERTOPIC_TOP_N_WORDS,
    pos_spacy_model: str = DEFAULT_POS_SPACY_MODEL,
    api_key: str | None = None,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
) -> "BERTopic":
    """Fit a BERTopic model with pre-computed reduced embeddings.

    Parameters
    ----------
    llm_provider : str
        ``"local"``, ``"huggingface_api"``, or ``"openrouter"``.
    pipeline_models : list[str], optional
        Models to run *before* the LLM in the main representation pipeline
        (e.g. ``["POS", "KeyBERT", "MMR"]``).
    parallel_models : list[str], optional
        Models stored separately in ``topic_aspects_`` for comparison
        (e.g. ``["MMR", "POS", "KeyBERT"]``).
    top_n_words : int
        Number of top words BERTopic stores per topic.
    pos_spacy_model : str
        spaCy model name for ``PartOfSpeech`` representation when ``"POS"``
        is active in pipeline or parallel models.

    Returns
    -------
    BERTopic
        Fitted topic model.
    """
    from .config import validate_provider
    validate_provider(
        llm_provider,
        valid={"local", "huggingface_api", "openrouter"},
        api_key=api_key,
        requires_key={"openrouter"},
        requires_import={
            "local": "transformers",
            "openrouter": "litellm",
            "huggingface_api": "litellm",
        },
    )
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    pipeline_models = pipeline_models or ["POS", "KeyBERT", "MMR"]
    parallel_models = parallel_models or ["MMR", "POS", "KeyBERT"]
    logger.info(
        "Preparing BERTopic components (pipeline=%s, parallel=%s) ...",
        pipeline_models,
        parallel_models,
    )

    # Build representation model
    rep_model = _build_representation_model(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_prompt=llm_prompt,
        pipeline_models=pipeline_models,
        parallel_models=parallel_models,
        mmr_diversity=mmr_diversity,
        llm_nr_docs=llm_nr_docs,
        llm_diversity=llm_diversity,
        llm_delay=llm_delay,
        keybert_model=keybert_model,
        api_key=api_key,
        pos_spacy_model=pos_spacy_model,
    )

    vectorizer = CountVectorizer(stop_words="english", min_df=min_df, ngram_range=(1, 3))
    ctfidf = ClassTfidfTransformer()

    # Embedding model for KeyBERT / find_topics()
    emb_model = None
    if "KeyBERT" in pipeline_models or "KeyBERT" in parallel_models:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer(keybert_model)
    elif embedding_model_name:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer(embedding_model_name)

    cluster_model = _create_cluster_model(clustering_method, clustering_params)

    topic_model = BERTopic(
        embedding_model=emb_model,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        representation_model=rep_model,
        top_n_words=top_n_words,
        verbose=True,
    )

    logger.info("Fitting BERTopic (LLM: %s/%s) ...", llm_provider, llm_model)

    track_llm_usage = cost_tracker is not None and llm_provider in ("openrouter", "huggingface_api")
    with _track_litellm_usage(enabled=track_llm_usage) as llm_usage:
        topic_model.fit_transform(documents, reduced_5d)

    _record_llm_usage(
        llm_usage,
        step="llm_labeling",
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
        openrouter_cost_mode=openrouter_cost_mode,
        cost_tracker=cost_tracker,
    )

    return topic_model


@contextmanager
def _track_litellm_usage(*, enabled: bool):
    """Capture LiteLLM usage for one operation via callback registration."""
    if not enabled:
        yield None
        return

    import litellm

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    def _cost_cb(
        kwargs: dict[str, Any],
        response: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Accumulate LiteLLM token usage and call metadata from callbacks."""
        del start_time, end_time
        stats = extract_usage_stats(response)
        usage["prompt_tokens"] += stats["prompt_tokens"]
        usage["completion_tokens"] += stats["completion_tokens"]
        usage["call_records"].append(
            {
                "generation_id": extract_generation_id(response),
                "direct_cost": extract_response_cost(kwargs=kwargs, response=response),
            }
        )

    litellm.success_callback.append(_cost_cb)
    try:
        yield usage
    finally:
        if _cost_cb in litellm.success_callback:
            litellm.success_callback.remove(_cost_cb)


def _record_llm_usage(
    usage: dict | None,
    *,
    step: str,
    llm_provider: str,
    llm_model: str,
    api_key: str | None,
    openrouter_cost_mode: str,
    cost_tracker: "CostTracker | None",
) -> None:
    """Persist captured LLM usage into the shared cost tracker."""
    if usage is None or cost_tracker is None:
        return

    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    if prompt_tokens + completion_tokens == 0:
        return

    cost_usd = None
    if llm_provider == "openrouter":
        cost_usd = _fetch_openrouter_costs(
            usage["call_records"],
            api_key,
            openrouter_cost_mode=openrouter_cost_mode,
        )

    cost_tracker.add(
        step=step,
        provider=llm_provider,
        model=llm_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


def _create_tracked_toponymy_namer(
    *,
    model: str,
    api_key: str,
    base_url: str,
    max_workers: int = 5,
) -> tuple[Any, dict]:
    """Create a Toponymy-compatible async OpenRouter wrapper with usage tracking."""
    from openai import OpenAI
    from toponymy.llm_wrappers import AsyncLLMWrapper

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}
    worker_count = max(1, int(max_workers))

    class TrackedAsyncOpenRouterNamer(AsyncLLMWrapper):
        """Toponymy async namer wrapper with retry, concurrency, and usage capture."""

        def __init__(self) -> None:
            """Initialize OpenAI client bindings for Toponymy naming."""
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
            self.max_workers = worker_count
            self._semaphore = asyncio.Semaphore(self.max_workers)

        async def _call_single(
            self,
            *,
            messages: list[dict[str, str]],
            temperature: float,
            max_tokens: int,
        ) -> str:
            """Call one prompt with shared retry logic and collect usage."""
            try:
                async with self._semaphore:
                    response = await asyncio.to_thread(
                        openrouter_chat_completion,
                        client=self.client,
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        retry_label="Toponymy OpenRouter labeling call",
                    )
                stats = openrouter_usage_from_response(response)
                usage["prompt_tokens"] += int(stats["prompt_tokens"])
                usage["completion_tokens"] += int(stats["completion_tokens"])
                usage["call_records"].append(stats["call_record"])
                return response.choices[0].message.content or ""
            except Exception as exc:
                warnings.warn(
                    f"Toponymy OpenRouter labeling failed: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return ""

        async def _call_llm_batch(
            self,
            prompts: list[str],
            temperature: float,
            max_tokens: int,
        ) -> list[str]:
            """Process a batch of plain-text prompts concurrently."""
            tasks = [
                self._call_single(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: list[str],
            user_prompts: list[str],
            temperature: float,
            max_tokens: int,
        ) -> list[str]:
            """Process a batch of system+user prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError("Number of system prompts must match number of user prompts")

            tasks = [
                self._call_single(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for system_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

    return TrackedAsyncOpenRouterNamer(), usage


def _build_toponymy_topic_info(topics: np.ndarray, topic_names: list[str] | None) -> pd.DataFrame:
    """Build BERTopic-like topic info rows for Toponymy outputs."""
    topic_names = topic_names or []
    rows: list[dict[str, Any]] = []
    for topic_id in sorted(int(t) for t in np.unique(topics)):
        if topic_id == -1:
            name = "Outlier Topic"
        elif 0 <= topic_id < len(topic_names):
            name = topic_names[topic_id]
        else:
            name = f"Topic {topic_id}"
        rows.append({"Topic": topic_id, "Name": name, "Main": name})
    return pd.DataFrame(rows)


def _instantiate_with_filtered_kwargs(
    cls: Any,
    params: dict[str, Any] | None,
    *,
    component_name: str,
) -> Any:
    """Instantiate *cls* with kwargs filtered to supported ``__init__`` parameters.

    This keeps notebook-side parameter dicts resilient across Toponymy/EVoC
    version differences and logs dropped keys explicitly.
    """
    kwargs = dict(params or {})

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"{component_name}.__init__ failed with parameters "
                f"{sorted(kwargs.keys())}: {exc}"
            ) from exc

    params_meta = list(sig.parameters.items())
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for _, p in params_meta)
    if has_var_kwargs:
        return cls(**kwargs)

    accepted: set[str] = set()
    for name, param in params_meta:
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            accepted.add(name)

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(k for k in kwargs.keys() if k not in accepted)

    if dropped:
        warnings.warn(
            f"Dropping unsupported {component_name} parameter(s): "
            f"{', '.join(dropped)}",
            UserWarning,
            stacklevel=2,
        )

    try:
        return cls(**filtered)
    except TypeError as exc:
        raise TypeError(
            f"{component_name}.__init__ failed with filtered parameters "
            f"{sorted(filtered.keys())}: {exc}"
        ) from exc


def _patch_clusterer_for_toponymy_kwargs(clusterer: Any) -> None:
    """Make strict clusterer methods tolerant to extra kwargs from Toponymy.fit.

    Some Toponymy versions pass layer kwargs (e.g. ``exemplar_delimiters``,
    ``prompt_format``) into ``clusterer.fit_predict``. Older EVoCClusterer
    implementations do not accept these keywords and raise ``TypeError``.
    This shim drops unsupported kwargs while keeping supported ones intact.
    """

    def _wrap_bound(method: Any) -> Any:
        """Wrap bound methods to drop unsupported keyword arguments."""
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return method

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return method

        accepted = set(sig.parameters.keys())

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            """Forward call while filtering to accepted keyword arguments."""
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            return method(*args, **filtered)

        return _wrapped

    for attr in ("fit_predict", "fit"):
        bound = getattr(clusterer, attr, None)
        if bound is None:
            continue
        wrapped = _wrap_bound(bound)
        if wrapped is not bound:
            setattr(clusterer, attr, wrapped)


def _normalize_toponymy_inputs(
    *,
    llm_provider: str,
    backend: str,
    api_key: str | None,
) -> tuple[str, str]:
    """Normalize and validate Toponymy backend/provider selections."""
    provider_norm = llm_provider.strip().lower()
    if provider_norm not in {"openrouter", "local"}:
        raise ValueError(
            f"Invalid llm_provider '{llm_provider}'. Expected 'openrouter' or 'local'."
        )
    if provider_norm == "openrouter" and not api_key:
        raise ValueError("api_key is required for Toponymy with OpenRouter.")

    backend_norm = backend.strip().lower()
    if backend_norm not in {"toponymy", "toponymy_evoc"}:
        raise ValueError(
            f"Invalid backend '{backend}'. Expected 'toponymy' or 'toponymy_evoc'."
        )
    return provider_norm, backend_norm


def _build_toponymy_clusterer(
    *,
    backend_norm: str,
    clusterer_params: dict[str, Any],
    toponymy_clusterer_cls: Any,
    embeddings: np.ndarray,
    clusterable_vectors: np.ndarray,
) -> tuple[Any, np.ndarray]:
    """Create configured Toponymy/EVoC clusterer and vectors for fit."""
    if backend_norm == "toponymy":
        clusterer = _instantiate_with_filtered_kwargs(
            toponymy_clusterer_cls,
            clusterer_params,
            component_name="ToponymyClusterer",
        )
        logger.info("  Clusterer: %s", clusterer.__class__.__name__)
        return clusterer, clusterable_vectors

    try:
        from toponymy.clustering import EVoCClusterer
    except Exception as exc:
        raise ImportError(
            "backend='toponymy_evoc' requires optional dependency 'evoc' and "
            "a Toponymy version that exposes EVoCClusterer."
        ) from exc

    clusterer = _instantiate_with_filtered_kwargs(
        EVoCClusterer,
        clusterer_params,
        component_name="EVoCClusterer",
    )
    _patch_clusterer_for_toponymy_kwargs(clusterer)
    logger.info("  Using raw embeddings for clustering with EVoCClusterer.")
    logger.info("  Clusterer: %s", clusterer.__class__.__name__)
    return clusterer, embeddings


def _build_toponymy_models(
    *,
    provider_norm: str,
    llm_model: str,
    embedding_model: str,
    api_key: str | None,
    openrouter_api_base: str,
    max_workers: int,
    cost_tracker: "CostTracker | None",
) -> tuple[Any, dict[str, Any] | None, Any]:
    """Build Toponymy naming and text-embedding components."""
    if provider_norm == "openrouter":
        llm_wrapper, llm_usage = _create_tracked_toponymy_namer(
            model=llm_model,
            api_key=api_key,
            base_url=openrouter_api_base,
            max_workers=max_workers,
        )
        text_embedding_model = OpenRouterEmbedder(
            api_key=api_key,
            model=embedding_model,
            api_base=openrouter_api_base,
            max_workers=max_workers,
        )
        return llm_wrapper, llm_usage, text_embedding_model

    try:
        from sentence_transformers import SentenceTransformer
        from toponymy.llm_wrappers import HuggingFaceNamer
    except Exception as exc:
        raise ImportError(
            "llm_provider='local' requires optional dependencies "
            "'sentence-transformers', 'transformers', and "
            "Toponymy's HuggingFaceNamer wrapper."
        ) from exc

    llm_wrapper = HuggingFaceNamer(
        model=llm_model,
        device_map="auto",
        torch_dtype="auto",
    )
    if cost_tracker is not None:
        logger.info(
            "  Local Toponymy LLM selected; token/cost tracking is unavailable for this step."
        )
    return llm_wrapper, None, SentenceTransformer(embedding_model)


def _fit_and_extract_toponymy_outputs(
    *,
    toponymy_cls: Any,
    documents: list[str],
    embeddings: np.ndarray,
    clusterable_vectors: np.ndarray,
    llm_wrapper: Any,
    text_embedding_model: Any,
    clusterer: Any,
    object_description: str,
    corpus_description: str,
    verbose: bool,
    backend_norm: str,
    provider_norm: str,
    llm_model: str,
    layer_index: int,
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    """Fit Toponymy model and extract one configured topic layer."""
    logger.info(
        "Fitting Toponymy backend='%s' (LLM: %s/%s) ...",
        backend_norm,
        provider_norm,
        llm_model,
    )
    topic_model = toponymy_cls(
        llm_wrapper=llm_wrapper,
        text_embedding_model=text_embedding_model,
        clusterer=clusterer,
        object_description=object_description,
        corpus_description=corpus_description,
        verbose=verbose,
    )
    topic_model.fit(
        documents,
        embedding_vectors=embeddings,
        clusterable_vectors=clusterable_vectors,
    )

    n_layers = len(topic_model.cluster_layers_)
    if layer_index < 0 or layer_index >= n_layers:
        raise ValueError(
            f"layer_index {layer_index} is out of range for {n_layers} available layers."
        )

    topics = np.asarray(topic_model.cluster_layers_[layer_index].cluster_labels, dtype=int)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = int((topics == -1).sum())
    logger.info(
        "  Selected Toponymy layer %s/%s: %s topics, %s outliers.",
        layer_index,
        n_layers - 1,
        n_topics,
        f"{n_outliers:,}",
    )

    topic_names = topic_model.topic_names_[layer_index]
    topic_info = _build_toponymy_topic_info(topics, topic_names)
    return topic_model, topics, topic_info


def fit_toponymy(
    documents: list[str],
    embeddings: np.ndarray,
    clusterable_vectors: np.ndarray,
    *,
    backend: str = "toponymy",
    layer_index: int = 0,
    llm_provider: str = "openrouter",
    llm_model: str = "google/gemini-3-flash-preview",
    embedding_model: str = "google/gemini-embedding-001",
    api_key: str | None = None,
    openrouter_api_base: str = DEFAULT_OPENROUTER_API_BASE,
    openrouter_cost_mode: str = "hybrid",
    max_workers: int = 5,
    clusterer_params: dict | None = None,
    object_description: str = "research papers",
    corpus_description: str = "collection of research papers",
    verbose: bool = True,
    cost_tracker: "CostTracker | None" = None,
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    """Fit a Toponymy topic model with configurable cluster backend.

    Parameters
    ----------
    backend : str
        ``"toponymy"`` (ToponymyClusterer) or ``"toponymy_evoc"`` (EVoCClusterer).
    llm_provider : str
        ``"openrouter"`` or ``"local"``.
    max_workers : int
        Max concurrent OpenRouter requests for Toponymy labeling and embedding.

    Returns
    -------
    tuple
        ``(toponymy_model, topics, topic_info)`` where ``topic_info`` has
        BERTopic-compatible columns ``Topic`` and ``Name``.
    """
    provider_norm, backend_norm = _normalize_toponymy_inputs(
        llm_provider=llm_provider,
        backend=backend,
        api_key=api_key,
    )
    from toponymy import Toponymy, ToponymyClusterer

    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    openrouter_api_base = normalize_openrouter_api_base(openrouter_api_base)
    max_workers = max(1, int(max_workers))
    clusterer_params = dict(clusterer_params or {})
    clusterer, clusterable_vectors_for_fit = _build_toponymy_clusterer(
        backend_norm=backend_norm,
        clusterer_params=clusterer_params,
        toponymy_clusterer_cls=ToponymyClusterer,
        embeddings=embeddings,
        clusterable_vectors=clusterable_vectors,
    )
    llm_wrapper, llm_usage, text_embedding_model = _build_toponymy_models(
        provider_norm=provider_norm,
        llm_model=llm_model,
        embedding_model=embedding_model,
        api_key=api_key,
        openrouter_api_base=openrouter_api_base,
        max_workers=max_workers,
        cost_tracker=cost_tracker,
    )
    topic_model, topics, topic_info = _fit_and_extract_toponymy_outputs(
        toponymy_cls=Toponymy,
        documents=documents,
        embeddings=embeddings,
        clusterable_vectors=clusterable_vectors_for_fit,
        llm_wrapper=llm_wrapper,
        text_embedding_model=text_embedding_model,
        clusterer=clusterer,
        object_description=object_description,
        corpus_description=corpus_description,
        verbose=verbose,
        backend_norm=backend_norm,
        provider_norm=provider_norm,
        llm_model=llm_model,
        layer_index=layer_index,
    )

    _record_llm_usage(
        llm_usage,
        step="llm_labeling_toponymy_evoc" if backend_norm == "toponymy_evoc" else "llm_labeling_toponymy",
        llm_provider=provider_norm,
        llm_model=llm_model,
        api_key=api_key,
        openrouter_cost_mode=openrouter_cost_mode,
        cost_tracker=cost_tracker,
    )
    if (
        cost_tracker is not None
        and provider_norm == "openrouter"
        and hasattr(text_embedding_model, "usage")
    ):
        emb_usage = text_embedding_model.usage
        total_tokens = int(emb_usage.get("total_tokens", 0))
        if total_tokens > 0:
            prompt_tokens = int(emb_usage.get("prompt_tokens", 0))
            embedder_workers = int(getattr(text_embedding_model, "max_workers", 5))
            cost_usd = _fetch_openrouter_costs(
                emb_usage.get("call_records", []),
                api_key,
                openrouter_cost_mode=openrouter_cost_mode,
                max_workers=embedder_workers,
            )
            cost_tracker.add(
                step="toponymy_embeddings",
                provider="openrouter",
                model=embedding_model,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )
    return topic_model, topics, topic_info


@contextmanager
def _suppress_manual_topics_warning():
    """Silence BERTopic's generic warning and replace with explicit pipeline logs."""
    target = (
        "Using a custom list of topic assignments may lead to errors if "
        "topic reduction techniques are used afterwards."
    )
    logger = logging.getLogger("BERTopic")

    class _MessageFilter(logging.Filter):
        """Filter out BERTopic's generic manual-topics warning."""

        def filter(self, record: logging.LogRecord) -> bool:
            """Return True for messages that should remain visible."""
            return target not in record.getMessage()

    warning_filter = _MessageFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


def _build_representation_model(
    *,
    llm_provider: str,
    llm_model: str,
    llm_prompt: str | None,
    pipeline_models: list[str],
    parallel_models: list[str],
    mmr_diversity: float,
    llm_nr_docs: int,
    llm_diversity: float,
    llm_delay: float,
    keybert_model: str | None,
    api_key: str | None,
    pos_spacy_model: str,
) -> dict[str, Any]:
    """Build BERTopic representation models for sequential and parallel use."""
    from bertopic.representation import MaximalMarginalRelevance, PartOfSpeech

    DEFAULT_PROMPT = (
        "You are an experienced researcher. You are labeling research topic clusters.\n\n"
        "Documents: [DOCUMENTS]\nKeywords: [KEYWORDS]\n\n"
        "Task: Generate EXACTLY ONE topic label of 4-7 words.\n"
        "Output format (single line): topic: <label>\n"
        "Do NOT write anything else."
    )
    prompt = llm_prompt or DEFAULT_PROMPT

    if "POS" in pipeline_models or "POS" in parallel_models:
        logger.info("  Initializing POS keyword extraction model: %s", pos_spacy_model)

    # Pipeline (sequential before LLM)
    pipe = []
    for name in pipeline_models:
        if name == "POS":
            pipe.append(PartOfSpeech(pos_spacy_model))
        elif name == "KeyBERT":
            from bertopic.representation import KeyBERTInspired
            pipe.append(KeyBERTInspired())
        elif name == "MMR":
            pipe.append(MaximalMarginalRelevance(diversity=mmr_diversity))

    # LLM
    pipe.append(_create_llm(llm_provider, llm_model, prompt, llm_nr_docs,
                             llm_diversity, llm_delay, api_key))

    result = {"Main": pipe if len(pipe) > 1 else pipe[0]}

    # Parallel models
    for name in parallel_models:
        if name == "MMR":
            result["MMR"] = MaximalMarginalRelevance(diversity=mmr_diversity)
        elif name == "POS":
            result["POS"] = PartOfSpeech(pos_spacy_model)
        elif name == "KeyBERT":
            from bertopic.representation import KeyBERTInspired
            result["KeyBERT"] = KeyBERTInspired()

    return result


def _create_llm(
    provider: str,
    model: str,
    prompt: str,
    nr_docs: int,
    diversity: float,
    delay: float,
    api_key: str | None,
) -> Any:
    """Create configured BERTopic LLM representation backend."""
    if provider == "local":
        from transformers import pipeline as hf_pipeline
        from bertopic.representation import TextGeneration

        logger.info("  Loading local LLM: %s", model)
        gen = hf_pipeline("text-generation", model=model, device_map="auto", torch_dtype="auto")
        return TextGeneration(
            gen, prompt=prompt,
            pipeline_kwargs={"do_sample": False, "max_new_tokens": 16, "num_return_sequences": 1},
        )
    elif provider in ("huggingface_api", "openrouter"):
        from bertopic.representation import LiteLLM

        kwargs: dict = {
            "model": model,
            "prompt": prompt,
            "nr_docs": nr_docs,
            "diversity": diversity,
            "delay_in_seconds": delay,
            "generator_kwargs": {"max_tokens": 16, "temperature": 0.0, "stop": ["\n"]},
        }
        if provider == "openrouter":
            if not model.startswith("openrouter/"):
                kwargs["model"] = f"openrouter/{model}"
            if api_key:
                kwargs["generator_kwargs"]["api_key"] = api_key
        return LiteLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Outlier reduction
# ---------------------------------------------------------------------------

def reduce_outliers(
    topic_model: "BERTopic",
    documents: list[str],
    topics: np.ndarray,
    reduced_5d: np.ndarray,
    *,
    threshold: float = 0.8,
    llm_provider: str = "local",
    llm_model: str = "google/gemma-3-1b-it",
    api_key: str | None = None,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
) -> np.ndarray:
    """Reduce outliers by re-assigning them to the nearest cluster.

    Recomputes topic representations afterwards, which is required for
    consistent labels after topic re-assignment.
    """
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    before = (topics == -1).sum()
    new_topics = topic_model.reduce_outliers(
        documents, topics, strategy="embeddings",
        embeddings=reduced_5d, threshold=threshold,
    )
    after = (np.array(new_topics) == -1).sum()
    logger.info("  Outliers: %s → %s", f"{before:,}", f"{after:,}")
    logger.info(
        "  Refreshing topic representations after outlier reassignment (not a full BERTopic refit)."
    )
    logger.info(
        "  Topic reduction must happen before this step when using manual topic assignments."
    )

    track_llm_usage = cost_tracker is not None and llm_provider in ("openrouter", "huggingface_api")
    with _track_litellm_usage(enabled=track_llm_usage) as llm_usage:
        with _suppress_manual_topics_warning():
            topic_model.update_topics(
                documents, topics=new_topics,
                vectorizer_model=topic_model.vectorizer_model,
                ctfidf_model=topic_model.ctfidf_model,
                representation_model=topic_model.representation_model,
            )

    _record_llm_usage(
        llm_usage,
        step="llm_labeling_post_outliers",
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
        openrouter_cost_mode=openrouter_cost_mode,
        cost_tracker=cost_tracker,
    )
    return np.array(new_topics)


# ---------------------------------------------------------------------------
# Build result DataFrame
# ---------------------------------------------------------------------------

def build_topic_dataframe(
    df: pd.DataFrame,
    topic_model: Any,
    topics: np.ndarray,
    reduced_2d: np.ndarray,
    embeddings: np.ndarray | None = None,
    topic_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Augment *df* with topic modeling results.

    Adds columns: ``embedding_2d_x``, ``embedding_2d_y``, ``topic_id``,
    representation columns from ``topic_aspects_``, and optionally
    ``full_embeddings``.
    """
    df = df.copy()
    df["embedding_2d_x"] = reduced_2d[:, 0]
    df["embedding_2d_y"] = reduced_2d[:, 1]
    df["topic_id"] = topics

    info = topic_info if topic_info is not None else topic_model.get_topic_info()
    base_cols = ["Name"]
    aspect_cols = [c for c in ("Main", "MMR", "POS", "KeyBERT") if c in info.columns]

    for col in base_cols + aspect_cols:
        if col in info.columns:
            mapping = {
                row["Topic"]: (", ".join(row[col]) if isinstance(row[col], list) else row[col])
                for _, row in info.iterrows()
            }
            df[col] = df["topic_id"].map(mapping)

    if embeddings is not None:
        df["full_embeddings"] = list(embeddings)

    # Set custom labels from LLM output
    if (
        hasattr(topic_model, "topic_representations_")
        and hasattr(topic_model, "set_topic_labels")
        and topic_model.topic_representations_
    ):
        labels = {}
        for tid, rep in topic_model.topic_representations_.items():
            if rep:
                labels[tid] = " | ".join(w for w, _ in rep[:3])
        labels[-1] = "Outlier Topic"
        topic_model.set_topic_labels(labels)

    # For Toponymy models, extract all hierarchical layer names as individual columns
    if hasattr(topic_model, "cluster_layers_"):
        for i, layer in enumerate(topic_model.cluster_layers_):
            df[f"Topic_Layer_{i}"] = layer.topic_name_vector

    return df
