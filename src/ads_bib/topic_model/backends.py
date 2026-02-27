"""Topic-model backends (BERTopic and Toponymy) and clustering orchestration."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import inspect
import logging
from typing import Any, Literal, TypeAlias, cast
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ads_bib._utils.hf_compat import raise_with_local_hf_compat_hint
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
    resolve_openrouter_costs,
)
from ads_bib.config import validate_provider
from ads_bib.topic_model.embeddings import OpenRouterEmbedder

logger = logging.getLogger("ads_bib.topic_model")

DEFAULT_CLUSTER_MIN_SIZE = 180
DEFAULT_BERTOPIC_TOP_N_WORDS = 20
DEFAULT_POS_SPACY_MODEL = "en_core_web_sm"
DEFAULT_BERTOPIC_LLM_MAX_NEW_TOKENS = 128
DEFAULT_TOPONYMY_LOCAL_LLM_MAX_NEW_TOKENS = 256
BERTopicLLMProvider: TypeAlias = Literal["local", "gguf", "huggingface_api", "openrouter"]
ToponymyLLMProvider: TypeAlias = Literal["local", "gguf", "openrouter"]
ToponymyBackend: TypeAlias = Literal["toponymy", "toponymy_evoc"]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _create_cluster_model(method: str, params: dict | None = None):
    """Build the configured clustering model once for all topic workflows."""
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
    """Cluster documents in the 5-D embedding space."""
    model = _create_cluster_model(method, params)
    clusters = model.fit_predict(reduced_5d)
    n_topics = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_outliers = int((clusters == -1).sum())
    logger.info(
        "  %s: %s topics, %s outliers (%.1f%%)",
        method.upper(),
        n_topics,
        f"{n_outliers:,}",
        n_outliers / len(clusters) * 100,
    )
    return clusters


# ---------------------------------------------------------------------------
# BERTopic LLM helpers
# ---------------------------------------------------------------------------

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
        cost_usd, _ = resolve_openrouter_costs(
            usage.get("call_records", []),
            mode=openrouter_cost_mode,
            api_key=api_key,
            max_workers=5,
            logger_obj=logger,
            total_label="OpenRouter cost",
        )

    cost_tracker.add(
        step=step,
        provider=llm_provider,
        model=llm_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


@contextmanager
def _suppress_manual_topics_warning():
    """Silence BERTopic's generic warning and replace with explicit pipeline logs."""
    target = (
        "Using a custom list of topic assignments may lead to errors if "
        "topic reduction techniques are used afterwards."
    )
    logger_name = logging.getLogger("BERTopic")

    class _MessageFilter(logging.Filter):
        """Filter out BERTopic's generic manual-topics warning."""

        def filter(self, record: logging.LogRecord) -> bool:
            """Return True for messages that should remain visible."""
            return target not in record.getMessage()

    warning_filter = _MessageFilter()
    logger_name.addFilter(warning_filter)
    try:
        yield
    finally:
        logger_name.removeFilter(warning_filter)


def _create_llm(
    provider: str,
    model: str,
    prompt: str,
    nr_docs: int,
    diversity: float,
    delay: float,
    llm_max_new_tokens: int,
    api_key: str | None,
) -> Any:
    """Create configured BERTopic LLM representation backend."""
    llm_max_new_tokens = max(1, int(llm_max_new_tokens))

    if provider == "local":
        from bertopic.representation import TextGeneration
        from transformers import pipeline as hf_pipeline

        logger.info("  Loading local LLM: %s", model)
        try:
            try:
                gen = hf_pipeline(
                    "text-generation",
                    model=model,
                    device_map="auto",
                    dtype="auto",
                )
            except TypeError:
                gen = hf_pipeline(
                    "text-generation",
                    model=model,
                    device_map="auto",
                    torch_dtype="auto",
                )
        except Exception as exc:
            raise_with_local_hf_compat_hint(model=model, use_case="topic labeling", exc=exc)
        return TextGeneration(
            gen,
            prompt=prompt,
            pipeline_kwargs={
                "do_sample": False,
                "max_new_tokens": llm_max_new_tokens,
                "num_return_sequences": 1,
            },
        )

    if provider == "gguf":
        from bertopic.representation import TextGeneration

        from ads_bib._utils.gguf_backend import LlamaCppTextGeneration, resolve_gguf_model

        model_path = resolve_gguf_model(model)
        gen = LlamaCppTextGeneration(model_path, max_new_tokens=llm_max_new_tokens)
        return TextGeneration(
            gen,
            prompt=prompt,
            pipeline_kwargs={"max_new_tokens": llm_max_new_tokens},
        )

    if provider in ("huggingface_api", "openrouter"):
        from bertopic.representation import LiteLLM

        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "nr_docs": nr_docs,
            "diversity": diversity,
            "delay_in_seconds": delay,
            "generator_kwargs": {
                "max_tokens": llm_max_new_tokens,
                "temperature": 0.0,
                "stop": ["\n"],
            },
        }
        if provider == "openrouter":
            if not model.startswith("openrouter/"):
                kwargs["model"] = f"openrouter/{model}"
            if api_key:
                kwargs["generator_kwargs"]["api_key"] = api_key
        return LiteLLM(**kwargs)

    raise ValueError(f"Unknown LLM provider: {provider}")


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
    llm_max_new_tokens: int,
    api_key: str | None,
    pos_spacy_model: str,
) -> dict[str, Any]:
    """Build BERTopic representation models for sequential and parallel use."""
    from bertopic.representation import MaximalMarginalRelevance, PartOfSpeech

    default_prompt = (
        "You are an experienced researcher. You are labeling research topic clusters.\n\n"
        "Documents: [DOCUMENTS]\nKeywords: [KEYWORDS]\n\n"
        "Task: Generate EXACTLY ONE topic label of 4-7 words.\n"
        "Output format (single line): topic: <label>\n"
        "Do NOT write anything else."
    )
    prompt = llm_prompt or default_prompt

    if "POS" in pipeline_models or "POS" in parallel_models:
        logger.info("  Initializing POS keyword extraction model: %s", pos_spacy_model)

    pipe: list[Any] = []
    for name in pipeline_models:
        if name == "POS":
            pipe.append(PartOfSpeech(pos_spacy_model))
        elif name == "KeyBERT":
            from bertopic.representation import KeyBERTInspired

            pipe.append(KeyBERTInspired())
        elif name == "MMR":
            pipe.append(MaximalMarginalRelevance(diversity=mmr_diversity))

    pipe.append(
        _create_llm(
            llm_provider,
            llm_model,
            prompt,
            llm_nr_docs,
            llm_diversity,
            llm_delay,
            llm_max_new_tokens,
            api_key,
        )
    )

    result: dict[str, Any] = {"Main": pipe if len(pipe) > 1 else pipe[0]}

    for name in parallel_models:
        if name == "MMR":
            result["MMR"] = MaximalMarginalRelevance(diversity=mmr_diversity)
        elif name == "POS":
            result["POS"] = PartOfSpeech(pos_spacy_model)
        elif name == "KeyBERT":
            from bertopic.representation import KeyBERTInspired

            result["KeyBERT"] = KeyBERTInspired()

    return result


# ---------------------------------------------------------------------------
# Toponymy helpers
# ---------------------------------------------------------------------------

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
    """Instantiate *cls* with kwargs filtered to supported ``__init__`` parameters."""
    kwargs = dict(params or {})

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"{component_name}.__init__ failed with parameters {sorted(kwargs.keys())}: {exc}"
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
            f"Dropping unsupported {component_name} parameter(s): {', '.join(dropped)}",
            UserWarning,
            stacklevel=2,
        )

    try:
        return cls(**filtered)
    except TypeError as exc:
        raise TypeError(
            f"{component_name}.__init__ failed with filtered parameters {sorted(filtered.keys())}: {exc}"
        ) from exc


def _patch_clusterer_for_toponymy_kwargs(clusterer: Any) -> None:
    """Make strict clusterer methods tolerant to extra kwargs from Toponymy.fit."""

    def _wrap_bound(method: Any) -> Any:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return method

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return method

        accepted = set(sig.parameters.keys())

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
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
    llm_provider: ToponymyLLMProvider | str,
    backend: ToponymyBackend | str,
    api_key: str | None,
) -> tuple[ToponymyLLMProvider, ToponymyBackend]:
    """Normalize and validate Toponymy backend/provider selections."""
    provider_norm = llm_provider.strip().lower()
    if provider_norm not in {"openrouter", "local", "gguf"}:
        raise ValueError(f"Invalid llm_provider '{llm_provider}'. Expected 'openrouter', 'local', or 'gguf'.")
    if provider_norm == "openrouter" and not api_key:
        raise ValueError("api_key is required for Toponymy with OpenRouter.")

    backend_norm = backend.strip().lower()
    if backend_norm not in {"toponymy", "toponymy_evoc"}:
        raise ValueError(f"Invalid backend '{backend}'. Expected 'toponymy' or 'toponymy_evoc'.")

    return (
        cast(ToponymyLLMProvider, provider_norm),
        cast(ToponymyBackend, backend_norm),
    )


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
    local_llm_max_new_tokens: int,
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

    if provider_norm == "gguf":
        from sentence_transformers import SentenceTransformer

        from ads_bib._utils.gguf_backend import _build_llama_cpp_namer, resolve_gguf_model

        model_path = resolve_gguf_model(llm_model)
        llm_wrapper = _build_llama_cpp_namer(
            model_path, max_new_tokens=local_llm_max_new_tokens,
        )
        try:
            text_embedding_model = SentenceTransformer(embedding_model)
        except Exception as exc:
            raise_with_local_hf_compat_hint(
                model=embedding_model, use_case="toponymy embeddings", exc=exc,
            )
        if cost_tracker is not None:
            logger.info("  GGUF Toponymy LLM selected; token/cost tracking is unavailable for this step.")
        return llm_wrapper, None, text_embedding_model

    try:
        from sentence_transformers import SentenceTransformer
        from toponymy.llm_wrappers import HuggingFaceNamer
    except Exception as exc:
        raise ImportError(
            "llm_provider='local' requires optional dependencies "
            "'sentence-transformers', 'transformers', and "
            "Toponymy's HuggingFaceNamer wrapper."
        ) from exc

    local_llm_max_new_tokens = max(1, int(local_llm_max_new_tokens))

    class _CappedDeterministicHuggingFaceNamer(HuggingFaceNamer):
        """Toponymy local namer with deterministic decoding and bounded output length."""

        def __init__(self, model: str, *, max_new_tokens: int, **kwargs):
            self._local_max_new_tokens = max(1, int(max_new_tokens))
            super().__init__(model=model, **kwargs)

        def _max_tokens(self, requested: int) -> int:
            return min(max(1, int(requested)), self._local_max_new_tokens)

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            del temperature
            response = self.llm(
                [{"role": "user", "content": prompt + self.extra_prompting}],
                return_full_text=False,
                max_new_tokens=self._max_tokens(max_tokens),
                do_sample=False,
                pad_token_id=self.llm.tokenizer.eos_token_id,
            )
            return response[0]["generated_text"]

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            del temperature
            response = self.llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                return_full_text=False,
                max_new_tokens=self._max_tokens(max_tokens),
                do_sample=False,
                pad_token_id=self.llm.tokenizer.eos_token_id,
            )
            return response[0]["generated_text"]

    try:
        try:
            llm_wrapper = _CappedDeterministicHuggingFaceNamer(
                model=llm_model,
                max_new_tokens=local_llm_max_new_tokens,
                device_map="auto",
                dtype="auto",
            )
        except TypeError:
            llm_wrapper = _CappedDeterministicHuggingFaceNamer(
                model=llm_model,
                max_new_tokens=local_llm_max_new_tokens,
                device_map="auto",
                torch_dtype="auto",
            )
    except Exception as exc:
        raise_with_local_hf_compat_hint(model=llm_model, use_case="toponymy labeling", exc=exc)

    try:
        text_embedding_model = SentenceTransformer(embedding_model)
    except Exception as exc:
        raise_with_local_hf_compat_hint(model=embedding_model, use_case="toponymy embeddings", exc=exc)

    if cost_tracker is not None:
        logger.info("  Local Toponymy LLM selected; token/cost tracking is unavailable for this step.")
    logger.info(
        "  Local Toponymy LLM max_new_tokens capped at %s per naming call.",
        local_llm_max_new_tokens,
    )
    return llm_wrapper, None, text_embedding_model


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
        raise ValueError(f"layer_index {layer_index} is out of range for {n_layers} available layers.")

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


# ---------------------------------------------------------------------------
# BERTopic fitting
# ---------------------------------------------------------------------------

def fit_bertopic(
    documents: list[str],
    reduced_5d: np.ndarray,
    *,
    llm_provider: BERTopicLLMProvider = "local",
    llm_model: str = "google/gemma-3-1b-it",
    llm_prompt: str | None = None,
    pipeline_models: list[str] | None = None,
    parallel_models: list[str] | None = None,
    mmr_diversity: float = 0.3,
    llm_nr_docs: int = 8,
    llm_diversity: float = 0.2,
    llm_delay: float = 0.3,
    llm_max_new_tokens: int = DEFAULT_BERTOPIC_LLM_MAX_NEW_TOKENS,
    embedding_model_name: str | None = None,
    keybert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_df: int = 2,
    clustering_method: str = "fast_hdbscan",
    clustering_params: dict | None = None,
    top_n_words: int = DEFAULT_BERTOPIC_TOP_N_WORDS,
    pos_spacy_model: str = DEFAULT_POS_SPACY_MODEL,
    show_progress: bool = True,
    api_key: str | None = None,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
) -> "BERTopic":
    """Fit BERTopic on pre-reduced document vectors.

    Parameters
    ----------
    documents : list[str]
        Input corpus; order must match *reduced_5d*.
    reduced_5d : np.ndarray
        Five-dimensional vectors used directly for clustering.
    llm_provider : str
        Labeling backend: ``"local"``, ``"gguf"``, ``"huggingface_api"``, ``"openrouter"``.
    llm_model : str
        LLM model used for topic naming.
    llm_max_new_tokens : int
        Maximum generated tokens per topic-label request.
    clustering_method : str
        Clustering backend passed into BERTopic (``"fast_hdbscan"`` or
        ``"hdbscan"``).
    clustering_params : dict, optional
        Parameters forwarded to the selected clustering backend.
    api_key : str, optional
        Required for ``llm_provider="openrouter"``.
    openrouter_cost_mode : str
        Cost resolution mode for OpenRouter usage.
    cost_tracker : CostTracker, optional
        Optional cost accumulator for LLM labeling calls.

    Returns
    -------
    BERTopic
        Fitted BERTopic instance.
    """
    validate_provider(
        llm_provider,
        valid={"local", "gguf", "huggingface_api", "openrouter"},
        api_key=api_key,
        requires_key={"openrouter"},
        requires_import={
            "local": "transformers",
            "gguf": "llama_cpp",
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
        llm_max_new_tokens=llm_max_new_tokens,
        api_key=api_key,
        pos_spacy_model=pos_spacy_model,
    )

    vectorizer = CountVectorizer(stop_words="english", min_df=min_df, ngram_range=(1, 3))
    ctfidf = ClassTfidfTransformer()

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
        with tqdm(total=1, desc="BERTopic fit", disable=not show_progress) as pbar:
            topic_model.fit_transform(documents, reduced_5d)
            pbar.update(1)

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


# ---------------------------------------------------------------------------
# Toponymy fitting
# ---------------------------------------------------------------------------

def fit_toponymy(
    documents: list[str],
    embeddings: np.ndarray,
    clusterable_vectors: np.ndarray,
    *,
    backend: ToponymyBackend = "toponymy",
    layer_index: int = 0,
    llm_provider: ToponymyLLMProvider = "openrouter",
    llm_model: str = "google/gemini-3-flash-preview",
    embedding_model: str = "google/gemini-embedding-001",
    api_key: str | None = None,
    openrouter_api_base: str = DEFAULT_OPENROUTER_API_BASE,
    openrouter_cost_mode: str = "hybrid",
    max_workers: int = 5,
    local_llm_max_new_tokens: int = DEFAULT_TOPONYMY_LOCAL_LLM_MAX_NEW_TOKENS,
    clusterer_params: dict | None = None,
    object_description: str = "research papers",
    corpus_description: str = "collection of research papers",
    verbose: bool = True,
    cost_tracker: "CostTracker | None" = None,
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    """Fit Toponymy (or Toponymy+EVoC) and return topic assignments.

    Parameters
    ----------
    documents : list[str]
        Input corpus; order must match *embeddings* and *clusterable_vectors*.
    embeddings : np.ndarray
        Full embedding matrix used for naming/context.
    clusterable_vectors : np.ndarray
        Vectors used for clustering (5D reduced vectors for ``backend="toponymy"``;
        raw high-dimensional vectors for ``backend="toponymy_evoc"``).
    backend : str
        ``"toponymy"`` or ``"toponymy_evoc"``.
    layer_index : int
        Hierarchical layer index selected as primary output topic_id.
    llm_provider : str
        Currently supported Toponymy naming provider.
    llm_model : str
        LLM model identifier for topic naming.
    local_llm_max_new_tokens : int
        Max generated tokens per local Toponymy naming call.
    embedding_model : str
        Text embedding model for Toponymy internals.
    api_key : str, optional
        Provider key where required.
    clusterer_params : dict, optional
        Backend-specific clustering parameters.
    cost_tracker : CostTracker, optional
        Optional usage/cost tracker.

    Returns
    -------
    tuple[Any, np.ndarray, pd.DataFrame]
        ``(topic_model, topics, topic_info)`` where *topics* is the selected
        layer assignment vector and *topic_info* contains ``Topic``/``Name``
        metadata.
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
        local_llm_max_new_tokens=local_llm_max_new_tokens,
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

    if cost_tracker is not None and provider_norm == "openrouter" and hasattr(text_embedding_model, "usage"):
        emb_usage = text_embedding_model.usage
        total_tokens = int(emb_usage.get("total_tokens", 0))
        if total_tokens > 0:
            prompt_tokens = int(emb_usage.get("prompt_tokens", 0))
            embedder_workers = int(getattr(text_embedding_model, "max_workers", 5))
            cost_usd, _ = resolve_openrouter_costs(
                emb_usage.get("call_records", []),
                mode=openrouter_cost_mode,
                api_key=api_key,
                max_workers=embedder_workers,
                logger_obj=logger,
                total_label="OpenRouter cost",
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
    show_progress: bool = True,
    api_key: str | None = None,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
) -> np.ndarray:
    """Reassign BERTopic outliers and refresh topic representations.

    Parameters
    ----------
    topic_model : BERTopic
        Fitted BERTopic model to update in-place.
    documents : list[str]
        Corpus used during BERTopic fitting.
    topics : np.ndarray
        Current topic assignments (including ``-1`` outliers).
    reduced_5d : np.ndarray
        Five-dimensional vectors used by BERTopic outlier reassignment.
    threshold : float
        Confidence threshold passed to ``topic_model.reduce_outliers``.
    llm_provider : str
        Labeling provider used by the representation model.
    llm_model : str
        Model name recorded for cost tracking.
    cost_tracker : CostTracker, optional
        Optional tracker for refresh-time LLM usage.

    Returns
    -------
    np.ndarray
        Updated topic assignment vector after outlier reassignment.

    Notes
    -----
    This function always calls ``update_topics`` after reassignment so topic
    labels/representations reflect the new assignments.
    """
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    before = int((topics == -1).sum())
    new_topics = topic_model.reduce_outliers(
        documents,
        topics,
        strategy="embeddings",
        embeddings=reduced_5d,
        threshold=threshold,
    )
    after = int((np.array(new_topics) == -1).sum())
    logger.info("  Outliers: %s → %s", f"{before:,}", f"{after:,}")
    logger.info("  Refreshing topic representations after outlier reassignment (not a full BERTopic refit).")
    logger.info("  Topic reduction must happen before this step when using manual topic assignments.")

    track_llm_usage = cost_tracker is not None and llm_provider in ("openrouter", "huggingface_api")
    with _track_litellm_usage(enabled=track_llm_usage) as llm_usage:
        with tqdm(total=1, desc="BERTopic refresh", disable=not show_progress) as pbar:
            with _suppress_manual_topics_warning():
                topic_model.update_topics(
                    documents,
                    topics=new_topics,
                    vectorizer_model=topic_model.vectorizer_model,
                    ctfidf_model=topic_model.ctfidf_model,
                    representation_model=topic_model.representation_model,
                )
            pbar.update(1)

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


__all__ = [
    "fit_bertopic",
    "fit_toponymy",
    "reduce_outliers",
]
