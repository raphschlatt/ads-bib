"""Topic-model backends (BERTopic and Toponymy) and clustering orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import inspect
import logging
from pathlib import Path
import time
from typing import Any, Literal, TypeAlias, cast
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ads_bib._utils.hf_compat import raise_with_local_hf_compat_hint
from ads_bib._utils.huggingface_api import (
    normalize_huggingface_model,
    normalize_huggingface_model_for_litellm,
    resolve_huggingface_api_key,
)
from ads_bib._utils.llama_server import (
    LlamaServerConfig,
    build_openai_client,
    ensure_llama_server,
)
from ads_bib._utils.logging import (
    capture_external_output,
    get_runtime_log_path,
    temporarily_raise_logger_level,
)
from ads_bib._utils.model_specs import ModelSpec
from ads_bib._utils.openrouter_client import (
    openrouter_chat_completion,
    openrouter_response_content,
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
from ads_bib.prompts import BERTOPIC_LABELING_GENERIC
from ads_bib.topic_model._runtime import (
    BERTOPIC_LLM_PROVIDER_IMPORTS,
    BERTOPIC_LLM_PROVIDERS,
    TOPONYMY_EMBEDDING_PROVIDERS,
    TOPONYMY_LLM_PROVIDERS,
)
from ads_bib.topic_model.embeddings import OpenRouterEmbedder

logger = logging.getLogger("ads_bib.topic_model")

DEFAULT_CLUSTER_MIN_SIZE = 180       # ~0.1-0.2% of typical corpus (87k-180k docs) -> 50-70 topics
DEFAULT_BERTOPIC_TOP_N_WORDS = 20     # Keywords per topic for c-TF-IDF representation
DEFAULT_POS_SPACY_MODEL = "en_core_web_md"
DEFAULT_KEYBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BERTOPIC_LLM_MAX_NEW_TOKENS = 128   # Concise topic labels (4-7 words)
DEFAULT_BERTOPIC_PIPELINE_MODELS = ["POS", "KeyBERT", "MMR"]
DEFAULT_BERTOPIC_PARALLEL_MODELS = ["MMR", "POS", "KeyBERT"]
DEFAULT_TOPONYMY_LOCAL_LLM_MAX_NEW_TOKENS = 128  # Keep hierarchy labels concise and readable
BERTopicLLMProvider: TypeAlias = Literal["local", "llama_server", "huggingface_api", "openrouter"]
ToponymyLLMProvider: TypeAlias = Literal["local", "llama_server", "openrouter"]
ToponymyEmbeddingProvider: TypeAlias = Literal["local", "openrouter"]
ToponymyBackend: TypeAlias = Literal["toponymy"]
ToponymyLayerIndex: TypeAlias = int | Literal["auto"] | None


def _raise_with_toponymy_import_hint(exc: ImportError, *, backend: str) -> None:
    """Raise an actionable import error for Toponymy's transitive dependency stack."""
    missing_module = exc.name if isinstance(exc, ModuleNotFoundError) else None
    message = (
        f"backend='{backend}' requires optional dependency 'toponymy' and its runtime dependencies."
    )
    if missing_module:
        message += f" Missing module: '{missing_module}'."
    if missing_module == "dask":
        message += " Toponymy imports 'vectorizers', which requires 'dask'."
    message += (
        " Install the missing package in your active Python environment, or reinstall the topic extras "
        "with `uv pip install -e \".[all,test]\"`."
    )
    raise ImportError(message) from exc

def _load_local_sentence_transformer(
    *,
    model: str,
    use_case: str,
    missing_dependency_message: str,
    runtime_log_path: Path | None,
) -> Any:
    """Load a local sentence-transformer under the runtime-log capture contract."""
    try:
        with capture_external_output(runtime_log_path or get_runtime_log_path()):
            with temporarily_raise_logger_level("transformers.modeling_utils", level=logging.ERROR):
                with temporarily_raise_logger_level(
                    "transformers.integrations.tensor_parallel",
                    level=logging.ERROR,
                ):
                    from sentence_transformers import SentenceTransformer

                    return SentenceTransformer(model)
    except ImportError as exc:
        raise ImportError(missing_dependency_message) from exc
    except Exception as exc:
        raise_with_local_hf_compat_hint(model=model, use_case=use_case, exc=exc)

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
            min_samples=3,                     # Low value reduces outliers; higher = stricter density requirement
            cluster_selection_method="eom",     # Excess of Mass: selects most persistent clusters
            cluster_selection_epsilon=0.02,     # Absorbs border points into nearby clusters
            allow_single_cluster=False,
        )
        # Note: fast_hdbscan does NOT support prediction_data or gen_min_span_tree.
        # Use standard hdbscan if you need approximate_predict() or condensed_tree_ analysis.
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
            prediction_data=True,              # Enables approximate_predict() for new documents
            gen_min_span_tree=True,             # Enables condensed_tree_ for cluster hierarchy analysis
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


def _configure_deterministic_generation_pipeline(generator: Any) -> None:
    """Normalize local HF generation defaults for deterministic topic labeling."""
    model = getattr(generator, "model", None)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    generation_config.do_sample = False
    generation_config.max_length = None

    # Some instruct checkpoints ship sampling-oriented defaults in generation_config.
    # For deterministic labeling we reset them to greedy-compatible values.
    for name, value in {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
    }.items():
        if hasattr(generation_config, name):
            setattr(generation_config, name, value)


@contextmanager
def _bridge_bertopic_label_progress(*, reporter: Any | None, desc: str):
    """Route BERTopic's per-topic labeling loops through the shared stage reporter."""
    if reporter is None:
        yield
        return

    patched: list[tuple[Any, Any]] = []
    state: dict[str, Any] = {"cm": None, "pbar": None, "total": None}

    def _refresh_total(total: int | None) -> None:
        pbar = state["pbar"]
        if pbar is None:
            cm = reporter.progress(total=total, desc=desc)
            state["cm"] = cm
            state["pbar"] = cm.__enter__()
            state["total"] = total
            return

        if total is None:
            return
        current_total = state["total"]
        if current_total is None:
            try:
                pbar.total = total
            except Exception:
                pass
            else:
                state["total"] = total
                refresh = getattr(pbar, "refresh", None)
                if callable(refresh):
                    refresh()
            return

        try:
            pbar.total = current_total + total
        except Exception:
            return
        state["total"] = current_total + total
        refresh = getattr(pbar, "refresh", None)
        if callable(refresh):
            refresh()

    def _infer_total(iterable: Any, explicit_total: Any) -> int | None:
        if explicit_total is not None:
            return int(explicit_total)
        try:
            return len(iterable)
        except Exception:
            return None

    def _update(amount: int = 1) -> None:
        pbar = state["pbar"]
        if pbar is not None:
            pbar.update(int(amount))

    def _bridge_tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
        del args
        total = _infer_total(iterable, kwargs.get("total"))
        _refresh_total(total)

        if iterable is None:
            return _ReporterTqdmProxy(update_callback=_update)

        class _ReporterIterable:
            def __iter__(self) -> Iterator[Any]:
                for item in iterable:
                    yield item
                    _update(1)

            def update(self, amount: int = 1) -> None:
                _update(amount)

            def close(self) -> None:
                return None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb
                return None

        return _ReporterIterable()

    class _ReporterTqdmProxy:
        def __init__(self, *, update_callback):
            self._update_callback = update_callback

        def update(self, amount: int = 1) -> None:
            self._update_callback(amount)

        def close(self) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    try:
        for module_name in (
            "bertopic.representation._textgeneration",
            "bertopic.representation._litellm",
            "bertopic.representation._openai",
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            patched.append((module, getattr(module, "tqdm", None)))
            setattr(module, "tqdm", _bridge_tqdm)
        yield
    finally:
        for module, original_tqdm in reversed(patched):
            setattr(module, "tqdm", original_tqdm)
        cm = state["cm"]
        if cm is not None:
            cm.__exit__(None, None, None)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from LLM output."""
    return text.replace("<think>", "").split("</think>", 1)[-1].strip()


def _strip_think_tags_from_topics(topic_model) -> None:
    """Post-process BERTopic representations to remove any leaked think tags."""
    reps = getattr(topic_model, "topic_representations_", None)
    if not reps:
        return
    for topic_id, words in reps.items():
        reps[topic_id] = [
            (_strip_think_tags(w), score) for w, score in words
        ]


def _new_usage_summary() -> dict[str, Any]:
    """Return an empty usage accumulator with the shared LLM schema."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}


def _merge_usage_summary(target: dict[str, Any], source: dict[str, Any] | None) -> None:
    """Merge one usage payload into another."""
    if not source:
        return
    target["prompt_tokens"] += int(source.get("prompt_tokens", 0))
    target["completion_tokens"] += int(source.get("completion_tokens", 0))
    target["call_records"].extend(list(source.get("call_records", [])))


def _consume_openrouter_representation_usage(representation_model: Any) -> dict[str, Any]:
    """Collect and reset usage from repo-owned OpenRouter BERTopic representations."""
    usage = _new_usage_summary()

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _visit(value)
            return
        if isinstance(node, (list, tuple)):
            for value in node:
                _visit(value)
            return
        consume = getattr(node, "consume_usage", None)
        if callable(consume):
            _merge_usage_summary(usage, consume())

    _visit(representation_model)
    return usage


def _create_openrouter_bertopic_representation(
    *,
    model: str,
    prompt: str,
    nr_docs: int,
    diversity: float,
    delay: float,
    max_tokens: int,
    api_key: str | None,
) -> Any:
    """Create a BERTopic representation model that calls OpenRouter directly."""
    from bertopic.representation import BaseRepresentation

    class _OpenRouterBERTopicRepresentation(BaseRepresentation):
        def __init__(self) -> None:
            self.model = model.removeprefix("openrouter/")
            self.prompt = prompt
            self.nr_docs = nr_docs
            self.diversity = diversity
            self.delay_in_seconds = delay
            self.max_tokens = max(1, int(max_tokens))
            self.api_key = api_key
            self.api_base = DEFAULT_OPENROUTER_API_BASE
            self._client = None
            self._usage = _new_usage_summary()

        def _get_client(self) -> Any:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            return self._client

        def consume_usage(self) -> dict[str, Any]:
            usage = _new_usage_summary()
            _merge_usage_summary(usage, self._usage)
            self._usage = _new_usage_summary()
            return usage

        def extract_topics(
            self,
            topic_model: Any,
            documents: pd.DataFrame,
            c_tf_idf: Any,
            topics: dict[int, list[tuple[str, float]]],
        ) -> dict[int, list[tuple[str, float]]]:
            repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
                c_tf_idf,
                documents,
                topics,
                500,
                self.nr_docs,
                self.diversity,
            )

            updated_topics: dict[int, list[tuple[str, float]]] = {}
            for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
                if self.delay_in_seconds:
                    time.sleep(self.delay_in_seconds)

                fallback = list(topics.get(topic, [(f"Topic {topic}", 1.0)]))
                try:
                    response = openrouter_chat_completion(
                        client=self._get_client(),
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": self._create_prompt(docs, topic, topics)},
                        ],
                        max_tokens=self.max_tokens,
                        temperature=0.0,
                        stop=["\n"],
                        retry_label="BERTopic OpenRouter labeling call",
                    )
                except Exception as exc:
                    logger.warning(
                        "  BERTopic OpenRouter labeling fallback for topic %s (model=%s): %s: %s",
                        topic,
                        self.model,
                        type(exc).__name__,
                        exc,
                    )
                    updated_topics[topic] = fallback
                    continue

                stats = openrouter_usage_from_response(response)
                self._usage["prompt_tokens"] += int(stats["prompt_tokens"])
                self._usage["completion_tokens"] += int(stats["completion_tokens"])
                self._usage["call_records"].append(stats["call_record"])

                content, content_state = openrouter_response_content(response)
                label = self._normalize_label(content) if content_state == "ok" and content is not None else ""
                if not label:
                    logger.warning(
                        "  BERTopic OpenRouter labeling fallback for topic %s (model=%s): content_state=%s",
                        topic,
                        self.model,
                        content_state,
                    )
                    updated_topics[topic] = fallback
                    continue

                updated_topics[topic] = [(label, 1)]

            return updated_topics

        def _create_prompt(
            self,
            docs: list[str],
            topic: int,
            topics: dict[int, list[tuple[str, float]]],
        ) -> str:
            prompt_text = self.prompt
            keywords = [word for word, _score in topics.get(topic, [])]
            if "[KEYWORDS]" in prompt_text:
                prompt_text = prompt_text.replace("[KEYWORDS]", " ".join(keywords))
            if "[DOCUMENTS]" in prompt_text:
                prompt_text = prompt_text.replace(
                    "[DOCUMENTS]",
                    "".join(f"- {doc[:255]}\n" for doc in docs),
                )
            return prompt_text

        @staticmethod
        def _normalize_label(content: str | None) -> str:
            if content is None:
                return ""
            label = _strip_think_tags(str(content)).strip()
            if not label:
                return ""
            label = label.splitlines()[0].strip()
            if label.lower().startswith("topic:"):
                label = label.split(":", 1)[1].strip()
            return label

    return _OpenRouterBERTopicRepresentation()


def _create_llm(
    provider: str,
    model: str,
    model_spec: ModelSpec | None,
    prompt: str,
    nr_docs: int,
    diversity: float,
    delay: float,
    llm_max_new_tokens: int,
    api_key: str | None,
    llama_server_config: LlamaServerConfig | None,
    runtime_log_path: Path | None,
) -> Any:
    """Create configured BERTopic LLM representation backend."""
    validate_provider(
        provider,
        valid=set(BERTOPIC_LLM_PROVIDERS),
        requires_import=BERTOPIC_LLM_PROVIDER_IMPORTS,
    )
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
        _configure_deterministic_generation_pipeline(gen)
        return TextGeneration(
            gen,
            prompt=prompt,
            pipeline_kwargs={
                "do_sample": False,
                "max_new_tokens": llm_max_new_tokens,
                "num_return_sequences": 1,
            },
        )

    if provider == "llama_server":
        from bertopic.representation import OpenAI as BERTopicOpenAI

        if model_spec is None:
            raise ValueError("llama_server provider requires a resolved ModelSpec.")
        handle = ensure_llama_server(
            model_spec=model_spec,
            config=llama_server_config or LlamaServerConfig(),
            runtime_log_path=runtime_log_path,
        )
        return BERTopicOpenAI(
            client=build_openai_client(handle=handle),
            model=Path(model_spec.resolve()).name,
            prompt=prompt,
            generator_kwargs={"max_tokens": llm_max_new_tokens, "temperature": 0.0},
            nr_docs=nr_docs,
            diversity=diversity,
        )

    if provider == "openrouter":
        return _create_openrouter_bertopic_representation(
            model=model,
            prompt=prompt,
            nr_docs=nr_docs,
            diversity=diversity,
            delay=delay,
            max_tokens=llm_max_new_tokens,
            api_key=api_key,
        )

    if provider == "huggingface_api":
        from bertopic.representation import LiteLLM

        resolved_model = model
        resolved_api_key = api_key
        resolved_model = normalize_huggingface_model(model)
        resolved_api_key = resolve_huggingface_api_key(api_key)

        kwargs: dict[str, Any] = {
            "model": resolved_model,
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
        kwargs["model"] = normalize_huggingface_model_for_litellm(resolved_model)
        if resolved_api_key:
            kwargs["generator_kwargs"]["api_key"] = resolved_api_key
        return LiteLLM(**kwargs)

    raise ValueError(f"Unknown LLM provider: {provider}")


def _build_representation_model(
    *,
    llm_provider: str,
    llm_model: str,
    llm_model_spec: ModelSpec | None,
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
    llama_server_config: LlamaServerConfig | None,
    runtime_log_path: Path | None,
) -> dict[str, Any]:
    """Build BERTopic representation models for sequential and parallel use."""
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic.representation import PartOfSpeech

    prompt = llm_prompt or BERTOPIC_LABELING_GENERIC

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
            llm_model_spec,
            prompt,
            llm_nr_docs,
            llm_diversity,
            llm_delay,
            llm_max_new_tokens,
            api_key,
            llama_server_config,
            runtime_log_path,
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
            except Exception as exc:
                logger.warning(
                    "  Toponymy OpenRouter labeling fallback (model=%s): %s: %s",
                    self.model,
                    type(exc).__name__,
                    exc,
                )
                return ""

            stats = openrouter_usage_from_response(response)
            usage["prompt_tokens"] += int(stats["prompt_tokens"])
            usage["completion_tokens"] += int(stats["completion_tokens"])
            usage["call_records"].append(stats["call_record"])
            content, content_state = openrouter_response_content(response)
            if content_state != "ok" or content is None:
                logger.warning(
                    "  Toponymy OpenRouter labeling fallback (model=%s): content_state=%s",
                    self.model,
                    content_state,
                )
                return ""
            return content

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
    embedding_provider: ToponymyEmbeddingProvider | str,
    backend: ToponymyBackend | str,
    api_key: str | None,
) -> tuple[ToponymyLLMProvider, ToponymyEmbeddingProvider, ToponymyBackend]:
    """Normalize and validate Toponymy backend/provider selections."""
    llm_provider_norm = llm_provider.strip().lower()
    if llm_provider_norm not in TOPONYMY_LLM_PROVIDERS:
        allowed = ", ".join(sorted(TOPONYMY_LLM_PROVIDERS))
        raise ValueError(f"Invalid llm_provider '{llm_provider}'. Expected one of: {allowed}.")
    embedding_provider_norm = embedding_provider.strip().lower()
    if embedding_provider_norm not in TOPONYMY_EMBEDDING_PROVIDERS:
        allowed = ", ".join(sorted(TOPONYMY_EMBEDDING_PROVIDERS))
        raise ValueError(f"Invalid embedding_provider '{embedding_provider}'. Expected one of: {allowed}.")
    if (llm_provider_norm == "openrouter" or embedding_provider_norm == "openrouter") and not api_key:
        raise ValueError("api_key is required for Toponymy with OpenRouter.")

    backend_norm = backend.strip().lower()
    if backend_norm != "toponymy":
        raise ValueError(f"Invalid backend '{backend}'. Expected 'toponymy'.")

    return (
        cast(ToponymyLLMProvider, llm_provider_norm),
        cast(ToponymyEmbeddingProvider, embedding_provider_norm),
        cast(ToponymyBackend, backend_norm),
    )


def _build_toponymy_clusterer(
    *,
    clusterer_params: dict[str, Any],
    toponymy_clusterer_cls: Any,
    clusterable_vectors: np.ndarray,
) -> tuple[Any, np.ndarray]:
    """Create configured Toponymy clusterer and vectors for fit."""
    clusterer = _instantiate_with_filtered_kwargs(
        toponymy_clusterer_cls,
        clusterer_params,
        component_name="ToponymyClusterer",
    )
    logger.info("  Clusterer: %s", clusterer.__class__.__name__)
    return clusterer, clusterable_vectors


_TOPONYMY_FIRST_LAYER_ERROR = "Not enough clusters found in the first layer"


def normalize_toponymy_layer_index(layer_index: ToponymyLayerIndex | str) -> int | Literal["auto"]:
    """Normalize user-provided Toponymy layer selection to ``int`` or ``"auto"``."""
    if layer_index is None:
        return "auto"

    if isinstance(layer_index, str):
        value = layer_index.strip().lower()
        if value in {"", "auto", "none", "null"}:
            return "auto"
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid toponymy_layer_index '{layer_index}'. Expected an integer or 'auto'."
            ) from exc

    if isinstance(layer_index, np.integer):
        return int(layer_index)
    if isinstance(layer_index, int):
        return layer_index

    raise TypeError(
        f"Invalid toponymy_layer_index type {type(layer_index).__name__}. "
        "Expected int, 'auto', or null."
    )


def _topic_count_from_labels(labels: np.ndarray) -> int:
    """Count non-outlier topics in one Toponymy layer."""
    unique = set(int(label) for label in np.asarray(labels, dtype=int))
    return len(unique) - (1 if -1 in unique else 0)


def _clusters_per_toponymy_layer(topic_model: Any) -> list[int]:
    """Return the number of non-outlier topics per Toponymy layer."""
    cluster_layers = getattr(topic_model, "cluster_layers_", None) or []
    return [
        _topic_count_from_labels(getattr(layer, "cluster_labels", np.array([], dtype=int)))
        for layer in cluster_layers
    ]


def _select_toponymy_layer_index(
    *,
    requested_layer_index: ToponymyLayerIndex | str,
    n_layers: int,
) -> tuple[int, Literal["auto", "manual"]]:
    """Resolve the active Toponymy layer index from explicit or automatic selection."""
    normalized = normalize_toponymy_layer_index(requested_layer_index)
    if normalized == "auto":
        return n_layers - 1, "auto"
    return normalized, "manual"


def get_toponymy_hierarchy_metadata(topic_model: Any) -> dict[str, Any] | None:
    """Return persisted Toponymy hierarchy metadata when available."""
    cluster_layers = getattr(topic_model, "cluster_layers_", None)
    if cluster_layers is None:
        return None

    layer_count = int(getattr(topic_model, "topic_layer_count_", len(cluster_layers)))
    primary_layer_index = getattr(topic_model, "topic_primary_layer_index_", None)
    selection = getattr(topic_model, "topic_primary_layer_selection_", None)
    clusters_per_layer = getattr(
        topic_model,
        "topic_clusters_per_layer_",
        _clusters_per_toponymy_layer(topic_model),
    )
    if primary_layer_index is None:
        return None

    return {
        "topic_layer_count": layer_count,
        "topic_primary_layer_index": int(primary_layer_index),
        "topic_clusters_per_layer": [int(value) for value in clusters_per_layer],
        "topic_primary_layer_selection": str(selection or "manual"),
    }


def _resolve_toponymy_cluster_param(
    *,
    clusterer: Any,
    clusterer_params: dict[str, Any],
    key: str,
) -> Any:
    """Resolve active Toponymy cluster parameter from explicit or instantiated config."""
    value = clusterer_params.get(key)
    if value is not None:
        return value

    clusterer_value = getattr(clusterer, key, None)
    if clusterer_value is not None:
        return clusterer_value

    kwargs = getattr(clusterer, "kwargs", None)
    if isinstance(kwargs, dict):
        return kwargs.get(key)
    return None


def _bridge_toponymy_fit_dtypes(
    *,
    embeddings: np.ndarray,
    clusterable_vectors: np.ndarray,
    backend_norm: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast Toponymy fit vectors to float32 when float16 would break Numba."""
    fit_embeddings = embeddings
    fit_clusterable_vectors = clusterable_vectors
    cast_targets: list[str] = []

    if embeddings.dtype == np.float16:
        fit_embeddings = embeddings.astype(np.float32, copy=False)
        cast_targets.append("embedding_vectors")

    if clusterable_vectors is embeddings:
        fit_clusterable_vectors = fit_embeddings
        if fit_clusterable_vectors is not clusterable_vectors:
            cast_targets.append("clusterable_vectors")
    elif clusterable_vectors.dtype == np.float16:
        fit_clusterable_vectors = clusterable_vectors.astype(np.float32, copy=False)
        cast_targets.append("clusterable_vectors")

    if cast_targets:
        logger.info(
            "  Toponymy dtype bridge | backend=%s | cast=%s -> float32",
            backend_norm,
            ",".join(cast_targets),
        )

    return fit_embeddings, fit_clusterable_vectors


def _build_toponymy_models(
    *,
    llm_provider_norm: str,
    embedding_provider_norm: str,
    llm_model: str,
    llm_model_spec: ModelSpec | None,
    embedding_model: str,
    api_key: str | None,
    openrouter_api_base: str,
    max_workers: int,
    local_llm_max_new_tokens: int,
    cost_tracker: "CostTracker | None",
    llama_server_config: LlamaServerConfig | None,
    runtime_log_path: Path | None,
) -> tuple[Any, dict[str, Any] | None, Any]:
    """Build Toponymy naming and text-embedding components."""
    if llm_provider_norm == "openrouter":
        llm_wrapper, llm_usage = _create_tracked_toponymy_namer(
            model=llm_model,
            api_key=api_key,
            base_url=openrouter_api_base,
            max_workers=max_workers,
        )
    elif llm_provider_norm == "llama_server":
        try:
            from toponymy.llm_wrappers import OpenAINamer
        except ImportError as exc:
            raise ImportError(
                "llm_provider='llama_server' requires optional dependency 'openai' "
                "and Toponymy's OpenAINamer wrapper."
            ) from exc
        if llm_model_spec is None:
            raise ValueError("llama_server provider requires a resolved ModelSpec.")
        handle = ensure_llama_server(
            model_spec=llm_model_spec,
            config=llama_server_config or LlamaServerConfig(),
            runtime_log_path=runtime_log_path,
        )
        llm_wrapper = OpenAINamer(
            api_key="local",
            model=Path(llm_model_spec.resolve()).name,
            base_url=handle.base_url,
        )
        if cost_tracker is not None:
            logger.info(
                "  llama_server Toponymy LLM selected; token/cost tracking is unavailable for this step."
            )
        llm_usage = None
    else:
        try:
            from toponymy.llm_wrappers import HuggingFaceNamer
        except Exception as exc:
            raise ImportError(
                "llm_provider='local' requires optional dependencies "
                "'transformers' and Toponymy's HuggingFaceNamer wrapper."
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
        local_generator = getattr(llm_wrapper, "llm", None)
        if local_generator is not None:
            _configure_deterministic_generation_pipeline(local_generator)

        llm_usage = None
        if cost_tracker is not None:
            logger.info("  Local Toponymy LLM selected; token/cost tracking is unavailable for this step.")
        logger.info(
            "  Local Toponymy LLM max_new_tokens capped at %s per naming call.",
            local_llm_max_new_tokens,
        )

    if embedding_provider_norm == "openrouter":
        text_embedding_model = OpenRouterEmbedder(
            api_key=api_key,
            model=embedding_model,
            api_base=openrouter_api_base,
            max_workers=max_workers,
        )
    else:
        text_embedding_model = _load_local_sentence_transformer(
            model=embedding_model,
            use_case="toponymy embeddings",
            missing_dependency_message=(
                "embedding_provider='local' requires optional dependency 'sentence-transformers'."
            ),
            runtime_log_path=runtime_log_path,
        )

    return llm_wrapper, llm_usage, text_embedding_model


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
    llm_provider_norm: str,
    llm_model: str,
    embedding_provider_norm: str,
    embedding_model: str,
    layer_index: ToponymyLayerIndex | str,
    clusterer_params: dict[str, Any],
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    """Fit Toponymy model and extract one configured topic layer."""
    logger.info(
        "Fitting Toponymy backend='%s' (LLM: %s/%s | embeddings: %s/%s) ...",
        backend_norm,
        llm_provider_norm,
        llm_model,
        embedding_provider_norm,
        embedding_model,
    )
    topic_model = toponymy_cls(
        llm_wrapper=llm_wrapper,
        text_embedding_model=text_embedding_model,
        clusterer=clusterer,
        object_description=object_description,
        corpus_description=corpus_description,
        verbose=verbose,
    )
    fit_embeddings, fit_clusterable_vectors = _bridge_toponymy_fit_dtypes(
        embeddings=embeddings,
        clusterable_vectors=clusterable_vectors,
        backend_norm=backend_norm,
    )
    try:
        topic_model.fit(
            documents,
            embedding_vectors=fit_embeddings,
            clusterable_vectors=fit_clusterable_vectors,
        )
    except ValueError as exc:
        if _TOPONYMY_FIRST_LAYER_ERROR in str(exc):
            min_clusters = _resolve_toponymy_cluster_param(
                clusterer=clusterer,
                clusterer_params=clusterer_params,
                key="min_clusters",
            )
            base_min_cluster_size = _resolve_toponymy_cluster_param(
                clusterer=clusterer,
                clusterer_params=clusterer_params,
                key="base_min_cluster_size",
            )
            raise ValueError(
                "Toponymy first-layer clustering failed "
                f"(backend='{backend_norm}', min_clusters={min_clusters}, "
                f"base_min_cluster_size={base_min_cluster_size}). "
                "Try lowering min_clusters and, if needed, lowering base_min_cluster_size."
            ) from exc
        raise

    n_layers = len(topic_model.cluster_layers_)
    if n_layers <= 0:
        raise ValueError(
            "Toponymy returned no layers. "
            "Lower Toponymy cluster thresholds or use toponymy_layer_index='auto'."
        )

    selected_layer_index, selection_mode = _select_toponymy_layer_index(
        requested_layer_index=layer_index,
        n_layers=n_layers,
    )
    if selected_layer_index < 0 or selected_layer_index >= n_layers:
        raise ValueError(
            f"layer_index {selected_layer_index} is out of range for {n_layers} available layers "
            f"(valid range: 0..{n_layers - 1}). "
            "Use toponymy_layer_index='auto' for the coarsest available overview layer."
        )

    topic_model.topic_layer_count_ = n_layers
    topic_model.topic_primary_layer_index_ = selected_layer_index
    topic_model.topic_primary_layer_selection_ = selection_mode
    topic_model.topic_clusters_per_layer_ = _clusters_per_toponymy_layer(topic_model)

    topics = np.asarray(topic_model.cluster_layers_[selected_layer_index].cluster_labels, dtype=int)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = int((topics == -1).sum())
    logger.info(
        "  Toponymy layers=%s | primary_layer=%s (%s) | clusters_per_layer=%s",
        n_layers,
        selected_layer_index,
        selection_mode,
        topic_model.topic_clusters_per_layer_,
    )
    logger.info(
        "  Selected Toponymy layer %s/%s: %s topics, %s outliers.",
        selected_layer_index,
        n_layers - 1,
        n_topics,
        f"{n_outliers:,}",
    )

    topic_names = topic_model.topic_names_[selected_layer_index]
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
    llm_model_repo: str | None = None,
    llm_model_file: str | None = None,
    llm_model_path: str | None = None,
    llm_prompt: str | None = None,
    pipeline_models: list[str] | None = None,
    parallel_models: list[str] | None = None,
    mmr_diversity: float = 0.3,
    llm_nr_docs: int = 8,
    llm_diversity: float = 0.2,
    llm_delay: float = 0.3,
    llm_max_new_tokens: int = DEFAULT_BERTOPIC_LLM_MAX_NEW_TOKENS,
    embedding_model_name: str | None = None,
    keybert_model: str = DEFAULT_KEYBERT_MODEL,
    min_df: int = 2,
    clustering_method: str = "fast_hdbscan",
    clustering_params: dict | None = None,
    top_n_words: int = DEFAULT_BERTOPIC_TOP_N_WORDS,
    pos_spacy_model: str = DEFAULT_POS_SPACY_MODEL,
    show_progress: bool = True,
    api_key: str | None = None,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
    llama_server_config: LlamaServerConfig | None = None,
    runtime_log_path: Path | None = None,
) -> "BERTopic":
    """Fit BERTopic on pre-reduced document vectors.

    Parameters
    ----------
    documents : list[str]
        Input corpus; order must match *reduced_5d*.
    reduced_5d : np.ndarray
        Five-dimensional vectors used directly for clustering.
    llm_provider : str
        Labeling backend: ``"local"``, ``"llama_server"``, ``"huggingface_api"``, ``"openrouter"``.
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
        Required for ``llm_provider="openrouter"`` and ``llm_provider="huggingface_api"``.
    openrouter_cost_mode : str
        Cost resolution mode for OpenRouter usage.
    cost_tracker : CostTracker, optional
        Optional cost accumulator for LLM labeling calls.

    Returns
    -------
    BERTopic
        Fitted BERTopic instance.
    """
    llm_model_spec = None
    if llm_provider == "llama_server":
        llm_model_spec = ModelSpec.from_fields(
            model_repo=llm_model_repo,
            model_file=llm_model_file,
            model_path=llm_model_path,
            legacy_value=llm_model,
            field_label="llm_model",
        )
        llm_model = llm_model_spec.display_name()
    elif llm_provider == "huggingface_api":
        llm_model = normalize_huggingface_model(llm_model)
        api_key = resolve_huggingface_api_key(api_key)
    validate_provider(
        llm_provider,
        valid=set(BERTOPIC_LLM_PROVIDERS),
        api_key=api_key,
        requires_key={"openrouter", "huggingface_api"},
    )
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)

    with capture_external_output(runtime_log_path or get_runtime_log_path()):
        from bertopic import BERTopic
        from bertopic.dimensionality import BaseDimensionalityReduction
        from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    if pipeline_models is None:
        pipeline_models = list(DEFAULT_BERTOPIC_PIPELINE_MODELS)
    if parallel_models is None:
        parallel_models = list(DEFAULT_BERTOPIC_PARALLEL_MODELS)
    logger.info(
        "Preparing BERTopic components (pipeline=%s, parallel=%s) ...",
        pipeline_models,
        parallel_models,
    )

    rep_model = _build_representation_model(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_model_spec=llm_model_spec,
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
        llama_server_config=llama_server_config,
        runtime_log_path=runtime_log_path,
    )

    n_docs = len(documents)
    safe_min_df = min(min_df, max(1, n_docs // 20))
    if safe_min_df != min_df:
        logger.warning("  min_df capped from %d to %d (dataset has %d documents)", min_df, safe_min_df, n_docs)
    vectorizer = CountVectorizer(stop_words="english", min_df=safe_min_df, ngram_range=(1, 3))
    ctfidf = ClassTfidfTransformer()

    emb_model = None
    if "KeyBERT" in pipeline_models or "KeyBERT" in parallel_models:
        emb_model = _load_local_sentence_transformer(
            model=keybert_model,
            use_case="bertopic keybert embeddings",
            missing_dependency_message=(
                "BERTopic keyword helpers require optional dependency 'sentence-transformers'."
            ),
            runtime_log_path=runtime_log_path,
        )
    elif embedding_model_name:
        emb_model = _load_local_sentence_transformer(
            model=embedding_model_name,
            use_case="bertopic embeddings",
            missing_dependency_message=(
                "BERTopic embeddings require optional dependency 'sentence-transformers'."
            ),
            runtime_log_path=runtime_log_path,
        )

    cluster_model = _create_cluster_model(clustering_method, clustering_params)

    topic_model = BERTopic(
        embedding_model=emb_model,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        representation_model=rep_model,
        top_n_words=top_n_words,
        verbose=False,
    )

    logger.info("Fitting BERTopic (LLM: %s/%s) ...", llm_provider, llm_model)

    track_litellm_usage = cost_tracker is not None and llm_provider == "huggingface_api"
    with _track_litellm_usage(enabled=track_litellm_usage) as llm_usage:
        with tqdm(total=1, desc="BERTopic fit", disable=not show_progress) as pbar:
            topic_model.fit_transform(documents, reduced_5d)
            pbar.update(1)

    if llm_provider == "openrouter":
        llm_usage = _consume_openrouter_representation_usage(
            getattr(topic_model, "representation_model", rep_model)
        )

    _record_llm_usage(
        llm_usage,
        step="llm_labeling",
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
        openrouter_cost_mode=openrouter_cost_mode,
        cost_tracker=cost_tracker,
    )

    _strip_think_tags_from_topics(topic_model)

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
    layer_index: ToponymyLayerIndex = "auto",
    llm_provider: ToponymyLLMProvider = "openrouter",
    llm_model: str = "google/gemini-3-flash-preview",
    llm_model_repo: str | None = None,
    llm_model_file: str | None = None,
    llm_model_path: str | None = None,
    embedding_provider: ToponymyEmbeddingProvider = "local",
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
    llama_server_config: LlamaServerConfig | None = None,
    runtime_log_path: Path | None = None,
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    """Fit Toponymy and return topic assignments.

    Parameters
    ----------
    documents : list[str]
        Input corpus; order must match *embeddings* and *clusterable_vectors*.
    embeddings : np.ndarray
        Full embedding matrix used for naming/context.
    clusterable_vectors : np.ndarray
        Vectors used for Toponymy clustering (typically 5D reduced vectors).
    backend : str
        ``"toponymy"``.
    layer_index : int | "auto" | None
        Hierarchical layer index selected as the working-layer compatibility
        view for ``topic_id``/``Name``.
        ``"auto"`` selects the coarsest available overview layer.
    llm_provider : str
        Currently supported Toponymy naming provider.
    llm_model : str
        LLM model identifier for topic naming.
    embedding_provider : str
        Text-embedding provider used by Toponymy internals.
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
    llm_provider_norm, embedding_provider_norm, backend_norm = _normalize_toponymy_inputs(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        backend=backend,
        api_key=api_key,
    )
    llm_model_spec = None
    if llm_provider_norm == "llama_server":
        llm_model_spec = ModelSpec.from_fields(
            model_repo=llm_model_repo,
            model_file=llm_model_file,
            model_path=llm_model_path,
            legacy_value=llm_model,
            field_label="llm_model",
        )
        llm_model = llm_model_spec.display_name()
    try:
        from toponymy import Toponymy, ToponymyClusterer
    except ImportError as exc:
        _raise_with_toponymy_import_hint(exc, backend=backend_norm)

    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    openrouter_api_base = normalize_openrouter_api_base(openrouter_api_base)
    max_workers = max(1, int(max_workers))
    clusterer_params = dict(clusterer_params or {})

    clusterer, clusterable_vectors_for_fit = _build_toponymy_clusterer(
        clusterer_params=clusterer_params,
        toponymy_clusterer_cls=ToponymyClusterer,
        clusterable_vectors=clusterable_vectors,
    )
    llm_wrapper, llm_usage, text_embedding_model = _build_toponymy_models(
        llm_provider_norm=llm_provider_norm,
        embedding_provider_norm=embedding_provider_norm,
        llm_model=llm_model,
        llm_model_spec=llm_model_spec,
        embedding_model=embedding_model,
        api_key=api_key,
        openrouter_api_base=openrouter_api_base,
        max_workers=max_workers,
        local_llm_max_new_tokens=local_llm_max_new_tokens,
        cost_tracker=cost_tracker,
        llama_server_config=llama_server_config,
        runtime_log_path=runtime_log_path,
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
        llm_provider_norm=llm_provider_norm,
        llm_model=llm_model,
        embedding_provider_norm=embedding_provider_norm,
        embedding_model=embedding_model,
        layer_index=layer_index,
        clusterer_params=clusterer_params,
    )

    _record_llm_usage(
        llm_usage,
        step="llm_labeling_toponymy",
        llm_provider=llm_provider_norm,
        llm_model=llm_model,
        api_key=api_key,
        openrouter_cost_mode=openrouter_cost_mode,
        cost_tracker=cost_tracker,
    )

    if (
        cost_tracker is not None
        and embedding_provider_norm == "openrouter"
        and hasattr(text_embedding_model, "usage")
    ):
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


def _invert_cluster_tree(
    cluster_tree: dict[tuple[int, int], list[tuple[int, int]]],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Invert ``{parent: [children]}`` to ``{child: parent}`` for O(1) lookup."""
    child_to_parent: dict[tuple[int, int], tuple[int, int]] = {}
    for parent, children in cluster_tree.items():
        for child in children:
            child_to_parent[child] = parent
    return child_to_parent


def reduce_toponymy_outliers(
    topic_model: Any,
    embeddings: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Reassign Toponymy outliers bottom-up with tree propagation.

    For each layer (finest → coarsest):

    1. **Tree propagation** (layers 1+): docs reassigned at the previous layer
       inherit their parent cluster via ``cluster_tree_``.
    2. **Embedding similarity**: remaining outliers are matched against the
       layer's ``centroid_vectors`` using cosine similarity.

    Parameters
    ----------
    topic_model
        Fitted Toponymy model with ``cluster_layers_``, ``cluster_tree_``,
        ``topic_names_``, and ``topic_primary_layer_index_``.
    embeddings : np.ndarray
        Full embedding vectors (must match dimensionality of
        ``layer.centroid_vectors``).
    threshold : float
        Minimum cosine similarity for reassignment (same semantics as
        the BERTopic ``outlier_threshold``).

    Returns
    -------
    np.ndarray
        Updated topic assignments for the primary (selected) layer.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cluster_layers = topic_model.cluster_layers_
    cluster_tree = getattr(topic_model, "cluster_tree_", None) or {}
    child_to_parent = _invert_cluster_tree(cluster_tree)

    prev_labels: np.ndarray | None = None

    for i, layer in enumerate(cluster_layers):
        labels = layer.cluster_labels
        before = int((labels == -1).sum())

        # Pass A — tree propagation from the layer below
        if i > 0 and prev_labels is not None:
            outlier_and_prev_assigned = (labels == -1) & (prev_labels != -1)
            for doc_idx in np.where(outlier_and_prev_assigned)[0]:
                parent = child_to_parent.get((i - 1, int(prev_labels[doc_idx])))
                if parent is not None and parent[0] == i:
                    labels[doc_idx] = parent[1]

        # Pass B — cosine similarity for remaining outliers
        outlier_mask = labels == -1
        centroids = getattr(layer, "centroid_vectors", None)
        if outlier_mask.any() and centroids is not None and len(centroids) > 0:
            sims = cosine_similarity(
                embeddings[outlier_mask], centroids,
            )
            sims[sims < threshold] = 0
            best = sims.argmax(axis=1)
            assigned = sims.max(axis=1) > 0
            labels[np.where(outlier_mask)[0][assigned]] = best[assigned]

        # Refresh per-doc label strings
        if hasattr(layer, "make_topic_name_vector"):
            layer.make_topic_name_vector()

        after = int((labels == -1).sum())
        logger.info("  Layer %s outliers: %s → %s", i, f"{before:,}", f"{after:,}")
        prev_labels = labels

    primary = getattr(topic_model, "topic_primary_layer_index_", len(cluster_layers) - 1)
    return np.asarray(cluster_layers[primary].cluster_labels, dtype=int)


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

    track_litellm_usage = cost_tracker is not None and llm_provider == "huggingface_api"
    with _track_litellm_usage(enabled=track_litellm_usage) as llm_usage:
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

    if llm_provider == "openrouter":
        llm_usage = _consume_openrouter_representation_usage(
            getattr(topic_model, "representation_model", None)
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


__all__ = [
    "fit_bertopic",
    "fit_toponymy",
    "reduce_outliers",
    "reduce_toponymy_outliers",
]
