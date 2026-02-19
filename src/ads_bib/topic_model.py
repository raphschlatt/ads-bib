"""Step 5 – Topic modeling backends (BERTopic and Toponymy).

Covers embedding, dimensionality reduction, clustering, and LLM-based
topic labeling with three interchangeable providers each for embeddings
and LLM labeling.
"""

from __future__ import annotations

import logging
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ads_bib._utils.openrouter_costs import (
    extract_generation_id,
    extract_response_cost,
    extract_usage_stats,
    normalize_openrouter_api_base,
    normalize_openrouter_cost_mode,
    summarize_openrouter_costs,
)

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
        print(
            f"  OpenRouter cost: ${total:.4f} "
            f"({summary['priced_calls']}/{summary['total_calls']} priced; "
            f"direct={summary['direct_priced_calls']}, "
            f"fetched={summary['fetched_priced_calls']}, "
            f"mode={summary['mode']})"
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
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    model_safe = model.replace("/", "_")
    cache_file = (cache_dir / f"embeddings_{provider}_{model_safe}.npz") if cache_dir else None

    # Check legacy cache (without provider prefix)
    if cache_file and not cache_file.exists():
        legacy = cache_dir / f"embeddings_{model_safe}.npz"
        if legacy.exists():
            print(f"  Using legacy cache: {legacy.name}")
            cache_file = legacy

    if cache_file and cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        print(f"  Loaded embeddings from cache: {cache_file.name}")
        return data["embeddings"]

    print(f"  Computing embeddings with {provider}/{model} ...")

    if provider == "local":
        emb = _embed_local(documents, model, batch_size, dtype)
    elif provider == "huggingface_api":
        emb = _embed_huggingface_api(documents, model, batch_size, dtype)
    elif provider == "openrouter":
        emb, usage = _embed_openrouter(documents, model, batch_size, dtype, api_key)
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
        np.savez_compressed(cache_file, embeddings=emb, model=model, provider=provider)
        print(f"  Saved: {cache_file.name}")

    return emb


def _embed_local(documents, model, batch_size, dtype):
    from sentence_transformers import SentenceTransformer

    print(f"  Loading local model: {model}")
    st = SentenceTransformer(model)
    emb = st.encode(documents, show_progress_bar=True, batch_size=batch_size)
    return emb.astype(dtype)


def _embed_huggingface_api(documents, model, batch_size, dtype):
    import litellm

    all_emb = []
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    for batch in tqdm(batches, desc="Embedding (HF API)"):
        for attempt in range(3):
            try:
                resp = litellm.embedding(model=model, input=batch)
                all_emb.extend(d["embedding"] for d in resp["data"])
                break
            except Exception:
                if attempt == 2:
                    raise
                import time; time.sleep(1 * (attempt + 1))
    return np.array(all_emb, dtype=dtype)


def _embed_openrouter(documents, model, batch_size, dtype, api_key):
    import litellm

    # litellm routes via "openrouter/" prefix automatically
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"

    all_emb = []
    total_pt, total_tokens = 0, 0
    call_records = []
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    for batch in tqdm(batches, desc="Embedding (OpenRouter)"):
        for attempt in range(3):
            try:
                resp = litellm.embedding(
                    model=model, input=batch,
                    api_key=api_key,
                )
                all_emb.extend(d["embedding"] for d in resp["data"])
                usage = extract_usage_stats(resp)
                total_pt += usage["prompt_tokens"]
                total_tokens += usage["total_tokens"]
                call_records.append(
                    {
                        "generation_id": extract_generation_id(resp),
                        "direct_cost": extract_response_cost(response=resp),
                    }
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                import time; time.sleep(1 * (attempt + 1))
    usage = {
        "prompt_tokens": total_pt,
        "total_tokens": total_tokens,
        "call_records": call_records,
    }
    return np.array(all_emb, dtype=dtype), usage


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    *,
    method: str = "pacmap",
    params_5d: dict | None = None,
    params_2d: dict | None = None,
    cache_dir: Path | None = None,
    cache_suffix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 5-D (for clustering) and 2-D (for visualisation) projections.

    Parameters
    ----------
    method : str
        ``"pacmap"`` or ``"umap"``.
    params_5d, params_2d : dict, optional
        Override default parameters for each projection.
    cache_dir : Path, optional
        Directory for ``.npy`` caches.
    cache_suffix : str
        Appended to cache filenames for uniqueness.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(reduced_5d, reduced_2d)``.
    """
    r5 = _reduce(embeddings, 5, method, params_5d or {}, cache_dir, f"5d_{cache_suffix}")
    r2 = _reduce(embeddings, 2, method, params_2d or {}, cache_dir, f"2d_{cache_suffix}")
    print(f"  5D shape: {r5.shape}, 2D shape: {r2.shape}")
    return r5, r2


def _reduce(embeddings, n_components, method, params, cache_dir, name):
    if cache_dir:
        path = cache_dir / f"reduced_{name}.npy"
        if path.exists():
            print(f"  Loaded {name} from cache")
            return np.load(path)

    print(f"  Computing {name} with {method.upper()} ...")
    if method == "pacmap":
        import pacmap
        defaults = dict(n_components=n_components, n_neighbors=60, MN_ratio=0.5,
                        FP_ratio=1.0, random_state=42, verbose=True)
        defaults.update(params)
        defaults["n_components"] = n_components
        model = pacmap.PaCMAP(**defaults)
    elif method == "umap":
        from umap import UMAP
        defaults = dict(n_components=n_components, n_neighbors=80, min_dist=0.05,
                        metric="cosine", random_state=42, verbose=True)
        defaults.update(params)
        defaults["n_components"] = n_components
        model = UMAP(**defaults)
    else:
        raise ValueError(f"Unknown dim reduction method: {method}")

    reduced = model.fit_transform(embeddings)

    if cache_dir:
        path = cache_dir / f"reduced_{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, reduced)
        print(f"  Saved: {path.name}")

    return reduced


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _create_cluster_model(method: str, params: dict | None = None):
    """Build the configured clustering model once for all topic workflows."""
    params = params or {}

    if method == "fast_hdbscan":
        import fast_hdbscan

        defaults = dict(
            min_cluster_size=180,
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
            min_cluster_size=180,
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
    print(f"  {method.upper()}: {n_topics} topics, {n_outliers:,} outliers ({n_outliers / len(clusters) * 100:.1f}%)")
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

    Returns
    -------
    BERTopic
        Fitted topic model.
    """
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    pipeline_models = pipeline_models or ["POS", "KeyBERT", "MMR"]
    parallel_models = parallel_models or ["MMR", "POS", "KeyBERT"]
    print(
        "Preparing BERTopic components "
        f"(pipeline={pipeline_models}, parallel={parallel_models}) ..."
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
        top_n_words=20,
        verbose=True,
    )

    print(f"Fitting BERTopic (LLM: {llm_provider}/{llm_model}) ...")

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

    def _cost_cb(kwargs, response, start_time, end_time):
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
) -> tuple[Any, dict]:
    """Create a Toponymy-compatible OpenAI wrapper and capture usage/cost metadata."""
    from openai import OpenAI
    from toponymy.llm_wrappers import LLMWrapper

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}

    class TrackedOpenAINamer(LLMWrapper):
        def __init__(self):
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model

        def _record_usage(self, response: Any) -> None:
            stats = extract_usage_stats(response)
            usage["prompt_tokens"] += stats["prompt_tokens"]
            usage["completion_tokens"] += stats["completion_tokens"]
            usage["call_records"].append(
                {
                    "generation_id": extract_generation_id(response),
                    "direct_cost": extract_response_cost(response=response),
                    "response": response,
                }
            )

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            self._record_usage(response)
            return response.choices[0].message.content or ""

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            self._record_usage(response)
            return response.choices[0].message.content or ""

    return TrackedOpenAINamer(), usage


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


def _patch_clusterer_for_toponymy_kwargs(clusterer: Any) -> None:
    """Make strict clusterer methods tolerant to extra kwargs from Toponymy.fit.

    Some Toponymy versions pass layer kwargs (e.g. ``exemplar_delimiters``,
    ``prompt_format``) into ``clusterer.fit_predict``. Older EVoCClusterer
    implementations do not accept these keywords and raise ``TypeError``.
    This shim drops unsupported kwargs while keeping supported ones intact.
    """

    def _wrap_bound(method: Any) -> Any:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return method

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return method

        accepted = set(sig.parameters.keys())

        def _wrapped(*args, **kwargs):
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
    openrouter_api_base: str = "https://openrouter.ai/api/v1",
    openrouter_cost_mode: str = "hybrid",
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
        Currently only ``"openrouter"`` is supported.

    Returns
    -------
    tuple
        ``(toponymy_model, topics, topic_info)`` where ``topic_info`` has
        BERTopic-compatible columns ``Topic`` and ``Name``.
    """
    if llm_provider != "openrouter":
        raise ValueError(
            "fit_toponymy currently supports llm_provider='openrouter' only."
        )
    if not api_key:
        raise ValueError("api_key is required for Toponymy with OpenRouter.")

    backend_norm = backend.strip().lower()
    if backend_norm not in {"toponymy", "toponymy_evoc"}:
        raise ValueError(
            f"Invalid backend '{backend}'. Expected 'toponymy' or 'toponymy_evoc'."
        )

    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.embedding_wrappers import OpenAIEmbedder

    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    openrouter_api_base = normalize_openrouter_api_base(openrouter_api_base)
    clusterer_params = dict(clusterer_params or {})

    if backend_norm == "toponymy":
        clusterer = ToponymyClusterer(**clusterer_params)
    else:
        try:
            from toponymy.clustering import EVoCClusterer
        except Exception as exc:
            raise ImportError(
                "backend='toponymy_evoc' requires optional dependency 'evoc' and "
                "a Toponymy version that exposes EVoCClusterer."
            ) from exc
        clusterer = EVoCClusterer(**clusterer_params)
        _patch_clusterer_for_toponymy_kwargs(clusterer)

    llm_wrapper, llm_usage = _create_tracked_toponymy_namer(
        model=llm_model,
        api_key=api_key,
        base_url=openrouter_api_base,
    )
    text_embedding_model = OpenAIEmbedder(
        api_key=api_key,
        model=embedding_model,
        base_url=openrouter_api_base,
    )

    print(f"Fitting Toponymy backend='{backend_norm}' (LLM: {llm_provider}/{llm_model}) ...")
    topic_model = Toponymy(
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
    print(
        f"  Selected Toponymy layer {layer_index}/{n_layers - 1}: "
        f"{n_topics} topics, {n_outliers:,} outliers."
    )

    topic_names = topic_model.topic_names_[layer_index]
    topic_info = _build_toponymy_topic_info(topics, topic_names)

    _record_llm_usage(
        llm_usage,
        step="llm_labeling_toponymy_evoc" if backend_norm == "toponymy_evoc" else "llm_labeling_toponymy",
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
        openrouter_cost_mode=openrouter_cost_mode,
        cost_tracker=cost_tracker,
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
        def filter(self, record):
            return target not in record.getMessage()

    warning_filter = _MessageFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


def _build_representation_model(*, llm_provider, llm_model, llm_prompt,
                                  pipeline_models, parallel_models,
                                  mmr_diversity, llm_nr_docs, llm_diversity,
                                  llm_delay, keybert_model, api_key):
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
        print("  Initializing POS keyword extraction model: en_core_web_sm")

    # Pipeline (sequential before LLM)
    pipe = []
    for name in pipeline_models:
        if name == "POS":
            pipe.append(PartOfSpeech("en_core_web_sm"))
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
            result["POS"] = PartOfSpeech("en_core_web_sm")
        elif name == "KeyBERT":
            from bertopic.representation import KeyBERTInspired
            result["KeyBERT"] = KeyBERTInspired()

    return result


def _create_llm(provider, model, prompt, nr_docs, diversity, delay, api_key):
    if provider == "local":
        from transformers import pipeline as hf_pipeline
        from bertopic.representation import TextGeneration

        print(f"  Loading local LLM: {model}")
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
    print(f"  Outliers: {before:,} → {after:,}")
    print("  Refreshing topic representations after outlier reassignment (not a full BERTopic refit).")
    print("  Topic reduction must happen before this step when using manual topic assignments.")

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

    return df
