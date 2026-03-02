"""Step 3 – Language detection and translation of non-English titles/abstracts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Literal, TypeAlias, TypedDict

import pandas as pd
from tqdm.auto import tqdm

from ads_bib._utils.cleaning import require_columns as _require_columns
from ads_bib._utils.openrouter_client import (
    openrouter_chat_completion,
    openrouter_usage_from_response,
)
from ads_bib._utils.openrouter_costs import (
    DEFAULT_OPENROUTER_API_BASE,
    normalize_openrouter_api_base,
    normalize_openrouter_cost_mode,
    resolve_openrouter_costs,
)
from ads_bib._utils import local_runtime

logger = logging.getLogger(__name__)

TranslationProvider: TypeAlias = Literal["openrouter", "gguf"]
GGUFParallelPolicy: TypeAlias = local_runtime.GGUFParallelPolicy
GGUFTokenBudgetMode: TypeAlias = local_runtime.GGUFTokenBudgetMode


class TranslationCostInfo(TypedDict):
    """Structured cost summary returned by :func:`translate_dataframe`."""

    prompt_tokens: int
    completion_tokens: int
    provider: TranslationProvider
    model: str
    cost_usd: float | None
    cost_mode: str | None
    cost_summary: dict[str, object] | None

# ---------------------------------------------------------------------------
# Language detection  (fasttext – fast & free)
# ---------------------------------------------------------------------------

_ft_model = None
_fasttext_predict_needs_compat = False
_fasttext_numpy2_warning_emitted = False
_FASTTEXT_NUMPY2_COPY_ERROR = "Unable to avoid copy while creating an array as requested."


def _get_ft_model(model_path: str | Path | None = None):
    """Lazy-load the fasttext language-identification model."""
    global _ft_model
    if _ft_model is None:
        import fasttext

        if model_path is None:
            model_path = "lid.176.bin"
        _ft_model = fasttext.load_model(str(model_path))
    return _ft_model


def _predict_language_fasttext_compat(model: object, text: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Fallback fasttext prediction path compatible with NumPy 2."""
    if "\n" in text:
        raise ValueError("predict processes one line at a time (remove '\\n')")
    predictions = model.f.predict(f"{text}\n", 1, 0.0, "strict")
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = ((), ())
    return labels, probs


def _predict_language(text: str, model_path: str | Path | None = None) -> str:
    global _fasttext_predict_needs_compat, _fasttext_numpy2_warning_emitted
    model = _get_ft_model(model_path)
    if _fasttext_predict_needs_compat:
        label, _ = _predict_language_fasttext_compat(model, text)
        return label[0].replace("__label__", "")
    try:
        label, _ = model.predict(text)
    except ValueError as exc:
        if _FASTTEXT_NUMPY2_COPY_ERROR not in str(exc):
            raise
        _fasttext_predict_needs_compat = True
        if not _fasttext_numpy2_warning_emitted:
            logger.warning(
                "fasttext-wheel detected NumPy 2 copy incompatibility; using compatibility path."
            )
            _fasttext_numpy2_warning_emitted = True
        label, _ = _predict_language_fasttext_compat(model, text)
    return label[0].replace("__label__", "")


def detect_languages(
    df: pd.DataFrame,
    columns: list[str],
    *,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add ``{col}_lang`` columns with ISO-639-1 language codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Column names whose language should be detected (e.g. ``["Title", "Abstract"]``).
    model_path : str or Path, optional
        Path to ``lid.176.bin``.  Falls back to CWD.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added language columns.
    """
    _require_columns(df, columns, function_name="detect_languages")
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna("")
        df[f"{col}_lang"] = df[col].apply(lambda t: _predict_language(t, model_path))
        n_non_en = (df[f"{col}_lang"] != "en").sum()
        logger.info("  %s: %s non-English entries detected", col, f"{n_non_en:,}")
    return df


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a highly accurate translator specializing in scientific and "
    "technical texts. Only translate the text. Do not comment or provide "
    "additional information."
)
DEFAULT_TRANSLATION_MAX_TOKENS = 2048
DEFAULT_GGUF_CHUNK_INPUT_TOKENS = 384
DEFAULT_GGUF_CHUNK_OVERLAP_TOKENS = 48
DEFAULT_GGUF_TRANSLATION_RETRIES = 2
DEFAULT_GGUF_PARALLEL_POLICY: GGUFParallelPolicy = "auto_calibrated"
DEFAULT_GGUF_WARMUP_BUDGET_SECONDS = 60
DEFAULT_GGUF_TITLE_MAX_TOKENS = 128
DEFAULT_GGUF_ABSTRACT_MAX_TOKENS = 768
DEFAULT_GGUF_CALIBRATION_MAX_TOKENS = 64
DEFAULT_GGUF_SHORT_TEXT_CHAR_THRESHOLD = 900


@dataclass(frozen=True)
class _GGUFRuntimeConfig:
    effective_workers: int
    n_ctx: int
    n_threads: int | None
    n_threads_batch: int | None
    policy: GGUFParallelPolicy
    gpu_offload_supported: bool
    calibrated: bool
    token_budget_mode: GGUFTokenBudgetMode
    auto_chunk: bool
    chunk_input_tokens: int
    chunk_overlap_tokens: int
    short_text_char_threshold: int


_GGUF_AUTOCAL_CACHE: dict[tuple[str, int, str, int, int], tuple[int, int, int]] = {}


def _make_gguf_runtime_config(
    *,
    plan: local_runtime.GGUFRuntimePlan,
    auto_chunk: bool,
    chunk_input_tokens: int,
    chunk_overlap_tokens: int,
) -> _GGUFRuntimeConfig:
    return _GGUFRuntimeConfig(
        effective_workers=max(1, int(plan.workers)),
        n_ctx=max(1, int(plan.n_ctx)),
        n_threads=max(1, int(plan.threads)),
        n_threads_batch=max(1, int(plan.threads_batch)),
        policy=plan.policy,
        gpu_offload_supported=bool(plan.gpu_offload_supported),
        calibrated=bool(plan.calibrated),
        token_budget_mode=plan.token_budget_mode,
        auto_chunk=bool(auto_chunk),
        chunk_input_tokens=max(1, int(chunk_input_tokens)),
        chunk_overlap_tokens=max(0, int(chunk_overlap_tokens)),
        short_text_char_threshold=max(
            DEFAULT_GGUF_SHORT_TEXT_CHAR_THRESHOLD,
            int(chunk_input_tokens) * 4,
        ),
    )


def _init_gguf_pool_worker(
    model_path: str,
    n_ctx: int,
    n_threads: int | None,
    n_threads_batch: int | None,
    preload_tokenizer: bool,
) -> None:
    from ads_bib._utils.gguf_backend import prime_gguf_translation_runtime

    prime_gguf_translation_runtime(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
        preload_tokenizer=preload_tokenizer,
    )


def _collect_calibration_samples(
    df: pd.DataFrame,
    *,
    columns: list[str],
    target_lang: str,
    limit: int = 6,
) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for col in columns:
        lang_col = f"{col}_lang"
        if lang_col not in df.columns:
            continue
        mask = (df[lang_col] != target_lang) & df[col].notna() & (df[col] != "")
        for _, row in df.loc[mask, [col, lang_col]].head(limit).iterrows():
            samples.append((str(row[col]), str(row[lang_col])))
            if len(samples) >= limit:
                return samples
    return samples


def _benchmark_runtime_candidate(
    *,
    model_path: str,
    target_lang: str,
    sample_texts: list[tuple[str, str]],
    workers: int,
    threads: int,
    threads_batch: int,
    n_ctx: int,
) -> float:
    """Return candidate docs/sec throughput using a short synthetic run."""
    if not sample_texts:
        return 0.0
    runtime = _GGUFRuntimeConfig(
        effective_workers=max(1, int(workers)),
        n_ctx=max(1, int(n_ctx)),
        n_threads=max(1, int(threads)),
        n_threads_batch=max(1, int(threads_batch)),
        policy="auto_calibrated",
        gpu_offload_supported=False,
        calibrated=False,
        token_budget_mode="global",
        auto_chunk=False,
        chunk_input_tokens=DEFAULT_GGUF_CHUNK_INPUT_TOKENS,
        chunk_overlap_tokens=DEFAULT_GGUF_CHUNK_OVERLAP_TOKENS,
        short_text_char_threshold=DEFAULT_GGUF_SHORT_TEXT_CHAR_THRESHOLD,
    )
    n_tasks = 1 if workers <= 1 else min(max(2, workers), 4)
    payloads = []
    for i in range(n_tasks):
        text, src_lang = sample_texts[i % len(sample_texts)]
        payloads.append(
            (
                i,
                text,
                src_lang,
                target_lang,
                model_path,
                DEFAULT_GGUF_CALIBRATION_MAX_TOKENS,
                runtime,
            )
        )

    start = time.perf_counter()
    if workers <= 1:
        for payload in payloads:
            _, translated, _, err = _translate_gguf_worker_task(payload)
            if translated is None:
                raise RuntimeError(err or "calibration failed")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_gguf_pool_worker,
            initargs=(
                model_path,
                n_ctx,
                threads,
                threads_batch,
                False,
            ),
        ) as pool:
            futures = [pool.submit(_translate_gguf_worker_task, payload) for payload in payloads]
            for future in as_completed(futures):
                _, translated, _, err = future.result()
                if translated is None:
                    raise RuntimeError(err or "calibration failed")
    elapsed = max(1e-9, time.perf_counter() - start)
    return len(payloads) / elapsed


def _resolve_auto_calibrated_runtime(
    *,
    max_workers: int,
    model_path: str,
    n_ctx: int,
    n_threads: int | None,
    n_threads_batch: int | None,
    token_budget_mode: GGUFTokenBudgetMode,
    warmup_budget_seconds: int,
    calibration_samples: list[tuple[str, str]],
    target_lang: str,
    auto_chunk: bool,
    chunk_input_tokens: int,
    chunk_overlap_tokens: int,
) -> _GGUFRuntimeConfig:
    from ads_bib._utils.gguf_backend import gguf_supports_gpu_offload

    gpu_supported = gguf_supports_gpu_offload()
    if gpu_supported:
        plan = local_runtime.resolve_gguf_runtime_plan(
            max_workers=max_workers,
            policy="stability_first",
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            token_budget_mode=token_budget_mode,
            gpu_offload_supported=True,
            calibrated=False,
        )
        return _make_gguf_runtime_config(
            plan=plan,
            auto_chunk=auto_chunk,
            chunk_input_tokens=chunk_input_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )

    cpu_total = local_runtime.cpu_count()
    cache_key = (
        model_path,
        cpu_total,
        local_runtime.gguf_provider_build_tag(),
        int(max_workers),
        int(n_ctx),
    )
    cached = _GGUF_AUTOCAL_CACHE.get(cache_key)
    if cached is not None:
        workers_cached, threads_cached, threads_batch_cached = cached
        plan = local_runtime.resolve_gguf_runtime_plan(
            max_workers=workers_cached,
            policy="max_throughput",
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=threads_cached,
            n_threads_batch=threads_batch_cached,
            token_budget_mode=token_budget_mode,
            gpu_offload_supported=False,
            calibrated=True,
        )
        plan = local_runtime.GGUFRuntimePlan(
            workers=workers_cached,
            threads=threads_cached,
            threads_batch=threads_batch_cached,
            n_ctx=plan.n_ctx,
            policy="auto_calibrated",
            gpu_offload_supported=False,
            calibrated=True,
            token_budget_mode=token_budget_mode,
        )
        return _make_gguf_runtime_config(
            plan=plan,
            auto_chunk=auto_chunk,
            chunk_input_tokens=chunk_input_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )

    fallback_plan = local_runtime.resolve_gguf_runtime_plan(
        max_workers=max_workers,
        policy="stability_first",
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
        token_budget_mode=token_budget_mode,
        gpu_offload_supported=False,
        calibrated=False,
    )
    if not calibration_samples:
        return _make_gguf_runtime_config(
            plan=fallback_plan,
            auto_chunk=auto_chunk,
            chunk_input_tokens=chunk_input_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )

    best: tuple[float, int, int, int] | None = None
    deadline = time.perf_counter() + max(1, int(warmup_budget_seconds))
    for cand_workers, cand_threads in local_runtime.build_gguf_calibration_candidates(
        max_workers=max_workers,
        cpu_total=cpu_total,
    ):
        if time.perf_counter() >= deadline:
            break
        remaining = deadline - time.perf_counter()
        if remaining < 5:
            break
        cand_threads_batch = max(1, min(cpu_total, max(cand_threads, cand_threads * 2)))
        try:
            score = _benchmark_runtime_candidate(
                model_path=model_path,
                target_lang=target_lang,
                sample_texts=calibration_samples,
                workers=cand_workers,
                threads=cand_threads,
                threads_batch=cand_threads_batch,
                n_ctx=n_ctx,
            )
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, cand_workers, cand_threads, cand_threads_batch)

    if best is None:
        return _make_gguf_runtime_config(
            plan=fallback_plan,
            auto_chunk=auto_chunk,
            chunk_input_tokens=chunk_input_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )

    _, win_workers, win_threads, win_threads_batch = best
    _GGUF_AUTOCAL_CACHE[cache_key] = (win_workers, win_threads, win_threads_batch)
    calibrated_plan = local_runtime.GGUFRuntimePlan(
        workers=max(1, int(win_workers)),
        threads=max(1, int(win_threads)),
        threads_batch=max(1, int(win_threads_batch)),
        n_ctx=max(1, int(n_ctx)),
        policy="auto_calibrated",
        gpu_offload_supported=False,
        calibrated=True,
        token_budget_mode=token_budget_mode,
    )
    return _make_gguf_runtime_config(
        plan=calibrated_plan,
        auto_chunk=auto_chunk,
        chunk_input_tokens=chunk_input_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )


def _resolve_gguf_runtime(
    *,
    max_workers: int,
    policy: GGUFParallelPolicy,
    model_path: str,
    n_ctx: int,
    n_threads: int | None,
    n_threads_batch: int | None,
    token_budget_mode: GGUFTokenBudgetMode,
    warmup_budget_seconds: int,
    calibration_samples: list[tuple[str, str]],
    target_lang: str,
    auto_chunk: bool,
    chunk_input_tokens: int,
    chunk_overlap_tokens: int,
) -> _GGUFRuntimeConfig:
    from ads_bib._utils.gguf_backend import gguf_supports_gpu_offload

    if policy == "auto_calibrated":
        return _resolve_auto_calibrated_runtime(
            max_workers=max_workers,
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            token_budget_mode=token_budget_mode,
            warmup_budget_seconds=warmup_budget_seconds,
            calibration_samples=calibration_samples,
            target_lang=target_lang,
            auto_chunk=auto_chunk,
            chunk_input_tokens=chunk_input_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )

    plan = local_runtime.resolve_gguf_runtime_plan(
        max_workers=max_workers,
        policy=policy,
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
        token_budget_mode=token_budget_mode,
        gpu_offload_supported=gguf_supports_gpu_offload(),
        calibrated=False,
    )
    return _make_gguf_runtime_config(
        plan=plan,
        auto_chunk=auto_chunk,
        chunk_input_tokens=chunk_input_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )


import threading

_thread_local = threading.local()


def _get_openai_client(api_key: str, api_base: str):
    """Get one OpenAI client per worker thread."""
    from openai import OpenAI

    normalized_base = normalize_openrouter_api_base(api_base)
    cache_key = (api_key, normalized_base)
    existing_key = getattr(_thread_local, "openai_client_key", None)
    if existing_key != cache_key:
        _thread_local.openai_client = OpenAI(api_key=api_key, base_url=normalized_base)
        _thread_local.openai_client_key = cache_key
    return _thread_local.openai_client


def _translate_openrouter(
    text: str,
    target_lang: str,
    model: str,
    api_key: str,
    api_base: str = DEFAULT_OPENROUTER_API_BASE,
    *,
    max_tokens: int = DEFAULT_TRANSLATION_MAX_TOKENS,
) -> tuple[str, int, int, str | None, float | None]:
    """Translate *text* via OpenRouter."""
    client = _get_openai_client(api_key, api_base)
    resp = openrouter_chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate the following scientific text to {target_lang}:\n\n{text}"},
        ],
        max_tokens=max_tokens,
        temperature=0,
        retry_label="OpenRouter translation call",
    )
    translated = resp.choices[0].message.content.strip()
    usage = openrouter_usage_from_response(resp)
    pt = usage["prompt_tokens"]
    ct = usage["completion_tokens"]
    gen_id = usage["call_record"]["generation_id"]
    direct_cost = usage["call_record"]["direct_cost"]
    return translated, pt, ct, gen_id, direct_cost


def _prepare_translation_targets(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_lang: str,
) -> tuple[str, pd.DataFrame]:
    """Initialize translated column and return rows that need translation."""
    lang_col = f"{source_col}_lang"
    if source_col not in df.columns:
        raise ValueError(
            f"Column '{source_col}' not found. Ensure source columns exist before translation."
        )
    target_col = f"{source_col}_{target_lang}"
    if lang_col not in df.columns:
        raise ValueError(f"Column '{lang_col}' not found. Run detect_languages() first.")
    df[target_col] = df[source_col]
    mask = (df[lang_col] != target_lang) & df[source_col].notna() & (df[source_col] != "")
    return target_col, df.loc[mask]


def _translate_rows_openrouter(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    to_translate: pd.DataFrame,
    target_lang: str,
    model: str,
    api_key: str | None,
    api_base: str,
    max_workers: int,
    max_tokens: int,
) -> tuple[int, int, list[dict[str, str | float | None]], list[tuple[object, str]]]:
    """Translate selected rows with OpenRouter and return usage/call metadata."""

    def _do_translate(idx_text: tuple[object, object]) -> tuple[object, str | None, int, int, str | None, float | None, str | None]:
        idx, text = idx_text
        try:
            translated, pt, ct, gen_id, direct_cost = _translate_openrouter(
                str(text),
                target_lang,
                model,
                api_key,
                api_base,
                max_tokens=max_tokens,
            )
            return idx, translated, pt, ct, gen_id, direct_cost, None
        except Exception as exc:
            return idx, None, 0, 0, None, None, f"{type(exc).__name__}: {exc}"

    total_pt = 0
    total_ct = 0
    call_records: list[dict[str, str | float | None]] = []
    failed: list[tuple[object, str]] = []

    items = list(zip(to_translate.index, to_translate[source_col]))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_do_translate, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(items), desc=f"  {source_col}"):
            idx, translated, pt, ct, gen_id, direct_cost, error_msg = future.result()
            if translated is not None:
                df.at[idx, target_col] = translated
                call_records.append({"generation_id": gen_id, "direct_cost": direct_cost})
            else:
                failed.append((idx, error_msg or "unknown error"))
            total_pt += pt
            total_ct += ct

    return total_pt, total_ct, call_records, failed


def _merge_translated_chunks(chunks: list[str]) -> str:
    """Merge chunk translations while trimming simple duplicated overlap text."""
    if not chunks:
        return ""
    merged = chunks[0].strip()
    for chunk in chunks[1:]:
        current = chunk.strip()
        if not current:
            continue
        prev_words = merged.split()
        next_words = current.split()
        overlap = 0
        max_overlap = min(24, len(prev_words), len(next_words))
        if max_overlap > 0:
            prev_fold = [w.casefold() for w in prev_words]
            next_fold = [w.casefold() for w in next_words]
            for n in range(max_overlap, 0, -1):
                if prev_fold[-n:] == next_fold[:n]:
                    overlap = n
                    break
        if overlap:
            next_words = next_words[overlap:]
        if next_words:
            merged = f"{merged} {' '.join(next_words)}".strip()
    return merged.strip()


def _translate_text_with_gguf(
    text: str,
    *,
    target_lang: str,
    source_lang: str,
    model_path: str,
    max_tokens: int,
    runtime: _GGUFRuntimeConfig,
    retries: int = DEFAULT_GGUF_TRANSLATION_RETRIES,
) -> tuple[str, int]:
    from ads_bib._utils.gguf_backend import split_text_by_gguf_tokens, translate_gguf

    chunks = [str(text)]
    if runtime.auto_chunk and len(str(text)) > runtime.short_text_char_threshold:
        chunks = split_text_by_gguf_tokens(
            str(text),
            model_path=model_path,
            max_input_tokens=runtime.chunk_input_tokens,
            overlap_tokens=runtime.chunk_overlap_tokens,
        )

    translated_chunks: list[str] = []
    for chunk in chunks:
        last_exc: Exception | None = None
        for attempt in range(max(0, retries) + 1):
            try:
                translated_chunks.append(
                    translate_gguf(
                        chunk,
                        target_lang,
                        source_lang=source_lang,
                        model_path=model_path,
                        n_ctx=runtime.n_ctx,
                        n_threads=runtime.n_threads,
                        n_threads_batch=runtime.n_threads_batch,
                        max_tokens=max_tokens,
                    )
                )
                last_exc = None
                break
            except Exception as exc:  # pragma: no cover - exercised via integration paths
                last_exc = exc
                if attempt >= retries:
                    raise
        if last_exc is not None:
            raise last_exc
    merged = _merge_translated_chunks(translated_chunks)
    return (merged if merged else str(text)), len(chunks)


def _translate_gguf_worker_task(
    payload: tuple[object, str, str, str, str, int, _GGUFRuntimeConfig],
) -> tuple[object, str | None, int, str | None]:
    idx, text, source_lang, target_lang, model_path, max_tokens, runtime = payload
    try:
        translated, chunk_count = _translate_text_with_gguf(
            text,
            target_lang=target_lang,
            source_lang=source_lang,
            model_path=model_path,
            max_tokens=max_tokens,
            runtime=runtime,
        )
        return idx, translated, chunk_count, None
    except Exception as exc:  # pragma: no cover - exercised via integration paths
        return idx, None, 0, f"{type(exc).__name__}: {exc}"


def _resolve_gguf_token_cap(
    *,
    source_col: str,
    text: str,
    hard_cap: int,
    runtime: _GGUFRuntimeConfig,
    title_max_tokens: int,
    abstract_max_tokens: int,
) -> int:
    if runtime.token_budget_mode == "global":
        return max(1, int(hard_cap))

    col_norm = source_col.strip().casefold()
    if col_norm == "title":
        return max(1, min(int(hard_cap), int(title_max_tokens)))
    if col_norm == "abstract":
        text_len = len(str(text))
        if text_len <= 900:
            adaptive = 384
        elif text_len <= 1800:
            adaptive = 512
        elif text_len <= 3200:
            adaptive = 640
        else:
            adaptive = int(abstract_max_tokens)
        return max(64, min(int(hard_cap), int(abstract_max_tokens), adaptive))

    return max(1, min(int(hard_cap), 768))


def _translate_rows_gguf(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    to_translate: pd.DataFrame,
    target_lang: str,
    model_path: str,
    max_tokens: int,
    runtime: _GGUFRuntimeConfig,
    title_max_tokens: int,
    abstract_max_tokens: int,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[list[tuple[object, str]], int, int, float]:
    """Translate selected rows with a local GGUF model."""
    lang_col = f"{source_col}_lang"
    failed: list[tuple[object, str]] = []
    chunked_docs = 0
    total_chunks = 0
    started = time.perf_counter()
    items = [
        (
            idx,
            str(text),
            str(to_translate.at[idx, lang_col]),
            _resolve_gguf_token_cap(
                source_col=source_col,
                text=str(text),
                hard_cap=max_tokens,
                runtime=runtime,
                title_max_tokens=title_max_tokens,
                abstract_max_tokens=abstract_max_tokens,
            ),
        )
        for idx, text in to_translate[source_col].items()
    ]
    if runtime.effective_workers <= 1 or len(items) <= 1:
        for idx, text, src_lang, cap_tokens in tqdm(items, total=len(items), desc=f"  {source_col}"):
            try:
                translated, chunk_count = _translate_text_with_gguf(
                    text,
                    target_lang=target_lang,
                    source_lang=src_lang,
                    model_path=model_path,
                    max_tokens=cap_tokens,
                    runtime=runtime,
                )
                df.at[idx, target_col] = translated
                if chunk_count > 1:
                    chunked_docs += 1
                total_chunks += chunk_count
            except Exception as exc:
                failed.append((idx, f"{type(exc).__name__}: {exc}"))
        elapsed = max(1e-9, time.perf_counter() - started)
        return failed, chunked_docs, total_chunks, elapsed

    owns_pool = pool is None
    if pool is None:
        pool = ProcessPoolExecutor(
            max_workers=runtime.effective_workers,
            initializer=_init_gguf_pool_worker,
            initargs=(
                model_path,
                runtime.n_ctx,
                runtime.n_threads,
                runtime.n_threads_batch,
                runtime.auto_chunk,
            ),
        )
    try:
        futures = [
            pool.submit(
                _translate_gguf_worker_task,
                (idx, text, src_lang, target_lang, model_path, cap_tokens, runtime),
            )
            for idx, text, src_lang, cap_tokens in items
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {source_col}"):
            idx, translated, chunk_count, error_msg = future.result()
            if translated is not None:
                df.at[idx, target_col] = translated
                if chunk_count > 1:
                    chunked_docs += 1
                total_chunks += chunk_count
            else:
                failed.append((idx, error_msg or "unknown error"))
    finally:
        if owns_pool:
            pool.shutdown(wait=True)
    elapsed = max(1e-9, time.perf_counter() - started)
    return failed, chunked_docs, total_chunks, elapsed


def _report_failed_translations(column: str, failed: list[tuple[object, str]]) -> None:
    """Print compact translation failure summary for one column."""
    if not failed:
        return
    logger.warning("  %s: %s translations failed", column, len(failed))
    sample = "; ".join(f"idx={idx} ({msg})" for idx, msg in failed[:3])
    logger.warning("    examples: %s", sample)


def _summarize_translation_cost(
    *,
    provider: TranslationProvider,
    call_records: list[dict[str, str | float | None]],
    openrouter_cost_mode: str,
    api_key: str | None,
    api_base: str,
    max_workers: int,
) -> tuple[float | None, dict[str, object] | None]:
    """Resolve translation costs when OpenRouter metadata is available."""
    if provider != "openrouter" or not call_records:
        return None, None

    total_cost_usd, cost_summary = resolve_openrouter_costs(
        call_records,
        mode=openrouter_cost_mode,
        api_key=api_key,
        api_base=api_base,
        max_workers=max_workers,
        retries=2,
        delay=0.5,
        wait_before_fetch=2.0,
        logger_obj=logger,
        total_label="Total translation cost",
        log_fetch_resolution=True,
    )
    return total_cost_usd, cost_summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    *,
    provider: TranslationProvider,
    model: str,
    target_lang: str = "en",
    api_key: str | None = None,
    api_base: str = DEFAULT_OPENROUTER_API_BASE,
    max_workers: int = 5,
    max_translation_tokens: int = DEFAULT_TRANSLATION_MAX_TOKENS,
    gguf_parallel_policy: GGUFParallelPolicy = DEFAULT_GGUF_PARALLEL_POLICY,
    gguf_warmup_budget_seconds: int = DEFAULT_GGUF_WARMUP_BUDGET_SECONDS,
    gguf_token_budget_mode: GGUFTokenBudgetMode = "column_aware",
    gguf_title_max_tokens: int = DEFAULT_GGUF_TITLE_MAX_TOKENS,
    gguf_abstract_max_tokens: int = DEFAULT_GGUF_ABSTRACT_MAX_TOKENS,
    gguf_threads: int | None = None,
    gguf_threads_batch: int | None = None,
    gguf_n_ctx: int = 4096,
    gguf_auto_chunk: bool = True,
    gguf_chunk_input_tokens: int = DEFAULT_GGUF_CHUNK_INPUT_TOKENS,
    gguf_chunk_overlap_tokens: int = DEFAULT_GGUF_CHUNK_OVERLAP_TOKENS,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
) -> tuple[pd.DataFrame, TranslationCostInfo]:
    """Translate non-English entries in *columns* and add ``{col}_en`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must already have ``{col}_lang`` columns (from :func:`detect_languages`).
    columns : list[str]
        Columns to translate (e.g. ``["Title", "Abstract"]``).
    provider : str
        ``"openrouter"`` or ``"gguf"``.
    model : str
        Model identifier (e.g. ``"gpt-4o"`` for OpenRouter,
        ``"mradermacher/translategemma-4b-it-GGUF"`` for GGUF).
    target_lang : str
        Target language code.
    api_key : str, optional
        Required for ``provider="openrouter"``.
    api_base : str
        OpenRouter API base URL.
    max_workers : int
        Concurrent workers for OpenRouter and GGUF document-level translation.
    max_translation_tokens : int
        Global hard token ceiling per translation call.
    gguf_parallel_policy : str
        GGUF worker policy: ``"auto_calibrated"``, ``"balanced_auto"``,
        ``"max_throughput"``, or ``"stability_first"``.
    gguf_warmup_budget_seconds : int
        Time budget for one-time CPU auto-calibration when
        ``gguf_parallel_policy="auto_calibrated"``.
    gguf_token_budget_mode : str
        ``"column_aware"`` (default) or ``"global"`` token budgeting.
    gguf_title_max_tokens : int
        Title translation cap used when ``gguf_token_budget_mode="column_aware"``.
    gguf_abstract_max_tokens : int
        Abstract translation cap used when ``gguf_token_budget_mode="column_aware"``.
    gguf_threads : int, optional
        Threads used by each GGUF worker model instance.
    gguf_threads_batch : int, optional
        Prompt/batch thread count used by each GGUF worker model instance.
    gguf_n_ctx : int
        GGUF context window.
    gguf_auto_chunk : bool
        If ``True``, split long texts into token chunks and merge translated chunks.
    gguf_chunk_input_tokens : int
        Maximum GGUF input tokens per chunk when ``gguf_auto_chunk=True``.
    gguf_chunk_overlap_tokens : int
        Chunk overlap in GGUF tokens when ``gguf_auto_chunk=True``.
    openrouter_cost_mode : str
        ``"hybrid"`` (default), ``"strict"``, or ``"fast"``.

    Returns
    -------
    tuple[pd.DataFrame, TranslationCostInfo]
        ``(translated_df, cost_info)`` where *cost_info* has keys
        ``prompt_tokens``, ``completion_tokens``, ``provider``, ``model``,
        ``cost_usd``, ``cost_mode``, ``cost_summary``.
    """
    from .config import validate_provider
    validate_provider(
        provider,
        valid={"openrouter", "gguf"},
        api_key=api_key,
        requires_key={"openrouter"},
        requires_import={"gguf": "llama_cpp"},
    )
    df = df.copy()
    if max_translation_tokens <= 0:
        raise ValueError("max_translation_tokens must be > 0.")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0.")
    if gguf_parallel_policy not in {"auto_calibrated", "balanced_auto", "max_throughput", "stability_first"}:
        raise ValueError(
            f"Invalid gguf_parallel_policy '{gguf_parallel_policy}'. "
            "Expected 'auto_calibrated', 'balanced_auto', 'max_throughput', or 'stability_first'."
        )
    if gguf_token_budget_mode not in {"column_aware", "global"}:
        raise ValueError(
            f"Invalid gguf_token_budget_mode '{gguf_token_budget_mode}'. "
            "Expected 'column_aware' or 'global'."
        )
    if gguf_warmup_budget_seconds <= 0:
        raise ValueError("gguf_warmup_budget_seconds must be > 0.")
    if gguf_title_max_tokens <= 0:
        raise ValueError("gguf_title_max_tokens must be > 0.")
    if gguf_abstract_max_tokens <= 0:
        raise ValueError("gguf_abstract_max_tokens must be > 0.")
    if gguf_parallel_policy != "auto_calibrated" and gguf_warmup_budget_seconds != DEFAULT_GGUF_WARMUP_BUDGET_SECONDS:
        logger.info(
            "  GGUF warmup budget ignored because policy=%s",
            gguf_parallel_policy,
        )
    if gguf_n_ctx <= 0:
        raise ValueError("gguf_n_ctx must be > 0.")
    if gguf_threads is not None and int(gguf_threads) <= 0:
        raise ValueError("gguf_threads must be > 0 when provided.")
    if gguf_threads_batch is not None and int(gguf_threads_batch) <= 0:
        raise ValueError("gguf_threads_batch must be > 0 when provided.")
    if gguf_chunk_input_tokens <= 0:
        raise ValueError("gguf_chunk_input_tokens must be > 0.")
    if gguf_chunk_overlap_tokens < 0:
        raise ValueError("gguf_chunk_overlap_tokens must be >= 0.")
    if gguf_chunk_overlap_tokens >= gguf_chunk_input_tokens:
        raise ValueError("gguf_chunk_overlap_tokens must be < gguf_chunk_input_tokens.")
    _require_columns(df, columns, function_name="translate_dataframe")
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    total_pt = 0
    total_ct = 0
    call_records: list[dict[str, str | float | None]] = []

    # Pre-resolve GGUF model path once (downloads on first call, then cached).
    gguf_model_path: str | None = None
    gguf_runtime: _GGUFRuntimeConfig | None = None
    calibration_samples: list[tuple[str, str]] = []
    gguf_pool: ProcessPoolExecutor | None = None
    if provider == "gguf":
        from ads_bib._utils.gguf_backend import resolve_gguf_model

        gguf_model_path = resolve_gguf_model(model)
        calibration_samples = _collect_calibration_samples(
            df,
            columns=columns,
            target_lang=target_lang,
            limit=6,
        )
        gguf_runtime = _resolve_gguf_runtime(
            max_workers=max_workers,
            policy=gguf_parallel_policy,
            model_path=gguf_model_path,
            n_ctx=gguf_n_ctx,
            n_threads=gguf_threads,
            n_threads_batch=gguf_threads_batch,
            token_budget_mode=gguf_token_budget_mode,
            warmup_budget_seconds=gguf_warmup_budget_seconds,
            calibration_samples=calibration_samples,
            target_lang=target_lang,
            auto_chunk=gguf_auto_chunk,
            chunk_input_tokens=gguf_chunk_input_tokens,
            chunk_overlap_tokens=gguf_chunk_overlap_tokens,
        )
        logger.info(
            "  GGUF runtime | policy=%s | calibrated=%s | workers=%s/%s | "
            "gpu_offload=%s | threads=%s | threads_batch=%s | n_ctx=%s | "
            "token_budget_mode=%s | chunking=%s(%s/%s)",
            gguf_runtime.policy,
            gguf_runtime.calibrated,
            gguf_runtime.effective_workers,
            max_workers,
            gguf_runtime.gpu_offload_supported,
            gguf_runtime.n_threads,
            gguf_runtime.n_threads_batch,
            gguf_runtime.n_ctx,
            gguf_runtime.token_budget_mode,
            gguf_runtime.auto_chunk,
            gguf_runtime.chunk_input_tokens,
            gguf_runtime.chunk_overlap_tokens,
        )
        if gguf_runtime.effective_workers > 1:
            gguf_pool = ProcessPoolExecutor(
                max_workers=gguf_runtime.effective_workers,
                initializer=_init_gguf_pool_worker,
                initargs=(
                    gguf_model_path,
                    gguf_runtime.n_ctx,
                    gguf_runtime.n_threads,
                    gguf_runtime.n_threads_batch,
                    gguf_runtime.auto_chunk,
                ),
            )

    try:
        for col in columns:
            en_col, to_translate = _prepare_translation_targets(
                df,
                source_col=col,
                target_lang=target_lang,
            )
            n = len(to_translate)
            if n == 0:
                logger.info("  %s: nothing to translate", col)
                continue

            logger.info(
                "  %s: translating %s entries with %s/%s ...",
                col,
                f"{n:,}",
                provider,
                model,
            )
            if provider == "openrouter":
                pt, ct, records, failed = _translate_rows_openrouter(
                    df,
                    source_col=col,
                    target_col=en_col,
                    to_translate=to_translate,
                    target_lang=target_lang,
                    model=model,
                    api_key=api_key,
                    api_base=api_base,
                    max_workers=max_workers,
                    max_tokens=max_translation_tokens,
                )
                total_pt += pt
                total_ct += ct
                call_records.extend(records)
            elif provider == "gguf":
                assert gguf_model_path is not None
                assert gguf_runtime is not None
                if gguf_runtime.token_budget_mode == "global":
                    cap_info = str(max_translation_tokens)
                elif col.strip().casefold() == "title":
                    cap_info = str(min(max_translation_tokens, gguf_title_max_tokens))
                elif col.strip().casefold() == "abstract":
                    cap_info = f"adaptive<={min(max_translation_tokens, gguf_abstract_max_tokens)}"
                else:
                    cap_info = "adaptive<=768"
                logger.info(
                    "  %s: token_cap[%s]=%s",
                    col,
                    gguf_runtime.token_budget_mode,
                    cap_info,
                )
                failed, chunked_docs, total_chunks, elapsed_s = _translate_rows_gguf(
                    df,
                    source_col=col,
                    target_col=en_col,
                    to_translate=to_translate,
                    target_lang=target_lang,
                    model_path=gguf_model_path,
                    max_tokens=max_translation_tokens,
                    runtime=gguf_runtime,
                    title_max_tokens=gguf_title_max_tokens,
                    abstract_max_tokens=gguf_abstract_max_tokens,
                    pool=gguf_pool,
                )
                docs_per_min = n * 60.0 / max(1e-9, elapsed_s)
                logger.info("  %s: throughput %.2f docs/min", col, docs_per_min)
                if chunked_docs > 0:
                    logger.info(
                        "  %s: chunked %s/%s texts (avg chunks/text=%.2f)",
                        col,
                        f"{chunked_docs:,}",
                        f"{n:,}",
                        total_chunks / max(1, n),
                    )
            else:
                raise ValueError(f"Unknown translation provider: {provider}")

            _report_failed_translations(col, failed)
    finally:
        if gguf_pool is not None:
            gguf_pool.shutdown(wait=True)

    total_cost_usd, cost_summary = _summarize_translation_cost(
        provider=provider,
        call_records=call_records,
        openrouter_cost_mode=openrouter_cost_mode,
        api_key=api_key,
        api_base=api_base,
        max_workers=max_workers,
    )

    cost_info: TranslationCostInfo = {
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "provider": provider,
        "model": model,
        "cost_usd": total_cost_usd,
        "cost_mode": openrouter_cost_mode if provider == "openrouter" else None,
        "cost_summary": cost_summary,
    }
    if cost_tracker is not None and total_pt + total_ct > 0:
        cost_tracker.add(
            step="translation",
            provider=provider,
            model=model,
            prompt_tokens=total_pt,
            completion_tokens=total_ct,
            cost_usd=total_cost_usd,
        )
    return df, cost_info
