"""Step 3 – Language detection and translation of non-English titles/abstracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import threading

import pandas as pd
from tqdm.auto import tqdm

from ads_bib._utils.openrouter_client import (
    openrouter_chat_completion,
    openrouter_usage_from_response,
)
from ads_bib._utils.openrouter_costs import (
    DEFAULT_OPENROUTER_API_BASE,
    fetch_generation_cost,
    normalize_openrouter_api_base,
    normalize_openrouter_cost_mode,
    summarize_openrouter_costs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection  (fasttext – fast & free)
# ---------------------------------------------------------------------------

_ft_model = None


def _get_ft_model(model_path: str | Path | None = None):
    """Lazy-load the fasttext language-identification model."""
    global _ft_model
    if _ft_model is None:
        import fasttext

        if model_path is None:
            model_path = "lid.176.bin"
        _ft_model = fasttext.load_model(str(model_path))
    return _ft_model


def _predict_language(text: str, model_path: str | Path | None = None) -> str:
    model = _get_ft_model(model_path)
    label, _ = model.predict(text)
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


def _fetch_generation_cost(
    generation_id: str,
    api_key: str,
    api_base: str = DEFAULT_OPENROUTER_API_BASE,
    retries: int = 3,
    delay: float = 1.0,
) -> float | None:
    """Backward-compatible wrapper around OpenRouter generation cost lookup."""
    return fetch_generation_cost(
        generation_id,
        api_key,
        api_base=api_base,
        retries=retries,
        delay=delay,
    )


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


def _translate_huggingface(
    text: str,
    target_lang: str,
    *,
    _pipeline=None,
    max_new_tokens: int = DEFAULT_TRANSLATION_MAX_TOKENS,
) -> str:
    """Translate *text* using a local HuggingFace model (e.g. TranslateGemma)."""
    prompt = f"Translate the following scientific text to {target_lang}:\n\n{text}"
    result = _pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    generated = result[0]["generated_text"]
    # TranslateGemma returns the full prompt + translation; strip the prompt
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    return generated


def _load_hf_pipeline(model: str):
    """Load a HuggingFace text-generation pipeline (cached)."""
    from transformers import pipeline as hf_pipeline

    logger.info("Loading local translation model: %s", model)
    return hf_pipeline(
        "text-generation",
        model=model,
        device_map="auto",
        torch_dtype="auto",
    )


def _prepare_translation_targets(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_lang: str,
) -> tuple[str, pd.DataFrame]:
    """Initialize translated column and return rows that need translation."""
    lang_col = f"{source_col}_lang"
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


def _translate_rows_huggingface(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    to_translate: pd.DataFrame,
    target_lang: str,
    hf_pipeline: object,
    max_new_tokens: int,
) -> list[tuple[object, str]]:
    """Translate selected rows with local HuggingFace pipeline."""
    failed: list[tuple[object, str]] = []
    for idx, text in tqdm(to_translate[source_col].items(), total=len(to_translate), desc=f"  {source_col}"):
        try:
            df.at[idx, target_col] = _translate_huggingface(
                str(text),
                target_lang,
                _pipeline=hf_pipeline,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            failed.append((idx, f"{type(exc).__name__}: {exc}"))
    return failed


def _report_failed_translations(column: str, failed: list[tuple[object, str]]) -> None:
    """Print compact translation failure summary for one column."""
    if not failed:
        return
    logger.warning("  %s: %s translations failed", column, len(failed))
    sample = "; ".join(f"idx={idx} ({msg})" for idx, msg in failed[:3])
    logger.warning("    examples: %s", sample)


def _summarize_translation_cost(
    *,
    provider: str,
    call_records: list[dict[str, str | float | None]],
    openrouter_cost_mode: str,
    api_key: str | None,
    api_base: str,
    max_workers: int,
) -> tuple[float | None, dict | None]:
    """Resolve translation costs when OpenRouter metadata is available."""
    if provider != "openrouter" or not call_records:
        return None, None

    cost_summary = summarize_openrouter_costs(
        call_records,
        mode=openrouter_cost_mode,
        api_key=api_key,
        api_base=api_base,
        max_workers=max_workers,
        retries=2,
        delay=0.5,
        wait_before_fetch=2.0,
    )
    total_cost_usd = cost_summary["total_cost_usd"]

    if cost_summary["fetch_attempted_calls"] > 0 and not cost_summary["fetch_skipped_no_api_key"]:
        logger.info(
            "  Resolved USD costs via /generation for %s calls (mode=%s) ...",
            cost_summary["fetch_attempted_calls"],
            openrouter_cost_mode,
        )
    if total_cost_usd is not None:
        logger.info(
            "  Total translation cost: $%.4f (%s/%s priced; direct=%s, fetched=%s, mode=%s)",
            total_cost_usd,
            cost_summary["priced_calls"],
            cost_summary["total_calls"],
            cost_summary["direct_priced_calls"],
            cost_summary["fetched_priced_calls"],
            openrouter_cost_mode,
        )
    return total_cost_usd, cost_summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    *,
    provider: str,
    model: str,
    target_lang: str = "en",
    api_key: str | None = None,
    api_base: str = DEFAULT_OPENROUTER_API_BASE,
    max_workers: int = 5,
    max_translation_tokens: int = DEFAULT_TRANSLATION_MAX_TOKENS,
    openrouter_cost_mode: str = "hybrid",
    cost_tracker: "CostTracker | None" = None,
) -> tuple[pd.DataFrame, dict]:
    """Translate non-English entries in *columns* and add ``{col}_en`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must already have ``{col}_lang`` columns (from :func:`detect_languages`).
    columns : list[str]
        Columns to translate (e.g. ``["Title", "Abstract"]``).
    provider : str
        ``"openrouter"`` or ``"huggingface"``.
    model : str
        Model identifier (e.g. ``"gpt-4o"`` for OpenRouter, ``"google/translategemma-4b-it"`` for HF).
    target_lang : str
        Target language code.
    api_key : str, optional
        Required for ``provider="openrouter"``.
    api_base : str
        OpenRouter API base URL.
    max_workers : int
        Concurrent workers for API-based translation.
    max_translation_tokens : int
        Token limit per translation call (`max_tokens` for OpenRouter and
        `max_new_tokens` for local HuggingFace pipelines).
    openrouter_cost_mode : str
        ``"hybrid"`` (default), ``"strict"``, or ``"fast"``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(translated_df, cost_info)`` where *cost_info* has keys
        ``prompt_tokens``, ``completion_tokens``, ``provider``, ``model``.
    """
    from .config import validate_provider
    validate_provider(
        provider,
        valid={"openrouter", "huggingface"},
        api_key=api_key,
        requires_key={"openrouter"},
        requires_import={"huggingface": "transformers"},
    )
    df = df.copy()
    if max_translation_tokens <= 0:
        raise ValueError("max_translation_tokens must be > 0.")
    openrouter_cost_mode = normalize_openrouter_cost_mode(openrouter_cost_mode)
    total_pt = 0
    total_ct = 0
    call_records: list[dict[str, str | float | None]] = []

    # Pre-load HF pipeline once (not per-row)
    hf_pipe: object | None = None
    if provider == "huggingface":
        hf_pipe = _load_hf_pipeline(model)

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
        elif provider == "huggingface":
            failed = _translate_rows_huggingface(
                df,
                source_col=col,
                target_col=en_col,
                to_translate=to_translate,
                target_lang=target_lang,
                hf_pipeline=hf_pipe,
                max_new_tokens=max_translation_tokens,
            )
        else:
            raise ValueError(f"Unknown translation provider: {provider}")

        _report_failed_translations(col, failed)

    total_cost_usd, cost_summary = _summarize_translation_cost(
        provider=provider,
        call_records=call_records,
        openrouter_cost_mode=openrouter_cost_mode,
        api_key=api_key,
        api_base=api_base,
        max_workers=max_workers,
    )

    cost_info = {
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
