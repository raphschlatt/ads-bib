"""Step 3 – Language detection and translation of non-English titles/abstracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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

logger = logging.getLogger(__name__)

TranslationProvider: TypeAlias = Literal["openrouter", "gguf", "nllb"]


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
# Translation helpers  (shared)
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


def _report_failed_translations(column: str, failed: list[tuple[object, str]]) -> None:
    """Print compact translation failure summary for one column."""
    if not failed:
        return
    logger.warning("  %s: %s translations failed", column, len(failed))
    sample = "; ".join(f"idx={idx} ({msg})" for idx, msg in failed[:3])
    logger.warning("    examples: %s", sample)


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GGUF backend  (GPU-accelerated, or CPU fallback)
# ---------------------------------------------------------------------------

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
    n_ctx: int,
    n_threads: int | None,
    n_threads_batch: int | None,
    auto_chunk: bool,
    chunk_input_tokens: int,
    chunk_overlap_tokens: int,
    retries: int = DEFAULT_GGUF_TRANSLATION_RETRIES,
) -> tuple[str, int]:
    from ads_bib._utils.gguf_backend import split_text_by_gguf_tokens, translate_gguf

    chunks = [str(text)]
    if auto_chunk and len(str(text)) > chunk_input_tokens * 4:
        chunks = split_text_by_gguf_tokens(
            str(text),
            model_path=model_path,
            max_input_tokens=chunk_input_tokens,
            overlap_tokens=chunk_overlap_tokens,
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
                        n_ctx=n_ctx,
                        n_threads=n_threads,
                        n_threads_batch=n_threads_batch,
                        max_tokens=max_tokens,
                    )
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    raise
        if last_exc is not None:
            raise last_exc
    merged = _merge_translated_chunks(translated_chunks)
    return (merged if merged else str(text)), len(chunks)


def _translate_rows_gguf(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    to_translate: pd.DataFrame,
    target_lang: str,
    model_path: str,
    max_tokens: int,
    n_ctx: int,
    n_threads: int | None,
    n_threads_batch: int | None,
    auto_chunk: bool,
    chunk_input_tokens: int,
    chunk_overlap_tokens: int,
) -> tuple[list[tuple[object, str]], int, int, float]:
    """Translate selected rows with a local GGUF model (single-worker)."""
    lang_col = f"{source_col}_lang"
    failed: list[tuple[object, str]] = []
    chunked_docs = 0
    total_chunks = 0
    started = time.perf_counter()

    items = list(zip(to_translate.index, to_translate[source_col], to_translate[lang_col]))
    for idx, text, src_lang in tqdm(items, total=len(items), desc=f"  {source_col}"):
        try:
            translated, chunk_count = _translate_text_with_gguf(
                str(text),
                target_lang=target_lang,
                source_lang=str(src_lang),
                model_path=model_path,
                max_tokens=max_tokens,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                auto_chunk=auto_chunk,
                chunk_input_tokens=chunk_input_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
            df.at[idx, target_col] = translated
            if chunk_count > 1:
                chunked_docs += 1
            total_chunks += chunk_count
        except Exception as exc:
            failed.append((idx, f"{type(exc).__name__}: {exc}"))
    elapsed = max(1e-9, time.perf_counter() - started)
    return failed, chunked_docs, total_chunks, elapsed


# ---------------------------------------------------------------------------
# NLLB backend  (CTranslate2 — fast CPU translation, 200+ languages)
# ---------------------------------------------------------------------------

# Mapping from fasttext ISO-639-1 codes to NLLB Flores-200 codes.
_NLLB_LANG_MAP: dict[str, str] = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "az": "azj_Latn",
    "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng", "bs": "bos_Latn",
    "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
    "et": "est_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn",
    "ga": "gle_Latn", "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "hy": "hye_Armn", "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn",
    "ja": "jpn_Jpan", "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr",
    "kn": "kan_Knda", "ko": "kor_Hang", "lt": "lit_Latn", "lv": "lvs_Latn",
    "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva",
    "ms": "zsm_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "no": "nob_Latn", "pa": "pan_Guru", "pl": "pol_Latn", "pt": "por_Latn",
    "ro": "ron_Latn", "ru": "rus_Cyrl", "si": "sin_Sinh", "sk": "slk_Latn",
    "sl": "slv_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai",
    "tl": "tgl_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab",
    "uz": "uzn_Latn", "vi": "vie_Latn", "zh": "zho_Hans",
}

_nllb_translator = None
_nllb_tokenizer = None
_nllb_model_path: str | None = None

_NLLB_INSTALL_HINT = (
    "NLLB translation requires 'ctranslate2' and 'sentencepiece'. Install with:\n"
    "  pip install ctranslate2 sentencepiece huggingface-hub"
)

_DEFAULT_NLLB_MODEL = "JustFrederik/nllb-200-distilled-600M-ct2-int8"


def _resolve_nllb_lang_code(iso_code: str) -> str | None:
    """Map a fasttext ISO-639-1 code to an NLLB Flores-200 code."""
    return _NLLB_LANG_MAP.get(iso_code)


def _ensure_nllb_model(model: str) -> tuple:
    """Load and cache the CTranslate2 NLLB translator + SentencePiece tokenizer."""
    global _nllb_translator, _nllb_tokenizer, _nllb_model_path
    if _nllb_translator is not None and _nllb_model_path == model:
        return _nllb_translator, _nllb_tokenizer

    try:
        import ctranslate2
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError(_NLLB_INSTALL_HINT) from exc

    from huggingface_hub import snapshot_download

    # Download or resolve model directory
    if Path(model).is_dir():
        model_dir = str(model)
    else:
        import warnings

        logger.info("Downloading NLLB model %s …", model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
            model_dir = snapshot_download(repo_id=model, local_dir_use_symlinks=False)

    import os

    # Doku: "increase inter_threads over intra_threads" for large data,
    # keep inter_threads * intra_threads <= physical core count
    cpu_count = max(1, int(os.cpu_count() or 1))
    inter = max(1, min(cpu_count, 4))
    intra = max(1, cpu_count // inter)
    _nllb_translator = ctranslate2.Translator(
        model_dir, device="cpu", compute_type="int8",
        inter_threads=inter, intra_threads=intra,
    )
    logger.info(
        "  CTranslate2 threads: inter=%d, intra=%d (cpus=%d)",
        inter, intra, cpu_count,
    )
    sp_model_path = Path(model_dir) / "sentencepiece.bpe.model"
    if not sp_model_path.exists():
        # Try common alternative names
        for alt_name in ["spm.model", "tokenizer.model"]:
            alt = Path(model_dir) / alt_name
            if alt.exists():
                sp_model_path = alt
                break
    _nllb_tokenizer = spm.SentencePieceProcessor()
    _nllb_tokenizer.Load(str(sp_model_path))
    _nllb_model_path = model
    logger.info("NLLB model loaded from %s", model_dir)
    return _nllb_translator, _nllb_tokenizer


def _translate_rows_nllb(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    to_translate: pd.DataFrame,
    target_lang: str,
    model: str,
) -> tuple[list[tuple[object, str]], float]:
    """Translate selected rows with NLLB/CTranslate2 using native batched inference."""
    lang_col = f"{source_col}_lang"
    failed: list[tuple[object, str]] = []
    started = time.perf_counter()
    skipped_langs: set[str] = set()

    tgt_code = _resolve_nllb_lang_code(target_lang)
    if tgt_code is None:
        raise ValueError(
            f"NLLB does not support target language '{target_lang}'. "
            f"Known codes: {sorted(_NLLB_LANG_MAP.keys())}"
        )

    # Fail fast: verify dependencies and load model before the loop
    translator, tokenizer = _ensure_nllb_model(model)

    # Partition rows: collect translatable items, skip unsupported languages
    indices: list[object] = []
    all_tokens: list[list[str]] = []
    all_prefixes: list[list[str]] = []

    for idx, row in to_translate.iterrows():
        src_lang_str = str(row[lang_col])
        src_code = _resolve_nllb_lang_code(src_lang_str)
        if src_code is None:
            if src_lang_str not in skipped_langs:
                logger.warning(
                    "  NLLB: no mapping for source language '%s' — skipping rows with this language",
                    src_lang_str,
                )
                skipped_langs.add(src_lang_str)
            failed.append((idx, f"Unsupported source language: {src_lang_str}"))
            continue
        tokens = tokenizer.Encode(str(row[source_col]), out_type=str)
        indices.append(idx)
        all_tokens.append([src_code] + tokens)
        all_prefixes.append([tgt_code])

    # Translate in sub-batches with tqdm progress.
    # CTranslate2 internally splits each sub-batch by max_batch_size (tokens)
    # and parallelizes across inter_threads workers.
    n = len(all_tokens)
    chunk_size = 64
    for start in tqdm(range(0, max(n, 1), chunk_size), desc=f"  {source_col}", disable=n == 0):
        end = min(start + chunk_size, n)
        try:
            batch_results = translator.translate_batch(
                all_tokens[start:end],
                target_prefix=all_prefixes[start:end],
                beam_size=4,
                batch_type="tokens",
                max_batch_size=1024,
                max_decoding_length=512,
            )
            for idx, result in zip(indices[start:end], batch_results):
                out_tokens = result.hypotheses[0]
                if out_tokens and out_tokens[0] == tgt_code:
                    out_tokens = out_tokens[1:]
                df.at[idx, target_col] = tokenizer.Decode(out_tokens)
        except Exception:
            # Fallback: translate one-by-one to isolate the bad row
            for i in range(start, end):
                try:
                    single_result = translator.translate_batch(
                        [all_tokens[i]],
                        target_prefix=[all_prefixes[i]],
                        beam_size=4,
                        max_decoding_length=512,
                    )
                    out_tokens = single_result[0].hypotheses[0]
                    if out_tokens and out_tokens[0] == tgt_code:
                        out_tokens = out_tokens[1:]
                    df.at[indices[i], target_col] = tokenizer.Decode(out_tokens)
                except Exception as row_exc:
                    failed.append((indices[i], f"{type(row_exc).__name__}: {row_exc}"))

    elapsed = max(1e-9, time.perf_counter() - started)
    return failed, elapsed


# ---------------------------------------------------------------------------
# Cost summarization  (OpenRouter only)
# ---------------------------------------------------------------------------

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
    # GGUF-specific (kept simple — single worker, optional chunking)
    gguf_n_ctx: int = 4096,
    gguf_threads: int | None = None,
    gguf_threads_batch: int | None = None,
    gguf_auto_chunk: bool = True,
    gguf_chunk_input_tokens: int = DEFAULT_GGUF_CHUNK_INPUT_TOKENS,
    gguf_chunk_overlap_tokens: int = DEFAULT_GGUF_CHUNK_OVERLAP_TOKENS,
    # OpenRouter cost tracking
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
        ``"openrouter"``, ``"gguf"``, or ``"nllb"``.
    model : str
        Model identifier. Examples:

        - OpenRouter: ``"gpt-4o"``
        - GGUF: ``"mradermacher/translategemma-4b-it-GGUF"``
        - NLLB: ``"JustFrederik/nllb-200-distilled-600M-ct2-int8"`` (default)
    target_lang : str
        Target language code.
    api_key : str, optional
        Required for ``provider="openrouter"``.
    api_base : str
        OpenRouter API base URL.
    max_workers : int
        Concurrent workers for OpenRouter.
    max_translation_tokens : int
        Token ceiling per translation call (OpenRouter and GGUF).
    gguf_n_ctx : int
        GGUF context window.
    gguf_threads : int, optional
        Threads for the GGUF model instance.
    gguf_threads_batch : int, optional
        Prompt/batch thread count for the GGUF model instance.
    gguf_auto_chunk : bool
        If ``True``, split long texts into token chunks (GGUF only).
    gguf_chunk_input_tokens : int
        Maximum GGUF input tokens per chunk.
    gguf_chunk_overlap_tokens : int
        Chunk overlap in tokens.
    openrouter_cost_mode : str
        ``"hybrid"`` (default), ``"strict"``, or ``"fast"``.
    cost_tracker : CostTracker, optional
        Aggregated cost tracker instance.

    Returns
    -------
    tuple[pd.DataFrame, TranslationCostInfo]
        ``(translated_df, cost_info)``.
    """
    from .config import validate_provider
    validate_provider(
        provider,
        valid={"openrouter", "gguf", "nllb"},
        api_key=api_key,
        requires_key={"openrouter"},
        requires_import={"gguf": "llama_cpp", "nllb": "ctranslate2"},
    )
    df = df.copy()
    if max_translation_tokens <= 0:
        raise ValueError("max_translation_tokens must be > 0.")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0.")
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

    # Pre-resolve GGUF model path once.
    gguf_model_path: str | None = None
    if provider == "gguf":
        from ads_bib._utils.gguf_backend import resolve_gguf_model
        gguf_model_path = resolve_gguf_model(model)
        logger.info(
            "  GGUF translation | n_ctx=%s | threads=%s | threads_batch=%s | chunking=%s(%s/%s)",
            gguf_n_ctx, gguf_threads, gguf_threads_batch,
            gguf_auto_chunk, gguf_chunk_input_tokens, gguf_chunk_overlap_tokens,
        )

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
            col, f"{n:,}", provider, model,
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
            failed, chunked_docs, total_chunks, elapsed_s = _translate_rows_gguf(
                df,
                source_col=col,
                target_col=en_col,
                to_translate=to_translate,
                target_lang=target_lang,
                model_path=gguf_model_path,
                max_tokens=max_translation_tokens,
                n_ctx=gguf_n_ctx,
                n_threads=gguf_threads,
                n_threads_batch=gguf_threads_batch,
                auto_chunk=gguf_auto_chunk,
                chunk_input_tokens=gguf_chunk_input_tokens,
                chunk_overlap_tokens=gguf_chunk_overlap_tokens,
            )
            docs_per_min = n * 60.0 / max(1e-9, elapsed_s)
            logger.info("  %s: throughput %.2f docs/min", col, docs_per_min)
            if chunked_docs > 0:
                logger.info(
                    "  %s: chunked %s/%s texts (avg chunks/text=%.2f)",
                    col, f"{chunked_docs:,}", f"{n:,}", total_chunks / max(1, n),
                )

        elif provider == "nllb":
            failed, elapsed_s = _translate_rows_nllb(
                df,
                source_col=col,
                target_col=en_col,
                to_translate=to_translate,
                target_lang=target_lang,
                model=model,
            )
            docs_per_min = n * 60.0 / max(1e-9, elapsed_s)
            logger.info("  %s: throughput %.2f docs/min", col, docs_per_min)

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
