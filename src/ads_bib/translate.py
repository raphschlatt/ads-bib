"""Step 3 – Language detection and translation of non-English titles/abstracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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
        print(f"  {col}: {n_non_en:,} non-English entries detected")
    return df


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a highly accurate translator specializing in scientific and "
    "technical texts. Only translate the text. Do not comment or provide "
    "additional information."
)


def _fetch_generation_cost(
    generation_id: str,
    api_key: str,
    api_base: str = "https://openrouter.ai/api/v1",
    retries: int = 3,
    delay: float = 1.0,
) -> float | None:
    """Query OpenRouter ``/api/v1/generation`` for actual USD cost."""
    import time
    import requests

    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/api/v1/generation?id={generation_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.ok:
                data = resp.json().get("data", {})
                cost = data.get("total_cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def _translate_openrouter(
    text: str,
    target_lang: str,
    model: str,
    api_key: str,
    api_base: str = "https://openrouter.ai/api/v1",
) -> tuple[str, int, int, str | None]:
    """Translate *text* via OpenRouter. Returns ``(translated, prompt_tokens, completion_tokens, generation_id)``."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate the following scientific text to {target_lang}:\n\n{text}"},
        ],
        max_tokens=2048,
        temperature=0,
    )
    translated = resp.choices[0].message.content.strip()
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    gen_id = getattr(resp, "id", None)
    return translated, pt, ct, gen_id


def _translate_huggingface(
    text: str,
    target_lang: str,
    *,
    _pipeline=None,
) -> str:
    """Translate *text* using a local HuggingFace model (e.g. TranslateGemma)."""
    prompt = f"Translate the following scientific text to {target_lang}:\n\n{text}"
    result = _pipeline(prompt, max_new_tokens=2048, do_sample=False)
    generated = result[0]["generated_text"]
    # TranslateGemma returns the full prompt + translation; strip the prompt
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    return generated


def _load_hf_pipeline(model: str):
    """Load a HuggingFace text-generation pipeline (cached)."""
    from transformers import pipeline as hf_pipeline

    print(f"Loading local translation model: {model}")
    return hf_pipeline(
        "text-generation",
        model=model,
        device_map="auto",
        torch_dtype="auto",
    )


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
    api_base: str = "https://openrouter.ai/api/v1",
    max_workers: int = 5,
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

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(translated_df, cost_info)`` where *cost_info* has keys
        ``prompt_tokens``, ``completion_tokens``, ``provider``, ``model``.
    """
    df = df.copy()
    total_pt, total_ct = 0, 0
    gen_ids = []  # OpenRouter generation IDs for cost lookup

    # Pre-load HF pipeline once (not per-row)
    hf_pipe = None
    if provider == "huggingface":
        hf_pipe = _load_hf_pipeline(model)

    for col in columns:
        lang_col = f"{col}_lang"
        en_col = f"{col}_{target_lang}"

        if lang_col not in df.columns:
            raise ValueError(f"Column '{lang_col}' not found. Run detect_languages() first.")

        # Start with original text
        df[en_col] = df[col]

        # Rows that need translation
        mask = (df[lang_col] != target_lang) & df[col].notna() & (df[col] != "")
        to_translate = df.loc[mask]
        n = len(to_translate)

        if n == 0:
            print(f"  {col}: nothing to translate")
            continue

        print(f"  {col}: translating {n:,} entries with {provider}/{model} ...")
        failed = []

        if provider == "openrouter":
            def _do_translate(idx_text):
                idx, text = idx_text
                try:
                    translated, pt, ct, gen_id = _translate_openrouter(
                        text, target_lang, model, api_key, api_base
                    )
                    return idx, translated, pt, ct, gen_id
                except Exception as exc:
                    return idx, None, 0, 0, None

            items = list(zip(to_translate.index, to_translate[col]))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_do_translate, item) for item in items]
                for future in tqdm(as_completed(futures), total=n, desc=f"  {col}"):
                    idx, translated, pt, ct, gen_id = future.result()
                    if translated is not None:
                        df.at[idx, en_col] = translated
                    else:
                        failed.append(idx)
                    total_pt += pt
                    total_ct += ct
                    if gen_id:
                        gen_ids.append(gen_id)

        elif provider == "huggingface":
            for idx, text in tqdm(to_translate[col].items(), total=n, desc=f"  {col}"):
                try:
                    df.at[idx, en_col] = _translate_huggingface(
                        text, target_lang, _pipeline=hf_pipe
                    )
                except Exception:
                    failed.append(idx)

        else:
            raise ValueError(f"Unknown translation provider: {provider}")

        if failed:
            print(f"  {col}: {len(failed)} translations failed")

    # Batch-fetch USD costs from OpenRouter generation API
    total_cost_usd = None
    if provider == "openrouter" and gen_ids and api_key:
        import time
        print(f"  Fetching USD costs for {len(gen_ids)} generations ...")
        time.sleep(2)  # brief delay for OpenRouter to finalize cost data
        costs = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(_fetch_generation_cost, gid, api_key, api_base, retries=2, delay=0.5): gid
                for gid in gen_ids
            }
            for fut in as_completed(futs):
                c = fut.result()
                if c is not None:
                    costs.append(c)
        if costs:
            total_cost_usd = sum(costs)
            print(f"  Total translation cost: ${total_cost_usd:.4f} ({len(costs)}/{len(gen_ids)} priced)")

    cost_info = {
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "provider": provider,
        "model": model,
        "cost_usd": total_cost_usd,
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
