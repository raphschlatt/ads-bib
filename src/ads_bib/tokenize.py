"""Step 4 – Tokenize full-text (Title + Abstract) with spaCy."""

from __future__ import annotations

from functools import lru_cache
import logging
import os
import subprocess
import sys

import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def default_n_process() -> int:
    """Return a conservative parallel worker default for spaCy."""
    cpu_count = os.cpu_count() or 1
    return min(max(cpu_count - 1, 1), 8)


@lru_cache(maxsize=8)
def _load_spacy_model(spacy_model: str, spacy_disable: tuple[str, ...]):
    """Load and cache spaCy model instances by model+disable signature."""
    import spacy

    disable = list(spacy_disable) if spacy_disable else None
    return spacy.load(spacy_model, disable=disable)


def ensure_spacy_model(
    *,
    spacy_model: str = "en_core_web_md",
    fallback_model: str = "en_core_web_lg",
    spacy_disable: tuple[str, ...] = ("ner", "parser", "textcat"),
    auto_download: bool = True,
) -> tuple[str, object]:
    """Ensure a spaCy model is available and return ``(model_name, nlp)``.

    If the preferred model is unavailable and ``auto_download`` is enabled,
    this function attempts ``python -m spacy download <model>`` once.
    On failure it falls back to ``fallback_model``.
    """
    disable = tuple(spacy_disable)
    try:
        return spacy_model, _load_spacy_model(spacy_model, disable)
    except Exception as exc:
        if auto_download:
            logger.warning(
                "spaCy model '%s' not available (%s: %s). Trying to install it now ...",
                spacy_model,
                type(exc).__name__,
                exc,
            )
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", spacy_model])
                return spacy_model, _load_spacy_model(spacy_model, disable)
            except Exception as install_exc:
                logger.warning(
                    "Automatic install failed (%s: %s). Falling back to '%s'.",
                    type(install_exc).__name__,
                    install_exc,
                    fallback_model,
                )
        else:
            logger.warning(
                "spaCy model '%s' unavailable (%s: %s). Falling back to '%s'.",
                spacy_model,
                type(exc).__name__,
                exc,
                fallback_model,
            )

    return fallback_model, _load_spacy_model(fallback_model, disable)


def tokenize_texts(
    df: pd.DataFrame,
    *,
    title_col: str = "Title_en",
    abstract_col: str = "Abstract_en",
    text_col: str = "full_text",
    token_col: str = "tokens",
    spacy_model: str = "en_core_web_md",
    min_token_length: int = 3,
    nlp: object | None = None,
    batch_size: int = 512,
    n_process: int | None = None,
    spacy_disable: tuple[str, ...] = ("ner", "parser", "textcat"),
    show_progress: bool = True,
) -> pd.DataFrame:
    """Create *full_text* from title + abstract, then tokenize.

    Tokenization: lemmatise, lowercase, remove stopwords / punctuation /
    numbers / non-alphabetic tokens, filter by minimum length.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *title_col* and *abstract_col*.
    title_col, abstract_col : str
        Column names for the (translated) title and abstract.
    text_col : str
        Name of the combined full-text column to create.
    token_col : str
        Name of the token list column to create.
    spacy_model : str
        spaCy model to load (default ``en_core_web_md``).
    min_token_length : int
        Minimum character length for a token to be kept.
    nlp : object, optional
        Preloaded spaCy ``Language`` object. When provided, ``spacy_model`` is ignored.
    batch_size : int
        Batch size passed to ``nlp.pipe``.
    n_process : int, optional
        Number of worker processes passed to ``nlp.pipe``.
        If ``None``, uses ``min(max(cpu_count - 1, 1), 8)``.
    spacy_disable : tuple[str, ...]
        spaCy pipeline components to disable when loading by model name.
    show_progress : bool
        Show a tqdm progress bar while consuming ``nlp.pipe`` output.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added *text_col* and *token_col* columns.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    if nlp is None:
        nlp = _load_spacy_model(spacy_model, tuple(spacy_disable))

    requested_n_process = default_n_process() if n_process is None else max(int(n_process), 1)
    df = df.copy()

    # Build full_text
    df[text_col] = (
        df[title_col].fillna("").astype(str)
        + ". "
        + df[abstract_col].fillna("").astype(str)
    )

    texts = df[text_col].tolist()

    def _tokenize_docs(proc_count: int) -> list[list[str]]:
        docs = nlp.pipe(texts, batch_size=batch_size, n_process=proc_count)
        tokens: list[list[str]] = []
        for doc in tqdm(
            docs,
            total=len(texts),
            desc="Tokenization",
            disable=not show_progress,
        ):
            tokens.append(
                [
                    token.lemma_.lower()
                    for token in doc
                    if (
                        not token.is_stop
                        and not token.is_punct
                        and not token.like_num
                        and token.is_alpha
                        and len(token.lemma_) >= min_token_length
                    )
                ]
            )
        return tokens

    logger.info(
        "Tokenizing %s documents with %s (n_process=%s, batch_size=%s) ...",
        f"{len(df):,}",
        spacy_model,
        requested_n_process,
        batch_size,
    )
    try:
        tokens = _tokenize_docs(requested_n_process)
        used_n_process = requested_n_process
    except Exception as exc:
        if requested_n_process == 1:
            raise
        logger.warning(
            "  spaCy multiprocessing failed (%s: %s). Retrying with n_process=1 ...",
            type(exc).__name__,
            exc,
        )
        tokens = _tokenize_docs(1)
        used_n_process = 1

    df[token_col] = tokens
    if used_n_process != requested_n_process:
        logger.info("  Done (fallback n_process=%s).", used_n_process)
    else:
        logger.info("  Done.")
    return df


def _default_n_process() -> int:
    """Backward-compatible alias for previous private helper name."""
    return default_n_process()
