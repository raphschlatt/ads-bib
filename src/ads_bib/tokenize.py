"""Step 4 – Tokenize full-text (Title + Abstract) with spaCy."""

from __future__ import annotations

import pandas as pd


def tokenize_texts(
    df: pd.DataFrame,
    *,
    title_col: str = "Title_en",
    abstract_col: str = "Abstract_en",
    text_col: str = "full_text",
    token_col: str = "tokens",
    spacy_model: str = "en_core_web_lg",
    min_token_length: int = 3,
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
        spaCy model to load (default ``en_core_web_lg``).
    min_token_length : int
        Minimum character length for a token to be kept.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added *text_col* and *token_col* columns.
    """
    import spacy

    nlp = spacy.load(spacy_model)
    df = df.copy()

    # Build full_text
    df[text_col] = (
        df[title_col].fillna("").astype(str)
        + ". "
        + df[abstract_col].fillna("").astype(str)
    )

    def _tokenize(text: str) -> list[str]:
        doc = nlp(text)
        return [
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

    print(f"Tokenizing {len(df):,} documents with {spacy_model} ...")
    df[token_col] = df[text_col].apply(_tokenize)
    print("  Done.")
    return df
