"""Data cleaning utilities for ADS export data."""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd


def clean_html(text: str) -> str:
    """Unescape HTML entities and strip HTML tags."""
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    return text


def clean_range(value: str) -> str:
    """Keep only the first number in a range and strip whitespace.

    ``"123-456"`` → ``"123"``, ``" 42 "`` → ``"42"``.
    """
    if not isinstance(value, str):
        return value
    value = value.split("-")[0]
    value = value.strip()
    return value


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to an ADS export DataFrame.

    1. Replace empty ``References`` (``'[]'``) with empty lists.
    2. Drop rows with non-numeric ``Year``; convert ``Year`` to int.
    3. Unescape HTML in ``Title``, ``Abstract``, ``full_text``.
    4. Remove hyphens from ``Author`` (preserves DOI hyphens).
    5. Normalise ``Issue``, ``Volume``, ``First Page``, ``Last Page``.
    """
    df = df.copy()

    if "References" in df.columns:
        df["References"] = df["References"].apply(
            lambda x: np.nan if x == "[]" else x
        )
        df["References"] = df["References"].fillna("").apply(
            lambda x: x if isinstance(x, list) else ([] if x == "" else x)
        )

    df = df[pd.to_numeric(df["Year"], errors="coerce").notnull()]
    df["Year"] = df["Year"].astype(int)

    for col in ("Title", "Abstract", "full_text"):
        if col in df.columns:
            df[col] = df[col].apply(clean_html)

    if "Author" in df.columns:
        df["Author"] = df["Author"].str.replace("-", "", regex=False)

    for col in ("Issue", "Volume", "First Page", "Last Page"):
        if col in df.columns:
            df[col] = df[col].apply(clean_range)

    return df
