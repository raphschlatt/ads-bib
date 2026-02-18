"""Thin I/O wrappers for consistent data persistence."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def save_json_lines(df: pd.DataFrame, path: Path | str) -> None:
    """Save a DataFrame as newline-delimited JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True)


def load_json_lines(path: Path | str) -> pd.DataFrame:
    """Load a newline-delimited JSON file into a DataFrame."""
    return pd.read_json(path, lines=True)


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Save a DataFrame as Snappy-compressed Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy")


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def save_pickle(data, path: Path | str) -> None:
    """Pickle an arbitrary object to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def load_pickle(path: Path | str):
    """Unpickle an object from *path*."""
    with open(path, "rb") as fh:
        return pickle.load(fh)
