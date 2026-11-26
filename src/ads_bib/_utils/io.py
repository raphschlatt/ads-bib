"""Thin I/O wrappers for consistent data persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def sha256_file(path: Path | str) -> str:
    """Calculate the SHA-256 hash of a file."""
    path = Path(path)
    if not path.exists():
        return ""
    
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


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


def lists_from_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with numpy-array cells turned into Python lists.

    Parquet hands list columns back as numpy arrays. Text made from such a cell differs
    from the text of a list (``['a' 'b']``, cut short with ``...`` above 1,000 entries),
    and ``if cell:`` raises. Frames built in memory or read from JSON lines hold lists.
    """
    converted = {
        column: df[column].map(lambda value: value.tolist() if isinstance(value, np.ndarray) else value)
        for column in df.columns[df.dtypes == object]
        if any(isinstance(value, np.ndarray) for value in df[column])
    }
    return df.assign(**converted) if converted else df


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame; list columns hold Python lists."""
    return lists_from_arrays(pd.read_parquet(path))


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
