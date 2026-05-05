"""Content-based snapshot helpers for notebook and CLI orchestration."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ads_bib._utils.io import (
    load_json_lines,
    load_parquet,
    save_json_lines,
    save_parquet,
)

logger = logging.getLogger(__name__)

_TRANSLATED_METADATA = "translated_snapshot_metadata.json"
_TOKENIZED_METADATA = "tokenized_snapshot_metadata.json"


def _copy_snapshot_pair(
    pub_path: Path,
    ref_path: Path,
    *,
    run_data_dir: Path | str | None,
) -> None:
    if run_data_dir is None:
        return

    run_data_dir = Path(run_data_dir)
    run_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pub_path, run_data_dir / pub_path.name)
    shutil.copy(ref_path, run_data_dir / ref_path.name)


def _write_snapshot_metadata(
    *,
    cache_dir: Path,
    filename: str,
    metadata: Mapping[str, Any] | None,
    run_data_dir: Path | str | None,
) -> None:
    if metadata is None:
        return
    path = cache_dir / filename
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if run_data_dir is not None:
        run_dir = Path(run_data_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, run_dir / filename)


def _assert_snapshot_metadata(
    *,
    cache_dir: Path,
    filename: str,
    expected_metadata: Mapping[str, Any] | None,
) -> None:
    if expected_metadata is None:
        return
    path = cache_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing snapshot metadata file: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata != dict(expected_metadata):
        raise FileNotFoundError(f"Snapshot metadata mismatch for {path}")


def save_translated_snapshot(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save translated publications/references to global cache and optional run snapshot."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_translated.json"
    ref_path = cache_dir / "references_translated.json"

    save_json_lines(publications, pub_path)
    save_json_lines(references, ref_path)
    _copy_snapshot_pair(pub_path, ref_path, run_data_dir=run_data_dir)
    _write_snapshot_metadata(
        cache_dir=cache_dir,
        filename=_TRANSLATED_METADATA,
        metadata=metadata,
        run_data_dir=run_data_dir,
    )

    logger.info("Translated checkpoint saved to global cache and local run folder.")
    return pub_path, ref_path


def load_translated_snapshot(
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load translated publications/references from cache and optional run snapshot."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_translated.json"
    ref_path = cache_dir / "references_translated.json"

    if not pub_path.exists() or not ref_path.exists():
        raise FileNotFoundError(
            "Missing translated cache files. "
            f"Expected: {pub_path} and {ref_path}"
        )
    _assert_snapshot_metadata(
        cache_dir=cache_dir,
        filename=_TRANSLATED_METADATA,
        expected_metadata=expected_metadata,
    )

    pubs = load_json_lines(pub_path)
    refs = load_json_lines(ref_path)
    _copy_snapshot_pair(pub_path, ref_path, run_data_dir=run_data_dir)

    return pubs, refs


def save_tokenized_snapshot(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save tokenized-publications + translated-references snapshot."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_tokenized.json"
    ref_path = cache_dir / "references_translated.json"

    save_json_lines(publications, pub_path)
    save_json_lines(references, ref_path)
    _copy_snapshot_pair(pub_path, ref_path, run_data_dir=run_data_dir)
    _write_snapshot_metadata(
        cache_dir=cache_dir,
        filename=_TOKENIZED_METADATA,
        metadata=metadata,
        run_data_dir=run_data_dir,
    )

    logger.info("Tokenized snapshot saved (publications tokenized; refs retained without tokenization).")
    return pub_path, ref_path


def load_tokenized_snapshot(
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tokenized-publications + translated-references snapshot.

    Raises
    ------
    FileNotFoundError
        If the expected cache files are missing.
    """
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_tokenized.json"
    ref_path = cache_dir / "references_translated.json"

    if not pub_path.exists() or not ref_path.exists():
        raise FileNotFoundError(
            f"Missing tokenized snapshot files: {pub_path} and/or {ref_path}"
        )
    _assert_snapshot_metadata(
        cache_dir=cache_dir,
        filename=_TOKENIZED_METADATA,
        expected_metadata=expected_metadata,
    )

    pubs = load_json_lines(pub_path)
    refs = load_json_lines(ref_path)
    logger.info(
        "Loaded tokenized snapshot: %s publications, %s references",
        f"{len(pubs):,}", f"{len(refs):,}",
    )
    _copy_snapshot_pair(pub_path, ref_path, run_data_dir=run_data_dir)

    return pubs, refs


def save_disambiguated_snapshot(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Save disambiguated publications/references snapshot."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_disambiguated.parquet"
    ref_path = cache_dir / "references_disambiguated.parquet"

    save_parquet(publications, pub_path)
    save_parquet(references, ref_path)
    _copy_snapshot_pair(pub_path, ref_path, run_data_dir=run_data_dir)

    logger.info("Disambiguated snapshot saved (publications, references).")
    return pub_path, ref_path


def load_disambiguated_snapshot(
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load disambiguated publications/references snapshot."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_disambiguated.parquet"
    ref_path = cache_dir / "references_disambiguated.parquet"

    if not pub_path.exists() or not ref_path.exists():
        raise FileNotFoundError(
            "Missing disambiguated snapshot files: "
            f"{pub_path} and/or {ref_path}"
        )

    pubs = load_parquet(pub_path)
    refs = load_parquet(ref_path)
    logger.info(
        "Loaded disambiguated snapshot: %s publications, %s references",
        f"{len(pubs):,}",
        f"{len(refs):,}",
    )
    _copy_snapshot_pair(pub_path, ref_path, run_data_dir=run_data_dir)

    return pubs, refs
