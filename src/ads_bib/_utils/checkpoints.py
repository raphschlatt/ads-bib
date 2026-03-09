"""Checkpoint helpers for notebook orchestration."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from ads_bib._utils.io import (
    load_json_lines,
    load_parquet,
    save_json_lines,
    save_parquet,
)

logger = logging.getLogger(__name__)


def save_translated_checkpoint(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Save translated publications/references to global cache and optional run snapshot."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_translated.json"
    ref_path = cache_dir / "references_translated.json"

    save_json_lines(publications, pub_path)
    save_json_lines(references, ref_path)

    if run_data_dir is not None:
        run_data_dir = Path(run_data_dir)
        run_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pub_path, run_data_dir / pub_path.name)
        shutil.copy(ref_path, run_data_dir / ref_path.name)

    logger.info("Translated checkpoint saved to global cache and local run folder.")
    return pub_path, ref_path


def load_translated_checkpoint(
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
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

    pubs = load_json_lines(pub_path)
    refs = load_json_lines(ref_path)

    if run_data_dir is not None:
        run_data_dir = Path(run_data_dir)
        run_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pub_path, run_data_dir / pub_path.name)
        shutil.copy(ref_path, run_data_dir / ref_path.name)

    return pubs, refs


def save_phase3_checkpoint(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Save Phase-3 outputs (tokenized pubs + refs frame) for Phase-4 restart."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_translated_tokenized.json"
    ref_path = cache_dir / "references_translated.json"

    save_json_lines(publications, pub_path)
    save_json_lines(references, ref_path)

    if run_data_dir is not None:
        run_data_dir = Path(run_data_dir)
        run_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pub_path, run_data_dir / pub_path.name)
        shutil.copy(ref_path, run_data_dir / ref_path.name)

    logger.info("Phase 3 checkpoint saved (publications tokenized; refs retained without tokenization).")
    return pub_path, ref_path


def load_phase3_checkpoint(
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase-3 outputs (tokenized pubs + translated refs) from global cache.

    Raises
    ------
    FileNotFoundError
        If the expected cache files are missing.
    """
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_translated_tokenized.json"
    ref_path = cache_dir / "references_translated.json"

    if not pub_path.exists() or not ref_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 3 cache files: {pub_path} and/or {ref_path}"
        )

    pubs = load_json_lines(pub_path)
    refs = load_json_lines(ref_path)
    logger.info(
        "Loaded Phase 3 checkpoint: %s publications, %s references",
        f"{len(pubs):,}", f"{len(refs):,}",
    )

    if run_data_dir is not None:
        run_data_dir = Path(run_data_dir)
        run_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pub_path, run_data_dir / pub_path.name)
        shutil.copy(ref_path, run_data_dir / ref_path.name)

    return pubs, refs


def save_phase4_checkpoint(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Save Phase-4 outputs (disambiguated publications, refs)."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_disambiguated.parquet"
    ref_path = cache_dir / "references_disambiguated.parquet"

    save_parquet(publications, pub_path)
    save_parquet(references, ref_path)

    if run_data_dir is not None:
        run_data_dir = Path(run_data_dir)
        run_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pub_path, run_data_dir / pub_path.name)
        shutil.copy(ref_path, run_data_dir / ref_path.name)

    logger.info("Phase 4 checkpoint saved (publications, references).")
    return pub_path, ref_path


def load_phase4_checkpoint(
    *,
    cache_dir: Path | str,
    run_data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase-4 outputs (disambiguated publications, refs)."""
    cache_dir = Path(cache_dir)
    pub_path = cache_dir / "publications_disambiguated.parquet"
    ref_path = cache_dir / "references_disambiguated.parquet"

    if not pub_path.exists() or not ref_path.exists():
        raise FileNotFoundError(
            "Missing Phase 4 cache files: "
            f"{pub_path} and/or {ref_path}"
        )

    pubs = load_parquet(pub_path)
    refs = load_parquet(ref_path)
    logger.info(
        "Loaded Phase 4 checkpoint: %s publications, %s references",
        f"{len(pubs):,}",
        f"{len(refs):,}",
    )

    if run_data_dir is not None:
        run_data_dir = Path(run_data_dir)
        run_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pub_path, run_data_dir / pub_path.name)
        shutil.copy(ref_path, run_data_dir / ref_path.name)

    return pubs, refs
