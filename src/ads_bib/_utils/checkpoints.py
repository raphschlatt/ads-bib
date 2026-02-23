"""Checkpoint helpers for notebook orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from ads_bib._utils.io import load_json_lines, save_json_lines


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

    print("Translated checkpoint saved to global cache and local run folder.")
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
    ref_path = cache_dir / "references_translated_tokenized.json"

    save_json_lines(publications, pub_path)
    save_json_lines(references, ref_path)

    if run_data_dir is not None:
        run_data_dir = Path(run_data_dir)
        run_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pub_path, run_data_dir / pub_path.name)
        shutil.copy(ref_path, run_data_dir / ref_path.name)

    print("Phase 3 checkpoint saved (publications tokenized; refs retained without tokenization).")
    return pub_path, ref_path
