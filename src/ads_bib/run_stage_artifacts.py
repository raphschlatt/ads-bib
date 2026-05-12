"""Run-local artifacts used to restart variants from individual stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ads_bib._stage_state import STAGE_ORDER, StageName, validate_stage_name
from ads_bib._utils.io import load_parquet, save_parquet
from ads_bib.run_manager import RunArtifactLayout

_SEARCH_RESULTS_FILE = "search_results.json"
_PUBLICATIONS_FILE = "publications.parquet"
_REFERENCES_FILE = "references.parquet"

_REUSED_ARTIFACT_DIRS: dict[StageName, tuple[str, ...]] = {
    "search": ("search",),
    "export": ("search", "export"),
    "translate": ("search", "export", "translated"),
    "tokenize": ("search", "export", "translated", "tokenized"),
    "author_disambiguation": ("search", "export", "translated", "tokenized", "and"),
    "embeddings": ("search", "export", "translated", "tokenized", "and"),
    "reduction": ("search", "export", "translated", "tokenized", "and"),
    "topic_fit": ("search", "export", "translated", "tokenized", "and"),
    "topic_dataframe": ("search", "export", "translated", "tokenized", "and", "dataset"),
    "visualize": ("search", "export", "translated", "tokenized", "and", "dataset", "plots"),
    "curate": ("search", "export", "translated", "tokenized", "and", "dataset", "plots"),
    "citations": (
        "search",
        "export",
        "translated",
        "tokenized",
        "and",
        "dataset",
        "plots",
        "citations",
    ),
}


def save_search_artifact(
    directory: Path | str,
    *,
    bibcodes: list[str],
    references: list[list[str]],
    esources: list[list[str]],
    fulltext_urls: list[str | None],
) -> Path:
    """Persist ADS search results inside one run directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _SEARCH_RESULTS_FILE
    payload = {
        "schema_version": 1,
        "bibcodes": bibcodes,
        "references": references,
        "esources": esources,
        "fulltext_urls": fulltext_urls,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_search_artifact(directory: Path | str) -> tuple[list[str], list[list[str]], list[list[str]], list[str | None]]:
    """Load ADS search results from one run directory."""
    path = Path(directory) / _SEARCH_RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Run search artifact not found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        list(payload.get("bibcodes", [])),
        [list(value) for value in payload.get("references", [])],
        [list(value) for value in payload.get("esources", [])],
        list(payload.get("fulltext_urls", [])),
    )


def save_frame_pair(
    directory: Path | str,
    *,
    publications: pd.DataFrame,
    refs: pd.DataFrame,
) -> tuple[Path, Path]:
    """Persist aligned publication/reference frames for one stage boundary."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    publications_path = directory / _PUBLICATIONS_FILE
    references_path = directory / _REFERENCES_FILE
    save_parquet(publications, publications_path)
    save_parquet(refs, references_path)
    return publications_path, references_path


def load_frame_pair(directory: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a run-local publication/reference frame pair."""
    directory = Path(directory)
    publications_path = directory / _PUBLICATIONS_FILE
    references_path = directory / _REFERENCES_FILE
    missing = [str(path) for path in (publications_path, references_path) if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Run frame artifacts are missing: {joined}")
    return load_parquet(publications_path), load_parquet(references_path)


def copy_reused_artifacts(
    *,
    base_run_path: Path | str,
    target_run_path: Path | str,
    reused_until: StageName | str | None,
) -> None:
    """Copy reused run artifacts into a variant run so it remains self-contained."""
    if reused_until is None:
        return
    stage = validate_stage_name(reused_until)
    base_layout = RunArtifactLayout.from_run_dir(base_run_path)
    target_layout = RunArtifactLayout.from_run_dir(target_run_path)
    for key in _REUSED_ARTIFACT_DIRS[stage]:
        source = base_layout.as_paths()[key]
        target = target_layout.as_paths()[key]
        if not source.exists():
            continue
        _copy_directory_contents(source, target)


def _copy_directory_contents(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def stage_is_reused(recomputed_from: StageName | str, stage: StageName | str) -> bool:
    """Return whether *stage* is before *recomputed_from*."""
    recomputed = validate_stage_name(recomputed_from)
    candidate = validate_stage_name(stage)
    return STAGE_ORDER.index(candidate) < STAGE_ORDER.index(recomputed)


__all__ = [
    "copy_reused_artifacts",
    "load_frame_pair",
    "load_search_artifact",
    "save_frame_pair",
    "save_search_artifact",
    "stage_is_reused",
]
