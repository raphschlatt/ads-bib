"""Helpers for run-local dataset bundle artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ads_bib._utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)


def ensure_run_references_artifact(
    *,
    refs: pd.DataFrame,
    run_data_dir: Path | str,
    force: bool = False,
) -> Path:
    run_data_dir = Path(run_data_dir)
    references_path = run_data_dir / "references.parquet"
    if force or not references_path.exists():
        save_parquet(refs, references_path)
    return references_path


def write_dataset_bundle(
    *,
    publications: pd.DataFrame,
    refs: pd.DataFrame | None,
    run_data_dir: Path | str,
    run_id: str,
    source_stage: str,
    and_enabled: bool,
) -> Path:
    run_data_dir = Path(run_data_dir)
    publications_path = run_data_dir / "publications.parquet"
    manifest_path = run_data_dir / "dataset_manifest.json"

    if refs is None:
        logger.info("Skipping dataset bundle export at %s: refs are not available.", source_stage)
        return publications_path

    references_path = ensure_run_references_artifact(
        refs=refs,
        run_data_dir=run_data_dir,
    )
    save_parquet(publications, publications_path)

    try:
        from ads_bib import __version__ as ads_bib_version
    except Exception:
        ads_bib_version = "0.0.0"

    coordinate_columns = [
        column
        for column in ("embedding_2d_x", "embedding_2d_y")
        if column in publications.columns
    ]
    manifest = {
        "schema_version": 1,
        "producer": "ads_bib",
        "producer_version": ads_bib_version,
        "run_id": run_id,
        "source_stage": source_stage,
        "and_enabled": bool(and_enabled),
        "publications_path": publications_path.name,
        "references_path": references_path.name,
        "coordinate_columns": coordinate_columns,
        "counts": {
            "publications": int(len(publications)),
            "references": int(len(refs)),
        },
        "has_author_uids": bool(
            "author_uids" in publications.columns and "author_uids" in refs.columns
        ),
        "has_author_display_names": bool(
            "author_display_names" in publications.columns
            and "author_display_names" in refs.columns
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info(
        "Dataset bundle exported at %s | publications=%s | references=%s",
        source_stage,
        f"{len(publications):,}",
        f"{len(refs):,}",
    )
    return publications_path


def load_curated_dataset(*, run_data_dir: Path | str) -> pd.DataFrame:
    curated_path = Path(run_data_dir) / "publications.parquet"
    if curated_path.exists():
        return load_parquet(curated_path)
    raise FileNotFoundError(f"Curated dataset not found at {curated_path}")


__all__ = [
    "ensure_run_references_artifact",
    "load_curated_dataset",
    "write_dataset_bundle",
]
