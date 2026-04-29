"""Helpers for run-local dataset bundle artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ads_bib._utils.io import load_parquet, save_parquet, sha256_file

logger = logging.getLogger(__name__)

_FRONT_COLUMNS: tuple[str, ...] = (
    "Bibcode",
    "Year",
    "Author",
    "author_display_names",
    "author_uids",
    "Title",
    "Title_lang",
    "Title_en",
    "Abstract",
    "Abstract_lang",
    "Abstract_en",
    "Journal",
    "Journal Abbreviation",
    "Citation Count",
    "DOI",
    "Affiliation",
    "Keywords",
    "Category",
    "Volume",
    "Issue",
    "First Page",
    "Last Page",
    "References",
    "full_text",
    "tokens",
    "topic_id",
    "Name",
    "Main",
    "MMR",
    "POS",
    "KeyBERT",
    "topic_primary_layer_index",
    "topic_layer_count",
)

_TAIL_COLUMNS: tuple[str, ...] = (
    "full_embeddings",
)

_TOPIC_INFO_FRONT_COLUMNS: tuple[str, ...] = (
    "Topic",
    "Count",
    "Name",
    "Main",
    "MMR",
    "POS",
    "KeyBERT",
    "Representation",
    "Representative_Docs",
)


def _is_embedding_column(column: str) -> bool:
    return column.startswith("embedding_5d_") or column in {"embedding_2d_x", "embedding_2d_y"}


def _embedding_sort_key(column: str) -> tuple[int, int, str]:
    if column.startswith("embedding_5d_"):
        suffix = column.removeprefix("embedding_5d_")
        if suffix.isdigit():
            return (0, int(suffix), column)
    if column == "embedding_2d_x":
        return (1, 0, column)
    if column == "embedding_2d_y":
        return (1, 1, column)
    return (10**6, 10**6, column)


def _topic_layer_sort_key(column: str) -> tuple[int, int, str]:
    if column.startswith("topic_layer_"):
        parts = column.split("_")
        if len(parts) >= 4 and parts[2].isdigit():
            kind = 0 if column.endswith("_id") else 1
            return (int(parts[2]), kind, column)
    if column.startswith("Topic_Layer_"):
        suffix = column.removeprefix("Topic_Layer_")
        if suffix.isdigit():
            return (int(suffix), 2, column)
    return (10**6, 10**6, column)


def _ordered_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    front = [column for column in _FRONT_COLUMNS if column in columns]
    topic_layers = sorted(
        [
            column
            for column in columns
            if (column.startswith("topic_layer_") or column.startswith("Topic_Layer_"))
            and column not in front
        ],
        key=_topic_layer_sort_key,
    )
    embeddings = sorted(
        [column for column in columns if _is_embedding_column(column)],
        key=_embedding_sort_key,
    )
    tail = [column for column in _TAIL_COLUMNS if column in columns]
    used = set(front) | set(topic_layers) | set(embeddings) | set(tail)
    middle = [column for column in columns if column not in used]
    return df.loc[:, [*front, *topic_layers, *middle, *embeddings, *tail]]


def _ordered_topic_info_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    front = [column for column in _TOPIC_INFO_FRONT_COLUMNS if column in columns]
    used = set(front)
    rest = [column for column in columns if column not in used]
    return df.loc[:, [*front, *rest]].copy()


def ensure_run_references_artifact(
    *,
    refs: pd.DataFrame,
    run_data_dir: Path | str,
    force: bool = False,
) -> Path:
    run_data_dir = Path(run_data_dir)
    references_path = run_data_dir / "references.parquet"
    if force or not references_path.exists():
        save_parquet(_ordered_dataset_frame(refs), references_path)
    return references_path


def ensure_run_topic_info_artifact(
    *,
    topic_info: pd.DataFrame | None,
    run_data_dir: Path | str,
    force: bool = True,
) -> Path | None:
    if topic_info is None:
        return None
    run_data_dir = Path(run_data_dir)
    topic_info_path = run_data_dir / "topic_info.parquet"
    if force or not topic_info_path.exists():
        save_parquet(_ordered_topic_info_frame(topic_info), topic_info_path)
    return topic_info_path


def _artifact_manifest(path: Path | None) -> dict[str, int | str] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return {
        "path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def write_dataset_bundle(
    *,
    publications: pd.DataFrame,
    refs: pd.DataFrame | None,
    topic_info: pd.DataFrame | None = None,
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
    topic_info_path = ensure_run_topic_info_artifact(
        topic_info=topic_info,
        run_data_dir=run_data_dir,
    )
    if topic_info_path is None:
        existing_topic_info_path = run_data_dir / "topic_info.parquet"
        topic_info_path = existing_topic_info_path if existing_topic_info_path.exists() else None
    publications = _ordered_dataset_frame(publications)
    save_parquet(publications, publications_path)

    try:
        from ads_bib import __version__ as ads_bib_version
    except Exception:
        ads_bib_version = "0.0.0"

    coordinate_columns = sorted(
        [column for column in publications.columns if _is_embedding_column(column)],
        key=_embedding_sort_key,
    )
    artifact_paths = {
        "publications": publications_path,
        "references": references_path,
        "topic_info": topic_info_path,
    }
    artifacts = {
        name: payload
        for name, path in artifact_paths.items()
        if (payload := _artifact_manifest(path)) is not None
    }

    manifest = {
        "schema_version": 1,
        "producer": "ads_bib",
        "producer_version": ads_bib_version,
        "run_id": run_id,
        "source_stage": source_stage,
        "and_enabled": bool(and_enabled),
        "publications_path": publications_path.name,
        "references_path": references_path.name,
        "topic_info_path": topic_info_path.name if topic_info_path is not None else None,
        "coordinate_columns": coordinate_columns,
        "counts": {
            "publications": int(len(publications)),
            "references": int(len(refs)),
        },
        "artifacts": artifacts,
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
