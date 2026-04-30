"""Helpers for run-local dataset bundle artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ads_bib._utils.io import load_parquet, save_parquet, sha256_file

logger = logging.getLogger(__name__)
_MAX_REPORT_EXAMPLES = 20

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


def _sync_topic_info_counts(
    topic_info: pd.DataFrame | None,
    publications: pd.DataFrame,
) -> pd.DataFrame | None:
    if topic_info is None:
        return None
    if not {"Topic", "Count"}.issubset(topic_info.columns) or "topic_id" not in publications.columns:
        return topic_info
    counts = publications["topic_id"].value_counts(dropna=False).to_dict()
    out = topic_info.copy()
    out["Count"] = out["Topic"].map(counts).fillna(0).astype("int64")
    return out


def _is_list_like(value: object) -> bool:
    return bool(pd.api.types.is_list_like(value)) and not isinstance(value, (str, bytes, dict))


def _as_list(value: object) -> list[object]:
    if _is_list_like(value):
        items = value.tolist() if hasattr(value, "tolist") else list(value)  # type: ignore[attr-defined]
        return list(items)
    if value is None:
        return []
    try:
        if bool(pd.isna(value)):
            return []
    except Exception:
        pass
    return [value]


def _has_meaningful_value(value: object) -> bool:
    if _is_list_like(value):
        return any(_has_meaningful_value(item) for item in _as_list(value))
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return bool(pd.notna(value))
    except Exception:
        return True


def _normalize_bibcode(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _normalize_reference_list(value: object) -> tuple[list[str], int, int]:
    refs: list[str] = []
    seen: set[str] = set()
    empty_removed = 0
    duplicates_removed = 0
    for item in _as_list(value):
        ref = _normalize_bibcode(item)
        if not ref:
            empty_removed += 1
            continue
        if ref in seen:
            duplicates_removed += 1
            continue
        refs.append(ref)
        seen.add(ref)
    return refs, empty_removed, duplicates_removed


def _reference_count(value: object) -> int:
    refs, _, _ = _normalize_reference_list(value)
    return len(refs)


def _row_non_empty_count(row: pd.Series) -> int:
    return sum(1 for value in row.tolist() if _has_meaningful_value(value))


def _best_duplicate_row_index(group: pd.DataFrame, order_column: str) -> Any:
    def _score(idx: Any) -> tuple[int, int, int]:
        row = group.loc[idx]
        return (
            _reference_count(row.get("References")),
            _row_non_empty_count(row),
            -int(row[order_column]),
        )

    return max(group.index, key=_score)


def _dedupe_bibcodes(
    frame: pd.DataFrame,
    *,
    frame_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {
        "input_rows": int(len(frame)),
        "empty_bibcodes_removed": 0,
        "duplicate_bibcodes": 0,
        "duplicate_rows_removed": 0,
        "duplicate_bibcode_examples": [],
    }
    if "Bibcode" not in frame.columns:
        report["missing_bibcode_column"] = True
        return frame.copy(), report

    order_column = "__ads_bib_bundle_order__"
    out = frame.copy()
    out[order_column] = range(len(out))
    out["Bibcode"] = out["Bibcode"].apply(_normalize_bibcode)

    empty_mask = out["Bibcode"].eq("")
    if empty_mask.any():
        report["empty_bibcodes_removed"] = int(empty_mask.sum())
        out = out.loc[~empty_mask].copy()

    if out.empty:
        report["output_rows"] = 0
        return out.drop(columns=[order_column], errors="ignore"), report

    duplicate_mask = out["Bibcode"].duplicated(keep=False)
    duplicate_keys = out.loc[duplicate_mask, "Bibcode"].drop_duplicates().tolist()
    report["duplicate_bibcodes"] = int(len(duplicate_keys))
    report["duplicate_bibcode_examples"] = duplicate_keys[:_MAX_REPORT_EXAMPLES]

    keep_indices: list[Any] = []
    for _, group in out.groupby("Bibcode", sort=False):
        if len(group) == 1:
            keep_indices.append(group.index[0])
        else:
            keep_indices.append(_best_duplicate_row_index(group, order_column))

    out = out.loc[keep_indices].sort_values(order_column).drop(columns=[order_column])
    removed = int(report["input_rows"]) - int(report["empty_bibcodes_removed"]) - len(out)
    report["duplicate_rows_removed"] = int(removed)
    report["output_rows"] = int(len(out))

    if report["empty_bibcodes_removed"] or report["duplicate_rows_removed"]:
        logger.info(
            "Prepared %s bundle rows: %s -> %s",
            frame_name,
            f"{int(report['input_rows']):,}",
            f"{len(out):,}",
        )
    return out.reset_index(drop=True), report


def _is_placeholder_author_uid(uid: str, display_name: str = "") -> bool:
    uid_norm = uid.strip().lower()
    display_norm = display_name.strip().lower()
    if uid_norm in {"", "nan", "none", "unknown", "no author", "no_author", "no-author", "n.author"}:
        return True
    uid_segments = {segment.strip().lower() for segment in uid_norm.split("::")}
    if uid_segments.intersection({"", "unknown", "no author", "no_author", "no-author", "n.author"}):
        return True
    if not uid_norm and display_norm in {"", "unknown", "no author", "no_author", "no-author"}:
        return True
    return False


def _clean_author_identity_columns(
    frame: pd.DataFrame,
    *,
    frame_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {
        "placeholder_uid_mentions_removed": 0,
        "duplicate_uid_mentions_removed": 0,
        "rows_with_uid_changes": 0,
        "placeholder_uid_examples": [],
    }
    if "author_uids" not in frame.columns:
        return frame.copy(), report

    out = frame.copy()
    has_display_names = "author_display_names" in out.columns
    cleaned_uids: list[list[str]] = []
    cleaned_names: list[list[str]] = []
    placeholder_examples: list[str] = []
    placeholder_example_seen: set[str] = set()

    for row_idx, uid_value in enumerate(out["author_uids"].tolist()):
        uid_items = [_normalize_bibcode(item) for item in _as_list(uid_value)]
        display_items = (
            [_normalize_bibcode(item) for item in _as_list(out.iloc[row_idx]["author_display_names"])]
            if has_display_names
            else []
        )
        seen: set[str] = set()
        row_uids: list[str] = []
        row_names: list[str] = []
        changed = False

        for idx, uid in enumerate(uid_items):
            display = display_items[idx] if idx < len(display_items) else ""
            if _is_placeholder_author_uid(uid, display):
                report["placeholder_uid_mentions_removed"] = int(report["placeholder_uid_mentions_removed"]) + 1
                if (
                    uid
                    and uid not in placeholder_example_seen
                    and len(placeholder_examples) < _MAX_REPORT_EXAMPLES
                ):
                    placeholder_examples.append(uid)
                    placeholder_example_seen.add(uid)
                changed = True
                continue
            if uid in seen:
                report["duplicate_uid_mentions_removed"] = int(report["duplicate_uid_mentions_removed"]) + 1
                changed = True
                continue
            seen.add(uid)
            row_uids.append(uid)
            row_names.append(display)

        if changed:
            report["rows_with_uid_changes"] = int(report["rows_with_uid_changes"]) + 1
        cleaned_uids.append(row_uids)
        cleaned_names.append(row_names)

    out["author_uids"] = cleaned_uids
    if has_display_names:
        out["author_display_names"] = cleaned_names
    report["placeholder_uid_examples"] = placeholder_examples
    if report["rows_with_uid_changes"]:
        logger.info(
            "Prepared %s author_uids: %s rows changed",
            frame_name,
            f"{int(report['rows_with_uid_changes']):,}",
        )
    return out, report


def _clean_publication_references(
    publications: pd.DataFrame,
    *,
    valid_reference_bibcodes: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {
        "empty_mentions_removed": 0,
        "duplicate_mentions_removed": 0,
        "missing_reference_bibcodes": 0,
        "missing_mentions_removed": 0,
        "missing_reference_examples": [],
    }
    if "References" not in publications.columns:
        return publications.copy(), report

    out = publications.copy()
    missing_counts: dict[str, int] = {}
    cleaned_rows: list[list[str]] = []
    for value in out["References"].tolist():
        refs, empty_removed, duplicate_removed = _normalize_reference_list(value)
        report["empty_mentions_removed"] = int(report["empty_mentions_removed"]) + empty_removed
        report["duplicate_mentions_removed"] = int(report["duplicate_mentions_removed"]) + duplicate_removed
        kept: list[str] = []
        for ref in refs:
            if ref not in valid_reference_bibcodes:
                missing_counts[ref] = missing_counts.get(ref, 0) + 1
                continue
            kept.append(ref)
        cleaned_rows.append(kept)

    out["References"] = cleaned_rows
    missing_examples = sorted(
        (
            {"bibcode": bibcode, "mentions_removed": count}
            for bibcode, count in missing_counts.items()
        ),
        key=lambda item: (-int(item["mentions_removed"]), str(item["bibcode"])),
    )
    report["missing_reference_bibcodes"] = int(len(missing_counts))
    report["missing_mentions_removed"] = int(sum(missing_counts.values()))
    report["missing_reference_examples"] = missing_examples[:_MAX_REPORT_EXAMPLES]
    return out, report


def prepare_dataset_bundle(
    publications: pd.DataFrame,
    refs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return cleaned final bundle frames plus a manifest-ready report."""
    report: dict[str, object] = {
        "schema_version": 1,
        "input_counts": {
            "publications": int(len(publications)),
            "references": int(len(refs)),
        },
    }

    pubs_clean, pub_report = _dedupe_bibcodes(publications, frame_name="publications")
    refs_clean, ref_report = _dedupe_bibcodes(refs, frame_name="references")
    pubs_clean, pub_refs_report = _clean_publication_references(
        pubs_clean,
        valid_reference_bibcodes=set(refs_clean["Bibcode"].astype(str)) if "Bibcode" in refs_clean.columns else set(),
    )
    pubs_clean, pub_author_report = _clean_author_identity_columns(
        pubs_clean,
        frame_name="publications",
    )
    refs_clean, ref_author_report = _clean_author_identity_columns(
        refs_clean,
        frame_name="references",
    )

    report.update(
        {
            "output_counts": {
                "publications": int(len(pubs_clean)),
                "references": int(len(refs_clean)),
            },
            "publications": {
                **pub_report,
                "references": pub_refs_report,
                "author_uids": pub_author_report,
            },
            "references": {
                **ref_report,
                "author_uids": ref_author_report,
            },
        }
    )
    return pubs_clean, refs_clean, report


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
    prepare_bundle: bool = True,
    cleaning_report: dict[str, object] | None = None,
) -> Path:
    run_data_dir = Path(run_data_dir)
    publications_path = run_data_dir / "publications.parquet"
    manifest_path = run_data_dir / "dataset_manifest.json"

    if refs is None:
        logger.info("Skipping dataset bundle export at %s: refs are not available.", source_stage)
        return publications_path

    if prepare_bundle:
        publications, refs, cleaning_report = prepare_dataset_bundle(publications, refs)
    elif cleaning_report is None:
        cleaning_report = {
            "schema_version": 1,
            "input_counts": {"publications": int(len(publications)), "references": int(len(refs))},
            "output_counts": {"publications": int(len(publications)), "references": int(len(refs))},
        }

    topic_info = _sync_topic_info_counts(topic_info, publications)
    references_path = ensure_run_references_artifact(
        refs=refs,
        run_data_dir=run_data_dir,
        force=True,
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
        "cleaning": cleaning_report,
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
    "prepare_dataset_bundle",
    "write_dataset_bundle",
]
