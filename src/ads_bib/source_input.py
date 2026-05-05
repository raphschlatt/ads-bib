"""External source-frame loading for non-ADS corpus inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ads_bib._utils.io import load_parquet

PUBLICATION_REQUIRED_COLUMNS = frozenset(
    {"Bibcode", "Year", "Author", "Title", "Abstract", "References"}
)
REFERENCE_REQUIRED_COLUMNS = frozenset({"Bibcode", "Author", "Title"})

_LIST_COLUMNS = frozenset(
    {"Author", "References", "author_uids", "author_display_names", "Keywords", "Category", "Affiliation"}
)
_COMMON_PUBLICATION_COLUMNS = (
    "Bibcode",
    "Year",
    "Author",
    "author_uids",
    "author_display_names",
    "Title",
    "Abstract",
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
    "source",
)
_COMMON_REFERENCE_COLUMNS = (
    "Bibcode",
    "Year",
    "Author",
    "author_uids",
    "author_display_names",
    "Title",
    "Abstract",
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
    "source",
)


def load_source_input_frames(
    *,
    publications_path: Path | str,
    references_path: Path | str,
    project_root: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize external publications/references parquet frames."""
    publications_file = _resolve_path(publications_path, project_root)
    references_file = _resolve_path(references_path, project_root)
    publications = load_parquet(publications_file)
    references = load_parquet(references_file)
    return normalize_source_input_frames(publications, references)


def normalize_source_input_frames(
    publications: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize external source frames into the ADS pipeline source shape."""
    pubs = _normalize_frame(publications, frame_name="publications", allowed_columns=_COMMON_PUBLICATION_COLUMNS)
    refs = _normalize_frame(references, frame_name="references", allowed_columns=_COMMON_REFERENCE_COLUMNS)

    _require_columns(pubs, PUBLICATION_REQUIRED_COLUMNS, frame_name="publications")
    _require_columns(refs, REFERENCE_REQUIRED_COLUMNS, frame_name="references")
    if "Abstract" not in refs.columns:
        refs["Abstract"] = ""

    pubs = _normalize_bibcodes(pubs, frame_name="publications")
    refs = _normalize_bibcodes(refs, frame_name="references")
    pubs = _coerce_list_columns(pubs)
    refs = _coerce_list_columns(refs)

    known_refs = set(refs["Bibcode"])
    pubs["References"] = [
        [ref for ref in refs_list if ref in known_refs]
        for refs_list in pubs["References"]
    ]
    return pubs.reset_index(drop=True), refs.reset_index(drop=True)


def _resolve_path(path: Path | str, project_root: Path | str | None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and project_root is not None:
        resolved = Path(project_root).expanduser() / resolved
    return resolved


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    frame_name: str,
    allowed_columns: tuple[str, ...],
) -> pd.DataFrame:
    out = frame.copy()
    if "Citation Count" not in out.columns and "citation_count" in out.columns:
        out["Citation Count"] = out["citation_count"]
    if "Journal" not in out.columns and "venue" in out.columns:
        out["Journal"] = out["venue"]

    keep = [column for column in allowed_columns if column in out.columns]
    if not keep:
        raise ValueError(f"{frame_name} source input has no usable columns")
    return out.loc[:, keep].copy()


def _require_columns(frame: pd.DataFrame, required: frozenset[str], *, frame_name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{frame_name} source input missing required columns: {joined}")


def _normalize_bibcodes(frame: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
    out = frame.copy()
    out["Bibcode"] = out["Bibcode"].map(_normalize_string)
    empty = out["Bibcode"].eq("")
    if empty.any():
        raise ValueError(f"{frame_name}.Bibcode contains {int(empty.sum())} empty values")
    duplicate = out["Bibcode"].duplicated(keep=False)
    if duplicate.any():
        examples = ", ".join(out.loc[duplicate, "Bibcode"].drop_duplicates().head(5).tolist())
        raise ValueError(f"{frame_name}.Bibcode contains duplicate values: {examples}")
    return out


def _coerce_list_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in sorted(_LIST_COLUMNS.intersection(out.columns)):
        out[column] = out[column].map(_coerce_list)
    return out


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parts = text.split(";") if ";" in text else [text]
        return _dedupe_strings(parts)
    if isinstance(value, bytes):
        return _coerce_list(value.decode("utf-8", errors="replace"))
    if isinstance(value, dict):
        return []
    if pd.api.types.is_list_like(value):
        return _dedupe_strings(list(value))
    try:
        if bool(pd.isna(value)):
            return []
    except Exception:
        pass
    return _dedupe_strings([value])


def _dedupe_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_string(value)
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except Exception:
        pass
    return str(value).strip()


__all__ = [
    "PUBLICATION_REQUIRED_COLUMNS",
    "REFERENCE_REQUIRED_COLUMNS",
    "load_source_input_frames",
    "normalize_source_input_frames",
]
