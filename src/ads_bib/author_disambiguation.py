"""Author name disambiguation adapter for external AND packages."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path

import pandas as pd

from ads_bib._utils.authors import author_list as _author_list
from ads_bib._utils.checkpoints import load_phase4_checkpoint, save_phase4_checkpoint
from ads_bib._utils.io import save_parquet

logger = logging.getLogger(__name__)

_MENTION_ASSIGNMENT_COLUMNS = {
    "mention_id",
    "author_uid",
    "author_display_name",
}
_AUTHOR_ENTITY_COLUMNS = {
    "author_uid",
    "author_display_name",
    "aliases",
    "mention_count",
    "document_count",
    "unique_mention_count",
    "display_name_method",
}

DisambiguateMentions = Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]


def _resolve_affiliation(value: object, position: int) -> object:
    if isinstance(value, list):
        return value[position] if position < len(value) else None
    if isinstance(value, tuple):
        return value[position] if position < len(value) else None
    if isinstance(value, pd.Series):
        return value.iloc[position] if position < len(value) else None
    return value


def _build_author_mentions(
    publications: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frames = (
        ("publication", publications),
        ("reference", references),
    )
    for document_type, frame in frames:
        if "Author" not in frame.columns:
            raise ValueError(f"{document_type}s DataFrame must contain an 'Author' column.")
        for record_row, row in enumerate(frame.itertuples(index=False), start=0):
            row_dict = row._asdict()
            authors = _author_list(row_dict.get("Author"))
            document_id = str(row_dict.get("Bibcode") or f"{document_type}:{record_row}")
            for author_position, raw_mention in enumerate(authors):
                rows.append(
                    {
                        "mention_id": f"{document_type}:{record_row}:{author_position}",
                        "document_id": document_id,
                        "document_type": document_type,
                        "record_row": record_row,
                        "author_position": author_position,
                        "raw_mention": raw_mention,
                        "affiliation": _resolve_affiliation(
                            row_dict.get("Affiliation"),
                            author_position,
                        ),
                        "year": row_dict.get("Year"),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "mention_id",
            "document_id",
            "document_type",
            "record_row",
            "author_position",
            "raw_mention",
            "affiliation",
            "year",
        ],
    )


def _validate_mention_assignments(
    mentions: pd.DataFrame,
    mention_assignments: pd.DataFrame,
) -> pd.DataFrame:
    missing = _MENTION_ASSIGNMENT_COLUMNS.difference(mention_assignments.columns)
    if missing:
        raise ValueError(
            "mention_assignments is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    assignments = mention_assignments.copy()
    if assignments["mention_id"].isna().any():
        raise ValueError("mention_assignments contains null mention_id values.")
    if assignments["mention_id"].duplicated().any():
        raise ValueError("mention_assignments must contain unique mention_id values.")
    if assignments["author_uid"].isna().any():
        raise ValueError("mention_assignments contains null author_uid values.")
    if assignments["author_display_name"].isna().any():
        raise ValueError("mention_assignments contains null author_display_name values.")

    expected = set(mentions["mention_id"].tolist())
    actual = set(assignments["mention_id"].tolist())
    if actual != expected:
        missing_ids = sorted(expected - actual)
        extra_ids = sorted(actual - expected)
        problems: list[str] = []
        if missing_ids:
            problems.append(f"missing mention_id(s): {missing_ids[:5]}")
        if extra_ids:
            problems.append(f"unexpected mention_id(s): {extra_ids[:5]}")
        raise ValueError("mention_assignments does not cover the mention contract: " + "; ".join(problems))

    return assignments


def _validate_author_entities(authors: pd.DataFrame) -> pd.DataFrame:
    missing = _AUTHOR_ENTITY_COLUMNS.difference(authors.columns)
    if missing:
        raise ValueError(
            "authors is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    author_entities = authors.copy()
    if author_entities["author_uid"].isna().any():
        raise ValueError("authors contains null author_uid values.")
    if author_entities["author_uid"].duplicated().any():
        raise ValueError("authors must contain unique author_uid values.")
    if author_entities["author_display_name"].isna().any():
        raise ValueError("authors contains null author_display_name values.")
    return author_entities


def _apply_assignments_to_frame(
    frame: pd.DataFrame,
    *,
    document_type: str,
    mention_assignments: pd.DataFrame,
) -> pd.DataFrame:
    frame_out = frame.copy()
    grouped = (
        mention_assignments.loc[mention_assignments["document_type"] == document_type]
        .sort_values(["record_row", "author_position"])
        .groupby("record_row", sort=True)
    )

    author_uids: list[list[str]] = [[] for _ in range(len(frame_out))]
    author_display_names: list[list[str]] = [[] for _ in range(len(frame_out))]

    for record_row, group in grouped:
        idx = int(record_row)
        author_uids[idx] = group["author_uid"].astype(str).tolist()
        author_display_names[idx] = group["author_display_name"].astype(str).tolist()

    frame_out["author_uids"] = author_uids
    frame_out["author_display_names"] = author_display_names
    return frame_out


def _save_run_snapshot_only(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    authors: pd.DataFrame,
    *,
    run_data_dir: Path | str,
) -> None:
    run_data_dir = Path(run_data_dir)
    save_parquet(publications, run_data_dir / "publications_disambiguated.parquet")
    save_parquet(references, run_data_dir / "references_disambiguated.parquet")
    save_parquet(authors, run_data_dir / "authors.parquet")


def _normalize_list_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert cached array-like cells back to plain Python lists."""
    frame_out = frame.copy()
    for column in columns:
        if column not in frame_out.columns:
            continue
        frame_out[column] = frame_out[column].apply(
            lambda value: list(value)
            if isinstance(value, (list, tuple, pd.Series))
            else value.tolist()
            if hasattr(value, "tolist")
            else []
            if pd.isna(value)
            else value
        )
    return frame_out


def apply_author_disambiguation(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    disambiguate_mentions: DisambiguateMentions,
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
    run_data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply an external mention-based AND runner and map results back to ADS frames."""
    if cache_dir is not None and not force_refresh:
        try:
            pubs_cached, refs_cached, authors_cached = load_phase4_checkpoint(
                cache_dir=cache_dir,
                run_data_dir=run_data_dir,
            )
        except FileNotFoundError:
            pass
        else:
            logger.info("Loaded cached author disambiguation outputs.")
            return (
                _normalize_list_columns(pubs_cached, ["author_uids", "author_display_names"]),
                _normalize_list_columns(refs_cached, ["author_uids", "author_display_names"]),
                authors_cached,
            )

    mentions = _build_author_mentions(publications, references)
    if mentions.empty:
        pubs_out = publications.copy()
        refs_out = references.copy()
        pubs_out["author_uids"] = [[] for _ in range(len(pubs_out))]
        pubs_out["author_display_names"] = [[] for _ in range(len(pubs_out))]
        refs_out["author_uids"] = [[] for _ in range(len(refs_out))]
        refs_out["author_display_names"] = [[] for _ in range(len(refs_out))]
        authors = pd.DataFrame(columns=sorted(_AUTHOR_ENTITY_COLUMNS))
        if cache_dir is not None:
            save_phase4_checkpoint(
                pubs_out,
                refs_out,
                authors,
                cache_dir=cache_dir,
                run_data_dir=run_data_dir,
            )
        elif run_data_dir is not None:
            _save_run_snapshot_only(
                pubs_out,
                refs_out,
                authors,
                run_data_dir=run_data_dir,
            )
        logger.info("Author disambiguation skipped: no author mentions found.")
        return pubs_out, refs_out, authors

    mention_assignments, authors = disambiguate_mentions(mentions.copy())
    mention_assignments = _validate_mention_assignments(mentions, mention_assignments)
    authors = _validate_author_entities(authors)
    mention_assignments = mention_assignments.merge(
        mentions[["mention_id", "document_type", "record_row", "author_position"]],
        on="mention_id",
        how="left",
        validate="one_to_one",
    )

    pubs_out = _apply_assignments_to_frame(
        publications,
        document_type="publication",
        mention_assignments=mention_assignments,
    )
    refs_out = _apply_assignments_to_frame(
        references,
        document_type="reference",
        mention_assignments=mention_assignments,
    )

    if cache_dir is not None:
        save_phase4_checkpoint(
            pubs_out,
            refs_out,
            authors,
            cache_dir=cache_dir,
            run_data_dir=run_data_dir,
        )
    elif run_data_dir is not None:
        _save_run_snapshot_only(
            pubs_out,
            refs_out,
            authors,
            run_data_dir=run_data_dir,
        )

    logger.info(
        "Author disambiguation complete | mentions=%s | unique authors=%s",
        f"{len(mentions):,}",
        f"{len(authors):,}",
    )
    return pubs_out, refs_out, authors


__all__ = ["apply_author_disambiguation"]
