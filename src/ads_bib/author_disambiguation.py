"""Source-based author disambiguation adapter for external AND packages."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ads_bib._utils.authors import author_list as _author_list
from ads_bib._utils.checkpoints import load_phase4_checkpoint, save_phase4_checkpoint
from ads_bib._utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)

_REQUIRED_SOURCE_COLUMNS = {"Bibcode", "Author", "Year"}
_AUTHOR_UID_COLUMN = "AuthorUID"
_AUTHOR_DISPLAY_NAME_COLUMN = "AuthorDisplayName"
_LIST_OUTPUT_COLUMNS = {
    _AUTHOR_UID_COLUMN: "author_uids",
    _AUTHOR_DISPLAY_NAME_COLUMN: "author_display_names",
}

SourceAndRunner = Callable[..., Any]


def _load_default_and_runner() -> SourceAndRunner:
    try:
        from author_name_disambiguation import run_infer_sources
    except ImportError as exc:
        raise ImportError(
            "Author disambiguation requires the optional `author_name_disambiguation` "
            "package with a `run_infer_sources` export."
        ) from exc
    return run_infer_sources


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def _normalize_list_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame_out = frame.copy()
    for column in columns:
        if column not in frame_out.columns:
            continue
        normalized: list[list[str]] = []
        for value in frame_out[column].tolist():
            if isinstance(value, (list, tuple, pd.Series)):
                items = list(value)
            elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
                items = list(value.tolist())
            elif pd.isna(value):
                items = []
            else:
                items = [value]
            normalized.append([str(item) for item in items])
        frame_out[column] = normalized
    return frame_out


def _normalize_output_list(value: object, *, column: str, expected_len: int) -> list[str]:
    if isinstance(value, (list, tuple, pd.Series)):
        items = list(value)
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        items = list(value.tolist())
    elif pd.isna(value):
        items = []
    else:
        raise ValueError(f"{column} must contain list-like values.")

    if len(items) != expected_len:
        raise ValueError(
            f"{column} length must match Author length for every row "
            f"(expected {expected_len}, got {len(items)})."
        )
    if any(item is None or pd.isna(item) for item in items):
        raise ValueError(f"{column} contains null entries.")
    return [str(item) for item in items]


def _validate_source_frame(frame: pd.DataFrame, *, frame_name: str) -> None:
    if frame.empty and len(frame.columns) == 0:
        return

    missing = sorted(_REQUIRED_SOURCE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {', '.join(missing)}")
    if "Title_en" not in frame.columns and "Title" not in frame.columns:
        raise ValueError(f"{frame_name} must contain Title_en or Title.")
    if "Abstract_en" not in frame.columns and "Abstract" not in frame.columns:
        raise ValueError(f"{frame_name} must contain Abstract_en or Abstract.")


def _resolve_dataset_id(dataset_id: str | None, run_data_dir: Path | str | None) -> str:
    if dataset_id is not None and str(dataset_id).strip():
        return str(dataset_id).strip()
    if run_data_dir is not None:
        return Path(run_data_dir).name
    raise ValueError("dataset_id is required when run_data_dir is not available.")


def _prepare_passthrough_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame_out = frame.copy()
    if "Author" in frame_out.columns:
        frame_out["author_uids"] = frame_out["Author"].apply(lambda value: [""] * len(_author_list(value)))
        frame_out["author_display_names"] = frame_out["Author"].apply(_author_list)
        frame_out["author_uids"] = frame_out["author_uids"].apply(
            lambda values: [] if all(value == "" for value in values) else values
        )
    return frame_out


def _stage_source_frames(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    bridge_root: Path,
) -> tuple[Path, Path | None]:
    inputs_dir = bridge_root / "inputs"
    publications_path = inputs_dir / "publications.parquet"
    save_parquet(publications, publications_path)

    references_path: Path | None = None
    if not references.empty:
        references_path = inputs_dir / "references.parquet"
        save_parquet(references, references_path)

    return publications_path, references_path


def _validate_and_normalize_output(
    staged_input: pd.DataFrame,
    output_frame: pd.DataFrame,
    *,
    frame_name: str,
) -> pd.DataFrame:
    if len(output_frame) != len(staged_input):
        raise ValueError(
            f"{frame_name} output row count must match staged input "
            f"(expected {len(staged_input)}, got {len(output_frame)})."
        )

    staged_bibcodes = staged_input.get("Bibcode", pd.Series(dtype=object)).astype(str).tolist()
    output_bibcodes = output_frame.get("Bibcode", pd.Series(dtype=object)).astype(str).tolist()
    if output_bibcodes != staged_bibcodes:
        raise ValueError(f"{frame_name} output Bibcode order must match staged input.")

    missing = [column for column in _LIST_OUTPUT_COLUMNS if column not in output_frame.columns]
    if missing:
        raise ValueError(f"{frame_name} output is missing required columns: {', '.join(missing)}")

    frame_out = output_frame.copy()
    expected_lengths = staged_input["Author"].apply(lambda value: len(_author_list(value))).tolist()
    for source_column, target_column in _LIST_OUTPUT_COLUMNS.items():
        normalized_rows: list[list[str]] = []
        for value, expected_len in zip(frame_out[source_column].tolist(), expected_lengths, strict=False):
            normalized_rows.append(
                _normalize_output_list(value, column=source_column, expected_len=expected_len)
            )
        frame_out[target_column] = normalized_rows
        frame_out = frame_out.drop(columns=[source_column])
    return frame_out


def _load_disambiguated_outputs(
    *,
    publications_input: pd.DataFrame,
    references_input: pd.DataFrame,
    publications_output_path: Path | str,
    references_output_path: Path | str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    publications_output = load_parquet(publications_output_path)
    pubs_out = _validate_and_normalize_output(
        publications_input,
        publications_output,
        frame_name="publications",
    )

    if references_input.empty:
        refs_out = references_input.copy()
        if "Author" in refs_out.columns:
            refs_out["author_uids"] = [[] for _ in range(len(refs_out))]
            refs_out["author_display_names"] = [[] for _ in range(len(refs_out))]
        return pubs_out, refs_out

    if references_output_path is None:
        raise ValueError("references output path is required when references were provided.")

    references_output = load_parquet(references_output_path)
    refs_out = _validate_and_normalize_output(
        references_input,
        references_output,
        frame_name="references",
    )
    return pubs_out, refs_out


def _copy_optional_source_assignments(result: Any, *, run_data_dir: Path | str | None) -> None:
    if run_data_dir is None:
        return

    source_assignments_path = _result_value(result, "source_author_assignments_path")
    if source_assignments_path is None:
        return

    source_path = Path(str(source_assignments_path))
    if not source_path.exists():
        return

    target_dir = Path(run_data_dir) / "and"
    target_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(load_parquet(source_path), target_dir / source_path.name)


def apply_author_disambiguation(
    publications: pd.DataFrame,
    references: pd.DataFrame | None = None,
    *,
    model_bundle: str | Path,
    cache_dir: Path | str,
    dataset_id: str | None = None,
    force_refresh: bool = False,
    run_data_dir: Path | str | None = None,
    infer_stage: str = "full",
    and_runner: SourceAndRunner | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run an external source-based AND package and load disambiguated ADS frames.

    Required source columns:
    - ``Bibcode``
    - ``Author``
    - ``Year``
    - ``Title_en`` or ``Title``
    - ``Abstract_en`` or ``Abstract``

    Optional source columns:
    - ``Affiliation``
    """
    references = pd.DataFrame() if references is None else references.copy()
    _validate_source_frame(publications, frame_name="publications")
    _validate_source_frame(references, frame_name="references")

    if not force_refresh:
        try:
            pubs_cached, refs_cached = load_phase4_checkpoint(
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
            )

    if publications.empty and references.empty:
        pubs_out = _prepare_passthrough_frame(publications)
        refs_out = _prepare_passthrough_frame(references)
        save_phase4_checkpoint(
            pubs_out,
            refs_out,
            cache_dir=cache_dir,
            run_data_dir=run_data_dir,
        )
        logger.info("Author disambiguation skipped: no source rows available.")
        return pubs_out, refs_out

    runner = and_runner or _load_default_and_runner()
    resolved_dataset_id = _resolve_dataset_id(dataset_id, run_data_dir)
    bridge_root = Path(cache_dir) / "and_bridge" / resolved_dataset_id
    publications_path, references_path = _stage_source_frames(
        publications,
        references,
        bridge_root=bridge_root,
    )

    result = runner(
        publications_path=publications_path,
        references_path=references_path,
        output_root=bridge_root / "output",
        dataset_id=resolved_dataset_id,
        model_bundle=model_bundle,
        force=force_refresh,
        infer_stage=infer_stage,
    )
    publications_output_path = _result_value(result, "publications_disambiguated_path")
    if publications_output_path is None:
        raise ValueError("AND runner result is missing publications_disambiguated_path.")

    pubs_out, refs_out = _load_disambiguated_outputs(
        publications_input=publications,
        references_input=references,
        publications_output_path=publications_output_path,
        references_output_path=_result_value(result, "references_disambiguated_path"),
    )

    save_phase4_checkpoint(
        pubs_out,
        refs_out,
        cache_dir=cache_dir,
        run_data_dir=run_data_dir,
    )
    _copy_optional_source_assignments(result, run_data_dir=run_data_dir)

    logger.info(
        "Author disambiguation complete | publications=%s | references=%s",
        f"{len(pubs_out):,}",
        f"{len(refs_out):,}",
    )
    return pubs_out, refs_out


__all__ = ["apply_author_disambiguation"]
