"""Source-based author disambiguation adapter for external AND packages."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import logging
from pathlib import Path
import shutil
from typing import Any
import warnings

import pandas as pd

from ads_bib._utils.authors import author_list as _author_list
from ads_bib._utils.checkpoints import (
    load_disambiguated_snapshot,
    save_disambiguated_snapshot,
)
from ads_bib._utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)

_REQUIRED_SOURCE_COLUMNS = {"Bibcode", "Author", "Year"}
_AUTHOR_UID_COLUMN = "AuthorUID"
_AUTHOR_DISPLAY_NAME_COLUMN = "AuthorDisplayName"
_AND_CACHE_METADATA_FILE = "author_disambiguation_cache.json"
_LIST_OUTPUT_COLUMNS = {
    _AUTHOR_UID_COLUMN: "author_uids",
    _AUTHOR_DISPLAY_NAME_COLUMN: "author_display_names",
}

SourceAndRunner = Callable[..., Any]


def _load_default_and_runner() -> SourceAndRunner:
    try:
        from author_name_disambiguation import disambiguate_sources, run_infer_sources
    except ImportError as exc:
        raise ImportError(
            "Author disambiguation requires `ads-and>=0.1.3` "
            "with `author_name_disambiguation.disambiguate_sources`. "
            "Update ads-bib with `pip install -U ads-bib` or `uv pip install -U ads-bib`."
        ) from exc

    def _runner(**kwargs: Any) -> Any:
        model_bundle = kwargs.pop("model_bundle", None)
        if model_bundle is None:
            return disambiguate_sources(**kwargs)

        output_dir = kwargs.pop("output_dir")
        runtime = str(kwargs.pop("runtime", "auto") or "auto").strip().lower()
        runtime_mode = None if runtime == "auto" else runtime
        return run_infer_sources(
            output_root=output_dir,
            model_bundle=model_bundle,
            runtime_mode=runtime_mode,
            **kwargs,
        )

    return _runner


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


_OPTIONAL_AND_ARTIFACT_PATH_KEYS = (
    "source_author_assignments_path",
    "author_entities_path",
    "mention_clusters_path",
    "summary_path",
    "stage_metrics_path",
    "go_no_go_path",
)


def _jsonable_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _source_frame_fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in (
            "Bibcode",
            "Author",
            "Year",
            "Title_en",
            "Title",
            "Abstract_en",
            "Abstract",
            "Affiliation",
        )
        if column in frame.columns
    ]
    if not columns:
        return hashlib.sha256(str(len(frame)).encode("utf-8")).hexdigest()
    comparable = frame.loc[:, columns].copy()
    for column in comparable.columns:
        comparable[column] = comparable[column].map(_jsonable_value)
    payload = comparable.to_json(orient="records", lines=True, force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_metadata_path(cache_dir: Path | str) -> Path:
    return Path(cache_dir) / _AND_CACHE_METADATA_FILE


def _build_cache_metadata(
    *,
    publications: pd.DataFrame,
    references: pd.DataFrame,
    backend: str,
    runtime: str,
    modal_gpu: str | None,
    model_bundle: str | Path | None,
    dataset_id: str,
    infer_stage: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "ads-and",
        "backend": backend,
        "runtime": runtime,
        "modal_gpu": modal_gpu,
        "model_bundle": None if model_bundle is None else str(model_bundle),
        "dataset_id": dataset_id,
        "infer_stage": infer_stage,
        "publications_fingerprint": _source_frame_fingerprint(publications),
        "references_fingerprint": _source_frame_fingerprint(references),
    }


def _load_matching_cache_metadata(cache_dir: Path | str, expected: dict[str, Any]) -> bool:
    metadata_path = _cache_metadata_path(cache_dir)
    if not metadata_path.exists():
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _save_cache_metadata(cache_dir: Path | str, metadata: dict[str, Any]) -> None:
    metadata_path = _cache_metadata_path(cache_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _copy_optional_and_artifacts(result: Any, *, run_data_dir: Path | str | None) -> None:
    if run_data_dir is None:
        return

    target_dir = Path(run_data_dir) / "and"
    copied_any = False
    for key in _OPTIONAL_AND_ARTIFACT_PATH_KEYS:
        raw_path = _result_value(result, key)
        if raw_path is None:
            continue
        source_path = Path(str(raw_path))
        if not source_path.exists() or not source_path.is_file():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_dir / source_path.name)
        copied_any = True
    if copied_any:
        logger.info("Copied author disambiguation diagnostics to %s", target_dir)


def _run_and_runner(runner: SourceAndRunner, **kwargs: Any) -> Any:
    try:
        from torch.jit import TracerWarning
    except Exception:
        TracerWarning = Warning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"torch\.tensor results are registered as constants in the trace\.",
            category=TracerWarning,
            module=r"transformers\.modeling_attn_mask_utils",
        )
        return runner(**kwargs)


def apply_author_disambiguation(
    publications: pd.DataFrame,
    references: pd.DataFrame | None = None,
    *,
    cache_dir: Path | str,
    backend: str = "local",
    runtime: str = "auto",
    modal_gpu: str | None = "l4",
    model_bundle: str | Path | None = None,
    dataset_id: str | None = None,
    force_refresh: bool = False,
    run_data_dir: Path | str | None = None,
    infer_stage: str = "full",
    progress: bool = True,
    progress_handler: Callable[[Any], None] | None = None,
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
    resolved_dataset_id = _resolve_dataset_id(dataset_id, run_data_dir)
    expected_cache_metadata = _build_cache_metadata(
        publications=publications,
        references=references,
        backend=backend,
        runtime=runtime,
        modal_gpu=modal_gpu,
        model_bundle=model_bundle,
        dataset_id=resolved_dataset_id,
        infer_stage=infer_stage,
    )

    if not force_refresh and _load_matching_cache_metadata(cache_dir, expected_cache_metadata):
        try:
            pubs_cached, refs_cached = load_disambiguated_snapshot(
                cache_dir=cache_dir,
            )
        except FileNotFoundError:
            pass
        else:
            logger.info("Loaded cached author disambiguation outputs.")
            return (
                _normalize_list_columns(pubs_cached, ["author_uids", "author_display_names"]),
                _normalize_list_columns(refs_cached, ["author_uids", "author_display_names"]),
            )
    elif not force_refresh and _cache_metadata_path(cache_dir).exists():
        logger.info("Ignoring cached author disambiguation outputs because cache metadata differs.")
    elif not force_refresh:
        logger.info("Ignoring cached author disambiguation outputs without ads-and metadata.")

    if publications.empty and references.empty:
        pubs_out = _prepare_passthrough_frame(publications)
        refs_out = _prepare_passthrough_frame(references)
        save_disambiguated_snapshot(
            pubs_out,
            refs_out,
            cache_dir=cache_dir,
        )
        logger.info("Author disambiguation skipped: no source rows available.")
        return pubs_out, refs_out

    runner = and_runner or _load_default_and_runner()
    bridge_root = Path(cache_dir) / "and_bridge" / resolved_dataset_id
    publications_path, references_path = _stage_source_frames(
        publications,
        references,
        bridge_root=bridge_root,
    )

    result = _run_and_runner(
        runner,
        publications_path=publications_path,
        references_path=references_path,
        output_dir=bridge_root / "output",
        dataset_id=resolved_dataset_id,
        backend=backend,
        runtime=runtime,
        modal_gpu=modal_gpu,
        model_bundle=model_bundle,
        force=force_refresh,
        infer_stage=infer_stage,
        progress=progress,
        progress_handler=progress_handler,
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

    save_disambiguated_snapshot(
        pubs_out,
        refs_out,
        cache_dir=cache_dir,
    )
    _save_cache_metadata(cache_dir, expected_cache_metadata)
    _copy_optional_and_artifacts(result, run_data_dir=run_data_dir)

    logger.info(
        "Author disambiguation complete | publications=%s | references=%s",
        f"{len(pubs_out):,}",
        f"{len(refs_out):,}",
    )
    return pubs_out, refs_out


__all__ = ["apply_author_disambiguation"]
