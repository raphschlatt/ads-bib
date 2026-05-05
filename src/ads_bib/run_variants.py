"""Planning helpers for creating run variants from completed runs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ads_bib._stage_state import (
    STAGE_ORDER,
    StageName,
    _earliest_invalidation_stage,
    validate_stage_name,
)
from ads_bib._utils.io import load_parquet
from ads_bib.pipeline import PipelineConfig, PipelineInitialState
from ads_bib.run_manager import ARTIFACT_LAYOUT_VERSION, RunArtifactLayout, _is_secret_key

_CONFIG_USED = "config_used.yaml"
_SUMMARY = "run_summary.yaml"
_LAYOUT_MIGRATION_HINT = (
    "Base run is not in the v0.2 artifact layout. Use a v0.2 run as the variant base."
)


@dataclass(frozen=True)
class RunVariantPlan:
    base_run_id: str
    base_run_path: Path
    config: PipelineConfig
    changed_keys: tuple[str, ...]
    requested_start_stage: StageName | None
    effective_start_stage: StageName
    stop_stage: StageName | None
    run_name: str
    reused_until: StageName | None
    initial_state: PipelineInitialState | None
    variant: dict[str, Any]


def plan_run_variant(
    *,
    from_run: str | Path,
    overrides: Mapping[str, Any] | None = None,
    start_stage: StageName | str | None = None,
    stop_stage: StageName | str | None = None,
    run_name: str | None = None,
    project_root: Path | str | None = None,
) -> RunVariantPlan:
    """Return an executable variant plan for a completed base run."""
    base_run_path = resolve_base_run_path(from_run, project_root=project_root)
    _ensure_base_run_layout(base_run_path)
    base_run_id = _read_base_run_id(base_run_path)
    base_config = load_base_run_config(base_run_path)
    base_data = base_config.to_dict()
    config_data = deepcopy(base_data)

    normalized_overrides = {str(key): value for key, value in (overrides or {}).items()}
    for key, value in normalized_overrides.items():
        _apply_override(config_data, key, value)

    requested_start = validate_stage_name(start_stage) if start_stage is not None else None
    requested_stop = validate_stage_name(stop_stage) if stop_stage is not None else None
    effective_start = requested_start or _automatic_start_stage(base_data, config_data)
    if requested_start is None and effective_start == "curate":
        effective_start = "topic_fit"
    if requested_stop is not None and STAGE_ORDER.index(requested_stop) < STAGE_ORDER.index(effective_start):
        raise ValueError("stop_stage must be after or equal to the effective start stage.")

    changed_keys = tuple(sorted(normalized_overrides)) or tuple(
        sorted(_changed_leaf_paths(base_data, config_data))
    )
    _anchor_author_disambiguation_cache(config_data, base_run_id=base_run_id)
    config = PipelineConfig.from_dict(config_data)

    variant_run_name = run_name or f"{config.run.run_name}_variant"
    reused_until = _previous_stage(effective_start)
    initial_state = hydrate_initial_state(
        base_run_path=base_run_path,
        start_stage=effective_start,
    )
    variant = {
        "base_run_id": base_run_id,
        "base_run_path": str(base_run_path),
        "changed_keys": list(changed_keys),
        "recomputed_from": effective_start,
        "reused_until": reused_until,
    }

    return RunVariantPlan(
        base_run_id=base_run_id,
        base_run_path=base_run_path,
        config=config,
        changed_keys=changed_keys,
        requested_start_stage=requested_start,
        effective_start_stage=effective_start,
        stop_stage=requested_stop,
        run_name=variant_run_name,
        reused_until=reused_until,
        initial_state=initial_state,
        variant=variant,
    )


def resolve_base_run_path(
    value: str | Path,
    *,
    project_root: Path | str | None = None,
) -> Path:
    """Resolve a run directory path or run id below ``runs/``."""
    raw = Path(value).expanduser()
    candidates: list[Path] = []
    if raw.exists():
        candidates.append(raw)

    root = Path(project_root or Path.cwd())
    runs_root = root / "runs"
    direct = runs_root / str(value)
    if direct.exists():
        candidates.append(direct)

    unique = _unique_existing_run_dirs(candidates)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        joined = "\n".join(str(path) for path in unique)
        raise ValueError(f"Run id '{value}' is ambiguous below {runs_root}:\n{joined}")
    raise FileNotFoundError(
        f"Could not find run '{value}'. Pass a run directory path or a run id below {runs_root}."
    )


def load_base_run_config(base_run_path: Path | str) -> PipelineConfig:
    config_path = Path(base_run_path) / _CONFIG_USED
    if not config_path.exists():
        raise FileNotFoundError(f"Base run config not found at {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return PipelineConfig.from_dict(_normalize_redacted_secrets(payload))


def hydrate_initial_state(
    *,
    base_run_path: Path | str,
    start_stage: StageName | str,
) -> PipelineInitialState | None:
    """Load practical downstream inputs from a base run when they are sufficient."""
    stage = validate_stage_name(start_stage)
    layout = RunArtifactLayout.from_run_dir(base_run_path)
    if stage == "citations":
        curated_df = _load_optional_parquet(layout.dataset / "publications.parquet")
        refs = _load_optional_parquet(layout.dataset / "references.parquet")
        if curated_df is None or refs is None:
            return None
        return PipelineInitialState(
            publications=curated_df,
            refs=refs,
            curated_df=curated_df,
            author_entities=_load_optional_parquet(layout.and_dir / "author_entities.parquet"),
        )
    if stage in {"visualize", "curate"}:
        topic_df = _load_optional_parquet(layout.dataset / "publications.parquet")
        if topic_df is None:
            return None
        return PipelineInitialState(
            publications=topic_df,
            refs=_load_optional_parquet(layout.dataset / "references.parquet"),
            topic_df=topic_df,
            topic_info=_load_optional_parquet(layout.dataset / "topic_info.parquet"),
        )
    return None


def format_variant_plan(plan: RunVariantPlan) -> str:
    changed = ", ".join(plan.changed_keys) if plan.changed_keys else "(none)"
    start_index = STAGE_ORDER.index(plan.effective_start_stage)
    stop_index = (
        STAGE_ORDER.index(plan.stop_stage)
        if plan.stop_stage is not None
        else len(STAGE_ORDER) - 1
    )
    reused_stages = ", ".join(STAGE_ORDER[:start_index]) or "(none)"
    recomputed_stages = ", ".join(STAGE_ORDER[start_index : stop_index + 1]) or "(none)"
    return "\n".join(
        [
            "Run variant dry run",
            f"Base run: {plan.base_run_id}",
            f"Base path: {plan.base_run_path}",
            f"Changed keys: {changed}",
            f"Reused stages: {reused_stages}",
            f"Recomputed stages: {recomputed_stages}",
            f"Reused until: {plan.reused_until or '(none)'}",
            f"Recomputed from: {plan.effective_start_stage}",
            f"Effective start stage: {plan.effective_start_stage}",
            f"Requested start: {plan.requested_start_stage or '(auto)'}",
            f"Requested stop: {plan.stop_stage or '(pipeline default)'}",
            f"Target run name: {plan.run_name}",
        ]
    )


def _automatic_start_stage(
    base_data: Mapping[str, Any],
    config_data: Mapping[str, Any],
) -> StageName:
    return _earliest_invalidation_stage(base_data, config_data) or validate_stage_name(
        _nested_get(config_data, ("run", "start_stage")) or "search"
    )


def _anchor_author_disambiguation_cache(config_data: dict[str, Any], *, base_run_id: str) -> None:
    and_config = config_data.setdefault("author_disambiguation", {})
    if not isinstance(and_config, dict):
        return
    if and_config.get("enabled") and not and_config.get("dataset_id"):
        and_config["dataset_id"] = base_run_id


def _apply_override(data: dict[str, Any], key: str, value: Any) -> None:
    current: dict[str, Any] = data
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("Override key cannot be empty.")
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _normalize_redacted_secrets(value: Any, *, key_name: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_redacted_secrets(item, key_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_redacted_secrets(item, key_name=key_name) for item in value]
    if value == "<redacted>" and key_name is not None and _is_secret_key(key_name):
        return None
    return value


def _changed_leaf_paths(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> set[str]:
    keys = set(previous) | set(current)
    changed: set[str] = set()
    for key in keys:
        prev = previous.get(key)
        cur = current.get(key)
        path = (*prefix, str(key))
        if isinstance(prev, Mapping) and isinstance(cur, Mapping):
            changed.update(_changed_leaf_paths(prev, cur, prefix=path))
        elif prev != cur:
            changed.add(".".join(path))
    return changed


def _previous_stage(stage: StageName) -> StageName | None:
    index = STAGE_ORDER.index(stage)
    if index == 0:
        return None
    return STAGE_ORDER[index - 1]


def _nested_get(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _read_base_run_id(base_run_path: Path) -> str:
    summary_path = base_run_path / _SUMMARY
    if summary_path.exists():
        try:
            payload = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
            run_id = payload.get("run", {}).get("run_id")
            if run_id:
                return str(run_id)
        except Exception:
            pass
    return base_run_path.name


def _ensure_base_run_layout(base_run_path: Path) -> None:
    summary_path = base_run_path / _SUMMARY
    if not summary_path.exists():
        raise ValueError(
            f"Base run summary not found at {summary_path}. "
            "Variants require a completed v0.2 run."
        )
    try:
        payload = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Base run summary is not readable at {summary_path}: {exc}") from exc
    if payload.get("artifact_layout_version") == ARTIFACT_LAYOUT_VERSION:
        return
    raise ValueError(_LAYOUT_MIGRATION_HINT)


def _load_optional_parquet(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_parquet(path)


def _unique_existing_run_dirs(candidates: list[Path]) -> list[Path]:
    unique: dict[Path, Path] = {}
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_dir() and (path / _CONFIG_USED).exists():
            unique[path] = path
    return list(unique)


__all__ = [
    "RunVariantPlan",
    "format_variant_plan",
    "hydrate_initial_state",
    "load_base_run_config",
    "plan_run_variant",
    "resolve_base_run_path",
]
