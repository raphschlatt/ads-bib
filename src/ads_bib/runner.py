"""High-level Python runner matching the public CLI run path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ads_bib._utils.llama_server import prepare_llama_server_runtime
from ads_bib.bootstrap import ensure_default_fasttext_model
from ads_bib.doctor import DoctorReport, collect_doctor_report
from ads_bib.pipeline import PipelineConfig, PipelineContext, StageName, run_pipeline
from ads_bib.presets import get_preset_names, load_preset_config

Notify = Callable[[str], None]

_TOPIC_STAGES = frozenset(
    {"embeddings", "reduction", "topic_fit", "topic_dataframe", "visualize", "curate"}
)


class RunBlockedError(RuntimeError):
    """Raised when a high-level run is blocked before pipeline execution."""

    def __init__(self, message: str, *, report: DoctorReport | None = None) -> None:
        super().__init__(message.rstrip())
        self.report = report


def parse_override(raw: str) -> tuple[str, object]:
    """Parse one CLI-style ``key=value`` override."""
    if "=" not in raw:
        raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
    key, value = raw.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def apply_override(data: dict[str, Any], key: str, value: object) -> None:
    """Apply one dotted-key override to a pipeline config dict."""
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


def load_run_config(
    *,
    preset: str | None = None,
    config: PipelineConfig | Mapping[str, Any] | Path | str | None = None,
    query: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    project_root: Path | str | None = None,
) -> PipelineConfig:
    """Resolve a preset/YAML/object source plus high-level overrides."""
    if (preset is None) == (config is None):
        raise ValueError("Exactly one of preset or config is required.")
    if query is not None and _overrides_search_query(overrides):
        raise ValueError("Pass search.query either via query or overrides, not both.")

    resolved = load_preset_config(preset) if preset is not None else _load_config(config)
    config_data = resolved.to_dict()

    if project_root is not None:
        config_data.setdefault("run", {})["project_root"] = str(project_root)

    for key, value in (overrides or {}).items():
        apply_override(config_data, str(key), value)
    if query is not None:
        apply_override(config_data, "search.query", query)

    return PipelineConfig.from_dict(config_data)


def run(
    *,
    preset: str | None = None,
    config: PipelineConfig | Mapping[str, Any] | Path | str | None = None,
    query: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    start_stage: StageName | None = None,
    stop_stage: StageName | None = None,
    run_name: str | None = None,
    project_root: Path | str | None = None,
    preflight: bool = True,
) -> PipelineContext:
    """Run the ADS pipeline from a packaged preset or config source."""
    resolved_config = load_run_config(
        preset=preset,
        config=config,
        query=query,
        overrides=overrides,
        project_root=project_root,
    )
    return run_resolved_config(
        resolved_config,
        start_stage=start_stage,
        stop_stage=stop_stage,
        run_name=run_name,
        project_root=project_root,
        preflight=preflight,
    )


def run_resolved_config(
    config: PipelineConfig,
    *,
    start_stage: StageName | None = None,
    stop_stage: StageName | None = None,
    run_name: str | None = None,
    project_root: Path | str | None = None,
    preflight: bool = True,
    notify: Notify | None = None,
) -> PipelineContext:
    """Run an already resolved config through the shared high-level path."""
    if preflight:
        _prepare_run(config, start_stage=start_stage, stop_stage=stop_stage, notify=notify)
    return run_pipeline(
        config,
        start_stage=start_stage,
        stop_stage=stop_stage,
        project_root=project_root,
        run_name=run_name,
    )


def format_run_preflight_report(report: DoctorReport) -> str:
    """Format the compact run-blocking preflight report used by the CLI and API."""
    start_stage = report.active_stages[0]
    end_stage = report.active_stages[-1]
    lines = [f"Run blocked by preflight checks for stages {start_stage} -> {end_stage}"]
    for check in report.failing_checks():
        lines.append(f"[FAIL] {check.name}: {check.detail}")
    lines.append(
        "Summary: "
        f"{report.ok_count} ok, {report.warn_count} warn, {report.fail_count} fail"
    )
    lines.append("Tip: run 'ads-bib doctor ...' for the full stage-aware report.")
    return "\n".join(lines)


def _load_config(
    config: PipelineConfig | Mapping[str, Any] | Path | str | None,
) -> PipelineConfig:
    if isinstance(config, PipelineConfig):
        return PipelineConfig.from_dict(config.to_dict())
    if isinstance(config, Mapping):
        return PipelineConfig.from_dict(deepcopy(dict(config)))
    if isinstance(config, (Path, str)):
        try:
            return PipelineConfig.from_yaml(config)
        except FileNotFoundError as exc:
            _maybe_raise_legacy_config_hint(config, exc)
    raise TypeError("config must be a PipelineConfig, mapping, or YAML path.")


def _maybe_raise_legacy_config_hint(raw_path: Path | str, exc: FileNotFoundError) -> None:
    config_path = Path(raw_path)
    preset_names = set(get_preset_names())
    stem = config_path.stem
    legacy_parent = tuple(part.lower() for part in config_path.parent.parts[-2:])
    looks_like_legacy_preset = stem in preset_names and legacy_parent == ("configs", "pipeline")

    if looks_like_legacy_preset:
        raise FileNotFoundError(
            f"Legacy preset file '{raw_path}' no longer exists. "
            f"Use 'ads-bib run --preset {stem} ...' directly, or write it first via "
            f"'ads-bib preset write {stem} --output {stem}.yaml'."
        ) from exc

    raise exc


def _overrides_search_query(overrides: Mapping[str, Any] | None) -> bool:
    if not overrides:
        return False
    if "search.query" in overrides:
        return True
    search_override = overrides.get("search")
    return isinstance(search_override, Mapping) and "query" in search_override


def _prepare_run(
    config: PipelineConfig,
    *,
    start_stage: StageName | None,
    stop_stage: StageName | None,
    notify: Notify | None,
) -> None:
    if config.translate.enabled:
        try:
            downloaded_fasttext = ensure_default_fasttext_model(
                project_root=config.run.project_root,
                configured_path=config.translate.fasttext_model,
            )
        except Exception as exc:
            raise RunBlockedError(
                f"Run blocked while preparing translate.fasttext_model: {exc}"
            ) from exc
        if downloaded_fasttext is not None and notify is not None:
            notify(f"Prepared translate.fasttext_model at {downloaded_fasttext}")

    report = collect_doctor_report(config, start_stage=start_stage, stop_stage=stop_stage)
    if report.has_failures():
        raise RunBlockedError(format_run_preflight_report(report), report=report)

    if _run_requires_llama_server(config, report.active_stages):
        try:
            runtime = prepare_llama_server_runtime(
                config=config.llama_server,
                project_root=config.run.project_root,
            )
        except Exception as exc:
            raise RunBlockedError(
                f"Run blocked while preparing llama_server.command: {exc}"
            ) from exc
        if (
            runtime.source == "managed_downloaded"
            and runtime.command is not None
            and notify is not None
        ):
            notify(f"Prepared llama_server.command at {runtime.command}")


def _run_requires_llama_server(config: PipelineConfig, active_stages: tuple[str, ...]) -> bool:
    if (
        "translate" in active_stages
        and config.translate.enabled
        and config.translate.provider == "llama_server"
    ):
        return True
    return bool(_TOPIC_STAGES.intersection(active_stages)) and (
        config.topic_model.llm_provider == "llama_server"
    )


__all__ = [
    "RunBlockedError",
    "run",
]
