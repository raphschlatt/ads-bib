"""Thin CLI for quality checks and config-driven pipeline runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable, Sequence

from ads_bib._utils.llama_server import prepare_llama_server_runtime
from ads_bib.bootstrap import bootstrap_workspace, ensure_default_fasttext_model
from ads_bib.doctor import collect_doctor_report, format_doctor_report
from ads_bib.pipeline import PipelineConfig, run_pipeline
from ads_bib.presets import (
    get_preset_names,
    get_preset_summary,
    load_preset_config,
    write_preset,
)

CommandRunner = Callable[[Sequence[str], dict[str, str] | None], int]
_TOPIC_STAGES = frozenset({"embeddings", "reduction", "topic_fit", "topic_dataframe", "visualize", "curate"})


def _run_command(command: Sequence[str], env: dict[str, str] | None = None) -> int:
    result = subprocess.run(list(command), check=False, env=env)
    return result.returncode


def run_quality_checks(*, run_command: CommandRunner | None = None) -> int:
    """Run the standard local quality gates."""
    runner = run_command or _run_command
    checks: list[tuple[Sequence[str], dict[str, str] | None]] = [
        ([sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"], None),
        ([sys.executable, "-m", "pytest", "-q"], {"PYTHONPATH": "src"}),
    ]

    for command, extra_env in checks:
        env = None
        if extra_env is not None:
            env = os.environ.copy()
            env.update(extra_env)
        if runner(command, env) != 0:
            return 1
    return 0


def _handle_check(_args: argparse.Namespace) -> int:
    return run_quality_checks()


def _parse_override(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
    key, value = raw.split("=", 1)
    import yaml

    return key.strip(), yaml.safe_load(value)


def _apply_override(data: dict[str, object], key: str, value: object) -> None:
    current: dict[str, object] = data
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


def _maybe_raise_legacy_config_hint(raw_path: str, exc: FileNotFoundError) -> None:
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


def _load_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if getattr(args, "config", None) is not None:
        try:
            config = PipelineConfig.from_yaml(args.config)
        except FileNotFoundError as exc:
            _maybe_raise_legacy_config_hint(args.config, exc)
    else:
        config = load_preset_config(args.preset)
    config_data = config.to_dict()

    for raw in getattr(args, "set_values", []) or []:
        key, value = _parse_override(raw)
        _apply_override(config_data, key, value)

    return PipelineConfig.from_dict(config_data)


def _format_run_preflight_report(report) -> str:
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
    return "\n".join(lines) + "\n"


def _run_requires_llama_server(config: PipelineConfig, active_stages: tuple[str, ...]) -> bool:
    if "translate" in active_stages and config.translate.enabled and config.translate.provider == "llama_server":
        return True
    return bool(_TOPIC_STAGES.intersection(active_stages)) and config.topic_model.llm_provider == "llama_server"


def _handle_run(args: argparse.Namespace) -> int:
    config = _load_config_from_args(args)
    if config.translate.enabled:
        try:
            downloaded_fasttext = ensure_default_fasttext_model(
                project_root=config.run.project_root,
                configured_path=config.translate.fasttext_model,
            )
        except Exception as exc:
            sys.stderr.write(f"Run blocked while preparing translate.fasttext_model: {exc}\n")
            return 1
        if downloaded_fasttext is not None:
            sys.stdout.write(f"Prepared translate.fasttext_model at {downloaded_fasttext}\n")

    report = collect_doctor_report(
        config,
        start_stage=args.from_stage,
        stop_stage=args.to_stage,
    )
    if report.has_failures():
        sys.stderr.write(_format_run_preflight_report(report))
        return 1
    if _run_requires_llama_server(config, report.active_stages):
        try:
            runtime = prepare_llama_server_runtime(
                config=config.llama_server,
                project_root=config.run.project_root,
            )
        except Exception as exc:
            sys.stderr.write(f"Run blocked while preparing llama_server.command: {exc}\n")
            return 1
        if runtime.source == "managed_downloaded" and runtime.command is not None:
            sys.stdout.write(f"Prepared llama_server.command at {runtime.command}\n")
    run_pipeline(
        config,
        start_stage=args.from_stage,
        stop_stage=args.to_stage,
        run_name=args.run_name,
    )
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    config = _load_config_from_args(args)
    report = collect_doctor_report(
        config,
        start_stage=args.from_stage,
        stop_stage=args.to_stage,
    )
    sys.stdout.write(format_doctor_report(report))
    return 1 if report.has_failures() else 0


def _handle_bootstrap(args: argparse.Namespace) -> int:
    if bool(args.preset) != bool(args.config):
        raise ValueError("bootstrap requires --preset and --config together, or neither of them.")
    lines = bootstrap_workspace(
        project_root=Path(args.project_root),
        preset_name=args.preset,
        config_output=args.config,
        env_file=args.env_file,
        download_fasttext=args.download_fasttext,
        force=args.force,
    )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _handle_preset_list(_args: argparse.Namespace) -> int:
    lines = [f"{name:<12} {get_preset_summary(name)}" for name in get_preset_names()]
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _handle_preset_write(args: argparse.Namespace) -> int:
    path = write_preset(args.name, args.output, overwrite=args.force)
    sys.stdout.write(f"Wrote preset '{args.name}' to {path}\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ads-bib",
        description="CLI for ADS pipeline runs, optional preset scaffolding, and support checks.",
    )
    subparsers = parser.add_subparsers(required=True)

    check_parser = subparsers.add_parser(
        "check",
        help="Run local quality gates (ruff + pytest).",
    )
    check_parser.set_defaults(handler=_handle_check)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Optional convenience command to scaffold a working directory for packaged CLI runs.",
    )
    bootstrap_parser.add_argument(
        "--project-root",
        default=".",
        help="Working directory root that should contain data/ and runs/.",
    )
    bootstrap_parser.add_argument(
        "--preset",
        choices=get_preset_names(),
        help="Optional packaged preset to materialize as YAML.",
    )
    bootstrap_parser.add_argument(
        "--config",
        help="Destination YAML path for the preset when --preset is used.",
    )
    bootstrap_parser.add_argument(
        "--env-file",
        default=".env",
        help="Env template path relative to --project-root.",
    )
    bootstrap_parser.add_argument(
        "--download-fasttext",
        action="store_true",
        help="Download lid.176.bin into data/models/ under --project-root.",
    )
    bootstrap_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files when they already exist.",
    )
    bootstrap_parser.set_defaults(handler=_handle_bootstrap)

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "Run the ADS pipeline from a packaged preset or YAML config file; "
            "performs required preflight checks first."
        ),
    )
    run_source = run_parser.add_mutually_exclusive_group(required=True)
    run_source.add_argument("--config", help="Path to YAML pipeline config.")
    run_source.add_argument(
        "--preset",
        choices=get_preset_names(),
        help="Official packaged preset name.",
    )
    run_parser.add_argument("--from", dest="from_stage", help="Optional stage to start from.")
    run_parser.add_argument("--to", dest="to_stage", help="Optional stage to stop after.")
    run_parser.add_argument("--run-name", help="Optional run name override.")
    run_parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values via dotted key=value pairs.",
    )
    run_parser.set_defaults(handler=_handle_run)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Optional support command to inspect stage-aware runtime prerequisites.",
    )
    doctor_source = doctor_parser.add_mutually_exclusive_group(required=True)
    doctor_source.add_argument("--config", help="Path to YAML pipeline config.")
    doctor_source.add_argument(
        "--preset",
        choices=get_preset_names(),
        help="Official packaged preset name.",
    )
    doctor_parser.add_argument("--from", dest="from_stage", help="Optional stage to start from.")
    doctor_parser.add_argument("--to", dest="to_stage", help="Optional stage to stop after.")
    doctor_parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values via dotted key=value pairs.",
    )
    doctor_parser.set_defaults(handler=_handle_doctor)

    preset_parser = subparsers.add_parser(
        "preset",
        help="List or write the official packaged starter presets.",
    )
    preset_subparsers = preset_parser.add_subparsers(required=True)

    preset_list_parser = preset_subparsers.add_parser(
        "list",
        help="List the official preset names.",
    )
    preset_list_parser.set_defaults(handler=_handle_preset_list)

    preset_write_parser = preset_subparsers.add_parser(
        "write",
        help="Write a packaged preset to a YAML file you can edit.",
    )
    preset_write_parser.add_argument("name", choices=get_preset_names(), help="Preset name.")
    preset_write_parser.add_argument("--output", required=True, help="Destination YAML path.")
    preset_write_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    preset_write_parser.set_defaults(handler=_handle_preset_write)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
