"""Thin CLI for quality checks and config-driven pipeline runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable, Sequence

from ads_bib.bootstrap import bootstrap_workspace
from ads_bib.doctor import collect_doctor_report, format_doctor_report
from ads_bib.pipeline import PipelineConfig
from ads_bib.presets import (
    get_preset_names,
    get_preset_summary,
    write_preset,
)
from ads_bib.runner import (
    RunBlockedError,
    load_run_config,
    parse_override as _parse_override,
    run_resolved_config,
)

CommandRunner = Callable[[Sequence[str], dict[str, str] | None], int]


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


def _load_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    overrides = dict(_parse_override(raw) for raw in getattr(args, "set_values", []) or [])
    return load_run_config(
        preset=getattr(args, "preset", None),
        config=getattr(args, "config", None),
        overrides=overrides,
    )


def _handle_run(args: argparse.Namespace) -> int:
    config = _load_config_from_args(args)
    try:
        run_resolved_config(
            config,
            start_stage=args.from_stage,
            stop_stage=args.to_stage,
            run_name=args.run_name,
            notify=lambda message: sys.stdout.write(f"{message}\n"),
        )
    except RunBlockedError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
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
