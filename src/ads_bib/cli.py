"""Thin CLI for quality checks and config-driven pipeline runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence

from ads_bib.pipeline import PipelineConfig, run_pipeline
from ads_bib.presets import (
    get_preset_names,
    get_preset_summary,
    load_preset_config,
    write_preset,
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


def _handle_run(args: argparse.Namespace) -> int:
    if args.config is not None:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = load_preset_config(args.preset)
    config_data = config.to_dict()

    for raw in args.set_values or []:
        key, value = _parse_override(raw)
        _apply_override(config_data, key, value)

    config = PipelineConfig.from_dict(config_data)
    if not str(config.search.query).strip():
        raise ValueError(
            "search.query is required. Set it in your YAML config or pass "
            "--set search.query='...' when using a preset."
        )
    run_pipeline(
        config,
        start_stage=args.from_stage,
        stop_stage=args.to_stage,
        run_name=args.run_name,
    )
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
        description="CLI for quality checks, preset management, and ADS pipeline runs.",
    )
    subparsers = parser.add_subparsers(required=True)

    check_parser = subparsers.add_parser(
        "check",
        help="Run local quality gates (ruff + pytest).",
    )
    check_parser.set_defaults(handler=_handle_check)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the ADS pipeline from a packaged preset or YAML config file.",
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
