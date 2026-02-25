"""Minimal CLI for repository-local quality checks."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Callable, Sequence

CommandRunner = Callable[[Sequence[str], dict[str, str] | None], int]


def _run_command(command: Sequence[str], env: dict[str, str] | None = None) -> int:
    result = subprocess.run(list(command), check=False, env=env)
    return result.returncode


def run_quality_checks(*, run_command: CommandRunner | None = None) -> int:
    """Run the standard local quality gates."""
    runner = run_command or _run_command
    checks: list[tuple[Sequence[str], dict[str, str] | None]] = [
        (["ruff", "check", "src", "tests", "scripts"], None),
        (["pytest", "-q"], {"PYTHONPATH": "src"}),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ads-bib",
        description="Notebook-first ADS pipeline helper commands.",
    )
    subparsers = parser.add_subparsers(required=True)

    check_parser = subparsers.add_parser(
        "check",
        help="Run local quality gates (ruff + pytest).",
    )
    check_parser.set_defaults(handler=_handle_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
