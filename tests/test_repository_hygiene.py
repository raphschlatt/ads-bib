from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_SKELETON = {"runs/.gitkeep", "runs/README.md"}
SECRET_VALUE_PATTERN = (
    r"sk-or-v1-[A-Za-z0-9_-]+|"
    r"sk-ant-[A-Za-z0-9_-]+|"
    r"sk-[A-Za-z0-9_-]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]+|"
    r"gh[opsur]_[A-Za-z0-9_]{20,}|"
    r"AIza[0-9A-Za-z_-]{35}"
)


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("git is not available")
    if check and result.returncode != 0:
        pytest.skip("repository metadata is not available")
    return result


def test_git_tracks_only_runs_skeleton() -> None:
    tracked = set(_git(["ls-files", "runs"]).stdout.splitlines())

    assert tracked <= RUNS_SKELETON


def test_head_does_not_contain_secret_like_values() -> None:
    result = _git(
        ["grep", "-n", "-I", "-E", SECRET_VALUE_PATTERN, "HEAD"],
        check=False,
    )

    if result.returncode == 1:
        return
    if result.returncode != 0:
        pytest.skip("git grep could not inspect HEAD")

    locations = []
    for line in result.stdout.splitlines():
        commit_path_line = line.split(":", 3)[:3]
        locations.append(":".join(commit_path_line))

    assert not locations, "Secret-like values found in tracked files: " + ", ".join(locations)
