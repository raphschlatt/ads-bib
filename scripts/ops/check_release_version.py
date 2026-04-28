"""Check release metadata consistency for a tag."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def fail(message: str) -> None:
    print(f"release metadata check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"missing {path.relative_to(ROOT)}")


def normalize_tag(raw_tag: str) -> str:
    tag = raw_tag.strip()
    if not tag:
        fail("release tag is required")
    if tag.startswith("refs/tags/"):
        tag = tag.removeprefix("refs/tags/")
    if not tag.startswith("v"):
        fail(f"tag must start with 'v': {raw_tag}")
    version = tag[1:]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        fail(f"tag must look like vX.Y.Z: {raw_tag}")
    return version


def pyproject_version() -> str:
    data = tomllib.loads(read_text(ROOT / "pyproject.toml"))
    try:
        return str(data["project"]["version"])
    except KeyError:
        fail("pyproject.toml missing [project].version")


def changelog_release(version: str) -> tuple[str, str]:
    changelog = read_text(ROOT / "CHANGELOG.md")
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\] - (?P<date>\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        re.MULTILINE,
    )
    match = pattern.search(changelog)
    if not match:
        fail(f"CHANGELOG.md missing release section for {version}")

    body_start = match.end()
    next_match = re.search(r"^## \[", changelog[body_start:], re.MULTILINE)
    body_end = body_start + next_match.start() if next_match else len(changelog)
    body = changelog[body_start:body_end].strip()
    if not body:
        fail(f"CHANGELOG.md release section for {version} is empty")
    return match.group("date"), body


def cff_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in read_text(ROOT / "CITATION.cff").splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


def zenodo_metadata() -> dict[str, object]:
    try:
        data = json.loads(read_text(ROOT / ".zenodo.json"))
    except json.JSONDecodeError as exc:
        fail(f".zenodo.json is invalid JSON: {exc}")
    if not isinstance(data, dict):
        fail(".zenodo.json must contain a JSON object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tag",
        nargs="?",
        default=os.environ.get("GITHUB_REF_NAME", ""),
        help="Release tag such as v0.1.0; defaults to GITHUB_REF_NAME.",
    )
    parser.add_argument(
        "--notes-out",
        type=Path,
        help="Write the matching CHANGELOG release body to this file.",
    )
    args = parser.parse_args()

    version = normalize_tag(args.tag)
    project_version = pyproject_version()
    if project_version != version:
        fail(f"pyproject.toml version {project_version!r} does not match tag v{version}")

    release_date, release_notes = changelog_release(version)

    cff = cff_fields()
    if cff.get("version") != version:
        fail(f"CITATION.cff version {cff.get('version')!r} does not match {version!r}")
    if cff.get("date-released") != release_date:
        fail(
            "CITATION.cff date-released "
            f"{cff.get('date-released')!r} does not match CHANGELOG.md {release_date!r}"
        )

    zenodo = zenodo_metadata()
    if zenodo.get("version") != version:
        fail(f".zenodo.json version {zenodo.get('version')!r} does not match {version!r}")
    if zenodo.get("publication_date") != release_date:
        fail(
            ".zenodo.json publication_date "
            f"{zenodo.get('publication_date')!r} does not match CHANGELOG.md {release_date!r}"
        )
    for key in ("title", "description", "creators", "upload_type", "access_right"):
        if not zenodo.get(key):
            fail(f".zenodo.json missing {key!r}")

    if args.notes_out:
        args.notes_out.parent.mkdir(parents=True, exist_ok=True)
        args.notes_out.write_text(release_notes + "\n", encoding="utf-8")

    print(f"release metadata ok: v{version} ({release_date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
