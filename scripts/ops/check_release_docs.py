"""Check small, release-critical documentation invariants."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def fail(message: str) -> None:
    print(f"release docs check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"missing {path.relative_to(ROOT)}")


def main() -> int:
    pyproject = tomllib.loads(read_text(ROOT / "pyproject.toml"))
    project = pyproject.get("project", {})
    package_name = project.get("name")
    docs_url = project.get("urls", {}).get("Documentation")

    if package_name != "ads-bib":
        fail(f"pyproject.toml project name must be 'ads-bib', got {package_name!r}")
    if docs_url != "https://raphschlatt.github.io/ads-bib/":
        fail(f"unexpected documentation URL in pyproject.toml: {docs_url!r}")

    readme = read_text(ROOT / "README.md")
    if "not yet on PyPI" in readme:
        fail("README.md still says ads-bib is not yet on PyPI")
    if "uv pip install ads-bib" not in readme:
        fail("README.md missing canonical PyPI install command")
    if "author_disambiguation.enabled=true" not in readme:
        fail("README.md missing the AND opt-in command hint")

    zensical = read_text(ROOT / "zensical.toml")
    if f'site_url = "{docs_url}"' not in zensical:
        fail("zensical.toml site_url does not match pyproject.toml Documentation URL")

    print("release docs ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
