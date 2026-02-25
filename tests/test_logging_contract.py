from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "ads_bib"


def _runtime_python_files() -> list[Path]:
    return sorted(path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _print_call_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            lines.append(int(node.lineno))
    return sorted(lines)


def test_runtime_modules_do_not_use_print_calls() -> None:
    violations: list[str] = []
    for path in _runtime_python_files():
        for lineno in _print_call_lines(path):
            rel = path.relative_to(PROJECT_ROOT)
            violations.append(f"{rel}:{lineno}")

    assert not violations, (
        "Runtime modules in src/ads_bib must use logging instead of print().\n"
        + "Found print() call(s):\n- "
        + "\n- ".join(violations)
    )
