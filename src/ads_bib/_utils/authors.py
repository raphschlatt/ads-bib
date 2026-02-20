"""Author formatting and parsing helpers shared across modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def author_list(value: object) -> list[str]:
    """Normalize author input into a cleaned list of author strings."""
    if isinstance(value, str):
        return [v.strip() for v in value.split(";") if v.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def author_text(value: object, *, separator: str = "; ") -> str:
    """Serialize author input to a separator-joined author string."""
    return separator.join(author_list(value))


def first_author_lastname(value: object) -> str | None:
    """Extract first author last name from list/string author input."""
    authors = author_list(value)
    if not authors:
        return None
    first = authors[0]
    if "," in first:
        return first.split(",", 1)[0].strip() or None
    parts = first.split()
    return parts[-1].strip() if parts else None
