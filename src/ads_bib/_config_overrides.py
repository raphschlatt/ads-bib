"""Strict dotted override handling for pipeline config dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints


def validate_config_mapping(data: Mapping[str, Any], schema_type: type) -> None:
    """Reject unknown keys in *data* according to the dataclass config schema."""
    _validate_mapping(data, schema_type, prefix=())


def apply_config_override(
    data: dict[str, Any],
    key: str,
    value: object,
    *,
    schema_type: type,
) -> None:
    """Apply one dotted-key override after validating the target path."""
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("Override key cannot be empty.")
    _validate_path(parts, schema_type)

    current: dict[str, Any] = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _validate_mapping(data: Mapping[str, Any], schema_type: object, *, prefix: tuple[str, ...]) -> None:
    if _is_mapping_type(schema_type):
        return
    if not _is_dataclass_type(schema_type):
        path = ".".join(prefix) or "<root>"
        raise ValueError(f"Config key '{path}' does not accept nested values.")

    field_types = _field_types(schema_type)
    allowed = set(field_types)
    for key, value in data.items():
        key_text = str(key)
        if key_text not in allowed:
            raise ValueError(f"Unknown config key '{'.'.join((*prefix, key_text))}'.")
        child_type = field_types[key_text]
        if isinstance(value, Mapping):
            _validate_mapping(value, child_type, prefix=(*prefix, key_text))


def _validate_path(parts: list[str], schema_type: object) -> None:
    current_type: object = schema_type
    walked: list[str] = []
    for index, part in enumerate(parts):
        walked.append(part)
        if _is_mapping_type(current_type):
            return
        if not _is_dataclass_type(current_type):
            raise ValueError(f"Config key '{'.'.join(walked)}' does not accept nested values.")
        field_types = _field_types(current_type)
        if part not in field_types:
            raise ValueError(f"Unknown config key '{'.'.join(walked)}'.")
        current_type = field_types[part]
        if index < len(parts) - 1 and not (
            _is_dataclass_type(current_type) or _is_mapping_type(current_type)
        ):
            raise ValueError(f"Config key '{'.'.join(walked)}' does not accept nested values.")


def _field_types(schema_type: object) -> dict[str, object]:
    hints = get_type_hints(schema_type)
    return {field.name: hints.get(field.name, field.type) for field in fields(schema_type)}


def _is_dataclass_type(value: object) -> bool:
    return isinstance(value, type) and is_dataclass(value)


def _is_mapping_type(value: object) -> bool:
    origin = get_origin(value)
    if origin in {dict, Mapping}:
        return True
    if value in {dict, Mapping, Any}:
        return True
    return any(_is_mapping_type(arg) for arg in get_args(value))
