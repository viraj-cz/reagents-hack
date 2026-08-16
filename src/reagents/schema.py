"""Minimal JSON-schema checks for demigod artifacts."""

from __future__ import annotations

from typing import Any


def validate_payload(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Validate `payload` against a small subset of JSON Schema (type/required/properties)."""
    return _check(payload, schema, path="$")


def _check(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                errors.extend(_check(value[key], subschema, f"{path}.{key}"))
    elif expected == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors.extend(_check(item, item_schema, f"{path}[{i}]"))
    elif expected == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string")
    elif expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path}: expected number")
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer")
    elif expected == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean")
    return errors
