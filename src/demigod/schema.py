"""Minimal JSON-Schema subset check for demigod artifact payloads.

Lives in `demigod`, not `reagents`, because it is needed INSIDE the sandbox: a
DEMI_GOD validates its own payload before writing the manifest, which turns a
schema mismatch into something the agent can still fix on its next turn rather
than a failure the orchestrator discovers after the sandbox is gone.

The dependency direction is deliberate. `reagents` imports this; `demigod` never
imports `reagents`. Only `demigod` is shipped into the sandbox image, so GOD's
planner prompts and inverse maps cannot be read by the agent they constrain.

Supports the subset the planner actually emits: type / required / properties /
items over object, array, string, number, integer, boolean.
"""

from __future__ import annotations

from typing import Any


def validate_payload(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable errors. Empty means valid."""
    if not schema:
        return []
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
