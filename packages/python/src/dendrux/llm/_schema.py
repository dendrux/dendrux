"""OpenAI strict-mode schema normalization.

Pydantic's model_json_schema() produces standard JSON Schema. OpenAI's
strict mode requires a few adjustments:
  - additionalProperties: false on every object
  - All properties must be in required
  - Root must be an object type

This is a lightweight pass, not an aggressive downgrade. OpenAI strict
mode supports most JSON Schema features (pattern, format, minimum,
maximum, $defs, $ref, anyOf, etc.). We only fix the known mismatches.
"""

from __future__ import annotations

import copy
from typing import Any


def normalize_for_openai_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON Schema for OpenAI strict mode.

    Returns a copy with:
    - additionalProperties: false on every object
    - All properties added to required
    - $defs preserved (OpenAI supports them)
    """
    schema = copy.deepcopy(schema)
    _normalize_object(schema)
    return schema


def _normalize_object(node: dict[str, Any]) -> None:
    """Recursively normalize a schema node for OpenAI strict mode."""
    if not isinstance(node, dict):
        return

    # Handle object types: add additionalProperties and required
    if node.get("type") == "object" and "properties" in node:
        node["additionalProperties"] = False
        # Ensure all properties are in required
        props = node.get("properties", {})
        existing_required = set(node.get("required", []))
        node["required"] = list(existing_required | set(props.keys()))

    # Recurse into properties
    for prop_schema in node.get("properties", {}).values():
        _normalize_object(prop_schema)

    # Recurse into items (arrays)
    items = node.get("items")
    if isinstance(items, dict):
        _normalize_object(items)

    # Recurse into anyOf / oneOf / allOf
    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    _normalize_object(variant)

    # Recurse into $defs
    defs = node.get("$defs")
    if isinstance(defs, dict):
        for def_schema in defs.values():
            _normalize_object(def_schema)
