"""Pre-execution validation of tool-call arguments against a tool's schema.

Models — especially small open-weight ones — sometimes emit tool calls
with misspelled or garbled argument names, missing required arguments,
or wrong value types. Calling the tool with such params produces a raw
Python ``TypeError`` (or an opaque MCP server error) that models recover
from poorly. This module checks the params against ``ToolDef.parameters``
before any tool code runs and builds a short, model-oriented error.

The validator is deliberately small and tolerant: it checks top-level
argument names, ``required``, basic JSON types (including type arrays such
as ``["string", "null"]``) and ``enum``. Any property schema that uses a
construct it does not implement (``anyOf``, ``oneOf``, ``allOf``, ``$ref``,
``not``, unknown type names, ...) is skipped rather than rejected, because
MCP servers can publish arbitrary JSON Schema.
"""

from __future__ import annotations

import json
from typing import Any

from dendrux.types import ToolCall, ToolDef, ToolResult

INVALID_PARAMS_CODE = "invalid_params"

_COMPOSITE_KEYWORDS = frozenset(
    {"anyOf", "oneOf", "allOf", "$ref", "not", "if", "then", "else", "$dynamicRef"}
)
_KNOWN_TYPES = frozenset({"string", "integer", "number", "boolean", "array", "object", "null"})


def validate_tool_params(tool_def: ToolDef, params: dict[str, Any]) -> list[str]:
    """Return every problem found in ``params`` for ``tool_def`` (empty if valid).

    Checks, in order: unknown argument names, missing required arguments,
    and basic JSON types / ``enum`` membership of top-level arguments.
    Pure and synchronous; never runs tool code.
    """
    return _collect_problems(tool_def, params)[0]


def _collect_problems(tool_def: ToolDef, params: dict[str, Any]) -> tuple[list[str], bool]:
    """Return ``(problems, names_wrong)``; ``names_wrong`` flags unknown/missing names."""
    schema = tool_def.parameters
    if not isinstance(schema, dict):
        return [], False
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = None
    name = tool_def.name
    problems: list[str] = []
    names_wrong = False

    if properties is not None and not _allows_unknown_names(schema):
        unknown = [key for key in params if key not in properties]
        if unknown:
            quoted = ", ".join(f"'{key}'" for key in unknown)
            noun = "argument" if len(unknown) == 1 else "arguments"
            problems.append(f"{name} has no {noun} {quoted}")
            names_wrong = True

    required = schema.get("required")
    if isinstance(required, list):
        missing = [key for key in required if isinstance(key, str) and key not in params]
        if missing:
            problems.append(f"{name} is missing required argument(s): {', '.join(missing)}")
            names_wrong = True

    if properties is not None:
        required_set = set(required) if isinstance(required, list) else set()
        for key, value in params.items():
            prop = properties.get(key)
            if not isinstance(prop, dict):
                continue
            if value is None and key not in required_set:
                continue
            problem = _check_value(prop, value)
            if problem:
                problems.append(f"{name} argument '{key}' {problem}")

    return problems, names_wrong


def format_invalid_params(
    tool_def: ToolDef, problems: list[str], *, list_valid_names: bool = True
) -> str:
    """Join validation problems into the standard model-facing message.

    ``list_valid_names`` appends the tool's declared argument names, which
    is what a model needs after an unknown or missing argument.
    """
    message = "; ".join(problems) + "."
    if list_valid_names:
        message += f" Valid arguments: {_valid_names(tool_def)}."
    return message


def check_tool_call(tool_call: ToolCall, tool_def: ToolDef | None) -> ToolResult | None:
    """Validate a tool call; return a failed ``ToolResult`` if invalid, else ``None``.

    Unknown tools (no ``ToolDef``) pass through so the executor reports them.
    """
    if tool_def is None:
        return None
    problems, names_wrong = _collect_problems(tool_def, tool_call.params or {})
    if not problems:
        return None
    message = format_invalid_params(tool_def, problems, list_valid_names=names_wrong)
    return ToolResult(
        name=tool_call.name,
        call_id=tool_call.id,
        payload=json.dumps({"error": message, "code": INVALID_PARAMS_CODE}),
        success=False,
        error=message,
    )


def _allows_unknown_names(schema: dict[str, Any]) -> bool:
    if any(keyword in schema for keyword in _COMPOSITE_KEYWORDS):
        return True
    if "patternProperties" in schema:
        return True
    additional = schema.get("additionalProperties", False)
    return additional is not False


def _valid_names(tool_def: ToolDef) -> str:
    properties = tool_def.parameters.get("properties") if tool_def.parameters else None
    if not isinstance(properties, dict) or not properties:
        return "(none)"
    return ", ".join(properties)


def _check_value(prop: dict[str, Any], value: Any) -> str | None:
    if any(keyword in prop for keyword in _COMPOSITE_KEYWORDS):
        return None

    declared = prop.get("type")
    if declared is not None:
        types = declared if isinstance(declared, list) else [declared]
        if not all(isinstance(t, str) and t in _KNOWN_TYPES for t in types):
            return None
        if not any(_matches_type(t, value) for t in types):
            return f"must be {' or '.join(types)}, got {_json_type_name(value)}"

    enum = prop.get("enum")
    if isinstance(enum, list) and not any(_json_equal(value, option) for option in enum):
        options = ", ".join(_dump(option) for option in enum)
        return f"must be one of {options}, got {_dump(value)}"

    return None


def _matches_type(json_type: str, value: Any) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return value is None


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return bool(a == b)


def _dump(value: Any) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
