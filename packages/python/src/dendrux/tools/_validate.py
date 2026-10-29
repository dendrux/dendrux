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

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from dendrux import _codes
from dendrux.tools._coerce import (
    ArgumentCoercionError,
    contains_placeholder,
    get_args_spec,
    normalize_arguments,
)
from dendrux.types import InputValidationError, ToolDef, ToolResult, _validate_persistable

if TYPE_CHECKING:
    from collections.abc import Collection

    from dendrux.types import ToolCall

logger = logging.getLogger(__name__)

INVALID_PARAMS_CODE = "invalid_params"
INVALID_TOOL_SCHEMA_CODE = "invalid_tool_schema"
INVALID_JSON_CODE = "invalid_json"
PII_PLACEHOLDER_CODE = "pii_placeholder"
ARGUMENTS_NOT_STABLE_CODE = "arguments_not_stable"

_COMPOSITE_KEYWORDS = frozenset(
    {"anyOf", "oneOf", "allOf", "$ref", "not", "if", "then", "else", "$dynamicRef"}
)
_KNOWN_TYPES = frozenset({"string", "integer", "number", "boolean", "array", "object", "null"})


class MalformedToolSchemaError(ValueError):
    """A tool's argument schema has a shape the validator cannot interpret.

    Raised by :func:`_collect_problems`; :func:`check_tool_call` turns it into
    a failed result with ``code="invalid_tool_schema"``. The message names the
    problem kind only, never schema text.
    """


# Every JSON Schema 2020-12 keyword (and its draft-07 spelling) whose value is a schema,
# a map of schemas or an array of schemas; walked by :func:`schema_problem`.
_SUBSCHEMA_MAPS = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
)
_SUBSCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems", "items")
_SUBSCHEMAS = (
    "items",
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contentSchema",
    "additionalItems",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
)
# Budget for the walk; past it the schema is refused (fail closed).
_SCHEMA_MAX_DEPTH = 32
_SCHEMA_MAX_NODES = 10_000
_SCHEMA_BUDGET_PROBLEM = "schema is too large or too deeply nested to check"


def _keyword_problem(node: dict[str, Any]) -> str | None:
    """Shape of the keywords the validator reads, in one (sub)schema.

    ``required``, when present (even as ``null``), must be a list of argument
    names; ``properties``, when present, must be an object.
    """
    if "required" in node:
        required = node["required"]
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            return "required must be a list of argument names"
    if "properties" in node and not isinstance(node["properties"], dict):
        return "properties must be an object"
    return None


def _subschemas(node: dict[str, Any]) -> list[dict[str, Any]]:
    children: list[Any] = []
    for key in _SUBSCHEMA_MAPS:
        if isinstance(node.get(key), dict):
            children.extend(node[key].values())
    for key in _SUBSCHEMA_LISTS:
        if isinstance(node.get(key), list):
            children.extend(node[key])
    children.extend(node.get(key) for key in _SUBSCHEMAS)
    return [child for child in children if isinstance(child, dict)]


def root_schema_problem(schema: Any) -> str | None:
    """:func:`_keyword_problem` for the root schema only: the cheap per-call check."""
    return _keyword_problem(schema) if isinstance(schema, dict) else None


def schema_problem(schema: Any) -> str | None:
    """Why ``schema`` cannot be validated against, or ``None`` when it can.

    Every (sub)schema reachable through the standard subschema keywords
    (``properties``, ``items``, ``anyOf``, ``$defs``, ...) is checked with
    :func:`_keyword_problem`, within a depth of 32 and 10,000 nodes; a schema
    past either budget is refused. The validator is tolerant of everything else
    (unknown keywords, odd property schemas, unsupported constructs) by design,
    and skips what it does not implement.
    """
    if not isinstance(schema, dict):
        return None
    stack: list[tuple[dict[str, Any], int]] = [(schema, 1)]
    nodes = 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > _SCHEMA_MAX_NODES or depth > _SCHEMA_MAX_DEPTH:
            return _SCHEMA_BUDGET_PROBLEM
        issue = _keyword_problem(node)
        if issue is not None:
            return issue
        stack.extend((child, depth + 1) for child in _subschemas(node))
    return None


class _Problem(NamedTuple):
    """A model-facing problem text plus the input that caused it (never rendered)."""

    text: str
    value: Any = None


def unpersistable_result(tool_call: ToolCall) -> ToolResult:
    """The ``invalid_params`` failure for unpersistable JSON; it quotes no content."""
    message = (
        f"Arguments for `{tool_call.name}` must be bounded finite JSON with persistable "
        "text. The call was not made; send it again with valid arguments."
    )
    return invalid_params_result(tool_call, message)


def _unknown_names_problem(tool_name: str, unknown: list[Any]) -> _Problem:
    """The "unknown argument" problem: a count only, never the names the model sent."""
    noun = "argument" if len(unknown) == 1 else "arguments"
    return _Problem(f"{tool_name} has {len(unknown)} unknown {noun}")


def validate_tool_params(tool_def: ToolDef, params: dict[str, Any]) -> list[str]:
    """Return every problem found in ``params`` for ``tool_def`` (empty if valid).

    Checks, in order: unknown argument names, missing required arguments,
    and basic JSON types / ``enum`` membership of top-level arguments.
    Problems name the argument and the constraint, never the value.
    Pure and synchronous; never runs tool code.
    """
    return [p.text for p in _problems_for(tool_def, params)[0]]


def _collect_problems(tool_def: ToolDef, params: dict[str, Any]) -> tuple[list[_Problem], bool]:
    """Return ``(problems, names_wrong)``; ``names_wrong`` flags unknown/missing names."""
    schema = tool_def.parameters
    if not isinstance(schema, dict):
        return [], False
    bad_schema = root_schema_problem(schema)
    if bad_schema is not None:
        raise MalformedToolSchemaError(bad_schema)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = None
    name = tool_def.name
    problems: list[_Problem] = []
    names_wrong = False

    if properties is not None and not _allows_unknown_names(schema):
        unknown = [key for key in params if key not in properties]
        if unknown:
            problems.append(_unknown_names_problem(name, unknown))
            names_wrong = True

    required = schema.get("required")
    if isinstance(required, list):
        missing = [key for key in required if isinstance(key, str) and key not in params]
        if missing:
            problems.append(
                _Problem(f"{name} is missing required argument(s): {', '.join(missing)}")
            )
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
                problems.append(_Problem(f"{name} argument '{key}' {problem}", value))

    return problems, names_wrong


_MALFORMED_SCHEMA: Any = object()  # marks the problem for an unusable schema


def _problems_for(tool_def: ToolDef, params: dict[str, Any]) -> tuple[list[_Problem], bool]:
    """:func:`_collect_problems`, with an unusable schema reported as one problem.

    The one problem carries ``_MALFORMED_SCHEMA`` as its value so callers can
    tell it from an argument problem; it holds no schema or input text.
    """
    try:
        return _collect_problems(tool_def, params)
    except (MalformedToolSchemaError, TypeError, ValueError, AttributeError, RecursionError):
        return [
            _Problem(f"{tool_def.name} has a malformed argument schema", _MALFORMED_SCHEMA)
        ], False


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


def format_invalid_json(tool_name: str, parse_error: str) -> str:
    """Build the standard model-facing message for unparseable arguments."""
    return (
        f"Your arguments for `{tool_name}` were not valid JSON ({parse_error}). "
        "Send the call again with valid JSON; for long content, write it in smaller pieces."
    )


def check_tool_json(tool_call: ToolCall) -> ToolResult | None:
    """Return a failed ``ToolResult`` if the call's arguments did not parse, else ``None``.

    The payload carries ``code: "invalid_json"`` and the parse error only,
    never the raw arguments.
    """
    if tool_call.parse_error is None:
        return None
    message = format_invalid_json(tool_call.name, tool_call.parse_error)
    return ToolResult(
        name=tool_call.name,
        call_id=tool_call.id,
        payload=json.dumps({"error": message, "code": INVALID_JSON_CODE}),
        success=False,
        error=message,
    )


def check_tool_call(
    tool_call: ToolCall,
    tool_def: ToolDef | None,
    *,
    placeholders: Collection[str] = (),
) -> ToolResult | None:
    """Validate a tool call; return a failed ``ToolResult`` if invalid, else ``None``.

    Unparseable arguments are rejected first (``invalid_json``). Unknown
    tools (no ``ToolDef``) otherwise pass through so the executor reports them.
    A problem caused by one of ``placeholders`` (redacted PII the tool may
    not receive) is reported as ``code="pii_placeholder"``. A tool whose own
    argument schema cannot be validated against is refused with
    ``code="invalid_tool_schema"`` instead of raising. Params holding U+0000
    or a lone surrogate (in any key or value), nonfinite numbers, or structures
    exceeding the shared persistence limits are refused with ``invalid_params``
    for every tool, known or not, before anything runs or pauses; the message
    quotes none of the content.
    """
    json_result = check_tool_json(tool_call)
    if json_result is not None:
        return json_result
    try:
        _validate_persistable(tool_call.params, "params")
    except InputValidationError:
        return unpersistable_result(tool_call)
    if tool_def is None:
        return None
    return _check_params(tool_call, tool_def, placeholders)


def _check_params(
    tool_call: ToolCall, tool_def: ToolDef, placeholders: Collection[str]
) -> ToolResult | None:
    problems, names_wrong = _problems_for(tool_def, tool_call.params or {})
    if not problems:
        return None
    if any(p.value is _MALFORMED_SCHEMA for p in problems):
        return malformed_schema_result(tool_call)
    if any(contains_placeholder(p.value, placeholders) for p in problems):
        return placeholder_result(tool_call)
    message = format_invalid_params(
        tool_def, [p.text for p in problems], list_valid_names=names_wrong
    )
    return invalid_params_result(tool_call, message)


def invalid_params_result(
    tool_call: ToolCall, message: str, *, code: str = INVALID_PARAMS_CODE
) -> ToolResult:
    """A failed ``ToolResult`` carrying a value-free validation message."""
    return ToolResult(
        name=tool_call.name,
        call_id=tool_call.id,
        payload=json.dumps({"error": message, "code": code}),
        success=False,
        error=message,
        meta={"code": code},
    )


def malformed_schema_result(tool_call: ToolCall) -> ToolResult:
    """The ``invalid_tool_schema`` failure: the tool's own schema cannot be validated against."""
    _codes.log_code(logger, _codes.INVALID_TOOL_SCHEMA)
    message = (
        f"`{tool_call.name}` cannot be called: its argument schema is malformed. "
        "The call was not made; do not retry it."
    )
    return invalid_params_result(tool_call, message, code=INVALID_TOOL_SCHEMA_CODE)


def placeholder_result(tool_call: ToolCall) -> ToolResult:
    """The ``pii_placeholder`` failure: a redacted value the tool may not receive."""
    message = (
        f"An argument for `{tool_call.name}` is a redacted placeholder, and this tool is not "
        "allowed to receive the real value. The value is withheld by policy: do not retry "
        "with the placeholder."
    )
    return invalid_params_result(tool_call, message, code=PII_PLACEHOLDER_CODE)


def prepare_tool_call(
    tool_call: ToolCall,
    tool_def: ToolDef | None,
    fn: Any,
    *,
    placeholders: Collection[str] = (),
) -> ToolCall | ToolResult:
    """JSON-Schema check, then lax coercion for local tools.

    Returns the call with **normalised** arguments (see
    :mod:`dendrux.tools._coerce`), or a failed ``ToolResult``. Tools without
    an arguments model (explicit ``parameters=``, MCP, client, skill tools)
    get the JSON-Schema check only and keep their arguments as sent.
    """
    invalid = check_tool_call(tool_call, tool_def, placeholders=placeholders)
    if invalid is not None:
        return invalid
    spec = get_args_spec(fn)
    if spec is None or tool_def is None:
        return tool_call
    try:
        normalized = normalize_arguments(spec, tool_call.params or {})
    except ArgumentCoercionError as exc:
        return coercion_failure(tool_call, tool_def, exc, placeholders=placeholders)
    # Always the normalised dict: ``3 == 3.0`` in Python, but hooks, approvals
    # and stored calls must see the JSON the function is called with.
    return dataclasses.replace(tool_call, params=normalized)


def coercion_failure(
    tool_call: ToolCall,
    tool_def: ToolDef,
    exc: ArgumentCoercionError,
    *,
    placeholders: Collection[str] = (),
) -> ToolResult:
    """The failed result for an :class:`ArgumentCoercionError` (value-free)."""
    if any(contains_placeholder(p.value, placeholders) for p in exc.problems):
        return placeholder_result(tool_call)
    problems = [
        f"{tool_def.name} argument '{p.path}' {_lower_first(p.message)}" for p in exc.problems
    ]
    message = format_invalid_params(tool_def, problems, list_valid_names=False)
    code = ARGUMENTS_NOT_STABLE_CODE if exc.unstable else INVALID_PARAMS_CODE
    return invalid_params_result(tool_call, message, code=code)


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


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
        return f"must be one of {options}"

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
