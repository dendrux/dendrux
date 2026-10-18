"""Tool decorator and schema generation.

The @tool decorator turns a plain Python function into a Dendrux tool
by attaching a ToolDef to it. The function itself is unchanged — it stays
callable as normal. Schema is auto-generated from type hints.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import inspect
import json
import re
import typing
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

from dendrux.types import ToolDef, ToolTarget

_TOOL_DEF_ATTR = "__tool_def__"

_TYPE_MAP: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    dict: "object",
    list: "array",
}

_SUPPORTED_TYPES_HINT = (
    "Supported types: str, int, float, bool, list[T], dict, dict[str, T], Literal[...], "
    "Enum subclasses, pydantic BaseModel, dataclasses, TypedDict, Annotated[T, ...], "
    "and Optional variants. For anything else pass an explicit schema with "
    "@tool(parameters=...)."
)


_SENTINEL: float = object()  # type: ignore[assignment]

DEFAULT_TOOL_TIMEOUT = 120.0


def tool(
    target: ToolTarget | str = ToolTarget.SERVER,
    parallel: bool = True,
    max_calls_per_run: int | None = None,
    timeout_seconds: float = _SENTINEL,
    *,
    parameters: dict[str, Any] | None = None,
    strict: bool = False,
) -> Callable[..., Any]:
    """Decorator that registers a function as a Dendrux tool.

    Args:
        target: Where the tool runs — "server" (default) or "client".
        parallel: Whether concurrent execution is allowed (default True).
        max_calls_per_run: Maximum calls per run. None = unlimited.
        timeout_seconds: Execution timeout in seconds. Default 120s.
        parameters: Explicit JSON Schema (``{"type": "object", ...}``) for the
            tool's arguments. Bypasses schema generation from type hints.
        strict: Opt into strict (schema-constrained) tool calling on OpenAI
            and Anthropic. The schema is normalized for OpenAI at decoration
            time (every property required, closed objects; free-form dicts
            are rejected) and OpenAI providers send ``"strict": true``.
            ``AnthropicProvider`` sends ``"strict": true`` with a per-request
            copy of that schema in which keywords Anthropic strict mode
            rejects (``minimum``/``maximum``, ``maxItems``, unsupported
            ``format`` or ``pattern``, ...) are moved into the description;
            a schema Anthropic cannot take strictly (recursion, tuples,
            untyped values, or over the request's 20 strict tools / 24
            optional / 16 union-typed parameter limits) is sent non-strict.

    Usage:
        @tool(target="server")
        async def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = fn.__name__
        if not name.isidentifier():
            raise ValueError(
                f"Tool name '{name}' is not a valid Python identifier. "
                f"Use a named function instead of a lambda or dynamic callable."
            )

        resolved_target = ToolTarget(target)
        if resolved_target not in (ToolTarget.SERVER, ToolTarget.CLIENT):
            raise ValueError(
                f"Tool target '{target}' is not supported. "
                f"Only 'server' and 'client' are implemented. "
                f"'human' and 'agent' are reserved for future use."
            )

        explicit_timeout = timeout_seconds is not _SENTINEL
        resolved_timeout = float(timeout_seconds) if explicit_timeout else DEFAULT_TOOL_TIMEOUT

        if parameters is not None:
            schema = _validate_explicit_parameters(parameters, name)
        else:
            schema = _generate_schema(fn)
        if strict:
            schema = _strict_schema(schema, name)

        tool_def = ToolDef(
            name=name,
            description=(fn.__doc__ or "").strip(),
            parameters=schema,
            target=resolved_target,
            parallel=parallel,
            max_calls_per_run=max_calls_per_run,
            timeout_seconds=resolved_timeout,
            has_explicit_timeout=explicit_timeout,
            strict=strict,
        )
        setattr(fn, _TOOL_DEF_ATTR, tool_def)
        return fn

    return decorator


# Fix introspection: replace the sentinel with 120.0 in the visible signature
# so IDEs, help(), and doc generators show a clean default.
_tool_sig = inspect.signature(tool)
tool.__signature__ = _tool_sig.replace(  # type: ignore[attr-defined]
    parameters=[
        p.replace(default=DEFAULT_TOOL_TIMEOUT) if p.name == "timeout_seconds" else p
        for p in _tool_sig.parameters.values()
    ]
)


def get_tool_def(fn: Callable[..., Any]) -> ToolDef:
    """Get ToolDef from a decorated function. Raises ValueError if not a tool."""
    tool_def: ToolDef | None = getattr(fn, _TOOL_DEF_ATTR, None)
    if tool_def is None:
        raise ValueError(f"'{fn.__name__}' is not a Dendrux tool. Decorate it with @tool().")
    return tool_def


def is_tool(fn: Callable[..., Any]) -> bool:
    """Check if a function is decorated with @tool."""
    return hasattr(fn, _TOOL_DEF_ATTR)


# ---------------------------------------------------------------------------
# Explicit schemas and strict mode
# ---------------------------------------------------------------------------


def _validate_explicit_parameters(parameters: Any, tool_name: str) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        raise TypeError(
            f"parameters= for tool '{tool_name}' must be a dict (a JSON Schema object), "
            f"got {type(parameters).__name__}."
        )
    if parameters.get("type") != "object":
        raise ValueError(
            f'parameters= for tool \'{tool_name}\' must be a JSON Schema with "type": "object" '
            f"at the root, got type={parameters.get('type')!r}."
        )
    return copy.deepcopy(parameters)


def _strict_schema(schema: dict[str, Any], tool_name: str) -> dict[str, Any]:
    from dendrux.llm._schema import find_open_object, normalize_for_openai_strict

    open_path = find_open_object(schema, "parameters")
    if open_path is not None:
        raise TypeError(
            f"Tool '{tool_name}' uses strict=True but {open_path} is a free-form object "
            f"(dict / dict[str, T] / an object without properties). OpenAI strict mode "
            f"requires every object to declare its properties. Use a BaseModel, dataclass, "
            f"or TypedDict instead, or drop strict=True."
        )
    return normalize_for_openai_strict(schema)


# ---------------------------------------------------------------------------
# Schema generation from type hints
# ---------------------------------------------------------------------------


def _generate_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON Schema from a function's type hints."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    doc_descriptions = _parse_docstring_args(fn.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        hint = hints.get(name)
        if hint is None:
            raise TypeError(
                f"Parameter '{name}' of tool '{fn.__name__}' has no type hint. "
                f"All tool parameters must be typed for schema generation."
            )

        try:
            prop = _type_to_schema(hint)
        except TypeError as e:
            raise TypeError(f"Parameter '{name}' of tool '{fn.__name__}': {e}") from e

        if not _param_has_annotated_description(hint) and name in doc_descriptions:
            prop["description"] = doc_descriptions[name]

        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            try:
                json.dumps(param.default)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Default value for parameter '{name}' of tool '{fn.__name__}' "
                    f"is not JSON-serializable: {param.default!r}"
                ) from e
            prop["default"] = param.default
        elif not _is_optional(_strip_annotated(hint)):
            required.append(name)

        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_schema(hint: Any, seen: frozenset[type] = frozenset()) -> dict[str, Any]:
    """Convert a Python type hint to a JSON Schema property."""
    origin = get_origin(hint)

    if origin is Annotated:
        schema = _type_to_schema(hint.__origin__, seen)
        description = _annotated_description(hint)
        if description is not None:
            schema["description"] = description
        return schema

    if _is_optional(hint):
        return _make_nullable(_type_to_schema(_unwrap_optional(hint), seen))

    # Reject non-optional union types (e.g., str | int)
    if origin is type(int | str) or origin is typing.Union:
        args = get_args(hint)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) > 1:
            raise TypeError(
                f"Union type {hint!r} is not supported in tool schemas. "
                f"Only Optional[T] (T | None) is allowed. "
                f"Got non-None types: {non_none!r}."
            )

    if origin is Literal:
        return _enum_schema(list(get_args(hint)), hint)

    if origin is list:
        args = get_args(hint)
        schema = {"type": "array"}
        if args:
            schema["items"] = _type_to_schema(args[0], seen)
        return schema

    if origin is dict:
        return _dict_schema(hint, seen)

    if isinstance(hint, type):
        if issubclass(hint, enum.Enum):
            return _enum_schema([member.value for member in hint], hint)
        if issubclass(hint, BaseModel):
            return _pydantic_schema(hint)
        if dataclasses.is_dataclass(hint):
            return _dataclass_schema(hint, seen)
        if _is_typeddict(hint):
            return _typeddict_schema(hint, seen)

    json_type = _TYPE_MAP.get(hint)
    if json_type:
        return {"type": json_type}

    raise TypeError(
        f"Type {hint!r} is not supported in Dendrux tool schemas. {_SUPPORTED_TYPES_HINT}"
    )


def _enum_schema(values: list[Any], hint: Any) -> dict[str, Any]:
    if not values:
        raise TypeError(f"{hint!r} has no values; an enum needs at least one.")
    json_types = {_literal_json_type(v, hint) for v in values}
    if json_types == {"integer", "number"}:
        json_types = {"number"}
    if len(json_types) != 1:
        raise TypeError(
            f"{hint!r} has mixed value types ({', '.join(sorted(json_types))}). "
            f"All Literal / Enum values must share one JSON type."
        )
    return {"type": json_types.pop(), "enum": values}


def _literal_json_type(value: Any, hint: Any) -> str:
    for py_type, json_type in ((bool, "boolean"), (int, "integer"), (float, "number")):
        if isinstance(value, py_type):
            return json_type
    if isinstance(value, str):
        return "string"
    raise TypeError(
        f"{hint!r} contains unsupported value {value!r}. "
        f"Literal / Enum values must be str, int, float, or bool."
    )


def _dict_schema(hint: Any, seen: frozenset[type]) -> dict[str, Any]:
    args = get_args(hint)
    if not args:
        return {"type": "object"}
    key_type, value_type = args
    if key_type is not str:
        raise TypeError(f"{hint!r}: dict keys must be str in tool schemas (JSON object keys).")
    if value_type is Any:
        return {"type": "object"}
    return {"type": "object", "additionalProperties": _type_to_schema(value_type, seen)}


def _make_nullable(schema: dict[str, Any]) -> dict[str, Any]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        schema["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list):
        if "null" not in schema_type:
            schema["type"] = [*schema_type, "null"]
    else:
        return {"anyOf": [schema, {"type": "null"}]}
    if "enum" in schema and None not in schema["enum"]:
        schema["enum"] = [*schema["enum"], None]
    return schema


def _pydantic_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    return typing.cast("dict[str, Any]", _inline_refs(raw, defs, ()))


def _inline_refs(node: Any, defs: dict[str, Any], stack: tuple[str, ...]) -> Any:
    """Resolve ``#/$defs/...`` refs so providers get a self-contained schema."""
    if isinstance(node, list):
        return [_inline_refs(item, defs, stack) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        name = ref.removeprefix("#/$defs/")
        if name not in defs:
            raise TypeError(f"Cannot resolve schema reference {ref!r}.")
        if name in stack:
            raise TypeError(
                f"Recursive model {name!r} cannot be inlined into a tool schema. "
                f"Pass an explicit schema with @tool(parameters=...)."
            )
        resolved = _inline_refs(defs[name], defs, (*stack, name))
        siblings = {k: _inline_refs(v, defs, stack) for k, v in node.items() if k != "$ref"}
        return {**resolved, **siblings}
    return {k: _inline_refs(v, defs, stack) for k, v in node.items()}


def _check_recursion(cls: type, seen: frozenset[type]) -> frozenset[type]:
    if cls in seen:
        raise TypeError(
            f"Recursive type {cls.__name__!r} cannot be inlined into a tool schema. "
            f"Pass an explicit schema with @tool(parameters=...)."
        )
    return seen | {cls}


def _dataclass_schema(cls: type, seen: frozenset[type]) -> dict[str, Any]:
    seen = _check_recursion(cls, seen)
    hints = get_type_hints(cls, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        prop = _type_to_schema(hints[field.name], seen)
        if field.default is not dataclasses.MISSING:
            if _is_json_serializable(field.default):
                prop["default"] = field.default
        elif field.default_factory is dataclasses.MISSING:
            required.append(field.name)
        properties[field.name] = prop
    return {"type": "object", "properties": properties, "required": required}


def _is_typeddict(cls: type) -> bool:
    return issubclass(cls, dict) and hasattr(cls, "__required_keys__")


def _typeddict_schema(cls: type, seen: frozenset[type]) -> dict[str, Any]:
    seen = _check_recursion(cls, seen)
    hints = get_type_hints(cls, include_extras=True)
    required_keys: frozenset[str] = getattr(cls, "__required_keys__", frozenset())
    properties: dict[str, Any] = {}
    for name, hint in hints.items():
        if get_origin(hint) in (typing.Required, typing.NotRequired):
            hint = get_args(hint)[0]
        properties[name] = _type_to_schema(hint, seen)
    required = [name for name in properties if name in required_keys]
    return {"type": "object", "properties": properties, "required": required}


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _strip_annotated(hint: Any) -> Any:
    while get_origin(hint) is Annotated:
        hint = hint.__origin__
    return hint


def _annotated_description(hint: Any) -> str | None:
    """Description from ``Annotated[T, Field(description=...)]`` or ``Annotated[T, "text"]``."""
    if get_origin(hint) is not Annotated:
        return None
    for meta in reversed(hint.__metadata__):
        if isinstance(meta, str):
            return meta
        description = getattr(meta, "description", None)
        if isinstance(description, str):
            return description
    return None


def _param_has_annotated_description(hint: Any) -> bool:
    if _annotated_description(hint) is not None:
        return True
    if _is_optional(hint):
        return any(_annotated_description(arg) is not None for arg in get_args(hint))
    return False


_ARGS_HEADERS = frozenset({"args:", "arguments:", "parameters:", "params:"})
_ARG_LINE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _parse_docstring_args(doc: str | None) -> dict[str, str]:
    """Extract ``name: text`` entries from a Google-style ``Args:`` section."""
    if not doc:
        return {}
    descriptions: dict[str, str] = {}
    section_indent: int | None = None
    arg_indent: int | None = None
    current: str | None = None
    for line in inspect.cleandoc(doc).splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if section_indent is None:
            if stripped.lower() in _ARGS_HEADERS:
                section_indent = indent
            continue
        if not stripped:
            continue
        if indent <= section_indent:
            break
        if arg_indent is None:
            arg_indent = indent
        if indent == arg_indent:
            match = _ARG_LINE.match(stripped)
            current = match.group(1) if match else None
            if match:
                descriptions[match.group(1)] = match.group(2).strip()
        elif current is not None:
            descriptions[current] = f"{descriptions[current]} {stripped}".strip()
    return {name: text for name, text in descriptions.items() if text}


def _is_optional(hint: Any) -> bool:
    """Check if a type hint is Optional (i.e., X | None or Optional[X])."""
    origin = get_origin(hint)
    if origin is not type(int | str) and origin is not typing.Union:
        return False
    args = get_args(hint)
    return type(None) in args


def _unwrap_optional(hint: Any) -> Any:
    """Extract the inner type from Optional[X] / X | None.

    Raises TypeError for multi-type unions like ``str | int | None`` because
    they cannot be represented as a single JSON Schema type. Only
    ``Optional[T]`` (i.e., ``T | None``) is supported.
    """
    args = get_args(hint)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) > 1:
        raise TypeError(
            f"Union type {hint!r} is not supported in tool schemas. "
            f"Only Optional[T] (T | None) is allowed. "
            f"Got non-None types: {non_none!r}."
        )
    return non_none[0] if non_none else str
