"""Tool decorator and schema generation.

The @tool decorator turns a plain Python function into a Dendrite tool
by attaching a ToolDef to it. The function itself is unchanged — it stays
callable as normal. Schema is auto-generated from type hints.
"""

from __future__ import annotations

import inspect
import json
import typing
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Callable

from dendrite.types import ToolDef, ToolTarget

_TOOL_DEF_ATTR = "__tool_def__"

_TYPE_MAP: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    dict: "object",
    list: "array",
}


_SENTINEL: float = object()  # type: ignore[assignment]

DEFAULT_TOOL_TIMEOUT = 120.0


def tool(
    target: ToolTarget | str = ToolTarget.SERVER,
    parallel: bool = True,
    priority: int = 0,
    max_calls_per_run: int | None = None,
    timeout_seconds: float = _SENTINEL,
) -> Callable[..., Any]:
    """Decorator that registers a function as a Dendrite tool.

    Args:
        target: Where the tool runs — "server" (default) or "client".
        parallel: Whether concurrent execution is allowed (default True).
        priority: Reserved — not enforced.
        max_calls_per_run: Maximum calls per run. None = unlimited.
        timeout_seconds: Execution timeout in seconds. Default 120s.

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

        schema = _generate_schema(fn)
        tool_def = ToolDef(
            name=name,
            description=(fn.__doc__ or "").strip(),
            parameters=schema,
            target=resolved_target,
            parallel=parallel,
            priority=priority,
            max_calls_per_run=max_calls_per_run,
            timeout_seconds=resolved_timeout,
            has_explicit_timeout=explicit_timeout,
        )
        setattr(fn, _TOOL_DEF_ATTR, tool_def)
        return fn

    return decorator


# Fix introspection: replace the sentinel with 120.0 in the visible signature
# so IDEs, help(), and doc generators show a clean default.
_tool_sig = inspect.signature(tool)
_tool_params = list(_tool_sig.parameters.values())
_tool_params[-1] = _tool_params[-1].replace(default=DEFAULT_TOOL_TIMEOUT)
tool.__signature__ = _tool_sig.replace(parameters=_tool_params)  # type: ignore[attr-defined]


def get_tool_def(fn: Callable[..., Any]) -> ToolDef:
    """Get ToolDef from a decorated function. Raises ValueError if not a tool."""
    tool_def: ToolDef | None = getattr(fn, _TOOL_DEF_ATTR, None)
    if tool_def is None:
        raise ValueError(f"'{fn.__name__}' is not a Dendrite tool. Decorate it with @tool().")
    return tool_def


def is_tool(fn: Callable[..., Any]) -> bool:
    """Check if a function is decorated with @tool."""
    return hasattr(fn, _TOOL_DEF_ATTR)


def _generate_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON Schema from a function's type hints."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

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

        prop = _type_to_schema(hint)

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
        elif not _is_optional(hint):
            required.append(name)

        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_schema(hint: type) -> dict[str, Any]:
    """Convert a Python type hint to a JSON Schema property."""
    if _is_optional(hint):
        inner = _unwrap_optional(hint)
        return _type_to_schema(inner)

    origin = get_origin(hint)

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

    if origin is list:
        args = get_args(hint)
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _type_to_schema(args[0])
        return schema

    if origin is dict:
        return {"type": "object"}

    json_type = _TYPE_MAP.get(hint)
    if json_type:
        return {"type": json_type}

    raise TypeError(
        f"Type {hint!r} is not supported in Dendrite tool schemas. "
        f"Supported types: str, int, float, bool, list, dict, and Optional variants. "
        f"If this is a custom type, convert it to a supported type in the tool signature."
    )


def _is_optional(hint: type) -> bool:
    """Check if a type hint is Optional (i.e., X | None or Optional[X])."""
    origin = get_origin(hint)
    if origin is not type(int | str) and origin is not typing.Union:
        return False
    args = get_args(hint)
    return type(None) in args


def _unwrap_optional(hint: type) -> type:
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
