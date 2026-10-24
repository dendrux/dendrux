"""Argument coercion for local ``@tool`` functions.

A tool whose schema is generated from its signature gets a Pydantic
arguments model, built once at decoration time with ``create_model``. Model
arguments are checked against the JSON Schema first (``tools/_validate.py``)
and then validated through this model in **lax** mode: ``3.0`` becomes ``3``,
a dict becomes the declared ``BaseModel`` / dataclass, a string becomes the
``Enum`` member, an ISO string becomes a ``datetime``, and ``Field``
constraints are enforced.

The validated arguments are dumped back to JSON (``model_dump(mode="json")``).
That **normalised** form is what hooks, approval events, persisted tool calls
and pause state see, and the function receives objects rebuilt from exactly
it, so what a human approves is what runs. A validator that does not give the
same normalised form when it runs again on its own output is rejected
(``code="arguments_not_stable"``) instead of executing something different
from what was approved.

Validation runs synchronously on the event loop, so slow or I/O-bound model
validators delay other runs in the process. Nested scalars are coerced laxly
(``"3"`` becomes ``3`` inside a nested model); only top-level JSON types are
checked strictly first.

Tools with an explicit ``parameters=`` schema, MCP tools and client tools have
no arguments model and keep JSON-Schema validation only.
"""

from __future__ import annotations

import inspect
import re
import typing
from typing import TYPE_CHECKING, Any, NamedTuple, get_type_hints

import pydantic_core
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from dendrux.tool import _VARIADIC_KINDS, _is_optional, _strip_annotated

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

_ARGS_SPEC_ATTR = "__tool_args_spec__"

# Built-in Pydantic error types whose message templates never contain the
# input value. ``value_error`` / ``assertion_error`` carry a custom
# validator's own text, and custom error types are arbitrary, so neither is
# ever shown.
_SAFE_ERROR_TYPES = frozenset(typing.get_args(pydantic_core.core_schema.ErrorType)) - {
    "value_error",
    "assertion_error",
}

_MAX_REPORTED_ERRORS = 10


class ArgsSpec(NamedTuple):
    """How to validate and call one local tool's arguments."""

    model: type[BaseModel]
    # (argument name, field name, positional-only) in signature order.
    params: tuple[tuple[str, str, bool], ...]


class ArgumentProblem(NamedTuple):
    """One value-free validation problem.

    ``path`` is the dotted argument path, ``message`` the constraint text.
    ``value`` is the offending input, kept only so callers can tell a
    redacted placeholder caused it; it is never rendered.
    """

    path: str
    message: str
    value: Any


class ArgumentCoercionError(Exception):
    """Arguments failed lax validation; ``problems`` are value-free."""

    def __init__(self, problems: list[ArgumentProblem], *, unstable: bool = False) -> None:
        super().__init__("; ".join(f"{p.path}: {p.message}" for p in problems))
        self.problems = problems
        self.unstable = unstable


def build_args_spec(fn: Callable[..., Any]) -> ArgsSpec:
    """Build the arguments model for ``fn`` from its signature.

    Field names are synthetic (``a0``, ``a1``, ...) with the argument name as
    the alias, so argument names never collide with ``BaseModel`` attributes.
    ``Optional[T]`` without a default defaults to ``None``, matching the
    generated schema, which does not mark it required.
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    params: list[tuple[str, str, bool]] = []
    for index, (name, param) in enumerate(sig.parameters.items()):
        if name == "self" or param.kind in _VARIADIC_KINDS:
            continue
        field_name = f"a{index}"
        hint = hints[name]
        if param.default is not inspect.Parameter.empty:
            info = Field(default=param.default, alias=name)
        elif _is_optional(_strip_annotated(hint)):
            info = Field(default=None, alias=name)
        else:
            info = Field(alias=name)
        fields[field_name] = (hint, info)
        params.append((name, field_name, param.kind is inspect.Parameter.POSITIONAL_ONLY))
    model = create_model(
        f"{getattr(fn, '__name__', 'tool')}_arguments",
        __config__=ConfigDict(extra="forbid", populate_by_name=False),
        **fields,
    )
    return ArgsSpec(model=model, params=tuple(params))


def attach_args_spec(fn: Callable[..., Any]) -> None:
    """Build and attach ``fn``'s arguments model (called by ``@tool``)."""
    setattr(fn, _ARGS_SPEC_ATTR, build_args_spec(fn))


def get_args_spec(fn: Any) -> ArgsSpec | None:
    """The arguments model attached by ``@tool``, or None (no coercion)."""
    spec = getattr(fn, _ARGS_SPEC_ATTR, None)
    return spec if isinstance(spec, ArgsSpec) else None


def _validate(spec: ArgsSpec, params: dict[str, Any]) -> BaseModel:
    try:
        return spec.model.model_validate(params)
    except ValidationError as exc:
        problems = _problems(exc)
    raise ArgumentCoercionError(problems)


def _dump(instance: BaseModel) -> dict[str, Any]:
    return instance.model_dump(mode="json", by_alias=True, exclude_unset=True)


def normalize_arguments(spec: ArgsSpec, params: dict[str, Any]) -> dict[str, Any]:
    """Validate ``params`` laxly and return their normalised JSON form.

    Raises :class:`ArgumentCoercionError` with value-free problems. The
    normalised form is checked to be a fixed point (validating it again
    gives itself); a validator that keeps changing its output is rejected.
    """
    normalized = _dump(_validate(spec, params))
    if _dump(_validate(spec, normalized)) != normalized:
        raise ArgumentCoercionError(
            [
                ArgumentProblem(
                    "(arguments)",
                    "the tool's validators change the arguments again on re-validation",
                    None,
                )
            ],
            unstable=True,
        )
    return normalized


def build_call(spec: ArgsSpec, normalized: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Positional and keyword arguments rebuilt from normalised JSON.

    Raises :class:`ArgumentCoercionError` when the arguments no longer
    validate (for example after a deploy changed the signature while the
    call waited for approval) or no longer normalise to the same JSON.
    """
    instance = _validate(spec, normalized)
    if _dump(instance) != normalized:
        raise ArgumentCoercionError(
            [
                ArgumentProblem(
                    "(arguments)",
                    "the arguments no longer normalise to the approved form",
                    None,
                )
            ],
            unstable=True,
        )
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, field_name, positional in spec.params:
        value = getattr(instance, field_name)
        if positional:
            args.append(value)
        else:
            kwargs[name] = value
    return args, kwargs


def _path(loc: tuple[int | str, ...]) -> str:
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        elif parts:
            parts.append(f".{item}")
        else:
            parts.append(str(item))
    return "".join(parts) or "(arguments)"


def _problems(exc: ValidationError) -> list[ArgumentProblem]:
    problems: list[ArgumentProblem] = []
    for error in exc.errors(include_url=False)[:_MAX_REPORTED_ERRORS]:
        kind = error.get("type", "")
        if kind in _SAFE_ERROR_TYPES:
            message = str(error.get("msg", "is invalid"))
        elif kind in ("value_error", "assertion_error"):
            message = "failed the tool's own validation"
        else:
            message = f"failed validation ({kind})" if kind else "is invalid"
        problems.append(
            ArgumentProblem(_path(tuple(error.get("loc", ()))), message, error.get("input"))
        )
    return problems


# Same shape as the PII guardrail's ``<<ENTITY_N>>`` placeholders.
_PLACEHOLDER_SHAPE = re.compile(r"<<[A-Z][A-Z_]*_\d+>>")


def contains_placeholder(value: Any, placeholders: Collection[str], _depth: int = 0) -> bool:
    """Whether ``value`` holds one of ``placeholders`` anywhere (bounded walk)."""
    if not placeholders or _depth > 20:
        return False
    if isinstance(value, str):
        return any(match in placeholders for match in _PLACEHOLDER_SHAPE.findall(value))
    if isinstance(value, dict):
        return any(contains_placeholder(v, placeholders, _depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_placeholder(v, placeholders, _depth + 1) for v in value)
    return False
