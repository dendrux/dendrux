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

import functools
import inspect
import re
from typing import TYPE_CHECKING, Any, NamedTuple, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from dendrux.tool import _VARIADIC_KINDS, _is_optional, _strip_annotated
from dendrux.tools._typing import normalize_tool_hints

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

_ARGS_SPEC_ATTR = "__tool_args_spec__"

# Pydantic error types whose message templates are fixed text or quote only
# the schema (limits, allowed values, type names), never the input. This is an
# allow-list: a type not listed (``value_error`` and ``assertion_error`` carry a
# custom validator's own text; ``union_tag_invalid``, ``uuid_parsing``,
# ``bytes_invalid_encoding`` and ``json_invalid`` quote parts of the input; custom
# and future types are arbitrary) is reported by its type name alone.
_SAFE_ERROR_TYPES = frozenset(
    {
        "missing",
        "extra_forbidden",
        "none_required",
        "frozen_field",
        "frozen_instance",
        "model_type",
        "model_attributes_type",
        "dataclass_type",
        "dataclass_exact_type",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "multiple_of",
        "finite_number",
        "too_short",
        "too_long",
        "string_type",
        "string_sub_type",
        "string_unicode",
        "string_too_short",
        "string_too_long",
        "string_pattern_mismatch",
        "enum",
        "literal_error",
        "dict_type",
        "mapping_type",
        "list_type",
        "tuple_type",
        "set_type",
        "frozen_set_type",
        "set_item_not_hashable",
        "bool_type",
        "bool_parsing",
        "int_type",
        "int_parsing",
        "int_parsing_size",
        "int_from_float",
        "float_type",
        "float_parsing",
        "bytes_type",
        "bytes_too_short",
        "bytes_too_long",
        "date_type",
        "date_parsing",
        "date_from_datetime_parsing",
        "date_from_datetime_inexact",
        "date_past",
        "date_future",
        "time_type",
        "time_parsing",
        "datetime_type",
        "datetime_parsing",
        "datetime_from_date_parsing",
        "datetime_past",
        "datetime_future",
        "timezone_naive",
        "timezone_aware",
        "timezone_offset",
        "time_delta_type",
        "time_delta_parsing",
        "union_tag_not_found",
        "url_type",
        "url_parsing",
        "url_syntax_violation",
        "url_too_long",
        "url_scheme",
        "uuid_type",
        "uuid_version",
        "decimal_type",
        "decimal_parsing",
        "decimal_max_digits",
        "decimal_max_places",
        "decimal_whole_digits",
        "complex_type",
        "is_instance_of",
        "is_subclass_of",
        "callable_type",
    }
)
# Fixed texts for types that are useful to the model but whose template quotes the input.
_FIXED_MESSAGES = {
    "union_tag_invalid": "Input does not match any of the expected tags",
    "uuid_parsing": "Input should be a valid UUID",
    "bytes_invalid_encoding": "Input should be validly encoded",
    "json_invalid": "Input should be valid JSON",
}
# Stands in for a dict key (or any other input-derived name) in an error location.
KEY_PLACEHOLDER = "<key>"
# Locations pydantic adds itself: ``dict[K, V]`` key and value positions.
_STRUCTURAL_LOC = frozenset({"[key]", "[value]"})
_MAX_SCHEMA_NODES = 20_000

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
    hints = normalize_tool_hints(get_type_hints(fn, include_extras=True), sig)
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
        problems = _problems(exc, declared_names(spec))
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


@functools.lru_cache(maxsize=512)
def _model_names(model: type[BaseModel]) -> frozenset[str]:
    names: set[str] = set()
    try:
        schema: Any = model.model_json_schema(by_alias=True, mode="validation")
    except Exception:
        return frozenset()
    stack: list[Any] = [schema]
    seen = 0
    while stack and seen < _MAX_SCHEMA_NODES:
        node = stack.pop()
        seen += 1
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                names.update(k for k in properties if isinstance(k, str))
            defs = node.get("$defs")
            if isinstance(defs, dict):
                names.update(k for k in defs if isinstance(k, str))
            mapping = (node.get("discriminator") or {}).get("mapping")
            if isinstance(mapping, dict):
                names.update(k for k in mapping if isinstance(k, str))
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return frozenset(names)


def declared_names(spec: ArgsSpec) -> frozenset[str]:
    """Names the tool's own signature and types declare: argument, field and type names.

    Only these may appear in an error location; any other name there came
    from the input (a dict key, an extra key) and is replaced by ``<key>``.
    """
    return _model_names(spec.model) | {name for name, _, _ in spec.params}


def _loc_name(item: str, declared: frozenset[str]) -> str:
    return item if item in declared or item in _STRUCTURAL_LOC else KEY_PLACEHOLDER


def _path(loc: tuple[int | str, ...], declared: frozenset[str] = frozenset()) -> str:
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        elif parts:
            parts.append(f".{_loc_name(item, declared)}")
        else:
            parts.append(_loc_name(item, declared))
    return "".join(parts) or "(arguments)"


def _problem_message(kind: str, error: Any) -> str:
    if kind in _SAFE_ERROR_TYPES:
        return str(error.get("msg", "is invalid"))
    if kind in _FIXED_MESSAGES:
        return _FIXED_MESSAGES[kind]
    if kind in ("value_error", "assertion_error"):
        return "failed the tool's own validation"
    return f"failed validation ({kind})" if kind else "is invalid"


def _problems(
    exc: ValidationError, declared: frozenset[str] = frozenset()
) -> list[ArgumentProblem]:
    return [
        ArgumentProblem(
            _path(tuple(error.get("loc", ())), declared),
            _problem_message(error.get("type", ""), error),
            error.get("input"),
        )
        for error in exc.errors(include_url=False)[:_MAX_REPORTED_ERRORS]
    ]


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
