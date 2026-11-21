"""A small JSON Schema checker for scripted answers with no Python type to validate with.

Prefer ``FakeModel.provider(name, output_type=Model)``, which validates with the real
Pydantic model. This checker works from an allowlist. It checks ``type``, ``properties``,
``required``, ``additionalProperties``, ``items`` (one schema), ``enum``, ``const``,
``anyOf``, ``allOf``, ``oneOf`` (exactly one), a local ``$ref`` (beside annotations only),
the numeric bounds, ``minLength``/``maxLength``, ``minItems``/``maxItems`` and the boolean
schemas ``true`` and ``false``. It ignores annotations such as ``title`` and ``default``.
Any other keyword or form (``format``, ``pattern``, an unknown keyword, an external or
missing ``$ref`` and so on) is refused with :class:`SchemaUnsupportedError`; it is never
passed silently. Reasons name the path and the kind of mismatch, never a value; a key
that comes from the script and not from the schema shows as ``<key>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

from pydantic import TypeAdapter, ValidationError
from pydantic_core.core_schema import ErrorType

if TYPE_CHECKING:
    from collections.abc import Callable

KEY = "<key>"

_TYPES: dict[str, tuple[type, ...]] = {
    "integer": (int, float),
    "string": (str,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}

_ANNOTATIONS = frozenset(
    {
        "$schema",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "discriminator",
    }
)
_CONTAINERS = ("$defs", "definitions")


def _is_schema(value: object) -> bool:
    return isinstance(value, (dict, bool))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_type(value: object) -> bool:
    names = value if isinstance(value, list) else [value]
    return bool(names) and all(isinstance(n, str) and n in _TYPES for n in names)


def _is_schema_map(value: object) -> bool:
    return isinstance(value, dict) and all(_is_schema(v) for v in value.values())


def _is_schema_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_schema(v) for v in value)


_FORMS: dict[str, Callable[[object], bool]] = {
    "type": _is_type,
    "properties": _is_schema_map,
    "$defs": _is_schema_map,
    "definitions": _is_schema_map,
    "required": lambda v: isinstance(v, list) and all(isinstance(n, str) for n in v),
    "additionalProperties": _is_schema,
    "items": _is_schema,
    "enum": lambda v: isinstance(v, list),
    "const": lambda v: True,
    "anyOf": _is_schema_list,
    "allOf": _is_schema_list,
    "oneOf": _is_schema_list,
    "$ref": lambda v: isinstance(v, str) and (v == "#" or v.startswith("#/")),
    "minimum": _is_number,
    "maximum": _is_number,
    "exclusiveMinimum": _is_number,
    "exclusiveMaximum": _is_number,
    "minLength": _is_count,
    "maxLength": _is_count,
    "minItems": _is_count,
    "maxItems": _is_count,
}

_BUILTIN_ERRORS = frozenset(get_args(ErrorType))

Schema = dict[str, Any] | bool


class SchemaUnsupportedError(ValueError):
    """The schema uses a keyword or form this checker does not implement."""

    def __init__(self, keyword: str) -> None:
        super().__init__(f"schema keyword {keyword!r} is not supported")
        self.keyword = keyword


def first_mismatch(value: object, schema: Schema) -> str | None:
    """The first way ``value`` breaks ``schema`` as ``"<path>: <kind>"``, or None.

    Raises :class:`SchemaUnsupportedError` for a keyword or form the checker does not
    implement, anywhere in the schema, before any value is checked; a ``$ref`` loop that
    never reaches the value is refused the same way.
    """
    _refuse_unsupported(schema, schema)
    try:
        return _check(value, schema, schema, "$")
    except RecursionError:
        raise SchemaUnsupportedError("$ref") from None


def _subschemas(node: dict[str, Any]) -> list[Schema]:
    found = [node[key] for key in ("items", "additionalProperties") if key in node]
    for key in ("properties", *_CONTAINERS):
        found += list(node.get(key, {}).values())
    for key in ("anyOf", "oneOf", "allOf"):
        found += node.get(key, [])
    return found


def _refuse_unsupported(node: Schema, root: Schema) -> None:
    if isinstance(node, bool):
        return
    for key, value in node.items():
        if key not in _ANNOTATIONS and not (key in _FORMS and _FORMS[key](value)):
            raise SchemaUnsupportedError(key)
    if "$ref" in node:
        _lookup(node["$ref"], root)
        if set(node) - _ANNOTATIONS - {"$ref", *_CONTAINERS}:
            raise SchemaUnsupportedError("$ref")
    for child in _subschemas(node):
        _refuse_unsupported(child, root)


def _lookup(ref: str, root: Schema) -> Schema:
    node: Any = root
    for part in ref[2:].split("/") if ref != "#" else ():
        key = part.replace("~1", "/").replace("~0", "~")
        node = node.get(key) if isinstance(node, dict) else None
    if isinstance(node, (dict, bool)):
        return node
    raise SchemaUnsupportedError("$ref")


def _type_ok(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return name == "boolean"
    if name == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    return isinstance(value, _TYPES[name])


def _check_combinators(
    value: object, schema: dict[str, Any], root: Schema, path: str
) -> str | None:
    any_of = schema.get("anyOf")
    if any_of and not any(_check(value, o, root, path) is None for o in any_of):
        return f"{path}: matches none of the allowed shapes"
    for option in schema.get("allOf", ()):
        found = _check(value, option, root, path)
        if found:
            return found
    one_of = schema.get("oneOf")
    if one_of:
        matches = sum(_check(value, o, root, path) is None for o in one_of)
        if matches != 1:
            return f"{path}: matches {matches} of the exclusive shapes, not exactly one"
    return None


def _check_choice(value: object, schema: dict[str, Any], root: Schema, path: str) -> str | None:
    found = _check_combinators(value, schema, root, path)
    if found:
        return found
    if "const" in schema and value != schema["const"]:
        return f"{path}: not the allowed constant"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: not one of the allowed values"
    return None


def _check_number(value: float, schema: dict[str, Any], path: str) -> str | None:
    if value < schema.get("minimum", value):
        return f"{path}: below the minimum"
    if value > schema.get("maximum", value):
        return f"{path}: above the maximum"
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        return f"{path}: not above the exclusive minimum"
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        return f"{path}: not below the exclusive maximum"
    return None


def _check_string(value: str, schema: dict[str, Any], path: str) -> str | None:
    if len(value) < schema.get("minLength", 0):
        return f"{path}: shorter than minLength"
    if len(value) > schema.get("maxLength", len(value)):
        return f"{path}: longer than maxLength"
    return None


def _check_array(value: list[Any], schema: dict[str, Any], root: Schema, path: str) -> str | None:
    if len(value) < schema.get("minItems", 0):
        return f"{path}: fewer than minItems"
    if len(value) > schema.get("maxItems", len(value)):
        return f"{path}: more than maxItems"
    items = schema.get("items", True)
    for index, item in enumerate(value):
        found = _check(item, items, root, f"{path}[{index}]")
        if found:
            return found
    return None


def _check_object(
    value: dict[str, Any], schema: dict[str, Any], root: Schema, path: str
) -> str | None:
    for name in schema.get("required", ()):
        if name not in value:
            return f"{path}.{name}: missing required property"
    properties = schema.get("properties", {})
    extra = schema.get("additionalProperties", True)
    for name, item in value.items():
        if name in properties:
            found = _check(item, properties[name], root, f"{path}.{name}")
        else:
            found = _check(item, extra, root, f"{path}.{KEY}")
        if found:
            return found
    return None


def _check_by_kind(value: object, schema: dict[str, Any], root: Schema, path: str) -> str | None:
    if isinstance(value, dict):
        return _check_object(value, schema, root, path)
    if isinstance(value, list):
        return _check_array(value, schema, root, path)
    if isinstance(value, str):
        return _check_string(value, schema, path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _check_number(value, schema, path)
    return None


def _check(value: object, schema: Schema, root: Schema, path: str) -> str | None:
    if isinstance(schema, bool):
        return None if schema else f"{path}: not allowed"
    if "$ref" in schema:
        return _check(value, _lookup(schema["$ref"], root), root, path)
    found = _check_choice(value, schema, root, path)
    if found:
        return found
    expected = schema.get("type")
    names = [expected] if isinstance(expected, str) else expected
    if names and not any(_type_ok(value, n) for n in names):
        return f"{path}: wrong type, expected {'/'.join(names)}"
    return _check_by_kind(value, schema, root, path)


def _declared_names(node: object, names: set[str]) -> set[str]:
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            names.update(k for k in props if isinstance(k, str))
        for child in node.values():
            _declared_names(child, names)
    elif isinstance(node, list):
        for child in node:
            _declared_names(child, names)
    return names


def _kind(error_type: str) -> str:
    """A built-in Pydantic error type, or ``custom_error``: custom types can carry data."""
    return error_type if error_type in _BUILTIN_ERRORS else "custom_error"


class TypeCheck:
    """Validate scripted JSON with the real Pydantic type, reporting paths and error types only."""

    def __init__(self, output_type: Any) -> None:
        self._adapter: TypeAdapter[Any] = TypeAdapter(output_type)
        self._names = _declared_names(self._adapter.json_schema(), set())
        self.label = str(getattr(output_type, "__name__", output_type))

    def problem(self, text: str | None) -> str | None:
        """``"<path>: <error type>; ..."`` for the first errors, or None when valid."""
        try:
            self._adapter.validate_json(text or "")
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_url=False)
            shown = [f"{self._path(e['loc'])}: {_kind(e['type'])}" for e in errors[:5]]
            extra = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
            return "; ".join(shown) + extra
        return None

    def _path(self, loc: tuple[int | str, ...]) -> str:
        parts = []
        for item in loc:
            if isinstance(item, int):
                parts.append(f"[{item}]")
            else:
                parts.append(f".{item if item in self._names else KEY}")
        return "$" + "".join(parts)
