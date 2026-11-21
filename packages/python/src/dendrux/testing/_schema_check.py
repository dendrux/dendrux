"""A small JSON Schema checker for scripted answers with no Python type to validate with.

Prefer ``FakeModel.provider(name, output_type=Model)``, which validates with the real
Pydantic model. This checker covers a fixed subset: ``type``, ``properties``,
``required``, ``additionalProperties`` (``false`` or a schema), ``items``, ``enum``,
``const``, ``anyOf``, ``allOf``, ``oneOf`` (exactly one), ``$ref``, the numeric bounds,
``minLength``/``maxLength`` and ``minItems``/``maxItems``. A schema that uses any other
validation keyword (``pattern``, ``multipleOf``, ``prefixItems`` and so on) is refused
with :class:`SchemaUnsupportedError`; it is never passed silently. Reasons name the path
and the kind of mismatch, never a value; a key that comes from the script and not from
the schema shows as ``<key>``.
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError

KEY = "<key>"

_UNSUPPORTED = frozenset(
    {
        "pattern",
        "multipleOf",
        "prefixItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "patternProperties",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "contains",
        "minContains",
        "maxContains",
        "unevaluatedProperties",
        "unevaluatedItems",
        "additionalItems",
    }
)


class SchemaUnsupportedError(ValueError):
    """The schema uses a validation keyword this checker does not implement."""

    def __init__(self, keyword: str) -> None:
        super().__init__(f"schema keyword {keyword!r} is not supported")
        self.keyword = keyword


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def first_mismatch(value: object, schema: dict[str, Any]) -> str | None:
    """The first way ``value`` breaks ``schema`` as ``"<path>: <kind>"``, or None.

    Raises :class:`SchemaUnsupportedError` for a keyword the checker does not implement.
    """
    _refuse_unsupported(schema)
    return _check(value, schema, schema, "$")


def _subschemas(node: dict[str, Any]) -> list[Any]:
    found: list[Any] = [node.get("items"), node.get("additionalProperties")]
    for key in ("properties", "$defs", "definitions"):
        children = node.get(key)
        found += list(children.values()) if isinstance(children, dict) else []
    for key in ("anyOf", "oneOf", "allOf"):
        found += list(node.get(key) or ())
    return found


def _refuse_unsupported(node: object) -> None:
    if not isinstance(node, dict):
        return
    for key in node:
        if key in _UNSUPPORTED or (key == "items" and isinstance(node[key], list)):
            raise SchemaUnsupportedError(key)
    for child in _subschemas(node):
        _refuse_unsupported(child)


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node if isinstance(node, dict) else {}


def _type_ok(value: object, name: str) -> bool:
    if name == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if isinstance(value, bool) and name == "number":
        return False
    return isinstance(value, _TYPES.get(name, (object,)))


def _check_combinators(
    value: object, schema: dict[str, Any], root: dict[str, Any], path: str
) -> str | None:
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and not any(_check(value, o, root, path) is None for o in any_of):
        return f"{path}: matches none of the allowed shapes"
    for option in schema.get("allOf") or ():
        found = _check(value, option, root, path)
        if found:
            return found
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        matches = sum(_check(value, o, root, path) is None for o in one_of)
        if matches != 1:
            return f"{path}: matches {matches} of the exclusive shapes, not exactly one"
    return None


def _check_choice(
    value: object, schema: dict[str, Any], root: dict[str, Any], path: str
) -> str | None:
    found = _check_combinators(value, schema, root, path)
    if found:
        return found
    if "const" in schema and value != schema["const"]:
        return f"{path}: not the allowed constant"
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        return f"{path}: not one of the allowed values"
    return None


def _check_number(value: float, schema: dict[str, Any], path: str) -> str | None:
    def bound(key: str) -> float | None:
        found = schema.get(key)
        return found if isinstance(found, (int, float)) and not isinstance(found, bool) else None

    low, high = bound("minimum"), bound("maximum")
    above, below = bound("exclusiveMinimum"), bound("exclusiveMaximum")
    if low is not None and value < low:
        return f"{path}: below the minimum"
    if high is not None and value > high:
        return f"{path}: above the maximum"
    if above is not None and value <= above:
        return f"{path}: not above the exclusive minimum"
    if below is not None and value >= below:
        return f"{path}: not below the exclusive maximum"
    return None


def _check_string(value: str, schema: dict[str, Any], path: str) -> str | None:
    if len(value) < schema.get("minLength", 0):
        return f"{path}: shorter than minLength"
    if len(value) > schema.get("maxLength", len(value)):
        return f"{path}: longer than maxLength"
    return None


def _check_array(
    value: list[Any], schema: dict[str, Any], root: dict[str, Any], path: str
) -> str | None:
    if len(value) < schema.get("minItems", 0):
        return f"{path}: fewer than minItems"
    if len(value) > schema.get("maxItems", len(value)):
        return f"{path}: more than maxItems"
    items = schema.get("items")
    if isinstance(items, dict):
        for index, item in enumerate(value):
            found = _check(item, items, root, f"{path}[{index}]")
            if found:
                return found
    return None


def _check_object(
    value: dict[str, Any], schema: dict[str, Any], root: dict[str, Any], path: str
) -> str | None:
    for name in schema.get("required", ()):
        if name not in value:
            return f"{path}.{name}: missing required property"
    properties = schema.get("properties", {})
    extra = schema.get("additionalProperties")
    for name, item in value.items():
        if name in properties:
            found = _check(item, properties[name], root, f"{path}.{name}")
        elif extra is False:
            found = f"{path}.{KEY}: property not allowed"
        elif isinstance(extra, dict):
            found = _check(item, extra, root, f"{path}.{KEY}")
        else:
            found = None
        if found:
            return found
    return None


def _check_by_kind(
    value: object, schema: dict[str, Any], root: dict[str, Any], path: str
) -> str | None:
    if isinstance(value, dict):
        return _check_object(value, schema, root, path)
    if isinstance(value, list):
        return _check_array(value, schema, root, path)
    if isinstance(value, str):
        return _check_string(value, schema, path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _check_number(value, schema, path)
    return None


def _check(value: object, schema: dict[str, Any], root: dict[str, Any], path: str) -> str | None:
    schema = _resolve(schema, root)
    found = _check_choice(value, schema, root, path)
    if found:
        return found
    expected = schema.get("type")
    names = [expected] if isinstance(expected, str) else expected
    if isinstance(names, list) and not any(_type_ok(value, n) for n in names):
        return f"{path}: wrong type, expected {'/'.join(map(str, names))}"
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
            shown = [f"{self._path(e['loc'])}: {e['type']}" for e in errors[:5]]
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
