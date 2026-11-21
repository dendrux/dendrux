"""A small JSON Schema checker for scripted structured responses.

It understands the subset Pydantic emits: ``type``, ``properties``, ``required``,
``additionalProperties`` (``false`` or a schema), ``items``, ``enum``, ``const``,
``anyOf``/``oneOf`` (exactly one), ``$ref``, ``minimum``/``maximum`` and their
exclusive forms, ``minLength``/``maxLength``/``pattern`` and ``minItems``/``maxItems``.

It checks the JSON Schema only. Pydantic validators that JSON Schema cannot express
are not checked, and Pydantic's lax coercion (``"1"`` or ``true`` for an int) is not
mirrored. Reasons name the path and the kind of mismatch, never a value; a key that
comes from the script and not from the schema shows as ``<key>``.
"""

from __future__ import annotations

import re
from typing import Any

KEY = "<key>"

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def first_mismatch(value: object, schema: dict[str, Any]) -> str | None:
    """The first way ``value`` breaks ``schema`` as ``"<path>: <kind>"``, or None."""
    return _check(value, schema, schema, "$")


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


def _check_choice(
    value: object, schema: dict[str, Any], root: dict[str, Any], path: str
) -> str | None:
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and not any(_check(value, o, root, path) is None for o in any_of):
        return f"{path}: matches none of the allowed shapes"
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        matches = sum(_check(value, o, root, path) is None for o in one_of)
        if matches != 1:
            return f"{path}: matches {matches} of the exclusive shapes, not exactly one"
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
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            matched = re.search(pattern, value)
        except re.error:
            return None  # a pattern Python cannot read is not checked
        if not matched:
            return f"{path}: does not match the pattern"
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
