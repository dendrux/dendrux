"""A small JSON Schema checker for scripted structured responses.

It understands the subset Pydantic emits (``type``, ``properties``, ``required``,
``items``, ``enum``, ``const``, ``anyOf``/``oneOf``, ``$ref``). Reasons name the
schema path and the kind of mismatch only, never a value.
"""

from __future__ import annotations

from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
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
    if isinstance(value, bool) and name in ("integer", "number"):
        return False
    return isinstance(value, _TYPES.get(name, (object,)))


def _check_choice(
    value: object, schema: dict[str, Any], root: dict[str, Any], path: str
) -> str | None:
    """The mismatch for ``anyOf``/``oneOf``/``const``/``enum``, or None."""
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and not any(
            _check(value, o, root, path) is None for o in options
        ):
            return f"{path}: matches none of the allowed shapes"
    if "const" in schema and value != schema["const"]:
        return f"{path}: not the allowed constant"
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        return f"{path}: not one of the allowed values"
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
    if isinstance(value, dict):
        return _check_object(value, schema, root, path)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            found = _check(item, schema["items"], root, f"{path}[{index}]")
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
    for name, item in value.items():
        if name in properties:
            found = _check(item, properties[name], root, f"{path}.{name}")
            if found:
                return found
        elif schema.get("additionalProperties") is False:
            return f"{path}.{name}: property not allowed"
    return None
