"""Strict-mode schema normalization for OpenAI and Anthropic.

Pydantic's model_json_schema() produces standard JSON Schema. Each vendor's
strict mode accepts a subset of it:

OpenAI strict mode needs ``additionalProperties: false`` on every object and
every property in ``required``; it supports most keywords (pattern, format,
minimum, maximum, $defs, $ref, anyOf, ...), so the pass is lightweight.

Anthropic strict mode needs ``additionalProperties: false`` on every object
and rejects a set of validation keywords with a 400. Those are moved into the
node's ``description`` so the model still sees them; ``required`` is left
alone (Anthropic supports optional properties). Some schemas cannot be made
strict at all (open objects, recursion, tuples, untyped values, over the
request-wide complexity limits); callers send those non-strict.
"""

from __future__ import annotations

import copy
import json
import re
import warnings
from typing import Any


def normalize_for_openai_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON Schema for OpenAI strict mode.

    Returns a copy with:
    - additionalProperties: false on every object
    - All properties added to required
    - $defs preserved (OpenAI supports them)
    """
    schema = copy.deepcopy(schema)
    _normalize_object(schema)
    return schema


def _normalize_object(node: dict[str, Any]) -> None:
    """Recursively normalize a schema node for OpenAI strict mode."""
    if not isinstance(node, dict):
        return

    # Handle object types: add additionalProperties and required
    node_type = node.get("type")
    is_object = node_type == "object" or (isinstance(node_type, list) and "object" in node_type)
    if is_object and "properties" in node:
        node["additionalProperties"] = False
        # Every property required, in declaration order (stable for prompt caching)
        props = node.get("properties", {})
        extra = [name for name in node.get("required", []) if name not in props]
        node["required"] = [*props, *extra]

    # Recurse into properties
    for prop_schema in node.get("properties", {}).values():
        _normalize_object(prop_schema)

    # Recurse into items (arrays)
    items = node.get("items")
    if isinstance(items, dict):
        _normalize_object(items)

    # Recurse into anyOf / oneOf / allOf
    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    _normalize_object(variant)

    # Recurse into $defs
    defs = node.get("$defs")
    if isinstance(defs, dict):
        for def_schema in defs.values():
            _normalize_object(def_schema)


def find_open_object(node: Any, path: str) -> str | None:
    """Return the path of the first free-form object in ``node``, or None.

    A free-form object is ``type: object`` with neither ``properties`` nor a
    ``$ref`` (``dict``, ``dict[str, T]``, a bare object). Strict modes on both
    vendors require objects to declare their properties.
    """
    if not isinstance(node, dict):
        return None
    node_type = node.get("type")
    is_object = node_type == "object" or (isinstance(node_type, list) and "object" in node_type)
    if is_object and "properties" not in node and "$ref" not in node:
        return path
    for key, sub in node.get("properties", {}).items():
        found = find_open_object(sub, f"{path}.{key}")
        if found:
            return found
    children: list[tuple[str, Any]] = [("[]", node.get("items"))]
    for key in ("anyOf", "oneOf", "allOf"):
        children.extend((f".{key}[{i}]", v) for i, v in enumerate(node.get(key) or []))
    for key in ("$defs", "definitions"):
        children.extend((f".{key}.{k}", v) for k, v in (node.get(key) or {}).items())
    for suffix, child in children:
        found = find_open_object(child, f"{path}{suffix}")
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# Anthropic strict mode
# ---------------------------------------------------------------------------

# Request-wide limits across every strict schema in one request (verified live
# 2026-10-18 on claude-sonnet-4-6 and claude-haiku-4-5).
ANTHROPIC_STRICT_MAX_TOOLS = 20
ANTHROPIC_STRICT_MAX_OPTIONAL = 24
ANTHROPIC_STRICT_MAX_UNIONS = 16

_ANTHROPIC_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)

# Keywords strict mode rejects, in the order their notes are appended.
# minItems (0/1 allowed), uniqueItems (false is a no-op), format (supported
# set allowed) and pattern (simple regexes allowed) are conditional.
_ANTHROPIC_STRIPPED = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "minProperties",
    "maxProperties",
    "propertyNames",
    "patternProperties",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "not",
    "format",
    "pattern",
    "discriminator",
)

_NOTE_TEMPLATES = {
    "minimum": "Must be >= {}.",
    "maximum": "Must be <= {}.",
    "exclusiveMinimum": "Must be > {}.",
    "exclusiveMaximum": "Must be < {}.",
    "multipleOf": "Must be a multiple of {}.",
    "minItems": "Must have at least {} items.",
    "maxItems": "Must have at most {} items.",
    "minProperties": "Must have at least {} properties.",
    "maxProperties": "Must have at most {} properties.",
    "pattern": "Must match the regex pattern {}.",
}

# A node carrying none of these has no concrete type; strict mode rejects it.
_TYPED_KEYS = frozenset({"type", "anyOf", "oneOf", "allOf", "$ref", "enum", "const", "properties"})

_RANGE_QUANTIFIER = re.compile(r"\{\d+(,\d*)?\}")
_SAFE_ESCAPE_LETTERS = frozenset("dwsDWSntr")


def normalize_for_anthropic_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON Schema for Anthropic strict mode.

    Returns a copy with ``additionalProperties: false`` on every object that
    has ``properties``, ``oneOf`` rewritten as ``anyOf``, and every keyword
    Anthropic strict mode rejects removed and restated in that node's
    ``description`` (e.g. ``"Must be >= 1."``). ``required`` is unchanged.
    The result is deterministic, so repeated calls give byte-identical JSON.
    """
    schema = copy.deepcopy(schema)
    _anthropic_normalize(schema)
    return schema


def is_anthropic_strict_compatible(schema: dict[str, Any]) -> bool:
    """True when ``schema`` can be sent with Anthropic ``strict: true``."""
    return anthropic_strict_problem(schema) is None


def anthropic_strict_problem(schema: dict[str, Any]) -> str | None:
    """Why ``schema`` cannot be sent with Anthropic ``strict: true``, or None.

    Incompatible: an open object (no ``properties`` and no ``$ref``, or
    ``additionalProperties`` other than false), recursion, an unresolvable or
    external ``$ref``, a tuple (``prefixItems``), a value with no type, an
    enum or const of objects or arrays, or more optional / union-typed parameters than
    one request allows.
    """
    return prepare_anthropic_strict(schema)[1]


def prepare_anthropic_strict(schema: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return ``(normalized_schema, problem)``; send strict only when problem is None."""
    normalized = normalize_for_anthropic_strict(schema)
    problem = _anthropic_open_problem(schema) or _anthropic_structure_problem(normalized)
    if problem is None:
        optional, unions = anthropic_strict_budget(normalized)
        if optional > ANTHROPIC_STRICT_MAX_OPTIONAL:
            problem = f"{optional} optional parameters (limit {ANTHROPIC_STRICT_MAX_OPTIONAL})"
        elif unions > ANTHROPIC_STRICT_MAX_UNIONS:
            problem = f"{unions} union-typed parameters (limit {ANTHROPIC_STRICT_MAX_UNIONS})"
    return normalized, problem


def anthropic_strict_budget(schema: dict[str, Any]) -> tuple[int, int]:
    """Count ``(optional_parameters, union_typed_parameters)`` the way Anthropic does.

    Every object reachable from the root counts its properties missing from
    ``required``; every reachable ``anyOf`` or type array counts as a union.
    A ``$defs`` entry counts once however often it is referenced, and unused
    definitions do not count.
    """
    defs = _anthropic_defs(schema)
    seen: set[str] = set()
    optional = unions = 0
    stack: list[Any] = [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        name = _anthropic_def_name(node.get("$ref"))
        if name is not None and name in defs and name not in seen:
            seen.add(name)
            stack.append(defs[name])
        props = node.get("properties")
        if isinstance(props, dict):
            required = set(node.get("required") or [])
            optional += sum(1 for key in props if key not in required)
        any_of = node.get("anyOf")
        node_type = node.get("type")
        if (isinstance(any_of, list) and len(any_of) > 1) or (
            isinstance(node_type, list) and len(node_type) > 1
        ):
            unions += 1
        stack.extend(_anthropic_children(node))
    return optional, unions


def _anthropic_normalize(node: Any) -> None:
    if not isinstance(node, dict):
        return
    if "oneOf" in node and "anyOf" not in node:
        rebuilt = {("anyOf" if key == "oneOf" else key): value for key, value in node.items()}
        node.clear()
        node.update(rebuilt)

    notes: list[str] = []
    for key in _ANTHROPIC_STRIPPED:
        if key in node and _anthropic_rejects(key, node[key]):
            note = _anthropic_note(key, node.pop(key))
            if note:
                notes.append(note)
    if notes:
        existing = node.get("description")
        joined = " ".join(notes)
        node["description"] = f"{existing}\n\n{joined}" if existing else joined

    if "properties" in node:
        node["additionalProperties"] = False

    items = node.get("items")
    if isinstance(items, dict) and not _TYPED_KEYS & items.keys():
        del node["items"]

    for child in _anthropic_children(node):
        _anthropic_normalize(child)
    for key in ("$defs", "definitions"):
        defs = node.get(key)
        if isinstance(defs, dict):
            for def_schema in defs.values():
                _anthropic_normalize(def_schema)


def _anthropic_children(node: dict[str, Any]) -> list[Any]:
    children: list[Any] = []
    props = node.get("properties")
    if isinstance(props, dict):
        children.extend(props.values())
    items = node.get("items")
    if isinstance(items, dict):
        children.append(items)
    for key in ("prefixItems", "anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            children.extend(variants)
    return children


def _anthropic_rejects(key: str, value: Any) -> bool:
    if key == "minItems":
        return value not in (0, 1)
    if key == "format":
        return value not in _ANTHROPIC_FORMATS
    if key == "pattern":
        return not (isinstance(value, str) and _anthropic_pattern_ok(value))
    return True


def _anthropic_note(key: str, value: Any) -> str | None:
    if key == "discriminator":
        return None
    if key == "uniqueItems":
        return "Items must be unique." if value is True else None
    if key == "format":
        return f"Format: {value}."
    template = _NOTE_TEMPLATES.get(key)
    if template is not None and not isinstance(value, bool):
        return template.format(json.dumps(value))
    return f"Constraint {key}: {json.dumps(value)}."


def _anthropic_pattern_ok(pattern: str) -> bool:
    """Conservative check that Anthropic's strict grammar accepts ``pattern``.

    Rejects group extensions other than ``(?:``, escapes other than
    ``\\d \\w \\s \\D \\W \\S \\n \\t \\r`` and escaped punctuation, and a
    ``{n,m}`` quantifier anywhere in a pattern that uses a negated element
    (``.``, ``[^...]``, ``\\D``, ``\\W``, ``\\S``) — all 400s in live probes.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            re.compile(pattern)
    except re.error:
        return False
    negated = has_range = in_class = False
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\":
            nxt = pattern[i + 1 : i + 2]
            if nxt.isalnum() and nxt not in _SAFE_ESCAPE_LETTERS:
                return False
            negated = negated or nxt in ("D", "W", "S")
            i += 2
            continue
        if in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
            if pattern[i + 1 : i + 2] == "^":
                negated = True
                i += 1
            if pattern[i + 1 : i + 2] == "]":
                i += 1
        elif char == "(" and pattern[i + 1 : i + 2] == "?" and pattern[i + 2 : i + 3] != ":":
            return False
        elif char == ".":
            negated = True
        elif char == "{" and _RANGE_QUANTIFIER.match(pattern, i):
            has_range = True
        i += 1
    return not (negated and has_range)


def _anthropic_open_problem(schema: dict[str, Any]) -> str | None:
    open_path = find_open_object(schema, "schema")
    if open_path is not None:
        return f"{open_path} is a free-form object"
    for path, node in _anthropic_walk(schema, "schema"):
        if node.get("additionalProperties", False) is not False:
            return f"{path} allows additional properties"
    return None


def _anthropic_structure_problem(schema: dict[str, Any]) -> str | None:
    defs = _anthropic_defs(schema)
    edges: dict[str | None, set[str]] = {}
    for path, node, owner in _anthropic_walk_owned(schema, "schema"):
        if not _TYPED_KEYS & node.keys():
            return f"{path} has no type"
        if "prefixItems" in node:
            return f"{path} is a tuple (prefixItems)"
        if "oneOf" in node:
            return f"{path} combines oneOf with anyOf"
        enum = node.get("enum")
        if isinstance(enum, list) and any(isinstance(v, (dict, list)) for v in enum):
            return f"{path} is an enum of objects or arrays"
        if isinstance(node.get("const"), (dict, list)):
            return f"{path} is a const object or array"
        ref = node.get("$ref")
        if ref is None:
            continue
        if ref == "#":
            return f"{path} is recursive (refers to the root)"
        name = _anthropic_def_name(ref)
        if name is None or name not in defs:
            return f"{path} has an unresolvable $ref {ref!r}"
        edges.setdefault(owner, set()).add(name)
    cycle = _anthropic_cycle(edges)
    if cycle is not None:
        return f"schema is recursive ($defs.{cycle})"
    return None


def _anthropic_walk(node: dict[str, Any], path: str) -> list[tuple[str, dict[str, Any]]]:
    return [(p, n) for p, n, _ in _anthropic_walk_owned(node, path)]


def _anthropic_walk_owned(
    root: dict[str, Any], path: str
) -> list[tuple[str, dict[str, Any], str | None]]:
    """Every schema node with its path and owning ``$defs`` name (None = root)."""
    out: list[tuple[str, dict[str, Any], str | None]] = []

    def visit(node: Any, node_path: str, owner: str | None) -> None:
        if not isinstance(node, dict):
            return
        out.append((node_path, node, owner))
        props = node.get("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                visit(sub, f"{node_path}.{key}", owner)
        items = node.get("items")
        if isinstance(items, dict):
            visit(items, f"{node_path}[]", owner)
        for key in ("prefixItems", "anyOf", "oneOf", "allOf"):
            for i, sub in enumerate(node.get(key) or []):
                visit(sub, f"{node_path}.{key}[{i}]", owner)
        for key in ("$defs", "definitions"):
            defs = node.get(key)
            if isinstance(defs, dict):
                for name, sub in defs.items():
                    visit(sub, f"{node_path}.{key}.{name}", name)

    visit(root, path, None)
    return out


def _anthropic_defs(schema: dict[str, Any]) -> dict[str, Any]:
    defs: dict[str, Any] = {}
    for key in ("definitions", "$defs"):
        found = schema.get(key)
        if isinstance(found, dict):
            defs.update(found)
    return defs


def _anthropic_def_name(ref: Any) -> str | None:
    if not isinstance(ref, str):
        return None
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            return ref[len(prefix) :]
    return None


def _anthropic_cycle(edges: dict[str | None, set[str]]) -> str | None:
    """Return a definition name on a reference cycle, or None."""
    done: set[str] = set()
    active: set[str] = set()

    def visit(name: str) -> str | None:
        if name in active:
            return name
        if name in done:
            return None
        active.add(name)
        for nxt in sorted(edges.get(name, ())):
            found = visit(nxt)
            if found is not None:
                return found
        active.discard(name)
        done.add(name)
        return None

    for start in sorted(k for k in edges if k is not None):
        found = visit(start)
        if found is not None:
            return found
    return None
