"""FakeModel review round 4: $ref targets stay inside $defs; const and enum use JSON equality."""

from __future__ import annotations

import pytest

from dendrux.testing import FakeModel, FakeModelScriptError, structured
from dendrux.testing._schema_check import SchemaUnsupportedError, first_mismatch
from dendrux.types import Message, Role

SENTINEL = "ROUND4-SCRIPTED-PAYLOAD-412"


def history() -> list[Message]:
    return [Message(role=Role.USER, content="question")]


# ---- R1 residual: a $ref may only name a preflighted $defs or definitions entry ----


@pytest.mark.parametrize("target", [{"format": "uuid"}, {"unknownKeyword": 1}, {"items": []}])
def test_reference_targets_cannot_escape_the_allowlist(target):
    value = [1] if "items" in target else SENTINEL
    schema = {"default": target, "properties": {"value": {"$ref": "#/default"}}}
    with pytest.raises(SchemaUnsupportedError) as caught:
        first_mismatch({"value": value}, schema)
    assert SENTINEL not in str(caught.value)


def test_a_referenced_sibling_bound_cannot_be_discarded():
    schema = {
        "$defs": {"N": {"type": "integer"}},
        "default": {"$ref": "#/$defs/N", "minimum": 5},
        "properties": {"value": {"$ref": "#/default"}},
    }
    with pytest.raises(SchemaUnsupportedError):
        first_mismatch({"value": 1}, schema)


@pytest.mark.parametrize(
    "target", [{"format": "uuid"}, {"items": []}, {"$ref": "#/$defs/N", "minimum": 5}]
)
async def test_provider_refusal_of_unchecked_reference_targets_is_recorded(target):
    payload = [1] if "items" in target else 1 if "$ref" in target else SENTINEL
    schema = {
        "default": target,
        "$defs": {"N": {"type": "integer"}},
        "properties": {"value": {"$ref": "#/default"}},
    }
    fake = FakeModel({"a": [structured({"value": payload})]})
    with pytest.raises(FakeModelScriptError) as caught:
        await fake.provider("a").complete(history(), output_schema=schema)
    assert caught.value.code == "unsupported_schema"
    assert SENTINEL not in str(caught.value)
    assert fake.script_errors() == [("a", 0, "unsupported_schema")]


@pytest.mark.parametrize(
    "ref", ["#", "#/default", "#/$defs", "#/$defs/A/properties/x", "#/properties/x", "#$defs/A"]
)
def test_only_defs_and_definitions_entries_can_be_referenced(ref):
    schema = {
        "$defs": {"A": {"properties": {"x": {"type": "integer"}}}},
        "default": {"type": "integer"},
        "properties": {"x": {"type": "integer"}, "y": {"$ref": ref}},
    }
    with pytest.raises(SchemaUnsupportedError) as caught:
        first_mismatch({}, schema)
    assert caught.value.keyword == "$ref"


@pytest.mark.parametrize("container", ["$defs", "definitions"])
def test_defs_and_definitions_entries_are_resolved_and_checked(container):
    schema = {container: {"a/b~c": {"type": "integer"}}, "$ref": f"#/{container}/a~1b~0c"}
    assert first_mismatch(1, schema) is None
    assert first_mismatch("x", schema) == "$: wrong type, expected integer"


def test_a_reference_cycle_that_never_reaches_the_value_is_refused():
    schema = {"$defs": {"A": {"allOf": [{"$ref": "#/$defs/A"}]}}, "$ref": "#/$defs/A"}
    with pytest.raises(SchemaUnsupportedError) as caught:
        first_mismatch(1, schema)
    assert caught.value.keyword == "$ref"


# ---- N2: const and enum compare as JSON values ----


@pytest.mark.parametrize(
    ("constraint", "value", "valid"),
    [
        ({"const": True}, 1, False),
        ({"const": False}, 0, False),
        ({"enum": [True]}, 1, False),
        ({"const": 1}, True, False),
        ({"const": [{"x": True}]}, [{"x": 1}], False),
        ({"enum": [{"x": False}]}, {"x": 0}, False),
        ({"oneOf": [{"const": True}, {"const": 1}]}, 1, True),
        ({"oneOf": [{"const": True}, {"const": 1}]}, True, True),
        ({"const": 1}, 1.0, True),
        ({"enum": [[1, {"x": 2.0}]]}, [1.0, {"x": 2}], True),
        ({"const": True}, True, True),
        ({"const": None}, None, True),
        ({"const": "1"}, 1, False),
        ({"const": {"x": 1}}, {"x": 1, "y": 2}, False),
        ({"const": [1]}, [1, 1], False),
    ],
)
def test_enum_and_const_use_json_equality(constraint, value, valid):
    schema = {"properties": {"value": constraint}}
    assert (first_mismatch({"value": value}, schema) is None) is valid
