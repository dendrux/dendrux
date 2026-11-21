"""FakeModel review round 5: a $ref names a $defs entry by a plain name, with no decoding."""

from __future__ import annotations

from typing import Generic, TypeVar

import pytest
from pydantic import BaseModel

from dendrux.testing import FakeModel, FakeModelScriptError, structured
from dendrux.testing._schema_check import SchemaUnsupportedError, first_mismatch
from dendrux.types import Message, Role

SENTINEL = "ROUND5-SCRIPTED-VALUE-533"

T = TypeVar("T")


def history() -> list[Message]:
    return [Message(role=Role.USER, content="question")]


# Each name is a real key; a validator that decodes the reference would pick another one.
ENCODED = [
    ("A", "%41"),
    ("a b", "a%20b"),
    ("a/b", "a%7E1b"),
    ("a~b", "a%7E0b"),
    ("é", "%C3%A9"),
    ("%41", "%2541"),
]
ESCAPED_OR_ODD = ["a~1b", "a~0b", "x y", "é", "a%b", "a+b", "a:b", "a#b"]


@pytest.mark.parametrize("container", ["$defs", "definitions"])
@pytest.mark.parametrize(("actual", "encoded"), ENCODED)
def test_percent_encoded_refs_are_refused(container, actual, encoded):
    schema = {
        container: {actual: {"minimum": 5}, encoded: True},
        "$ref": f"#/{container}/{encoded}",
    }
    with pytest.raises(SchemaUnsupportedError) as caught:
        first_mismatch(1, schema)
    assert caught.value.keyword == "$ref"


@pytest.mark.parametrize("container", ["$defs", "definitions"])
@pytest.mark.parametrize(("actual", "encoded"), ENCODED[:3])
async def test_provider_records_a_value_free_refusal_for_encoded_refs(container, actual, encoded):
    schema = {
        container: {actual: {"type": "string", "minLength": 100}, encoded: True},
        "properties": {"value": {"$ref": f"#/{container}/{encoded}"}},
    }
    fake = FakeModel({"a": [structured({"value": SENTINEL})]})
    with pytest.raises(FakeModelScriptError) as caught:
        await fake.provider("a").complete(history(), output_schema=schema)
    assert caught.value.code == "unsupported_schema"
    assert "'$ref'" in str(caught.value) and SENTINEL not in str(caught.value)
    assert fake.script_errors() == [("a", 0, "unsupported_schema")]


@pytest.mark.parametrize("name", ESCAPED_OR_ODD)
def test_names_outside_the_plain_set_are_refused_even_when_the_key_exists(name):
    schema = {"$defs": {name: {"type": "integer"}}, "$ref": f"#/$defs/{name}"}
    with pytest.raises(SchemaUnsupportedError) as caught:
        first_mismatch(1, schema)
    assert caught.value.keyword == "$ref"


class Inner(BaseModel):
    n: int


class Outer(BaseModel):
    inner: Inner
    items: list[Inner]


class Box(BaseModel, Generic[T]):
    item: T


class Holder(BaseModel):
    boxed: Box[Inner]
    numbers: Box[int]


@pytest.mark.parametrize("model", [Outer, Holder])
def test_pydantic_def_names_fit_the_plain_set(model):
    schema = model.model_json_schema()
    good = model.model_validate(
        {"inner": {"n": 1}, "items": [{"n": 2}]}
        if model is Outer
        else {"boxed": {"item": {"n": 1}}, "numbers": {"item": 2}}
    ).model_dump(mode="json")
    assert first_mismatch(good, schema) is None
    bad = (
        {"inner": {"n": "x"}, "items": []} if model is Outer else {**good, "numbers": {"item": "x"}}
    )
    assert first_mismatch(bad, schema) is not None
