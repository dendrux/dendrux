"""FakeModel review round 3: allowlisted schemas, untrusted error codes, predicate identity."""

from __future__ import annotations

import json
import traceback
from datetime import date  # noqa: TC003 - pydantic reads field types at runtime
from uuid import UUID  # noqa: TC003 - pydantic reads field types at runtime

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import PydanticCustomError
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.db.models import Base
from dendrux.loops.single import SingleCall
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.testing import FakeModel, FakeModelScriptError, structured
from dendrux.testing._schema_check import SchemaUnsupportedError, first_mismatch
from dendrux.types import Message, Role

SENTINEL = "ROUND3-SCRIPTED-VALUE-311"


def history() -> list[Message]:
    return [Message(role=Role.USER, content="question")]


class RustPattern(BaseModel):
    value: str = Field(pattern=r"^\p{L}+$")


class UuidAnswer(BaseModel):
    value: UUID


class DateAnswer(BaseModel):
    value: date


class StrictAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int
    mapping: dict[str, int] = {}


class ValidatorAnswer(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def refuse(cls, value: str) -> str:
        raise ValueError(value)


class CustomCodeAnswer(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def refuse(cls, value: str) -> str:
        # Pydantic accepts any string as a custom error type, including scripted data.
        raise PydanticCustomError(value, "rejected")


# ---- R1: the raw checker refuses anything outside its allowlist ----

UNSUPPORTED = [
    ({"type": "string", "format": "uuid"}, SENTINEL, "format"),
    ({"type": "string", "format": "date"}, SENTINEL, "format"),
    ({"type": "string", "unknownValidationKeyword": 3}, SENTINEL, "unknownValidationKeyword"),
    ({"type": "object", "dependencies": {"x": ["y"]}}, {"x": 1}, "dependencies"),
    ({"$dynamicRef": "#schema"}, 1, "$dynamicRef"),
    ({"$recursiveRef": "#"}, 1, "$recursiveRef"),
    ({"$ref": "https://example.invalid/schema"}, 1, "$ref"),
    ({"$ref": "#/$defs/missing"}, 1, "$ref"),
    ({"$defs": {"B": {"type": "integer"}}, "$ref": "#/$defs/B", "minimum": 5}, 1, "$ref"),
    ({"type": "object", "properties": {"x": {"format": "uuid"}}}, {"x": SENTINEL}, "format"),
    ({"type": "uuid"}, 1, "type"),
    ({"minimum": "5"}, 1, "minimum"),
    ({"exclusiveMinimum": True}, 1, "exclusiveMinimum"),
    ({"type": "string", "minLength": "2"}, "x", "minLength"),
    ({"required": "x"}, {}, "required"),
    ({"anyOf": {"type": "string"}}, 1, "anyOf"),
    ({"$ref": "#"}, 1, "$ref"),
    ({"allOf": [{"$ref": "#"}]}, 1, "$ref"),
]


@pytest.mark.parametrize(("schema", "value", "keyword"), UNSUPPORTED)
def test_raw_checker_refuses_what_is_not_on_its_allowlist(schema, value, keyword):
    with pytest.raises(SchemaUnsupportedError) as caught:
        first_mismatch(value, schema)
    assert caught.value.keyword == keyword
    assert SENTINEL not in str(caught.value)


@pytest.mark.parametrize(
    ("schema", "value", "ok"),
    [
        ({"type": "array", "items": False}, [1], False),
        ({"type": "array", "items": False}, [], True),
        ({"type": "object", "properties": {"x": False}}, {"x": 1}, False),
        ({"type": "object", "properties": {"x": False}}, {}, True),
        ({"type": "object", "properties": {"x": True}}, {"x": [1]}, True),
        ({"type": "object", "additionalProperties": True}, {"k": 1}, True),
        ({"anyOf": [False, {"type": "integer"}]}, 1, True),
        (False, 1, False),
        (True, 1, True),
    ],
)
def test_boolean_schemas_allow_everything_or_nothing(schema, value, ok):
    assert (first_mismatch(value, schema) is None) is ok


def test_annotations_refs_and_keyword_named_properties_stay_valid():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$comment": "c",
        "type": "object",
        "title": "T",
        "description": "D",
        "properties": {
            "pattern": {"type": "string", "default": "ok", "examples": ["a"], "deprecated": True},
            "inner": {"$ref": "#/$defs/Inner", "description": "an annotation beside $ref"},
        },
        "required": ["pattern"],
        "$defs": {"Inner": {"type": "integer", "readOnly": True, "writeOnly": False}},
    }
    assert first_mismatch({"pattern": "yes", "inner": 2}, schema) is None
    assert (
        first_mismatch({"pattern": "yes", "inner": "x"}, schema)
        == "$.inner: wrong type, expected integer"
    )


class Node(BaseModel):
    name: str
    children: list[Node] = []


def test_recursive_local_refs_are_supported():
    schema = Node.model_json_schema()
    assert first_mismatch({"name": "a", "children": [{"name": "b"}]}, schema) is None
    assert first_mismatch({"name": "a", "children": [{"name": 1}]}, schema)


@pytest.mark.parametrize("model", [UuidAnswer, DateAnswer])
async def test_raw_provider_refuses_pydantic_formats(model):
    value = {"value": SENTINEL}
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(value))
    fake = FakeModel({"a": [structured(value)]})
    with pytest.raises(FakeModelScriptError) as caught:
        await fake.provider("a").complete(history(), output_schema=model.model_json_schema())
    assert caught.value.code == "unsupported_schema"
    assert "'format'" in str(caught.value) and SENTINEL not in str(caught.value)
    assert fake.script_errors() == [("a", 0, "unsupported_schema")]


async def test_raw_provider_refuses_a_false_property_without_crashing():
    fake = FakeModel({"a": [structured({"x": SENTINEL})]})
    schema = {"type": "object", "properties": {"x": False}}
    with pytest.raises(FakeModelScriptError) as caught:
        await fake.provider("a").complete(history(), output_schema=schema)
    assert caught.value.code == "schema_mismatch" and SENTINEL not in str(caught.value)


# ---- N1: Pydantic error kinds come from a fixed allowlist ----


def _chain_texts(error: BaseException) -> list[str]:
    texts = [str(error), repr(error), "".join(traceback.format_exception_only(error))]
    current = error.__cause__ or error.__context__
    while current is not None:
        texts.append(str(current))
        current = current.__cause__ or current.__context__
    return texts


@pytest.mark.parametrize(
    "model",
    [RustPattern, UuidAnswer, DateAnswer, StrictAnswer, ValidatorAnswer, CustomCodeAnswer],
)
async def test_bound_provider_exception_and_chain_are_value_free(model):
    payload: dict[str, object] = {"value": SENTINEL}
    if model is StrictAnswer:
        payload.update({SENTINEL: "x", "mapping": {SENTINEL: "x"}})
    fake = FakeModel({"a": [structured(payload)]})
    provider = fake.provider("a", output_type=model)
    with pytest.raises(FakeModelScriptError) as caught:
        await provider.complete(history(), output_schema=model.model_json_schema())
    assert caught.value.code == "schema_mismatch"
    texts = _chain_texts(caught.value)
    assert all(SENTINEL not in text for text in texts), texts
    assert fake.script_errors() == [("a", 0, "schema_mismatch")]


async def test_custom_error_codes_show_as_a_fixed_kind():
    fake = FakeModel({"a": [structured({"value": SENTINEL})]})
    with pytest.raises(FakeModelScriptError) as caught:
        await fake.provider("a", output_type=CustomCodeAnswer).complete(history())
    assert str(caught.value).endswith("$.value: custom_error")


async def test_built_in_error_kinds_are_still_shown():
    fake = FakeModel({"a": [structured({"value": SENTINEL})]})
    with pytest.raises(FakeModelScriptError) as caught:
        await fake.provider("a", output_type=UuidAnswer).complete(history())
    assert str(caught.value).endswith("$.value: uuid_parsing")


# Not here: CustomCodeAnswer with an unbound provider. Its schema is a plain string, so
# the script is valid for it; the leak then comes from SingleCall's own validation
# message (dendrux/llm/structured.py), outside FakeModel. Bind output_type instead.
@pytest.mark.parametrize(
    ("model", "streaming", "bound"),
    [
        (UuidAnswer, False, False),
        (UuidAnswer, False, True),
        (UuidAnswer, True, True),
        (CustomCodeAnswer, False, True),
        (CustomCodeAnswer, True, True),
    ],
)
async def test_saved_run_errors_and_all_events_are_value_free(tmp_path, model, streaming, bound):
    fake = FakeModel({"a": [structured({"value": SENTINEL})]})
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'case.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store = store_for_engine(engine)
    provider = fake.provider("a", output_type=model) if bound else fake.provider("a")
    agent = Agent(
        name="a",
        prompt="p",
        provider=provider,
        loop=SingleCall(),
        output_type=None if streaming else model,
        state_store=store,
    )
    errors: list[str] = []
    stream_texts: list[str] = []
    try:
        if streaming:
            async with agent.stream("go", tenant_id="t") as stream:
                stream_texts += [str(event) async for event in stream]
            run_id = stream.result.run_id
            errors.append(stream.result.error or "")
        else:
            with pytest.raises(Exception) as caught:
                await agent.run("go", tenant_id="t")
            run_id = caught.value.run_id
            errors.append(str(caught.value))
        saved = await store.get_run(run_id)
        events = await store.get_run_events(run_id)
        assert saved.error
        assert any(e.event_type == "run.error" for e in events)
        surfaces = {
            "exception_or_result": errors,
            "saved_error": [saved.error],
            "events": [str(e.data) for e in events],
            "stream_events": stream_texts,
        }
        leaked = [name for name, texts in surfaces.items() if any(SENTINEL in t for t in texts)]
        assert not leaked
    finally:
        await agent.close()
        await engine.dispose()
