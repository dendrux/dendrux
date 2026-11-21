"""FakeModel review round 2: real-model validation, honest refusals, fingerprints, ownership."""

from __future__ import annotations

import json
import threading

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.db.models import Base
from dendrux.loops.single import SingleCall
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.testing import FakeModel, FakeModelScriptError, call, say, structured, when
from dendrux.types import LLMResponse, Message, Role, ToolDef

SENTINEL = "ROUND2-SENTINEL-VALUE-55"


def history() -> list[Message]:
    return [Message(role=Role.USER, content="question")]


def predicate(enabled: bool):
    return lambda messages: enabled


class RustPattern(BaseModel):
    value: str = Field(pattern=r"^\p{L}+$")


class MultipleModel(BaseModel):
    value: int = Field(multiple_of=2)


class TypedTuple(BaseModel):
    value: tuple[int, str]


class RustSet(BaseModel):
    value: str = Field(pattern=r"^[a-z&&[^b]]+$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int
    mapping: dict[str, int] = {}


# ---- R1: validate with the real model; refuse what the raw checker cannot judge ----

CASES = [
    (RustPattern, {"value": SENTINEL}),
    (MultipleModel, {"value": 3}),
    (TypedTuple, {"value": ["bad", 1]}),
]


@pytest.mark.parametrize(("model", "value"), CASES)
async def test_bound_output_type_refuses_invalid_value_with_paths_only(model, value):
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(value))
    fake = FakeModel({"a": [structured(value)]})
    provider = fake.provider("a", output_type=model)
    with pytest.raises(FakeModelScriptError) as caught:
        await provider.complete(history(), output_schema=model.model_json_schema())
    assert caught.value.code == "schema_mismatch"
    assert SENTINEL not in str(caught.value) and "$.value" in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.parametrize(("model", "value"), CASES)
async def test_raw_schema_path_refuses_what_it_cannot_judge(model, value):
    fake = FakeModel({"a": [structured(value)]})
    with pytest.raises(FakeModelScriptError) as caught:
        await fake.provider("a").complete(history(), output_schema=model.model_json_schema())
    assert caught.value.code == "unsupported_schema"
    assert SENTINEL not in str(caught.value)


async def test_bound_output_type_accepts_a_rust_only_pattern():
    fake = FakeModel({"a": [structured({"value": "a"})]})
    out = await fake.provider("a", output_type=RustSet).complete(
        history(), output_schema=RustSet.model_json_schema()
    )
    assert json.loads(out.text or "") == {"value": "a"}
    fake.assert_all_used()


async def test_bound_output_type_never_shows_script_keys_or_values():
    value = {"value": SENTINEL, f"{SENTINEL}-K": 1, "mapping": {f"{SENTINEL}-M": "x"}}
    fake = FakeModel({"a": [structured(value)]})
    with pytest.raises(FakeModelScriptError) as caught:
        await fake.provider("a", output_type=Strict).complete(history())
    assert SENTINEL not in str(caught.value) and "<key>" in str(caught.value)


async def test_run_with_bound_output_type_leaks_nothing_to_run_error_or_events(tmp_path):
    fake = FakeModel({"a": [structured({"value": SENTINEL})]})
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'r1.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store = store_for_engine(engine)
    agent = Agent(
        name="a",
        prompt="p",
        provider=fake.provider("a", output_type=RustPattern),
        loop=SingleCall(),
        output_type=RustPattern,
        state_store=store,
    )
    try:
        with pytest.raises(FakeModelScriptError) as caught:
            await agent.run("go", tenant_id="t")
        texts = [str(caught.value), (await store.get_run(caught.value.run_id)).error or ""]
        events = await store.get_run_events(caught.value.run_id)
        texts += [str(e.data) for e in events if e.event_type == "run.error"]
        assert all(SENTINEL not in t for t in texts)
        with pytest.raises(AssertionError, match="schema_mismatch"):
            fake.assert_no_script_errors()
    finally:
        await agent.close()
        await engine.dispose()


# ---- R2: fingerprints ----


def test_fingerprint_distinguishes_captured_predicate_values():
    expected = FakeModel({"a": [when(predicate(False), say("answer"))]})
    other = FakeModel({"a": [when(predicate(True), say("answer"))]})
    with pytest.raises(AssertionError, match="same script"):
        expected.assert_no_script_errors(other)


@pytest.mark.parametrize("field", ["stop_reason", "reasoning"])
async def test_fingerprint_distinguishes_behavioral_raw_fields(field):
    values = {"stop_reason": ("end_turn", "max_tokens"), "reasoning": ("one", "two")}
    left, right = values[field]
    expected = FakeModel({"a": [LLMResponse(text="answer", **{field: left})]})
    other = FakeModel({"a": [LLMResponse(text="answer", **{field: right})]})
    await other.provider("a").complete(history())
    with pytest.raises(AssertionError, match="same script"):
        expected.assert_all_used(other)


async def test_fingerprint_accepts_equal_tool_mappings_in_any_key_order():
    left = {"first": 1, "second": {"x": 2, "y": 3}}
    right = {"second": {"y": 3, "x": 2}, "first": 1}
    expected = FakeModel({"a": [call("t", **left)]})
    other = FakeModel({"a": [call("t", **right)]})
    await other.provider("a").complete(history())
    expected.assert_all_used(other)


# ---- R3: call records ----


@pytest.mark.parametrize("target", ["schema", "message_meta", "tool_schema"])
async def test_call_records_are_independent_on_each_read(target):
    fake = FakeModel({"a": [say("ok")]})
    tool_def = ToolDef(name="t", description="d", parameters={"type": "object"})
    await fake.provider("a").complete(history(), tools=[tool_def], output_schema={"type": "object"})
    first = fake.calls("a")[0]
    if target == "schema":
        first.output_schema["type"] = "string"
        assert fake.calls("a")[0].output_schema["type"] == "object"
    elif target == "message_meta":
        first.messages[0].meta["changed"] = True
        assert "changed" not in fake.calls("a")[0].messages[0].meta
    else:
        first.tools[0].parameters["type"] = "string"
        assert fake.calls("a")[0].tools[0].parameters["type"] == "object"


# ---- R4: predicates stay caller-owned ----


class PurePredicate:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.unrelated_lock = threading.Lock()

    def __call__(self, messages: list[Message]) -> bool:
        return self.answer


@pytest.mark.parametrize("bound", [False, True])
async def test_pure_callable_predicate_remains_caller_owned(bound):
    obj = PurePredicate(True)
    callback = obj.__call__ if bound else obj
    fake = FakeModel({"a": [when(callback, say("ok"))]})
    assert (await fake.provider("a").complete(history())).text == "ok"


# ---- the raw checker: refuse, never pass silently ----


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "pattern": "^a"},
        {"type": "integer", "multipleOf": 2},
        {"type": "array", "prefixItems": [{"type": "integer"}]},
        {"type": "array", "items": [{"type": "integer"}]},
        {"type": "object", "properties": {"a": {"type": "array", "uniqueItems": True}}},
        {"allOf": [{"not": {"type": "string"}}]},
    ],
)
def test_checker_refuses_unsupported_keywords(schema):
    from dendrux.testing._schema_check import SchemaUnsupportedError, first_mismatch

    with pytest.raises(SchemaUnsupportedError):
        first_mismatch(1, schema)


def test_checker_allows_a_property_named_like_a_keyword_and_checks_all_of():
    from dendrux.testing._schema_check import first_mismatch

    schema = {"type": "object", "properties": {"pattern": {"type": "string"}}}
    assert first_mismatch({"pattern": "x"}, schema) is None
    all_of = {"allOf": [{"type": "integer"}, {"minimum": 3}]}
    assert first_mismatch(5, all_of) is None and first_mismatch(2, all_of)
