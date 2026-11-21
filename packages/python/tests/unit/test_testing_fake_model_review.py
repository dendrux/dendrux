"""FakeModel review round 1: value-free errors, script-error tracking, checker, ownership."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.db.models import Base
from dendrux.loops.single import SingleCall
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.testing import FakeModel, FakeModelScriptError, call, calls, say, structured, when
from dendrux.testing._schema_check import first_mismatch
from dendrux.types import LLMResponse, Message, Role, RunStatus, ToolCall, UsageStats


def history(assistant: int = 0, text: str = "question") -> list[Message]:
    return [Message(role=Role.USER, content=text)] + [
        Message(role=Role.ASSISTANT, content="a") for _ in range(assistant)
    ]


class ForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class MappingModel(BaseModel):
    mapping: dict[str, int]


async def _single_call_run(tmp_path, fake, output_type):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'one.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store = store_for_engine(engine)
    agent = Agent(
        name="a",
        prompt="p",
        provider=fake.provider("a"),
        loop=SingleCall(),
        output_type=output_type,
        state_store=store,
    )
    try:
        with pytest.raises(FakeModelScriptError) as caught:
            await agent.run("go", tenant_id="t")
        run_id = caught.value.run_id
        error = (await store.get_run(run_id)).error
        events = await store.get_run_events(run_id)
        event_text = str([e.data for e in events if e.event_type == "run.error"])
        return caught.value, error, event_text
    finally:
        await agent.close()
        await engine.dispose()


def _texts(exc: BaseException) -> str:
    out = []
    while exc is not None:
        out.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    return " | ".join(out)


# ---- 1. value-free errors -------------------------------------------------


@pytest.mark.parametrize(
    ("model", "script", "sentinel"),
    [
        (ForbidModel, {"value": 1, "SENTINEL-KEY-71": "x"}, "SENTINEL-KEY-71"),
        (MappingModel, {"mapping": {"SENTINEL-MAPKEY-72": "SENTINEL-VAL-72"}}, "SENTINEL-"),
    ],
)
async def test_script_payload_text_never_reaches_errors_or_events(
    tmp_path, model, script, sentinel
):
    fake = FakeModel({"a": [structured(script)]})
    exc, error, event_text = await _single_call_run(tmp_path, fake, model)
    assert isinstance(exc, FakeModelScriptError)
    for text in (_texts(exc), error or "", event_text):
        assert sentinel not in text
    assert "<key>" in str(exc)


# ---- 2. script failures are remembered -----------------------------------


async def test_failed_structured_answer_is_not_marked_used():
    fake = FakeModel({"a": [structured({"value": "bad"})]})
    with pytest.raises(FakeModelScriptError):
        await fake.provider("a").complete(history(), output_schema=ForbidModel.model_json_schema())
    with pytest.raises(AssertionError, match=r"a@0"):
        fake.assert_all_used()
    with pytest.raises(AssertionError, match="schema_mismatch"):
        fake.assert_no_script_errors()


async def test_exhaustion_and_unmatched_when_are_recorded_without_text():
    fake = FakeModel({"a": [say("one")], "b": [when(lambda m: False, say("x"))]})
    secret = "SENTINEL-HISTORY-TEXT"
    with pytest.raises(FakeModelScriptError):
        await fake.provider("a").complete(history(1, secret))
    with pytest.raises(FakeModelScriptError):
        await fake.provider("b").complete(history(0, secret))
    with pytest.raises(AssertionError) as info:
        fake.assert_no_script_errors()
    text = str(info.value)
    assert "a@1" in text and "exhausted" in text and "b@0" in text and "no_when_match" in text
    assert secret not in text
    assert fake.script_errors() == [("a", 1, "exhausted"), ("b", 0, "no_when_match")]


def _fanout(tmp_path, fake, children, parent_calls):
    kids = [Agent(name=n, description="d", prompt="p", provider=fake.provider(n)) for n in children]
    return Agent(
        name="parent",
        prompt="p",
        provider=fake.provider("parent"),
        subagents=kids,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'fan.db'}",
    )


async def _go(parent, streaming):
    if streaming:
        async with parent.stream("go", tenant_id="t") as source:
            async for _ in source:
                pass
        return source.result
    return await parent.run("go", tenant_id="t")


@pytest.mark.parametrize("streaming", [False, True])
async def test_fanout_empty_child_script_fails_all_used(tmp_path, streaming):
    fake = FakeModel(
        {
            "parent": [calls(call("broken", task="x"), call("good", task="y")), say("done")],
            "broken": [],
            "good": [say("ok")],
        }
    )
    parent = _fanout(tmp_path, fake, ["broken", "good"], None)
    result = await _go(parent, streaming)
    assert result.status == RunStatus.SUCCESS  # runtime child isolation is unchanged
    with pytest.raises(AssertionError, match=r"broken@0"):
        fake.assert_all_used()
    with pytest.raises(AssertionError, match="exhausted"):
        fake.assert_no_script_errors()


@pytest.mark.parametrize("streaming", [False, True])
async def test_fanout_missing_when_branch_fails_all_used(tmp_path, streaming):
    fake = FakeModel(
        {
            "parent": [
                calls(call("helper", task="alpha"), call("helper", task="beta")),
                say("done"),
            ],
            "helper": [when(lambda m: "alpha" in str(m[-1].content), say("ok"))],
        }
    )
    parent = _fanout(tmp_path, fake, ["helper"], None)
    result = await _go(parent, streaming)
    assert result.status == RunStatus.SUCCESS
    with pytest.raises(AssertionError, match=r"helper@0"):
        fake.assert_all_used()


async def test_clean_run_passes_both_assertions():
    fake = FakeModel({"a": [say("x")]})
    await fake.provider("a").complete(history())
    fake.assert_no_script_errors()
    fake.assert_all_used()


# ---- 3. schema checker -----------------------------------------------------


class Bounds(BaseModel):
    gt: int = Field(default=1, gt=0)
    ge: int = Field(default=1, ge=1)
    lt: int = Field(default=1, lt=5)
    le: int = Field(default=1, le=5)
    text: str = Field(default="okay", min_length=2, max_length=5)
    items: list[int] = Field(default_factory=lambda: [1], min_length=1, max_length=2)
    mapping: dict[str, int] = Field(default_factory=dict)
    number: float = Field(default=1.0, gt=0.5)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"gt": 1},
        {"gt": 0},
        {"gt": -1},
        {"ge": 0},
        {"ge": 1},
        {"lt": 5},
        {"lt": 4},
        {"le": 6},
        {"le": 5},
        {"text": "x"},
        {"text": "okayok"},
        {"text": "no"},
        {"text": "ok"},
        {"items": []},
        {"items": [1, 2, 3]},
        {"items": [1, 2]},
        {"mapping": {"k": "v"}},
        {"mapping": {"k": 2}},
        {"number": 0.5},
        {"number": 0.6},
        {"gt": 3.0},
        {"gt": 3.5},
        {"gt": True},
    ],
)
def test_checker_agrees_with_pydantic_on_constraints(change):
    schema = Bounds.model_json_schema()
    text = json.dumps(change)
    try:
        Bounds.model_validate_json(text)
        expected = True
    except ValidationError:
        expected = False
    if change.get("gt") is True:  # lax pydantic coerces; JSON Schema does not (documented)
        expected = False
    assert (first_mismatch(change, schema) is None) == expected


def test_integral_float_is_an_integer_and_one_of_means_exactly_one():
    assert first_mismatch({"value": 1.0}, {"properties": {"value": {"type": "integer"}}}) is None
    assert first_mismatch(1.5, {"type": "integer"})
    one_of = {"oneOf": [{"type": "integer"}, {"type": "number"}]}
    assert first_mismatch(1, one_of)  # matches both
    assert first_mismatch(1.5, one_of) is None
    assert first_mismatch("s", one_of)


def test_additional_properties_schema_and_extra_key_placeholder():
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    problem = first_mismatch({"SENTINEL-K": "v"}, schema)
    assert problem and "SENTINEL" not in problem and "<key>" in problem
    declared = {"type": "object", "properties": {"known": {"type": "integer"}}}
    assert "$.known" in first_mismatch({"known": "v"}, declared)
    assert "[1]" in first_mismatch([1, "x"], {"type": "array", "items": {"type": "integer"}})


# ---- 4. aliases -------------------------------------------------------------


class AliasInner(BaseModel):
    n: int = Field(alias="wire_n")


class AliasOuter(BaseModel):
    value: int = Field(alias="wire_value")
    inner: AliasInner
    many: list[AliasInner]


@pytest.mark.parametrize("model", ["plain", "nested"])
async def test_structured_uses_the_schema_keys(model):
    if model == "plain":
        obj, cls = AliasInner(wire_n=2), AliasInner
    else:
        obj = AliasOuter(wire_value=1, inner=AliasInner(wire_n=2), many=[AliasInner(wire_n=3)])
        cls = AliasOuter
    out = (
        await FakeModel({"a": [structured(obj)]})
        .provider("a")
        .complete(history(), output_schema=cls.model_json_schema())
    )
    assert cls.model_validate_json(out.text or "") == obj


# ---- 5. ownership ----------------------------------------------------------


async def test_returned_payload_mutation_cannot_change_later_answers():
    script = {"a": [call("edit", payload={"items": [1]}), call("edit", payload={"items": [1]})]}
    pristine = copy.deepcopy(script)
    fake = FakeModel(script)
    p = fake.provider("a")
    first = await p.complete(history())
    first.tool_calls[0].params["payload"]["items"].append(2)
    script["a"][0].calls[0][1]["payload"]["items"].append(3)  # caller edits own script later
    again = await FakeModel(pristine).provider("a").complete(history())
    repeated = await p.complete(history())  # same slot, same history
    assert repeated.tool_calls[0].params == again.tool_calls[0].params
    assert repeated.tool_calls[0].params["payload"]["items"] == [1]


async def test_raw_response_usage_and_call_records_are_copies():
    raw = LLMResponse(
        text="ok", tool_calls=[ToolCall(name="t", params={"x": [1]})], usage=UsageStats()
    )
    fake = FakeModel({"a": [raw]})
    msgs = history()
    schema = {"type": "object"}
    first = await fake.provider("a").complete(msgs, output_schema=schema)
    first.usage.total_tokens = 99
    first.tool_calls[0].params["x"].append(2)
    record = fake.calls("a")[0]
    record.output_schema["type"] = "string"
    record.messages[0].meta["changed"] = True
    assert raw.usage.total_tokens == 0 and raw.tool_calls[0].params == {"x": [1]}
    assert schema["type"] == "object" and "changed" not in msgs[0].meta
    again = await fake.provider("a").complete(msgs)
    assert again.usage.total_tokens == 0 and again.tool_calls[0].params == {"x": [1]}


# ---- 6. assert_all_used compatibility ---------------------------------------


async def test_assert_all_used_refuses_a_different_script():
    expected = FakeModel({"a": [say("SENTINEL-EXPECTED")]})
    unrelated = FakeModel({"a": [say("SENTINEL-OTHER")]})
    await unrelated.provider("a").complete(history())
    with pytest.raises(AssertionError) as info:
        expected.assert_all_used(unrelated)
    assert "SENTINEL" not in str(info.value) and "same script" in str(info.value)


async def test_assert_all_used_accepts_a_rebuilt_equal_script():
    def script():
        return {"a": [say("x"), when(lambda m: True, call("t", n=1))]}

    first, second = FakeModel(script()), FakeModel(script())
    await first.provider("a").complete(history())
    await second.provider("a").complete(history(1))
    second.assert_all_used(first)
