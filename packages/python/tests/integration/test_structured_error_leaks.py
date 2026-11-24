"""Rejected structured answers stay off persisted rows, formatted logs, child runs and spans."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

from dendrux import Agent, SingleCall, tool
from dendrux.llm.mock import MockLLM
from dendrux.llm.structured import structured_complete
from dendrux.loops.base import BaseNotifier
from dendrux.types import (
    LLMResponse,
    Message,
    ProviderCapabilities,
    Role,
    RunStatus,
    StructuredOutputValidationError,
    ToolCall,
    ToolResult,
    UsageStats,
)

S = "MODEL_KEY_SENTINEL_7f3a91"
NUMBER = 734910286517


class Nested(BaseModel):
    model_config = {"extra": "forbid"}
    count: int = 1


class DictCollision(BaseModel):
    known: int = Field(0, alias=S)
    values: dict[str, int]


class ExtraCollision(BaseModel):
    known: int = Field(0, alias=S)
    nested: Nested


class Tagged(BaseModel):
    kind: Literal["MODEL_KEY_SENTINEL_7f3a91"]
    count: int


class Other(BaseModel):
    kind: Literal["other"]
    count: int


class UnionCollision(BaseModel):
    known: int = Field(0, alias=S)
    item: Annotated[Tagged | Other, Field(discriminator="kind")]


class CustomLoc(BaseModel):
    known: int = Field(0, alias=S)
    value: str | int

    @model_validator(mode="after")
    def reject(self) -> CustomLoc:
        raise ValidationError.from_exception_data(
            "custom", [{"type": "int_parsing", "loc": (self.value,), "input": self.value}]
        )


class Plain(BaseModel):
    count: int


CASES = [
    ("dict_key", DictCollision, {"values": {S: "bad"}}, S),
    ("nested_extra_key", ExtraCollision, {"nested": {S: 1}}, S),
    ("union_tag", UnionCollision, {"item": {"kind": S, "count": "bad"}}, S),
    ("custom_string_loc", CustomLoc, {"value": S}, S),
    ("custom_numeric_loc", CustomLoc, {"value": NUMBER}, str(NUMBER)),
]


def _provider(text: str) -> MockLLM:
    llm = MockLLM([LLMResponse(text=text, usage=UsageStats(input_tokens=1, output_tokens=1))])
    llm.capabilities = ProviderCapabilities(supports_structured_output=True)
    return llm


@pytest.mark.parametrize(("name", "model", "data", "marker"), CASES, ids=[c[0] for c in CASES])
async def test_locations_are_never_copied_into_errors(
    name: str, model: type[BaseModel], data: dict[str, Any], marker: str, store: Any
) -> None:
    agent = Agent(
        strict_tenancy=False,
        provider=_provider(json.dumps(data)),
        prompt="t",
        loop=SingleCall(),
        output_type=model,
        state_store=store,
    )
    with pytest.raises(StructuredOutputValidationError) as info:
        await agent.run("hi")
    run_id = (await store.list_runs())[0].id
    record = await store.get_run(run_id)
    events = await store.get_run_events(run_id, event_types=["run.error"])
    surfaces = {
        "str": str(info.value),
        "validation_error": info.value.validation_error,
        "column": record.error,
        "events": repr([e.data for e in events]),
    }
    assert not {k: v for k, v in surfaces.items() if marker in str(v)}
    assert "." not in info.value.validation_error.split("(")[1]  # no location paths at all


async def test_summary_is_counts_and_deduplicated_kinds() -> None:
    class Many(BaseModel):
        a: int
        b: int
        c: str

    llm = _provider('{"a": "x", "b": "y"}')
    with pytest.raises(StructuredOutputValidationError) as info:
        await structured_complete(llm, [Message(role=Role.USER, content="x")], Many)
    assert info.value.validation_error == "3 validation errors (int_parsing, missing)"


class _FailingCompleted(BaseNotifier):
    async def on_llm_call_completed(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("static callback failure")


class _FailingRun(BaseNotifier):
    async def on_run_failed(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("static callback failure")


@pytest.mark.parametrize("notifier", [_FailingCompleted, _FailingRun], ids=["completed", "failed"])
async def test_secondary_callback_failure_logs_no_model_value(
    notifier: type[BaseNotifier], store: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    agent = Agent(
        strict_tenancy=False,
        provider=_provider(json.dumps({"count": S})),
        prompt="t",
        loop=SingleCall(),
        output_type=Plain,
        state_store=store,
    )
    with pytest.raises(StructuredOutputValidationError) as info:
        await agent.run("hi", notifier=notifier())
    fmt = logging.Formatter("%(name)s %(message)s")
    ours = [r for r in caplog.records if r.name.startswith("dendrux")]
    assert any(r.exc_info for r in ours), "the callback failure must have been logged"
    assert not [r for r in ours if S in fmt.format(r)]
    # The developer's in-memory view is kept.
    assert S in info.value.raw_response
    assert info.value.__cause__ is None
    assert S not in "".join(traceback.format_exception(info.value))


@tool(target="client")
async def review_client_tool(sheet: str) -> str:
    """Read a client sheet."""
    return ""


@pytest.mark.parametrize("entry", ["run", "resume", "submit_tool_results"])
async def test_react_parent_never_persists_child_model_values(entry: str, store: Any) -> None:
    child = Agent(
        name="scorer",
        description="Score a draft",
        prompt="score",
        provider=_provider(json.dumps({"values": {S: "bad"}})),
        loop=SingleCall(),
        output_type=DictCollision,
    )
    responses = [
        LLMResponse(
            tool_calls=[
                ToolCall(name="scorer", params={"task": "score"}, provider_tool_call_id="child")
            ]
        ),
        LLMResponse(text="root done"),
    ]
    if entry != "run":
        responses.insert(
            0,
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="review_client_tool",
                        params={"sheet": "a"},
                        provider_tool_call_id="client",
                    )
                ]
            ),
        )
    root = Agent(
        name="lead",
        prompt="lead",
        provider=MockLLM(responses),
        state_store=store,
        strict_tenancy=False,
        subagents=[child],
        tools=[review_client_tool],
    )
    result = await root.run("go")
    if entry != "run":
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        report = ToolResult(
            name="review_client_tool", call_id=result.pending.tool_calls[0].id, payload='"a"'
        )
        submit = root.resume if entry == "resume" else root.submit_tool_results
        if entry == "resume":
            result = await submit(
                result.run_id, tool_results=[report], expected_pause_id=result.pause_id
            )
        else:
            result = await submit(result.run_id, [report], expected_pause_id=result.pause_id)
    assert result.status == RunStatus.SUCCESS
    errors = [r.error for r in await store.list_runs() if r.error]
    assert len(errors) == 1
    assert S not in repr(errors)


async def test_default_otel_span_has_no_rejected_text(store: Any) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from dendrux.notifiers.otel import OpenTelemetryNotifier

    tracer_provider = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    notifier = OpenTelemetryNotifier(tracer=tracer_provider.get_tracer("t"))
    agent = Agent(
        strict_tenancy=False,
        provider=_provider(json.dumps({"count": S})),
        prompt="t",
        loop=SingleCall(),
        output_type=Plain,
        state_store=store,
    )
    try:
        with pytest.raises(StructuredOutputValidationError):
            await agent.run("hi", notifier=notifier)
        spans = exporter.get_finished_spans()
        assert spans
        assert S not in repr([s.to_json() for s in spans])
    finally:
        tracer_provider.shutdown()
