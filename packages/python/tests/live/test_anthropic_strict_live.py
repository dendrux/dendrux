"""Live checks for Anthropic strict mode (0.4.1, T2).

Scenarios, each on claude-sonnet-4-6 and claude-haiku-4-5:
    ST1  @tool(strict=True) with a nested model, Field(ge=.., le=..), an
         optional field, an enum, a pattern and a capped list (the tool
         schema generator inlines models): sent with "strict": true, the
         arguments validate, and an agent run executes the tool.
    ST2  structured output with the same shapes, here as $defs/$ref: sent
         strict, validates (direct call and an Agent with output_type).
    ST3  structured output with a dict[str, str] field: sent non-strict
         (unchanged payload) and still validates.
    ST4  structured output whose strict grammar is too large (60 fields with
         max_length=5000): Anthropic 400s the strict request, Dendrux retries
         once non-strict, the output validates and provider_request shows
         the non-strict request.
    ST5  a strict tool whose pattern the normaliser keeps but Anthropic's
         regex budget rejects, streamed: retried non-strict before any
         event, the stream completes.
    ST6  AnthropicProvider(strict=False): the strict tool and the structured
         output go out exactly as in 0.4.0 (no "strict") and still work.

An HTTP 200 alone never passes: the request evidence must show the strict
flag (or its absence) and the output must validate against the model.

Run (keys come from the environment only):
    DENDRUX_LIVE=1 DENDRUX_LIVE_RESULTS=/path/results.json \
        pytest tests/live/test_anthropic_strict_live.py -v
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pytest
from pydantic import BaseModel, Field, create_model

from dendrux import Agent, tool
from dendrux.llm import anthropic as anthropic_mod
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.llm.structured import structured_complete
from dendrux.loops.single import SingleCall
from dendrux.tool import get_tool_def
from dendrux.types import Message, Role, RunStatus, StreamEventType, ToolDef

MODELS = ("claude-sonnet-4-6", "claude-haiku-4-5")


@dataclass(frozen=True)
class Case:
    model: str
    scenario: str
    family: str = "anthropic"
    route_id: str = "anthropic-strict"


class Size(StrEnum):
    SMALL = "small"
    LARGE = "large"


class Address(BaseModel):
    city: str
    postcode: str = Field(pattern=r"^\d{5}$", description="Five digits.")


class Order(BaseModel):
    item: str
    quantity: int = Field(ge=1, le=50)
    size: Size
    gift_note: str | None = None
    ship_to: Address
    tags: list[str] = Field(max_length=3)


class Labelled(BaseModel):
    title: str
    labels: dict[str, str]


_seen: list[dict[str, Any]] = []


@tool(strict=True)
async def place_order(order: Order) -> str:
    """Place an order for an item."""
    parsed = order if isinstance(order, Order) else Order.model_validate(order)
    _seen.append(parsed.model_dump(mode="json"))
    return f"order placed: {parsed.quantity} x {parsed.item}"


ORDER_PROMPT = (
    "Place an order: 3 small blue mugs, gift note 'Happy birthday', shipped to Berlin "
    "postcode 10115, tagged 'kitchen' and 'gift'."
)


def _cases(scenario: str) -> list[Case]:
    return [Case(model=m, scenario=scenario) for m in MODELS]


def _provider(model: str) -> AnthropicProvider:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    return AnthropicProvider(model=model, max_tokens=1024)


@pytest.mark.parametrize("case", _cases("ST1"), ids=lambda c: f"{c.model}-ST1")
async def test_strict_tool(case: Case, rec: dict[str, Any]) -> None:
    provider = _provider(case.model)
    td = get_tool_def(place_order)
    response = await provider.complete(
        [
            Message(role=Role.SYSTEM, content="Use the place_order tool."),
            Message(role=Role.USER, content=ORDER_PROMPT),
        ],
        tools=[td],
    )
    sent = response.provider_request["tools"][0]
    assert sent["strict"] is True
    quantity = sent["input_schema"]["properties"]["order"]["properties"]["quantity"]
    assert "minimum" not in quantity
    assert response.tool_calls, response.text
    order = Order.model_validate(response.tool_calls[0].params["order"])
    assert order.quantity == 3
    rec["args"] = response.tool_calls[0].params
    rec["usage"] = {"in": response.usage.input_tokens, "out": response.usage.output_tokens}

    _seen.clear()
    async with Agent(
        name="strict-live",
        provider=_provider(case.model),
        prompt="Use the place_order tool, then confirm in one sentence.",
        tools=[place_order],
    ) as agent:
        result = await agent.run(ORDER_PROMPT)
    assert result.status is RunStatus.SUCCESS, result.error
    assert _seen and _seen[0]["quantity"] == 3
    rec["agent_answer"] = (result.answer or "")[:200]


@pytest.mark.parametrize("case", _cases("ST2"), ids=lambda c: f"{c.model}-ST2")
async def test_strict_structured_output(case: Case, rec: dict[str, Any]) -> None:
    provider = _provider(case.model)
    response, order = await structured_complete(
        provider,
        [Message(role=Role.USER, content=f"Extract the order. {ORDER_PROMPT}")],
        Order,
    )
    sent = response.provider_request["tools"][0]
    assert sent["strict"] is True
    assert {"Address", "Size"} <= set(sent["input_schema"]["$defs"])
    assert sent["input_schema"]["properties"]["quantity"]["description"].endswith(
        "Must be >= 1. Must be <= 50."
    )
    assert order.quantity == 3
    assert order.size is Size.SMALL
    assert order.ship_to.postcode == "10115"
    rec["output"] = order.model_dump(mode="json")

    async with Agent(
        name="strict-live-structured",
        provider=_provider(case.model),
        prompt="Extract the order.",
        loop=SingleCall(),
        output_type=Order,
    ) as agent:
        result = await agent.run(ORDER_PROMPT)
    assert result.status is RunStatus.SUCCESS, result.error
    assert isinstance(result.output, Order)
    assert result.output.quantity == 3


@pytest.mark.parametrize("case", _cases("ST3"), ids=lambda c: f"{c.model}-ST3")
async def test_dict_schema_falls_back_non_strict(case: Case, rec: dict[str, Any]) -> None:
    provider = _provider(case.model)
    schema = Labelled.model_json_schema()
    response, out = await structured_complete(
        provider,
        [
            Message(
                role=Role.USER,
                content="Title 'Q3 report'; labels: team=finance, status=draft.",
            )
        ],
        Labelled,
    )
    sent = response.provider_request["tools"][0]
    assert "strict" not in sent
    assert sent["input_schema"] == schema
    assert out.labels.get("team") == "finance"
    rec["output"] = out.model_dump(mode="json")


Wide = create_model(  # type: ignore[call-overload]
    "Wide", **{f"field_{i}": (str, Field(min_length=1, max_length=5000)) for i in range(60)}
)


def _strict_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dendrux.llm.anthropic"
    ]


@pytest.mark.parametrize("case", _cases("ST4"), ids=lambda c: f"{c.model}-ST4")
async def test_grammar_too_large_falls_back(
    case: Case, rec: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    anthropic_mod._STRICT_REJECTED.clear()
    anthropic_mod._STRICT_WARNED.clear()
    _provider(case.model)  # skips without a key
    provider = AnthropicProvider(model=case.model, max_tokens=4096)
    with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
        response, out = await structured_complete(
            provider,
            [Message(role=Role.USER, content="Fill every field with one short lowercase word.")],
            Wide,
        )
    sent = response.provider_request["tools"][0]
    assert "strict" not in sent
    assert sent["input_schema"] == Wide.model_json_schema()
    assert out.field_0
    warned = _strict_warnings(caplog)
    assert len(warned) == 1 and "structured_output" in warned[0]
    rec["warning"] = warned[0][:400]


@pytest.mark.parametrize("case", _cases("ST5"), ids=lambda c: f"{c.model}-ST5")
async def test_regex_budget_falls_back_in_stream(
    case: Case, rec: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    anthropic_mod._STRICT_REJECTED.clear()
    anthropic_mod._STRICT_WARNED.clear()
    params = {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^[a-z]{1,100000}$"}},
        "required": ["code"],
        "additionalProperties": False,
    }
    td = ToolDef(name="save_code", description="Save a code.", parameters=params, strict=True)
    provider = _provider(case.model)
    with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
        events = [
            e
            async for e in provider.complete_stream(
                [Message(role=Role.USER, content="Save the code 'abc' with the save_code tool.")],
                tools=[td],
            )
        ]
    done = events[-1]
    assert done.type is StreamEventType.DONE
    assert done.raw.provider_request["tools"] == [
        {"name": "save_code", "description": "Save a code.", "input_schema": params}
    ]
    assert done.raw.tool_calls and done.raw.tool_calls[0].params["code"] == "abc"
    warned = _strict_warnings(caplog)
    assert len(warned) == 1 and "save_code" in warned[0]
    rec["warning"] = warned[0][:400]


@pytest.mark.parametrize("case", _cases("ST6"), ids=lambda c: f"{c.model}-ST6")
async def test_strict_opt_out(case: Case, rec: dict[str, Any]) -> None:
    _provider(case.model)  # skips without a key
    provider = AnthropicProvider(model=case.model, max_tokens=1024, strict=False)
    td = get_tool_def(place_order)
    response = await provider.complete(
        [
            Message(role=Role.SYSTEM, content="Use the place_order tool."),
            Message(role=Role.USER, content=ORDER_PROMPT),
        ],
        tools=[td],
    )
    sent = response.provider_request["tools"][0]
    assert sent == {"name": td.name, "description": td.description, "input_schema": td.parameters}
    assert response.tool_calls
    structured, order = await structured_complete(
        provider, [Message(role=Role.USER, content=f"Extract the order. {ORDER_PROMPT}")], Order
    )
    assert "strict" not in structured.provider_request["tools"][0]
    assert structured.provider_request["tools"][0]["input_schema"] == Order.model_json_schema()
    assert order.quantity == 3
