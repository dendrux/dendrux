"""0.8 T2: the model-facing batch rules (spec section 3) and exclusive registration (D4)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from dendrux import Agent, tool
from dendrux.errors import SubagentDefinitionError
from dendrux.loops import _managed_batch
from dendrux.loops._managed_batch import BatchDecision, decide_batch, effective_limit
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import Limits, LLMResponse, ToolCall
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-brief-91c4"


def _decide(managed: list[str], *, client: int = 0, exclusive: set[str] | None = None, limit: int):
    return decide_batch(managed, client_calls=client, exclusive=exclusive or set(), limit=limit)


# --- the decision table -------------------------------------------------------


def test_no_managed_call_is_not_a_managed_batch() -> None:
    assert _decide([], client=2, limit=4) == BatchDecision("none")


def test_one_managed_call_runs_as_the_single_path() -> None:
    assert _decide(["a"], limit=1) == BatchDecision("single", 1)
    assert _decide(["a"], limit=8) == BatchDecision("single", 1)


def test_one_managed_call_with_a_client_call_is_refused() -> None:
    assert _decide(["a"], client=1, limit=4) == BatchDecision("refuse", 1, "client_call")


@pytest.mark.parametrize("n", [2, 3, 4])
def test_two_to_limit_managed_calls_fan_out(n: int) -> None:
    assert _decide([f"s{i}" for i in range(n)], limit=4) == BatchDecision("fanout", n)


@pytest.mark.parametrize(("n", "limit"), [(5, 4), (33, 32), (2, 1), (3, 2)])
def test_over_the_limit_is_refused(n: int, limit: int) -> None:
    assert _decide(["s"] * n, limit=limit) == BatchDecision("refuse", n, "fanout_limit")


@pytest.mark.parametrize(("n", "limit"), [(32, 32), (4, 4), (2, 2)])
def test_exactly_the_limit_fans_out(n: int, limit: int) -> None:
    assert _decide([f"s{i}" for i in range(n)], limit=limit) == BatchDecision("fanout", n)


def test_thirty_two_and_thirty_three_at_the_ceiling() -> None:
    assert _decide(["s"] * 32, limit=32).action == "fanout"
    assert _decide(["s"] * 33, limit=32) == BatchDecision("refuse", 33, "fanout_limit")


def test_several_calls_to_the_same_subagent_are_allowed() -> None:
    assert _decide(["a", "a", "a"], limit=4) == BatchDecision("fanout", 3)


def test_exclusive_alone_runs() -> None:
    assert _decide(["x"], exclusive={"x"}, limit=4) == BatchDecision("single", 1)


def test_exclusive_with_another_managed_call_is_refused() -> None:
    assert _decide(["x", "a"], exclusive={"x"}, limit=4) == BatchDecision("refuse", 2, "exclusive")
    assert _decide(["a", "x"], exclusive={"x"}, limit=4) == BatchDecision("refuse", 2, "exclusive")


def test_exclusive_called_twice_is_refused() -> None:
    assert _decide(["x", "x"], exclusive={"x"}, limit=4) == BatchDecision("refuse", 2, "exclusive")


def test_server_calls_never_matter_so_the_decision_ignores_them() -> None:
    # Server calls are not an input at all: only managed names and the client count.
    assert _decide(["a", "b"], limit=4).action == "fanout"


def test_precedence_client_call_then_fanout_limit_then_exclusive() -> None:
    # client_call beats everything: it is the one rule that survives any limit.
    assert _decide(["x", "a", "b"], client=1, exclusive={"x"}, limit=2).reason == "client_call"
    # fanout_limit beats exclusive: the limit is the cheaper count to fix first.
    assert _decide(["x", "a", "b"], exclusive={"x"}, limit=2).reason == "fanout_limit"
    assert _decide(["x", "a"], exclusive={"x"}, limit=2).reason == "exclusive"


def test_limit_one_refuses_every_multi_call_batch_as_in_0_7() -> None:
    assert _decide(["a", "b"], limit=1) == BatchDecision("refuse", 2, "fanout_limit")
    assert _decide(["x", "a"], exclusive={"x"}, limit=1).reason == "fanout_limit"


# --- effective limit ------------------------------------------------------------


def test_effective_limit_is_the_configured_limit_when_capable() -> None:
    assert effective_limit(Limits(max_parallel_subagents=7), fanout_capable=True) == 7
    assert effective_limit(Limits(), fanout_capable=True) == 4


def test_effective_limit_is_one_without_the_store_capability() -> None:
    assert effective_limit(Limits(max_parallel_subagents=7), fanout_capable=False) == 1


# --- exclusive registration (D4) ------------------------------------------------


def _child(name: str) -> Agent:
    return Agent(
        name=name,
        description="Does one job.",
        prompt="Work.",
        provider=ScriptedLLM([LLMResponse(text="never")]),
    )


@tool()
async def lookup(x: int) -> str:
    """Look something up."""
    return "ok"


@tool(target="client")
async def read_sheet(sheet: str) -> str:
    """Client tool."""
    return "never"


def _parent(*subagents: Agent, exclusive: Any = None, **kwargs: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="Lead.",
        tools=[lookup, read_sheet],
        subagents=list(subagents),
        exclusive_subagents=exclusive,
        **kwargs,
    )


def test_exclusive_names_are_stored_on_the_agent() -> None:
    agent = _parent(_child("a"), _child("b"), exclusive=["b"])
    assert agent._exclusive_subagents == frozenset({"b"})


def test_exclusive_defaults_to_none() -> None:
    assert _parent(_child("a"))._exclusive_subagents == frozenset()
    assert _parent(_child("a"), exclusive=[])._exclusive_subagents == frozenset()


def test_exclusive_is_keyword_only_and_copied() -> None:
    names = ["a"]
    agent = _parent(_child("a"), exclusive=names)
    names.append("zzz")
    assert agent._exclusive_subagents == frozenset({"a"})


@pytest.mark.parametrize(
    ("exclusive", "reason"),
    [
        (["ghost"], "exclusive_unknown"),
        (["lookup"], "exclusive_unknown"),  # a tool, not a registered subagent
        (["a", "a"], "exclusive_duplicate"),
        ("a", "exclusive_invalid"),
        ([1], "exclusive_invalid"),
        ({"a"}, "exclusive_invalid"),
    ],
)
def test_exclusive_registration_is_validated(exclusive: Any, reason: str) -> None:
    with pytest.raises(SubagentDefinitionError) as info:
        _parent(_child("a"), exclusive=exclusive)
    assert info.value.reason == reason
    assert isinstance(info.value, ValueError)


def test_exclusive_without_any_subagent_is_refused() -> None:
    with pytest.raises(SubagentDefinitionError) as info:
        Agent(name="solo", prompt="p", provider=ScriptedLLM([]), exclusive_subagents=["a"])
    assert info.value.reason == "exclusive_unknown"


def test_error_names_the_subagent_by_field_not_in_the_message() -> None:
    with pytest.raises(SubagentDefinitionError) as info:
        _parent(_child("a"), exclusive=["ghost"])
    assert info.value.subagent_name == "ghost"
    assert "ghost" not in str(info.value)


# --- the call site: refusals carry their reason; N >= 2 stays refused on 0.8 T2 -----


def _call(name: str, params: dict[str, Any], pid: str) -> ToolCall:
    return ToolCall(name=name, params=params, provider_tool_call_id=pid)


class _Recorder(BaseRecorder):
    def __init__(self) -> None:
        self.results: list[Any] = []

    async def on_tool_completed(
        self, run_id: str, tool_call: ToolCall, tool_result: Any, iteration: int
    ) -> None:
        self.results.append(tool_result)


async def _run(agent: Agent, response: LLMResponse) -> tuple[_Recorder, Any]:
    recorder = _Recorder()
    llm = ScriptedLLM([response, LLMResponse(text="done")])
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        recorder=recorder,
    )
    return recorder, result


async def test_two_managed_calls_are_refused_with_fanout_limit_and_counts(caplog) -> None:
    agent = _parent(_child("a"), _child("b"))
    response = LLMResponse(
        tool_calls=[_call("a", {"task": BRIEF}, "p1"), _call("b", {"task": BRIEF}, "p2")]
    )
    with caplog.at_level(logging.WARNING):
        recorder, _ = await _run(agent, response)
    assert len(recorder.results) == 2
    for result in recorder.results:
        assert result.meta == {
            "code": "managed_fanout_limit",
            "reason": "fanout_limit",
            "limit": 1,
            "requested": 2,
        }
        body = json.loads(result.payload)
        assert body["reason"] == "fanout_limit"
        assert (body["limit"], body["requested"]) == (1, 2)
        assert body["error"] == (
            "Refused: 2 subagent calls in one response; the limit is 1. "
            "Call at most 1 subagent at once."
        )
    assert "managed_batch_unsupported" not in caplog.text
    assert "managed_fanout_limit" in caplog.text
    assert BRIEF not in caplog.text


async def test_managed_plus_client_call_is_refused_with_client_call_and_0_7_message() -> None:
    agent = _parent(_child("a"))
    response = LLMResponse(
        tool_calls=[_call("a", {"task": BRIEF}, "p1"), _call("read_sheet", {"sheet": "S"}, "p2")]
    )
    recorder, _ = await _run(agent, response)
    assert len(recorder.results) == 2
    for result in recorder.results:
        assert result.meta == {"code": "managed_batch_unsupported", "reason": "client_call"}
        assert json.loads(result.payload) == {
            "error": _managed_batch._MANAGED_BATCH_MESSAGE,
            "reason": "client_call",
        }


async def test_managed_plus_server_call_still_runs_the_single_path() -> None:
    agent = _parent(_child("a"))
    response = LLMResponse(
        tool_calls=[_call("a", {"task": BRIEF}, "p1"), _call("lookup", {"x": 1}, "p2")]
    )
    _, result = await _run(agent, response)
    assert result.status.value == "running"  # handed off to the runner, not refused


async def test_policy_from_the_agent_drives_classification_when_capable() -> None:
    agent = _parent(
        _child("a"), _child("b"), exclusive=["b"], limits=Limits(max_parallel_subagents=3)
    )
    lookups = await agent.get_tool_lookups()
    both = [_call("a", {"task": "t"}, "p1"), _call("b", {"task": "t"}, "p2")]
    capable = _managed_batch.batch_policy(agent, fanout_capable=True)
    assert capable.limit == 3
    assert _managed_batch.classify_batch(both, lookups, capable).reason == "exclusive"
    only_a = [_call("a", {"task": "t"}, "p1"), _call("a", {"task": "t"}, "p2")]
    assert _managed_batch.classify_batch(only_a, lookups, capable).action == "fanout"
    incapable = _managed_batch.batch_policy(agent, fanout_capable=False)
    assert incapable.limit == 1
    assert _managed_batch.classify_batch(only_a, lookups, incapable).reason == "fanout_limit"


def test_exclusive_payload_carries_reason_and_the_exclusive_text() -> None:
    tc = _call("a", {"task": "t"}, "p1")
    result = _managed_batch._managed_batch_result(tc, BatchDecision("refuse", 2, "exclusive"))
    assert json.loads(result.payload) == {
        "error": _managed_batch._EXCLUSIVE_MESSAGE,
        "reason": "exclusive",
    }
    assert result.meta["reason"] == "exclusive"


def test_fanout_limit_payload_above_one_names_the_limit() -> None:
    tc = _call("a", {"task": "t"}, "p1")
    decision = BatchDecision("refuse", 5, "fanout_limit")
    body = json.loads(_managed_batch._managed_batch_result(tc, decision, limit=4).payload)
    assert body["limit"] == 4 and body["requested"] == 5 and "at most 4" in body["error"]


def test_result_without_a_decision_is_the_plain_0_7_body() -> None:
    result = _managed_batch._managed_batch_result(_call("a", {"task": "t"}, "p1"))
    assert json.loads(result.payload) == {"error": _managed_batch._MANAGED_BATCH_MESSAGE}


@pytest.mark.parametrize("exclusive", [None, [], ["a"], ["ghost"]])
def test_invalid_subagent_entry_is_refused_before_exclusive_names(exclusive: Any) -> None:
    with pytest.raises(SubagentDefinitionError) as info:
        Agent(
            name="lead",
            prompt="p",
            provider=ScriptedLLM([]),
            subagents=[object()],  # type: ignore[list-item]
            exclusive_subagents=exclusive,
        )
    assert info.value.reason == "not_agent"


@pytest.mark.parametrize("limit", [1, 2, 4, 32])
def test_fanout_refusal_code_is_distinct(limit):
    result = _managed_batch._managed_batch_result(
        ToolCall("a", {"task": BRIEF}),
        BatchDecision("refuse", limit + 1, "fanout_limit"),
        limit=limit,
    )
    assert result.meta["code"] == "managed_fanout_limit"


@pytest.mark.parametrize("limit", [1, 2, 4, 32])
def test_fanout_refusal_text_states_requested_count_and_limit(limit):
    result = _managed_batch._managed_batch_result(
        ToolCall("a", {"task": BRIEF}),
        BatchDecision("refuse", limit + 1, "fanout_limit"),
        limit=limit,
    )
    subagents = "subagent" if limit == 1 else "subagents"
    expected = (
        f"Refused: {limit + 1} subagent calls in one response; the limit is {limit}. "
        f"Call at most {limit} {subagents} at once."
    )
    assert result.error == json.loads(result.payload)["error"] == expected
    assert BRIEF not in result.payload
