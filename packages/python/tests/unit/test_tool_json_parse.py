"""Malformed tool-call JSON is a failed tool call, never a failed run."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.llm._helpers import NOT_AN_OBJECT_ERROR, build_tool_call, parse_tool_json
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.tool import tool
from dendrux.tools._validate import (
    INVALID_JSON_CODE,
    check_tool_call,
    check_tool_json,
    format_invalid_json,
)
from dendrux.types import (
    AgentStep,
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunEventType,
    RunStatus,
    ToolCall,
    ToolDef,
    UsageStats,
    _message_from_dict,
    _message_to_dict,
    tool_call_parse_meta,
)

DOCSTRING_RAW = (
    '{"path": "build_deck.py", "content": "#!/usr/bin/env python3\\n"""Build the deck"""\\n"}'
)
DOCSTRING_ERROR = "arguments must be valid JSON"


def _bad_call(name: str = "write_file", raw: str = DOCSTRING_RAW, **kwargs: Any) -> ToolCall:
    params, error = parse_tool_json(raw)
    assert error is not None
    return ToolCall(name=name, params=params, raw_arguments=raw, parse_error=error, **kwargs)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


class TestParseToolJson:
    def test_valid_object(self) -> None:
        assert parse_tool_json('{"a": 1, "b": [2]}') == ({"a": 1, "b": [2]}, None)

    def test_dict_passthrough(self) -> None:
        assert parse_tool_json({"a": 1}) == ({"a": 1}, None)

    @pytest.mark.parametrize("raw", ["", "   ", "\n\t", "null"])
    def test_empty_whitespace_and_null_are_empty_params_without_error(self, raw: str) -> None:
        assert parse_tool_json(raw) == ({}, None)

    def test_trailing_garbage(self) -> None:
        assert parse_tool_json('{"a": 1} trailing') == ({}, "arguments must be valid JSON")

    def test_truncated(self) -> None:
        assert parse_tool_json('{"a": 1, "b":') == ({}, "arguments must be valid JSON")

    def test_unescaped_docstring(self) -> None:
        assert parse_tool_json(DOCSTRING_RAW) == ({}, DOCSTRING_ERROR)

    def test_invalid_escape(self) -> None:
        assert parse_tool_json('{"a": "C:\\q"}') == ({}, "arguments must be valid JSON")

    @pytest.mark.parametrize("raw", ["[1, 2]", '"text"', "42", "true"])
    def test_non_object_json_is_an_error(self, raw: str) -> None:
        assert parse_tool_json(raw) == ({}, NOT_AN_OBJECT_ERROR)

    def test_error_never_contains_raw_text(self) -> None:
        secret = '{"content": "TOP-SECRET-FILE-BODY" oops}'
        _, error = parse_tool_json(secret)
        assert error is not None
        assert "TOP-SECRET" not in error


class TestBuildToolCall:
    def test_valid_arguments(self) -> None:
        tc = build_tool_call(
            "add", '{"a": 1}', provider_tool_call_id="c1", provider="openai", model="m"
        )
        assert tc.params == {"a": 1}
        assert tc.parse_error is None
        assert tc.raw_arguments is None
        assert tc.provider_tool_call_id == "c1"

    def test_malformed_arguments_logged_without_raw_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="dendrux.llm._helpers"):
            tc = build_tool_call(
                "write_file",
                DOCSTRING_RAW,
                provider_tool_call_id="c1",
                provider="openai",
                model="m",
            )
        assert tc.params == {}
        assert tc.raw_arguments == DOCSTRING_RAW
        assert tc.parse_error == DOCSTRING_ERROR
        messages = [r.getMessage() for r in caplog.records]
        assert any(f"raw_len={len(DOCSTRING_RAW)}" in m for m in messages)
        assert not any("Build the deck" in m for m in messages)

    def test_raw_arguments_not_in_repr(self) -> None:
        assert "Build the deck" not in repr(_bad_call())


# ------------------------------------------------------------------
# Serialization: parse_error + raw length only, never the raw text
# ------------------------------------------------------------------


class TestSerialization:
    def test_parse_meta(self) -> None:
        assert tool_call_parse_meta(ToolCall(name="x")) == {}
        assert tool_call_parse_meta(_bad_call()) == {
            "parse_error": DOCSTRING_ERROR,
            "raw_length": len(DOCSTRING_RAW),
        }

    def test_message_round_trip_keeps_parse_error_drops_raw(self) -> None:
        tc = _bad_call()
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        d = _message_to_dict(msg)
        assert "Build the deck" not in json.dumps(d)
        assert d["tool_calls"][0]["raw_length"] == len(DOCSTRING_RAW)
        restored = _message_from_dict(d).tool_calls
        assert restored is not None
        assert restored[0].parse_error == DOCSTRING_ERROR
        assert restored[0].raw_arguments is None
        assert restored[0].id == tc.id

    def test_pause_state_round_trip(self) -> None:
        tc = _bad_call()
        state = PauseState(
            agent_name="a",
            pending_tool_calls=[],
            history=[Message(role=Role.ASSISTANT, content="", tool_calls=[tc])],
            steps=[AgentStep(reasoning=None, action=tc, meta={"all_tool_calls": [tc]})],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        data = state.to_dict()
        assert "Build the deck" not in json.dumps(data)
        restored = PauseState.from_dict(json.loads(json.dumps(data)))
        assert restored.history[0].tool_calls[0].parse_error == DOCSTRING_ERROR  # type: ignore[index]
        action = restored.steps[0].action
        assert isinstance(action, ToolCall)
        assert action.parse_error == DOCSTRING_ERROR
        assert restored.steps[0].meta["all_tool_calls"][0].parse_error == DOCSTRING_ERROR

    def test_legacy_dict_without_parse_error(self) -> None:
        d = {"role": "assistant", "content": "", "tool_calls": [{"name": "x", "id": "1"}]}
        restored = _message_from_dict(d).tool_calls
        assert restored is not None
        assert restored[0].parse_error is None


# ------------------------------------------------------------------
# check_tool_json / check_tool_call
# ------------------------------------------------------------------


class TestCheckToolJson:
    def test_parsed_call_passes(self) -> None:
        assert check_tool_json(ToolCall(name="x", params={"a": 1})) is None

    def test_standard_message(self) -> None:
        assert format_invalid_json("write_file", "arguments must be valid JSON") == (
            "Your arguments for `write_file` were not valid JSON "
            "(arguments must be valid JSON). Send the call again with valid JSON; "
            "for long content, write it in smaller pieces."
        )

    def test_failed_result_shape(self) -> None:
        tc = _bad_call()
        result = check_tool_json(tc)
        assert result is not None
        assert result.success is False
        assert result.call_id == tc.id
        assert result.error == format_invalid_json("write_file", DOCSTRING_ERROR)
        assert json.loads(result.payload) == {"error": result.error, "code": INVALID_JSON_CODE}
        assert "Build the deck" not in result.payload

    def test_check_tool_call_rejects_json_before_schema(self) -> None:
        td = ToolDef(
            name="write_file",
            description="",
            parameters={"type": "object", "properties": {"path": {}}, "required": ["path"]},
        )
        result = check_tool_call(_bad_call(), td)
        assert result is not None
        assert json.loads(result.payload)["code"] == INVALID_JSON_CODE
        assert "missing required" not in result.payload

    def test_check_tool_call_rejects_json_for_unknown_tool(self) -> None:
        result = check_tool_call(_bad_call(name="nope"), None)
        assert result is not None
        assert json.loads(result.payload)["code"] == INVALID_JSON_CODE


# ------------------------------------------------------------------
# Loop behaviour
# ------------------------------------------------------------------

WRITES: list[dict[str, str]] = []


@tool()
async def write_file(path: str, content: str) -> str:
    """Write a file."""
    WRITES.append({"path": path, "content": content})
    return f"wrote {path}"


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Read a sheet range on the client."""
    return ""


@pytest.fixture(autouse=True)
def _reset_writes() -> None:
    WRITES.clear()


class _Capture(BaseRecorder):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.completed: list[tuple[ToolCall, Any]] = []

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        self.started.append(tool_call.name)

    async def on_tool_completed(
        self, run_id: str, tool_call: ToolCall, tool_result: Any, iteration: int
    ) -> None:
        self.completed.append((tool_call, tool_result))


def _tool_messages(llm: MockLLM, call_index: int) -> list[Message]:
    return [m for m in llm.call_history[call_index]["messages"] if m.role == Role.TOOL]


GOOD_PARAMS = {"path": "build_deck.py", "content": '#!/usr/bin/env python3\n"""Build the deck"""\n'}


def _good_call() -> ToolCall:
    return ToolCall(name="write_file", params=dict(GOOD_PARAMS), provider_tool_call_id="t2")


class TestLoop:
    async def test_malformed_call_rejected_and_model_recovers(self) -> None:
        bad = _bad_call(provider_tool_call_id="t1")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[bad]),
                LLMResponse(tool_calls=[_good_call()]),
                LLMResponse(text="done"),
            ]
        )
        agent = Agent(prompt="p", tools=[write_file], provider=llm)
        recorder = _Capture()

        result = await agent.run("go", notifier=recorder)

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "done"
        assert WRITES == [GOOD_PARAMS]
        (first,) = _tool_messages(llm, 1)
        assert first.call_id == bad.id
        assert first.meta == {"is_error": True}
        assert json.loads(first.content) == {
            "error": format_invalid_json("write_file", DOCSTRING_ERROR),
            "code": "invalid_json",
        }
        assert recorder.started == ["write_file"]
        rejected_call, rejected = recorder.completed[0]
        assert rejected_call.parse_error == DOCSTRING_ERROR
        assert rejected.success is False
        assert recorder.completed[1][1].success is True

    async def test_mixed_batch_runs_valid_call(self) -> None:
        bad = _bad_call(provider_tool_call_id="t1")
        good = _good_call()
        llm = MockLLM([LLMResponse(tool_calls=[bad, good]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[write_file], provider=llm)
        recorder = _Capture()

        result = await agent.run("go", notifier=recorder)

        assert result.status == RunStatus.SUCCESS
        assert WRITES == [GOOD_PARAMS]
        msgs = {m.call_id: m for m in _tool_messages(llm, 1)}
        assert "invalid_json" in msgs[bad.id].content
        assert msgs[good.id].meta == {}
        assert recorder.started == ["write_file"]

    async def test_malformed_call_to_approval_tool_does_not_pause(self) -> None:
        llm = MockLLM([LLMResponse(tool_calls=[_bad_call()]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[write_file], require_approval=["write_file"], provider=llm)

        result = await agent.run("go")

        assert result.status == RunStatus.SUCCESS
        assert WRITES == []
        assert "invalid_json" in _tool_messages(llm, 1)[0].content

    async def test_malformed_call_to_client_tool_does_not_pause(self) -> None:
        bad = _bad_call(name="read_range", raw='{"sheet": S1}')
        llm = MockLLM([LLMResponse(tool_calls=[bad]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[read_range], provider=llm)

        result = await agent.run("go")

        assert result.status == RunStatus.SUCCESS

    async def test_malformed_call_to_denied_tool_reports_parse_error(self) -> None:
        llm = MockLLM([LLMResponse(tool_calls=[_bad_call()]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[write_file], deny=["write_file"], provider=llm)

        await agent.run("go")

        content = _tool_messages(llm, 1)[0].content
        assert "invalid_json" in content and "denied" not in content

    async def test_stream_emits_failed_tool_result_and_no_run_error(self) -> None:
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[_bad_call()]),
                LLMResponse(tool_calls=[_good_call()]),
                LLMResponse(text="ok"),
            ]
        )
        agent = Agent(prompt="p", tools=[write_file], provider=llm)

        events = [e async for e in agent.stream("go")]

        types = [e.type for e in events]
        assert RunEventType.RUN_ERROR not in types
        assert types[-1] == RunEventType.RUN_COMPLETED
        results = [e.tool_result for e in events if e.type == RunEventType.TOOL_RESULT]
        assert [r.success for r in results if r is not None] == [False, True]
        assert results[0] is not None
        assert results[0].error == format_invalid_json("write_file", DOCSTRING_ERROR)
        assert WRITES == [GOOD_PARAMS]

    async def test_model_that_never_recovers_is_bounded_by_max_iterations(self) -> None:
        llm = MockLLM([LLMResponse(tool_calls=[_bad_call()]) for _ in range(3)])
        agent = Agent(prompt="p", tools=[write_file], provider=llm, max_iterations=3)

        result = await agent.run("go")

        assert result.status == RunStatus.MAX_ITERATIONS
        assert WRITES == []
