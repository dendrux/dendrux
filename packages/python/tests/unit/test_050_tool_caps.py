"""0.5.0 parallel siblings stop and B12 / input and result caps."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest

import dendrux
from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop, _execute_tool
from dendrux.media._image import Image
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.tools import build_tool_lookups, cap_tool_result, tool_pool_stats
from dendrux.tools._limits import cap_value_payload
from dendrux.tools._lookups import make_call_limiter
from dendrux.types import ApprovalDecision, Limits, LLMResponse, RunStatus, ToolCall, ToolResult

LOG: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    LOG.clear()


class Recorder(BaseRecorder):
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.tools: list[tuple[ToolCall, ToolResult]] = []

    async def on_tool_completed(self, run_id, tool_call, tool_result, iteration) -> None:
        if tool_call.name == self.fail_on and "code" not in tool_result.meta:
            raise RuntimeError("constraint violation")
        self.tools.append((tool_call, tool_result))


async def _run(agent: Agent, llm: MockLLM, recorder: Recorder | None = None):
    return await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        recorder=recorder,
    )


# ------------------------------------------------------------------
# a run-ending failure stops parallel siblings
# ------------------------------------------------------------------


@tool()
async def fast(x: int) -> int:
    """Fast."""
    return x


@tool()
async def slow_side_effect(x: int) -> int:
    """Slow, with a side effect."""
    await asyncio.sleep(0.3)
    LOG.append("slow_side_effect EXECUTED")
    return x


SYNC_RELEASE = threading.Event()


@tool(timeout_seconds=10)
def sync_side_effect(x: int) -> int:
    """Sync, with a side effect."""
    LOG.append("sync started")
    SYNC_RELEASE.wait(5)
    LOG.append("sync finished")
    return x


class TestParallelSiblings:
    async def test_async_sibling_is_cancelled_and_recorded(self) -> None:
        rec = Recorder(fail_on="fast")
        calls = [
            ToolCall(name="fast", params={"x": 1}),
            ToolCall(name="slow_side_effect", params={"x": 2}),
        ]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[fast, slow_side_effect])

        with pytest.raises(RuntimeError, match="constraint violation"):
            await _run(agent, llm, rec)

        await asyncio.sleep(0.5)
        assert LOG == []  # the sibling never applied its effect
        [(call, result)] = rec.tools
        assert call.name == "slow_side_effect"
        assert result.success is False
        assert json.loads(result.payload)["code"] == "cancelled"
        # No row is written after the failure propagated.
        await asyncio.sleep(0.1)
        assert len(rec.tools) == 1

    async def test_started_sync_sibling_is_outcome_unknown(self) -> None:
        SYNC_RELEASE.clear()
        rec = Recorder(fail_on="slow_side_effect")
        calls = [
            ToolCall(name="sync_side_effect", params={"x": 1}),
            ToolCall(name="slow_side_effect", params={"x": 2}),
        ]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[sync_side_effect, slow_side_effect])

        with pytest.raises(RuntimeError):
            await _run(agent, llm, rec)

        [(call, result)] = rec.tools
        assert call.name == "sync_side_effect"
        assert json.loads(result.payload)["code"] == "outcome_unknown"
        assert tool_pool_stats().zombies >= 1
        SYNC_RELEASE.set()
        deadline = time.monotonic() + 5
        while "sync finished" not in LOG and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert "sync finished" in LOG

    async def test_a_sibling_that_already_returned_is_not_recorded_twice(self) -> None:
        class SlowRecorder(Recorder):
            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration) -> None:
                if tool_call.name == "fast":
                    await asyncio.sleep(0.05)
                    raise RuntimeError("constraint violation")
                self.tools.append((tool_call, tool_result))
                await asyncio.sleep(0.5)  # still recording when the failure lands

        @tool()
        def b_sync(x: int) -> int:
            """Quick sync tool."""
            LOG.append("b ran")
            return x

        rec = SlowRecorder()
        calls = [ToolCall(name="fast", params={"x": 1}), ToolCall(name="b_sync", params={"x": 2})]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="ok")])
        with pytest.raises(RuntimeError, match="constraint violation"):
            await _run(Agent(prompt="p", tools=[fast, b_sync]), llm, rec)
        assert LOG == ["b ran"]
        rows = [(c.name, r.success, r.payload) for c, r in rec.tools]
        assert rows == [("b_sync", True, "2")]

    async def test_a_finished_sibling_cut_off_mid_record_is_logged_not_rewritten(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class CancelBeforeWrite(Recorder):
            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration) -> None:
                if tool_call.name == "fast":
                    await asyncio.sleep(0.05)
                    raise RuntimeError("constraint violation")
                await asyncio.sleep(0.5)  # cancelled before the row is written
                self.tools.append((tool_call, tool_result))

        @tool()
        async def c_async(x: int) -> int:
            """Quick async tool."""
            return x

        rec = CancelBeforeWrite()
        calls = [ToolCall(name="fast", params={"x": 1}), ToolCall(name="c_async", params={"x": 3})]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="ok")])
        with caplog.at_level("WARNING"), pytest.raises(RuntimeError):
            await _run(Agent(prompt="p", tools=[fast, c_async]), llm, rec)
        # No synthetic "cancelled" row for a call that ran.
        assert rec.tools == []
        assert any("tool_row_may_be_missing" in r.getMessage() for r in caplog.records)

    async def test_successful_batches_are_unchanged(self) -> None:
        rec = Recorder()
        calls = [ToolCall(name="fast", params={"x": n}) for n in range(5)]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="ok")])
        result = await _run(Agent(prompt="p", tools=[fast]), llm, rec)
        assert result.status == RunStatus.SUCCESS
        assert [r.payload for _, r in rec.tools] == ["0", "1", "2", "3", "4"]

    async def test_outer_cancellation_stops_every_sibling(self) -> None:
        calls = [ToolCall(name="slow_side_effect", params={"x": n}) for n in range(3)]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="ok")])
        task = asyncio.create_task(_run(Agent(prompt="p", tools=[slow_side_effect]), llm))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.4)
        assert LOG == []


# ------------------------------------------------------------------
# B12: tool-result cap
# ------------------------------------------------------------------


@tool()
def big_text(n: int) -> str:
    """Big text."""
    return "é" * n


@tool()
def big_rows(n: int) -> list[dict[str, int]]:
    """Big rows."""
    return [{"row": i} for i in range(n)]


class TestResultCap:
    async def test_oversized_string_is_truncated_with_a_marker(self) -> None:
        limits = Limits(max_tool_result_bytes=1000)
        limiter = make_call_limiter(2)
        limiter.limits = limits
        lookups = build_tool_lookups([big_text], call_limiter=limiter)
        result = await _execute_tool(
            ToolCall(name="big_text", params={"n": 5000}),
            lookups,
            run_id="00000000000000000000000102",
        )
        assert result.success is True
        assert len(result.payload.encode()) <= 1000
        text = json.loads(result.payload)
        assert text.startswith("é") and "[truncated by Dendrux" in text
        assert result.meta["code"] == "size_limit" and result.meta["truncated"] is True
        assert result.meta["bytes"] > 1000

    async def test_oversized_structure_is_replaced_with_valid_json(self) -> None:
        limiter = make_call_limiter(2)
        limiter.limits = Limits(max_tool_result_bytes=500)
        lookups = build_tool_lookups([big_rows], call_limiter=limiter)
        result = await _execute_tool(ToolCall(name="big_rows", params={"n": 1000}), lookups)
        assert result.success is False
        body = json.loads(result.payload)
        assert body["error"] == "result too large" and body["code"] == "size_limit"
        assert body["bytes"] > 500

    async def test_agent_limits_reach_the_tool_path(self) -> None:
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="big_rows", params={"n": 1000})]),
                LLMResponse(text="ok"),
            ]
        )
        agent = Agent(
            prompt="p", provider=llm, tools=[big_rows], limits=Limits(max_tool_result_bytes=300)
        )
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        [tool_msg] = [m for m in llm.call_history[1]["messages"] if m.role.value == "tool"]
        assert json.loads(tool_msg.content)["code"] == "size_limit"

    async def test_default_cap_is_one_mebibyte(self) -> None:
        lookups = build_tool_lookups([big_text])
        fits = await _execute_tool(ToolCall(name="big_text", params={"n": 100_000}), lookups)
        assert "code" not in fits.meta
        over = await _execute_tool(ToolCall(name="big_text", params={"n": 1_000_000}), lookups)
        assert over.meta["code"] == "size_limit"
        assert len(over.payload.encode()) <= 1024 * 1024

    def test_cap_value_payload_leaves_small_results_alone(self) -> None:
        assert cap_value_payload({"a": 1}, '{"a": 1}', 100) == ('{"a": 1}', None)

    def test_client_results_are_capped_too(self) -> None:
        text = ToolResult(name="t", call_id="c", payload=json.dumps("x" * 5000))
        capped = cap_tool_result(text, 1000)
        assert capped.success is True and len(capped.payload.encode()) <= 1000
        assert "[truncated by Dendrux" in json.loads(capped.payload)
        rows = ToolResult(name="t", call_id="c", payload=json.dumps([1] * 5000))
        replaced = cap_tool_result(rows, 1000)
        assert replaced.success is False
        assert json.loads(replaced.payload)["code"] == "size_limit"
        garbage = ToolResult(name="t", call_id="c", payload="{" * 5000)
        assert json.loads(cap_tool_result(garbage, 1000).payload)["code"] == "size_limit"
        small = ToolResult(name="t", call_id="c", payload='"ok"')
        assert cap_tool_result(small, 1000) is small


# ------------------------------------------------------------------
# B12: input cap at every entry point, before any I/O
# ------------------------------------------------------------------


class _NoIOStore:
    """A store that fails the test if anything touches it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"store accessed before the input cap: {name}")


def _capped_agent(**kw: Any) -> Agent:
    llm = MockLLM([LLMResponse(text="ok")])
    agent = Agent(prompt="p", provider=llm, limits=Limits(max_input_bytes=16), **kw)
    agent._state_store = _NoIOStore()  # type: ignore[assignment]
    return agent


class TestInputCap:
    async def test_run_rejects_oversized_text(self) -> None:
        agent = _capped_agent()
        with pytest.raises(ValueError, match="max_input_bytes"):
            await agent.run("x" * 17)

    async def test_text_is_measured_in_utf8_bytes_and_images_are_free(self) -> None:
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="p", provider=llm, limits=Limits(max_input_bytes=16))
        with pytest.raises(ValueError, match="max_input_bytes"):
            await agent.run("é" * 9)  # 18 bytes
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f"
            b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with pytest.raises(ValueError, match="max_input_bytes"):
            await agent.run(["x" * 10, "y" * 10])
        # 16 bytes of text plus an image is within the cap: the image doesn't count.
        try:
            await agent.run(["x" * 8, Image.from_bytes(png), "y" * 8])
        except ValueError as exc:
            assert "max_input_bytes" not in str(exc)

    def test_stream_rejects_at_call_time(self) -> None:
        agent = _capped_agent()
        with pytest.raises(ValueError, match="max_input_bytes"):
            agent.stream("x" * 17)

    async def test_resume_and_submit_paths_reject_before_io(self) -> None:
        agent = _capped_agent()
        big = "x" * 17
        with pytest.raises(ValueError, match="user_input"):
            await agent.resume("00000000000000000000000107", user_input=big)
        with pytest.raises(ValueError, match="user_input"):
            await agent.submit_input("00000000000000000000000107", big)
        with pytest.raises(ValueError, match="rejection_reason"):
            await agent.submit_approval(
                "00000000000000000000000107", approved=False, rejection_reason=big
            )
        with pytest.raises(ValueError, match="decision reasons"):
            await agent.submit_approval(
                "00000000000000000000000107", decisions={"c1": ApprovalDecision.deny(big)}
            )
        with pytest.raises(ValueError, match="user_input"):
            agent.resume_stream("00000000000000000000000107", user_input=big)

    async def test_direct_runner_callers_are_capped(self) -> None:
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="p", provider=llm, limits=Limits(max_input_bytes=16))
        with pytest.raises(ValueError, match="max_input_bytes"):
            await dendrux.run(agent, provider=llm, user_input="x" * 17)
        assert llm.call_history == []

    async def test_within_the_cap_runs_normally(self, caplog: pytest.LogCaptureFixture) -> None:
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="p", provider=llm, limits=Limits(max_input_bytes=16))
        result = await agent.run("x" * 16)
        assert result.status == RunStatus.SUCCESS
        assert agent._tool_call_limiter.limits is agent._limits

    async def test_rejection_is_logged_with_a_stable_code(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = _capped_agent()
        with caplog.at_level("WARNING"), pytest.raises(ValueError):
            await agent.submit_input("00000000000000000000000108", "x" * 40)
        [record] = [r for r in caplog.records if "input_size_limit" in r.getMessage()]
        message = record.getMessage()
        assert record.run_id == "00000000000000000000000108" and record.bytes == 40
        assert "xxxx" not in message
