"""``ReActLoop.run`` and ``run_stream`` write the same records for the same script.

Each scenario runs once through ``run`` and once through ``run_stream`` with
identical providers, spies and (where used) a scripted run-control store.
The recorder, notifier and governance sequences must match exactly, and so
must the final result. The stream's own ``RunEvent``s are not part of the
comparison except for the terminal event type, which is pinned per scenario.
A notifier that fails on call ``w`` puts ``notifier_warnings`` on later results,
so result meta is compared with and without warnings.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from dendrux import Agent, tool
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED, FailureReason
from dendrux.llm.mock import MockLLM
from dendrux.loops._helpers import guardrail_meta
from dendrux.loops.base import BaseNotifier, BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.runtime.state import NoteRecord
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    LLMResponse,
    RunEventType,
    RunResult,
    RunStatus,
    ToolCall,
    ToolDecision,
    UsageStats,
    generate_ulid,
)

RUN_ID = generate_ulid()


@tool(parallel=False)
async def work(n: int = 0) -> str:
    return f"ok {n}"


@tool(parallel=False)
async def nope() -> str:
    return "never"


@tool(parallel=False, max_calls_per_run=1)
async def lim() -> str:
    return "lim"


@tool(parallel=False)
async def hdeny() -> str:
    return "never"


@tool(parallel=False)
async def gated() -> str:
    return "never"


def _hook(ctx: Any) -> ToolDecision | None:
    return ToolDecision.deny("hook says no") if ctx.tool_name == "hdeny" else None


def _usage() -> UsageStats:
    return UsageStats(input_tokens=3, output_tokens=2, total_tokens=5)


def _calls(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(tool_calls=list(calls), usage=_usage())


def _text(text: str) -> LLMResponse:
    return LLMResponse(text=text, usage=_usage())


class _Spy:
    def __init__(self) -> None:
        self.log: list[tuple[Any, ...]] = []

    def recorder(self) -> BaseRecorder:
        log = self.log

        class Rec(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                log.append(("rec.msg", iteration, message.role.value, message.call_id,
                            message.content))  # fmt: skip

            async def on_llm_call_started(self, run_id, iteration, **kw):
                log.append(("rec.llm_started", iteration))

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                names = [tc.name for tc in response.tool_calls or ()]
                log.append(("rec.llm", iteration, response.text, names))

            async def on_llm_call_failed(self, run_id, iteration, error, **kw):
                log.append(("rec.llm_failed", iteration, type(error).__name__))

            async def on_tool_started(self, run_id, tool_call, iteration):
                log.append(("rec.tool_started", iteration, tool_call.id))

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                log.append(("rec.tool", iteration, tool_call.id, tool_result.payload))

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
                log.append(("rec.gov", iteration, event_type, data, correlation_id))

        return Rec()

    def notifier(self) -> BaseNotifier:
        log = self.log

        class Not(BaseNotifier):
            async def on_message_appended(self, run_id, message, iteration):
                log.append(("not.msg", iteration, message.call_id))

            async def on_llm_call_started(self, run_id, iteration, **kw):
                log.append(("not.llm_started", iteration))

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                log.append(("not.llm", iteration))

            async def on_tool_started(self, run_id, tool_call, iteration):
                log.append(("not.tool_started", iteration, tool_call.id))

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                log.append(("not.tool", iteration, tool_call.id))
                if tool_call.id == "w":
                    # A failing notifier puts notifier_warnings on every later result.
                    raise RuntimeError("notifier down")

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
                log.append(("not.gov", iteration, event_type, correlation_id))

        return Not()


class _Control:
    def __init__(self, cancel: bool = False, pause: bool = False, notes: bool = False) -> None:
        self.cancel_requested = cancel
        self.pause_requested = pause
        self.has_pending_notes = notes


class _Store:
    """Run-control reads answered from a script (one entry per read), then idle."""

    def __init__(
        self, script: list[_Control], notes: list[str] | None = None, storage_format: int = 1
    ) -> None:
        self.storage_format = storage_format
        self.script = list(script)
        self.notes = [
            NoteRecord(id=f"note{i}", run_id="run", text=t) for i, t in enumerate(notes or [])
        ]
        self.reads = 0
        self.delivered: list[tuple[list[str], int]] = []

    async def get_run_control(self, run_id: str) -> _Control:
        self.reads += 1
        return self.script.pop(0) if self.script else _Control()

    async def get_notes(self, run_id: str, *, undelivered_only: bool = False) -> list[NoteRecord]:
        return self.notes

    async def mark_notes_delivered(self, run_id: str, ids: list[str], *, iteration: int) -> None:
        self.delivered.append((ids, iteration))
        self.notes = []

    async def get_run(self, run_id: str) -> Any:
        return SimpleNamespace(id=run_id, tenant_id=None, meta={})


def _summary(result: RunResult) -> dict[str, Any]:
    pause = result.meta.get("pause_state")
    return {
        "status": result.status,
        "answer": result.answer,
        "error": result.error,
        "iteration_count": result.iteration_count,
        "usage": (result.usage.input_tokens, result.usage.output_tokens),
        "steps": len(result.steps),
        "meta_keys": list(result.meta),
        "warnings": list(result.meta.get("notifier_warnings", [])),
        "failure_reason": result.meta.get("failure_reason"),
        "pending": [tc.id for tc in pause.pending_tool_calls] if pause else None,
        "gates": pause.approval_gates if pause else None,
        "pause_iteration": pause.iteration if pause else None,
        "pause_reason": getattr(pause, "pause_reason", None) if pause else None,
    }


async def _both(
    make_agent: Any,
    responses: list[LLMResponse],
    *,
    script: list[_Control] | None = None,
    notes: list[str] | None = None,
    storage_format: int = 1,
) -> tuple[tuple[list[Any], dict[str, Any], Any], tuple[list[Any], dict[str, Any], Any]]:
    out = []
    for mode in ("run", "stream"):
        spy = _Spy()
        store = _Store(script, notes, storage_format) if script is not None else None
        kwargs: dict[str, Any] = {
            "agent": make_agent(),
            "provider": MockLLM(list(responses)),
            "strategy": NativeToolCalling(),
            "user_input": "go",
            "run_id": RUN_ID,
            "recorder": spy.recorder(),
            "notifier": spy.notifier(),
            "state_store": store,
        }
        if mode == "run":
            result = await ReActLoop().run(**kwargs)
            terminal = None
        else:
            events = [e async for e in ReActLoop().run_stream(**kwargs)]
            result = events[-1].run_result
            terminal = events[-1].type
        assert result is not None
        out.append((spy.log, _summary(result), terminal, store))
    (run_log, run_sum, _, run_store), (stream_log, stream_sum, terminal, stream_store) = out
    assert stream_log == run_log
    assert stream_sum == run_sum
    if run_store is not None:
        assert stream_store.reads == run_store.reads
        assert stream_store.delivered == run_store.delivered
    return (run_log, run_sum, None), (stream_log, stream_sum, terminal)


async def test_deny_hook_limit_and_unknown_tool_match() -> None:
    def make() -> Agent:
        return Agent(prompt="p", tools=[work, nope, lim, hdeny], deny=["nope"], before_tool=_hook)

    responses = [
        _calls(
            ToolCall("nope", id="d"),
            ToolCall("hdeny", id="hd"),
            ToolCall("lim", id="l1"),
            ToolCall("lim", id="l2"),
            ToolCall("ghost", id="u"),
            ToolCall("work", {"n": 1}, id="w"),
        ),
        _text("done"),
    ]
    (log, summary, _), (_, _, terminal) = await _both(make, responses)
    assert terminal == RunEventType.RUN_COMPLETED
    assert summary["status"] == RunStatus.SUCCESS and summary["answer"] == "done"
    assert len(summary["warnings"]) == 1
    governance = [(e[2], e[4]) for e in log if e[0] == "rec.gov"]
    assert governance == [("policy.denied", "d"), ("policy.denied", "hd")]
    tool_rows = [e[2] for e in log if e[0] == "rec.tool"]
    assert tool_rows == ["u", "l2", "l1", "w"]


async def test_approval_pause_matches() -> None:
    def make() -> Agent:
        return Agent(prompt="p", tools=[work, gated], require_approval=["gated"])

    responses = [_calls(ToolCall("gated", id="g"), ToolCall("work", id="w"))]
    (log, summary, _), (_, _, terminal) = await _both(make, responses)
    assert terminal == RunEventType.RUN_PAUSED
    assert summary["status"] == RunStatus.WAITING_APPROVAL
    assert summary["pending"] == ["g", "w"]
    assert summary["warnings"] == [] and "notifier_warnings" not in summary["meta_keys"]
    assert [e[2] for e in log if e[0] == "rec.gov"] == ["approval.requested"]


async def test_client_tool_pause_and_clarification_free_finish_match() -> None:
    @tool(target="client")
    async def ask() -> str:
        return ""

    def make() -> Agent:
        return Agent(prompt="p", tools=[work, ask])

    responses = [_calls(ToolCall("work", id="w"), ToolCall("ask", id="a"))]
    (_, summary, _), (_, _, terminal) = await _both(make, responses)
    assert terminal == RunEventType.RUN_PAUSED
    assert summary["status"] == RunStatus.WAITING_CLIENT_TOOL
    assert summary["pending"] == ["a"]
    assert summary["meta_keys"] == ["pause_state", "notifier_warnings"]


async def test_max_iterations_matches() -> None:
    def make() -> Agent:
        return Agent(prompt="p", tools=[work], max_iterations=2)

    responses = [_calls(ToolCall("work", id="w1")), _calls(ToolCall("work", id="w2"))]
    (_, summary, _), (_, _, terminal) = await _both(make, responses)
    assert terminal == RunEventType.RUN_COMPLETED
    assert summary["status"] == RunStatus.MAX_ITERATIONS
    assert summary["iteration_count"] == 2


@pytest.mark.parametrize(
    "control,status,terminal",
    [
        (_Control(cancel=True), RunStatus.CANCELLED, RunEventType.RUN_CANCELLED),
        (_Control(pause=True), RunStatus.PAUSED, RunEventType.RUN_PAUSED),
    ],
)
async def test_top_of_iteration_control_matches(control, status, terminal) -> None:
    def make() -> Agent:
        return Agent(prompt="p", tools=[work])

    responses = [_calls(ToolCall("work", id="w")), _text("never")]
    # Reads: top of iteration 1, after the provider returned, top of iteration 2.
    script = [_Control(), _Control(), control]
    (_, summary, _), (_, _, got) = await _both(make, responses, script=script)
    assert got == terminal
    assert summary["status"] == status
    assert summary["iteration_count"] == 1
    assert "notifier_warnings" in summary["meta_keys"]


async def test_inbox_delivery_matches() -> None:
    def make() -> Agent:
        return Agent(prompt="p", tools=[work])

    responses = [_calls(ToolCall("work", id="w")), _text("done")]
    script = [_Control(), _Control(), _Control(notes=True)]
    (log, summary, _), (_, _, terminal) = await _both(make, responses, script=script, notes=["hi"])
    assert terminal == RunEventType.RUN_COMPLETED
    assert summary["status"] == RunStatus.SUCCESS
    assert ("rec.gov", 2, "inbox.delivered", {"note_id": "note0"}, "note0") in log


async def test_pending_limit_matches() -> None:
    def make() -> Agent:
        agent = Agent(prompt="p", tools=[work])
        agent._limits = replace(agent._limits, max_pending_calls=1)
        return agent

    responses = [_calls(ToolCall("work", id="a"), ToolCall("work", id="b"))]
    (_, summary, _), (_, _, terminal) = await _both(
        make, responses, script=[], storage_format=STORAGE_FORMAT_CHECKPOINTED
    )
    assert terminal == RunEventType.RUN_ERROR
    assert summary["failure_reason"] == FailureReason.PENDING_LIMIT.value


@pytest.mark.parametrize("warnings", [[], ["on_tool_completed failed at iteration 1"]])
def test_stream_meta_is_guardrail_meta_without_an_engine(warnings: list[str]) -> None:
    """The stream's hand-built meta dicts equal ``guardrail_meta(None, warnings)``."""
    hand_built = {"notifier_warnings": warnings} if warnings else {}
    built = guardrail_meta(None, warnings)
    assert built == hand_built and list(built) == list(hand_built)
    assert all(built[k] is hand_built[k] for k in built)
    pause = object()
    hand_pause: dict[str, Any] = {"pause_state": pause}
    if warnings:
        hand_pause["notifier_warnings"] = warnings
    shared = {"pause_state": pause, **guardrail_meta(None, warnings)}
    assert shared == hand_pause and list(shared) == list(hand_pause)
