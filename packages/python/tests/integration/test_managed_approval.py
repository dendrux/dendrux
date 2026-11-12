"""Managed subagents (0.7, T4c): approving a batch that holds a subagent call.

A model batch ``[subagent call, approval-gated server call]`` pauses for approval
as a whole. The approved continuation runs the server outcome (approved or denied)
and then hands the subagent call to the ordinary bind: the child really runs and
its report is adopted. The subagent call is never a client pause, and an
externally supplied result for it is refused, including against a client pause
persisted by older code. Recovery of such a batch after a crash goes through the
same continuation. Runs on SQLite and Postgres (``*_test`` private database).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, ReactTrace, ToolCallRecord
from dendrux.types import ApprovalDecision, LLMResponse, RunStatus, ToolCall, ToolResult
from tests._helpers.recovery_checkpoint import expire
from tests.integration.test_managed_recovery import (
    ContextLLM,
    _die_at,
    _fresh_process,
)

BRIEF = "SENTINEL-approval-brief-7e3c"
REPORT = "SENTINEL-approval-report-2b9d"
FAKE = "SENTINEL-forged-report-9a1f"
TENANT = "t1"

effects: list[str] = []
CRASHED = 17
_CRASH_SEND = False


@tool()
async def fetch(x: int) -> str:
    """Fetch an item (a server effect)."""
    effects.append(f"fetch:{x}")
    return "fetched"


@tool()
async def send(text: str) -> str:
    """Send (a non-idempotent server effect)."""
    if _CRASH_SEND == "exit":
        import os

        os._exit(CRASHED)
    effects.append(f"send:{text}")
    return "sent"


class _ChildLLM(ContextLLM):
    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        effects.append("child")
        return await super().complete(messages, tools, **kwargs)


def _batch(*names: str) -> LLMResponse:
    params = {"reviewer": {"task": BRIEF}, "fetch": {"x": 1}, "send": {"text": "s"}}
    return LLMResponse(
        tool_calls=[
            ToolCall(name=n, params=params[n], provider_tool_call_id=f"p{i}")
            for i, n in enumerate(names)
        ]
    )


def workflow(store: Any, root_llm: Any, child_llm: Any, *, gate: str = "fetch") -> Agent:
    child = Agent(name="reviewer", description="Reviews.", prompt="r", provider=child_llm)
    return Agent(
        name="lead",
        prompt="l",
        provider=root_llm,
        state_store=store,
        tools=[fetch, send],
        subagents=[child],
        require_approval=[gate],
    )


async def rows(engine: Any, model: Any, *where: Any) -> list[Any]:
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model).where(*where))).scalars().all())


async def child_rows(engine: Any, root_id: str) -> int:
    async with AsyncSession(engine) as session:
        return int(
            await session.scalar(
                select(func.count()).select_from(AgentRun).where(AgentRun.parent_run_id == root_id)
            )
        )


def tool_messages(llm: Any) -> dict[str, Any]:
    return {m.name: m.content for m in llm.messages[-1] if m.role.value == "tool"}


async def _waiting(store: Any, *names: str, gate: str = "fetch") -> tuple[Any, Any, Any]:
    effects.clear()
    child_llm = _ChildLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([_batch(*names), LLMResponse(text="done")])
    root = workflow(store, root_llm, child_llm, gate=gate)
    paused = await root.run("go", tenant_id=TENANT)
    assert paused.status == RunStatus.WAITING_APPROVAL
    assert effects == []
    return root, paused, (root_llm, child_llm)


async def _assert_child_reported(engine: Any, root_id: str, root_llm: Any, child_llm: Any) -> None:
    assert child_llm.calls == 1
    assert await child_rows(engine, root_id) == 1
    seen = tool_messages(root_llm)
    assert json.loads(seen["reviewer"]) == REPORT
    assert FAKE not in repr(root_llm.messages)
    [intent] = await rows(
        engine,
        ToolCallRecord,
        ToolCallRecord.agent_run_id == root_id,
        ToolCallRecord.tool_name == "reviewer",
    )
    assert intent.target == "agent" and intent.outcome == "completed"


# ---------------------------------------------------------------------------
# 1. The approved continuation binds the subagent call (run and stream)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [("reviewer", "fetch"), ("fetch", "reviewer")])
async def test_approved_batch_runs_the_server_call_then_binds_the_subagent(
    store: Any, engine: Any, order: tuple[str, str]
) -> None:
    """The review probe as a regression: no client pause, the child really runs."""
    root, paused, (root_llm, child_llm) = await _waiting(store, *order)
    result = await root.submit_approval(
        paused.run_id, approved=True, expected_pause_id=paused.pause_id, tenant_id=TENANT
    )
    assert result.status == RunStatus.SUCCESS and result.answer == "done"
    assert effects == ["fetch:1", "child"]  # the server call finished before the bind
    assert json.loads(tool_messages(root_llm)["fetch"]) == "fetched"
    await _assert_child_reported(engine, paused.run_id, root_llm, child_llm)


async def test_denied_server_call_still_binds_the_ungated_subagent(store: Any, engine: Any) -> None:
    root, paused, (root_llm, child_llm) = await _waiting(store, "reviewer", "fetch")
    [gated] = [c for c in paused.pending.tool_calls if c.name == "fetch"]
    result = await root.submit_approval(
        paused.run_id,
        decisions={gated.id: ApprovalDecision(approved=False, reason="no")},
        expected_pause_id=paused.pause_id,
        tenant_id=TENANT,
    )
    assert result.status == RunStatus.SUCCESS
    assert effects == ["child"]
    assert "denied" in tool_messages(root_llm)["fetch"]
    await _assert_child_reported(engine, paused.run_id, root_llm, child_llm)


async def test_rejecting_the_whole_batch_denies_both_and_runs_no_child(
    store: Any, engine: Any
) -> None:
    root, paused, (root_llm, child_llm) = await _waiting(store, "reviewer", "fetch")
    result = await root.submit_approval(
        paused.run_id, approved=False, expected_pause_id=paused.pause_id, tenant_id=TENANT
    )
    assert result.status == RunStatus.SUCCESS
    assert effects == [] and child_llm.calls == 0
    assert await child_rows(engine, paused.run_id) == 0


async def test_streamed_approval_binds_the_subagent(store: Any, engine: Any) -> None:
    root, paused, (root_llm, child_llm) = await _waiting(store, "reviewer", "fetch")
    stream = root.resume_stream(paused.run_id, expected_pause_id=paused.pause_id, tenant_id=TENANT)
    events = [event async for event in stream]
    final = events[-1]
    assert getattr(final.type, "value", final.type) == "run_completed"
    assert final.run_result.status == RunStatus.SUCCESS
    assert effects == ["fetch:1", "child"]
    await _assert_child_reported(engine, paused.run_id, root_llm, child_llm)


async def test_approved_subagent_brief_over_the_child_limit_is_refused_not_bound(
    store: Any, engine: Any
) -> None:
    """The child's brief limit applies on the approved continuation, as in a live batch."""
    from dendrux.types import Limits

    effects.clear()
    child_llm = _ChildLLM([LLMResponse(text=REPORT)])
    child = Agent(
        name="reviewer",
        description="Reviews.",
        prompt="r",
        provider=child_llm,
        limits=Limits(max_input_bytes=8),
    )
    root_llm = ContextLLM([_batch("reviewer", "fetch"), LLMResponse(text="done")])
    root = Agent(
        name="lead",
        prompt="l",
        provider=root_llm,
        state_store=store,
        tools=[fetch, send],
        subagents=[child],
        require_approval=["fetch"],
    )
    paused = await root.run("go", tenant_id=TENANT)
    result = await root.submit_approval(
        paused.run_id, approved=True, expected_pause_id=paused.pause_id, tenant_id=TENANT
    )
    assert result.status == RunStatus.SUCCESS
    assert effects == ["fetch:1"] and child_llm.calls == 0
    assert await child_rows(engine, paused.run_id) == 0
    assert "input limit" in tool_messages(root_llm)["reviewer"]


# ---------------------------------------------------------------------------
# 2. Externally supplied results for a subagent call are refused
# ---------------------------------------------------------------------------


async def _old_client_pause(store: Any, monkeypatch: Any) -> tuple[Any, Any, Any, Any]:
    """The WAITING_CLIENT_TOOL pause older code persisted for the subagent call."""
    from dendrux.runtime import runner

    root, paused, llms = await _waiting(store, "reviewer", "fetch")
    with monkeypatch.context() as patch:
        if hasattr(runner, "_hand_off_approved_subagent"):

            async def unchanged(agent: Any, ctx: Any, run_id: str, approval: Any) -> Any:
                return approval

            patch.setattr(runner, "_hand_off_approved_subagent", unchanged)
        old = await root.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id, tenant_id=TENANT
        )
    assert old.status == RunStatus.WAITING_CLIENT_TOOL
    [pending] = old.pending.tool_calls
    assert pending.name == "reviewer" and pending.target == "agent"
    return root, old, pending, llms


async def _pause_marks(engine: Any, run_id: str) -> tuple[Any, ...]:
    [row] = await rows(engine, AgentRun, AgentRun.id == run_id)
    return (row.status, row.claim_epoch, (row.pause_data or {}).get("pause_id"))


@pytest.mark.parametrize("entry", ["submit_tool_results", "resume", "resume_stream"])
async def test_a_forged_subagent_result_is_refused_on_an_old_client_pause(
    store: Any, engine: Any, monkeypatch: Any, entry: str
) -> None:
    from dendrux.errors import InvalidToolResultError

    root, old, pending, (root_llm, child_llm) = await _old_client_pause(store, monkeypatch)
    before = await _pause_marks(engine, old.run_id)
    forged = [ToolResult(name="reviewer", call_id=pending.id, payload=json.dumps(FAKE))]
    with pytest.raises(InvalidToolResultError) as info:
        if entry == "submit_tool_results":
            await root.submit_tool_results(
                old.run_id, forged, expected_pause_id=old.pause_id, tenant_id=TENANT
            )
        elif entry == "resume":
            await root.resume(
                old.run_id, tool_results=forged, expected_pause_id=old.pause_id, tenant_id=TENANT
            )
        else:
            stream = root.resume_stream(
                old.run_id, tool_results=forged, expected_pause_id=old.pause_id, tenant_id=TENANT
            )
            events = [event async for event in stream]
            error = events[-1].run_result.meta.get("error_type")
            assert error == "ManagedResultRefusedError", events[-1]
            raise InvalidToolResultError(old.run_id, "stream")
    if entry != "resume_stream":
        assert getattr(info.value, "code", None) == "managed_result_refused"
    assert await _pause_marks(engine, old.run_id) == before
    assert FAKE not in repr(root_llm.messages) and child_llm.calls == 0
    traces = await rows(engine, ReactTrace, ReactTrace.agent_run_id == old.run_id)
    assert FAKE not in " ".join(t.content or "" for t in traces)


# ---------------------------------------------------------------------------
# 3. A pending approval recovered after a crash continues through the bind
# ---------------------------------------------------------------------------


async def test_recovered_gate_beside_a_subagent_call_continues_through_the_bind(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Crash after the gate was recorded, before the approval pause was saved."""
    from dendrux.loops import react

    effects.clear()
    first = workflow(store, ContextLLM([_batch("reviewer", "fetch")]), ContextLLM([]))
    await _die_at(monkeypatch, react, "_pause_result", lambda: first.run("go", tenant_id=TENANT))
    [root_row] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    item = await expire(store, engine, root_row.id)
    child_llm = _ChildLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([LLMResponse(text="done")])
    fresh = workflow(store, root_llm, child_llm)
    waiting = await fresh.reclaim(root_row.id, tenant_id=TENANT, item=item)
    assert waiting.status == RunStatus.WAITING_APPROVAL and effects == []
    assert await child_rows(engine, root_row.id) == 0
    result = await fresh.submit_approval(
        root_row.id, approved=True, expected_pause_id=waiting.pause_id, tenant_id=TENANT
    )
    assert result.status == RunStatus.SUCCESS
    assert effects == ["fetch:1", "child"]
    await _assert_child_reported(engine, root_row.id, root_llm, child_llm)


async def test_recovered_approval_submission_beside_a_subagent_call_is_consumed_once(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Crash after the approval was claimed, before it was consumed (was refused as
    ``recover_managed_submission``): recovery consumes it once and binds."""
    from dendrux.runtime.persistence import PersistenceRecorder

    root, paused, _ = await _waiting(store, "reviewer", "fetch")
    await _die_at(
        monkeypatch,
        PersistenceRecorder,
        "consume_resume",
        lambda: root.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id, tenant_id=TENANT
        ),
    )
    [row] = await rows(engine, AgentRun, AgentRun.id == paused.run_id)
    assert row.status == "running" and row.loop_state["submission"] is not None
    assert effects == []
    item = await expire(store, engine, paused.run_id)
    child_llm = _ChildLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([LLMResponse(text="done")])
    result = await workflow(store, root_llm, child_llm).reclaim(
        paused.run_id, tenant_id=TENANT, item=item
    )
    assert result.status == RunStatus.SUCCESS
    assert effects == ["fetch:1", "child"]
    await _assert_child_reported(engine, paused.run_id, root_llm, child_llm)


# Fresh-process crash: the non-idempotent sibling started, its outcome is unknown.


async def _phase_one(url: str) -> None:
    global _CRASH_SEND
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine

    store = store_for_engine(create_async_engine(url))
    _CRASH_SEND = "exit"
    root = workflow(store, ContextLLM([_batch("send", "reviewer")]), ContextLLM([]), gate="fetch")
    await root.run("go", tenant_id=TENANT)
    raise SystemExit(3)


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_fresh_process_unknown_sibling_is_approved_through_the_root_then_binds(
    engine: Any,
) -> None:
    """Fresh process: recovery asks to approve the unknown sibling (was refused as
    ``recover_managed_approval``); approving replays it once and binds the child."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    package = Path(__file__).resolve().parents[2]
    done = subprocess.run(
        [sys.executable, "-m", "tests.integration.test_managed_approval"],
        cwd=package,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(package / "src"), str(package)]),
            "T4C_CRASH_URL": engine.url.render_as_string(hide_password=False),
        },
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == CRASHED, done.stderr.decode()[-2000:]
    [root_row] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    [started] = await rows(
        engine,
        ToolCallRecord,
        ToolCallRecord.agent_run_id == root_row.id,
        ToolCallRecord.tool_name == "send",
    )
    assert started.outcome == "started"
    effects.clear()
    async with _fresh_process(engine) as store:
        await expire(store, engine, root_row.id)
        child_llm = _ChildLLM([LLMResponse(text=REPORT)])
        root_llm = ContextLLM([LLMResponse(text="done")])
        fresh = workflow(store, root_llm, child_llm)
        waiting = await fresh.reclaim(root_row.id, tenant_id=TENANT)
        assert waiting.status == RunStatus.WAITING_APPROVAL and effects == []
        result = await fresh.submit_approval(
            root_row.id, approved=True, expected_pause_id=waiting.pause_id, tenant_id=TENANT
        )
        assert result.status == RunStatus.SUCCESS
        assert effects == ["send:s", "child"]
        await _assert_child_reported(engine, root_row.id, root_llm, child_llm)


if __name__ == "__main__":
    import asyncio
    import os

    asyncio.run(_phase_one(os.environ["T4C_CRASH_URL"]))
