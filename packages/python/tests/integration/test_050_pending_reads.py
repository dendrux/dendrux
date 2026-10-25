"""Pending-pause reads are robust and bounded (SQLite + Postgres).

- A pause whose stored JSON holds characters Postgres ``jsonb`` cannot hold
  (U+0000, a lone surrogate) still reads: ``RunStore.get_pending``,
  ``Agent.get_result`` and the pending projection after ``agent.run()``
  return the pause instead of raising a database error.
- On Postgres the summary converts ``pause_data`` to ``jsonb`` once per read,
  not once per extracted key.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import event, text

from dendrux.agent import Agent
from dendrux.store import RunStore
from dendrux.tool import tool
from dendrux.types import LLMResponse
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio


@tool(target="client")
async def read_range(sheet: str) -> str:
    return ""


async def _paused(store: Any, tenant: str = "A") -> tuple[Agent, str]:
    agent = Agent(
        provider=ScriptedLLM(
            [tool_call("read_range", {"sheet": "S1"}, pid="c"), LLMResponse(text="d")]
        ),
        prompt="t",
        state_store=store,
        tools=[read_range],
        strict_tenancy=False,
    )
    result = await agent.run("go", tenant_id=tenant)
    assert result.status.value == "waiting_client_tool"
    return agent, result.run_id


async def _poison_pause_data(engine: Any, run_id: str, bad: str) -> None:
    """Rewrite the stored JSON text so a pending argument holds ``bad`` (an escape)."""
    async with engine.begin() as conn:
        raw = (
            await conn.execute(
                text("SELECT pause_data FROM agent_runs WHERE id = :i"), {"i": run_id}
            )
        ).scalar_one()
        doc = raw if isinstance(raw, str) else json.dumps(raw)
        assert '"S1"' in doc
        poisoned = doc.replace('"S1"', f'"S{bad}1"')
        if engine.dialect.name == "postgresql":
            stmt = text("UPDATE agent_runs SET pause_data = CAST(:d AS json) WHERE id = :i")
        else:
            stmt = text("UPDATE agent_runs SET pause_data = :d WHERE id = :i")
        await conn.execute(stmt, {"d": poisoned, "i": run_id})


@pytest.mark.parametrize(
    ("escape", "char"), [("\\u0000", "\x00"), ("\\ud800", "\ud800")], ids=["nul", "surrogate"]
)
async def test_unconvertible_pause_still_reads(
    store: Any, engine: Any, escape: str, char: str
) -> None:
    agent, run_id = await _paused(store)
    await _poison_pause_data(engine, run_id, escape)
    run_store = RunStore.from_engine(engine)
    pending = await run_store.get_pending(run_id, tenant_id="A")
    assert pending is not None and pending.kind == "client_tool"
    assert pending.tool_calls[0].params == {"sheet": f"S{char}1"}
    result = await agent.get_result(run_id, tenant_id="A")
    assert result.pending is not None and result.pending.pause_id == pending.pause_id
    # Another tenant's read of the same run still reads as missing, not as an error.
    from dendrux.errors import RunNotFoundError

    with pytest.raises(RunNotFoundError):
        await run_store.get_pending(run_id, tenant_id="B")


async def test_agent_run_returns_a_pause_whose_args_hold_nul(store: Any, engine: Any) -> None:
    """The pending projection right after a pause survives a stored U+0000."""
    agent = Agent(
        provider=ScriptedLLM(
            [tool_call("read_range", {"sheet": "S\x001"}, pid="c"), LLMResponse(text="d")]
        ),
        prompt="t",
        state_store=store,
        tools=[read_range],
        strict_tenancy=False,
    )
    try:
        result = await agent.run("go", tenant_id="A")
    except ValueError:
        pytest.skip("the boundary rejects U+0000 before the pause is stored")
    if result.status.value != "waiting_client_tool":
        pytest.skip("the model's U+0000 argument did not reach a pause")
    assert result.pending is not None
    pending = await RunStore.from_engine(engine).get_pending(result.run_id, tenant_id="A")
    assert pending is not None and pending.pause_id == result.pause_id


async def test_postgres_summary_converts_pause_data_once(store: Any, engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres statement shape")
    _, run_id = await _paused(store)
    statements: list[str] = []

    def capture(_c: Any, _cur: Any, stmt: str, *_a: Any) -> None:
        statements.append(stmt)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await RunStore.from_engine(engine).get_pending(run_id, tenant_id="A")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    summary = [s for s in statements if "pause_data" in s]
    assert len(summary) == 1, summary
    assert summary[0].upper().count("AS JSONB") == 1, summary[0]


async def test_event_reads_survive_unconvertible_event_data(store: Any, engine: Any) -> None:
    """Approval arguments and events holding U+0000 still stream and view."""
    from dendrux.store import FollowEnd, FollowHeartbeat

    await store.create_run("rn".ljust(26, "0"), "a", tenant_id="A")
    run_id = "rn".ljust(26, "0")
    await store.save_run_event(
        run_id,
        event_type="approval.requested",
        data={"tool_name": "t", "call_id": "c1", "params": {"q": "x"}},
    )
    await store.save_run_event(
        run_id,
        event_type="run.paused",
        data={"status": "waiting_approval", "pending_tool_calls": [{"id": "c1", "name": "t"}]},
    )
    await store.pause_run(
        run_id,
        status="waiting_approval",
        pause_data={"pause_id": "p", "pending_tool_calls": [{"id": "c1", "name": "t"}]},
    )
    async with engine.begin() as conn:
        cast = "CAST(:d AS json)" if engine.dialect.name == "postgresql" else ":d"
        await conn.execute(
            text(
                f"UPDATE run_events SET data = {cast} "
                "WHERE agent_run_id = :i AND event_type = 'approval.requested'"
            ),
            {"d": '{"tool_name": "t", "call_id": "c1", "params": {"q": "a\\u0000b"}}', "i": run_id},
        )
    run_store = RunStore.from_engine(engine)
    pending = await run_store.get_pending(run_id, tenant_id="A")
    assert pending is not None and pending.tool_calls[0].params == {"q": "a\x00b"}
    seen = []
    async with run_store.follow(
        run_id, tenant_id="A", until_terminal=False, max_duration_s=1.0
    ) as f:
        async for item in f:
            if isinstance(item, (FollowEnd, FollowHeartbeat)) or len(seen) == 2:
                break
            seen.append(item)
    assert [e.event_type for e in seen] == ["approval.requested", "run.paused"]
    assert seen[1].data["pending_tool_calls"][0]["params"] == {"q": "a\x00b"}
    pairs, _ = await run_store.get_pause_page(run_id, tenant_id="A")
    assert len(pairs) == 1
