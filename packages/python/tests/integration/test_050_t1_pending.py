"""0.5.0 T1 change round on the backend matrix (SQLite + Postgres).

- ``RunStore.get_pending``: every pause kind, the bounded JSON extraction on
  both dialects, and the fail-closed tenant scope.
- B2 amendment: with the router defaults (``include_payloads=False``) the
  client-tools flow and the approval flow still see the active pause's
  arguments; past pauses and other payloads stay hidden.
- ``RunStore.follow`` under concurrency: many followers of one run share one
  poller and all receive every event.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dendrux.errors import RunNotFoundError
from dendrux.http import make_read_router
from dendrux.store import UNSCOPED, EventFeed, FollowEnd, RunStore, StoredEvent, redact_event
from dendrux.types import ToolResult
from tests.integration.test_tenant_run_control import _pause

pytestmark = pytest.mark.asyncio


@pytest.fixture
def run_store(engine: Any) -> RunStore:
    return RunStore.from_engine(engine)


async def _allow() -> None:
    return None


def _client(run_store: RunStore) -> AsyncClient:
    app = FastAPI()
    app.include_router(
        make_read_router(store=run_store, authorize=_allow, tenant_resolver=UNSCOPED)
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


# ---------------------------------------------------------------------------
# get_pending
# ---------------------------------------------------------------------------


async def test_client_tool_pause(store: Any, run_store: RunStore) -> None:
    paused = await _pause("client", store, "acme")
    pending = await run_store.get_pending(paused.run_id, tenant_id="acme")
    assert pending is not None
    assert (pending.kind, pending.status) == ("client_tool", "waiting_client_tool")
    (call,) = pending.tool_calls
    assert (call.name, call.target, call.params, call.approval) == (
        "read_range",
        "client",
        {"sheet": "S1"},
        None,
    )
    raw = await store.get_pause_state(paused.run_id)
    assert call.id == raw["pending_tool_calls"][0]["id"]
    assert pending.to_dict()["pending_tool_calls"][0]["params"] == {"sheet": "S1"}


async def test_approval_pause_carries_args_and_gate(store: Any, run_store: RunStore) -> None:
    paused = await _pause("approval", store, None)
    pending = await run_store.get_pending(paused.run_id, tenant_id=None)
    assert pending is not None and pending.kind == "approval"
    (call,) = pending.tool_calls
    assert (call.name, call.params, call.args_hidden) == ("refund", None, True)
    assert call.approval is not None and call.approval == {
        "source": "static",
        "reason": "requires_approval",
    }


async def test_requested_pause_has_reason(store: Any, run_store: RunStore) -> None:
    paused = await _pause("paused", store, None)
    pending = await run_store.get_pending(paused.run_id, tenant_id=UNSCOPED)
    assert pending is not None
    assert (pending.kind, pending.reason) == ("paused", "requested")


async def test_input_pause_has_the_question(store: Any, run_store: RunStore) -> None:
    await store.create_run("ri", "agent", tenant_id="t")
    await store.pause_run(
        "ri",
        status="waiting_human_input",
        pause_data={
            "agent_name": "agent",
            "pending_tool_calls": [],
            "history": [{"role": "user", "content": "x" * 5000}],
            "steps": [
                {"reasoning": None, "action": {"type": "finish", "answer": "a"}},
                {
                    "reasoning": None,
                    "action": {"type": "clarification", "question": "Which?", "options": ["A"]},
                },
            ],
            "pause_message": "[note] brackets are kept",
        },
    )
    pending = await run_store.get_pending("ri", tenant_id="t")
    assert pending is not None
    assert (pending.kind, pending.question, pending.options) == ("input", "Which?", ["A"])
    assert pending.message == "[note] brackets are kept"


async def test_not_paused_is_none(store: Any, run_store: RunStore) -> None:
    await store.create_run("rr", "agent")
    assert await run_store.get_pending("rr", tenant_id=None) is None
    await store.finalize_run("rr", status="success")
    assert await run_store.get_pending("rr", tenant_id=None) is None


async def test_scope_fails_closed(store: Any, run_store: RunStore) -> None:
    paused = await _pause("client", store, "acme")
    for tenant in ("other", None):
        with pytest.raises(RunNotFoundError):
            await run_store.get_pending(paused.run_id, tenant_id=tenant)
    with pytest.raises(RunNotFoundError):
        await run_store.get_pending("missing", tenant_id="acme")
    with pytest.raises(ValueError, match="tenant_id"):
        await run_store.get_pending(paused.run_id, tenant_id="")
    with pytest.raises(TypeError):
        await run_store.get_pending(paused.run_id)  # type: ignore[call-arg]


async def test_pending_read_does_not_load_history(
    store: Any, run_store: RunStore, engine: Any
) -> None:
    """One statement, and the pause history column is never selected whole."""
    from sqlalchemy import event

    paused = await _pause("client", store, None)
    statements: list[str] = []

    def capture(_c: Any, _cur: Any, statement: str, *_a: Any) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await run_store.get_pending(paused.run_id, tenant_id=None)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1
    assert "history" not in selects[0]


# ---------------------------------------------------------------------------
# B2 amendment: the default router still serves the client-tools and approval flows
# ---------------------------------------------------------------------------


def _sse_events(body: str) -> list[dict[str, Any]]:
    out = []
    for block in body.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in lines and lines.get("event", "message") == "message":
            out.append(json.loads(lines["data"]))
    return out


async def test_client_tools_flow_with_defaults(store: Any, run_store: RunStore) -> None:
    paused = await _pause("client", store, None)
    run_id = paused.run_id
    async with _client(run_store) as http:
        events = (await http.get(f"/runs/{run_id}/events")).json()["items"]
        call = next(e for e in events if e["event_type"] == "run.paused")["data"][
            "pending_tool_calls"
        ][0]
        assert call["params"] == {
            "sheet": "S1"
        }  # what executeLocally(call.name, call.params) needs
        pauses = (await http.get(f"/runs/{run_id}/pauses")).json()["items"]
        assert pauses[-1]["pending_tool_calls"][0]["params"] == {"sheet": "S1"}

        result = await paused.agent.submit_tool_results(
            run_id, [ToolResult(name="read_range", call_id=call["id"], payload='"rows"')]
        )
        assert result.status.value == "success"

        # The pause is history now: its arguments are hidden again.
        events = (await http.get(f"/runs/{run_id}/events")).json()["items"]
        past = next(e for e in events if e["event_type"] == "run.paused")
        assert "params" not in past["data"]["pending_tool_calls"][0]
        pauses = (await http.get(f"/runs/{run_id}/pauses")).json()["items"]
        assert "params" not in pauses[-1]["pending_tool_calls"][0]
        stream = (await http.get(f"/runs/{run_id}/events/stream")).text
        paused_frame = next(e for e in _sse_events(stream) if e["event_type"] == "run.paused")
        assert "params" not in paused_frame["data"]["pending_tool_calls"][0]


async def test_approval_flow_with_defaults(store: Any, run_store: RunStore) -> None:
    paused = await _pause("approval", store, None)
    run_id = paused.run_id
    async with _client(run_store) as http:
        events = (await http.get(f"/runs/{run_id}/events")).json()["items"]
        data = next(e for e in events if e["event_type"] == "run.paused")["data"]
        assert data["status"] == "waiting_approval"
        assert data["pending_tool_calls"][0]["params"] is None
        assert data["pending_tool_calls"][0]["args_hidden"] is True
    pending = await run_store.get_pending(run_id, tenant_id=UNSCOPED)
    assert pending is not None and pending.tool_calls[0].params is None
    assert pending.tool_calls[0].args_hidden is True
    result = await paused.agent.submit_approval(run_id, approved=True)
    assert result.status.value == "success"


async def test_active_pause_args_reach_a_live_sse_client(store: Any, run_store: RunStore) -> None:
    paused = await _pause("client", store, None)
    follow = run_store.follow(paused.run_id, tenant_id=None)
    async with follow as f:
        items = []
        while True:
            item = await asyncio.wait_for(anext(f), 10)
            if isinstance(item, StoredEvent):
                items.append(item)
                if item.event_type == "run.paused":
                    break
    call = items[-1].data["pending_tool_calls"][0]
    assert call["params"] == {"sheet": "S1"}


# ---------------------------------------------------------------------------
# follow under concurrency
# ---------------------------------------------------------------------------


async def test_many_followers_one_poller(store: Any, run_store: RunStore) -> None:
    await store.create_run("r1", "agent", tenant_id="T")
    feed = EventFeed(run_store, poll_interval_s=0.01, poll_max_s=0.05, status_every=1)
    polls = {"n": 0}
    original = run_store.get_events

    async def counting(run_id: str, **kw: Any) -> Any:
        polls["n"] += 1
        return await original(run_id, **kw)

    run_store.get_events = counting  # type: ignore[method-assign]
    n_followers, n_events = 60, 30

    async def follower() -> list[Any]:
        async with feed.follow("r1", tenant_id="T") as f:
            return [item async for item in f]

    tasks = [asyncio.create_task(follower()) for _ in range(n_followers)]
    try:
        async with asyncio.timeout(20):
            while feed.followers < n_followers:
                await asyncio.sleep(0.01)
        assert feed.watched_runs == 1
        before = polls["n"]
        await asyncio.sleep(0.2)
        assert polls["n"] - before < 60  # idle polling is per run, not per follower

        async def writer(i: int) -> None:
            await store.save_run_event("r1", event_type=f"e{i}")

        await asyncio.gather(*(writer(i) for i in range(n_events)))
        await store.finalize_run("r1", status="success")
        results = await asyncio.wait_for(asyncio.gather(*tasks), 30)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for items in results:
        assert [i.sequence_index for i in items if isinstance(i, StoredEvent)] == list(
            range(n_events)
        )
        assert items[-1] == FollowEnd("terminal", n_events - 1)
    assert feed.watched_runs == 0 and feed.followers == 0


# ---------------------------------------------------------------------------
# Approval arguments are shown as the model sent them (placeholders), never
# de-anonymised, on every read path; the approved tool still gets real values.
# ---------------------------------------------------------------------------


async def test_pii_approval_pause_never_shows_real_values(store: Any, run_store: RunStore) -> None:
    from dendrux import Agent, tool
    from dendrux.guardrails import PII
    from dendrux.llm.mock import MockLLM
    from dendrux.types import LLMResponse, ToolCall

    received: list[str] = []

    @tool()
    async def send_mail(to: str) -> str:
        received.append(to)
        return "sent"

    llm = MockLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(name="send_mail", params={"to": "<<EMAIL_ADDRESS_1>>"})]
            ),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        provider=llm,
        prompt="p",
        tools=[send_mail],
        require_approval=["send_mail"],
        guardrails=[PII()],
        state_store=store,
    )
    # Emulate T4's producer contract; this branch's loop has no params yet.
    original_save = store.save_run_event

    async def save_with_model_args(run_id: str, **kw: Any) -> int:
        if kw["event_type"] == "approval.requested":
            kw["data"] = {**kw["data"], "params": {"to": "<<EMAIL_ADDRESS_1>>"}}
        return await original_save(run_id, **kw)

    store.save_run_event = save_with_model_args
    real = "bob@example.com"
    paused = await agent.run(f"mail {real} please", tenant_id="acme")
    run_id = paused.run_id
    assert paused.status.value == "waiting_approval"
    # The stored pause holds the de-anonymised value (that is what will execute).
    assert (await store.get_pause_state(run_id))["pending_tool_calls"][0]["params"]["to"] == real

    # Developer-owned read route recipe: get_events -> redact_event(active_pause=True).
    raw_events = await run_store.get_events(run_id, tenant_id="acme")
    raw_pause = next(event for event in raw_events if event.event_type == "run.paused")
    assert real in json.dumps(raw_pause.data)
    recipe_view = redact_event(raw_pause, active_pause=True)
    assert recipe_view.data["pending_tool_calls"][0]["params"] is None
    assert recipe_view.data["pending_tool_calls"][0]["args_hidden"] is True
    assert real not in json.dumps(recipe_view.data)
    for include_payloads in (False, True):
        direct = redact_event(raw_pause, active_pause=True, include_payloads=include_payloads)
        assert direct.data["pending_tool_calls"][0]["params"] is None
        assert direct.data["pending_tool_calls"][0]["args_hidden"] is True
        assert real not in json.dumps(direct.data)
        scoped = await run_store.get_event_view(
            run_id, raw_pause, tenant_id="acme", include_payloads=include_payloads
        )
        assert scoped.data["pending_tool_calls"][0]["params"] == {"to": "<<EMAIL_ADDRESS_1>>"}
        assert real not in json.dumps(scoped.data)
        with pytest.raises(RunNotFoundError):
            await run_store.get_event_view(run_id, raw_pause, tenant_id="other")

    pending = await run_store.get_pending(run_id, tenant_id="acme")
    assert pending is not None
    assert pending.tool_calls[0].params == {"to": "<<EMAIL_ADDRESS_1>>"}
    assert real not in json.dumps(pending.to_dict())

    for include_payloads in (False, True):
        app = FastAPI()
        app.include_router(
            make_read_router(
                store=run_store,
                authorize=_allow,
                tenant_resolver=lambda: "acme",
                include_payloads=include_payloads,
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as http:
            events = (await http.get(f"/runs/{run_id}/events")).json()["items"]
            data = next(e for e in events if e["event_type"] == "run.paused")["data"]
            assert data["pending_tool_calls"][0]["params"] == {"to": "<<EMAIL_ADDRESS_1>>"}
            assert real not in json.dumps(events)
            pauses = (await http.get(f"/runs/{run_id}/pauses")).json()["items"]
            assert pauses[-1]["pending_tool_calls"][0]["params"] == {"to": "<<EMAIL_ADDRESS_1>>"}
            assert real not in json.dumps(pauses)

    follow = run_store.follow(run_id, tenant_id="acme", max_duration_s=0.5)
    async with follow as f:
        items = [item async for item in f]
    shown = [i for i in items if isinstance(i, StoredEvent)]
    assert any(i.event_type == "run.paused" for i in shown)
    assert real not in json.dumps([i.data for i in shown])

    result = await agent.submit_approval(run_id, approved=True, tenant_id="acme")
    assert result.status.value == "success"
    assert received == [real]  # the approved tool still receives the real value


@pytest.mark.parametrize(
    "status", ["waiting_approval", "waiting_human_input", "waiting_client_tool"]
)
@pytest.mark.parametrize("recorded,mapped", [(True, True), (False, True), (False, False)])
async def test_pause_display_uses_recorded_args_on_every_path(
    store: Any,
    run_store: RunStore,
    status: str,
    recorded: bool,
    mapped: bool,
) -> None:
    from dendrux.http.read_router import _sse_frames

    await store.create_run("display", "a", tenant_id="T")
    # Reused call IDs from older pauses must never supply the new approval view.
    await store.save_run_event(
        "display",
        event_type="approval.requested",
        data={"call_id": "c", "params": {"old": "<<OLD>>"}},
    )
    await store.save_run_event("display", event_type="run.paused", data={})
    await store.save_run_event("display", event_type="run.resumed", data={})
    await store.save_run_event(
        "display",
        event_type="approval.requested",
        data={"call_id": "c", "params": {"superseded": True}},
    )
    model = {"phone": "<<PHONE_1>>", "literal": "Bobcat", "email": "<<EMAIL_1>>"}
    data = {"call_id": "c", **({"params": model} if recorded else {})}
    await store.save_run_event("display", event_type="approval.requested", data=data)
    execution = {"phone": 1234567890, "literal": "Bobcat", "email": "bob@example.com"}
    pause = {"pending_tool_calls": [{"id": "c", "name": "send", "params": execution}]}
    await store.pause_run(
        "display",
        status=status,
        pause_data=pause,
        pii_mapping={
            "<<PHONE_1>>": "+1 234 567 890",
            "<<EMAIL_1>>": "Bob@Example.com",
            "<<PERSON_1>>": "Bob",
        }
        if mapped
        else {},
    )
    await store.save_run_event("display", event_type="run.paused", data={"status": status, **pause})
    pending = await run_store.get_pending("display", tenant_id="T")
    expected = execution if status == "waiting_client_tool" else model if recorded else None
    assert pending.tool_calls[0].params == expected
    assert pending.tool_calls[0].args_hidden == (expected is None)
    for payloads in (False, True):
        app = FastAPI()
        app.include_router(
            make_read_router(
                store=run_store,
                authorize=_allow,
                tenant_resolver=lambda: "T",
                include_payloads=payloads,
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as http:
            events = (await http.get("/runs/display/events")).json()["items"]
            pair = (await http.get("/runs/display/pauses")).json()["items"][-1]
        call = events[-1]["data"]["pending_tool_calls"][0]
        assert call["params"] == expected
        assert pair["pending_tool_calls"][0]["params"] == expected
        if expected is None:
            assert call["args_hidden"] is True
        async with run_store.follow(
            "display", tenant_id="T", include_payloads=payloads, max_duration_s=0.1
        ) as follow:
            frames = [frame async for frame in _sse_frames(follow)]
        shown = [json.loads(f.split("data: ")[1]) for f in frames if "event: message\n" in f]
        assert shown[-1]["event_type"] == "run.paused"
        assert shown[-1]["data"]["pending_tool_calls"][0]["params"] == expected


async def test_pause_page_pairs_at_boundary_and_reports_truncation(
    store: Any, run_store: RunStore
) -> None:
    await store.create_run("pages", "a")
    for i in range(5):
        await store.save_run_event("pages", event_type="run.paused", data={"pause_id": str(i)})
        await store.save_run_event("pages", event_type="run.resumed")
    items, truncated = await run_store.get_pause_page("pages", limit=2)
    assert truncated and len(items) == 2
    assert [p.resume_sequence_index for p in items] == [1, 3]
    items, truncated = await run_store.get_pause_page("pages", limit=5)
    assert len(items) == 5 and not truncated
    async with _client(run_store) as http:
        body = (await http.get("/runs/pages/pauses?limit=2")).json()
        assert body["truncated"] and body["limit"] == 2 and len(body["items"]) == 2
        assert (await http.get("/runs/pages/pauses?limit=1001")).status_code == 422


async def test_delegation_traversal_bounds_and_projection(store: Any, engine: Any) -> None:
    from sqlalchemy import event

    await store.create_run("root", "a")
    for i in range(6):
        await store.create_run(f"c{i}", "a", parent_run_id="root")
    await store.create_run("leaf", "a", parent_run_id="c0")
    queries = []

    def capture(_c: Any, _cur: Any, stmt: str, *a: Any) -> None:
        queries.append(stmt)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        bounded = await store.get_delegation_info("root", max_rows=3)
        assert bounded.truncated and len(bounded.children) == 2
        assert bounded.subtree_summary.descendant_count == 2
        depth = await store.get_delegation_info("root", max_depth=1)
        assert depth.truncated and depth.subtree_summary.descendant_count == 6
        whole = await store.get_delegation_info("root")
        assert not whole.truncated and whole.subtree_summary.descendant_count == 7
        ancestors = await store.get_delegation_info("leaf", max_depth=1)
        assert ancestors.truncated and not ancestors.ancestry_complete
        root_only = await store.get_delegation_info("root", max_rows=1)
        assert root_only.truncated and not root_only.children
        leaf = await store.get_delegation_info("leaf", max_rows=3)
        assert not leaf.truncated and leaf.ancestry_complete
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert not any("pause_data" in q or "pii_mapping" in q or "input_data" in q for q in queries)
    for kw in ({"max_rows": 0}, {"max_rows": 10001}, {"max_depth": -1}, {"max_depth": 101}):
        with pytest.raises(ValueError):
            await store.get_delegation_info("root", **kw)
