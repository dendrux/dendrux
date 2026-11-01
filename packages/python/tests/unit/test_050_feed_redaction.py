"""Redacted feeds hide free text and past arguments in every event type.

``redact_event(include_payloads=False)`` (and so ``RunStore.follow``,
``get_event_view`` and ``get_pause_page``) must hide:

- the arguments of a **past** approval request (``approval.requested.params``);
- human or client text: ``run.resumed.user_input``, approval denial reasons,
  a requested pause's message on a past pause;
- exception text (``run.error.error``, ``mcp.error.error``) and image
  descriptions (``image.state.described[].text``).

Codes, counts, ids and Dendrux's own constant reasons stay visible.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import EventFeed, FollowEnd, FollowHeartbeat, RunStore, StoredEvent, redact_event

if TYPE_CHECKING:
    from dendrux.runtime.stores.sql.base import SQLStateStore

SECRET = "SECRET-TEXT-4242"


def _event(event_type: str, data: dict[str, Any], seq: int = 0) -> StoredEvent:
    return StoredEvent(
        sequence_index=seq,
        iteration_index=0,
        event_type=event_type,
        correlation_id=None,
        data=data,
        created_at=None,
    )


class TestRedactFreeText:
    def test_past_approval_request_hides_params(self) -> None:
        ev = _event(
            "approval.requested",
            {
                "tool_name": "sql",
                "call_id": "c1",
                "reason": "requires_approval",
                "params": {"q": SECRET},
            },
        )
        shown = redact_event(ev).data
        assert "params" not in shown
        assert shown == {"tool_name": "<unknown_tool>", "reason": "requires_approval"}
        assert redact_event(ev, active_pause=True).data["params"] == {"q": SECRET}
        assert redact_event(ev, include_payloads=True).data["params"] == {"q": SECRET}

    def test_resumed_user_input_hidden(self) -> None:
        ev = _event("run.resumed", {"resumed_from": "waiting_human_input", "user_input": SECRET})
        assert redact_event(ev).data == {"resumed_from": "waiting_human_input"}
        assert redact_event(ev, include_payloads=True).data["user_input"] == SECRET

    def test_decided_reasons_hidden_but_constants_kept(self) -> None:
        ev = _event(
            "approval.decided",
            {
                "decision": "mixed",
                "run_id": "r",
                "decisions": [
                    {"call_id": "a", "tool_name": "t", "decision": "denied", "reason": SECRET},
                    {
                        "call_id": "b",
                        "tool_name": "t",
                        "decision": "denied",
                        "reason": "denied_by_policy",
                    },
                    {
                        "call_id": "c",
                        "tool_name": "t",
                        "decision": "denied",
                        "reason": "User declined to run this tool.",
                    },
                    {"call_id": "d", "tool_name": "t", "decision": "approved", "reason": None},
                ],
            },
        )
        data = redact_event(ev).data
        assert SECRET not in json.dumps(data)
        reasons = [d.get("reason") for d in data["decisions"]]
        assert reasons == [None, "denied_by_policy", "User declined to run this tool.", None]
        assert all(d["decision"] for d in data["decisions"])
        assert redact_event(ev, include_payloads=True).data["decisions"][0]["reason"] == SECRET
        assert ev.data["decisions"][0]["reason"] == SECRET  # input never mutated

    def test_error_text_hidden_codes_kept(self) -> None:
        err = _event("run.error", {"error": SECRET, "error_info": {"category": "rate_limit"}})
        assert redact_event(err).data == {"error_info": {"category": "rate_limit"}}
        mcp = _event("mcp.error", {"error": SECRET, "error_type": "E", "code": "connection_failed"})
        assert redact_event(mcp).data == {"code": "connection_failed"}

    def test_image_descriptions_hidden(self) -> None:
        ev = _event(
            "image.state",
            {"described": [{"sha256": "ab" * 32, "text": SECRET}], "placeholders": []},
        )
        assert redact_event(ev).data == {"described": [{"sha256": "ab" * 32}], "placeholders": []}

    def test_pause_message_hidden_codes_kept(self) -> None:
        ev = _event(
            "run.paused",
            {
                "status": "paused",
                "reason": "requested",
                "message": SECRET,
                "pending_tool_calls": [],
            },
        )
        assert "message" not in redact_event(ev).data
        assert redact_event(ev).data["reason"] == "requested"
        assert "message" not in redact_event(ev, active_pause=True).data
        assert redact_event(ev, include_payloads=True).data["message"] == SECRET

    @pytest.mark.parametrize(
        "event_type",
        ["run.resumed", "approval.decided", "approval.requested", "run.error", "image.state"],
    )
    def test_non_dict_shapes_never_crash(self, event_type: str) -> None:
        ev = _event(event_type, {"decisions": "x", "described": [1, None], "params": 3})
        redact_event(ev)


# ---------------------------------------------------------------------------
# Through RunStore.follow and get_event_view (SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path: Any) -> Any:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'redact.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def state(engine: Any) -> SQLStateStore:
    return store_for_engine(engine)


async def _write_history(state: SQLStateStore, *, active: bool) -> None:
    """A run with a past approval pause; optionally a second, active one."""
    await state.create_run("r1", "a", tenant_id="T")
    approval_pause = {
        "status": "waiting_approval",
        "pending_tool_calls": [{"id": "c1", "name": "sql", "target": "server", "params": {"q": 1}}],
    }
    await state.save_run_event(
        "r1",
        event_type="approval.requested",
        data={
            "tool_name": "sql",
            "call_id": "c1",
            "reason": "requires_approval",
            "params": {"q": SECRET},
        },
    )
    await state.save_run_event("r1", event_type="run.paused", data=approval_pause)
    await state.save_run_event(
        "r1",
        event_type="approval.decided",
        data={
            "decision": "rejected",
            "decisions": [
                {"call_id": "c1", "tool_name": "sql", "decision": "denied", "reason": SECRET}
            ],
        },
    )
    await state.save_run_event(
        "r1",
        event_type="run.resumed",
        data={"resumed_from": "waiting_approval", "user_input": SECRET},
    )
    if active:
        await state.save_run_event(
            "r1",
            event_type="approval.requested",
            data={
                "tool_name": "sql",
                "call_id": "c2",
                "reason": "requires_approval",
                "params": {"q": "now"},
            },
        )
        await state.save_run_event(
            "r1",
            event_type="run.paused",
            data={
                "status": "waiting_approval",
                "pending_tool_calls": [
                    {"id": "c2", "name": "sql", "target": "server", "params": {"q": 2}}
                ],
            },
        )
        await state.pause_run("r1", status="waiting_approval", pause_data={})
    else:
        await state.finalize_run("r1", status="success")


async def _follow_all(store: RunStore, **kw: Any) -> list[StoredEvent]:
    feed = EventFeed(store, poll_interval_s=0.01, poll_max_s=0.02, status_every=1, heartbeat_s=0.2)
    out: list[StoredEvent] = []
    async with feed.follow("r1", tenant_id="T", until_terminal=True, max_duration_s=2.0, **kw) as f:
        async for item in f:
            if isinstance(item, FollowEnd):
                break
            if isinstance(item, FollowHeartbeat):
                break
            out.append(item)
    await feed.aclose()
    return out


async def test_follow_finished_run_leaks_no_free_text(state: Any, engine: Any) -> None:
    await _write_history(state, active=False)
    store = RunStore.from_engine(engine)
    events = await _follow_all(store)
    assert [e.event_type for e in events][:4] == [
        "approval.requested",
        "run.paused",
        "approval.decided",
        "run.resumed",
    ]
    assert SECRET not in json.dumps([e.data for e in events])
    trusted = await _follow_all(store, include_payloads=True)
    assert SECRET in json.dumps([e.data for e in trusted])


async def test_follow_active_approval_request_keeps_its_params(state: Any, engine: Any) -> None:
    await _write_history(state, active=True)
    store = RunStore.from_engine(engine)
    events = await _follow_all(store)
    past, active = [e.data for e in events if e.event_type == "approval.requested"]
    assert "params" not in past and "call_id" not in past  # a past call id is hidden too
    assert active["call_id"] == "c2" and active["params"] == {"q": "now"}
    assert SECRET not in json.dumps([e.data for e in events])


async def test_event_view_and_pause_page_hide_free_text(state: Any, engine: Any) -> None:
    await _write_history(state, active=True)
    store = RunStore.from_engine(engine)
    raw = await store.get_events("r1", tenant_id="T")
    views = [await store.get_event_view("r1", e, tenant_id="T") for e in raw]
    assert SECRET not in json.dumps([v.data for v in views])
    active_req = [v for v in views if v.event_type == "approval.requested"][-1]
    assert active_req.data["params"] == {"q": "now"}
    pairs, _ = await store.get_pause_page("r1", tenant_id="T")
    assert pairs[0].user_input is None
    trusted, _ = await store.get_pause_page("r1", tenant_id="T", include_payloads=True)
    assert trusted[0].user_input == SECRET


async def test_live_pause_change_refreshes_the_shared_view(state: Any, engine: Any) -> None:
    """A follower that cached the old pause still shows the new active request."""
    import asyncio

    await _write_history(state, active=True)
    store = RunStore.from_engine(engine)
    feed = EventFeed(store, poll_interval_s=0.01, poll_max_s=0.02, status_every=1, heartbeat_s=5)
    async with feed.follow("r1", tenant_id="T", max_duration_s=5.0) as f:
        backfill = [await asyncio.wait_for(anext(f), 5) for _ in range(6)]
        assert backfill[4].data["params"] == {"q": "now"}  # active request (c2)
        await state.save_run_event("r1", event_type="run.resumed", data={})
        await state.save_run_event(
            "r1",
            event_type="approval.requested",
            data={"tool_name": "sql", "call_id": "c3", "params": {"q": "live"}},
        )
        await state.save_run_event(
            "r1",
            event_type="run.paused",
            data={
                "status": "waiting_approval",
                "pending_tool_calls": [{"id": "c3", "name": "sql", "params": {"q": 3}}],
            },
        )
        live = [await asyncio.wait_for(anext(f), 5) for _ in range(3)]
    await feed.aclose()
    assert live[1].event_type == "approval.requested"
    # The request is committed before its pause: shown once the pause is active.
    assert live[1].data.get("params") in (None, {"q": "live"})
    assert live[2].data["pending_tool_calls"][0]["params"] == {"q": "live"}
