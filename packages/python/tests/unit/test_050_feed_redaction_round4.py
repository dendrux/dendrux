"""Call ids and image tool names keep their provenance (round 4).

Threat model: custom providers, strategies, tools, hooks and MCP servers are
untrusted text sources for redacted readers.

- ``ToolCall.id`` can be chosen by a custom provider, so redacted views never
  show a tool call id (``call_id``, pending ``id``, ``submitted_results`` and
  ``decisions`` ids, and the event ``correlation_id``), except for the active
  pause, whose ids a client needs to act on (the same exception as its
  arguments). ``llm.completed`` and ``inbox.delivered`` keep their
  correlation ids (Dendrux-generated interaction and note ids).
- ``image.state.guardrail_verdicts[].tool_name`` comes from a strategy-built
  message; it is shown only when it is a tool name the run recorded as a
  declared call, else ``"<unknown_tool>"``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import EventFeed, FollowEnd, FollowHeartbeat, RunStore, StoredEvent, redact_event

SECRET = "probe-secret-5831"


def _event(event_type: str, data: Any, correlation_id: str | None = SECRET) -> StoredEvent:
    return StoredEvent(
        sequence_index=0,
        iteration_index=0,
        event_type=event_type,
        correlation_id=correlation_id,
        data=data,
        created_at=None,
    )


CALL_ID_EVENTS = [
    (
        "run.paused",
        {"status": "waiting_client_tool", "pending_tool_calls": [{"id": SECRET, "name": "t"}]},
    ),
    ("approval.requested", {"tool_name": "t", "call_id": SECRET, "reason": "requires_approval"}),
    ("policy.substituted", {"tool_name": "t", "call_id": SECRET}),
    ("policy.denied", {"tool_name": "t", "call_id": SECRET, "reason": "denied_by_policy"}),
    (
        "run.resumed",
        {
            "resumed_from": "waiting_client_tool",
            "submitted_results": [{"call_id": SECRET, "name": "t", "success": True}],
        },
    ),
    (
        "approval.decided",
        {
            "decision": "approved",
            "decisions": [{"call_id": SECRET, "tool_name": "t", "decision": "approved"}],
        },
    ),
    ("guardrail.restore_skipped", {"tool_name": "t", "call_id": SECRET, "count": 1}),
    ("tool.completed", {"tool_name": "t", "success": True}),
    ("policy.hook_error", {"tool_name": "t", "call_id": SECRET}),
    ("policy.hook_warning", {"tool_name": "t", "call_id": SECRET}),
]


@pytest.mark.parametrize(("event_type", "data"), CALL_ID_EVENTS, ids=[e[0] for e in CALL_ID_EVENTS])
def test_call_ids_hidden_outside_the_active_pause(event_type: str, data: dict[str, Any]) -> None:
    shown = redact_event(_event(event_type, data))
    assert SECRET not in json.dumps(shown.data) and shown.correlation_id is None
    trusted = redact_event(_event(event_type, data), include_payloads=True)
    assert SECRET in json.dumps(trusted.data) + str(trusted.correlation_id)


def test_active_pause_keeps_the_ids_a_client_needs() -> None:
    paused = redact_event(_event(*CALL_ID_EVENTS[0]), active_pause=True)
    assert paused.data["pending_tool_calls"][0]["id"] == SECRET
    requested = redact_event(_event(*CALL_ID_EVENTS[1]), active_pause=True)
    assert requested.data["call_id"] == SECRET and requested.correlation_id == SECRET


def test_dendrux_correlation_ids_kept() -> None:
    llm = redact_event(_event("llm.completed", {"input_tokens": 1}, correlation_id="INTERACTION"))
    assert llm.correlation_id == "INTERACTION"
    note = redact_event(_event("inbox.delivered", {"note_id": "N1"}, correlation_id="N1"))
    assert note.correlation_id == "N1" and note.data == {"note_id": "N1"}


def test_standalone_image_verdict_tool_name_is_a_placeholder() -> None:
    state = {"guardrail_verdicts": [{"sha256": "ab" * 32, "action": "block", "tool_name": SECRET}]}
    shown = redact_event(_event("image.state", state, correlation_id=None)).data
    assert shown["guardrail_verdicts"][0]["tool_name"] == "<unknown_tool>"


@pytest.fixture
async def stores(tmp_path: Any) -> Any:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'r4.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    state = SQLAlchemyStateStore(eng)
    yield state, RunStore.from_engine(eng)
    await eng.dispose()


async def test_image_verdict_tool_name_checked_against_recorded_calls(stores: Any) -> None:
    state, store = stores
    await state.create_run("r1", "a", tenant_id="T")
    await state.save_tool_call(
        "r1",
        tool_call_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        provider_tool_call_id=None,
        tool_name="screenshot",
        target="server",
        params={},
        result_payload='"ok"',
        success=True,
        duration_ms=1,
        iteration_index=0,
    )
    await state.save_tool_call(
        "r1",
        tool_call_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
        provider_tool_call_id=None,
        tool_name=SECRET,
        target="server",
        params={},
        result_payload="{}",
        success=False,
        duration_ms=1,
        iteration_index=0,
        error_message=f"Unknown tool: {SECRET}. No tools are available.",
    )
    verdicts = [
        {"sha256": "ab" * 32, "action": "block", "tool_name": "screenshot"},
        {"sha256": "cd" * 32, "action": "block", "tool_name": SECRET},
    ]
    # Positive evidence that "screenshot" is a declared, non-MCP tool (round 6).
    await state.save_run_event(
        "r1", event_type="tools.declared", data={"local_tool_names": ["screenshot", SECRET]}
    )
    await state.save_run_event(
        "r1", event_type="image.state", data={"guardrail_verdicts": verdicts}
    )
    await state.finalize_run("r1", status="success")
    _, raw = await store.get_events("r1", tenant_id="T")
    view = await store.get_event_view("r1", raw, tenant_id="T")
    names = [v["tool_name"] for v in view.data["guardrail_verdicts"]]
    assert names == ["screenshot", "<unknown_tool>"]
    feed = EventFeed(store, poll_interval_s=0.01, poll_max_s=0.02, heartbeat_s=1.0)
    async with feed.follow("r1", tenant_id="T", max_duration_s=5.0) as f:
        items = [i async for i in f if not isinstance(i, (FollowEnd, FollowHeartbeat))]
    await feed.aclose()
    assert [v["tool_name"] for v in items[1].data["guardrail_verdicts"]] == names
