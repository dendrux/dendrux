"""Redacted views keep only developer-declared identifiers and constant codes.

- Every public ``include_payloads`` / ``active_pause`` / other boolean flag of
  the store requires a real ``bool``: ``"false"`` from a query string must
  never switch redaction off.
- Hook-written reasons, unknown skill names, a pause's message, MCP-server
  tool names that were skipped and vendor header text are hidden.
- Malformed stored shapes (a dict reason, a dict status) never crash a reader.
- Every event type Dendrux writes is listed in the redaction review registry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import EventFeed, RunStore, StoredEvent, _follow, redact_event

SECRET = "probe-secret-5831"
NOT_BOOLS = ["false", "true", "", 0, 1, None, [], "False"]


def _event(event_type: str, data: dict[str, Any], seq: int = 0) -> StoredEvent:
    return StoredEvent(
        sequence_index=seq,
        iteration_index=0,
        event_type=event_type,
        correlation_id="c1",
        data=data,
        created_at=None,
    )


def _shown(event_type: str, data: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return redact_event(_event(event_type, data), **kw).data or {}


@pytest.mark.parametrize("bad", NOT_BOOLS)
def test_redact_event_flags_must_be_bool(bad: Any) -> None:
    ev = _event("run.resumed", {"user_input": SECRET})
    with pytest.raises(ValueError, match="must be a bool") as info:
        redact_event(ev, include_payloads=bad)
    assert SECRET not in str(info.value)
    with pytest.raises(ValueError, match="must be a bool"):
        redact_event(ev, active_pause=bad)


def test_hook_reasons_hidden_constants_kept() -> None:
    denied = _shown(
        "policy.denied", {"tool_name": "t", "call_id": "c", "reason": SECRET, "source": "hook"}
    )
    assert denied == {"tool_name": "<unknown_tool>", "source": "hook"}  # standalone: no evidence
    for constant in ("denied_by_policy", "policy hook error"):
        assert _shown("policy.denied", {"reason": constant})["reason"] == constant
    requested = _shown(
        "approval.requested", {"tool_name": "t", "call_id": "c", "reason": SECRET, "source": "hook"}
    )
    assert "reason" not in requested
    assert _shown("approval.requested", {"reason": "requires_approval"})["reason"] == (
        "requires_approval"
    )
    trusted = _shown("policy.denied", {"reason": SECRET}, include_payloads=True)
    assert trusted["reason"] == SECRET


def test_unknown_skill_name_is_a_placeholder() -> None:
    shown = _shown("skill.activation_failed", {"name": SECRET, "reason": "not_found"})
    assert shown == {"name": "<unknown_skill>", "reason": "not_found"}
    declared = _shown("skill.activation_failed", {"name": "billing", "reason": "denied"})
    assert declared["name"] == "billing"
    assert _shown("skill.activation_failed", {"name": SECRET}, include_payloads=True)["name"] == (
        SECRET
    )


def test_pause_message_hidden_even_for_the_active_pause() -> None:
    data = {"status": "paused", "reason": "requested", "message": SECRET, "pending_tool_calls": []}
    assert "message" not in _shown("run.paused", data, active_pause=True)
    assert _shown("run.paused", data, include_payloads=True)["message"] == SECRET


def test_mcp_and_vendor_text_hidden() -> None:
    skipped = _shown(
        "mcp.tools_skipped",
        {
            "skipped": [
                {"source_name": "s", "namespace": "n", "tool": SECRET, "reason": "name_unsafe"},
                {
                    "source_name": "s",
                    "namespace": "n",
                    "tool": "dev_name",
                    "reason": "unknown_tool",
                },
            ]
        },
    )
    assert SECRET not in json.dumps(skipped)
    assert skipped["skipped"][1]["tool"] == "dev_name"
    retry = _shown("provider.retry", {"status_code": 429, "retry_after": SECRET, "attempt": 1})
    assert "retry_after" not in retry
    assert _shown("provider.retry", {"retry_after": "12"})["retry_after"] == 12


@pytest.mark.parametrize(
    ("event_type", "data"),
    [
        ("approval.decided", {"decisions": [{"decision": "denied", "reason": {"note": "x"}}]}),
        ("approval.decided", {"decisions": [{"reason": ["x"]}, "x", None]}),
        ("run.paused", {"status": {"x": 1}, "pending_tool_calls": [{"id": "c"}]}),
        ("approval.requested", {"call_id": {"x": 1}, "params": {"q": 1}}),
        ("policy.denied", {"reason": {"x": 1}}),
        ("skill.activation_failed", {"name": {"x": 1}, "reason": ["not_found"]}),
        ("mcp.tools_skipped", {"skipped": [{"tool": "t", "reason": {"x": 1}}, 3]}),
        ("tool.completed", {"tool_name": {"x": 1}, "success": "yes", "code": []}),
    ],
)
def test_malformed_shapes_never_crash(event_type: str, data: dict[str, Any]) -> None:
    shown = _shown(event_type, data)
    assert "note" not in json.dumps(shown)
    redact_event(_event(event_type, data), active_pause=True)


def test_every_written_event_type_is_reviewed() -> None:
    src = Path(__file__).resolve().parents[2] / "src" / "dendrux"
    pattern = re.compile(
        r'"((?:run|tool|llm|approval|policy|guardrail|skill|mcp|budget|governance|provider|'
        r'inbox|image)\.[a-z_]+)"'
    )
    written = set()
    for path in src.rglob("*.py"):
        if "notifiers" in path.parts:
            continue  # tracing span names, never stored run events
        written |= set(pattern.findall(path.read_text()))
    assert written, "pattern found nothing"
    reviewed = set(getattr(_follow, "REVIEWED_EVENT_TYPES", {}))
    assert written <= reviewed, sorted(written - reviewed)


# ---------------------------------------------------------------------------
# Store boundaries reject non-bool flags with a fixed message
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Any) -> Any:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'r1.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    state = SQLAlchemyStateStore(eng)
    await state.create_run("r1", "a", tenant_id="T")
    await state.save_run_event("r1", event_type="run.resumed", data={"user_input": SECRET})
    await state.finalize_run("r1", status="success")
    yield RunStore.from_engine(eng)
    await eng.dispose()


@pytest.mark.parametrize("bad", NOT_BOOLS)
async def test_store_flags_must_be_bool(store: RunStore, bad: Any) -> None:
    ev = (await store.get_events("r1", tenant_id="T"))[0]
    calls = [
        lambda: store.follow("r1", tenant_id="T", include_payloads=bad),
        lambda: store.follow("r1", tenant_id="T", until_terminal=bad),
        lambda: EventFeed(store).follow("r1", tenant_id="T", include_payloads=bad),
        lambda: store.get_event_view("r1", ev, tenant_id="T", include_payloads=bad),
        lambda: store.get_pause_page("r1", tenant_id="T", include_payloads=bad),
        lambda: store.get_pauses("r1", tenant_id="T", include_payloads=bad),
        lambda: store.get_notes("r1", tenant_id="T", undelivered_only=bad),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="must be a bool") as info:
            result = call()
            if hasattr(result, "__await__"):
                await result
        assert SECRET not in str(info.value)


@pytest.mark.parametrize("bad", ["5", True, float("nan"), float("inf"), 0, -1])
async def test_follow_duration_is_validated(store: RunStore, bad: Any) -> None:
    with pytest.raises(ValueError):
        store.follow("r1", tenant_id="T", max_duration_s=bad)


async def test_trusted_get_pauses_keeps_the_answer(store: RunStore) -> None:
    state = store._state
    await state.create_run("r2", "a", tenant_id="T")
    await state.save_run_event("r2", event_type="run.paused", data={"status": "paused"})
    await state.save_run_event("r2", event_type="run.resumed", data={"user_input": SECRET})
    assert (await store.get_pauses("r2", tenant_id="T"))[0].user_input is None
    trusted = await store.get_pauses("r2", tenant_id="T", include_payloads=True)
    assert trusted[0].user_input == SECRET
