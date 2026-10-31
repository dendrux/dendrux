"""Redacted views are a field-level allowlist.

- Provider-reported text (``llm.completed.model``, ``error_info.model`` /
  ``provider_code`` / ``safe_message``, ``image.rejected.model``) and text
  built from model call ids (``guardrail.blocked.location`` / ``error``) are
  hidden; Dendrux's enumerated fields stay.
- ``tool.completed.code`` keeps only codes Dendrux writes; any other code is
  ``"<unknown_code>"``.
- Keys a reviewed event type does not list are hidden; unreviewed event types
  show no data.
- A non-object ``event.data`` reads as empty data on every reader path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import EventFeed, FollowEnd, FollowHeartbeat, RunStore, StoredEvent, redact_event

SECRET = "probe-secret-5831"


def _shown(event_type: str, data: Any, **kw: Any) -> Any:
    ev = StoredEvent(
        sequence_index=0,
        iteration_index=0,
        event_type=event_type,
        correlation_id="c1",
        data=data,
        created_at=None,
    )
    return redact_event(ev, **kw).data


ERROR_INFO = {
    "category": "rate_limit",
    "retryable": True,
    "safe_message": f"anthropic rate_limit (HTTP 429, code {SECRET})",
    "error_type": "ProviderRateLimitError",
    "provider": "anthropic",
    "model": SECRET,
    "status_code": 429,
    "provider_code": SECRET,
    "timeout_reason": None,
}
KEPT_INFO = {
    "category": "rate_limit",
    "retryable": True,
    "error_type": "ProviderRateLimitError",
    "provider": "anthropic",
    "status_code": 429,
    "timeout_reason": None,
}


@pytest.mark.parametrize(
    ("event_type", "data"),
    [
        ("llm.completed", {"input_tokens": 3, "model": SECRET, "phase": "commentary"}),
        ("llm.failed", {"error_info": ERROR_INFO, "timeout_reason": None, "duration_ms": 5}),
        ("run.error", {"error": SECRET, "status": "error", "error_info": ERROR_INFO}),
        (
            "image.rejected",
            {"image_id": "img_abc", "sha256": "ab" * 32, "model": SECRET, "status_code": 400},
        ),
        (
            "guardrail.blocked",
            {
                "direction": "incoming",
                "error": f"Guardrail 'pii' blocked: 1 EMAIL at message 2 (tool_call_id={SECRET})",
                "code": "guardrail_blocked",
                "kinds": ["EMAIL_ADDRESS"],
                "counts": {"EMAIL_ADDRESS": 1},
                "location": f"message 2 (role=tool, tool_call_id={SECRET})",
            },
        ),
        ("run.completed", {"status": "success", "extra": SECRET}),
        ("tool.completed", {"tool_name": "t", "success": True, "parse_error": SECRET}),
        ("custom.unreviewed", {"anything": SECRET}),
    ],
)
def test_provider_and_model_text_hidden(event_type: str, data: dict[str, Any]) -> None:
    shown = _shown(event_type, data)
    assert SECRET not in json.dumps(shown)
    assert SECRET in json.dumps(_shown(event_type, data, include_payloads=True))


def test_dendrux_fields_survive() -> None:
    assert _shown("llm.failed", {"error_info": ERROR_INFO})["error_info"] == KEPT_INFO
    completed = _shown("llm.completed", {"input_tokens": 3, "model": "m", "phase": "commentary"})
    assert completed == {"input_tokens": 3, "phase": "commentary"}
    blocked = _shown(
        "guardrail.blocked",
        {"direction": "incoming", "code": "guardrail_blocked", "kinds": ["EMAIL_ADDRESS"]},
    )
    assert blocked == {
        "direction": "incoming",
        "code": "guardrail_blocked",
        "kinds": ["EMAIL_ADDRESS"],
    }


@pytest.mark.parametrize(
    ("code", "shown"),
    [
        ("invalid_params", "invalid_params"),
        ("timeout_outcome_unknown", "timeout_outcome_unknown"),
        ("tool_pool_saturated", "tool_pool_saturated"),
        (SECRET, "<unknown_code>"),
        ("x", "<unknown_code>"),
        ({"x": 1}, "<unknown_code>"),
    ],
)
def test_tool_codes_are_enumerated(code: Any, shown: str) -> None:
    data = _shown("tool.completed", {"tool_name": "t", "success": True, "code": code})
    assert data["code"] == shown
    assert data["tool_name"] == "<unknown_tool>"  # standalone: no MCP evidence


@pytest.mark.parametrize("bad", [["bad"], "text", 5, True])
def test_non_object_data_reads_as_empty(bad: Any) -> None:
    for event_type in ("run.resumed", "run.paused", "tool.completed", "custom.x"):
        assert _shown(event_type, bad) == {}
        assert _shown(event_type, bad, active_pause=True) == {}
        assert _shown(event_type, bad, include_payloads=True) == bad


@pytest.fixture
async def store(tmp_path: Any) -> Any:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'r2.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    state = store_for_engine(eng)
    await state.create_run("r1", "a", tenant_id="T")
    for event_type in ("run.paused", "approval.requested", "run.resumed", "tool.completed"):
        await state.save_run_event("r1", event_type=event_type, data=["bad"])  # type: ignore[arg-type]
    await state.finalize_run("r1", status="success")
    yield RunStore.from_engine(eng)
    await eng.dispose()


async def test_readers_survive_non_object_rows(store: RunStore) -> None:
    raw = await store.get_events("r1", tenant_id="T")
    assert len(raw) == 4
    for include_payloads in (False, True):
        views = [
            await store.get_event_view("r1", e, tenant_id="T", include_payloads=include_payloads)
            for e in raw
        ]
        assert [v.data for v in views] == ([{}] * 4 if not include_payloads else [["bad"]] * 4)
        feed = EventFeed(store, poll_interval_s=0.01, poll_max_s=0.02, heartbeat_s=1.0)
        seen = []
        async with feed.follow(
            "r1", tenant_id="T", include_payloads=include_payloads, max_duration_s=5.0
        ) as f:
            async for item in f:
                if isinstance(item, (FollowEnd, FollowHeartbeat)):
                    break
                seen.append(item)
        await feed.aclose()
        assert len(seen) == 4
        pairs = await store.get_pauses("r1", tenant_id="T", include_payloads=include_payloads)
        assert len(pairs) == 1 and pairs[0].pending_tool_calls == []
