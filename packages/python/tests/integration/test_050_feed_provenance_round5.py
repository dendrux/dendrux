"""Redacted views under the untrusted-source threat model (round 5, SQLite + Postgres).

Custom providers, strategies, tools, hooks and MCP servers are untrusted text
sources for redacted readers.

- An approval request counts as the active pause's only inside that pause's
  own event window; a custom provider that reuses a call id across pauses
  never makes an older request's arguments visible.
- MCP-server tool names are never shown: ``mcp.connected`` gives a count, and
  a tool event naming an MCP tool shows ``"<mcp_tool>"``.
- Provider labels in ``error_info`` and ``image.rejected`` are Dendrux's
  built-in adapter names or ``"<custom_provider>"``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.store import FollowEnd, FollowHeartbeat, RunStore, StoredEvent, redact_event
from dendrux.tool import tool
from dendrux.types import LLMResponse, ToolCall
from tests._helpers.scripted_provider import ScriptedLLM

pytestmark = pytest.mark.asyncio

PAST = "PAST-SECRET-5831"
NOW = "now-visible"
SECRET = "probe-secret-5831"


@tool()
async def run_sql(q: str) -> str:
    return "rows"


def _reused(q: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(name="run_sql", params={"q": q}, id="repeat-id", provider_tool_call_id="p")
        ]
    )


async def _follow(run_store: RunStore, run_id: str) -> list[StoredEvent]:
    out: list[StoredEvent] = []
    async with run_store.follow(
        run_id, tenant_id="A", until_terminal=False, max_duration_s=1.0
    ) as f:
        async for item in f:
            if isinstance(item, (FollowEnd, FollowHeartbeat)):
                break
            out.append(item)
    return out


async def test_reused_call_id_never_shows_an_older_pause(store: Any, engine: Any) -> None:
    llm = ScriptedLLM([_reused(PAST), _reused(NOW), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="t",
        state_store=store,
        tools=[run_sql],
        require_approval=["run_sql"],
        strict_tenancy=False,
    )
    first = await agent.run("go", tenant_id="A")
    second = await agent.submit_approval(
        first.run_id, approved=True, tenant_id="A", expected_pause_id=first.pause_id
    )
    assert second.status.value == "waiting_approval"
    run_store = RunStore.from_engine(engine)
    for events in (
        await _follow(run_store, first.run_id),
        [
            await run_store.get_event_view(first.run_id, e, tenant_id="A")
            for e in await run_store.get_events(first.run_id, tenant_id="A")
        ],
    ):
        requests = [e for e in events if e.event_type == "approval.requested"]
        assert len(requests) == 2
        assert PAST not in json.dumps([e.data for e in events])
        assert "params" not in requests[0].data and "call_id" not in requests[0].data
        assert requests[1].data["params"] == {"q": NOW}
    pending = await run_store.get_pending(first.run_id, tenant_id="A")
    assert pending is not None and pending.tool_calls[0].params == {"q": NOW}


async def test_mcp_tool_names_are_never_shown(store: Any, engine: Any) -> None:
    await store.create_run("rm".ljust(26, "0"), "a", tenant_id="A")
    run_id = "rm".ljust(26, "0")
    await store.save_run_event(
        run_id,
        event_type="mcp.connected",
        data={"source_name": "srv", "namespace": "srv", "tool_count": 1, "tool_names": [SECRET]},
    )
    await store.save_run_event(
        run_id,
        event_type="tool.completed",
        data={"tool_name": SECRET, "target": "server", "success": True},
    )
    await store.save_run_event(
        run_id,
        event_type="approval.decided",
        data={"decision": "approved", "decisions": [{"tool_name": SECRET, "decision": "approved"}]},
    )
    await store.finalize_run(run_id, status="success")
    run_store = RunStore.from_engine(engine)
    events = await _follow(run_store, run_id)
    assert SECRET not in json.dumps([e.data for e in events])
    connected, completed, decided = events
    assert connected.data == {"source_name": "srv", "namespace": "srv", "tool_count": 1}
    assert completed.data["tool_name"] == "<mcp_tool>"
    assert decided.data["decisions"][0]["tool_name"] == "<mcp_tool>"
    views = [
        await run_store.get_event_view(run_id, e, tenant_id="A")
        for e in await run_store.get_events(run_id, tenant_id="A")
    ]
    assert SECRET not in json.dumps([v.data for v in views])


def _event(event_type: str, data: dict[str, Any]) -> StoredEvent:
    return StoredEvent(
        sequence_index=0,
        iteration_index=0,
        event_type=event_type,
        correlation_id=None,
        data=data,
        created_at=None,
    )


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai_responses", "openrouter"])
async def test_builtin_provider_names_kept(provider: str) -> None:
    info = {"category": "rate_limit", "provider": provider}
    assert (
        redact_event(_event("llm.failed", {"error_info": info})).data["error_info"]["provider"]
        == provider
    )


async def test_custom_provider_label_is_a_placeholder() -> None:
    for event_type, data in (
        ("llm.failed", {"error_info": {"category": "rate_limit", "provider": SECRET}}),
        ("run.error", {"error_info": {"category": "rate_limit", "provider": SECRET}}),
        ("image.rejected", {"image_id": "img_abc", "provider": SECRET}),
    ):
        shown = redact_event(_event(event_type, data)).data
        assert SECRET not in json.dumps(shown)
        assert "<custom_provider>" in json.dumps(shown)


async def test_standalone_tool_names_need_evidence() -> None:
    shown = redact_event(_event("tool.completed", {"tool_name": "local", "success": True})).data
    assert shown["tool_name"] == "<unknown_tool>"


async def test_builtin_provider_set_matches_the_adapters() -> None:
    import re
    from pathlib import Path

    from dendrux.store._redact import BUILTIN_PROVIDERS

    llm = Path(__file__).resolve().parents[2] / "src" / "dendrux" / "llm"
    pattern = re.compile(r'(?:provider|provider_name|_image_provider_name)\s*=\s*"([a-z_]+)"')
    labels = set()
    for name in ("anthropic.py", "openai.py", "openai_responses.py", "openrouter.py"):
        labels |= set(pattern.findall((llm / name).read_text()))
    assert labels and labels <= BUILTIN_PROVIDERS, labels - BUILTIN_PROVIDERS


# ---------------------------------------------------------------------------
# Round 6: positive evidence only. A missing or partial MCP record never makes
# a tool name safe to show.
# ---------------------------------------------------------------------------


async def _tool_run(store: Any, run_id: str, name: str, *, connected: bool) -> None:
    await store.create_run(run_id, "a", tenant_id="A")
    if connected:
        await store.save_run_event(
            run_id,
            event_type="mcp.connected",
            data={"source_name": "srv", "namespace": "srv", "tool_count": 1, "tool_names": [name]},
        )
    await store.save_run_event(
        run_id, event_type="tool.completed", data={"tool_name": name, "success": True}
    )
    await store.finalize_run(run_id, status="success")


async def test_custom_store_that_omits_mcp_records_shows_no_name(engine: Any) -> None:
    from dendrux.runtime.state import SQLAlchemyStateStore

    class OmittingStore(SQLAlchemyStateStore):
        async def get_run_events(self, run_id: str, **kw: Any) -> Any:
            rows = await super().get_run_events(run_id, **kw)
            return [r for r in rows if r.event_type != "mcp.connected"]

    state = OmittingStore(engine)
    name = f"mcp__{SECRET}"
    await _tool_run(state, "rc".ljust(26, "0"), name, connected=True)
    run_store = RunStore(state)
    (completed,) = await run_store.get_events("rc".ljust(26, "0"), tenant_id="A")
    view = await run_store.get_event_view("rc".ljust(26, "0"), completed, tenant_id="A")
    assert view.data["tool_name"] == "<unknown_tool>"
    events = await _follow(run_store, "rc".ljust(26, "0"))
    assert SECRET not in json.dumps([e.data for e in events])


async def test_older_run_without_records_shows_no_name(store: Any, engine: Any) -> None:
    await _tool_run(store, "ro".ljust(26, "0"), SECRET, connected=False)
    run_store = RunStore.from_engine(engine)
    (completed,) = await run_store.get_events("ro".ljust(26, "0"), tenant_id="A")
    view = await run_store.get_event_view("ro".ljust(26, "0"), completed, tenant_id="A")
    assert view.data["tool_name"] == "<unknown_tool>"


async def test_declared_local_tools_are_positive_evidence(store: Any, engine: Any) -> None:
    run_id = "rd".ljust(26, "0")
    await store.create_run(run_id, "a", tenant_id="A")
    await store.save_run_event(
        run_id, event_type="tools.declared", data={"local_tool_names": ["lookup"]}
    )
    for name in ("lookup", "use_skill", "other"):
        await store.save_run_event(
            run_id, event_type="tool.completed", data={"tool_name": name, "success": True}
        )
    await store.finalize_run(run_id, status="success")
    run_store = RunStore.from_engine(engine)
    events = await _follow(run_store, run_id)
    declared, *completed = events
    assert declared.data == {"count": 1}
    assert [e.data["tool_name"] for e in completed] == ["lookup", "use_skill", "<unknown_tool>"]


async def test_mcp_tool_with_a_reserved_name_is_still_an_mcp_tool(store: Any, engine: Any) -> None:
    """MCP evidence is checked before Dendrux's reserved skill-tool names."""
    run_id = "rs".ljust(26, "0")
    await _tool_run(store, run_id, "use_skill", connected=True)
    run_store = RunStore.from_engine(engine)
    events = await _follow(run_store, run_id)
    assert events[-1].data["tool_name"] == "<mcp_tool>"
