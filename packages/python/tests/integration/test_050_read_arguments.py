"""Hostile read arguments fail as typed validation errors (SQLite + Postgres).

A route that forwards a URL id, cursor or filter to ``RunStore`` must get a
``ValueError`` with a fixed message, never a driver error carrying SQL text,
and never a ``TypeError``. The message does not echo the value. Valid edge
values keep working.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.store import UNSCOPED, RunStore
from dendrux.tool import tool
from dendrux.types import LLMResponse
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio

NUL = "bad\x00id"
LONE = "bad\ud800id"
HUGE = 10**30


@tool(target="client")
async def read_range(sheet: str) -> str:
    return ""


@pytest.fixture
async def paused(store: Any) -> str:
    agent = Agent(
        provider=ScriptedLLM(
            [tool_call("read_range", {"sheet": "S"}, pid="c"), LLMResponse(text="d")]
        ),
        prompt="t",
        state_store=store,
        tools=[read_range],
        strict_tenancy=False,
    )
    return (await agent.run("go", tenant_id="A")).run_id


Call = Callable[[RunStore, str], Any]

HOSTILE: list[tuple[str, Call]] = [
    ("get_run nul id", lambda s, r: s.get_run(NUL)),
    ("get_run surrogate id", lambda s, r: s.get_run(LONE)),
    ("get_run int id", lambda s, r: s.get_run(5)),
    ("get_run none id", lambda s, r: s.get_run(None)),
    ("get_run huge id", lambda s, r: s.get_run("a" * 1_000_000)),
    ("get_run nul tenant", lambda s, r: s.get_run(r, tenant_id=NUL)),
    ("get_run huge tenant", lambda s, r: s.get_run(r, tenant_id="t" * 100_000)),
    ("get_pending int id", lambda s, r: s.get_pending(5, tenant_id="A")),
    ("get_pending nul id", lambda s, r: s.get_pending(NUL, tenant_id="A")),
    ("get_pending nul tenant", lambda s, r: s.get_pending(r, tenant_id=NUL)),
    ("get_events huge cursor", lambda s, r: s.get_events(r, after_sequence_index=HUGE)),
    ("get_events str cursor", lambda s, r: s.get_events(r, after_sequence_index="3")),
    ("get_events bool cursor", lambda s, r: s.get_events(r, after_sequence_index=True)),
    ("get_events huge limit", lambda s, r: s.get_events(r, limit=HUGE)),
    ("get_events str limit", lambda s, r: s.get_events(r, limit="5")),
    ("get_events 100k types", lambda s, r: s.get_events(r, event_types=["a"] * 100_000)),
    ("get_events nul type", lambda s, r: s.get_events(r, event_types=[NUL])),
    ("get_events type str", lambda s, r: s.get_events(r, event_types="run.paused")),
    ("get_events nul id", lambda s, r: s.get_events(NUL)),
    ("get_llm_calls huge offset", lambda s, r: s.get_llm_calls(r, offset=HUGE)),
    ("get_llm_calls huge iteration", lambda s, r: s.get_llm_calls(r, iteration=HUGE)),
    ("get_tool_invocations str limit", lambda s, r: s.get_tool_invocations(r, limit="5")),
    ("get_traces huge offset", lambda s, r: s.get_traces(r, offset=HUGE)),
    ("get_traces nul id", lambda s, r: s.get_traces(NUL)),
    ("get_notes nul id", lambda s, r: s.get_notes(NUL, tenant_id="A")),
    ("get_pii_mapping nul id", lambda s, r: s.get_pii_mapping(NUL)),
    ("get_pause_page nul id", lambda s, r: s.get_pause_page(NUL)),
    ("get_subtree nul id", lambda s, r: s.get_subtree(NUL)),
    ("list_runs str limit", lambda s, r: s.list_runs(limit="5")),
    ("list_runs huge limit", lambda s, r: s.list_runs(limit=HUGE)),
    ("list_runs str offset", lambda s, r: s.list_runs(offset="5")),
    ("list_runs nul status", lambda s, r: s.list_runs(status=NUL)),
    ("list_runs nul status list", lambda s, r: s.list_runs(status=["running", NUL])),
    ("list_runs 100k statuses", lambda s, r: s.list_runs(status=["x"] * 100_000)),
    ("list_runs nul agent", lambda s, r: s.list_runs(agent_name=NUL)),
    ("list_runs nul parent", lambda s, r: s.list_runs(parent_run_id=NUL)),
    ("list_runs nul tenant", lambda s, r: s.list_runs(tenant_id=NUL)),
    ("list_runs nul metadata value", lambda s, r: s.list_runs(metadata_filter={"k": NUL})),
    (
        "list_runs nested nul metadata",
        lambda s, r: s.list_runs(metadata_filter={"k": [{"x": NUL}]}),
    ),
    ("list_runs metadata not dict", lambda s, r: s.list_runs(metadata_filter=[("k", 1)])),
    ("list_runs metadata nan", lambda s, r: s.list_runs(metadata_filter={"k": float("nan")})),
    ("list_runs nested metadata value", lambda s, r: s.list_runs(metadata_filter={"k": {"n": 1}})),
    ("list_runs bad metadata key", lambda s, r: s.list_runs(metadata_filter={"a.b": 1})),
    ("list_runs huge metadata key", lambda s, r: s.list_runs(metadata_filter={"k" * 10_000: 1})),
    (
        "list_runs 1000 metadata keys",
        lambda s, r: s.list_runs(metadata_filter={f"k{i}": i for i in range(1000)}),
    ),
    ("count_runs nul metadata value", lambda s, r: s.count_runs(metadata_filter={"k": NUL})),
    ("count_runs nul agent", lambda s, r: s.count_runs(agent_name=NUL)),
    ("event_view nul id", lambda s, r: s.get_event_view(NUL, None, tenant_id="A")),
]


@pytest.mark.parametrize(("label", "call"), HOSTILE, ids=[h[0] for h in HOSTILE])
async def test_hostile_argument_is_a_validation_error(
    engine: Any, paused: str, label: str, call: Call, caplog: pytest.LogCaptureFixture
) -> None:
    store = RunStore.from_engine(engine)
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as info:
        await call(store, paused)
    message = str(info.value)
    assert type(info.value) is ValueError, label
    assert "SELECT" not in message.upper() and "\x00" not in message and len(message) < 300
    assert any(getattr(r, "code", None) == "read_argument_invalid" for r in caplog.records)


async def test_follow_rejects_hostile_cursor_and_id(engine: Any, paused: str) -> None:
    store = RunStore.from_engine(engine)
    with pytest.raises(ValueError):
        store.follow(paused, tenant_id="A", after_sequence_index=HUGE)
    with pytest.raises(ValueError):
        store.follow(NUL, tenant_id="A")
    with pytest.raises(ValueError):
        store.follow(paused, tenant_id="t" * 100_000)
    await store.close()


async def test_read_image_with_hostile_hash_reads_as_missing(engine: Any, paused: str) -> None:
    from dendrux.errors import BlobNotFoundError

    store = RunStore.from_engine(engine)
    for bad in (NUL, "z" * 64, 5):
        with pytest.raises(BlobNotFoundError):
            await store.read_image(paused, bad)  # type: ignore[arg-type]


async def test_valid_edge_values_still_work(engine: Any, paused: str) -> None:
    store = RunStore.from_engine(engine)
    assert await store.get_events(paused, after_sequence_index=2**31 - 1) == []
    assert len(await store.get_events(paused, after_sequence_index=-(2**31))) > 0
    assert await store.get_events(paused, event_types=[]) == []
    assert await store.get_run("a" * 255) is None
    assert await store.get_run(paused, tenant_id="t" * 255) is None
    assert len(await store.list_runs(limit=0)) == 1  # clamped, as before
    assert len(await store.list_runs(status=["waiting_client_tool"], tenant_id=UNSCOPED)) == 1
    assert await store.list_runs(metadata_filter={"k": "é", "n": 1, "b": True, "f": 1.5}) == []
    after = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
    assert len(await store.list_runs(started_after=after)) == 1
    assert await store.count_runs(agent_name="x" * 255) == 0
