"""I-01 / I-02: image lifecycle through the public entry points on SQLite + Postgres.

The §6 canonical sample end to end (client-tool screenshot, pause, resume on a
fresh store + Agent), early typed errors, the external-``BlobStore`` second
process, and a v0.3.0 text pause resuming unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from dendrux.agent import Agent
from dendrux.errors import ImageLimitError, UnsupportedModalityError
from dendrux.llm.mock import MockLLM
from dendrux.media import Image, ImagePolicy, InMemoryBlobStore
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult

PNG = (Path(__file__).resolve().parents[1] / "fixtures" / "images" / "zebra42.png").read_bytes()
SHA = hashlib.sha256(PNG).hexdigest()
IMG = {
    "type": "image",
    "v": 1,
    "kind": "blob",
    "sha256": SHA,
    "media_type": "image/png",
    "bytes": len(PNG),
    "width": 400,
    "height": 200,
}


@tool(target="client")
async def take_screenshot() -> str:
    """Capture the screen on the client."""
    return "never runs on the server"


def _mock(responses: list[LLMResponse]) -> MockLLM:
    # C adds ``image_support``; MockLLM then reports image capability "yes".
    return MockLLM(responses, image_support="yes")  # type: ignore[call-arg]


def _agent(store: Any, responses: list[LLMResponse], **kw: Any) -> Agent:
    kw.setdefault("strict_tenancy", False)
    return Agent(
        name="screen_agent",
        prompt="You read screenshots.",
        provider=_mock(responses),
        tools=[take_screenshot],
        state_store=store,
        **kw,
    )


async def _rows(engine: Any, sql: str, **params: Any) -> list[Any]:
    async with engine.connect() as conn:
        return list((await conn.execute(text(sql), params)).mappings())


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


async def test_canonical_sample_end_to_end(engine: Any) -> None:
    store = SQLAlchemyStateStore(engine)
    agent = _agent(
        store,
        [LLMResponse(text=None, tool_calls=[ToolCall(name="take_screenshot", params={})])],
    )
    paused = await agent.run(
        "Take a screenshot and tell me the code word shown in it.", tenant_id="acme"
    )
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    run_id = paused.run_id
    pause = await store.get_pause_state(run_id)
    assert "image_view" not in pause
    call_id = pause["pending_tool_calls"][0]["id"]
    iteration = pause["iteration"]

    # "Process exit": a fresh store and Agent; on Postgres also a new engine
    # (an in-memory SQLite database does not survive its engine).
    if engine.dialect.name == "postgresql":
        from sqlalchemy.ext.asyncio import create_async_engine

        url = engine.url.render_as_string(hide_password=False)
        await engine.dispose()
        engine = create_async_engine(url)
    store2 = SQLAlchemyStateStore(engine)
    agent2 = _agent(store2, [LLMResponse(text="The code word is ZEBRA42.")])
    result = await agent2.submit_tool_results(
        run_id,
        [
            ToolResult(
                name="take_screenshot",
                call_id=call_id,
                payload=json.dumps("Captured."),
                images=(Image.from_bytes(PNG),),
            )
        ],
    )
    assert result.status == RunStatus.SUCCESS
    assert result.answer == "The code word is ZEBRA42."

    blobs = await _rows(engine, "SELECT * FROM dendrux_blobs")
    assert len(blobs) == 1
    blob = blobs[0]
    assert (blob["tenant_scope"], blob["sha256"], blob["state"], blob["storage"]) == (
        "t:acme",
        SHA,
        "live",
        "db",
    )
    assert (blob["width"], blob["height"], blob["frame_count"]) == (400, 200, 1)
    assert bytes(blob["data"]) == PNG and blob["ingest_transform"] is None

    refs = await _rows(
        engine,
        "SELECT owner_kind, owner_id, ordinal, tenant_scope FROM dendrux_blob_refs "
        "WHERE run_id = :r ORDER BY owner_kind",
        r=run_id,
    )
    trace = next(t for t in await store2.get_traces(run_id) if t.role == "tool")
    assert sorted((r["owner_kind"], r["owner_id"], r["ordinal"]) for r in refs) == sorted(
        [
            ("submission", f"{iteration}:{call_id}", 0),
            ("tool_call", call_id, 0),
            ("trace", str(trace.order_index), 1),
        ]
    )
    assert {r["tenant_scope"] for r in refs} == {"t:acme"}

    run = (
        await _rows(engine, "SELECT input_data, pause_data FROM agent_runs WHERE id=:r", r=run_id)
    )[0]
    assert _json(run["input_data"]) == {
        "input": "Take a screenshot and tell me the code word shown in it."
    }
    assert _json(run["pause_data"]) is None

    traces = await store2.get_traces(run_id)
    tool_trace = next(t for t in traces if t.role == "tool")
    assert tool_trace.content == '"Captured."'
    assert tool_trace.meta["dendrux.parts"] == [{"type": "text", "text": '"Captured."'}, IMG]

    calls = await _rows(
        engine, "SELECT meta, target FROM tool_calls WHERE agent_run_id=:r", r=run_id
    )
    assert _json(calls[0]["meta"])["dendrux.images"] == [IMG]

    events = await store2.get_run_events(run_id)
    resumed = next(e for e in events if e.event_type == "run.resumed")
    assert resumed.data["submitted_results"][0]["image_count"] == 1
    completed = next(e for e in events if e.event_type == "tool.completed")
    assert completed.data["image_count"] == 1
    assert not [e for e in events if e.event_type.startswith("image.")]

    llm = await store2.get_llm_interactions(run_id)  # type: ignore[attr-defined]
    second = llm[-1]
    tool_msg = next(m for m in second.semantic_request["messages"] if m["role"] == "tool")
    assert tool_msg["parts"][1] == IMG
    assert second.semantic_request["image_stats"]["count"] == 1

    dumped = json.dumps([str(dict(r)) for r in refs + blobs[:0]]) + json.dumps(
        [t.meta for t in traces], default=str
    )
    assert "iVBORw0K" not in dumped and "data:image" not in dumped
    if engine.dialect.name == "postgresql":
        await engine.dispose()


async def test_stream_errors_create_no_run_and_no_blob(engine: Any) -> None:
    store = SQLAlchemyStateStore(engine)
    agent = _agent(
        store, [LLMResponse(text="x")], images=ImagePolicy(capability_overrides={"mock": "no"})
    )
    with pytest.raises(UnsupportedModalityError):
        async for _ in agent.stream(["look", Image.from_bytes(PNG)]):
            pass
    agent2 = _agent(store, [LLMResponse(text="x")], images=ImagePolicy(max_image_bytes=100))
    with pytest.raises(ImageLimitError):
        async for _ in agent2.stream(["look", Image.from_bytes(PNG)]):
            pass
    assert await store.count_runs() == 0
    assert await _rows(engine, "SELECT sha256 FROM dendrux_blobs") == []


async def test_external_blob_store_second_process(engine: Any) -> None:
    from dendrux.store import RunStore

    blob_store = InMemoryBlobStore()
    store = SQLAlchemyStateStore(engine)
    agent = _agent(
        store,
        [LLMResponse(text=None, tool_calls=[ToolCall(name="take_screenshot", params={})])],
        blob_store=blob_store,
    )
    paused = await agent.run(["look at this", Image.from_bytes(PNG)], tenant_id="acme")
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    blob = (await _rows(engine, "SELECT storage, data FROM dendrux_blobs"))[0]
    assert blob["storage"] == "external" and blob["data"] is None

    reader = RunStore(SQLAlchemyStateStore(engine), blob_store=blob_store)  # type: ignore[call-arg]
    assert await reader.read_image(paused.run_id, SHA, tenant_id="acme") == PNG

    call_id = (await store.get_pause_state(paused.run_id))["pending_tool_calls"][0]["id"]
    agent2 = _agent(SQLAlchemyStateStore(engine), [LLMResponse(text="done")], blob_store=blob_store)
    result = await agent2.submit_tool_results(
        paused.run_id, [ToolResult(name="take_screenshot", call_id=call_id, payload='"ok"')]
    )
    assert result.status == RunStatus.SUCCESS


async def test_v030_text_pause_resumes_unchanged(engine: Any) -> None:
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, 1)
    """A text-only pause has no image keys and resumes exactly as in v0.3.0."""
    store = SQLAlchemyStateStore(engine)
    agent = Agent(
        strict_tenancy=False,
        name="screen_agent",
        prompt="p",
        provider=MockLLM(
            [LLMResponse(text=None, tool_calls=[ToolCall(name="take_screenshot", params={})])]
        ),
        tools=[take_screenshot],
        state_store=store,
    )
    paused = await agent.run("shot")
    pause = await store.get_pause_state(paused.run_id)
    assert "image_view" not in pause
    assert all("parts" not in m for m in pause["history"])
    call_id = pause["pending_tool_calls"][0]["id"]
    agent2 = Agent(
        strict_tenancy=False,
        name="screen_agent",
        prompt="p",
        provider=MockLLM([LLMResponse(text="done")]),
        tools=[take_screenshot],
        state_store=SQLAlchemyStateStore(engine),
    )
    result = await agent2.submit_tool_results(
        paused.run_id, [ToolResult(name="take_screenshot", call_id=call_id, payload='"ok"')]
    )
    assert result.status == RunStatus.SUCCESS
    events = await store.get_run_events(paused.run_id)
    resumed = next(e for e in events if e.event_type == "run.resumed")
    assert resumed.data["submitted_results"] == [
        {"call_id": call_id, "name": "take_screenshot", "success": True}
    ]
    record = await store.get_run(paused.run_id)
    assert record.input_data == {"input": "shot"}


async def test_recorder_gets_provider_for_token_estimates(engine: Any) -> None:
    """``PersistenceRecorder(provider=)``: image_stats.estimated_tokens is filled."""

    class ClaudeLikeMock(MockLLM):
        def image_capability_key(self, model: str) -> tuple[str, str, str]:
            return ("anthropic", "", model)

    store = SQLAlchemyStateStore(engine)
    provider = ClaudeLikeMock(
        [LLMResponse(text="ZEBRA42")],
        model="claude-sonnet-4-6",
        image_support="yes",  # type: ignore[call-arg]
    )
    agent = Agent(strict_tenancy=False, name="a", prompt="p", provider=provider, state_store=store)
    result = await agent.run(["What does this say?", Image.from_bytes(PNG)])
    llm = await store.get_llm_interactions(result.run_id)  # type: ignore[attr-defined]
    assert llm[0].semantic_request["image_stats"] == {
        "count": 1,
        "bytes": len(PNG),
        "estimated_tokens": 120,
    }
