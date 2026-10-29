"""Image persistence on the SQL catalog (SQLite + Postgres matrix).

Covers §3 (schema, edges in the record's transaction, tenant assertion),
§3.4 row shapes, §6 sample rows, tenant isolation and uniform not-found,
``RunStore(blob_store=)`` second-process reads, orphan-edge GC on SQLite
without foreign keys, the ``last_used_at`` grace, the Postgres DB clock
(N-14) and the P-04 pause-size bound.
"""

from __future__ import annotations

import base64
import json
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db.models import Base
from dendrux.errors import BlobNotFoundError, InvalidImageError
from dendrux.media import BlobEdge, Image, InMemoryBlobStore, ingest
from dendrux.runtime._blob_catalog import BlobServices
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import RunStore
from dendrux.types import (
    ImagePart,
    LLMResponse,
    Message,
    Role,
    TextPart,
    ToolCall,
    ToolResult,
    UsageStats,
)
from tests.unit.test_blob_catalog import _SurfaceProvider, make_png, runtime_for, store_png

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def age_blob(engine: Any, scope: str, sha256: str, hours: float) -> None:
    """Move a blob's last_used_at / state_changed_at ``hours`` into the past."""
    async with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            await conn.execute(
                text(
                    "UPDATE dendrux_blobs SET last_used_at = now() - make_interval(hours => :h),"
                    " state_changed_at = now() - make_interval(hours => :h)"
                    " WHERE tenant_scope = :s AND sha256 = :sha"
                ),
                {"h": int(hours), "s": scope, "sha": sha256},
            )
        else:
            from datetime import UTC, datetime

            old = (datetime.now(UTC) - timedelta(hours=hours)).replace(tzinfo=None)
            await conn.execute(
                text(
                    "UPDATE dendrux_blobs SET last_used_at = :t, state_changed_at = :t"
                    " WHERE tenant_scope = :s AND sha256 = :sha"
                ),
                {"t": old.isoformat(sep=" "), "s": scope, "sha": sha256},
            )


async def rows(engine: Any, sql: str, **params: Any) -> list[Any]:
    async with engine.connect() as conn:
        return list((await conn.execute(text(sql), params)).all())


async def typed(engine: Any, *cols: Any) -> list[Any]:
    """Rows through the models' column types (JSON decoded on both dialects)."""
    from sqlalchemy import select

    async with engine.connect() as conn:
        return list((await conn.execute(select(*cols))).all())


async def edge_rows(engine: Any) -> list[tuple[Any, ...]]:
    return [
        tuple(r)
        for r in await rows(
            engine,
            "SELECT tenant_scope, sha256, run_id, owner_kind, owner_id, ordinal"
            " FROM dendrux_blob_refs ORDER BY owner_kind, owner_id, ordinal",
        )
    ]


def assert_clean(exc: BaseException) -> None:
    assert exc.__cause__ is None and exc.__context__ is None


# ---------------------------------------------------------------------------
# Rows and storage
# ---------------------------------------------------------------------------


async def test_db_storage_one_row_per_scope_and_sha(engine, store) -> None:
    services = BlobServices.for_state_store(store)
    data = make_png(1)
    p1 = await store_png(services, "t:a", data)
    p2 = await store_png(services, "t:a", data)
    p3 = await store_png(services, "t:b", data)
    assert p1.ref == p2.ref == p3.ref
    found = await rows(
        engine,
        "SELECT tenant_scope, state, storage, byte_length, width, height, frame_count,"
        " length(generation), data, ingest_transform, purged_at FROM dendrux_blobs"
        " ORDER BY tenant_scope",
    )
    assert [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in found] == [
        ("t:a", "live", "db", len(data), 4, 3, 1, 26),
        ("t:b", "live", "db", len(data), 4, 3, 1, 26),
    ]
    assert bytes(found[0][8]) == data and found[0][9] is None and found[0][10] is None
    assert await services.accessor.read(p1.ref, tenant_id="a") == data
    info = await services.accessor.info(p1.ref.sha256, tenant_id="b")
    assert info is not None and info.storage == "db" and info.created_at.tzinfo is not None


async def test_external_storage_rows_hold_no_bytes(engine, store) -> None:
    blob_store = InMemoryBlobStore()
    services = BlobServices.for_state_store(store, blob_store=blob_store)
    data = make_png(2)
    part = await store_png(services, "-", data)
    (row,) = await rows(engine, "SELECT state, storage, data, generation FROM dendrux_blobs")
    assert row[0] == "live" and row[1] == "external" and row[2] is None
    assert blob_store.object_names() == [f"-/{part.ref.sha256}/{row[3]}"]
    assert await services.accessor.read(part.ref) == data


# ---------------------------------------------------------------------------
# Edges written in the record's transaction
# ---------------------------------------------------------------------------


async def _paused_run(store: SQLAlchemyStateStore, run_id: str, tenant: str | None) -> None:
    await store.create_run(run_id, "a", tenant_id=tenant)
    await store.pause_run(
        run_id, status="waiting_client_tool", pause_data={"agent_name": "a", "iteration": 1}
    )


async def test_edges_on_every_write_method(engine, store) -> None:
    # This fault injection / pause blob exercises the format-1 write path.
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, 1)
    services = BlobServices.for_state_store(store)
    part = await store_png(services, "t:acme", make_png(3))
    s = part.ref.sha256
    run = "R" * 26
    await store.create_run(
        run,
        "a",
        tenant_id="acme",
        input_data={"input": "q", "dendrux.parts": [{"type": "text", "text": "q"}]},
        blob_refs=[BlobEdge("t:acme", s, "input", "input", 1)],
        blob_refs_scope="t:acme",
    )
    await store.save_trace(
        run,
        "tool",
        "x",
        order_index=2,
        blob_refs=[BlobEdge("t:acme", s, "trace", "2", 1)],
        blob_refs_scope="t:acme",
    )
    await store.save_tool_call(
        run,
        tool_call_id="C1",
        provider_tool_call_id="P1",
        tool_name="shot",
        target="client",
        params=None,
        result_payload='"Captured."',
        success=True,
        duration_ms=0,
        iteration_index=1,
        blob_refs=[BlobEdge("t:acme", s, "tool_call", "C1", 0)],
        blob_refs_scope="t:acme",
    )
    await store.pause_run(run, status="waiting_client_tool", pause_data={"iteration": 1})
    assert await store.submit_and_claim(
        run,
        expected_status="waiting_client_tool",
        submitted_data={"submitted_tool_results": []},
        blob_refs=[BlobEdge("t:acme", s, "submission", "1:C1", 0)],
        blob_refs_scope="t:acme",
    )
    await services.catalog.add_blob_refs(run, "t:acme", [BlobEdge("t:acme", s, "seed", "seed", 0)])  # type: ignore[union-attr]
    assert await edge_rows(engine) == [
        ("t:acme", s, run, "input", "input", 1),
        ("t:acme", s, run, "seed", "seed", 0),
        ("t:acme", s, run, "submission", "1:C1", 0),
        ("t:acme", s, run, "tool_call", "C1", 0),
        ("t:acme", s, run, "trace", "2", 1),
    ]
    refs = await services.accessor.references(s, tenant_id="acme")
    assert {(r.owner_kind, r.owner_id, r.ordinal) for r in refs} == {
        ("input", "input", 1),
        ("seed", "seed", 0),
        ("submission", "1:C1", 0),
        ("tool_call", "C1", 0),
        ("trace", "2", 1),
    }
    assert all(r.run_id == run and r.created_at.tzinfo is not None for r in refs)


async def test_retried_edge_write_is_idempotent(engine, store) -> None:
    services = BlobServices.for_state_store(store)
    part = await store_png(services, "-", make_png(4))
    await store.create_run("R1", "a")
    edge = BlobEdge("-", part.ref.sha256, "seed", "seed", 0)
    await services.catalog.add_blob_refs("R1", "-", [edge])  # type: ignore[union-attr]
    await services.catalog.add_blob_refs("R1", "-", [edge])  # type: ignore[union-attr]
    # Distinct trace rows may repeat an edge owner without duplicating that edge.
    for order in range(2):
        await store.save_trace(
            "R1",
            "user",
            "x",
            order_index=order,
            blob_refs=[BlobEdge("-", part.ref.sha256, "trace", "0", 0)],
            blob_refs_scope="-",
        )
    assert len(await edge_rows(engine)) == 2


async def _count(engine: Any, table: str) -> int:
    return int((await rows(engine, f"SELECT count(*) FROM {table}"))[0][0])


async def test_cross_tenant_edge_writes_rejected_and_rolled_back(engine, store) -> None:
    # This fault injection / pause blob exercises the format-1 write path.
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, 1)
    services = BlobServices.for_state_store(store)
    sha_b = (await store_png(services, "t:b", make_png(5))).ref.sha256
    sha_a = (await store_png(services, "t:a", make_png(6))).ref.sha256
    edge_b = [BlobEdge("t:b", sha_b, "trace", "0", 0)]
    errors: list[BaseException] = []

    with pytest.raises(BlobNotFoundError) as e1:
        await store.create_run("RA", "a", tenant_id="a", blob_refs=edge_b, blob_refs_scope="t:b")
    errors.append(e1.value)
    with pytest.raises(BlobNotFoundError) as e1b:  # scope claims a, edge says b
        await store.create_run("RA", "a", tenant_id="a", blob_refs=edge_b, blob_refs_scope="t:a")
    errors.append(e1b.value)
    assert await _count(engine, "agent_runs") == 0

    await _paused_run(store, "RA", "a")
    with pytest.raises(BlobNotFoundError) as e2:
        await store.save_trace(
            "RA", "user", "x", order_index=0, blob_refs=edge_b, blob_refs_scope="t:b"
        )
    errors.append(e2.value)
    with pytest.raises(BlobNotFoundError) as e3:
        await store.save_tool_call(
            "RA",
            tool_call_id="C",
            provider_tool_call_id=None,
            tool_name="t",
            target="server",
            params=None,
            result_payload="1",
            success=True,
            duration_ms=0,
            iteration_index=1,
            blob_refs=edge_b,
            blob_refs_scope="t:b",
        )
    errors.append(e3.value)
    with pytest.raises(BlobNotFoundError) as e4:
        await store.submit_and_claim(
            "RA",
            expected_status="waiting_client_tool",
            submitted_data={"x": 1},
            blob_refs=edge_b,
            blob_refs_scope="t:b",
        )
    errors.append(e4.value)
    with pytest.raises(BlobNotFoundError) as e5:
        await services.catalog.add_blob_refs("RA", "t:b", edge_b)  # type: ignore[union-attr]
    errors.append(e5.value)
    # The tenant-b sha under tenant a's scope: the blob does not exist there.
    with pytest.raises(BlobNotFoundError) as e6:
        await store.save_tool_call(
            "RA",
            tool_call_id="C9",
            provider_tool_call_id=None,
            tool_name="t",
            target="server",
            params=None,
            result_payload="1",
            success=True,
            duration_ms=0,
            iteration_index=1,
            blob_refs=[BlobEdge("t:a", sha_b, "tool_call", "C9", 0)],
            blob_refs_scope="t:a",
        )
    errors.append(e6.value)
    # A trace edge to it is skipped (no edge, no error, no oracle): same as absent.
    await store.save_trace(
        "RA",
        "user",
        "skipped",
        order_index=9,
        blob_refs=[BlobEdge("t:a", sha_b, "trace", "9", 0)],
        blob_refs_scope="t:a",
    )
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM react_traces"))
    for exc in errors:
        assert_clean(exc)
    assert await _count(engine, "react_traces") == 0
    assert await _count(engine, "tool_calls") == 0
    assert await _count(engine, "dendrux_blob_refs") == 0
    run = await store.get_run("RA")
    assert run is not None and run.status == "waiting_client_tool"
    assert (await store.get_pause_state("RA")) == {"agent_name": "a", "iteration": 1}
    # The legitimate edge still works.
    await store.save_trace(
        "RA",
        "user",
        "x",
        order_index=0,
        blob_refs=[BlobEdge("t:a", sha_a, "trace", "0", 0)],
        blob_refs_scope="t:a",
    )
    assert await _count(engine, "dendrux_blob_refs") == 1


async def test_edge_to_purged_or_missing_blob_rolls_back(engine, store) -> None:
    services = BlobServices.for_state_store(store)
    s = (await store_png(services, "-", make_png(7))).ref.sha256
    await store.create_run("R1", "a")
    assert await services.accessor.purge(s, reason="erase")
    with pytest.raises(InvalidImageError) as purged:
        await store.save_tool_call(
            "R1",
            tool_call_id="C9",
            provider_tool_call_id=None,
            tool_name="t",
            target="server",
            params=None,
            result_payload="1",
            success=True,
            duration_ms=0,
            iteration_index=1,
            blob_refs=[BlobEdge("-", s, "tool_call", "C9", 0)],
            blob_refs_scope="-",
        )
    assert purged.value.reason == "purged"
    assert_clean(purged.value)
    with pytest.raises(BlobNotFoundError):
        await store.save_tool_call(
            "R1",
            tool_call_id="C9",
            provider_tool_call_id=None,
            tool_name="t",
            target="server",
            params=None,
            result_payload="1",
            success=True,
            duration_ms=0,
            iteration_index=1,
            blob_refs=[BlobEdge("-", "c" * 64, "tool_call", "C9", 0)],
            blob_refs_scope="-",
        )
    with pytest.raises(BlobNotFoundError):  # unknown run
        await services.catalog.add_blob_refs("nope", "-", [BlobEdge("-", s, "seed", "seed", 0)])  # type: ignore[union-attr]
    with pytest.raises(InvalidImageError):  # seed edges need a live blob too
        await services.catalog.add_blob_refs("R1", "-", [BlobEdge("-", s, "seed", "seed", 0)])  # type: ignore[union-attr]
    assert await _count(engine, "tool_calls") == 0


async def test_trace_edges_skip_non_live_blobs(engine, store) -> None:
    """A trace of a purged/absent image is written without that edge; live ones keep theirs."""
    services = BlobServices.for_state_store(store)
    purged = (await store_png(services, "-", make_png(70))).ref.sha256
    live = (await store_png(services, "-", make_png(71))).ref.sha256
    await store.create_run("R1", "a")
    assert await services.accessor.purge(purged)
    await store.save_trace(
        "R1",
        "user",
        "retried",
        order_index=0,
        blob_refs=[
            BlobEdge("-", purged, "trace", "0", 1),
            BlobEdge("-", live, "trace", "0", 2),
            BlobEdge("-", "d" * 64, "trace", "0", 3),
        ],
        blob_refs_scope="-",
    )
    await store.save_trace(
        "R1",
        "user",
        "only purged",
        order_index=1,
        blob_refs=[BlobEdge("-", purged, "trace", "1", 0)],
        blob_refs_scope="-",
    )
    assert await _count(engine, "react_traces") == 2
    assert await edge_rows(engine) == [("-", live, "R1", "trace", "0", 2)]
    # Mixed with a non-trace edge to the purged blob: still a typed error, rolled back.
    with pytest.raises(InvalidImageError):
        await store.save_tool_call(
            "R1",
            tool_call_id="C1",
            provider_tool_call_id=None,
            tool_name="t",
            target="server",
            params=None,
            result_payload="1",
            success=True,
            duration_ms=0,
            iteration_index=1,
            blob_refs=[
                BlobEdge("-", live, "tool_call", "C1", 0),
                BlobEdge("-", purged, "tool_call", "C1", 1),
            ],
            blob_refs_scope="-",
        )
    assert await _count(engine, "tool_calls") == 0
    # Tenant mismatch is never skipped, even for trace edges.
    with pytest.raises(BlobNotFoundError):
        await store.save_trace(
            "R1",
            "user",
            "x",
            order_index=2,
            blob_refs=[BlobEdge("t:x", live, "trace", "2", 0)],
            blob_refs_scope="t:x",
        )


# ---------------------------------------------------------------------------
# Run-scoped reads
# ---------------------------------------------------------------------------


async def test_read_image_authorisation(engine, store) -> None:
    run_store = RunStore(store)
    services = BlobServices.for_state_store(store)
    data = make_png(8)
    other = make_png(9)
    sa = (await store_png(services, "t:a", data)).ref.sha256
    sb_only = (await store_png(services, "t:b", other)).ref.sha256
    await store_png(services, "t:b", data)  # same bytes, tenant b
    await store.create_run(
        "RA",
        "a",
        tenant_id="a",
        blob_refs=[BlobEdge("t:a", sa, "input", "input", 0)],
        blob_refs_scope="t:a",
    )
    await store.create_run(
        "RB",
        "a",
        tenant_id="b",
        blob_refs=[BlobEdge("t:b", sa, "input", "input", 0)],
        blob_refs_scope="t:b",
    )
    await store.create_run("RN", "a")

    assert await run_store.read_image("RA", sa) == data
    assert await run_store.read_image("RA", sa, tenant_id="a") == data
    assert await run_store.read_image("RB", sa, tenant_id="b") == data
    failures = [
        run_store.read_image("RA", sb_only),  # tenant b's blob, no edge
        run_store.read_image("RA", sa, tenant_id="b"),  # wrong tenant
        run_store.read_image("RA", sa, tenant_id=None),  # untenanted scope
        run_store.read_image("RN", sa),  # no edge
        run_store.read_image("missing", sa),
    ]
    messages = []
    for coro in failures:
        with pytest.raises(BlobNotFoundError) as info:
            await coro
        assert_clean(info.value)
        messages.append(type(info.value))
    assert set(messages) == {BlobNotFoundError}
    # Purged: the edge stays (audit) but the bytes are gone.
    assert await run_store.blobs.purge(sa, tenant_id="a")
    with pytest.raises(BlobNotFoundError):
        await run_store.read_image("RA", sa)
    assert await run_store.read_image("RB", sa) == data
    assert len(await run_store.blobs.references(sa, tenant_id="a")) == 1


# ---------------------------------------------------------------------------
# §6 sample rows through the recorder
# ---------------------------------------------------------------------------


async def test_canonical_sample_rows(engine, store) -> None:
    # This fault injection / pause blob exercises the format-1 write path.
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, 1)
    from tests.unit.test_blob_catalog import make_png as _mk

    png = _mk(42, width=400, height=200)
    services = BlobServices.for_state_store(store)
    rt = runtime_for(services, "t:acme")
    run = "R" * 26
    await store.create_run(
        run, "screen_agent", tenant_id="acme", input_data={"input": "Take a screenshot."}
    )
    await store.pause_run(run, status="waiting_client_tool", pause_data={"iteration": 1})
    (part,) = await ingest(
        [Image.from_bytes(png)], policy=rt.policy, runtime=rt, origin="client_tool"
    )
    img = part.ref.to_dict()
    submitted = [
        {
            "name": "take_screenshot",
            "call_id": "C1",
            "payload": '"Captured."',
            "success": True,
            "error": None,
            "duration_ms": 0,
            "images": [img],
        }
    ]
    assert await store.submit_and_claim(
        run,
        expected_status="waiting_client_tool",
        submitted_data={"submitted_tool_results": submitted},
        blob_refs=[BlobEdge("t:acme", part.ref.sha256, "submission", "1:C1", 0)],
        blob_refs_scope="t:acme",
    )
    rec = PersistenceRecorder(
        store,
        run,
        model="claude-sonnet-4-6",
        provider_name="AnthropicProvider",
        provider=_SurfaceProvider("anthropic"),
        initial_order_index=2,
        target_lookup={"take_screenshot": "client"},
        tenant_scope="t:acme",
    )
    result = ToolResult(name="take_screenshot", call_id="C1", payload='"Captured."', images=(part,))
    tc = ToolCall(name="take_screenshot", params={}, id="C1", provider_tool_call_id="P1")
    await rec.on_tool_completed(run, tc, result, 2)
    tool_msg = Message.with_parts(
        Role.TOOL,
        [TextPart('"Captured."'), part],
        name="take_screenshot",
        call_id="C1",
    )
    await rec.on_message_appended(run, tool_msg, 2)
    b64 = base64.b64encode(png).decode()
    await rec.on_llm_call_completed(
        run,
        LLMResponse(
            text="The code word is ZEBRA42.",
            usage=UsageStats(input_tokens=10, output_tokens=5),
            provider_request={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": [
                                    {"type": "text", "text": '"Captured."'},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": b64,
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ]
            },
        ),
        2,
        semantic_messages=[Message(role=Role.USER, content="Take a screenshot."), tool_msg],
    )

    s = part.ref.sha256
    assert await edge_rows(engine) == [
        ("t:acme", s, run, "submission", "1:C1", 0),
        ("t:acme", s, run, "tool_call", "C1", 0),
        ("t:acme", s, run, "trace", "2", 1),
    ]
    from dendrux.db.models import AgentRun, LLMInteraction, ReactTrace, RunEvent, ToolCallRecord

    ((result_col, meta, target),) = await typed(
        engine, ToolCallRecord.result, ToolCallRecord.meta, ToolCallRecord.target
    )
    assert result_col == "Captured." and target == "client"
    assert meta == {"dendrux.images": [img]}
    ((content, tmeta),) = await typed(engine, ReactTrace.content, ReactTrace.meta)
    assert content == '"Captured."'
    assert tmeta["dendrux.parts"] == [{"type": "text", "text": '"Captured."'}, img]
    ((sem, prov),) = await typed(
        engine, LLMInteraction.semantic_request, LLMInteraction.provider_request
    )
    assert sem["messages"][1] == {
        "role": "tool",
        "content": '"Captured."',
        "call_id": "C1",
        "name": "take_screenshot",
        "parts": [{"type": "text", "text": '"Captured."'}, img],
    }
    assert sem["image_stats"] == {"count": 1, "bytes": len(png), "estimated_tokens": 120}
    image_block = prov["messages"][0]["content"][0]["content"][1]
    assert image_block["source"]["data"] == "dendrux-image:sha256:" + s
    ((run_input,),) = await typed(engine, AgentRun.input_data)
    assert run_input == {"input": "Take a screenshot."}
    events = await typed(engine, RunEvent.event_type, RunEvent.data)
    tool_event = next(e[1] for e in events if e[0] == "tool.completed")
    assert tool_event["image_count"] == 1

    # Leak scan over every persisted column except the blob bytes themselves.
    dump: list[str] = []
    for table in Base.metadata.sorted_tables:
        cols = [
            c.name
            for c in table.columns
            if not (table.name == "dendrux_blobs" and c.name == "data")
        ]
        for r in await rows(engine, f"SELECT {', '.join(cols)} FROM {table.name}"):
            dump.append(repr(tuple(r)))
    blob = "\n".join(dump)
    assert "iVBORw0K" not in blob and "data:image" not in blob and b64[:32] not in blob
    assert png.hex()[:64] not in blob


# ---------------------------------------------------------------------------
# GC grace, orphan edges, DB clock
# ---------------------------------------------------------------------------


async def test_gc_grace_and_reuse_touch(engine, store) -> None:
    services = BlobServices.for_state_store(store)
    data = make_png(10)
    s = (await store_png(services, "-", data)).ref.sha256
    result = await services.accessor.collect_garbage(older_than=timedelta(hours=1))
    assert result.deleted_blobs == 0
    await age_blob(engine, "-", s, 3)
    await store_png(services, "-", data)  # reuse refreshes last_used_at
    assert (
        await services.accessor.collect_garbage(older_than=timedelta(hours=1))
    ).deleted_blobs == 0
    await age_blob(engine, "-", s, 3)
    result = await services.accessor.collect_garbage(older_than=timedelta(hours=1))
    assert result.deleted_blobs == 1 and result.bytes_freed == len(data)
    assert await _count(engine, "dendrux_blobs") == 0


async def test_referenced_blob_survives_gc(engine, store) -> None:
    services = BlobServices.for_state_store(store)
    s = (await store_png(services, "-", make_png(11))).ref.sha256
    await store.create_run(
        "R1", "a", blob_refs=[BlobEdge("-", s, "input", "input", 0)], blob_refs_scope="-"
    )
    await age_blob(engine, "-", s, 48)
    result = await services.accessor.collect_garbage(older_than=timedelta(hours=1))
    assert result.deleted_blobs == 0 and result.skipped_referenced == 0
    assert await _count(engine, "dendrux_blobs") == 1


async def test_orphan_edges_collected_on_sqlite_without_fk(tmp_path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'nofk.db'}"
    engine = create_async_engine(url, poolclass=NullPool)  # production default: FKs off
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        store = SQLAlchemyStateStore(engine)
        services = BlobServices.for_state_store(store)
        s = (await store_png(services, "-", make_png(12))).ref.sha256
        await store.create_run(
            "R1", "a", blob_refs=[BlobEdge("-", s, "input", "input", 0)], blob_refs_scope="-"
        )
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM agent_runs WHERE id = 'R1'"))  # app SQL
        assert await _count(engine, "dendrux_blob_refs") == 1  # no cascade on SQLite
        await age_blob(engine, "-", s, 48)
        result = await services.accessor.collect_garbage(older_than=timedelta(hours=1))
        assert result.deleted_orphan_edges == 1 and result.deleted_blobs == 1
        assert await _count(engine, "dendrux_blob_refs") == 0
        assert await _count(engine, "dendrux_blobs") == 0
    finally:
        await engine.dispose()


async def test_run_delete_cascades_edges(engine, store) -> None:
    services = BlobServices.for_state_store(store)
    s = (await store_png(services, "-", make_png(13))).ref.sha256
    await store.create_run(
        "R1", "a", blob_refs=[BlobEdge("-", s, "input", "input", 0)], blob_refs_scope="-"
    )
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM agent_runs WHERE id = 'R1'"))
    assert await _count(engine, "dendrux_blob_refs") == 0  # FK cascade (PG; test SQLite has FKs on)
    from sqlalchemy.exc import IntegrityError

    await store.create_run(
        "R2", "a", blob_refs=[BlobEdge("-", s, "input", "input", 0)], blob_refs_scope="-"
    )
    with pytest.raises(IntegrityError):  # RESTRICT: a referenced blob row cannot be deleted
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM dendrux_blobs"))


async def test_postgres_db_clock_ignores_skewed_host(engine, store, monkeypatch) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres uses the DB clock; SQLite uses the host clock (documented)")
    from datetime import UTC, datetime

    import dendrux.media._ingest as ingest_mod
    import dendrux.runtime._blob_catalog as cat

    skewed = lambda: datetime.now(UTC) + timedelta(hours=2)  # noqa: E731
    monkeypatch.setattr(cat, "_py_now", skewed)
    monkeypatch.setattr(ingest_mod, "_now", skewed)
    services = BlobServices.for_state_store(store)
    s = (await store_png(services, "-", make_png(14))).ref.sha256
    info = await services.accessor.info(s)
    assert info is not None
    db_now = await store.current_time()
    assert abs((info.last_used_at - db_now).total_seconds()) < 60
    result = await services.accessor.collect_garbage(older_than=timedelta(hours=1))
    assert result.deleted_blobs == 0


# ---------------------------------------------------------------------------
# External store: second process
# ---------------------------------------------------------------------------


async def test_external_store_second_process_read(tmp_path, engine) -> None:
    if engine.dialect.name == "postgresql":
        url = engine.url.render_as_string(hide_password=False)
    else:
        url = f"sqlite+aiosqlite:///{tmp_path / 'proc.db'}"
        e = create_async_engine(url, poolclass=NullPool)
        async with e.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await e.dispose()
    shared = InMemoryBlobStore()
    data = make_png(15)

    proc1 = RunStore.from_database_url(url, blob_store=shared)
    try:
        state1 = proc1._state
        services1 = BlobServices.for_state_store(state1, blob_store=shared)
        part = await store_png(services1, "t:acme", data)
        await state1.create_run(
            "R1",
            "a",
            tenant_id="acme",
            blob_refs=[BlobEdge("t:acme", part.ref.sha256, "input", "input", 0)],
            blob_refs_scope="t:acme",
        )
    finally:
        await proc1.close()

    async with RunStore.from_database_url(url, blob_store=shared) as proc2:
        assert await proc2.read_image("R1", part.ref.sha256, tenant_id="acme") == data
    async with RunStore.from_database_url(url) as no_store:
        from dendrux.errors import ImagePersistenceUnsupportedError

        with pytest.raises(ImagePersistenceUnsupportedError) as info:
            await no_store.read_image("R1", part.ref.sha256)
        assert info.value.reason == "blob_store_not_configured"


# ---------------------------------------------------------------------------
# P-04: pause JSON grows <= 512 B per image
# ---------------------------------------------------------------------------


async def test_pause_json_bounded_with_ten_large_images(engine, store) -> None:
    services = BlobServices.for_state_store(store)
    rt = runtime_for(services, "-")
    size = 5 * 1024 * 1024 - 4096
    images = [Image.from_bytes(make_png(100 + i, pad=size)) for i in range(10)]
    parts = await ingest(images, policy=rt.policy, runtime=rt, origin="user")
    assert all(p.ref.byte_length <= 5 * 1024 * 1024 for p in parts)
    from dendrux.types import PauseState
    from dendrux.types import UsageStats as Usage

    text_only = PauseState(
        agent_name="a",
        pending_tool_calls=[],
        history=[Message(role=Role.USER, content="look")],
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=Usage(),
    )
    with_images = PauseState(
        agent_name="a",
        pending_tool_calls=[],
        history=[Message.with_parts(Role.USER, [TextPart("look"), *parts])],
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=Usage(),
    )
    await store.create_run("R1", "a")
    await store.pause_run("R1", status="waiting_client_tool", pause_data=with_images.to_dict())
    stored = await store.get_pause_state("R1")
    assert stored is not None
    growth = len(json.dumps(stored)) - len(json.dumps(text_only.to_dict()))
    assert growth <= 10 * 512
    for d in stored["history"][0]["parts"][1:]:
        assert len(json.dumps(d)) <= 512
    assert PauseState.from_dict(stored).history[0].images == tuple(ImagePart(p.ref) for p in parts)
