"""in-memory blob catalog, resolver, accessor, recorder shapes.

The SQL catalog and the fault/race matrix live in
``tests/integration/test_image_persistence.py`` and
``tests/integration/test_blob_lifecycle_faults.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import (
    BlobIntegrityError,
    BlobNotFoundError,
    ImageLimitError,
    ImagePersistenceUnsupportedError,
    InvalidImageError,
    PersistenceError,
)
from dendrux.media import (
    BlobCatalog,
    BlobEdge,
    BlobKey,
    BlobResolver,
    Image,
    ImagePolicy,
    ImageRuntime,
    InMemoryBlobStore,
    ingest,
)
from dendrux.runtime._blob_catalog import (
    BlobServices,
    CachingBlobResolver,
    CatalogBlobAccessor,
    InMemoryBlobCatalog,
    as_blob_catalog,
    edges_for_parts,
    scrub_inline_images,
)
from dendrux.runtime.persistence import PersistenceRecorder, image_dict, image_stats
from dendrux.types import (
    ImagePart,
    ImageRef,
    LLMResponse,
    Message,
    Role,
    TextPart,
    ToolCall,
    ToolResult,
    UsageStats,
)

# ---------------------------------------------------------------------------
# Helpers (also imported by the integration tests)
# ---------------------------------------------------------------------------


def make_png(seed: int = 0, *, width: int = 4, height: int = 3, pad: int = 0) -> bytes:
    """A structurally valid PNG; ``seed`` changes the bytes, ``pad`` adds a private chunk."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes([seed % 256, 7, 9]) * width for _ in range(height))
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
    out += chunk(b"tEXt", b"seed\x00" + str(seed).encode())
    if pad:
        out += chunk(b"prVt", b"\x00" * pad)
    out += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return out


def runtime_for(services: BlobServices, scope: str, **policy: Any) -> ImageRuntime:
    return ImageRuntime(
        policy=ImagePolicy(**policy),
        catalog=services.catalog,
        blob_store=services.blob_store,
        resolver=services.resolver,
        tenant_scope=scope,
    )


async def store_png(services: BlobServices, scope: str, data: bytes) -> ImagePart:
    rt = runtime_for(services, scope)
    (part,) = await ingest([Image.from_bytes(data)], policy=rt.policy, runtime=rt, origin="user")
    return part


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Protocol conformance and wiring
# ---------------------------------------------------------------------------


async def test_in_memory_services_implement_protocols() -> None:
    services = BlobServices.in_memory()
    assert isinstance(services.catalog, BlobCatalog)
    assert isinstance(services.resolver, BlobResolver)
    assert services.memory is services.catalog
    assert as_blob_catalog(object()) is None


async def test_custom_state_store_without_catalog_raises_state_store() -> None:
    services = BlobServices.for_state_store(object())
    assert services.catalog is None and services.resolver is None
    with pytest.raises(ImagePersistenceUnsupportedError) as info:
        await services.accessor.info("a" * 64)
    assert info.value.reason == "state_store"


# ---------------------------------------------------------------------------
# In-memory catalog: ingest, read, tenants
# ---------------------------------------------------------------------------


async def test_ingest_store_and_read_round_trip() -> None:
    services = BlobServices.in_memory()
    data = make_png(1)
    part = await store_png(services, "t:acme", data)
    assert part.ref.sha256 == sha(data)
    assert await services.accessor.read(part.ref, tenant_id="acme") == data
    assert await services.accessor.read(part.ref.sha256, tenant_id="acme") == data
    info = await services.accessor.info(part.ref.sha256, tenant_id="acme")
    assert info is not None and info.state == "live" and info.storage == "external"
    # Same bytes again: one row, one object.
    again = await store_png(services, "t:acme", data)
    assert again.ref == part.ref
    store = services.blob_store.inner  # type: ignore[union-attr]
    assert len(store.object_names()) == 1


async def test_tenant_isolation_uniform_not_found() -> None:
    services = BlobServices.in_memory()
    data = make_png(2)
    part = await store_png(services, "t:a", data)
    with pytest.raises(BlobNotFoundError) as other:
        await services.accessor.read(part.ref, tenant_id="b")
    with pytest.raises(BlobNotFoundError) as absent:
        await services.accessor.read("f" * 64, tenant_id="b")
    assert type(other.value) is type(absent.value)
    assert str(other.value).replace(part.ref.sha256[:12], "X") == str(absent.value).replace(
        "f" * 12, "X"
    )
    assert await services.accessor.info(part.ref.sha256, tenant_id="b") is None
    # Same bytes in tenant b: separate row.
    await store_png(services, "t:b", data)
    assert await services.accessor.read(part.ref, tenant_id="b") == data


async def test_read_rejects_mismatched_descriptor_and_bad_sha() -> None:
    services = BlobServices.in_memory()
    part = await store_png(services, "-", make_png(3))
    lie = ImageRef(
        sha256=part.ref.sha256,
        media_type="image/png",
        byte_length=part.ref.byte_length + 1,
        width=4,
        height=3,
    )
    with pytest.raises(InvalidImageError) as info:
        await services.accessor.read(lie)
    assert info.value.reason == "descriptor_mismatch"
    with pytest.raises(ValueError):
        await services.accessor.read("not-a-sha")


async def test_resolver_verifies_store_bytes() -> None:
    services = BlobServices.in_memory()
    part = await store_png(services, "-", make_png(4))
    info = await services.catalog.get_row("-", part.ref.sha256)  # type: ignore[union-attr]
    assert info is not None
    inner = services.blob_store.inner  # type: ignore[union-attr]
    key = BlobKey("-", part.ref.sha256, info.generation)
    await inner.put(key, make_png(5), media_type="image/png")  # tampered
    fresh = CachingBlobResolver(services.catalog, services.blob_store)  # type: ignore[arg-type]
    with pytest.raises(BlobIntegrityError) as err:
        await fresh.read("-", info)
    assert err.value.check in ("length", "sha256")
    await inner.delete(key)
    with pytest.raises(BlobNotFoundError):
        await fresh.read("-", info)


async def test_resolver_cache_is_generation_keyed_lru() -> None:
    services = BlobServices.in_memory()
    a = make_png(6)
    b = make_png(7)
    pa = await store_png(services, "-", a)
    pb = await store_png(services, "-", b)
    catalog = services.catalog
    assert catalog is not None
    resolver = CachingBlobResolver(catalog, services.blob_store, max_bytes=len(a) + 1)
    ia = await catalog.get_row("-", pa.ref.sha256)
    ib = await catalog.get_row("-", pb.ref.sha256)
    assert ia and ib
    assert await resolver.read("-", ia) == a
    assert resolver.cached_bytes == len(a)
    assert await resolver.read("-", ib) == b  # evicts a
    assert resolver.cached_bytes == len(b)
    store = services.blob_store.inner  # type: ignore[union-attr]
    calls = len(store.calls)
    assert await resolver.read("-", ib) == b
    assert len(store.calls) == calls  # cache hit
    # A non-live row is never served from cache.
    from dataclasses import replace

    with pytest.raises(BlobNotFoundError):
        await resolver.read("-", replace(ib, state="purged"))


async def test_db_storage_resolver_without_store() -> None:
    catalog = InMemoryBlobCatalog()
    services_rt = ImageRuntime(
        policy=ImagePolicy(),
        catalog=catalog,
        blob_store=None,
        resolver=None,
        tenant_scope="-",
    )
    data = make_png(8)
    (part,) = await ingest(
        [Image.from_bytes(data)], policy=services_rt.policy, runtime=services_rt, origin="user"
    )
    info = await catalog.get_row("-", part.ref.sha256)
    assert info is not None and info.storage == "db" and info.state == "live"
    accessor = CatalogBlobAccessor(catalog)
    assert await accessor.read(part.ref) == data
    assert await catalog.read_blob_data(BlobKey("-", part.ref.sha256, "wrong")) is None


async def test_external_row_without_store_is_unsupported() -> None:
    services = BlobServices.in_memory()
    part = await store_png(services, "-", make_png(9))
    bare = CatalogBlobAccessor(services.catalog)
    with pytest.raises(ImagePersistenceUnsupportedError) as info:
        await bare.read(part.ref)
    assert info.value.reason == "blob_store_not_configured"
    with pytest.raises(ImagePersistenceUnsupportedError):
        await bare.purge(part.ref.sha256)


# ---------------------------------------------------------------------------
# In-memory edges, purge, GC
# ---------------------------------------------------------------------------


def _age(services: BlobServices, scope: str, sha256: str, hours: float) -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    memory = services.memory
    assert memory is not None
    row = memory._rows[(scope, sha256)]
    old = datetime.now(UTC) - timedelta(hours=hours)
    row.info = replace(row.info, last_used_at=old, state_changed_at=old)


async def test_in_memory_edges_released_at_run_end_and_gc() -> None:
    services = BlobServices.in_memory()
    memory = services.memory
    assert memory is not None
    part = await store_png(services, "t:a", make_png(10))
    s = part.ref.sha256
    await memory.add_blob_refs("R1", "t:a", [BlobEdge("t:a", s, "input", "input", 0)])
    await memory.add_blob_refs("R1", "t:a", [BlobEdge("t:a", s, "input", "input", 0)])
    assert len(await services.accessor.references(s, tenant_id="a")) == 1
    assert await memory.run_references("t:a", "R1", s)
    assert not await memory.run_references("t:b", "R1", s)
    _age(services, "t:a", s, 30)
    result = await services.accessor.collect_garbage()
    assert result.deleted_blobs == 0  # referenced during the run
    memory.release_run("R1")
    _age(services, "t:a", s, 30)
    result = await services.accessor.collect_garbage()
    assert result.deleted_blobs == 1 and result.bytes_freed == part.ref.byte_length
    assert services.blob_store.inner.object_names() == []  # type: ignore[union-attr]
    with pytest.raises(BlobNotFoundError):
        await services.accessor.read(part.ref, tenant_id="a")


async def test_in_memory_edge_tenant_mismatch_and_purged() -> None:
    services = BlobServices.in_memory()
    memory = services.memory
    assert memory is not None
    part = await store_png(services, "t:a", make_png(11))
    s = part.ref.sha256
    with pytest.raises(BlobNotFoundError):
        await memory.add_blob_refs("R1", "t:a", [BlobEdge("t:b", s, "input", "input", 0)])
    await memory.add_blob_refs("R1", "t:a", [BlobEdge("t:a", s, "input", "input", 0)])
    with pytest.raises(BlobNotFoundError):
        await memory.add_blob_refs("R1", "t:b", [BlobEdge("t:b", s, "seed", "seed", 0)])
    with pytest.raises(BlobNotFoundError):
        await memory.add_blob_refs("R2", "t:a", [BlobEdge("t:a", "e" * 64, "seed", "seed", 0)])
    assert await services.accessor.purge(s, tenant_id="a", reason="gdpr") is True
    with pytest.raises(InvalidImageError) as info:
        await memory.add_blob_refs("R3", "t:a", [BlobEdge("t:a", s, "seed", "seed", 0)])
    assert info.value.reason == "purged"


async def test_purge_semantics() -> None:
    services = BlobServices.in_memory()
    data = make_png(12)
    part = await store_png(services, "-", data)
    s = part.ref.sha256
    assert await services.accessor.purge("d" * 64) is False
    assert await services.accessor.purge(s, reason="erase") is True
    assert await services.accessor.purge(s) is True  # already recorded
    info = await services.accessor.info(s)
    assert info is not None and info.state == "purged" and info.purge_reason == "erase"
    assert info.purged_at is not None
    assert services.blob_store.inner.object_names() == []  # type: ignore[union-attr]
    with pytest.raises(BlobNotFoundError):
        await services.accessor.read(part.ref)
    with pytest.raises(InvalidImageError) as err:
        await store_png(services, "-", data)
    assert err.value.reason == "purged"
    with pytest.raises(ValueError):
        await services.accessor.purge(s, reason="x" * 201)


async def test_gc_arguments_validated() -> None:
    services = BlobServices.in_memory()
    with pytest.raises(ValueError):
        await services.accessor.collect_garbage(older_than=timedelta(minutes=59))
    with pytest.raises(ValueError):
        await services.accessor.collect_garbage(limit=0)


async def test_full_store_triggers_one_gc_pass() -> None:
    first = make_png(13, pad=600)
    second = make_png(14, pad=600)
    services = BlobServices.in_memory(blob_store=InMemoryBlobStore(max_bytes=len(first) + 100))
    p1 = await store_png(services, "-", first)
    _age(services, "-", p1.ref.sha256, 2)  # finished run, unreferenced, past 1 h
    p2 = await store_png(services, "-", second)
    assert await services.accessor.info(p1.ref.sha256) is None
    assert await services.accessor.read(p2.ref) == second
    # Referenced blob is kept: a full store then fails with ImageLimitError.
    memory = services.memory
    assert memory is not None
    await memory.add_blob_refs("R", "-", [BlobEdge("-", p2.ref.sha256, "input", "input", 0)])
    _age(services, "-", p2.ref.sha256, 2)
    with pytest.raises(ImageLimitError):
        await store_png(services, "-", make_png(15, pad=600))


def _slip_edge_after_mark(memory: InMemoryBlobCatalog, s: str) -> None:
    from dendrux.runtime._blob_catalog import _MemEdge

    real = memory.mark_deleting_row

    async def mark_then_edge(*args: Any, **kwargs: Any) -> tuple[str, str] | None:
        marked = await real(*args, **kwargs)
        memory._run_scopes["R2"] = "-"
        memory._add_edge(
            ("R2", "seed", "seed", 0),
            _MemEdge(BlobEdge("-", s, "seed", "seed", 0), "R2", part_created()),
        )
        return marked

    memory.mark_deleting_row = mark_then_edge  # type: ignore[method-assign]


async def test_in_memory_orphan_edges() -> None:
    services = BlobServices.in_memory()
    memory = services.memory
    assert memory is not None
    part = await store_png(services, "-", make_png(16))
    s = part.ref.sha256
    await memory.add_blob_refs("R", "-", [BlobEdge("-", s, "input", "input", 0)])
    memory._run_scopes.pop("R")  # run gone without release
    assert await memory.delete_orphan_edges(10) == 1
    assert await memory.has_edges("-", s) is False
    assert memory._by_blob == {} and memory._by_run == {}


async def test_gc_edge_after_mark_external_stays_deleting() -> None:
    """P2-1: never revert an external row (an ingest may have deleted its object)."""
    services = BlobServices.in_memory()
    memory = services.memory
    assert memory is not None
    part = await store_png(services, "-", make_png(17))
    s = part.ref.sha256
    _age(services, "-", s, 30)
    _slip_edge_after_mark(memory, s)
    result = await services.accessor.collect_garbage()
    assert result.skipped_referenced == 1 and result.deleted_blobs == 0
    info = await memory.get_row("-", s)
    assert info is not None and info.state == "deleting"
    assert len(services.blob_store.inner.object_names()) == 1  # type: ignore[union-attr]
    # Not listed for recovery while referenced (no starvation), and a re-ingest revives it.
    assert await memory.stuck_rows(part_created(), 10) == []
    assert (await store_png(services, "-", make_png(17))).ref == part.ref
    assert await services.accessor.read(part.ref) == make_png(17)


async def test_gc_edge_after_mark_db_storage_reverts() -> None:
    catalog = InMemoryBlobCatalog()
    rt = ImageRuntime(
        policy=ImagePolicy(), catalog=catalog, blob_store=None, resolver=None, tenant_scope="-"
    )
    data = make_png(18)
    (part,) = await ingest([Image.from_bytes(data)], policy=rt.policy, runtime=rt, origin="user")
    s = part.ref.sha256
    from dataclasses import replace
    from datetime import UTC, datetime

    row = catalog._rows[("-", s)]
    row.info = replace(row.info, last_used_at=datetime.now(UTC) - timedelta(hours=30))
    _slip_edge_after_mark(catalog, s)
    result = await CatalogBlobAccessor(catalog).collect_garbage()
    assert result.skipped_referenced == 1
    info = await catalog.get_row("-", s)
    assert info is not None and info.state == "live"
    assert await CatalogBlobAccessor(catalog).read(part.ref) == data


def part_created() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Recorder shapes (fake store)
# ---------------------------------------------------------------------------


def _ref(n: int = 1) -> ImageRef:
    return ImageRef(
        sha256=f"{n:064x}", media_type="image/png", byte_length=1000, width=400, height=200
    )


class _FakeStore:
    def __init__(self) -> None:
        self.save_trace = AsyncMock()
        self.save_tool_call = AsyncMock()
        self.save_run_event = AsyncMock()
        self.save_llm_interaction = AsyncMock()
        self.save_usage = AsyncMock()
        self.touch_progress = AsyncMock()


async def test_recorder_text_only_passes_no_new_kwargs() -> None:
    store = _FakeStore()
    rec = PersistenceRecorder(store, "R", tenant_scope="t:a")  # type: ignore[arg-type]
    await rec.on_message_appended("R", Message(role=Role.USER, content="hi"), 1)
    kwargs = store.save_trace.call_args.kwargs
    assert "blob_refs" not in kwargs and kwargs["meta"] == {"iteration": 1}
    tc = ToolCall(name="t", params={}, id="C1")
    await rec.on_tool_completed("R", tc, ToolResult(name="t", call_id="C1", payload='"ok"'), 1)
    kwargs = store.save_tool_call.call_args.kwargs
    assert "blob_refs" not in kwargs and "meta" not in kwargs
    assert "image_count" not in store.save_run_event.call_args.kwargs["data"]


async def test_recorder_trace_and_tool_call_images() -> None:
    store = _FakeStore()
    rec = PersistenceRecorder(store, "R", tenant_scope="t:acme", initial_order_index=2)  # type: ignore[arg-type]
    ref = _ref()
    msg = Message.with_parts(
        Role.TOOL, [TextPart('"Captured."'), ImagePart(ref)], name="shot", call_id="C1"
    )
    await rec.on_message_appended("R", msg, 2)
    kwargs = store.save_trace.call_args.kwargs
    assert kwargs["meta"]["dendrux.parts"] == [
        {"type": "text", "text": '"Captured."'},
        ref.to_dict(),
    ]
    assert kwargs["blob_refs"] == [BlobEdge("t:acme", ref.sha256, "trace", "2", 1)]
    assert kwargs["blob_refs_scope"] == "t:acme"

    tc = ToolCall(name="shot", params={}, id="C1")
    result = ToolResult(
        name="shot",
        call_id="C1",
        payload='"Captured."',
        images=(ImagePart(ref, detail="low"),),
        notes=("[image from MCP could not be used: corrupt]",),
    )
    await rec.on_tool_completed("R", tc, result, 2)
    kwargs = store.save_tool_call.call_args.kwargs
    assert kwargs["meta"]["dendrux.images"] == [{**ref.to_dict(), "detail": "low"}]
    assert kwargs["meta"]["dendrux.image_notes"] == ["[image from MCP could not be used: corrupt]"]
    assert kwargs["blob_refs"] == [BlobEdge("t:acme", ref.sha256, "tool_call", "C1", 0)]
    assert store.save_run_event.call_args.kwargs["data"]["image_count"] == 1


async def test_recorder_keeps_runtime_supplied_user_parts() -> None:
    store = _FakeStore()
    rec = PersistenceRecorder(store, "R", tenant_scope="-")  # type: ignore[arg-type]
    ref = _ref(2)
    wire = [{"type": "text", "text": "framed"}, ref.to_dict()]
    await rec.on_message_appended(
        "R", Message(role=Role.USER, content="raw", meta={"dendrux.parts": wire}), 1
    )
    kwargs = store.save_trace.call_args.kwargs
    assert kwargs["meta"]["dendrux.parts"] == wire
    assert kwargs["blob_refs"] == [BlobEdge("-", ref.sha256, "trace", "0", 1)]


async def test_recorder_without_scope_rejects_edges() -> None:
    store = _FakeStore()
    rec = PersistenceRecorder(store, "R")  # type: ignore[arg-type]
    msg = Message.with_parts(Role.USER, [ImagePart(_ref())])
    with pytest.raises(ImagePersistenceUnsupportedError) as info:
        await rec.on_message_appended("R", msg, 1)
    assert info.value.reason == "state_store"
    store.save_trace.assert_not_called()


async def test_recorder_llm_interaction_parts_stats_and_scrub() -> None:
    store = _FakeStore()
    rec = PersistenceRecorder(
        store,  # type: ignore[arg-type]
        "R",
        model="claude-sonnet-4-6",
        provider_name="AnthropicProvider",
        tenant_scope="t:acme",
        provider=_SurfaceProvider("anthropic"),
    )
    png = make_png(20)
    ref = ImageRef(
        sha256=sha(png), media_type="image/png", byte_length=len(png), width=400, height=200
    )
    b64 = base64.b64encode(png).decode()
    semantic = [
        Message(role=Role.USER, content="hi"),
        Message.with_parts(Role.TOOL, [TextPart('"x"'), ImagePart(ref)], name="t", call_id="C1"),
    ]
    provider_request = {
        "messages": [
            {
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    }
                ]
            },
            {"content": [{"image_url": {"url": f"data:image/png;base64,{b64}"}}]},
            {"content": "plain text stays"},
        ]
    }
    response = LLMResponse(
        text="ok",
        usage=UsageStats(input_tokens=1, output_tokens=1),
        provider_request=provider_request,
    )
    await rec.on_llm_call_completed("R", response, 2, semantic_messages=semantic)
    kwargs = store.save_llm_interaction.call_args.kwargs
    req = kwargs["semantic_request"]
    assert "parts" not in req["messages"][0]
    assert req["messages"][1]["parts"][1] == ref.to_dict()
    assert req["image_stats"] == {"count": 1, "bytes": len(png), "estimated_tokens": 120}
    dumped = json.dumps(kwargs["provider_request"])
    assert b64[:40] not in dumped and "data:image" not in dumped
    assert dumped.count("dendrux-image:sha256:" + sha(png)) == 2
    assert "plain text stays" in dumped
    assert provider_request["messages"][0]["content"][0]["source"]["data"] == b64  # not mutated


async def test_recorder_text_only_llm_interaction_untouched() -> None:
    store = _FakeStore()
    rec = PersistenceRecorder(store, "R", provider_name="AnthropicProvider")  # type: ignore[arg-type]
    provider_request = {"messages": [{"content": "data:image/png;base64,AAAA"}]}
    response = LLMResponse(
        text="ok",
        usage=UsageStats(input_tokens=1, output_tokens=1),
        provider_request=provider_request,
    )
    await rec.on_llm_call_completed(
        "R", response, 1, semantic_messages=[Message(role=Role.USER, content="hi")]
    )
    kwargs = store.save_llm_interaction.call_args.kwargs
    assert kwargs["provider_request"] is provider_request
    assert "image_stats" not in kwargs["semantic_request"]


class _SurfaceProvider:
    def __init__(self, surface: str) -> None:
        self.surface = surface

    def image_capability_key(self, model: str) -> tuple[str, str, str]:
        return (self.surface, "", model)


def test_image_stats_family_from_capability_key() -> None:
    msgs = [Message.with_parts(Role.USER, [ImagePart(_ref())])]
    unknown = {"count": 1, "bytes": 1000, "estimated_tokens": None}
    assert image_stats(msgs, provider=None, model="claude-sonnet-4-6") == unknown
    assert image_stats(msgs, provider=object(), model="claude-sonnet-4-6") == unknown
    assert image_stats(msgs, provider=_SurfaceProvider("CustomLLM"), model="m") == unknown
    assert image_stats([Message(role=Role.USER, content="x")], provider=None, model=None) is None
    for surface, model in (
        ("anthropic", "claude-sonnet-4-6"),
        ("openrouter", "anthropic/claude-haiku-4.5"),
    ):
        stats = image_stats(msgs, provider=_SurfaceProvider(surface), model=model)
        assert stats is not None and stats["estimated_tokens"] == 120
    stats = image_stats(msgs, provider=_SurfaceProvider("chat"), model="gpt-5-mini")
    assert stats is not None and stats["estimated_tokens"] == 110
    from dendrux.llm.anthropic import AnthropicProvider

    real = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
    stats = image_stats(msgs, provider=real, model="claude-sonnet-4-6")
    assert stats is not None and stats["estimated_tokens"] == 120


def test_image_dict_requires_ingested_images() -> None:
    assert image_dict(_ref()) == _ref().to_dict()
    with pytest.raises(TypeError):
        image_dict(Image.from_bytes(b"x"))


def test_edges_for_parts_and_scrub_edge_cases() -> None:
    ref = _ref()
    edges = edges_for_parts("-", "input", "input", [{"type": "text", "text": "a"}, ref.to_dict()])
    assert edges == [BlobEdge("-", ref.sha256, "input", "input", 1)]
    same = {"a": [1, "x"], "b": {"type": "base64", "data": "dendrux-image:sha256:" + "0" * 64}}
    assert scrub_inline_images(same) is same
    bad = scrub_inline_images({"type": "base64", "media_type": "image/png", "data": "!!!"})
    assert bad == {"type": "base64", "media_type": "image/png", "data": "dendrux-image:redacted"}


# ---------------------------------------------------------------------------
# Fix round (review-B)
# ---------------------------------------------------------------------------


class _PgError(Exception):
    def __init__(self, sqlstate: str) -> None:
        super().__init__(f"pg error {sqlstate}")
        self.sqlstate = sqlstate


async def test_retry_treats_deadlock_and_serialization_as_transient(monkeypatch) -> None:
    from sqlalchemy.exc import DBAPIError, IntegrityError

    import dendrux.runtime.durability as durability

    monkeypatch.setattr(durability, "_BASE_BACKOFF_S", 0.0)
    for code in ("40P01", "40001"):
        calls: list[int] = []

        async def flaky(code: str = code, calls: list[int] = calls) -> str:
            calls.append(1)
            if len(calls) == 1:
                raise DBAPIError("UPDATE dendrux_blobs", None, _PgError(code))
            return "ok"

        assert await durability.retry_transient_db(flaky, label="t", run_id="r") == "ok"
        assert len(calls) == 2
    for exc in (
        DBAPIError("x", None, _PgError("23505")),
        IntegrityError("x", None, _PgError("23503")),
    ):
        calls = []

        async def fails(exc: Exception = exc, calls: list[int] = calls) -> None:
            calls.append(1)
            raise exc

        with pytest.raises(PersistenceError):
            await durability.retry_transient_db(fails, label="t", run_id="r")
        assert len(calls) == 1
    assert durability.is_transient_db_error(DBAPIError("x", None, _PgError("40P01")))
    assert not durability.is_transient_db_error(ValueError("x"))


async def test_purge_evicts_resolver_cache() -> None:
    services = BlobServices.in_memory()
    part = await store_png(services, "-", make_png(30))
    assert await services.accessor.read(part.ref)
    resolver = services.resolver
    assert isinstance(resolver, CachingBlobResolver) and resolver.cached_bytes > 0
    assert await services.accessor.purge(part.ref.sha256)
    assert resolver.cached_bytes == 0


async def test_purge_external_without_store_records_intent_first() -> None:
    services = BlobServices.in_memory()
    part = await store_png(services, "-", make_png(31))
    s = part.ref.sha256
    bare = CatalogBlobAccessor(services.catalog)
    with pytest.raises(ImagePersistenceUnsupportedError) as info:
        await bare.purge(s, reason="erase")
    assert info.value.reason == "blob_store_not_configured"
    row = await services.catalog.get_row("-", s)  # type: ignore[union-attr]
    assert row is not None and row.state == "purging"  # bytes unreadable, re-ingest blocked
    with pytest.raises(InvalidImageError):
        await store_png(services, "-", make_png(31))
    # GC without a store leaves it alone; with the store it finishes the purge.
    assert (await bare.collect_garbage()).completed_purges == 0
    assert (await services.accessor.collect_garbage()).completed_purges == 1
    assert services.blob_store.inner.object_names() == []  # type: ignore[union-attr]
    assert await services.accessor.purge(s) is True


async def test_mark_purging_row_reports_storage() -> None:
    services = BlobServices.in_memory()
    part = await store_png(services, "-", make_png(32))
    memory = services.memory
    assert memory is not None
    marked = await memory.mark_purging_row("-", part.ref.sha256, None)
    assert marked is not None and marked[1] == "external"
    assert await memory.mark_purging_row("-", part.ref.sha256, None) is None


async def test_in_memory_owner_key_collision_is_typed() -> None:
    services = BlobServices.in_memory()
    memory = services.memory
    assert memory is not None
    a = (await store_png(services, "-", make_png(33))).ref.sha256
    b = (await store_png(services, "-", make_png(34))).ref.sha256
    await memory.add_blob_refs("R", "-", [BlobEdge("-", a, "trace", "0", 0)])
    await memory.add_blob_refs("R", "-", [BlobEdge("-", a, "trace", "0", 0)])  # retry: no-op
    with pytest.raises(BlobNotFoundError):
        await memory.add_blob_refs("R", "-", [BlobEdge("-", b, "trace", "0", 0)])
    with pytest.raises(ValueError):
        await memory.add_blob_refs(
            "R", "-", [BlobEdge("-", a, "trace", "1", 0), BlobEdge("-", b, "trace", "1", 0)]
        )
    assert [r.owner_id for r in await memory.blob_references("-", a)] == ["0"]
    assert await memory.blob_references("-", b) == []


def test_scrub_covers_all_image_shapes() -> None:
    png = make_png(40, pad=300)
    b64 = base64.b64encode(png).decode()
    marker = "dendrux-image:sha256:" + sha(png)
    request = {
        "a": f"data:image/png;base64,{b64}",  # plain image data URL
        "b": f"data:image/png;charset=utf-8;base64,{b64}",  # with parameters
        "c": f'see <img src="data:image/jpeg;base64,{b64}"> (inline)',  # embedded in text
        "d": {"type": "base64", "media_type": "image/png", "data": b64},  # Anthropic source
        "e": {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},  # Chat
        "f": {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},  # Responses
        # Image slots trust the slot, not the declared type.
        "g": {
            "type": "image_url",
            "image_url": {"url": f"data:application/octet-stream;base64,{b64}", "detail": "low"},
        },
        "h": "short text, not base64",
    }
    out = scrub_inline_images(request)
    assert out["a"] == out["b"] == marker
    assert out["c"] == f'see <img src="{marker}"> (inline)'
    assert out["d"]["data"] == marker and out["d"]["media_type"] == "image/png"
    assert out["e"] == {"type": "image_url", "image_url": {"url": marker}}
    assert out["f"] == {"type": "input_image", "image_url": marker}
    assert out["g"]["image_url"] == {"url": marker, "detail": "low"}
    assert out["h"] is request["h"]
    dumped = json.dumps(out)
    assert b64[:40] not in dumped and "data:image" not in dumped
    tiny = "data:image/gif;base64,R0lGODlhAQABAAAAACw="
    assert scrub_inline_images(tiny).startswith("dendrux-image:")
    https = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    assert scrub_inline_images(https) is https


def test_scrub_keeps_forensic_payloads_byte_faithful() -> None:
    """Only image structures are scrubbed: reasoning blobs, signatures and tokens survive."""
    png = make_png(41, pad=300)
    b64 = base64.b64encode(png).decode()
    opaque = base64.b64encode(bytes(range(256)) * 3).decode()  # 1024 base64 chars
    token = "tok_" + "A1b2C3d4" * 37  # 300 chars, base64-like
    request = {
        "input": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": opaque, "summary": []},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": opaque},
                    {"type": "redacted_thinking", "data": opaque},
                ],
            },
            {
                "role": "assistant",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": opaque, "format": "anthropic-claude-v1"}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": f"session token: {token}"},
            {"type": "base64", "media_type": "application/pdf", "data": opaque},
            {"note": f"data:application/octet-stream;base64,{opaque}"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
        ]
    }
    snapshot = json.dumps(request)
    out = scrub_inline_images(request)
    assert json.dumps(request) == snapshot  # caller object untouched
    items = out["input"]
    assert items[0] is request["input"][0]
    assert items[1] is request["input"][1]
    assert items[2] is request["input"][2]
    assert items[3] is request["input"][3]
    assert items[4] is request["input"][4]
    assert items[5] is request["input"][5]
    assert items[0]["encrypted_content"] == opaque
    assert items[1]["content"][0]["signature"] == opaque
    assert items[2]["reasoning_details"][0]["data"] == opaque
    assert token in items[3]["content"]
    assert items[6]["image_url"] == "dendrux-image:sha256:" + sha(png)


async def test_in_memory_trace_edges_skip_non_live() -> None:
    services = BlobServices.in_memory()
    memory = services.memory
    assert memory is not None
    s = (await store_png(services, "-", make_png(41))).ref.sha256
    assert await services.accessor.purge(s)
    await memory.add_blob_refs("R", "-", [BlobEdge("-", s, "trace", "0", 0)])
    assert await memory.blob_references("-", s) == []
    with pytest.raises(InvalidImageError):
        await memory.add_blob_refs("R", "-", [BlobEdge("-", s, "seed", "seed", 0)])
