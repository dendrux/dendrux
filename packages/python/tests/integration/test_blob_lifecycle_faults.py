"""Blob lifecycle fault injection and races (§5.12), on SQLite and Postgres.

Crashes are simulated with a ``BaseException`` raised between two steps (GC
only swallows ``Exception``), then a **fresh** store instance runs
``collect_garbage`` with no re-ingest unless the test says so. Races are
injected deterministically by wrapping one catalog / object-store call.
Every test ends with the invariant: no stored object outlives a
``pending|live|purging|deleting`` row naming its generation.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import text

from dendrux.errors import BlobNotFoundError, InvalidImageError
from dendrux.media import BlobEdge, BlobKey, InMemoryBlobStore
from dendrux.runtime._blob_catalog import BlobServices, write_blob_refs
from dendrux.runtime.state import SQLAlchemyStateStore
from tests.unit.test_blob_catalog import make_png, store_png

HOUR = timedelta(hours=1)
CONCURRENT = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


class Crash(BaseException):
    """Simulated process death between two lifecycle steps."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def services_for(engine: Any, blob_store: InMemoryBlobStore) -> BlobServices:
    return BlobServices.for_state_store(SQLAlchemyStateStore(engine), blob_store=blob_store)


def catalog(services: BlobServices) -> Any:
    assert services.catalog is not None
    return services.catalog


async def age(engine: Any, scope: str, sha256: str, seconds: int) -> None:
    async with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            await conn.execute(
                text(
                    "UPDATE dendrux_blobs SET last_used_at = now() - make_interval(secs => :s),"
                    " state_changed_at = now() - make_interval(secs => :s)"
                    " WHERE tenant_scope = :scope AND sha256 = :sha"
                ),
                {"s": seconds, "scope": scope, "sha": sha256},
            )
        else:
            from datetime import UTC, datetime

            old = (datetime.now(UTC) - timedelta(seconds=seconds)).replace(tzinfo=None)
            await conn.execute(
                text(
                    "UPDATE dendrux_blobs SET last_used_at = :t, state_changed_at = :t"
                    " WHERE tenant_scope = :scope AND sha256 = :sha"
                ),
                {"t": old.isoformat(sep=" "), "scope": scope, "sha": sha256},
            )


async def blob_rows(engine: Any) -> list[tuple[str, str, str, str]]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT tenant_scope, sha256, generation, state FROM dendrux_blobs")
        )
        return [tuple(r) for r in result.all()]  # type: ignore[misc]


async def assert_no_orphan_objects(engine: Any, blob_store: InMemoryBlobStore) -> None:
    named = {
        BlobKey(scope, sha, gen).object_name
        for scope, sha, gen, state in await blob_rows(engine)
        if state in ("pending", "live", "purging", "deleting")
    }
    assert set(blob_store.object_names()) <= named


def wrap(obj: Any, name: str, before: Any = None, after: Any = None) -> None:
    """Wrap ``obj.name`` with async hooks run before / after the real call."""
    real = getattr(obj, name)

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if before is not None:
            await before(*args, **kwargs)
        result = await real(*args, **kwargs)
        if after is not None:
            await after(result, *args, **kwargs)
        return result

    setattr(obj, name, wrapper)


def crash_after(obj: Any, name: str) -> None:
    async def boom(*_: Any, **__: Any) -> None:
        raise Crash

    wrap(obj, name, after=boom)


def crash_before(obj: Any, name: str) -> None:
    async def boom(*_: Any, **__: Any) -> None:
        raise Crash

    wrap(obj, name, before=boom)


async def live_external(
    engine: Any, seed: int
) -> tuple[InMemoryBlobStore, BlobServices, str, bytes]:
    blob_store = InMemoryBlobStore()
    services = services_for(engine, blob_store)
    data = make_png(seed)
    part = await store_png(services, "-", data)
    return blob_store, services, part.ref.sha256, data


async def row_of(engine: Any, sha: str) -> tuple[str, str, str, str] | None:
    return next((r for r in await blob_rows(engine) if r[1] == sha), None)


# ---------------------------------------------------------------------------
# Purge crash windows (N-4)
# ---------------------------------------------------------------------------


async def test_crash_after_mark_purging_from_live(engine) -> None:
    blob_store, services, s, _ = await live_external(engine, 1)
    crash_before(blob_store, "delete")
    with pytest.raises(Crash):
        await services.accessor.purge(s)
    blob_store2 = blob_store
    blob_store2.delete = InMemoryBlobStore.delete.__get__(blob_store2)  # type: ignore[method-assign]
    assert (await row_of(engine, s))[3] == "purging"  # type: ignore[index]
    result = await services_for(engine, blob_store2).accessor.collect_garbage(older_than=HOUR)
    assert result.completed_purges == 1
    assert (await row_of(engine, s))[3] == "purged"  # type: ignore[index]
    assert blob_store.object_names() == []


async def test_crash_after_mark_purging_from_pending(engine) -> None:
    blob_store = InMemoryBlobStore()
    services = services_for(engine, blob_store)
    crash_before(catalog(services), "promote")  # ingest dies between put and promote
    data = make_png(2)
    with pytest.raises(Crash):
        await store_png(services, "-", data)
    s = (await blob_rows(engine))[0][1]
    assert (await row_of(engine, s))[3] == "pending"  # type: ignore[index]
    assert len(blob_store.object_names()) == 1
    purger = services_for(engine, blob_store)
    crash_after(catalog(purger), "mark_purging")
    with pytest.raises(Crash):
        await purger.accessor.purge(s)
    result = await services_for(engine, blob_store).accessor.collect_garbage(older_than=HOUR)
    assert result.completed_purges == 1
    assert (await row_of(engine, s))[3] == "purged"  # type: ignore[index]
    assert blob_store.object_names() == []
    with pytest.raises(InvalidImageError) as info:
        await store_png(services_for(engine, blob_store), "-", data)
    assert info.value.reason == "purged"
    assert blob_store.object_names() == []


async def test_crash_after_mark_purging_from_deleting(engine) -> None:
    blob_store, services, s, data = await live_external(engine, 3)
    await age(engine, "-", s, 7200)
    gc = services_for(engine, blob_store)
    crash_after(catalog(gc), "mark_deleting")
    with pytest.raises(Crash):
        await gc.accessor.collect_garbage(older_than=HOUR)
    assert (await row_of(engine, s))[3] == "deleting"  # type: ignore[index]
    purger = services_for(engine, blob_store)
    crash_after(catalog(purger), "mark_purging")
    with pytest.raises(Crash):
        await purger.accessor.purge(s)
    await services_for(engine, blob_store).accessor.collect_garbage(older_than=HOUR)
    assert (await row_of(engine, s))[3] == "purged"  # type: ignore[index]
    assert blob_store.object_names() == []
    # Purge of a deleting row then re-ingest: erasure wins.
    with pytest.raises(InvalidImageError):
        await store_png(services_for(engine, blob_store), "-", data)


async def test_crash_after_purge_object_delete(engine) -> None:
    blob_store, services, s, _ = await live_external(engine, 4)
    crash_before(catalog(services), "finish_purge")
    with pytest.raises(Crash):
        await services.accessor.purge(s, reason="gdpr")
    assert blob_store.object_names() == []
    assert (await row_of(engine, s))[3] == "purging"  # type: ignore[index]
    result = await services_for(engine, blob_store).accessor.collect_garbage(older_than=HOUR)
    assert result.completed_purges == 1
    info = await services_for(engine, blob_store).accessor.info(s)
    assert info is not None and info.state == "purged" and info.purge_reason == "gdpr"


# ---------------------------------------------------------------------------
# GC crash windows
# ---------------------------------------------------------------------------


async def test_crash_after_mark_deleting(engine) -> None:
    blob_store, services, s, _ = await live_external(engine, 5)
    await age(engine, "-", s, 7200)
    crash_after(catalog(services), "mark_deleting")
    with pytest.raises(Crash):
        await services.accessor.collect_garbage(older_than=HOUR)
    assert len(blob_store.object_names()) == 1
    result = await services_for(engine, blob_store).accessor.collect_garbage(older_than=HOUR)
    assert result.retried_deletes == 1 and result.deleted_blobs == 1
    assert await blob_rows(engine) == [] and blob_store.object_names() == []


async def test_crash_after_gc_object_delete(engine) -> None:
    blob_store, services, s, _ = await live_external(engine, 6)
    await age(engine, "-", s, 7200)
    crash_before(catalog(services), "delete_row")
    with pytest.raises(Crash):
        await services.accessor.collect_garbage(older_than=HOUR)
    assert blob_store.object_names() == []
    assert (await row_of(engine, s))[3] == "deleting"  # type: ignore[index]
    await services_for(engine, blob_store).accessor.collect_garbage(older_than=HOUR)
    assert await blob_rows(engine) == []


@pytest.mark.parametrize("when", ["before_put", "before_promote"])
async def test_crash_during_external_create_collected_after_grace(engine, when) -> None:
    blob_store = InMemoryBlobStore()
    services = services_for(engine, blob_store)
    if when == "before_put":
        crash_before(blob_store, "put")
    else:
        crash_before(catalog(services), "promote")
    with pytest.raises(Crash):
        await store_png(services, "-", make_png(7))
    blob_store.put = InMemoryBlobStore.put.__get__(blob_store)  # type: ignore[method-assign]
    ((_, s, _, state),) = await blob_rows(engine)
    assert state == "pending"
    fresh = services_for(engine, blob_store)
    assert (await fresh.accessor.collect_garbage(older_than=HOUR)).deleted_blobs == 0  # in grace
    await age(engine, "-", s, 7200)
    result = await services_for(engine, blob_store).accessor.collect_garbage(older_than=HOUR)
    assert result.deleted_blobs == 1
    assert await blob_rows(engine) == [] and blob_store.object_names() == []


async def test_crashed_create_then_reingest_adopts_new_generation(engine) -> None:
    blob_store = InMemoryBlobStore()
    services = services_for(engine, blob_store)
    crash_before(catalog(services), "promote")
    data = make_png(8)
    with pytest.raises(Crash):
        await store_png(services, "-", data)
    ((_, s, g1, _),) = await blob_rows(engine)
    await age(engine, "-", s, 120)  # older than the 60 s in-flight window
    part = await store_png(services_for(engine, blob_store), "-", data)
    ((_, _, g2, state),) = await blob_rows(engine)
    assert state == "live" and g2 != g1
    assert blob_store.object_names() == [BlobKey("-", s, g2).object_name]
    assert await services_for(engine, blob_store).accessor.read(part.ref) == data


# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------


async def test_two_concurrent_ingests_one_row_one_object(engine) -> None:
    blob_store = InMemoryBlobStore()
    a, b = services_for(engine, blob_store), services_for(engine, blob_store)
    data = make_png(9)
    pa, pb = await asyncio.gather(store_png(a, "-", data), store_png(b, "-", data))
    assert pa.ref == pb.ref
    ((_, s, g, state),) = await blob_rows(engine)
    assert state == "live" and blob_store.object_names() == [BlobKey("-", s, g).object_name]


async def test_in_flight_pending_is_finished_by_second_ingest(engine) -> None:
    blob_store = InMemoryBlobStore()
    first = services_for(engine, blob_store)
    crash_before(catalog(first), "promote")
    data = make_png(10)
    with pytest.raises(Crash):
        await store_png(first, "-", data)
    ((_, s, g, _),) = await blob_rows(engine)  # fresh pending: same key, same bytes
    await store_png(services_for(engine, blob_store), "-", data)
    assert await blob_rows(engine) == [("-", s, g, "live")]
    assert blob_store.object_names() == [BlobKey("-", s, g).object_name]


async def test_stale_pending_adoption_racing_gc(engine) -> None:
    """N-3: GC marks the stale pending row while an ingest is adopting it."""
    blob_store = InMemoryBlobStore()
    first = services_for(engine, blob_store)
    crash_before(catalog(first), "promote")
    data = make_png(11)
    with pytest.raises(Crash):
        await store_png(first, "-", data)
    ((_, s, g1, _),) = await blob_rows(engine)
    await age(engine, "-", s, 7200)
    gc = services_for(engine, blob_store)
    gc_cat = catalog(gc)
    marked: list[str | None] = []

    async def gc_marks_between_delete_and_adopt(key: BlobKey) -> None:
        if not marked and key.generation == g1:
            now = await gc_cat.current_time()
            marked.append(
                await gc_cat.mark_deleting("-", s, now - HOUR, stale_pending_cutoff=now - HOUR)
            )

    ingester = services_for(engine, blob_store)
    wrap(blob_store, "delete", before=gc_marks_between_delete_and_adopt)
    part = await store_png(ingester, "-", data)
    assert marked == [g1]
    # GC resumes after the ingest: its object delete and delete_row target g1 only.
    await blob_store.delete(BlobKey("-", s, g1))
    assert await gc_cat.delete_row("-", s, g1) is False
    ((_, _, g3, state),) = await blob_rows(engine)
    assert state == "live" and g3 != g1
    assert blob_store.object_names() == [BlobKey("-", s, g3).object_name]
    assert await ingester.accessor.read(part.ref) == data


async def test_purge_between_put_and_promote_wins(engine) -> None:
    """N-4: the in-flight ingest raises purged and removes the object it wrote."""
    blob_store = InMemoryBlobStore()
    services = services_for(engine, blob_store)
    purger = services_for(engine, blob_store)
    data = make_png(12)

    async def purge_first(scope: str, sha256: str, generation: str) -> None:
        assert await purger.accessor.purge(sha256)

    wrap(catalog(services), "promote", before=purge_first)
    with pytest.raises(InvalidImageError) as info:
        await store_png(services, "-", data)
    assert info.value.reason == "purged"
    ((_, _, _, state),) = await blob_rows(engine)
    assert state == "purged" and blob_store.object_names() == []


async def test_ingest_revives_while_gc_paused_before_delete_row(engine) -> None:
    blob_store, services, s, data = await live_external(engine, 13)
    ((_, _, g1, _),) = await blob_rows(engine)
    await age(engine, "-", s, 7200)
    gc = services_for(engine, blob_store)
    ingester = services_for(engine, blob_store)
    revived: list[bool] = []

    async def ingest_first(scope: str, sha256: str, generation: str) -> None:
        if not revived:
            revived.append(True)
            await store_png(ingester, "-", data)

    wrap(catalog(gc), "delete_row", before=ingest_first)
    result = await gc.accessor.collect_garbage(older_than=HOUR)
    assert result.deleted_blobs == 0
    ((_, _, g2, state),) = await blob_rows(engine)
    assert state == "live" and g2 != g1
    assert blob_store.object_names() == [BlobKey("-", s, g2).object_name]


async def test_gc_crash_after_mark_then_ingest_revives(engine) -> None:
    blob_store, services, s, data = await live_external(engine, 14)
    ((_, _, g1, _),) = await blob_rows(engine)
    await age(engine, "-", s, 7200)
    crash_after(catalog(services), "mark_deleting")
    with pytest.raises(Crash):
        await services.accessor.collect_garbage(older_than=HOUR)
    part = await store_png(services_for(engine, blob_store), "-", data)
    ((_, _, g2, state),) = await blob_rows(engine)
    assert state == "live" and g2 != g1
    assert ("delete", BlobKey("-", s, g1).object_name) in blob_store.calls
    assert blob_store.object_names() == [BlobKey("-", s, g2).object_name]
    await services_for(engine, blob_store).accessor.collect_garbage(older_than=HOUR)
    assert await services_for(engine, blob_store).accessor.read(part.ref) == data


async def test_db_storage_revive_and_gc(engine, store) -> None:
    services = BlobServices.for_state_store(store)
    data = make_png(15)
    part = await store_png(services, "-", data)
    await age(engine, "-", part.ref.sha256, 7200)
    crash_after(catalog(services), "mark_deleting")
    with pytest.raises(Crash):
        await services.accessor.collect_garbage(older_than=HOUR)
    fresh = BlobServices.for_state_store(SQLAlchemyStateStore(engine))
    assert await store_png(fresh, "-", data) == part  # revived straight to live with data
    assert await fresh.accessor.read(part.ref) == data


# ---------------------------------------------------------------------------
# Edge transaction vs GC (N-1)
# ---------------------------------------------------------------------------


@CONCURRENT
async def test_edge_claim_blocks_concurrent_gc_mark(engine) -> None:
    blob_store, services, s, _ = await live_external(engine, 16)
    store = SQLAlchemyStateStore(engine)
    await store.create_run("R1", "a")
    await age(engine, "-", s, 7200)
    gc_cat = catalog(services_for(engine, blob_store))
    now = await gc_cat.current_time()

    async with store._session_factory() as session:
        await write_blob_refs(
            session,
            run_id="R1",
            blob_refs=[BlobEdge("-", s, "seed", "seed", 0)],
            blob_refs_scope="-",
        )
        task = asyncio.create_task(
            gc_cat.mark_deleting("-", s, now - HOUR, stale_pending_cutoff=now - HOUR)
        )
        await asyncio.sleep(0.3)
        assert not task.done()  # waits on the claimed row / write lock
        await session.commit()
    assert await task is None
    ((_, _, _, state),) = await blob_rows(engine)
    assert state == "live"
    assert len(await services.accessor.references(s)) == 1


async def test_gc_marked_first_edge_write_fails_typed(engine, store) -> None:
    blob_store, services, s, _ = await live_external(engine, 17)
    await store.create_run("R1", "a")
    await age(engine, "-", s, 7200)
    cat = catalog(services)
    now = await cat.current_time()
    assert await cat.mark_deleting("-", s, now - HOUR, stale_pending_cutoff=now - HOUR)
    with pytest.raises(BlobNotFoundError) as info:
        await store.save_trace(
            "R1",
            "user",
            "x",
            order_index=0,
            blob_refs=[BlobEdge("-", s, "trace", "0", 0)],
            blob_refs_scope="-",
        )
    assert info.value.__cause__ is None and info.value.__context__ is None
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM react_traces"))).scalar_one() == 0


async def test_lost_race_at_recheck_reverts_to_live(engine) -> None:
    blob_store, services, s, data = await live_external(engine, 18)
    await age(engine, "-", s, 7200)
    cat = catalog(services)
    real = cat.has_edges
    calls: list[int] = []

    async def first_true(scope: str, sha256: str) -> bool:
        calls.append(1)
        return True if len(calls) == 1 else await real(scope, sha256)

    cat.has_edges = first_true
    result = await services.accessor.collect_garbage(older_than=HOUR)
    assert result.skipped_referenced == 1 and result.deleted_blobs == 0
    ((_, _, _, state),) = await blob_rows(engine)
    assert state == "live" and len(blob_store.object_names()) == 1
    assert await services_for(engine, blob_store).accessor.read(s) == data


async def test_lost_race_at_delete_row_leaves_row_for_revive(engine, caplog) -> None:
    blob_store, services, s, data = await live_external(engine, 19)
    store = SQLAlchemyStateStore(engine)
    await store.create_run("R1", "a")
    await age(engine, "-", s, 7200)

    async def edge_slips_in(key: BlobKey) -> None:
        async with engine.begin() as conn:  # bypasses the claim: an injected race
            await conn.execute(
                text(
                    "INSERT INTO dendrux_blob_refs (id, tenant_scope, sha256, run_id, owner_kind,"
                    " owner_id, ordinal, created_at)"
                    " VALUES ('E1', '-', :sha, 'R1', 'seed', 'seed', 0, CURRENT_TIMESTAMP)"
                ),
                {"sha": s},
            )

    wrap(blob_store, "delete", after=lambda *_a, **_k: edge_slips_in(None))  # type: ignore[arg-type]
    with caplog.at_level("ERROR"):
        result = await services.accessor.collect_garbage(older_than=HOUR)
    assert result.skipped_referenced == 1 and result.deleted_blobs == 0
    assert "img_" + s[:12] in caplog.text and "iVBORw0K" not in caplog.text
    ((_, _, _, state),) = await blob_rows(engine)
    assert state == "deleting"
    second = await services_for(engine, InMemoryBlobStore()).accessor.collect_garbage(
        older_than=HOUR
    )
    assert second.skipped_referenced == 1 and second.deleted_blobs == 0
    # The referencing run sees a typed not-found until the bytes are ingested again.
    with pytest.raises(BlobNotFoundError):
        await services.accessor.read(s)
    fresh_store = InMemoryBlobStore()
    await store_png(services_for(engine, fresh_store), "-", data)
    assert await services_for(engine, fresh_store).accessor.read(s) == data
