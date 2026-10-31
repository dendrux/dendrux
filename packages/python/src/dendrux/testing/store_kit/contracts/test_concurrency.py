"""Concurrency: two live instances over the same data, many callers, one outcome.

Each test starts N callers together through a barrier, spread over the store the factory made
and an independent second instance. The database, not process memory, must decide: exactly one
winner for a compare-and-swap, no lost update, no duplicated or skipped sequence number, and a
cap that is never exceeded.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.errors import NoteLimitExceededError, PersistenceError
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import (
    EVENT,
    complete_call_kwargs,
    start_call_kwargs,
    tool_call_kwargs,
    ulid,
)
from dendrux.types import compute_idempotency_fingerprint

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.testing.store_kit._kit import Kit

concurrent = contract("concurrency", uses=("second",))
CALLERS = 8


async def race(calls: list[Callable[..., Awaitable[Any]]]) -> list[Any]:
    """Run all calls at once; exceptions come back as values (never swallowed by gather)."""
    barrier = asyncio.Barrier(len(calls))

    async def gated(call: Callable[..., Awaitable[Any]]) -> Any:
        await barrier.wait()
        return await call()

    results = await asyncio.gather(*(gated(c) for c in calls), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result
    return list(results)


async def stores_of(kit: Kit) -> list[Any]:
    return [kit.store, await kit.second()]


def unexpected(results: list[Any], allowed: tuple[type[BaseException], ...] = ()) -> list[Any]:
    return [r for r in results if isinstance(r, BaseException) and not isinstance(r, allowed)]


@concurrent
async def test_parallel_claims_have_exactly_one_winner(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.paused_run()
    results = await race(
        [
            (
                lambda s=stores[i % 2]: s.claim_paused_run(
                    run_id, expected_status="waiting_client_tool"
                )
            )
            for i in range(CALLERS)
        ]
    )
    assert unexpected(results) == []
    winners = [r for r in results if r is not False]
    assert len(winners) == 1, f"exactly one claim may win, got {len(winners)}"
    record = await stores[0].get_run(run_id)
    assert record is not None and record.status == "running"
    if dendrux_kit.declaration.has("fenced_writes"):
        assert record.claim_epoch == 1, "one claim, one epoch step"


@concurrent
async def test_parallel_submit_and_claim_has_exactly_one_winner(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.paused_run()

    def submit(store: Any, index: int) -> Callable[[], Awaitable[Any]]:
        envelope = {
            "submission_id": ulid(),
            "kind": "tool_results",
            "pause_id": "kit-pause",
            "call_ids": ["c1"],
        }
        return lambda: store.submit_and_claim(
            run_id,
            expected_status="waiting_client_tool",
            submitted_data={"submitted_tool_results": [{"call_id": "c1", "payload": str(index)}]},
            submission=envelope,
        )

    results = await race([submit(stores[i % 2], i) for i in range(CALLERS)])
    assert unexpected(results) == []
    assert len([r for r in results if r is not False]) == 1


@concurrent
async def test_parallel_claims_of_different_runs_all_win(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    runs = [await dendrux_kit.paused_run() for _ in range(CALLERS)]
    results = await race(
        [
            (
                lambda s=stores[i % 2], r=run: s.claim_paused_run(
                    r, expected_status="waiting_client_tool"
                )
            )
            for i, run in enumerate(runs)
        ]
    )
    assert unexpected(results) == []
    assert all(r is not False for r in results), "no false conflict between independent runs"


@concurrent
async def test_parallel_finalizes_have_exactly_one_winner_and_no_lost_update(
    dendrux_kit: Kit,
) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.new_run()
    results = await race(
        [
            (
                lambda s=stores[i % 2], i=i: s.finalize_run(
                    run_id,
                    status="success",
                    answer=f"answer {i}",
                    expected_current_status="running",
                )
            )
            for i in range(CALLERS)
        ]
    )
    assert unexpected(results) == []
    winners = [i for i, r in enumerate(results) if r is True]
    assert len(winners) == 1
    record = await stores[0].get_run(run_id)
    assert record is not None
    assert (record.status, record.answer) == ("success", f"answer {winners[0]}")


@concurrent
async def test_a_cancel_racing_a_pause_leaves_one_consistent_outcome(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.new_run()
    cancel, pause = await race(
        [
            lambda: stores[0].request_cancel(run_id),
            lambda: stores[1].try_pause_run(
                run_id, status="waiting_client_tool", pause_data={"pause_id": "p"}
            ),
        ]
    )
    assert unexpected([cancel, pause]) == []
    record = await stores[0].get_run(run_id)
    assert record is not None
    if pause:
        assert record.status == "waiting_client_tool"
    else:
        assert record.status == "running" and record.cancel_requested, "a lost pause means a cancel"
    assert cancel is True and record.cancel_requested, "the cancel flag is never lost"


@concurrent
async def test_parallel_events_get_distinct_gapless_sequences(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.new_run()
    count = CALLERS * 3
    results = await race(
        [
            (lambda s=stores[i % 2], i=i: s.save_run_event(run_id, event_type=EVENT, data={"i": i}))
            for i in range(count)
        ]
    )
    assert unexpected(results) == []
    assert sorted(results) == list(range(count)), "every sequence once, none skipped"
    stored = await stores[0].get_run_events(run_id)
    assert [e.sequence_index for e in stored] == list(range(count))
    assert {e.data["i"] for e in stored} == set(range(count))


@concurrent
async def test_parallel_notes_never_exceed_the_cap(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.new_run()
    cap = 3
    results = await race(
        [
            (
                lambda s=stores[i % 2], i=i: s.add_note(
                    run_id, note_id=ulid(), text=str(i), max_notes=cap
                )
            )
            for i in range(CALLERS * 2)
        ]
    )
    assert unexpected(results, (NoteLimitExceededError,)) == []
    added = [r for r in results if r is True]
    refused = [r for r in results if isinstance(r, NoteLimitExceededError)]
    assert len(added) == cap
    assert len(refused) == len(results) - cap
    assert len(await stores[0].get_notes(run_id)) == cap


@concurrent
async def test_a_note_racing_a_finalize_is_stored_or_refused_never_lost(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    for _ in range(4):
        run_id = await dendrux_kit.new_run()
        added, finalized = await race(
            [
                lambda r=run_id: stores[0].add_note(r, note_id=ulid(), text="n"),
                lambda r=run_id: stores[1].finalize_run(r, status="success", answer="a"),
            ]
        )
        assert unexpected([added, finalized]) == []
        assert finalized is True
        notes = await stores[0].get_notes(run_id)
        assert len(notes) == (1 if added else 0), "a stored note is reported, a refused one absent"


@concurrent
async def test_parallel_creates_with_one_idempotency_key_make_one_run(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    fingerprint = compute_idempotency_fingerprint("KitAgent", "x")
    results = await race(
        [
            (
                lambda s=stores[i % 2]: s.create_run(
                    ulid(), "KitAgent", idempotency_key="race", idempotency_fingerprint=fingerprint
                )
            )
            for i in range(CALLERS)
        ]
    )
    assert unexpected(results) == []
    assert sorted(r.outcome for r in results).count("created") == 1
    assert len({r.run_id for r in results}) == 1, "every caller sees the one run"
    assert await stores[0].count_runs() == 1


@concurrent
async def test_parallel_tool_calls_with_one_identity_store_one_row(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.new_run()
    kwargs = tool_call_kwargs(tool_call_id="same")
    results = await race(
        [
            (lambda s=stores[i % 2]: s.save_tool_call(run_id, **kwargs, **_row_id(dendrux_kit)))
            for i in range(CALLERS)
        ]
    )
    assert unexpected(results, (PersistenceError,)) == []
    assert len(await stores[0].get_tool_calls(run_id)) == 1


def _row_id(kit: Kit) -> dict[str, str]:
    return {"row_id": ulid()} if kit.declaration.has("retry_ids") else {}


@pytest.mark.parametrize("order", [0, 1])
@contract("concurrency", needs=("retry_ids",), uses=("second",))
async def test_parallel_traces_for_one_position_have_one_winner(
    dendrux_kit: Kit, order: int
) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.new_run()
    results = await race(
        [
            (
                lambda s=stores[i % 2], i=i: s.save_trace(
                    run_id, "user", f"writer {i}", order_index=order, trace_id=ulid()
                )
            )
            for i in range(CALLERS)
        ]
    )
    assert unexpected(results, (PersistenceError,)) == []
    assert len([r for r in results if isinstance(r, str)]) == 1
    assert all(r.code == "write_conflict" for r in results if isinstance(r, PersistenceError))
    assert len(await stores[0].get_traces(run_id)) == 1


@contract("concurrency", needs=("atomic_completion",), uses=("second",), format2=True)
async def test_parallel_completions_patch_the_pending_set_without_loss(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    run_id = await dendrux_kit.new_run()
    ids = ["a", "b", "c", "d"]
    await stores[0].checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "dispatch", "pending": [{"call_id": c} for c in ids]},
        pii_mapping=None,
        trace=None,
    )
    for call_id in ids:
        await stores[0].start_tool_call(run_id, claim_epoch=0, **start_call_kwargs(call_id))
    results = await race(
        [
            (
                lambda s=stores[i % 2], c=call_id, o=i + 1: s.complete_tool_call(
                    run_id, claim_epoch=0, **complete_call_kwargs(c, o)
                )
            )
            for i, call_id in enumerate(ids)
        ]
    )
    assert unexpected(results) == []
    state = await stores[0].get_loop_state(run_id)
    assert state is not None and state["pending"] == [], "every completion removed its own call"
    assert len(await stores[0].get_traces(run_id)) == len(ids)
    assert len(await stores[0].get_run_events(run_id)) == 2 * len(ids)


@contract("concurrency", needs=("recovery",), uses=("second", "clock"), format2=True)
async def test_two_markers_find_each_stale_run_once(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    runs = []
    for _ in range(3):
        run_id = await dendrux_kit.new_run(tenant_id="t")
        await stores[0].save_run_event(run_id, event_type="run.started")
        await stores[0].checkpoint_step(
            run_id,
            claim_epoch=0,
            loop_state={"v": 1, "loop": "react", "phase": "dispatch"},
            pii_mapping=None,
            trace=None,
        )
        runs.append(run_id)
    threshold = await dendrux_kit.age()
    cutoff = datetime.now(UTC) - threshold
    options = {
        "cutoff": cutoff,
        "batch_size": 10,
        "max_runs": 100,
        "max_returned": 10,
        "reclaim_limit": 3,
        "cursor": None,
    }
    results = await race([lambda s=s: s.mark_lease_expired(**options) for s in stores])
    assert unexpected(results) == []
    for run_id in runs:
        epochs = {
            r.claim_epoch for result in results for r in result.reclaimable if r.run_id == run_id
        }
        assert len(epochs) == 1, "both markers must agree on the epoch they observed"
        assert (await stores[0].get_run(run_id)).claim_epoch == epochs.pop()


def _blob_info(sha: str, generation: str) -> Any:
    from dendrux.media._store import BlobInfo

    now = datetime.now(UTC)
    return BlobInfo(
        sha256=sha,
        media_type="image/png",
        byte_length=4,
        width=1,
        height=1,
        frame_count=1,
        storage="db",
        state="live",
        generation=generation,
        created_at=now,
        last_used_at=now,
    )


def _needs_blob_catalog(store: Any) -> None:
    from dendrux.media._catalog import BlobCatalog

    if not isinstance(store, BlobCatalog):
        pytest.skip("the store has no blob catalog (images are not supported)")


@concurrent
async def test_parallel_blob_inserts_of_one_content_make_one_row(dendrux_kit: Kit) -> None:
    stores = await stores_of(dendrux_kit)
    _needs_blob_catalog(stores[0])
    scope, sha = "kit-scope", "a" * 64

    def insert(store: Any) -> Callable[[], Awaitable[Any]]:
        generation = ulid()
        return lambda: store.insert_row(
            scope,
            sha,
            _blob_info(sha, generation),
            state="live",
            generation=generation,
            data=b"\x89PNG",
        )

    results = await race([insert(stores[i % 2]) for i in range(CALLERS)])
    assert unexpected(results) == []
    assert results.count(True) == 1, "exactly one insert may create the row"
    row = await stores[0].get_row(scope, sha)
    assert row is not None and row.state == "live"


@concurrent
async def test_parallel_references_to_one_blob_are_all_recorded(dendrux_kit: Kit) -> None:
    from dendrux.media._catalog import BlobEdge
    from dendrux.media._runtime import tenant_scope_for

    stores = await stores_of(dendrux_kit)
    _needs_blob_catalog(stores[0])
    scope, sha = tenant_scope_for("blobs"), "b" * 64
    generation = ulid()
    assert await stores[0].insert_row(
        scope,
        sha,
        _blob_info(sha, generation),
        state="live",
        generation=generation,
        data=b"\x89PNG",
    )
    runs = [await dendrux_kit.new_run(tenant_id="blobs") for _ in range(CALLERS)]
    traces = [ulid() for _ in runs]
    results = await race(
        [
            (
                lambda s=stores[i % 2], r=run, t=trace: s.save_trace(
                    r,
                    "user",
                    "image",
                    order_index=0,
                    **({"trace_id": t} if dendrux_kit.declaration.has("retry_ids") else {}),
                    blob_refs=[BlobEdge(scope, sha, "trace", t, 0)],
                    blob_refs_scope=scope,
                )
            )
            for i, (run, trace) in enumerate(zip(runs, traces, strict=True))
        ]
    )
    assert unexpected(results) == []
    refs = await stores[0].blob_references(scope, sha)
    assert len(refs) == CALLERS, "no reference may be lost to a race on the shared blob"
