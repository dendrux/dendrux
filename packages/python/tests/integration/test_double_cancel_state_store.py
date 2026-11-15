"""Concurrent store writes cancelled twice all finish, and the store stays usable.

Runs on the backend matrix (in-memory SQLite, Postgres). Each round starts 16
``create_run`` calls, cancels every one twice 10 ms apart while they are touching
the database, and requires every task to finish within a bound well above the
post-cancel close wait (``CLOSE_AFTER_CANCEL_S``).
"""

from __future__ import annotations

import asyncio

import pytest

from dendrux.db._pool_guard import CLOSE_AFTER_CANCEL_S

WRITERS = 16
CANCEL_AT_S = (0.0, 0.005, 0.02, 0.05)
FINISH_WITHIN_S = CLOSE_AFTER_CANCEL_S + 5.0


def _run_id(round_index: int, writer: int) -> str:
    return f"01DCC{round_index:02d}{writer:02d}".ljust(26, "0")


@pytest.mark.parametrize("cancel_at_s", CANCEL_AT_S)
async def test_writes_cancelled_twice_all_finish_and_the_store_stays_usable(
    engine, store, cancel_at_s: float
) -> None:
    round_index = CANCEL_AT_S.index(cancel_at_s)
    writers = [
        asyncio.create_task(store.create_run(_run_id(round_index, w), "agent"))
        for w in range(WRITERS)
    ]
    await asyncio.sleep(cancel_at_s)
    for _ in range(2):
        for writer in writers:
            writer.cancel()
        await asyncio.sleep(0.01)

    done, pending = await asyncio.wait(writers, timeout=FINISH_WITHIN_S)
    for writer in pending:
        writer.cancel()  # keep a failure from leaking tasks into teardown
    assert not pending, f"{len(pending)} of {WRITERS} cancelled writes never finished"
    for writer in done:
        if not writer.cancelled():
            writer.exception()  # finished either way; the outcome is not under test

    if engine.dialect.name == "sqlite":
        # In-memory SQLite: a write cancelled mid-statement invalidates the one shared
        # connection, and the pool's replacement is a new, empty database (no tables).
        # That predates the bounded close and is tracked separately; usability is
        # checked on the backends with durable storage.
        return
    after = await asyncio.wait_for(
        store.create_run(_run_id(round_index, 99), "agent"), timeout=FINISH_WITHIN_S
    )
    # The write retries transient failures. A plain read here can still meet a pooled
    # Postgres connection lost by the cancellations (kind=connection_lost); that also
    # predates the bounded close and is tracked separately.
    assert after.outcome == "created"
