"""An oversized statement: the server's answer is classified; its usual silence fails safe.

On both servers, asyncmy almost never sees the server's 1153 (packet too large): the server
closes the connection while the client is still writing, so the driver reports 2006 or 2013.
Those writes must fail safe through the existing connection-loss path (nothing persisted, the
store usable afterwards). When the server does answer 1153 or 1301, the store raises the typed
``StatementTooLargeError`` after exactly one attempt. That answer is injected here, because the
real servers do not give it.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import event, insert, select
from sqlalchemy.exc import OperationalError

from dendrux.db.models import AgentRun
from dendrux.errors import PersistenceError, StatementTooLargeError, StoreOutcomeUnknown
from dendrux.runtime.stores._errors import classify_error
from dendrux.store import MySQLStateStore
from dendrux.types import generate_ulid

from .test_060_p5_e_wiring import migrated_url, server_packet  # noqa: F401

MIB = 1024 * 1024


class DriverError(Exception):
    pass


async def run_ids(store: MySQLStateStore) -> set[str]:
    async with store._session_factory() as session:
        return set((await session.execute(select(AgentRun.id))).scalars())


async def test_the_servers_answer_becomes_the_typed_error_after_one_attempt(
    migrated_url: str,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    packet = await server_packet(migrated_url)
    store = MySQLStateStore.from_url(migrated_url)
    attempts = []

    def server_says_too_large(conn: Any, cursor: Any, statement: str, *rest: Any) -> None:
        if statement.lstrip().upper().startswith("INSERT INTO AGENT_RUNS"):
            attempts.append(statement)
            raise OperationalError(
                statement, (), DriverError(1153, "Got a packet bigger than 'max_allowed_packet'")
            )

    try:
        await store.get_run("warm")  # discovers max_allowed_packet
        event.listen(store._engine.sync_engine, "before_cursor_execute", server_says_too_large)
        with pytest.raises(StatementTooLargeError) as caught:
            await store.create_run(generate_ulid(), "agent", meta={"k": "secret-value"})
        event.remove(store._engine.sync_engine, "before_cursor_execute", server_says_too_large)
        error = caught.value
        assert not isinstance(error, StoreOutcomeUnknown)
        assert len(attempts) == 1
        assert any(getattr(r, "code", None) == "store_statement_limit" for r in caplog.records)
        assert error.limit == packet
        assert error.size is None
        assert str(packet) in str(error)
        assert "secret-value" not in str(error)
        assert await run_ids(store) == set()
        run_id = generate_ulid()
        await store.create_run(run_id, "agent")
        assert await run_ids(store) == {run_id}
    finally:
        await store._engine.dispose()


async def test_an_oversized_insert_fails_safe_and_the_store_survives(
    migrated_url: str,  # noqa: F811
) -> None:
    packet = await server_packet(migrated_url)
    store = MySQLStateStore.from_url(migrated_url)
    oversized = []

    def watch(conn: Any, cursor: Any, statement: str, parameters: Any, *rest: Any) -> None:
        if statement.lstrip().upper().startswith("INSERT INTO AGENT_RUNS") and any(
            isinstance(value, str) and len(value) > packet for value in parameters
        ):
            oversized.append(1)

    try:
        await store.get_run("warm")
        event.listen(store._engine.sync_engine, "before_cursor_execute", watch)
        with pytest.raises(PersistenceError) as caught:
            await store.create_run(generate_ulid(), "agent", meta={"k": "a" * (packet + MIB)})
        # Whatever the server answered, the failure is a persistence error, not a hang or a
        # retry loop: one attempt, nothing persisted, and the next normal write succeeds.
        assert len(oversized) == 1
        assert isinstance(
            caught.value, (StoreOutcomeUnknown, StatementTooLargeError, PersistenceError)
        )
        assert await run_ids(store) == set()
        run_id = generate_ulid()
        await store.create_run(run_id, "agent")
        assert await run_ids(store) == {run_id}
    finally:
        await store._engine.dispose()


async def test_an_oversized_executemany_fails_safe_and_the_store_survives(
    migrated_url: str,  # noqa: F811
) -> None:
    packet = await server_packet(migrated_url)
    store = MySQLStateStore.from_url(migrated_url)
    try:
        await store.get_run("warm")
        rows = [
            {
                "id": generate_ulid(),
                "agent_name": "a",
                "status": "running",
                "input_data": {},
                "meta": {"k": "a" * (packet + MIB)},
            }
            for _ in range(2)
        ]
        with pytest.raises(OperationalError) as caught:
            async with store._session_factory() as session:
                await session.execute(insert(AgentRun), rows)
                await session.commit()
        # Not an answer from the server but a lost connection: the existing connection-loss path.
        assert classify_error(caught.value).kind in ("connection_lost", "statement_too_large")
        assert await run_ids(store) == set()
        run_id = generate_ulid()
        await store.create_run(run_id, "agent")
        assert await run_ids(store) == {run_id}
    finally:
        await store._engine.dispose()
