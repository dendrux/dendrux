"""A statement larger than max_allowed_packet is refused before it is sent.

Without the preflight the server drops the connection mid-write. Sizes derive from the
server's real ``max_allowed_packet`` (MySQL 8.4: 64 MiB, MariaDB 10.11: 16 MiB).
"""

from __future__ import annotations

import pytest

from dendrux.errors import PersistenceError, StatementTooLargeError
from dendrux.store import MySQLStateStore
from dendrux.types import generate_ulid

from .test_060_p5_e_wiring import migrated_url, server_packet  # noqa: F401

MIB = 1024 * 1024


async def test_unicode_metadata_that_escapes_past_the_packet_is_refused_and_the_store_survives(
    migrated_url: str,  # noqa: F811
) -> None:
    packet = await server_packet(migrated_url)
    # Each e-acute is 2 bytes of text, 6 once the JSON column escapes it and 7 on the wire.
    letters = (packet + MIB) // 7
    store = MySQLStateStore.from_url(migrated_url)
    try:
        with pytest.raises(StatementTooLargeError) as caught:
            await store.create_run(generate_ulid(), "agent", meta={"k": "é" * letters})
        assert caught.value.limit == packet
        assert caught.value.size > packet
        assert isinstance(caught.value, PersistenceError)
        # Nothing was sent, so the same store works straight away.
        run_id = generate_ulid()
        await store.create_run(run_id, "agent", meta={"k": "small"})
        assert (await store.get_run(run_id)) is not None
    finally:
        await store._engine.dispose()


async def test_a_large_statement_that_fits_is_not_refused(migrated_url: str) -> None:  # noqa: F811
    packet = await server_packet(migrated_url)
    store = MySQLStateStore.from_url(migrated_url)
    try:
        run_id = generate_ulid()
        await store.create_run(run_id, "agent", meta={"k": "a" * (packet - 2 * MIB)})
        assert (await store.get_run(run_id)) is not None
    finally:
        await store._engine.dispose()
