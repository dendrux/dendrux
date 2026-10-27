"""Deterministic coverage of two paths the CRAP ratchet scores but no other test reliably runs.

`get_session` on a pool with real connections (Postgres, a file SQLite database) and
`InMemoryBlobCatalog.adopt_pending`. Without these, whether the lines ran depended on
which incidental race or error path the random test order hit, and the ratchet flaked.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db import session as db_session
from dendrux.db._pool_guard import shares_one_connection
from dendrux.media import BlobInfo
from dendrux.runtime._blob_catalog import InMemoryBlobCatalog

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


@pytest.fixture
async def file_engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 's.db'}")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE t (v INTEGER)"))
    assert not shares_one_connection(engine), "this test needs a pool with real connections"
    yield engine
    await engine.dispose()


async def _count(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        return int((await conn.execute(text("SELECT count(*) FROM t"))).scalar_one())


class TestGetSessionOnAPoolWithRealConnections:
    async def test_commits_on_success(self, file_engine: AsyncEngine) -> None:
        async with db_session.get_session(file_engine) as session:
            await session.execute(text("INSERT INTO t VALUES (1)"))
        assert await _count(file_engine) == 1

    async def test_rolls_back_and_reraises_on_error(self, file_engine: AsyncEngine) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with db_session.get_session(file_engine) as session:
                await session.execute(text("INSERT INTO t VALUES (1)"))
                raise RuntimeError("boom")
        assert await _count(file_engine) == 0

    async def test_the_shared_engine_uses_the_cached_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'g.db'}")
        await db_session.reset_engine()
        try:
            engine = await db_session.get_engine()
            assert not shares_one_connection(engine)
            async with db_session.get_session() as session:
                await session.execute(text("CREATE TABLE t (v INTEGER)"))
                await session.execute(text("INSERT INTO t VALUES (1)"))
            with pytest.raises(RuntimeError, match="boom"):
                async with db_session.get_session() as session:
                    await session.execute(text("INSERT INTO t VALUES (2)"))
                    raise RuntimeError("boom")
            assert await _count(engine) == 1
        finally:
            await db_session.reset_engine()


def _pending_row(sha: str, generation: str) -> BlobInfo:
    now = datetime.now(UTC)
    return BlobInfo(
        sha256=sha,
        media_type="image/png",
        byte_length=3,
        width=1,
        height=1,
        frame_count=1,
        storage="db",
        state="pending",
        generation=generation,
        created_at=now,
        last_used_at=now,
        state_changed_at=now,
    )


class TestInMemoryAdoptPending:
    SHA = "a" * 64

    async def _catalog(self) -> InMemoryBlobCatalog:
        catalog = InMemoryBlobCatalog()
        inserted = await catalog.insert_row(
            "-",
            self.SHA,
            _pending_row(self.SHA, "g1"),
            state="pending",
            generation="g1",
            data=b"png",
        )
        assert inserted
        return catalog

    async def test_adopts_a_pending_row_with_the_expected_generation(self) -> None:
        catalog = await self._catalog()
        assert await catalog.adopt_pending("-", self.SHA, "g1", "g2") is True
        row = await catalog.get_row("-", self.SHA)
        assert row is not None and row.generation == "g2" and row.state == "pending"

    async def test_refuses_a_stale_generation(self) -> None:
        catalog = await self._catalog()
        assert await catalog.adopt_pending("-", self.SHA, "other", "g2") is False
        row = await catalog.get_row("-", self.SHA)
        assert row is not None and row.generation == "g1"

    async def test_refuses_a_row_that_is_no_longer_pending(self) -> None:
        catalog = await self._catalog()
        assert await catalog.promote("-", self.SHA, "g1") is True
        assert await catalog.adopt_pending("-", self.SHA, "g1", "g2") is False

    async def test_refuses_an_unknown_blob(self) -> None:
        assert await InMemoryBlobCatalog().adopt_pending("-", "b" * 64, "g1", "g2") is False
