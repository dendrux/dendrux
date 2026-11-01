"""The models on real MySQL 8.4 and MariaDB 10.11 servers (``create_all``).

Covers the column variants (NO PAD collation per server, LONGTEXT/LONGBLOB sizes,
UTC microsecond timestamps), the generated tenant scope and the settings seed.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db.models import AgentRun, Base, Blob, DendruxSetting, ReactTrace

COLLATIONS = {False: "utf8mb4_0900_bin", True: "utf8mb4_nopad_bin"}


async def _create_all(url: str):  # noqa: ANN202
    engine = create_async_engine(url, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def test_create_all_builds_every_table_and_seeds_an_active_format(mysql_url: str) -> None:
    engine = await _create_all(mysql_url)
    try:
        async with engine.connect() as conn:
            settings = dict(
                (await conn.execute(select(DendruxSetting.key, DendruxSetting.value))).all()
            )
            tables = set(
                (
                    await conn.execute(
                        text(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = DATABASE() AND engine = 'InnoDB'"
                        )
                    )
                ).scalars()
            )
        assert settings == {"storage_format": "2", "activate_step": "done"}
        assert tables == set(Base.metadata.tables)
    finally:
        await engine.dispose()


async def test_every_identifier_column_has_the_servers_nopad_collation(mysql_url: str) -> None:
    engine = await _create_all(mysql_url)
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT table_name, column_name, data_type, collation_name, "
                        "generation_expression FROM information_schema.columns "
                        "WHERE table_schema = DATABASE() AND data_type IN ('varchar', 'char')"
                    )
                )
            ).all()
        expected = COLLATIONS[engine.dialect.is_mariadb]
        assert len(rows) > 40
        assert {row[3] for row in rows} == {expected}
        scope = [row for row in rows if row[:2] == ("agent_runs", "tenant_scope")]
        assert len(scope) == 1
        assert "coalesce" in scope[0][4].lower() and "tenant_id" in scope[0][4]
    finally:
        await engine.dispose()


async def test_payloads_beyond_64k_and_utc_microseconds_round_trip(mysql_url: str) -> None:
    engine = await _create_all(mysql_url)
    india = dt.timezone(dt.timedelta(hours=5, minutes=30))
    stamp = dt.datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=india)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                insert(AgentRun).values(
                    id="r1",
                    agent_name="a",
                    error="e" * 1_048_576,
                    created_at=stamp,
                    updated_at=stamp,
                    status_changed_at=stamp,
                )
            )
            await conn.execute(
                insert(ReactTrace).values(
                    id="t1",
                    agent_run_id="r1",
                    role="user",
                    content="c" * 1_048_576,
                    order_index=0,
                    created_at=stamp,
                    updated_at=stamp,
                )
            )
            await conn.execute(
                insert(Blob).values(
                    tenant_scope="t",
                    sha256="0" * 64,
                    generation="g",
                    state="live",
                    media_type="image/png",
                    byte_length=4 * 1_048_576,
                    width=1,
                    height=1,
                    storage="db",
                    data=b"\x00" * (4 * 1_048_576),
                )
            )
        async with engine.connect() as conn:
            created, error = (await conn.execute(select(AgentRun.created_at, AgentRun.error))).one()
            content = (await conn.execute(select(ReactTrace.content))).scalar_one()
            data = (await conn.execute(select(Blob.data))).scalar_one()
            raw = (
                await conn.execute(text("SELECT CAST(created_at AS CHAR) FROM agent_runs"))
            ).scalar_one()
        assert created == stamp and created.tzinfo == dt.UTC
        assert raw == "2026-01-01 21:34:05.123456"  # stored as UTC, microseconds kept
        assert len(error) == len(content) == 1_048_576
        assert len(data) == 4 * 1_048_576
    finally:
        await engine.dispose()


async def test_status_clock_server_default_is_utc(mysql_url: str) -> None:
    engine = await _create_all(mysql_url)
    try:
        async with engine.begin() as conn:
            # A writer that omits the clock (a raw insert) gets the database's UTC time.
            await conn.execute(text("SET time_zone = '+05:30'"))
            await conn.execute(
                text(
                    "INSERT INTO agent_runs (id, agent_name, status, iteration_count, "
                    "delegation_level, total_input_tokens, total_output_tokens, "
                    "total_cache_read_tokens, total_cache_creation_tokens, created_at, "
                    "updated_at) VALUES ('r1', 'a', 'pending', 0, 0, 0, 0, 0, 0, "
                    "UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))"
                )
            )
            clock = (
                await conn.execute(
                    text(
                        "SELECT TIMESTAMPDIFF(SECOND, status_changed_at, UTC_TIMESTAMP(6)) "
                        "FROM agent_runs"
                    )
                )
            ).scalar_one()
        assert abs(clock) <= 5
    finally:
        await engine.dispose()
