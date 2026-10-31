"""MySQL-family DDL from the models, compiled offline for MySQL 8.4 and MariaDB 10.11.

Identifier, tenant and key columns need a NO PAD binary collation chosen from the server
(``utf8mb4_0900_bin`` on MySQL, ``utf8mb4_nopad_bin`` on MariaDB; ``utf8mb4_bin`` pads and
is forbidden). Large text/binary need LONG variants, timestamps ``DATETIME(6)`` in UTC.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest
from sqlalchemy import CHAR, DateTime, LargeBinary, String, Text, create_mock_engine, event
from sqlalchemy.schema import CreateTable

from dendrux.db import models
from dendrux.db.models import Base

from .test_060_p5_ddl_golden import _normalise

COLLATIONS = {"mysql": "utf8mb4_0900_bin", "mariadb": "utf8mb4_nopad_bin"}


def _engine(server: str, statements: list[str] | None = None):  # noqa: ANN202
    sink = statements if statements is not None else []

    def executor(sql, *multiparams, **params):  # noqa: ANN001, ANN002, ANN003
        sink.append(_normalise(str(sql.compile(dialect=engine.dialect))))

    engine = create_mock_engine("mysql+asyncmy://", executor)
    engine.dialect.is_mariadb = server == "mariadb"
    engine.info = {}
    return engine


def _column_sql(server: str, table_name: str) -> dict[str, str]:
    """Column name -> its compiled column specification in CREATE TABLE."""
    engine = _engine(server)
    ddl = str(CreateTable(Base.metadata.tables[table_name]).compile(dialect=engine.dialect))
    specs = {}
    for line in ddl.splitlines():
        match = re.match(r"\s+`?(\w+)`? (.+?),?\s*$", line)
        if match and match.group(1) not in {"PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE"}:
            specs[match.group(1)] = match.group(2)
    return specs


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_every_string_column_has_the_servers_nopad_binary_collation(server: str) -> None:
    collation = COLLATIONS[server]
    checked = 0
    for table in Base.metadata.sorted_tables:
        specs = _column_sql(server, table.name)
        for column in table.columns:
            if isinstance(column.type, String) and not isinstance(column.type, Text):
                spec = specs[column.name]
                assert f"CHARACTER SET utf8mb4 COLLATE {collation}" in spec, (table.name, spec)
                assert "utf8mb4_bin" not in spec
                checked += 1
    assert checked > 40


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_large_payload_and_time_columns_use_mysql_variants(server: str) -> None:
    for table in Base.metadata.sorted_tables:
        specs = _column_sql(server, table.name)
        for column in table.columns:
            spec = specs[column.name]
            if isinstance(column.type, Text):
                assert spec.startswith("LONGTEXT"), (table.name, column.name, spec)
            elif isinstance(column.type, LargeBinary):
                assert spec.startswith("LONGBLOB"), (table.name, column.name, spec)
            elif isinstance(column.type, DateTime):
                assert spec.startswith("DATETIME(6)"), (table.name, column.name, spec)
            elif isinstance(column.type, CHAR):
                assert spec.startswith("CHAR("), spec


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_status_clock_server_default_is_utc_with_microseconds(server: str) -> None:
    spec = _column_sql(server, "agent_runs")["status_changed_at"]
    assert spec == "DATETIME(6) NOT NULL DEFAULT (UTC_TIMESTAMP(6))"


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_tables_are_innodb_utf8mb4(server: str) -> None:
    engine = _engine(server)
    for table in Base.metadata.sorted_tables:
        ddl = _normalise(str(CreateTable(table).compile(dialect=engine.dialect)))
        options = ddl.rsplit(")", 1)[1].split()
        assert "ENGINE=InnoDB" in options, (table.name, options)
        assert ("DEFAULT", "CHARSET=utf8mb4") in zip(options, options[1:], strict=False)


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_create_all_uses_a_generated_tenant_scope_and_composite_started_index(
    server: str,
) -> None:
    statements: list[str] = []
    engine = _engine(server, statements)
    event.remove(Base.metadata, "after_create", models._seed_storage_settings)
    try:
        Base.metadata.create_all(engine, checkfirst=False)
    finally:
        event.listen(Base.metadata, "after_create", models._seed_storage_settings)
    # The expression index fails on MariaDB 10.11 (1064); MySQL-family servers get none.
    assert not any("INDEX" in s and "coalesce" in s.lower() for s in statements)
    collation = COLLATIONS[server]
    assert (
        "ALTER TABLE agent_runs ADD COLUMN tenant_scope VARCHAR(255) CHARACTER SET utf8mb4 "
        f"COLLATE {collation} GENERATED ALWAYS AS (COALESCE(tenant_id, '')) STORED"
    ) in statements
    assert (
        "CREATE UNIQUE INDEX uq_agent_runs_tenant_idempotency_key "
        "ON agent_runs (tenant_scope, idempotency_key)"
    ) in statements
    assert "CREATE INDEX ix_run_events_started ON run_events (agent_run_id, event_type)" in (
        statements
    )
    alter = statements.index(
        next(s for s in statements if s.startswith("ALTER TABLE agent_runs ADD COLUMN"))
    )
    unique = statements.index(
        next(s for s in statements if "uq_agent_runs_tenant_idempotency_key" in s)
    )
    assert alter < unique
    assert sum("uq_agent_runs_tenant_idempotency_key" in s for s in statements) == 1
    assert sum("ix_run_events_started" in s for s in statements) == 1


def _dialect(server: str):  # noqa: ANN202
    return _engine(server).dialect


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_datetimes_bind_as_naive_utc_and_read_back_aware(server: str) -> None:
    dialect = _dialect(server)
    column = Base.metadata.tables["agent_runs"].c.created_at
    impl = column.type.dialect_impl(dialect)
    bind = impl.bind_processor(dialect)
    result = impl.result_processor(dialect, None)
    assert bind is not None and result is not None
    india = dt.timezone(dt.timedelta(hours=5, minutes=30))
    aware = dt.datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=india)
    stored = bind(aware)
    assert stored == dt.datetime(2026, 1, 1, 21, 34, 5, 123456)
    assert stored.tzinfo is None
    assert result(stored) == aware
    assert result(stored).tzinfo == dt.UTC
    assert bind(None) is None and result(None) is None
    with pytest.raises(ValueError, match="aware"):
        bind(dt.datetime(2026, 1, 1))  # naive: refused, never guessed
