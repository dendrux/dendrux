"""MySQL-family connection policy, checked before any store operation."""

# ruff: noqa: F401, F811
import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.runtime.stores.sql.mysql import MySQLStateStore
from tests.integration._mysql import mysql_engine, store
from tests.integration.test_sql_store_rx_checks import (
    test_autocommit_refused_before_twenty_claims,
    test_schema_driver_errors_are_typed,
)


@pytest.mark.parametrize(
    "setting,message",
    [
        ("sql_mode=''", "strict sql_mode"),
        ("character_set_client=latin1", "utf8mb4"),
        ("character_set_connection=utf8mb3", "utf8mb4"),
        ("character_set_results=latin1", "utf8mb4"),
    ],
)
async def test_unsafe_mysql_session_refused(backend_engine, setting, message):
    borrowed = create_async_engine(
        backend_engine.url,
        isolation_level="READ COMMITTED",
        connect_args={"charset": "utf8mb4"},
    )

    @event.listens_for(borrowed.sync_engine, "connect")
    def configure(connection, record):
        cursor = connection.cursor()
        cursor.execute(f"SET SESSION {setting}")
        cursor.close()

    try:
        with pytest.raises(ValueError, match=message):
            await MySQLStateStore(borrowed).get_run("absent")
    finally:
        await borrowed.dispose()


@pytest.fixture
def backend_engine(mysql_engine):
    return mysql_engine
