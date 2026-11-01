"""SQLite startup stays additive and never activates legacy files."""

from sqlalchemy import create_engine, inspect, text

from dendrux.db._sqlite_sync import create_sqlite_schema_sync, sync_sqlite_schema
from tests.migrations.test_060_activate import INDEXES, NEW_COLUMNS, prior_schema


def test_sync_adds_columns_settings_without_uniqueness(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    try:
        with engine.begin() as conn:
            prior_schema(conn)
        with engine.connect() as conn:
            create_sqlite_schema_sync(conn)
            assert "dendrux_settings" in inspect(conn).get_table_names()
            for table, columns in NEW_COLUMNS.items():
                assert set(columns) <= {c["name"] for c in inspect(conn).get_columns(table)}
            assert dict(conn.execute(text("SELECT key, value FROM dendrux_settings")).all()) == {
                "storage_format": "1"
            }
            for name, (table, _, old) in INDEXES.items():
                names = {i["name"] for i in inspect(conn).get_indexes(table)}
                assert name not in names
                if old:
                    assert old in names
            assert sync_sqlite_schema(conn) == []
    finally:
        engine.dispose()


def test_partial_legacy_schema_never_builds_activation_indexes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'partial.db'}")
    try:
        with engine.begin() as conn:
            prior_schema(conn)
            conn.exec_driver_sql("DROP TABLE react_traces")
        with engine.connect() as conn:
            create_sqlite_schema_sync(conn)
            assert "uq_react_traces_run_order" not in {
                i["name"] for i in inspect(conn).get_indexes("react_traces")
            }
    finally:
        engine.dispose()


async def test_runtime_startup_checks_active_indexes(tmp_path):
    import pytest
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db._sqlite_sync import create_sqlite_schema
    from dendrux.db.models import Base
    from dendrux.errors import SchemaOutOfDateError

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'active.db'}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("DROP INDEX uq_react_traces_run_order"))
        with pytest.raises(SchemaOutOfDateError):
            await create_sqlite_schema(engine)
    finally:
        await engine.dispose()
