"""Select built-in SQL stores at the engine boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.runtime.stores.sql.base import SQLStateStore


def store_for_engine(engine: AsyncEngine, **opts: Any) -> SQLStateStore:
    """Construct the store matching an engine's SQLAlchemy dialect name."""
    name = engine.dialect.name
    if name == "postgresql":
        from dendrux.runtime.stores.sql.postgres import PostgresStateStore

        return PostgresStateStore(engine, **opts)
    if name == "sqlite":
        from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

        return SQLiteStateStore(engine, **opts)
    if name == "mysql":
        from dendrux.runtime.stores.sql.mysql import MySQLStateStore

        return MySQLStateStore(engine, **opts)
    raise ValueError("Unsupported SQL dialect; subclass SQLStateStore to provide an adapter")
