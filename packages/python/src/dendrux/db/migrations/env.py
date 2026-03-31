"""Alembic env.py — async migration runner for Dendrux.

Supports both SQLite (default) and Postgres (via DENDRUX_DATABASE_URL).
Runs migrations using SQLAlchemy's async engine.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from dendrux.db.models import Base
from dendrux.db.session import get_database_url

# Alembic Config object — access to alembic.ini values
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata

# Override sqlalchemy.url from environment — but only if the CLI hasn't already set it
# (e.g. via `dendrux db migrate --url postgres://...`)
# TODO(post-alpha): use a dedicated sentinel value in alembic.ini instead of comparing
# against the literal default URL — this breaks if someone changes alembic.ini's default.
existing_url = config.get_main_option("sqlalchemy.url")
if not existing_url or existing_url == "sqlite+aiosqlite:///./dendrux.db":
    config.set_main_option("sqlalchemy.url", get_database_url())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL without a live connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):  # noqa: ANN001
    """Configure context and run migrations within a connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using an async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — with a live async connection."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
