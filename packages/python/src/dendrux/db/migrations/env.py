"""Alembic env.py — async migration runner for Dendrux.

Supports SQLite (default), Postgres and MySQL/MariaDB (via DENDRUX_DATABASE_URL).
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
    fileConfig(config.config_file_name, disable_existing_loggers=False)

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
    if url and url.split(":", 1)[0].split("+", 1)[0] in {"mysql", "mariadb"}:
        raise RuntimeError("MySQL/MariaDB migrations need a live connection (no --sql).")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def _configure_and_run(connection):  # noqa: ANN001
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_mysql(connection):  # noqa: ANN001
    """MySQL/MariaDB: fresh-baseline bootstrap or upgrade, under the maintenance GET_LOCK.

    Upgrading to head bootstraps an empty database (or resumes one), upgrades a stamped
    Dendrux schema, and refuses anything else. Read-only commands (``current``) run
    normally under the lock; other targets (downgrades, stamps) are refused because the
    pre-baseline revisions do not apply to these servers.
    """
    from dendrux.db._dialects import mysql

    script = context.script
    head = script.get_current_head()
    try:
        destination = context.get_revision_argument()
    except KeyError:  # no destination: a read-only command such as `alembic current`
        with mysql.maintenance_lock(connection):
            _configure_and_run(connection)
        return
    if destination != head:
        raise RuntimeError(
            "On MySQL/MariaDB Dendrux migrates only up to head (`dendrux db migrate`); "
            "restore a backup to go back."
        )
    upgradeable = mysql.upgradeable_revisions(script, head)
    mysql.migrate(
        connection,
        head=head,
        upgradeable=upgradeable,
        upgrade=lambda: _configure_and_run(connection),
    )


def do_run_migrations(connection):  # noqa: ANN001
    """Lock before Alembic reads the revision; retain it through all commits."""
    from dendrux.db._dialects import is_mysql_family
    from dendrux.db._dialects.postgres import migration_lock

    if is_mysql_family(connection.dialect):
        _run_mysql(connection)
        return
    if connection.dialect.name == "postgresql":
        with migration_lock(connection):
            _configure_and_run(connection)
        return
    _configure_and_run(connection)


async def run_async_migrations() -> None:
    """Run migrations using an async engine."""
    from dendrux.db._engine_options import (
        is_mysql_family_url,
        mysql_engine_options,
        normalized_url,
    )

    section = dict(config.get_section(config.config_ini_section, {}))
    url = section.get("sqlalchemy.url", "")
    options: dict = {}
    if url and is_mysql_family_url(url):
        # The same READ COMMITTED / UTC / driver policy as every other engine Dendrux creates.
        section["sqlalchemy.url"] = normalized_url(url).render_as_string(hide_password=False)
        options = mysql_engine_options({})
    connectable = async_engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool, **options
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — with a live async connection."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
