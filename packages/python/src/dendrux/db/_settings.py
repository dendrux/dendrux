"""Small, transaction-local storage settings operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from dendrux.db._dialects import is_mysql_family

if TYPE_CHECKING:
    from sqlalchemy import Connection


def put_setting(conn: Connection, key: str, value: str, *, overwrite: bool = True) -> None:
    """Write a setting without committing the caller's transaction."""
    if is_mysql_family(conn.dialect):
        from dendrux.db._dialects.mysql import put_setting as put_mysql_setting

        put_mysql_setting(conn, key, value, overwrite=overwrite)
        return
    action = (
        "DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at"
        if overwrite
        else "DO NOTHING"
    )
    conn.execute(
        text(
            "INSERT INTO dendrux_settings (key, value, updated_at) "
            f"VALUES (:key, :value, CURRENT_TIMESTAMP) ON CONFLICT (key) {action}"
        ),
        {"key": key, "value": value},
    )


def seed_settings(conn: Connection, *, active: bool) -> None:
    """Seed fresh schemas active and existing schemas pre-activation, without flipping either."""
    put_setting(conn, "storage_format", "2" if active else "1", overwrite=False)
    if active:
        put_setting(conn, "activate_step", "done", overwrite=False)
