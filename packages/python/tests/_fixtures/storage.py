"""Explicit storage-format selection for compatibility and recovery tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dendrux.db._settings import put_setting

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


async def pin_storage_format(engine: AsyncEngine, fmt: int) -> None:
    """Pin the test database without changing production create_all defaults."""
    if fmt not in (1, 2):
        raise ValueError("storage format must be 1 or 2")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: put_setting(c, "storage_format", str(fmt)))
