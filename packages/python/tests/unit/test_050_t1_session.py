"""0.5.0: store resolution (B6 deprecation) and DB-URL masking in ``db/session.py``.

The Postgres event-loop guard is tested in
``tests/integration/test_050_t1_r1_migration.py``.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from dendrux import Agent
from dendrux.db import session
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse


async def test_reinitialize_error_masks_passwords(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session, "_engine", object())
    monkeypatch.setattr(session, "_engine_url", "postgresql+asyncpg://u:first-secret@h/db")
    with pytest.raises(RuntimeError) as caught:
        session._published("postgresql+asyncpg://u:second-secret@h/db?password=q")
    message = str(caught.value)
    assert "first-secret" not in message and "second-secret" not in message
    assert "password=q" not in message
    assert "***" in message


async def test_env_store_requires_explicit_configuration(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'env.db'}")
    agent = Agent(provider=MockLLM([]), prompt="p")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="state_store=.*database_url="):
            await agent.run("hi", tenant_id=None)
    assert not caught
    assert session._engine is None


async def test_explicit_store_does_not_warn(tmp_path: Any) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = Agent(
            strict_tenancy=False,
            provider=MockLLM([LLMResponse(text="ok")]),
            prompt="p",
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'own.db'}",
        )
        await agent.run("hi", tenant_id=None)
        await agent.close()
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


async def test_sqlite_global_engine_is_not_loop_bound(tmp_path: Any) -> None:
    """The loop guard applies to Postgres only; SQLite connections are not loop-bound."""
    import asyncio

    from sqlalchemy import text

    await session.reset_engine()
    try:
        engine = await session.get_engine(f"sqlite+aiosqlite:///{tmp_path / 'g.db'}")

        def other_loop() -> int:
            async def use() -> int:
                async with engine.connect() as conn:
                    return int((await conn.execute(text("SELECT 1"))).scalar_one())

            return asyncio.run(use())

        assert await asyncio.to_thread(other_loop) == 1
    finally:
        await session.reset_engine()
