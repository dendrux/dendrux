"""Examples reload absent pending projections before submitting work."""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.db._settings import seed_settings
from dendrux.db.models import Base
from dendrux.store import SQLiteStateStore


@pytest.mark.parametrize("example", ["39_mysql_store", "40_live_updates"])
@pytest.mark.parametrize("changed_pause", [False, True])
async def test_example_reloads_missing_pending(monkeypatch, capsys, example, changed_pause):
    path = Path(__file__).parents[2] / "examples" / f"{example}.py"
    spec = importlib.util.spec_from_file_location(example, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = Agent.run

    async def without_projection(self, *args, **kwargs):
        result = await run(self, *args, **kwargs)
        result.pending = None
        result.pause_id = None
        return result

    monkeypatch.setattr(Agent, "run", without_projection)
    if changed_pause:
        monkeypatch.setattr(module.RunStore, "get_pending", AsyncMock(return_value=None))
    if example.startswith("39"):
        # Exercise the example flow locally with SQLite, without a MySQL connection.
        engine = create_async_engine("sqlite+aiosqlite://")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda sync: seed_settings(sync, active=True))
        monkeypatch.setenv(module.URL_ENV, "sqlite+aiosqlite://")
        monkeypatch.setattr(module, "create_async_engine", lambda *args, **kwargs: engine)
        monkeypatch.setattr(module, "MySQLStateStore", SQLiteStateStore)
        assert await module.main() == 0
    else:
        assert await module.main("loopback") == 0
    output = capsys.readouterr().out.lower()
    assert ("pause changed" if changed_pause else "finished: success") in output
