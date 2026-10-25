"""Portable pool-lifecycle unit checks; runnable with stdlib unittest on Python 3.13.

The subject is the actual source module. Only its database-option and awaited
close dependencies are doubled, so this also runs without installed SDK deps.
Real SQLAlchemy/asyncpg integration is tested separately.
"""

import atexit
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch


async def _close_fully(close):
    await close()


def _load_subject():
    options = types.ModuleType("dendrux.db._engine_options")
    for name in ("engine_options", "frozen_options", "normalized_url"):
        setattr(options, name, Mock())
    guard = types.ModuleType("dendrux.db._pool_guard")
    guard.close_fully = _close_fully
    source = Path(__file__).resolve().parents[2] / "src/dendrux/db/_shared_pool.py"
    spec = importlib.util.spec_from_file_location("_retirement_subject", source)
    assert spec is not None and spec.loader is not None
    subject = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {options.__name__: options, guard.__name__: guard, spec.name: subject},
    ):
        spec.loader.exec_module(subject)
    atexit.unregister(subject._retire_at_exit)
    return subject


class PoolRetirementTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.subject = _load_subject()

    def entry(self, *, closed=True, driver="asyncpg", refs=1):
        engine = Mock()
        engine.dispose = AsyncMock()
        engine.dialect.driver = driver
        engine.url = "postgresql://USER_SECRET:PASSWORD_SECRET@HOST_SECRET/DB_SECRET"
        loop = Mock()
        loop.is_closed.return_value = closed
        ready = Mock()
        ready.done.return_value = True
        return self.subject._Entry(engine=engine, ready=ready, loop=loop, refs=refs)

    async def test_closed_owner_detaches_and_warns_without_identifiers(self):
        entry = self.entry()
        with self.assertLogs(self.subject.__name__, level="WARNING") as captured:
            await self.subject._dispose_entry(entry)
        entry.engine.dispose.assert_awaited_once_with(close=False)
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertEqual(record.code, "shared_engine_loop_closed")
        self.assertIn("agent.close()", record.getMessage())
        self.assertIn("async with Agent", record.getMessage())
        for secret in ("USER_SECRET", "PASSWORD_SECRET", "HOST_SECRET", "DB_SECRET"):
            self.assertNotIn(secret, str(record.__dict__))
        entry.ready.done.assert_not_called()
        entry.ready.cancel.assert_not_called()

    async def test_closed_owner_never_inspects_or_awaits_the_dead_loop_future(self):
        for driver in ("asyncpg", "aiosqlite", "unknown"):
            with self.subTest(driver=driver):
                entry = self.entry(driver=driver)
                entry.ready.done.side_effect = AssertionError("must not touch a dead-loop future")
                with self.assertLogs(self.subject.__name__, level="WARNING"):
                    await self.subject._dispose_entry(entry)
                entry.engine.dispose.assert_awaited_once_with(close=False)
                entry.ready.cancel.assert_not_called()

    async def test_closed_entry_warns_once_across_repeated_releases(self):
        entry = self.entry(refs=2)
        key = object()
        self.subject._entries[key] = entry
        self.subject._by_engine[entry.engine] = key
        with self.assertLogs(self.subject.__name__, level="WARNING") as captured:
            await self.subject.release(entry.engine)
            await self.subject.release(entry.engine)
        self.assertEqual(len(captured.records), 1)
        entry.engine.dispose.assert_awaited_once_with(close=False)
        self.assertFalse(self.subject._entries)
        self.assertFalse(self.subject._by_engine)

    async def test_live_owner_still_disposes_gracefully(self):
        entry = self.entry(closed=False)
        with self.assertNoLogs(self.subject.__name__, level="WARNING"):
            await self.subject._dispose_entry(entry)
        entry.engine.dispose.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
