"""Store session factories on multi-connection pools close each session exactly once.

``_CheckedSessionFactory`` over ``_ClosingSessionFactory``, as the SQL store wires them
for pools that do not share one connection. A gated fake session pins the lifecycle:
one close on success, on a body error and on a failed connection check, and a
cancellation that stays a cancellation, including one that lands during a slow close.
"""

from __future__ import annotations

import asyncio

import pytest

from dendrux.db import _pool_guard
from dendrux.runtime.stores.sql._sql_sessions import (
    _CheckedSessionFactory,
    _ClosingSessionFactory,
)

BOUND_S = 0.1


class _FakeSession:
    def __init__(self, close_gate: asyncio.Event | None = None) -> None:
        self.closes = 0
        self.close_finished = 0
        self._gate = close_gate

    async def close(self) -> None:
        self.closes += 1
        if self._gate is not None:
            await self._gate.wait()
        self.close_finished += 1


class _FakeStore:
    def __init__(self, *, check_fails: bool = False) -> None:
        self.check_fails = check_fails

    async def _check_connection(self, session: object) -> None:
        if self.check_fails:
            raise ConnectionError("check failed")


def _factory(session: _FakeSession, store: _FakeStore) -> _CheckedSessionFactory:
    return _CheckedSessionFactory(store, _ClosingSessionFactory(lambda: session))  # type: ignore[arg-type,return-value]


async def test_success_closes_once() -> None:
    session = _FakeSession()
    async with _factory(session, _FakeStore())() as yielded:
        assert yielded is session
    assert session.closes == 1


async def test_a_body_error_closes_once_and_propagates() -> None:
    session = _FakeSession()
    with pytest.raises(LookupError):
        async with _factory(session, _FakeStore())():
            raise LookupError("body")
    assert session.closes == 1


async def test_a_failed_connection_check_closes_once() -> None:
    session = _FakeSession()
    with pytest.raises(ConnectionError):
        async with _factory(session, _FakeStore(check_fails=True))():
            pytest.fail("the body must not run after a failed check")
    assert session.closes == 1


async def test_a_cancelled_body_stays_cancelled_and_closes_once() -> None:
    session = _FakeSession()
    entered = asyncio.Event()

    async def caller() -> None:
        async with _factory(session, _FakeStore())():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(caller())
    await entered.wait()
    task.cancel()
    await asyncio.wait({task}, timeout=5)
    assert task.cancelled()
    assert session.closes == 1


async def test_a_cancel_during_a_stuck_close_is_bounded_and_stays_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_pool_guard, "CLOSE_AFTER_CANCEL_S", BOUND_S)
    gate = asyncio.Event()
    session = _FakeSession(close_gate=gate)

    async def caller() -> None:
        async with _factory(session, _FakeStore())():
            pass

    task = asyncio.create_task(caller())
    while session.closes == 0:
        await asyncio.sleep(0)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=BOUND_S * 20)
    assert done, "a cancelled caller waited on a stuck close past the bound"
    assert task.cancelled()
    assert session.close_finished == 0  # abandoned, still running

    gate.set()
    for _ in range(100):
        if session.close_finished:
            break
        await asyncio.sleep(0.01)
    assert (session.closes, session.close_finished) == (1, 1)
