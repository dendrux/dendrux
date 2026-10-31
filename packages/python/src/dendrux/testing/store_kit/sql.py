"""Kit helpers for stores built on ``SQLStateStore``: a fault injector and a clock.

``SqlFaultInjector`` is a ready ``dendrux_fault_injector``; ``SqlClock`` is a ready
``dendrux_clock``. Both work on any SQLAlchemy engine, so a subclass of ``SQLStateStore`` on a new
database gets them for free.

The fault injector
``arm(store, point, kind=..., after=n, sticky=...)`` makes a write transaction on that store's
engine fail. The ``after`` first write transactions are let through; then the fault fires once,
or on every write transaction until ``disarm`` when ``sticky``. ``fired(store)`` says how many
times it fired since the last ``arm``, so a contract can assert the cut really happened.

``before_commit``  raise just before COMMIT (nothing persists);
``after_commit``   raise just after COMMIT (the write persists: a lost acknowledgement);
``mid_write``      raise on the second write statement of the transaction, or before COMMIT when
                   it has only one (the transaction's writes must not persist).

``kind="error"`` raises a connection-loss ``OperationalError``;
``kind="cancel"`` raises ``asyncio.CancelledError``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from dendrux.db.models import AgentRun

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

_WRITES = ("INSERT", "UPDATE", "DELETE")


class _Arm:
    def __init__(self, point: str, kind: str, after: int, sticky: bool) -> None:
        self.point = point
        self.kind = kind
        self.after = after
        self.sticky = sticky
        self.writes = 0  # write statements in the transaction being built
        self.passed = False  # the transaction that just committed was let through

    def failure(self) -> BaseException:
        if self.kind == "cancel":
            return asyncio.CancelledError()
        return OperationalError(
            "kit fault", None, ConnectionResetError("kit fault"), connection_invalidated=True
        )


class SqlFaultInjector:
    def __init__(self) -> None:
        self._armed: dict[Any, _Arm] = {}  # sync Engine -> armed fault
        self._fired: dict[Any, int] = {}  # sync Engine -> times the last armed fault fired
        self._installed: set[Any] = set()
        self._session_hooks = False

    # -- the kit's interface -----------------------------------------------------------------

    def arm(
        self,
        store: Any,
        point: str,
        *,
        kind: str = "error",
        after: int = 0,
        sticky: bool = False,
    ) -> None:
        engine = store._engine.sync_engine
        self._fired[engine] = 0
        if engine not in self._installed:
            self._installed.add(engine)
            event.listen(engine, "before_cursor_execute", self._on_execute)
        if not self._session_hooks:
            self._session_hooks = True
            event.listen(Session, "before_commit", self._before_commit)
            event.listen(Session, "after_commit", self._after_commit)
            event.listen(Session, "after_rollback", self._after_rollback)
        self._armed[engine] = _Arm(point, kind, after, sticky)

    def fired(self, store: Any) -> int:
        return self._fired.get(store._engine.sync_engine, 0)

    def _fire(self, engine: Any, armed: _Arm) -> BaseException:
        self._fired[engine] = self._fired.get(engine, 0) + 1
        if not armed.sticky:
            del self._armed[engine]
        armed.writes = 0
        return armed.failure()

    def disarm(self, store: Any | None = None) -> None:
        if store is None:
            self._armed.clear()
        else:
            self._armed.pop(store._engine.sync_engine, None)

    def close(self) -> None:
        self._armed.clear()
        for engine in self._installed:
            event.remove(engine, "before_cursor_execute", self._on_execute)
        if self._session_hooks:
            event.remove(Session, "before_commit", self._before_commit)
            event.remove(Session, "after_commit", self._after_commit)
            event.remove(Session, "after_rollback", self._after_rollback)
            self._session_hooks = False
        self._installed.clear()

    # -- SQLAlchemy listeners ----------------------------------------------------------------

    def _on_execute(self, conn: Any, cursor: Any, statement: str, *_: Any) -> None:
        armed = self._armed.get(conn.engine)
        if armed is None or not statement.lstrip().upper().startswith(_WRITES):
            return
        armed.writes += 1
        if armed.point == "mid_write" and armed.after == 0 and armed.writes >= 2:
            raise self._fire(conn.engine, armed)

    def _armed_for(self, session: Session) -> tuple[Any, _Arm] | None:
        try:
            engine = session.get_bind()
        except Exception:
            return None
        armed = self._armed.get(engine)
        return (engine, armed) if armed is not None and armed.writes else None

    def _before_commit(self, session: Session) -> None:
        found = self._armed_for(session)
        if found is None:
            return
        engine, armed = found
        if armed.after > 0:
            armed.after -= 1  # a write transaction let through
            armed.writes = 0
            armed.passed = True
            return
        armed.passed = False
        if armed.point in ("before_commit", "mid_write"):
            raise self._fire(engine, armed)

    def _after_commit(self, session: Session) -> None:
        try:
            engine = session.get_bind()
        except Exception:
            return
        armed = self._armed.get(engine)
        if armed is None:
            return
        if armed.passed:
            armed.passed = False
            return
        if armed.point == "after_commit" and armed.writes and armed.after == 0:
            raise self._fire(engine, armed)
        armed.writes = 0

    def _after_rollback(self, session: Session) -> None:
        try:
            armed = self._armed.get(session.get_bind())
        except Exception:
            return
        if armed is not None:
            armed.writes = 0


_TIME_COLUMNS = (
    "last_progress_at",
    "status_changed_at",
    "updated_at",
    "created_at",
    "lease_expired_at",
    "last_reclaimed_at",
)


class SqlClock:
    """A ``dendrux_clock`` for SQL stores: ``advance`` ages every stored run clock.

    The store reads wall-clock time, so the clock moves the world forward by rewriting the
    stored timestamps ``delta`` into the past. ``engine`` is read at each call, so it can be a
    function returning the engine of the store the test is using.
    """

    def __init__(self, engine: AsyncEngine | Callable[[], AsyncEngine]) -> None:
        self._engine = engine

    async def advance(self, delta: timedelta) -> None:
        engine = self._engine() if callable(self._engine) else self._engine
        columns = [getattr(AgentRun, name) for name in _TIME_COLUMNS]
        async with engine.begin() as conn:
            rows = (await conn.execute(select(AgentRun.id, *columns))).mappings().all()
            for row in rows:
                values = {}
                for name in _TIME_COLUMNS:
                    stamp = row[name]
                    if stamp is not None:
                        values[name] = (
                            stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)
                        ) - delta
                if values:
                    await conn.execute(
                        update(AgentRun).where(AgentRun.id == row["id"]).values(**values)
                    )
