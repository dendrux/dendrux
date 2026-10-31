"""A fault injector for the repo's SQL stores (the optional ``dendrux_fault_injector``).

``arm(store, point, kind=...)`` makes the next write transaction on that store's engine fail:

``before_commit``  raise just before COMMIT (nothing persists);
``after_commit``   raise just after COMMIT (the write persists: a lost acknowledgement);
``mid_write``      raise on the second write statement of the transaction, or before COMMIT when
                   it has only one (the transaction's writes must not persist).

One shot. ``kind="error"`` raises a connection-loss ``OperationalError``;
``kind="cancel"`` raises ``asyncio.CancelledError``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

_WRITES = ("INSERT", "UPDATE", "DELETE")


class _Arm:
    def __init__(self, point: str, kind: str) -> None:
        self.point = point
        self.kind = kind
        self.writes = 0

    def failure(self) -> BaseException:
        if self.kind == "cancel":
            return asyncio.CancelledError()
        return OperationalError("kit fault", None, ConnectionResetError("kit fault"))


class SqlFaultInjector:
    def __init__(self) -> None:
        self._armed: dict[Any, _Arm] = {}  # sync Engine -> armed fault
        self._installed: set[Any] = set()
        self._session_hooks = False

    # -- the kit's interface -----------------------------------------------------------------

    def arm(self, store: Any, point: str, *, kind: str = "error") -> None:
        engine = store._engine.sync_engine
        if engine not in self._installed:
            self._installed.add(engine)
            event.listen(engine, "before_cursor_execute", self._on_execute)
        if not self._session_hooks:
            self._session_hooks = True
            event.listen(Session, "before_commit", self._before_commit)
            event.listen(Session, "after_commit", self._after_commit)
        self._armed[engine] = _Arm(point, kind)

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
            self._session_hooks = False
        self._installed.clear()

    # -- SQLAlchemy listeners ----------------------------------------------------------------

    def _on_execute(self, conn: Any, cursor: Any, statement: str, *_: Any) -> None:
        armed = self._armed.get(conn.engine)
        if armed is None or not statement.lstrip().upper().startswith(_WRITES):
            return
        armed.writes += 1
        if armed.point == "mid_write" and armed.writes >= 2:
            del self._armed[conn.engine]
            raise armed.failure()

    def _armed_for(self, session: Session) -> tuple[Any, _Arm] | None:
        try:
            engine = session.get_bind()
        except Exception:
            return None
        armed = self._armed.get(engine)
        return (engine, armed) if armed is not None and armed.writes else None

    def _before_commit(self, session: Session) -> None:
        found = self._armed_for(session)
        if found and found[1].point in ("before_commit", "mid_write"):
            del self._armed[found[0]]
            raise found[1].failure()

    def _after_commit(self, session: Session) -> None:
        found = self._armed_for(session)
        if found and found[1].point == "after_commit":
            del self._armed[found[0]]
            raise found[1].failure()
