"""``_PendingReader`` edge paths: legacy stores and database errors.

- A custom store without ``get_active_pause_sequence`` finds no active pause
  when the run is not paused, when its event reader cannot be inspected, or
  when the last pause was resumed (pause arguments then stay hidden).
- The Postgres summary only falls back to a whole-document read for a
  conversion error; any other database error propagates unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.exc import DBAPIError

from dendrux.errors import PersistenceError
from dendrux.runtime._pending import _PendingReader
from dendrux.runtime.stores.sql.postgres import PostgresStateStore


class _LegacyStore:
    def __init__(self, status: str, events: list[str]) -> None:
        self.status = status
        self.events = events

    async def get_run(self, run_id: str) -> Any:
        return SimpleNamespace(status=self.status, tenant_id=None)

    async def get_run_events(
        self, run_id: str, *, limit: int | None = None, event_types: Any = None
    ) -> list[Any]:
        return [SimpleNamespace(event_type=t, sequence_index=i) for i, t in enumerate(self.events)]


async def test_legacy_store_not_paused_has_no_active_pause() -> None:
    reader = _PendingReader(_LegacyStore("success", ["run.paused"]))
    assert await reader._active_pause_sequence("r", None) is None


async def test_legacy_store_resumed_tail_has_no_active_pause() -> None:
    reader = _PendingReader(_LegacyStore("waiting_approval", ["run.paused", "run.resumed"]))
    assert await reader._active_pause_sequence("r", None) is None
    paused = _PendingReader(_LegacyStore("waiting_approval", ["run.resumed", "run.paused"]))
    assert await paused._active_pause_sequence("r", None) == 1


async def test_legacy_store_with_uninspectable_reader_fails_closed() -> None:
    store = _LegacyStore("waiting_approval", ["run.paused"])
    store.get_run_events = object()  # type: ignore[assignment]  # not a signature
    assert await _PendingReader(store)._active_pause_sequence("r", None) is None


class _DriverError(Exception):
    def __init__(self, sqlstate: str | None) -> None:
        super().__init__("driver text")
        self.sqlstate = sqlstate


def _dbapi_error(sqlstate: str | None) -> DBAPIError:
    return DBAPIError("SELECT 1", {}, _DriverError(sqlstate))


@pytest.mark.parametrize(
    ("code", "expected"), [("22P05", True), ("22P02", True), ("40001", False), (None, False)]
)
def test_only_json_conversion_errors_fall_back(code: str | None, expected: bool) -> None:
    assert (
        PostgresStateStore.json_conversion_error(
            object.__new__(PostgresStateStore), _dbapi_error(code)
        )
        is expected
    )


async def test_other_database_errors_propagate() -> None:
    class _Session:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def execute(self, stmt: Any) -> Any:
            raise _dbapi_error("40001")

    state = object.__new__(PostgresStateStore)
    state._session_factory = _Session
    with pytest.raises(PersistenceError):
        await state.get_pause_summary("r", tenant_id=None)
