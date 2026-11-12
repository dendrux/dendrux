"""Managed subagents (0.7, T4): pure recovery and sweep decisions, no database."""

from __future__ import annotations

import logging
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from dendrux.errors import ManagedStateError, ManagedUnsupportedError
from dendrux.runtime._reclaim import _ManagedPending, _refuse_managed_approval
from dendrux.runtime.stores.sql.base import _SweepScan
from dendrux.types import ToolCall, generate_ulid

CALL = ToolCall(name="reviewer", params={"task": "t"}, id="m1")
RUN = generate_ulid()


def _action(kind: str, call_id: str = "s1") -> Any:
    return SimpleNamespace(kind=kind, call_id=call_id)


@pytest.mark.parametrize("kind", ["dispatch", "observe", "replay", "resolve"])
def test_a_managed_batch_without_an_approval_decision_is_completed(kind: str) -> None:
    _refuse_managed_approval(_ManagedPending(CALL, False), RUN, [_action(kind)], {}, None)
    _refuse_managed_approval(_ManagedPending(CALL, True), RUN, [], {}, {})


@pytest.mark.parametrize("kind", ["approval", "approval_replay"])
def test_a_recorded_approval_decision_keeps_the_refusal(kind: str) -> None:
    with pytest.raises(ManagedUnsupportedError) as info:
        _refuse_managed_approval(_ManagedPending(CALL, False), RUN, [_action(kind)], {}, None)
    assert info.value.operation == "recover_managed_approval"


def test_a_gate_raised_during_recovery_keeps_the_refusal() -> None:
    with pytest.raises(ManagedUnsupportedError):
        _refuse_managed_approval(_ManagedPending(CALL, False), RUN, (), {}, {"s1": {}})


def test_a_bound_intent_for_a_pending_call_is_invalid_state() -> None:
    with pytest.raises(ManagedStateError) as info:
        _refuse_managed_approval(_ManagedPending(CALL, True), RUN, [], {"m1": object()}, None)
    assert info.value.reason == "binding_outstanding"


def test_no_pending_managed_call_is_never_refused() -> None:
    _refuse_managed_approval(None, RUN, [_action("approval")], {"m1": object()}, {"x": {}})


class _Store:
    def json_get(self, col: Any, path: Any, as_type: Any) -> Any:
        return None


def _row(version: Any, reason: Any, child: Any) -> Any:
    return SimpleNamespace(
        id=generate_ulid(), wait_version=version, wait_reason=reason, wait_child=child
    )


@pytest.mark.parametrize(
    ("version", "reason", "child", "skipped", "reported"),
    [
        (2, None, None, False, False),
        (None, "requested", None, False, False),
        (None, None, None, False, False),
        (3, "managed_wait", "c1", True, False),
        (3, "managed_wait", None, True, True),
        (3, "managed_wait", "", True, True),
        (2, "managed_wait", "c1", True, True),
        (3, "requested", "c1", True, True),
    ],
)
def test_waiting_rows_are_classified_from_their_projected_marks(
    caplog: Any, version: Any, reason: Any, child: Any, skipped: bool, reported: bool
) -> None:
    scan = _SweepScan(_Store(), waiting=True, older_than=timedelta(hours=1))  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        assert scan.skipped(_row(version, reason, child)) is skipped
    codes = [getattr(r, "code", None) for r in caplog.records]
    assert ("managed_state_invalid" in codes) is reported


@pytest.mark.parametrize("cursor", [None, "01OLD_RUNNING_ID"])
def test_sweep_cursor_accepts_legacy_running_positions(cursor: str | None) -> None:
    from dendrux.runtime.sweep import _SweepCursor

    progress = _SweepCursor.decode(cursor)
    assert progress.running == cursor and progress.waiting is None
    assert not progress.running_done and not progress.waiting_done
    assert progress.encode(with_waiting=False) == cursor


@pytest.mark.parametrize("done", [(True, False), (False, True), (False, False)])
def test_sweep_cursor_preserves_both_positions_and_completed_categories(done: Any) -> None:
    from datetime import UTC, datetime

    from dendrux.runtime.sweep import _SweepCursor

    position = ("paused", datetime(2026, 1, 1, 1, 2, 3, 456789, tzinfo=UTC), "wait-id")
    progress = _SweepCursor("running-id", position, *done)
    encoded = progress.encode(with_waiting=True)
    assert encoded is not None
    assert _SweepCursor.decode(encoded) == progress
    progress.running_done = progress.waiting_done = True
    assert progress.encode(with_waiting=True) is None


@pytest.mark.parametrize(
    "payload",
    [
        "bad-json",
        "[]",
        "[123,null,false,false]",
        "[null,null,1,false]",
        '[null,["running","2026-01-01","id"],false,false]',
        '[null,["paused","invalid-date","id"],false,false]',
        '[null,["paused","2026-01-01",1],false,false]',
    ],
)
def test_sweep_rejects_malformed_composite_cursors(payload: str) -> None:
    from dendrux.runtime.sweep import _SweepCursor

    with pytest.raises(ValueError, match="Invalid sweep cursor"):
        _SweepCursor.decode("sweep:v1:" + payload)


async def test_custom_waiting_sweep_keeps_its_legacy_call_signature() -> None:
    from unittest.mock import AsyncMock

    from dendrux.runtime.sweep import _sweep_waiting, _SweepCursor

    method = AsyncMock(return_value=[])
    custom = SimpleNamespace(sweep_abandoned_runs=method)
    progress = _SweepCursor(waiting_done=True)
    assert await _sweep_waiting(custom, timedelta(hours=1), {}, progress) == []
    method.assert_awaited_once_with(older_than=timedelta(hours=1))


@pytest.mark.parametrize("family", ["mysql", "mariadb", "sqlite", "postgresql"])
def test_waiting_continuation_uses_the_dialect_range_form(family: str) -> None:
    from datetime import UTC, datetime

    from sqlalchemy.dialects import mysql, postgresql, sqlite
    from sqlalchemy.dialects.mysql.mariadb import MariaDBDialect

    from dendrux.runtime.stores.sql.mysql import MySQLStateStore
    from dendrux.runtime.stores.sql.postgres import PostgresStateStore
    from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

    store_class, dialect = {
        "mysql": (MySQLStateStore, mysql.dialect()),
        "mariadb": (MySQLStateStore, MariaDBDialect()),
        "sqlite": (SQLiteStateStore, sqlite.dialect()),
        "postgresql": (PostgresStateStore, postgresql.dialect()),
    }[family]
    scan = _SweepScan(object.__new__(store_class), waiting=True, older_than=timedelta(hours=1))
    page = scan.page("paused", (datetime(2026, 1, 1, tzinfo=UTC), "middle"), 37)
    clause = str(page.whereclause.compile(dialect=dialect))
    assert "agent_runs.status =" in clause
    assert "agent_runs.status_changed_at <" in clause
    if family in {"mysql", "mariadb"}:
        assert " OR " in clause
        assert "agent_runs.status_changed_at =" in clause
        assert "agent_runs.id >" in clause
        assert "(agent_runs.status_changed_at, agent_runs.id)" not in clause
    else:
        assert "(agent_runs.status_changed_at, agent_runs.id) >" in clause
