"""Configured payload limits are compared with the database's statement capacity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime.stores._limits_fit import ensure_limits_fit, required_statement_bytes
from dendrux.store import SQLiteStateStore
from dendrux.types import Limits, LLMResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

MIB = 1024 * 1024


def test_default_limits_fit_the_smallest_shipped_server_default() -> None:
    # MariaDB 10.11 ships a 16 MiB packet; MySQL 8.4 ships 64 MiB.
    assert required_statement_bytes(Limits()) < 16 * MIB


def test_every_limits_bounded_payload_reserves_json_and_sql_escaping() -> None:
    base = required_statement_bytes(Limits())
    assert required_statement_bytes(Limits(max_input_bytes=2 * MIB)) - base == 12 * MIB
    assert required_statement_bytes(Limits(max_tool_result_bytes=3 * MIB)) - base == 24 * MIB
    # Metadata gets the same worst case: ASCII escaping of a Unicode value needs 3.5x, and
    # a control character 6x.
    assert required_statement_bytes(Limits(max_metadata_bytes=128 * 1024)) - base == 12 * 64 * 1024
    # Only the largest text payload shares a statement with the scaffolding.
    both = required_statement_bytes(Limits(max_input_bytes=2 * MIB, max_tool_result_bytes=2 * MIB))
    assert both == required_statement_bytes(Limits(max_input_bytes=2 * MIB))


def test_a_five_mib_unicode_metadata_limit_is_not_accepted_by_a_16_mib_server() -> None:
    # Reviewer scenario: with 1-byte text limits an allowed value needs 17,500,015 bytes
    # on the wire; the earlier 2x estimate (10,551,308) accepted a 16 MiB server.
    limits = Limits(max_input_bytes=1, max_tool_result_bytes=1, max_metadata_bytes=5 * MIB)
    assert required_statement_bytes(limits) >= 17_500_015
    assert required_statement_bytes(limits) > 16 * MIB


class _Reporting:
    def __init__(self, capacity: int | None) -> None:
        self.capacity = capacity
        self.calls = 0

    async def statement_capacity(self) -> int | None:
        self.calls += 1
        return self.capacity


async def test_a_store_without_the_seam_or_without_a_limit_is_accepted() -> None:
    await ensure_limits_fit(object(), Limits(max_input_bytes=10**12))
    await ensure_limits_fit(_Reporting(None), Limits(max_input_bytes=10**12))


async def test_limits_that_fit_exactly_are_accepted() -> None:
    needed = required_statement_bytes(Limits())
    await ensure_limits_fit(_Reporting(needed), Limits())


async def test_refusal_names_both_numbers_and_logs_a_stable_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    limits = Limits(max_tool_result_bytes=8 * MIB)
    needed = required_statement_bytes(limits)
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as caught:
        await ensure_limits_fit(_Reporting(16 * MIB), limits)
    message = str(caught.value)
    assert str(needed) in message
    assert str(16 * MIB) in message
    assert "max_allowed_packet" in message
    [record] = [r for r in caplog.records if getattr(r, "code", None) == "store_statement_limit"]
    assert (record.bytes, record.limit) == (needed, 16 * MIB)  # type: ignore[attr-defined]


def report(monkeypatch: pytest.MonkeyPatch, reporting: _Reporting) -> None:
    """Make every SQLiteStateStore report ``reporting``'s capacity (a store class opts in)."""

    async def statement_capacity(self: SQLiteStateStore) -> int | None:
        return await reporting.statement_capacity()

    monkeypatch.setattr(SQLiteStateStore, "statement_capacity", statement_capacity, raising=False)


@pytest.fixture
async def store() -> AsyncIterator[SQLiteStateStore]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    from dendrux.db._sqlite_sync import create_sqlite_schema

    await create_sqlite_schema(engine)
    try:
        yield SQLiteStateStore(engine)
    finally:
        await engine.dispose()


def _agent(store: SQLiteStateStore, **kwargs: Any) -> Agent:
    return Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="ok"), LLMResponse(text="ok")]),
        state_store=store,
        strict_tenancy=False,
        **kwargs,
    )


async def test_the_agent_checks_once_per_store_and_then_runs(
    store: SQLiteStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting = _Reporting(64 * MIB)
    report(monkeypatch, reporting)
    agent = _agent(store)
    assert (await agent.run("a")).answer == "ok"
    assert (await agent.run("b")).answer == "ok"
    assert reporting.calls == 1


@pytest.mark.parametrize(
    "entry",
    ["run", "resume", "submit_input", "submit_approval", "cancel_run", "get_result"],
)
async def test_every_entry_path_refuses_before_touching_the_database(
    store: SQLiteStateStore, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting = _Reporting(MIB)
    report(monkeypatch, reporting)
    agent = _agent(store)
    with pytest.raises(ValueError, match="max_allowed_packet"):
        if entry == "run":
            await agent.run("a")
        elif entry == "resume":
            await agent.resume("run-1")
        elif entry == "submit_input":
            await agent.submit_input("run-1", "x")
        elif entry == "submit_approval":
            await agent.submit_approval("run-1", approved=True)
        elif entry == "cancel_run":
            await agent.cancel_run("run-1")
        else:
            await agent.get_result("run-1", tenant_id=None)
    assert await store.list_runs() == []
    # A refusal is not remembered as success: the next call is refused as well.
    with pytest.raises(ValueError, match="max_allowed_packet"):
        await agent.run("again")
    assert reporting.calls == 2


async def test_stream_reports_the_refusal_as_a_run_error(
    store: SQLiteStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    report(monkeypatch, _Reporting(MIB))
    async with _agent(store).stream("a") as events:
        seen = [event async for event in events]
    errors = [event for event in seen if event.type.value == "run_error"]
    assert len(errors) == 1
    assert "max_allowed_packet" in errors[0].error
    assert await store.list_runs() == []


async def test_resume_stream_raises_on_first_iteration(
    store: SQLiteStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    report(monkeypatch, _Reporting(MIB))
    with pytest.raises(ValueError, match="max_allowed_packet"):
        async with _agent(store).resume_stream("run-1") as events:
            async for _ in events:
                pass
    assert await store.list_runs() == []


async def test_smaller_configured_limits_fit_a_small_server(
    store: SQLiteStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    limits = Limits(max_input_bytes=64 * 1024, max_tool_result_bytes=64 * 1024)
    reporting = _Reporting(required_statement_bytes(limits))
    report(monkeypatch, reporting)
    assert (await _agent(store, limits=limits).run("a")).answer == "ok"
