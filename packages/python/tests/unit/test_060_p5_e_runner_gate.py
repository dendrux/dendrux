"""The runner functions check configured limits against the effective store themselves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

import dendrux
from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime import runner
from dendrux.store import SQLiteStateStore
from dendrux.types import LLMResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

MIB = 1024 * 1024
MESSAGE = "max_allowed_packet"


@pytest.fixture
async def store(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SQLiteStateStore]:
    from dendrux.db._sqlite_sync import create_sqlite_schema

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_sqlite_schema(engine)
    calls: list[int] = []

    async def statement_capacity(self: SQLiteStateStore) -> int | None:
        calls.append(1)
        return 1  # one byte: nothing fits

    monkeypatch.setattr(SQLiteStateStore, "statement_capacity", statement_capacity, raising=False)
    built = SQLiteStateStore(engine)
    built.capacity_calls = calls  # type: ignore[attr-defined]
    try:
        yield built
    finally:
        await engine.dispose()


def agent() -> Agent:
    return Agent(prompt="p", provider=MockLLM([LLMResponse(text="ok")]), strict_tenancy=False)


async def refused(call: Any) -> None:
    """The call is refused with the limit error, by raising or as the stream's run_error."""
    try:
        result = call()
        if hasattr(result, "__aenter__"):
            async with result as events:
                seen = [event async for event in events]
            errors = [e for e in seen if e.type.value == "run_error"]
            assert errors, "the stream finished without refusing"
            assert MESSAGE in errors[0].error
            return
        await result
    except ValueError as exc:
        assert MESSAGE in str(exc)
        return
    pytest.fail("the call was not refused")


async def test_dendrux_run_with_an_explicit_store_is_checked(store: SQLiteStateStore) -> None:
    a = agent()
    await refused(lambda: dendrux.run(a, provider=a.provider, user_input="x", state_store=store))
    assert store.capacity_calls  # type: ignore[attr-defined]
    assert await store.list_runs() == []


async def test_run_stream_with_an_explicit_store_is_checked(store: SQLiteStateStore) -> None:
    a = agent()
    await refused(
        lambda: runner.run_stream(a, provider=a.provider, user_input="x", state_store=store)
    )
    assert store.capacity_calls  # type: ignore[attr-defined]
    assert await store.list_runs() == []


async def test_retry_with_an_explicit_store_is_checked(store: SQLiteStateStore) -> None:
    a = agent()
    await refused(lambda: runner.retry("run-1", agent=a, provider=a.provider, state_store=store))
    assert store.capacity_calls  # type: ignore[attr-defined]


@pytest.mark.parametrize("entry", ["resume", "resume_with_input", "resume_stream"])
async def test_resume_entry_points_with_an_explicit_store_are_checked(
    store: SQLiteStateStore, entry: str
) -> None:
    a = agent()
    if entry == "resume":
        call = lambda: runner.resume("run-1", None, state_store=store, agent=a, provider=a.provider)  # noqa: E731
    elif entry == "resume_with_input":
        call = lambda: runner.resume_with_input(  # noqa: E731
            "run-1", "hello", state_store=store, agent=a, provider=a.provider
        )
    else:
        call = lambda: runner.resume_stream(  # noqa: E731
            "run-1", agent=a, provider=a.provider, state_store=store
        )
    await refused(call)
    assert store.capacity_calls  # type: ignore[attr-defined]


async def test_the_check_runs_once_per_store_across_runner_calls(
    store: SQLiteStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def roomy(self: SQLiteStateStore) -> int | None:
        store.capacity_calls.append(1)  # type: ignore[attr-defined]
        return 64 * MIB

    monkeypatch.setattr(SQLiteStateStore, "statement_capacity", roomy, raising=False)
    a = agent()
    await dendrux.run(
        a, provider=MockLLM([LLMResponse(text="ok")]), user_input="x", state_store=store
    )
    await dendrux.run(
        a, provider=MockLLM([LLMResponse(text="ok")]), user_input="y", state_store=store
    )
    assert len(store.capacity_calls) == 1  # type: ignore[attr-defined]
