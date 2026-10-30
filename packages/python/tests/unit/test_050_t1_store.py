"""0.5.0: store behaviour — tenant-scoped idempotency, store-allocated
event indexes, the atomic note cap, RunStore tenant scoping (B11), list-view
projection, UTC filter normalisation and B13.

Runs on file SQLite (independent connections). The Postgres races are in
``tests/integration/test_050_t1_pg_races.py``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from functools import wraps
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent, tool
from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.db.models import Base
from dendrux.errors import NoteLimitExceededError, PersistenceError, RunNotFoundError
from dendrux.llm.mock import MockLLM
from dendrux.runtime.state import UNSCOPED, SQLAlchemyStateStore
from dendrux.store import RunStore
from dendrux.types import IdempotencyConflictError, Limits, LLMResponse, RunStatus, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM, tool_calls


@pytest.fixture
async def engine(tmp_path: Any) -> Any:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")

    @event.listens_for(eng.sync_engine, "connect")
    def _fk(dbapi_conn: Any, _rec: Any) -> None:
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def state(engine: Any) -> SQLAlchemyStateStore:
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def store(engine: Any) -> RunStore:
    return RunStore.from_engine(engine)


def _answers(*texts: str) -> MockLLM:
    return MockLLM([LLMResponse(text=t) for t in texts])


# ---------------------------------------------------------------------------
# tenant-scoped idempotency
# ---------------------------------------------------------------------------


class TestTenantIdempotency:
    async def test_same_key_in_two_tenants_creates_two_runs(self, state: Any) -> None:
        """Tenant isolation: tenant B must not get tenant A's run or answer."""
        llm = _answers("A secret balance", "B balance", "unused")
        agent = Agent(
            strict_tenancy=False, name="bank", prompt="p", provider=llm, state_store=state
        )
        a = await agent.run("What is my balance?", tenant_id="tenantA", idempotency_key="req-42")
        b = await agent.run("What is my balance?", tenant_id="tenantB", idempotency_key="req-42")
        assert a.run_id != b.run_id
        assert b.answer == "B balance"
        assert llm.calls_made == 2

    async def test_different_input_in_another_tenant_is_not_a_conflict(self, state: Any) -> None:
        agent = Agent(
            strict_tenancy=False,
            name="bank",
            prompt="p",
            provider=_answers("a", "b"),
            state_store=state,
        )
        await agent.run("q1", tenant_id="tenantA", idempotency_key="k")
        result = await agent.run("something else", tenant_id="tenantB", idempotency_key="k")
        assert result.answer == "b"

    async def test_same_tenant_replay_returns_the_cached_run(self, state: Any) -> None:
        llm = _answers("00000000000000000000000111", "never")
        agent = Agent(
            strict_tenancy=False, name="bank", prompt="p", provider=llm, state_store=state
        )
        first = await agent.run("q", tenant_id="t1", idempotency_key="k")
        again = await agent.run("q", tenant_id="t1", idempotency_key="k")
        assert again.run_id == first.run_id
        assert again.answer == "00000000000000000000000111"
        assert llm.calls_made == 1

    async def test_same_tenant_conflict_names_the_run(self, state: Any) -> None:
        agent = Agent(
            strict_tenancy=False,
            name="bank",
            prompt="p",
            provider=_answers("x", "y"),
            state_store=state,
        )
        first = await agent.run("q", tenant_id="t1", idempotency_key="k")
        with pytest.raises(IdempotencyConflictError) as caught:
            await agent.run("other", tenant_id="t1", idempotency_key="k")
        assert caught.value.run_id == first.run_id

    async def test_untenanted_and_tenanted_scopes_are_separate(self, state: Any) -> None:
        await state.create_run(
            "00000000000000000000000105", "a", idempotency_key="k", idempotency_fingerprint="f"
        )
        created = await state.create_run(
            "00000000000000000000000117",
            "a",
            tenant_id="t",
            idempotency_key="k",
            idempotency_fingerprint="f",
        )
        assert created.outcome == "created"
        assert (await state._check_idempotency("k", "f")).run_id == "00000000000000000000000105"
        assert (
            await state._check_idempotency("k", "f", tenant_id="t")
        ).run_id == "00000000000000000000000117"
        assert await state._check_idempotency("k", "f", tenant_id="other") is None

    async def test_unique_violation_in_the_same_tenant_resolves(self, state: Any) -> None:
        """A racing insert that loses on the unique index reads the winner back."""
        await state.create_run(
            "00000000000000000000000105",
            "a",
            tenant_id="t",
            idempotency_key="k",
            idempotency_fingerprint="f",
        )
        original = state._check_idempotency
        calls = {"n": 0}

        async def miss_first(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return None if calls["n"] == 1 else await original(*args, **kwargs)

        state._check_idempotency = miss_first
        result = await state.create_run(
            "00000000000000000000000117",
            "a",
            tenant_id="t",
            idempotency_key="k",
            idempotency_fingerprint="f",
        )
        assert (result.run_id, result.outcome) == ("00000000000000000000000105", "existing_active")

    async def test_legacy_global_unique_fails_closed_without_run_id(
        self, engine: Any, state: Any
    ) -> None:
        """A pre-0.5 file keeps a global UNIQUE: another tenant's key conflicts, never leaks."""
        async with engine.begin() as conn:
            await conn.exec_driver_sql(
                "CREATE UNIQUE INDEX ix_agent_runs_idempotency_key ON agent_runs (idempotency_key)"
            )
        await state.create_run(
            "00000000000000000000000105",
            "a",
            tenant_id="A",
            idempotency_key="k",
            idempotency_fingerprint="f",
        )
        with pytest.raises(IdempotencyConflictError) as caught:
            await state.create_run(
                "00000000000000000000000117",
                "a",
                tenant_id="B",
                idempotency_key="k",
                idempotency_fingerprint="f",
            )
        assert caught.value.run_id is None
        assert "00000000000000000000000105" not in str(caught.value)
        assert await state.get_run("00000000000000000000000117") is None

    async def test_legacy_inline_unique_file_after_sync(self, tmp_path: Any) -> None:
        from tests.unit.test_sqlite_schema_sync import _build_old_db

        db = tmp_path / "legacy.db"
        _build_old_db(db)
        eng = create_async_engine(f"sqlite+aiosqlite:///{db}")
        try:
            await create_sqlite_schema(eng)
            legacy = SQLAlchemyStateStore(eng)
            await legacy.create_run(
                "n1", "a", tenant_id="A", idempotency_key="k", idempotency_fingerprint="f"
            )
            same = await legacy.create_run(
                "n2", "a", tenant_id="A", idempotency_key="k", idempotency_fingerprint="f"
            )
            assert same.run_id == "n1"
            with pytest.raises(IdempotencyConflictError) as caught:
                await legacy.create_run(
                    "00000000000000000000000116",
                    "a",
                    tenant_id="B",
                    idempotency_key="k",
                    idempotency_fingerprint="f",
                )
            assert caught.value.run_id is None
        finally:
            await eng.dispose()

    async def test_unrelated_integrity_error_is_not_classified(self, state: Any) -> None:

        await state.create_run("dup", "a")
        with pytest.raises(PersistenceError):
            await state.create_run("dup", "a", idempotency_key="k", idempotency_fingerprint="f")

    async def test_store_rejects_empty_tenant(self, state: Any) -> None:
        with pytest.raises(ValueError, match="invalid_identifier"):
            await state.create_run("00000000000000000000000105", "a", tenant_id="")

    def test_violation_classifier(self) -> None:
        from dendrux.runtime.state import _is_idempotency_violation

        assert _is_idempotency_violation(
            Exception("UNIQUE constraint failed: index 'uq_agent_runs_tenant_idempotency_key'")
        )
        assert _is_idempotency_violation(
            Exception("UNIQUE constraint failed: agent_runs.idempotency_key")
        )
        assert _is_idempotency_violation(
            Exception(
                'duplicate key value violates unique constraint "ix_agent_runs_idempotency_key"'
            )
        )
        assert not _is_idempotency_violation(Exception("UNIQUE constraint failed: agent_runs.id"))
        assert not _is_idempotency_violation(Exception("the idempotency_key column is odd"))


# ---------------------------------------------------------------------------
# the store allocates event indexes
# ---------------------------------------------------------------------------


class TestEventAllocation:
    async def test_indexes_start_at_zero_and_are_returned(self, state: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        seqs = [
            await state.save_run_event("00000000000000000000000105", event_type=f"e{i}")
            for i in range(3)
        ]
        assert seqs == [0, 1, 2]
        assert [
            e.sequence_index for e in await state.get_run_events("00000000000000000000000105")
        ] == [0, 1, 2]

    async def test_runs_are_numbered_independently(self, state: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        await state.create_run("00000000000000000000000117", "a")
        assert await state.save_run_event("00000000000000000000000105", event_type="x") == 0
        assert await state.save_run_event("00000000000000000000000117", event_type="x") == 0
        assert await state.save_run_event("00000000000000000000000105", event_type="y") == 1

    async def test_fields_are_stored(self, state: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        await state.save_run_event(
            "00000000000000000000000105",
            event_type="t",
            iteration_index=3,
            correlation_id="c1",
            data={"k": [1]},
        )
        (ev,) = await state.get_run_events("00000000000000000000000105")
        assert (ev.iteration_index, ev.correlation_id, ev.data) == (3, "c1", {"k": [1]})
        assert ev.created_at is not None and ev.created_at.tzinfo is not None

    async def test_cas_paths_continue_the_same_sequence(self, state: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        await state.save_run_event("00000000000000000000000105", event_type="run.started")
        seq = await state.finalize_with_event(
            "00000000000000000000000105",
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
        )
        assert seq == 1
        assert await state.save_run_event("00000000000000000000000105", event_type="late") == 2

    async def test_concurrent_writers_get_distinct_contiguous_indexes(self, state: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        seqs = await asyncio.gather(
            *(
                state.save_run_event("00000000000000000000000105", event_type=f"e{i}")
                for i in range(25)
            )
        )
        assert sorted(seqs) == list(range(25))

    async def test_an_explicit_index_is_refused(self, state: Any) -> None:
        """0.4.1-style callers fail loudly instead of writing a caller-chosen index."""
        await state.create_run("00000000000000000000000105", "a")
        with pytest.raises(TypeError):
            await state.save_run_event(
                "00000000000000000000000105", event_type="x", sequence_index=7
            )  # type: ignore[call-arg]

    async def test_retried_write_never_hides_events_from_a_cursor(self, state: Any) -> None:
        """A transient failure: a transient failure used to commit an earlier index late."""

        @tool()
        async def a(x: int) -> int:
            return x

        @tool()
        async def b(x: int) -> int:
            await asyncio.sleep(0.01)
            return x

        original = state.save_run_event
        failed = {"done": False}

        @wraps(original)
        async def flaky(run_id: str, **kwargs: Any) -> int:
            if kwargs.get("event_type") == "tool.completed" and not failed["done"]:
                failed["done"] = True
                raise OperationalError("x", {}, Exception("database is locked"))
            return await original(run_id, **kwargs)

        state.save_run_event = flaky
        llm = ScriptedLLM(
            [tool_calls(("a", {"x": 1}, "p1"), ("b", {"x": 2}, "p2")), LLMResponse(text="ok")]
        )
        agent = Agent(
            strict_tenancy=False,
            name="o",
            prompt="p",
            tools=[a, b],
            provider=llm,
            state_store=state,
        )
        seen: list[int] = []
        stop = asyncio.Event()

        async def poll() -> None:
            cursor = None
            while not stop.is_set():
                runs = await state.list_runs()
                if runs:
                    for ev in await state.get_run_events(runs[0].id, after_sequence_index=cursor):
                        seen.append(ev.sequence_index)
                        cursor = ev.sequence_index
                await asyncio.sleep(0.002)

        poller = asyncio.create_task(poll())
        result = await agent.run("go")
        await asyncio.sleep(0.05)
        stop.set()
        await poller
        all_seq = [e.sequence_index for e in await state.get_run_events(result.run_id)]
        assert all_seq == list(range(len(all_seq)))
        assert sorted(set(all_seq) - set(seen)) == []

    async def test_resume_does_not_read_the_whole_event_log(self, state: Any) -> None:
        """No O(history) event read per resume: the store allocates the next index."""
        from tests.integration.test_tenant_run_control import _pause

        paused = await _pause("client", state, None)
        pending = (await state.get_pause_state(paused.run_id))["pending_tool_calls"][0]
        reads: list[Any] = []
        original = state.get_run_events

        async def spy(run_id: str, **kwargs: Any) -> Any:
            reads.append(kwargs)
            return await original(run_id, **kwargs)

        state.get_run_events = spy
        from dendrux.types import ToolResult

        await paused.agent.submit_tool_results(
            paused.run_id, [ToolResult(name=pending["name"], call_id=pending["id"], payload='"ok"')]
        )
        assert (await state.get_run(paused.run_id)).status == "success"
        assert all(r.get("limit") is not None or r.get("event_types") for r in reads)


# ---------------------------------------------------------------------------
# Note cap
# ---------------------------------------------------------------------------


class TestNoteCap:
    async def test_cap_raises_once_full(self, state: Any, caplog: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        for i in range(3):
            assert await state.add_note(
                "00000000000000000000000105", note_id=f"n{i}", text="t", max_notes=3
            )
        with caplog.at_level(logging.WARNING), pytest.raises(NoteLimitExceededError) as caught:
            await state.add_note(
                "00000000000000000000000105",
                note_id="00000000000000000000000116",
                text="t",
                max_notes=3,
            )
        assert (caught.value.run_id, caught.value.limit) == ("00000000000000000000000105", 3)
        assert isinstance(caught.value, ValueError)
        assert any(getattr(r, "code", None) == "note_limit" for r in caplog.records)
        assert len(await state.get_notes("00000000000000000000000105")) == 3

    async def test_terminal_missing_or_foreign_run_is_false_not_full(self, state: Any) -> None:
        await state.create_run("done", "a", tenant_id="t")
        await state.add_note("done", note_id="n0", text="t", max_notes=1)
        await state.finalize_run("done", status="success")
        assert not await state.add_note("done", note_id="n1", text="t", max_notes=1)
        assert not await state.add_note("missing", note_id="n2", text="t", max_notes=1)
        await state.create_run("live", "a", tenant_id="t")
        await state.add_note("live", note_id="00000000000000000000000116", text="t", max_notes=1)
        assert not await state.add_note(
            "live", note_id="n4", text="t", tenant_id="other", max_notes=1
        )

    async def test_no_cap_without_max_notes(self, state: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        for i in range(5):
            assert await state.add_note("00000000000000000000000105", note_id=f"n{i}", text="t")

    async def test_invalid_cap_rejected(self, state: Any) -> None:
        with pytest.raises(ValueError, match="max_notes"):
            await state.add_note("00000000000000000000000105", note_id="n", text="t", max_notes=0)

    async def test_concurrent_submitters_never_exceed_the_cap(self, state: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        results = await asyncio.gather(
            *(
                state.add_note(
                    "00000000000000000000000105", note_id=f"n{i:02d}", text="t", max_notes=5
                )
                for i in range(20)
            ),
            return_exceptions=True,
        )
        assert sum(r is True for r in results) == 5
        assert all(isinstance(r, NoteLimitExceededError) for r in results if r is not True)
        assert len(await state.get_notes("00000000000000000000000105")) == 5

    async def test_agent_submit_note_enforces_limits(self, state: Any) -> None:
        from tests.integration.test_tenant_run_control import _pause

        paused = await _pause("client", state, "t1")
        agent = Agent(
            strict_tenancy=False,
            provider=MockLLM([]),
            prompt="p",
            state_store=state,
            limits=Limits(max_notes_per_run=2),
        )
        await agent.submit_note(paused.run_id, "one", tenant_id="t1")
        await agent.submit_note(paused.run_id, "two", tenant_id="t1")
        with pytest.raises(NoteLimitExceededError):
            await agent.submit_note(paused.run_id, "three", tenant_id="t1")


# ---------------------------------------------------------------------------
# RunStore tenant scoping (B11)
# ---------------------------------------------------------------------------


async def _seed_two_tenants(state: Any) -> None:
    for run_id, tenant in (("ra", "A"), ("rb", "B"), ("rn", None)):
        await state.create_run(run_id, "agent", tenant_id=tenant, input_data={"q": run_id})
        await state.save_trace(run_id, "user", f"{run_id} secret", order_index=0)
        await state.save_run_event(run_id, event_type="run.started", data={"k": run_id})
        await state.save_tool_call(
            run_id,
            tool_call_id=f"tc-{run_id}",
            provider_tool_call_id=None,
            tool_name="t",
            target="server",
            params={"p": 1},
            result_payload='"00000000000000000000000102"',
            success=True,
            duration_ms=1,
            iteration_index=0,
        )
        await state.save_llm_interaction(
            run_id, iteration_index=0, usage=UsageStats(input_tokens=1, output_tokens=1)
        )
        await state.pause_run(
            run_id, status="waiting_client_tool", pause_data={"x": 1}, pii_mapping={"<P>": "v"}
        )
        await state.save_run_event(run_id, event_type="run.paused", data={"status": "w"})


class TestRunStoreScoping:
    async def test_foreign_run_reads_like_a_missing_one(self, state: Any, store: RunStore) -> None:
        await _seed_two_tenants(state)
        assert (await store.get_run("ra", tenant_id="A")).run_id == "ra"
        assert await store.get_run("ra", tenant_id="B") is None
        assert await store.get_run("ra", tenant_id=None) is None
        assert await store.get_events("ra", tenant_id="B") == []
        assert await store.get_traces("ra", tenant_id="B") == []
        assert await store.get_tool_invocations("ra", tenant_id="B") == []
        assert await store.get_llm_calls("ra", tenant_id="B") == []
        assert await store.get_pii_mapping("ra", tenant_id="B") == {}
        assert await store.get_pauses("ra", tenant_id="B") == []
        with pytest.raises(RunNotFoundError):
            await anext(aiter(store.stream_events("ra", tenant_id="B")))

    async def test_in_scope_reads_return_data(self, state: Any, store: RunStore) -> None:
        await _seed_two_tenants(state)
        assert len(await store.get_events("rb", tenant_id="B")) == 2
        assert (await store.get_traces("rb", tenant_id="B"))[0].content == "rb secret"
        assert len(await store.get_tool_invocations("rb", tenant_id="B")) == 1
        assert len(await store.get_llm_calls("rb", tenant_id="B")) == 1
        assert await store.get_pii_mapping("rb", tenant_id="B") == {"<P>": "v"}
        assert len(await store.get_pauses("rb", tenant_id="B")) == 1
        assert (await store.get_run("rn", tenant_id=None)).run_id == "rn"
        assert len(await store.get_events("rn", tenant_id=None)) == 2

    async def test_unscoped_reads_everything(self, state: Any, store: RunStore) -> None:
        await _seed_two_tenants(state)
        assert (await store.get_run("ra")).tenant_id == "A"
        assert len(await store.get_events("rb", tenant_id=UNSCOPED)) == 2

    async def test_list_and_count_tenant_semantics(self, state: Any, store: RunStore) -> None:
        await _seed_two_tenants(state)
        assert {r.run_id for r in await store.list_runs()} == {"ra", "rb", "rn"}
        assert {r.run_id for r in await store.list_runs(tenant_id=UNSCOPED)} == {"ra", "rb", "rn"}
        assert [r.run_id for r in await store.list_runs(tenant_id="A")] == ["ra"]
        assert [r.run_id for r in await store.list_runs(tenant_id=None)] == ["rn"]
        assert await store.count_runs() == 3
        assert await store.count_runs(tenant_id=None) == 1
        assert await state.count_runs(tenant_id="B") == 1
        assert {r.tenant_id for r in await store.list_runs()} == {"A", "B", None}

    async def test_invalid_scope_fails_closed(self, store: RunStore) -> None:
        for bad in ("", 7):
            with pytest.raises(ValueError, match="tenant_id"):
                await store.get_run("ra", tenant_id=bad)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="tenant_id"):
                await store.list_runs(tenant_id=bad)  # type: ignore[arg-type]

    async def test_custom_store_without_tenant_kwarg_is_scoped_by_lookup(self, state: Any) -> None:
        class Legacy:
            """A custom store whose reads predate tenant_id (and one hiding behind **kwargs)."""

            async def get_run(self, run_id: str) -> Any:
                return await state.get_run(run_id)

            async def get_run_events(self, run_id: str, **kwargs: Any) -> Any:
                assert "tenant_id" not in kwargs
                return await state.get_run_events(run_id, **kwargs)

        await _seed_two_tenants(state)
        legacy = RunStore(Legacy())  # type: ignore[arg-type]
        assert await legacy.get_events("ra", tenant_id="B") == []
        assert len(await legacy.get_events("ra", tenant_id="A")) == 2
        assert await legacy.get_run("ra", tenant_id="B") is None

    async def test_detail_and_summary_carry_tenant_and_error_info(
        self, state: Any, store: RunStore
    ) -> None:
        from dendrux.types import ErrorInfo

        info = ErrorInfo(category="timeout", retryable=False, safe_message="openai timeout")
        await state.create_run(
            "re", "agent", tenant_id="T", meta={"dendrux.error_info": info.to_dict()}
        )
        detail = await store.get_run("re")
        assert detail.tenant_id == "T"
        assert detail.error_info == info
        assert (await store.list_runs())[0].tenant_id == "T"
        await state.create_run("rx", "agent", meta={"dendrux.error_info": "garbage"})
        assert (await store.get_run("rx")).error_info is None

    async def test_pause_pair_reads_pause_id(self, state: Any, store: RunStore) -> None:
        await state.create_run("rp", "agent")
        await state.save_run_event("rp", event_type="run.paused", data={"pause_id": "P1"})
        assert (await store.get_pauses("rp"))[0].pause_id == "P1"

    async def test_list_runs_never_loads_pause_data(self, state: Any, engine: Any) -> None:
        await state.create_run("00000000000000000000000105", "a")
        await state.pause_run(
            "00000000000000000000000105", status="waiting_client_tool", pause_data={"big": "x"}
        )
        statements: list[str] = []

        @event.listens_for(engine.sync_engine, "before_cursor_execute")
        def _capture(_c: Any, _cur: Any, statement: str, *_a: Any) -> None:
            statements.append(statement)

        (record,) = await state.list_runs()
        assert record.id == "00000000000000000000000105"
        select_sql = next(s for s in statements if s.lstrip().upper().startswith("SELECT"))
        assert "pause_data" not in select_sql
        assert "pii_mapping" not in select_sql

    async def test_offset_is_capped(self, state: Any, store: RunStore) -> None:
        with pytest.raises(ValueError, match="offset"):
            await store.list_runs(offset=10_001)
        with pytest.raises(ValueError, match="offset"):
            await state.list_runs(offset=10_001)
        assert await store.list_runs(offset=10_000) == []

    async def test_started_filters_use_utc_instants(self, state: Any, store: RunStore) -> None:
        """a non-UTC offset compares the same instant on SQLite."""
        await state.create_run("00000000000000000000000105", "a")
        created = (await state.get_run("00000000000000000000000105")).created_at
        plus5 = dt.timezone(dt.timedelta(hours=5))
        just_before = (created - dt.timedelta(seconds=1)).astimezone(plus5)
        just_after = (created + dt.timedelta(seconds=1)).astimezone(plus5)
        assert [r.run_id for r in await store.list_runs(started_after=just_before)] == [
            "00000000000000000000000105"
        ]
        assert await store.list_runs(started_after=just_after) == []
        assert [r.id for r in await state.list_runs(started_before=just_after)] == [
            "00000000000000000000000105"
        ]
        assert await store.count_runs(started_before=just_before) == 0


# ---------------------------------------------------------------------------
# B13: retry raises RunNotFoundError for missing and foreign runs
# ---------------------------------------------------------------------------


class TestRetryNotFound:
    async def test_missing_and_foreign_raise_the_same_error(self, state: Any) -> None:
        agent = Agent(
            strict_tenancy=False,
            name="a",
            prompt="p",
            provider=_answers("x", "y", "z"),
            state_store=state,
        )
        done = await agent.run("q", tenant_id="A")
        assert done.status == RunStatus.SUCCESS
        with pytest.raises(RunNotFoundError) as missing:
            await agent.retry("nope", tenant_id="A")
        with pytest.raises(RunNotFoundError) as foreign:
            await agent.retry(done.run_id, tenant_id="B")
        assert isinstance(foreign.value, LookupError)
        assert type(missing.value) is type(foreign.value)
        assert str(foreign.value) == f"Run '{done.run_id}' not found."


class TestLegacyEventWriter:
    async def test_legacy_index_zero_writer_is_rejected_before_write(self) -> None:
        from dendrux.runtime.runner import _emit_event

        class Legacy:
            written = False

            async def save_run_event(self, run_id: str, sequence_index: int = 0, **kw: Any) -> int:
                self.written = True
                return sequence_index

        legacy = Legacy()
        with pytest.raises(TypeError, match="sequence_index"):
            await _emit_event(legacy, "00000000000000000000000102", "tick")
        assert not legacy.written

    @pytest.mark.parametrize("value", [None, True, "0", -1])
    def test_non_integer_event_result_warns_loudly(self, value: Any) -> None:
        from dendrux.runtime.state import validate_event_index

        with pytest.warns(RuntimeWarning, match="allocated non-negative integer"):
            validate_event_index(value)
