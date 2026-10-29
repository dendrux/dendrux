"""Status writer inventory and provider-chain diagnostic boundaries."""

import ast
import inspect
import json
import logging
from dataclasses import dataclass

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dendrux.db.models import AgentRun
from dendrux.errors import ProviderTimeoutError
from dendrux.loops import _helpers
from dendrux.loops.base import BaseNotifier
from dendrux.notifiers.composite import CompositeNotifier
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.runtime import state
from dendrux.runtime.context import safe_exception_info

PRIVATE = "CHAIN-PRIVATE-SENTINEL-765"


def test_status_writer_inventory():
    tree = ast.parse(inspect.getsource(state))
    writers = {}
    direct = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Nested retry closures belong to the surrounding public operation.
        if function.name.startswith("_attempt") or function.name == "attempt":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "_status_values":
                    writers[function.name] = writers.get(function.name, 0) + 1
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "values"
                    and any(k.arg == "status" for k in node.keywords)
                ):
                    direct.append(function.name)
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "AgentRun"
                    and any(k.arg == "status" for k in node.keywords)
                ):
                    direct.append(function.name)
            if (
                isinstance(node, ast.Dict)
                and function.name != "_status_values"
                and any(isinstance(k, ast.Constant) and k.value == "status" for k in node.keys)
            ):
                direct.append(function.name)
    assert not direct, f"status writes must use the transition helper: {direct}"
    assert writers == {
        "create_run": 1,
        "finalize_run": 1,
        "_cas_finalize": 1,
        "try_pause_run": 1,
        "pause_with_event": 1,
        "pause_run": 1,
        "claim_paused_run": 1,
        "submit_and_claim": 1,
        "release_run_claim": 1,
        "_sweep_runs": 1,
        "_scan_expired_runs": 1,
        "reclaim_run": 1,
    }, f"audit every status writer: {writers}"
    # Heartbeats must not change any indexed columns, including predicates/expressions.
    indexed = {col.name for index in AgentRun.__table__.indexes for col in index.columns}
    assert not indexed & {"updated_at", "last_progress_at"}, indexed
    assert AgentRun.__table__.c.status_changed_at.nullable is False


class SelfCopyError(TypeError):
    def __copy__(self):
        return self


@dataclass(frozen=True)
class FrozenSecondaryError(TypeError):
    message: str

    def __copy__(self):
        return type(self)(self.message)


@pytest.mark.parametrize("kind", [TypeError, SelfCopyError, FrozenSecondaryError])
def test_secondary_failure_keeps_own_stack_without_provider_chain(caplog, kind):
    error = ProviderTimeoutError(
        message=PRIVATE, provider="fixture", timeout_reason="call_deadline"
    )
    try:
        raise error
    except ProviderTimeoutError:
        try:
            raise kind("secondary bug")
        except TypeError as secondary:
            original = secondary.__context__
            logging.getLogger("dendrux.test").error(
                "finalize failed", exc_info=safe_exception_info()
            )
            assert secondary.__context__ is original
    assert PRIVATE not in caplog.text
    assert f"{kind.__name__}: secondary bug" in caplog.text, "keep secondary type and message"
    assert "raise kind" in caplog.text, "keep secondary own stack"
    assert "During handling" not in caplog.text


@pytest.mark.parametrize("sink", ["helper_run", "helper_llm", "composite_run", "composite_llm"])
async def test_notifier_failure_does_not_leak_provider_chain(caplog, sink):
    class Broken(BaseNotifier):
        async def on_run_failed(self, *args, **kwargs):
            raise ValueError("secondary notifier bug")

        async def on_llm_call_failed(self, *args, **kwargs):
            raise ValueError("secondary notifier bug")

    error = ProviderTimeoutError(
        message=PRIVATE, provider="fixture", timeout_reason="call_deadline"
    )
    notifier = CompositeNotifier([Broken()]) if sink.startswith("composite") else Broken()
    try:
        raise error
    except ProviderTimeoutError:
        if sink.endswith("run"):
            await _helpers.notify_run_failed(notifier, "run", error)
        else:
            await _helpers.notify_llm_failed(notifier, "run", 1, error)
    assert PRIVATE not in caplog.text, "notifier failure stacks must omit provider chains"
    assert "ValueError: secondary notifier bug" in caplog.text


async def test_wrapped_provider_error_span_has_only_own_stack():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    notifier = OpenTelemetryNotifier(tracer=provider.get_tracer("test"))
    error = ProviderTimeoutError(
        message=PRIVATE, provider="fixture", timeout_reason="call_deadline"
    )
    try:
        raise error
    except ProviderTimeoutError:
        try:
            raise TypeError("secondary span bug")
        except TypeError as secondary:
            await notifier.on_run_started("run")
            await notifier.on_run_failed("run", secondary)
    span = exporter.get_finished_spans()[0]
    serialized = json.dumps([dict(e.attributes) for e in span.events])
    assert PRIVATE not in serialized, "span exception stack must omit provider chain"
    event = span.events[0]
    assert event.name == "provider.error"
    assert event.attributes["dendrux.error.category"] == "timeout"
    assert event.attributes["dendrux.error.timeout_reason"] == "call_deadline"
    assert event.attributes["dendrux.error.safe_message"] == error.safe_message
    assert span.status.description == error.safe_message
    assert "dendrux.error.provider_code" not in event.attributes
    diagnostic = next(e for e in span.events if e.name == "exception")
    assert diagnostic.attributes["exception.type"] == "TypeError"
    stack = diagnostic.attributes["exception.stacktrace"]
    assert "test_wrapped_provider_error_span_has_only_own_stack" in stack
    assert __file__ in stack and "line " in stack
    assert "raise TypeError" not in stack and "raise error" not in stack
    assert "secondary span bug" not in serialized
    provider.shutdown()


def test_oracle_validates_additive_status_clock():
    from tests._helpers.v030_oracle import _strip_050_additions

    row, problem = _strip_050_additions(
        "agent_runs", {"status": "running", "status_changed_at": "<ts>"}
    )
    assert problem is None and "status_changed_at" not in row
    _, problem = _strip_050_additions("agent_runs", {"status_changed_at": None})
    assert problem is not None, "the new non-null clock must be validated before projection"


@pytest.mark.parametrize("concurrent", [False, True])
def test_sqlite_status_clock_backfill_is_atomic_and_preserves_old_rows(tmp_path, concurrent):
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor

    from sqlalchemy import create_engine

    from dendrux.db._sqlite_sync import create_sqlite_schema_sync
    from tests.unit.test_sqlite_schema_sync import _build_old_db

    path = tmp_path / "status-clock.db"
    _build_old_db(path)

    def upgrade():
        engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 10})
        try:
            with engine.connect() as conn:
                return create_sqlite_schema_sync(conn)
        finally:
            engine.dispose()

    if concurrent:
        with ThreadPoolExecutor(max_workers=4) as pool:
            changes = list(pool.map(lambda _: upgrade(), range(4)))
        assert sum(bool(change) for change in changes) == 1
    else:
        assert upgrade()
    with sqlite3.connect(path) as conn:
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(agent_runs)")}
        assert columns["status_changed_at"][3] == 1, "backfilled clock must be NOT NULL"
        assert (
            conn.execute(
                "SELECT count(*) FROM agent_runs WHERE status_changed_at = updated_at"
            ).fetchone()[0]
            == 2
        )
        assert conn.execute("SELECT count(*) FROM agent_runs").fetchone()[0] == 2
    assert upgrade() == []


def test_sweep_index_validation_error_does_not_claim_uniqueness(monkeypatch):
    import importlib.util
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace

    import sqlalchemy
    from alembic import op

    from dendrux.db import _index_check

    monkeypatch.setattr(_index_check, "index_matches", lambda *args: False)
    path = (
        Path(state.__file__).parents[1] / "db/migrations/versions/2026_10_25_bound_sweep_pages.py"
    )
    spec = importlib.util.spec_from_file_location("sweep_migration_wording", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        op, "get_bind", lambda: SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    )
    monkeypatch.setattr(op, "get_context", lambda: SimpleNamespace(autocommit_block=nullcontext))
    monkeypatch.setattr(op, "execute", lambda *args: None)
    monkeypatch.setattr(
        sqlalchemy,
        "inspect",
        lambda bind: SimpleNamespace(get_columns=lambda _: [{"name": "status_changed_at"}]),
    )
    monkeypatch.setattr(module, "_online_pg_clock", lambda: None)
    # This wording unit test stubs DDL; real bounded waits have PG coverage.
    monkeypatch.setattr(module, "build_index_concurrently", lambda *args, **kw: None)
    with pytest.raises(RuntimeError) as caught:
        module.upgrade()
    assert "uniqueness constraint" not in str(caught.value), (
        "sweep-index errors must not promise unrelated uniqueness"
    )
    assert "definition" in str(caught.value)


def test_sqlite_clock_and_backfill_roll_back_on_later_schema_failure(tmp_path, monkeypatch):
    import sqlite3

    from sqlalchemy import create_engine

    from dendrux.db import _sqlite_sync
    from tests.unit.test_sqlite_schema_sync import _build_old_db

    path = tmp_path / "rollback-clock.db"
    _build_old_db(path)

    def fail(conn, table, changes):
        assert (
            conn.exec_driver_sql(
                "SELECT count(*) FROM agent_runs WHERE status_changed_at=updated_at"
            ).scalar_one()
            == 2
        )
        raise RuntimeError("fixture schema write failure")

    monkeypatch.setattr(_sqlite_sync, "_reconcile_indexes", fail)
    engine = create_engine(f"sqlite:///{path}")
    try:
        with (
            engine.connect() as conn,
            pytest.raises(RuntimeError, match="fixture schema write failure"),
        ):
            _sqlite_sync.create_sqlite_schema_sync(conn)
    finally:
        engine.dispose()
    with sqlite3.connect(path) as conn:
        assert "status_changed_at" not in {
            row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")
        }
        assert conn.execute("SELECT count(*) FROM agent_runs").fetchone()[0] == 2


@pytest.mark.parametrize("missing_created", [False, True])
def test_sqlite_status_clock_coalesces_nullable_legacy_timestamps(
    tmp_path, monkeypatch, missing_created
):
    import sqlite3

    from sqlalchemy import create_engine

    from dendrux.db._sqlite_sync import create_sqlite_schema_sync
    from tests.unit import test_sqlite_schema_sync as legacy

    # The standard legacy fixture requires both timestamps. Also exercise a
    # permissive older/custom table, without weakening that existing fixture.
    monkeypatch.setitem(legacy._AGENT_RUNS_COLUMNS, "updated_at", "DATETIME")
    monkeypatch.setitem(legacy._AGENT_RUNS_COLUMNS, "created_at", "DATETIME")
    path = tmp_path / "nullable-clock.db"
    legacy._build_old_db(path)
    run_id = legacy._LEGACY_ROWS[0][0]
    with sqlite3.connect(path) as conn:
        created = conn.execute(
            "SELECT created_at FROM agent_runs WHERE id=?", (run_id,)
        ).fetchone()[0]
        conn.execute("UPDATE agent_runs SET updated_at=NULL WHERE id=?", (run_id,))
        if missing_created:
            conn.execute("UPDATE agent_runs SET created_at=NULL WHERE id=?", (run_id,))
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.connect() as conn:
            create_sqlite_schema_sync(conn)
    finally:
        engine.dispose()
    with sqlite3.connect(path) as conn:
        info = next(
            row
            for row in conn.execute("PRAGMA table_info(agent_runs)")
            if row[1] == "status_changed_at"
        )
        constant = conn.execute("SELECT " + info[4]).fetchone()[0]
        actual = conn.execute(
            "SELECT status_changed_at, updated_at, created_at FROM agent_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        assert info[3] == 1 and actual[0] is not None
        assert actual[0] == (constant if missing_created else created)
        assert actual[1] is None
        assert actual[2] == (None if missing_created else created)


def test_standard_legacy_schema_does_not_allow_null_timestamps(tmp_path):
    import sqlite3

    from tests.unit.test_sqlite_schema_sync import _build_old_db

    path = tmp_path / "required-clock.db"
    _build_old_db(path)
    with sqlite3.connect(path) as conn:
        for column in ("updated_at", "created_at"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"UPDATE agent_runs SET {column}=NULL")


def test_sqlite_status_clock_raw_default_roundtrips_like_orm_datetime(tmp_path):
    import datetime as dt

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from dendrux.db._sqlite_sync import create_sqlite_schema_sync
    from tests.unit.test_sqlite_schema_sync import _build_old_db

    path = tmp_path / "default-clock.db"
    _build_old_db(path)
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.connect() as conn:
            create_sqlite_schema_sync(conn)
        with Session(engine) as session:
            session.add(AgentRun(id="orm-clock", agent_name="clock", status="running"))
            session.commit()
        with engine.begin() as conn:
            # Intentionally omit status_changed_at so SQLite uses the server default.
            conn.exec_driver_sql("""
                INSERT INTO agent_runs
                    (id, agent_name, status, iteration_count, delegation_level,
                     total_input_tokens, total_output_tokens, created_at, updated_at)
                SELECT 'raw-clock', agent_name, status, iteration_count, delegation_level,
                       total_input_tokens, total_output_tokens, created_at, updated_at
                FROM agent_runs WHERE id='orm-clock'
            """)
            stored = conn.exec_driver_sql(
                "SELECT status_changed_at FROM agent_runs WHERE id='raw-clock'"
            ).scalar_one()
        with Session(engine) as session:
            raw = session.get(AgentRun, "raw-clock").status_changed_at
            normal = session.get(AgentRun, "orm-clock").status_changed_at
            assert isinstance(raw, dt.datetime) and isinstance(normal, dt.datetime)
            bind = engine.dialect.type_descriptor(AgentRun.status_changed_at.type).bind_processor(
                engine.dialect
            )
            assert stored == bind(raw), "default must use SQLAlchemy's SQLite datetime format"
            assert raw.tzinfo == normal.tzinfo, (
                "default and ORM rows must have consistent timezone semantics"
            )
    finally:
        engine.dispose()
