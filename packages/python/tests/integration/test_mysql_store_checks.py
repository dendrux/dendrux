"""Connection facts and JSON edge cases on each supported server."""

# ruff: noqa: F401, F811
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import AgentRun
from dendrux.runtime.stores.sql.mysql import MySQLStateStore
from tests.integration._mysql import mysql_engine, store


async def test_from_url_connection_settings_and_facts(mysql_engine):
    owned = MySQLStateStore.from_url(mysql_engine.url.render_as_string(hide_password=False))
    try:
        await owned.get_run("absent")
        facts = owned._dialect_facts
        assert facts.server_family == ("mariadb" if mysql_engine.dialect.is_mariadb else "mysql")
        assert facts.max_allowed_packet >= 1024 * 1024
        assert owned.supports_insert_returning is mysql_engine.dialect.is_mariadb
        async with owned._engine.connect() as conn:
            assert await conn.get_isolation_level() == "READ COMMITTED"
            assert await conn.scalar(text("SELECT @@session.time_zone")) == "+00:00"
    finally:
        await owned._engine.dispose()


async def test_borrowed_repeatable_read_refused(mysql_engine):
    borrowed = create_async_engine(mysql_engine.url, isolation_level="REPEATABLE READ")
    try:
        with pytest.raises(ValueError, match='isolation_level="READ COMMITTED"'):
            await MySQLStateStore(borrowed).get_run("absent")
    finally:
        await borrowed.dispose()


async def test_non_innodb_table_refused(mysql_engine):
    async with mysql_engine.begin() as conn:
        await conn.execute(text("ALTER TABLE dendrux_settings ENGINE=MyISAM"))
    with pytest.raises(ValueError, match="InnoDB"):
        await MySQLStateStore(mysql_engine).get_run("absent")


@pytest.mark.parametrize("doc", [None, {}, {"steps": []}, {"steps": [1]}])
async def test_json_negative_index_out_of_range(store, doc):
    await store.create_run("last", "agent", meta=doc)
    async with store._session_factory() as session:
        assert (
            await session.scalar(
                select(store.json_get(AgentRun.meta, ("steps", -2), "json")).where(
                    AgentRun.id == "last"
                )
            )
            is None
        )


async def test_tenant_variants_coexist_and_filter(store):
    from dendrux.types import generate_ulid

    pairs = {}
    for tenant in [None, "t", "t ", "T", "é", "e\u0301"]:
        run_id = generate_ulid()
        pairs[tenant] = run_id
        await store.create_run(
            run_id,
            "agent",
            tenant_id=tenant,
            idempotency_key="shared",
            idempotency_fingerprint="same",
        )
    for tenant, run_id in pairs.items():
        assert (
            await store.create_run(
                generate_ulid(),
                "agent",
                tenant_id=tenant,
                idempotency_key="shared",
                idempotency_fingerprint="same",
            )
        ).run_id == run_id
        for other, foreign_id in pairs.items():
            if other != tenant:
                assert await store.get_run(foreign_id, tenant_id=tenant) is None


@pytest.mark.parametrize("doc", [None, {}, {"other": 1}])
async def test_json_literal_patch_preserves_types(store, doc):
    from sqlalchemy import update

    from dendrux.runtime.stores.sql._types import AppendArray, SetFields

    await store.create_run("patch", "agent", meta=doc)
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == "patch")
            .values(
                meta=store.json_patch_expr(
                    AgentRun.meta, SetFields({"a.b": {"bool": True}, "items": []})
                )
            )
        )
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == "patch")
            .values(
                meta=store.json_patch_expr(AgentRun.meta, AppendArray(("items",), {"ok": True}))
            )
        )
        value = await session.scalar(select(AgentRun.meta).where(AgentRun.id == "patch"))
        assert value["a.b"] == {"bool": True}
        assert value["items"] == [{"ok": True}]


async def test_spike_json_truth_table(store):
    from sqlalchemy import JSON, literal, null

    cases = [
        (null(), False, False, None, None),
        (literal(None, JSON), False, False, None, None),
        (literal({}, JSON), False, False, None, None),
        (literal({"x": None}, JSON), True, False, None, None),
        (literal({"x": True}, JSON), False, True, True, None),
        (literal({"x": False}, JSON), False, False, False, None),
        (literal({"x": "true"}, JSON), False, False, None, "true"),
        (literal({"x": 1}, JSON), False, False, None, None),
        (literal({"x": []}, JSON), False, False, None, None),
    ]
    async with store._session_factory() as session:
        for col, is_null, contains, boolean, string in cases:
            row = (
                await session.execute(
                    select(
                        store.json_is_null(col, ("x",)),
                        store.json_contains(col, ("x",), True),
                        store.json_get(col, ("x",), "bool"),
                        store.json_get(col, ("x",), "str"),
                    )
                )
            ).one()
            assert tuple(row) == (is_null, contains, boolean, string)


async def test_json_identifier_built_before_first_connection(mysql_engine):
    from sqlalchemy import JSON, literal

    owned = MySQLStateStore.from_url(mysql_engine.url.render_as_string(hide_password=False))
    try:
        expression = owned.json_get(literal({"x": "t"}, JSON), ("x",), "str") == "t "
        async with owned._session_factory() as session:
            assert await session.scalar(select(expression)) is False
    finally:
        await owned._engine.dispose()


@pytest.mark.parametrize("operation", ["children", "cancel"])
async def test_frontier_membership_is_chunked(store, operation):
    ids = [f"absent-{i}" for i in range(1500)]
    if operation == "children":
        assert await store.get_children_page(ids, after=None, limit=10, tenant_id=None) == []
    else:
        assert await store.mark_cancel_requested(ids, tenant_id=None) == 0
    assert len(store.id_in(AgentRun.id, ids).clauses) == 2


async def test_unrelated_unique_constraint_is_not_absorbed(store):
    from sqlalchemy import func
    from sqlalchemy.exc import IntegrityError

    from dendrux.db.models import RunEvent
    from dendrux.types import generate_ulid

    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    await store.save_run_event(run_id, event_type="one")
    conflict = store._conflict(RunEvent.__table__, ["id"])
    plan = store.insert_if_absent_stmt(
        RunEvent.__table__,
        dict(
            id=generate_ulid(), agent_run_id=run_id, event_type="PRIVATE_SENTINEL", sequence_index=0
        ),
        conflict=conflict,
    )
    async with store._session_factory() as session:
        with pytest.raises(IntegrityError) as caught:
            await store._execute_insert_plan(session, plan, conflict)
        safe = store.classify_db_error(caught.value)
        assert safe.kind == "unique_violation" and safe.code == 1062
        assert safe.constraint == "uq_run_events_run_seq"
        assert "PRIVATE_SENTINEL" not in repr(safe)
        assert (
            await session.scalar(
                select(func.count()).select_from(RunEvent).where(RunEvent.agent_run_id == run_id)
            )
            == 1
        )
