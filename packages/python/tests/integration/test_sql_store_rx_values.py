"""One scalar metadata truth table for every supported SQL family."""

import pytest


@pytest.fixture
def backend_engine(engine):
    return engine


async def test_metadata_scalar_truth_table(store):
    documents = {
        "string": {"k": "abc"},
        "array": {"k": ["abc", "x"]},
        "integer": {"k": 1},
        "real": {"k": 1.0},
        "true": {"k": True},
        "false": {"k": False},
        "zero": {"k": 0},
        "null": {"k": None},
        "missing": {},
        "object": {"k": {"nested": "abc"}},
        "upper": {"k": "ABC"},
        "space": {"k": "abc "},
        "unicode": {"k": "é"},
        **{f"large-{n}": {"k": n} for n in (10**12, 2**62, 2**63 - 1, 2**64 - 1)},
    }
    for run_id, doc in documents.items():
        await store.create_run(run_id, "agent", meta=doc)
    for value, expected in [
        ("abc", {"string"}),
        (1, {"integer", "real"}),
        (1.0, {"integer", "real"}),
        (True, {"true"}),
        (False, {"false"}),
        (0, {"zero"}),
        (None, {"null"}),
        ("absent", set()),
        ("é", {"unicode"}),
        *((n, {f"large-{n}"}) for n in (10**12, 2**62, 2**63 - 1, 2**64 - 1)),
    ]:
        rows = await store.list_runs(metadata_filter={"k": value})
        assert {row.id for row in rows} == expected, (value, rows)
        assert await store.count_runs(metadata_filter={"k": value}) == len(expected)


async def test_pause_count_chunks_large_input(backend_engine, store):
    from sqlalchemy import event

    await store.create_run("present", "agent")
    await store.save_run_event("present", event_type="run.paused")
    lengths = []

    def capture(conn, cursor, sql, params, context, many):
        if sql.startswith("SELECT") and "count(" in sql and "run_events" in sql:
            lengths.append(len(params))

    event.listen(backend_engine.sync_engine, "before_cursor_execute", capture)
    try:
        ids = [f"absent-{i}" for i in range(2500)] + ["present", "present"]
        assert await store.count_pauses_per_run(ids) == {"present": 1}
        assert len(lengths) >= 3
        assert max(lengths) <= 1001
    finally:
        event.remove(backend_engine.sync_engine, "before_cursor_execute", capture)


async def test_foreign_note_does_not_wait_for_advisory_lock(backend_engine, store):
    import asyncio

    from sqlalchemy import text

    if backend_engine.dialect.name != "postgresql":
        pytest.skip("Postgres advisory note lock")
    from dendrux.runtime.state import _NOTE_LOCK_NAMESPACE

    await store.create_run("owner", "agent", tenant_id="owner")
    async with backend_engine.begin() as conn:
        await conn.execute(
            text("SELECT pg_advisory_xact_lock(:ns, hashtext('owner'))"),
            {"ns": _NOTE_LOCK_NAMESPACE},
        )
        assert (
            await asyncio.wait_for(
                store.add_note(
                    "owner", note_id="foreign", text="note", max_notes=1, tenant_id="foreign"
                ),
                10,
            )
            is False
        )


async def test_note_scope_precheck_only_on_postgres(backend_engine, store):
    from sqlalchemy import event

    await store.create_run("note", "agent", tenant_id="owner")
    selects = []

    def capture(conn, cursor, sql, params, context, many):
        if sql.startswith("SELECT") and "agent_runs" in sql:
            selects.append(sql)

    event.listen(backend_engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await store.add_note(
            "note", note_id="n", text="note", tenant_id="owner", max_notes=2
        )
    finally:
        event.remove(backend_engine.sync_engine, "before_cursor_execute", capture)
    expected = 1 if backend_engine.dialect.name in {"mysql", "mariadb", "postgresql"} else 0
    assert len(selects) == expected
