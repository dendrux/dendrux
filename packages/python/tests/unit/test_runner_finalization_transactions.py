"""Atomic finalization pins SQL statements and transaction grouping byte-for-byte."""

import json
from pathlib import Path

import pytest
from sqlalchemy import event

from dendrux.runtime import runner
from dendrux.types import RunEvent, RunEventType
from tests._fixtures.storage import pin_storage_format
from tests.unit.test_runner_failure_save import entry_case, observer
from tests.unit.test_runner_failure_save import store as store  # noqa: F401


@pytest.mark.parametrize("entry", ["failure", "task", "run_stream", "resume_stream"])
@pytest.mark.parametrize("won", [False, True])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_finalization_transaction_bytes(store, monkeypatch, entry, won, storage_format):
    await pin_storage_format(store._engine, storage_format)
    stream = iterator = None
    if entry.endswith("stream"):
        agent, invoke = await entry_case(store, entry)

        async def loop_stream(**kwargs):
            yield RunEvent(type=RunEventType.TEXT_DELTA, text="partial")

        monkeypatch.setattr(agent._loop, "run_stream", loop_stream)
        stream = invoke(observer())
        iterator = aiter(stream)
        async for item in iterator:
            if item.type == RunEventType.TEXT_DELTA:
                break
        run_id = stream.run_id
    else:
        run_id = "finalization-snapshot"
        await store.create_run(run_id, agent_name="snapshot", input_data={})
    if not won:
        await store.finalize_run(
            run_id, status="success", answer="winner", expected_current_status="running"
        )

    statements = []

    def begin(conn):
        statements.append("BEGIN")

    def commit(conn):
        statements.append("COMMIT")

    def rollback(conn):
        statements.append("ROLLBACK")

    def execute(conn, cursor, statement, parameters, context, executemany):
        statements.append(" ".join(statement.split()))

    engine = store._engine.sync_engine
    listeners = [
        ("begin", begin),
        ("commit", commit),
        ("rollback", rollback),
        ("before_cursor_execute", execute),
    ]
    for name, listener in listeners:
        event.listen(engine, name, listener)
    try:
        if entry == "failure":
            await runner._persist_run_failure(store, run_id, RuntimeError("failure"))
        elif entry == "task":
            await runner._persist_task_cancellation(store, run_id, interrupted=True)
        else:
            await stream.aclose()
            await iterator.aclose()
    finally:
        for name, listener in listeners:
            event.remove(engine, name, listener)
    wire = json.dumps(statements, indent=2) + "\n"
    expected = (
        Path(__file__).with_name("fixtures")
        / f"finalization_sql_{entry}_{won}_{storage_format}.json"
    )
    assert wire == expected.read_text()
