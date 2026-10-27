"""Resuming reads an index without loading stored trace content."""

import pytest
from sqlalchemy import event, insert

from dendrux.db.models import ReactTrace
from tests.integration.test_tenant_run_control import _pause


@pytest.mark.parametrize("stream", [False, True])
async def test_resume_does_not_fetch_trace_payloads(store, engine, monkeypatch, stream):
    paused = await _pause("input", store, "A")
    async with store._session_factory() as session:
        await session.execute(
            insert(ReactTrace),
            [
                dict(
                    id=f"t{i:025d}",
                    agent_run_id=paused.run_id,
                    role="user",
                    content="large trace" * 100,
                    order_index=100 + i,
                )
                for i in range(2500)
            ],
        )
        await session.commit()
    rows_returned = 0
    original = store.get_traces
    queries = []

    async def count_traces(*args, **kwargs):
        nonlocal rows_returned
        rows = await original(*args, **kwargs)
        rows_returned += len(rows)
        return rows

    monkeypatch.setattr(store, "get_traces", count_traces)

    def capture(conn, cursor, statement, parameters, context, executemany):
        if "react_traces" in statement and statement.lstrip().upper().startswith("SELECT"):
            queries.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        if stream:
            async for _ in paused.agent.resume_stream(
                paused.run_id, user_input="yes", tenant_id="A"
            ):
                pass
        else:
            await paused.agent.resume(paused.run_id, user_input="yes", tenant_id="A")
        assert rows_returned == 0
        assert len(queries) == 1
        assert "content" not in queries[0]
        assert "LIMIT" in queries[0]
        assert await store.get_next_trace_index(paused.run_id) > 2600
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await paused.agent.close()
