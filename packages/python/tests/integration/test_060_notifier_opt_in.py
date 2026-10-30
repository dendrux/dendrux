"""Notification SQL is absent unless the writer explicitly enables a channel."""

import pytest
from sqlalchemy import event

from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import generate_ulid


@pytest.mark.parametrize("options", [{}, {"notify_channel": "dendrux_events"}])
async def test_agent_forwards_writer_opt_in(engine, options):
    from dendrux import Agent

    agent = Agent(
        prompt="p",
        database_url=engine.url.render_as_string(hide_password=False),
        database_options=options,
    )
    try:
        writer = await agent._resolve_state_store()
        assert writer.notify_channel == options.get("notify_channel")
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "options", [{}, {"notify_channel": None}, {"notify_channel": "custom_events"}]
)
async def test_writer_notify_is_opt_in(engine, options):
    writer = SQLAlchemyStateStore(engine, **options)
    run = generate_ulid()
    await writer.create_run(run, "notification-opt-in")
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await writer.save_run_event(run, event_type="llm.completed")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    enabled = engine.dialect.name == "postgresql" and options.get("notify_channel") is not None
    assert sum("pg_notify" in sql for sql in statements) == int(enabled), statements
    assert len(await writer.get_run_events(run)) == 1
