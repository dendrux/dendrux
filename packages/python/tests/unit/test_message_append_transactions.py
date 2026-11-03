"""Format-2 append SQL and transaction boundaries captured before extraction."""

import json
from pathlib import Path

import pytest
from sqlalchemy import event

from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.state import ClaimResult, claimed_store
from dendrux.types import AgentStep, Clarification, Message, Role, ToolCall, generate_ulid
from tests._fixtures.storage import pin_storage_format
from tests.unit.test_message_append_plan import live_data, pending
from tests.unit.test_runner_failure_save import store as store  # noqa: F401


@pytest.mark.parametrize("case", ["calls", "result", "clarification", "final"])
async def test_message_append_transaction_bytes(store, case):
    await pin_storage_format(store._engine, 2)
    run_id = generate_ulid()
    await store.create_run(run_id, agent_name="snapshot", input_data={})
    r = PersistenceRecorder(
        claimed_store(store, run_id, ClaimResult(0, 2, None, None)), run_id, initial_order_index=10
    )
    assert r._format2
    r._loop_state = {"v": 1, "pending": [], "phase": "model_next"}
    message = Message(Role.ASSISTANT, "answer")
    if case == "calls":
        message = Message(
            Role.ASSISTANT,
            "calls",
            tool_calls=[ToolCall("server", id="a"), ToolCall("client", id="b")],
        )
        r._loop_source = lambda: live_data()
    elif case == "result":
        r._loop_state["pending"] = [pending("a", 3)]
        message = Message(Role.TOOL, "result", call_id="a", name="server")
    elif case == "clarification":
        r._loop_source = lambda: live_data(AgentStep(None, Clarification("question")))
    statements = []
    listeners = [
        ("begin", lambda conn: statements.append("BEGIN")),
        ("commit", lambda conn: statements.append("COMMIT")),
        ("rollback", lambda conn: statements.append("ROLLBACK")),
        (
            "before_cursor_execute",
            lambda conn, cursor, statement, parameters, context, executemany: statements.append(
                " ".join(statement.split())
            ),
        ),
    ]
    engine = store._engine.sync_engine
    for name, listener in listeners:
        event.listen(engine, name, listener)
    try:
        await r.on_message_appended(run_id, message, 1)
    finally:
        for name, listener in listeners:
            event.remove(engine, name, listener)
    wire = json.dumps(statements, indent=2) + "\n"
    expected = Path(__file__).with_name("fixtures") / f"message_append_{case}.json"
    assert wire == expected.read_text()
