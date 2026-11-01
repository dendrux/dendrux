"""State-derived index requirements shared by startup and db check (B3)."""

import pytest
from sqlalchemy import create_engine, text

from dendrux.db import _schema_check as schema
from dendrux.db.models import Base


@pytest.mark.parametrize("state", ["pre-activation", "activating", "active"])
def test_b3_index_window(state):
    old = {"ix_react_traces_agent_run_id_order", "ix_run_events_agent_run_id_seq"}
    unique = {"uq_react_traces_run_order", "uq_run_events_run_seq", "uq_tool_calls_run_call"}
    baseline = set(schema.required_indexes()) - old - unique
    present = baseline | (unique if state == "active" else old)
    assert schema.compare_indexes(present, Base.metadata.tables, state=state) == []
    if state == "activating":
        assert schema.compare_indexes(baseline | unique, Base.metadata.tables, state=state) == []
        assert (
            schema.compare_indexes(baseline | unique | old, Base.metadata.tables, state=state) == []
        )
        assert schema.compare_indexes(baseline, Base.metadata.tables, state=state)
    if state == "active":
        assert schema.compare_indexes(present | old, Base.metadata.tables, state=state)


def test_invalid_settings_fail_closed():
    engine = create_engine("sqlite://")
    try:
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM dendrux_settings WHERE key='activate_step'"))
            with pytest.raises(RuntimeError, match="settings"):
                schema.read_schema_phase(conn)
    finally:
        engine.dispose()
