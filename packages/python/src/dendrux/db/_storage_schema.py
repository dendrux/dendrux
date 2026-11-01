"""Frozen format-2 DDL shared by migration, startup and activation."""

from __future__ import annotations

ADDITIONS = {
    "agent_runs": {
        "storage_format": "SMALLINT NOT NULL DEFAULT 1",
        "reclaim_attempts": "SMALLINT NOT NULL DEFAULT 0",
        "reclaim_limit": "SMALLINT NULL",
        "lease_expired_at": "TIMESTAMPTZ NULL",
        "last_reclaimed_at": "TIMESTAMPTZ NULL",
        "loop_state": "JSONB NULL",
        "trace_link": "VARCHAR(64) NULL",
    },
    "tool_calls": {
        "outcome": "VARCHAR(16) NOT NULL DEFAULT 'completed'",
        "idempotent": "BOOLEAN NOT NULL DEFAULT false",
        "replayed_at": "TIMESTAMPTZ NULL",
        "content_digest": "CHAR(64) NULL",
    },
    "react_traces": {
        "trace_kind": "VARCHAR(16) NOT NULL DEFAULT 'turn'",
        "content_digest": "CHAR(64) NULL",
    },
    "run_events": {"content_digest": "CHAR(64) NULL"},
}

# name -> (table, natural-key column, replaced index or None)
ACTIVATION_INDEXES = {
    "uq_react_traces_run_order": (
        "react_traces",
        "order_index",
        "ix_react_traces_agent_run_id_order",
    ),
    "uq_tool_calls_run_call": ("tool_calls", "tool_call_id", None),
    "uq_run_events_run_seq": ("run_events", "sequence_index", "ix_run_events_agent_run_id_seq"),
}


def index_ddl(name: str, *, concurrently: bool = False) -> str:
    """Exact unique-index definition; names come only from the frozen registry."""
    table, column, _ = ACTIVATION_INDEXES[name]
    concurrent = " CONCURRENTLY" if concurrently else ""
    return (
        f"CREATE UNIQUE INDEX{concurrent} IF NOT EXISTS {name} ON {table} (agent_run_id, {column})"
    )
