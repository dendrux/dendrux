"""Database enums — status values persisted to the DB.

Separate from types.RunStatus because DB enums are frozen once migrated
(adding a value = new Alembic migration). We only include statuses that
the runtime can actually produce.

Note: These are stored as VARCHAR strings in the DB, not DB-level enums,
so adding values here is safe without a migration for SQLite. Postgres
deployments using native enums would need an ALTER TYPE migration.
"""

from enum import StrEnum


class AgentRunStatus(StrEnum):
    """Status of an agent run in the database."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    MAX_ITERATIONS = "max_iterations"
    CANCELLED = "cancelled"
    WAITING_CLIENT_TOOL = "waiting_client_tool"
    WAITING_HUMAN_INPUT = "waiting_human_input"
    WAITING_APPROVAL = "waiting_approval"
