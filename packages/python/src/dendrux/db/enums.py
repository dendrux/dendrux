"""Database enums — status values persisted to the DB.

Separate from types.RunStatus because DB enums are frozen once migrated
(adding a value = new Alembic migration). We only include statuses that
the runtime can actually produce.

Note: These are stored as VARCHAR strings in the DB, not DB-level enums,
so adding values here is safe without a migration for SQLite. Postgres
deployments using native enums would need an ALTER TYPE migration.
"""

from enum import StrEnum

from dendrux.types import RunStatus


class AgentRunStatus(StrEnum):
    """Status of an agent run in the database."""

    PENDING = RunStatus.PENDING.value
    RUNNING = RunStatus.RUNNING.value
    SUCCESS = RunStatus.SUCCESS.value
    ERROR = RunStatus.ERROR.value
    MAX_ITERATIONS = RunStatus.MAX_ITERATIONS.value
    CANCELLED = RunStatus.CANCELLED.value
    WAITING_CLIENT_TOOL = RunStatus.WAITING_CLIENT_TOOL.value
    WAITING_HUMAN_INPUT = RunStatus.WAITING_HUMAN_INPUT.value
    WAITING_APPROVAL = RunStatus.WAITING_APPROVAL.value
    PAUSED = RunStatus.PAUSED.value
