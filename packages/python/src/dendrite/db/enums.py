"""Database enums — status values persisted to the DB.

Separate from types.RunStatus because DB enums are frozen once migrated
(adding a value = new Alembic migration). We only include statuses that
the runtime can actually produce.

Sprint 2 statuses: PENDING, RUNNING, SUCCESS, ERROR, MAX_ITERATIONS, CANCELLED.
The WAITING_* statuses (WAITING_CLIENT_TOOL, WAITING_HUMAN_INPUT, WAITING_APPROVAL)
are added via migration in Sprint 3 when pause/resume exists.
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
