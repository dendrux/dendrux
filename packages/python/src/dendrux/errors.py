"""Public exception hierarchy for developers writing their own HTTP routes.

Methods like :meth:`Agent.submit_tool_results`, :meth:`Agent.submit_input`,
:meth:`Agent.submit_approval`, and :meth:`Agent.cancel_run` raise these
exceptions on precondition failures. Developers catch them and map to
HTTP status codes (or CLI prompts, or log lines) in their own code —
Dendrux never forces a status-code contract.

Suggested HTTP mappings:

    RunNotFoundError            -> 404
    RunNotPausedError           -> 409
    PauseStatusMismatchError    -> 409
    RunAlreadyClaimedError      -> 409
    RunAlreadyTerminalError     -> 409
    InvalidToolResultError      -> 400
    ReplayRunError              -> 400 (404 when reason == "not_found")
    PersistenceNotConfiguredError -> 500
    SchemaNotInitializedError   -> 500
    SkillMissingError           -> 409
    ContextLengthExceededError  -> 413 or 400
    SkillMismatchError          -> 409
    SqliteSchemaSyncError       -> 500 (startup: legacy SQLite file needs migration)

``SkillValidationError`` is raised while building a :class:`~dendrux.skills.Skill`
(a configuration error, not a request error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dendrux.types import RunStatus


class RunNotFoundError(LookupError):
    """Raised when ``run_id`` does not exist in the store."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Run '{run_id}' not found.")


class RunNotPausedError(RuntimeError):
    """Raised when a submit method is called on a run that is not paused.

    The run is either still executing, already terminal, or in a status
    that cannot accept a resume signal. Use :meth:`Agent.cancel_run` to
    stop an active run; terminal runs cannot be resumed.
    """

    def __init__(self, run_id: str, current_status: RunStatus) -> None:
        self.run_id = run_id
        self.current_status = current_status
        super().__init__(f"Run '{run_id}' is not paused (current status: {current_status.value}).")


class PauseStatusMismatchError(RuntimeError):
    """Raised when a submit method is called for the wrong pause type.

    Example: calling :meth:`Agent.submit_tool_results` on a run that is
    ``WAITING_HUMAN_INPUT`` rather than ``WAITING_CLIENT_TOOL``. The run
    is paused, but not for the kind of signal this method delivers.
    """

    def __init__(
        self,
        run_id: str,
        current_status: RunStatus,
        expected_status: RunStatus,
    ) -> None:
        self.run_id = run_id
        self.current_status = current_status
        self.expected_status = expected_status
        super().__init__(
            f"Run '{run_id}' is {current_status.value}, "
            f"but this method requires {expected_status.value}."
        )


class RunAlreadyClaimedError(RuntimeError):
    """Raised when a concurrent submit has already claimed this pause.

    Two submits raced; one won the CAS transition from paused to
    running. The loser receives this error. The run continues under
    the winning caller — retries should poll for completion, not
    re-submit.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"Run '{run_id}' was claimed by another submit. Poll for completion instead."
        )


class RunAlreadyTerminalError(RuntimeError):
    """Raised by submit methods when the run has already reached a terminal state.

    :meth:`Agent.cancel_run` does **not** raise this — cancellation on
    a terminal run is a silent no-op.
    """

    def __init__(self, run_id: str, current_status: RunStatus) -> None:
        self.run_id = run_id
        self.current_status = current_status
        super().__init__(
            f"Run '{run_id}' is already terminal ({current_status.value}); cannot resume."
        )


class InvalidToolResultError(ValueError):
    """Raised when submitted tool results do not match the pending set.

    Every result's ``tool_call_id`` must correspond to a pending call
    on the paused run; unknown or duplicate ids cause this error and no
    partial claim is made.
    """

    def __init__(self, run_id: str, detail: str) -> None:
        self.run_id = run_id
        self.detail = detail
        super().__init__(f"Invalid tool results for run '{run_id}': {detail}")


class PersistenceNotConfiguredError(RuntimeError):
    """Raised when an Agent method requires persistence but none is configured.

    Resume-family methods need a ``database_url`` or ``state_store`` on
    the agent. This is a configuration bug, not a runtime condition —
    it will raise on every call until the agent is reconfigured.
    """

    def __init__(self) -> None:
        super().__init__(
            "This operation requires persistence. Pass database_url or state_store "
            "to the Agent constructor."
        )


class SchemaNotInitializedError(RuntimeError):
    """Raised when the configured database has no Dendrux tables yet.

    SQLite auto-creates its schema on first use, so this never fires
    there. Postgres (and other backends) require an explicit migration
    step before the first run. The message points at the CLI commands
    that set the schema up.
    """

    def __init__(self) -> None:
        super().__init__(
            "Dendrux tables not found in the configured database. "
            "Run migrations before the first run:\n\n"
            "    export DENDRUX_DATABASE_URL=<your-database-url>\n"
            "    dendrux db migrate\n\n"
            "SQLite auto-creates its schema; Postgres and other backends "
            "require this step. Check status with 'dendrux db status'."
        )


class ReplayRunError(ValueError):
    """Raised when a run passed via ``agent.run(runs=...)`` cannot be replayed.

    ``reason`` is one of ``"not_found"``, ``"tenant_mismatch"``,
    ``"not_terminal"``, ``"duplicate"``. Raised before any run is created
    or any model is called.
    """

    def __init__(self, run_id: str, reason: str, detail: str) -> None:
        self.run_id = run_id
        self.reason = reason
        super().__init__(f"Cannot replay run '{run_id}': {detail}")


class SkillValidationError(ValueError):
    """Raised when a skill's content or metadata is invalid.

    ``field`` names the offending field (``"name"``, ``"files"``,
    ``"frontmatter"``, ...) when one applies.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


@dataclass(frozen=True)
class SkillDigestMismatch:
    """One skill whose content changed between run creation and continuation."""

    name: str
    expected_digest: str
    actual_digest: str


class SkillContinuityError(RuntimeError):
    """Raised when a paused run's skills are not what the run was created with.

    Base class of :class:`SkillMissingError` and :class:`SkillMismatchError`.
    The run is left in its paused status. ``missing`` and ``mismatched``
    are sorted by skill name.
    """

    def __init__(
        self,
        run_id: str,
        *,
        missing: Iterable[str] = (),
        mismatched: Iterable[SkillDigestMismatch] = (),
    ) -> None:
        self.run_id = run_id
        self.missing: tuple[str, ...] = tuple(sorted(missing))
        self.mismatched: tuple[SkillDigestMismatch, ...] = tuple(
            sorted(mismatched, key=lambda m: m.name)
        )
        parts: list[str] = []
        if self.missing:
            parts.append("missing skills: " + ", ".join(self.missing))
        if self.mismatched:
            parts.append(
                "changed skills: "
                + ", ".join(
                    f"{m.name} (expected {m.expected_digest}, got {m.actual_digest})"
                    for m in self.mismatched
                )
            )
        detail = "; ".join(parts) if parts else "skills do not match"
        super().__init__(
            f"Run '{run_id}' cannot continue: {detail}. "
            "Attach the same skill content the run was created with."
        )


class SkillMissingError(SkillContinuityError):
    """Raised when a skill the run was created with is not attached to the agent.

    Any digest mismatches found in the same check are reported in ``mismatched``.
    """


class SkillMismatchError(SkillContinuityError):
    """Raised when every skill is attached but some have different content."""


class ContextLengthExceededError(RuntimeError):
    """Raised when an LLM endpoint rejects a request as too large for the model's context window.

    Raised by the OpenAI-compatible providers (and therefore
    ``OpenRouterProvider``) for both batch and streaming calls. The message
    never contains prompt content; the original SDK exception is chained as
    ``__cause__`` for debugging.

    Attributes:
        status_code: HTTP status the endpoint returned (``None`` for an
            in-stream error event).
        provider_code: Provider error code when one was sent
            (for example ``"context_length_exceeded"``), else ``None``.
        model: Model name the request targeted, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        model: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.provider_code = provider_code
        self.model = model
        super().__init__(message)


class SqliteSchemaSyncError(RuntimeError):
    """A SQLite database cannot be upgraded in place to the current schema.

    Raised at startup when an older Dendrux SQLite file is missing a column
    that SQLite cannot add (NOT NULL without a constant default), when an
    index the schema needs collides with an unrelated object, or when an
    ``ALTER`` fails. The message names the table, column or index. Run
    ``dendrux db migrate`` against the database, or recreate the file.
    """


__all__ = [
    "ContextLengthExceededError",
    "InvalidToolResultError",
    "PauseStatusMismatchError",
    "PersistenceNotConfiguredError",
    "ReplayRunError",
    "RunAlreadyClaimedError",
    "RunAlreadyTerminalError",
    "RunNotFoundError",
    "RunNotPausedError",
    "SchemaNotInitializedError",
    "SkillContinuityError",
    "SkillDigestMismatch",
    "SkillMismatchError",
    "SkillMissingError",
    "SkillValidationError",
    "SqliteSchemaSyncError",
]
