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
    PersistenceNotConfiguredError -> 500
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


__all__ = [
    "InvalidToolResultError",
    "PauseStatusMismatchError",
    "PersistenceNotConfiguredError",
    "RunAlreadyClaimedError",
    "RunAlreadyTerminalError",
    "RunNotFoundError",
    "RunNotPausedError",
]
