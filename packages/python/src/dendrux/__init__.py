"""Dendrux — The runtime for agents that act in the real world."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("dendrux")
except PackageNotFoundError:  # editable install before metadata exists
    __version__ = "0.0.0+dev"

from dendrux.agent import Agent
from dendrux.errors import (
    InvalidToolResultError,
    PauseStatusMismatchError,
    PersistenceNotConfiguredError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.loops.single import SingleCall
from dendrux.runtime.context import DelegationDepthExceededError
from dendrux.runtime.runner import run
from dendrux.runtime.sweep import sweep
from dendrux.tool import tool
from dendrux.types import (
    Budget,
    CreateRunResult,
    GovernanceEventType,
    IdempotencyConflictError,
    RunAlreadyActiveError,
    StructuredOutputValidationError,
)

__all__ = [
    "Agent",
    "Budget",
    "CreateRunResult",
    "DelegationDepthExceededError",
    "GovernanceEventType",
    "IdempotencyConflictError",
    "InvalidToolResultError",
    "PauseStatusMismatchError",
    "PersistenceNotConfiguredError",
    "RunAlreadyActiveError",
    "RunAlreadyClaimedError",
    "RunAlreadyTerminalError",
    "RunNotFoundError",
    "RunNotPausedError",
    "SingleCall",
    "StructuredOutputValidationError",
    "run",
    "sweep",
    "tool",
]
