"""Dendrux — The runtime for agents that act in the real world."""

__version__ = "0.1.0a3"

from dendrux.agent import Agent
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

# bridge() requires FastAPI (pip install dendrux[bridge]).
# We must bind it at module level to shadow the subpackage name —
# __getattr__ can't override subpackage resolution.
# Only catch the specific missing-dependency error; real bridge bugs
# propagate normally.
try:
    from dendrux.bridge import bridge
except ImportError as _err:
    if "fastapi" in str(_err).lower() or "uvicorn" in str(_err).lower():
        _missing_err = _err

        def bridge(*args, **kwargs):  # type: ignore[misc,no-untyped-def]
            """Stub — raises when bridge extras are not installed."""
            raise ImportError(
                "bridge requires optional dependencies. Install with: pip install 'dendrux[bridge]'"
            ) from _missing_err
    else:
        raise

__all__ = [
    "Agent",
    "Budget",
    "CreateRunResult",
    "DelegationDepthExceededError",
    "GovernanceEventType",
    "IdempotencyConflictError",
    "RunAlreadyActiveError",
    "SingleCall",
    "StructuredOutputValidationError",
    "bridge",
    "run",
    "sweep",
    "tool",
]
