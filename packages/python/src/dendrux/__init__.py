"""Dendrux — The runtime for agents that act in the real world."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("dendrux")
except PackageNotFoundError:  # editable install before metadata exists
    __version__ = "0.0.0+dev"

from dendrux.agent import Agent
from dendrux.context_blocks import ContextBlock
from dendrux.errors import (
    ContextLengthExceededError,
    InvalidToolResultError,
    PauseStatusMismatchError,
    PersistenceNotConfiguredError,
    ReplayRunError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
    SchemaNotInitializedError,
    SkillContinuityError,
    SkillDigestMismatch,
    SkillMismatchError,
    SkillMissingError,
    SkillValidationError,
)
from dendrux.loops.single import SingleCall
from dendrux.pricing import ModelPricing, PriceTable
from dendrux.runtime.context import DelegationDepthExceededError
from dendrux.runtime.runner import run
from dendrux.runtime.sweep import sweep
from dendrux.skills import Skill, SkillExecutionRequest, SkillExecutionResult, SkillExecutor
from dendrux.tool import tool
from dendrux.types import (
    Budget,
    CreateRunResult,
    GovernanceEventType,
    IdempotencyConflictError,
    ReplaySummary,
    RunAlreadyActiveError,
    StructuredOutputValidationError,
    ToolResultContext,
    ToolResultTransform,
)

__all__ = [
    "Agent",
    "Budget",
    "ContextBlock",
    "ContextLengthExceededError",
    "CreateRunResult",
    "DelegationDepthExceededError",
    "GovernanceEventType",
    "IdempotencyConflictError",
    "InvalidToolResultError",
    "ModelPricing",
    "PauseStatusMismatchError",
    "PersistenceNotConfiguredError",
    "PriceTable",
    "ReplayRunError",
    "ReplaySummary",
    "RunAlreadyActiveError",
    "RunAlreadyClaimedError",
    "RunAlreadyTerminalError",
    "RunNotFoundError",
    "RunNotPausedError",
    "SchemaNotInitializedError",
    "SingleCall",
    "Skill",
    "SkillContinuityError",
    "SkillDigestMismatch",
    "SkillExecutionRequest",
    "SkillExecutionResult",
    "SkillExecutor",
    "SkillMismatchError",
    "SkillMissingError",
    "SkillValidationError",
    "StructuredOutputValidationError",
    "ToolResultContext",
    "ToolResultTransform",
    "run",
    "sweep",
    "tool",
]
