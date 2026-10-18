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
    SchemaOutOfDateError,
    SkillContinuityError,
    SkillDigestMismatch,
    SkillMismatchError,
    SkillMissingError,
    SkillValidationError,
)
from dendrux.loops.single import SingleCall
from dendrux.media import Image, ToolOutput
from dendrux.pricing import ModelPricing, PriceTable
from dendrux.runtime.context import DelegationDepthExceededError, aheartbeat, heartbeat
from dendrux.runtime.runner import run
from dendrux.runtime.sweep import sweep
from dendrux.skills import Skill, SkillExecutionRequest, SkillExecutionResult, SkillExecutor
from dendrux.tool import tool
from dendrux.types import (
    ApprovalDecision,
    BeforeToolHook,
    Budget,
    CreateRunResult,
    GovernanceEventType,
    IdempotencyConflictError,
    ImagePart,
    ImageRef,
    PauseOutcome,
    ReplaySummary,
    RunAlreadyActiveError,
    StopReason,
    StructuredOutputValidationError,
    TextPart,
    ToolCallContext,
    ToolDecision,
    ToolResultContext,
    ToolResultTransform,
)

__all__ = [
    "Agent",
    "ApprovalDecision",
    "BeforeToolHook",
    "Budget",
    "ContextBlock",
    "ContextLengthExceededError",
    "CreateRunResult",
    "DelegationDepthExceededError",
    "GovernanceEventType",
    "IdempotencyConflictError",
    "Image",
    "ImagePart",
    "ImageRef",
    "InvalidToolResultError",
    "ModelPricing",
    "PauseOutcome",
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
    "SchemaOutOfDateError",
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
    "StopReason",
    "StructuredOutputValidationError",
    "TextPart",
    "ToolCallContext",
    "ToolDecision",
    "ToolOutput",
    "ToolResultContext",
    "ToolResultTransform",
    "aheartbeat",
    "heartbeat",
    "run",
    "sweep",
    "tool",
]
