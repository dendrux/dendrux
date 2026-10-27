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
    BudgetConfigurationError,
    ContentFilteredError,
    ContextLengthExceededError,
    GovernanceMismatchError,
    InvalidToolResultError,
    ModelRefusalError,
    NoteLimitExceededError,
    PauseStatusMismatchError,
    PersistenceNotConfiguredError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderStreamIncompleteError,
    ProviderTimeoutError,
    ProviderUnavailableError,
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
    TenantScopeRequiredError,
)
from dendrux.loops.single import SingleCall
from dendrux.media import Image, ToolOutput
from dendrux.pricing import ModelPricing, PriceTable
from dendrux.runtime.context import DelegationDepthExceededError, aheartbeat, heartbeat
from dendrux.runtime.runner import run
from dendrux.runtime.state import UNSCOPED
from dendrux.runtime.sweep import sweep
from dendrux.skills import Skill, SkillExecutionRequest, SkillExecutionResult, SkillExecutor
from dendrux.tool import tool
from dendrux.tools import tool_pool_stats
from dendrux.types import (
    ApprovalDecision,
    BeforeToolHook,
    Budget,
    CreateRunResult,
    ErrorInfo,
    GovernanceEventType,
    IdempotencyConflictError,
    ImagePart,
    ImageRef,
    Limits,
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
    "BudgetConfigurationError",
    "ContentFilteredError",
    "ContextBlock",
    "ContextLengthExceededError",
    "CreateRunResult",
    "DelegationDepthExceededError",
    "ErrorInfo",
    "GovernanceEventType",
    "GovernanceMismatchError",
    "IdempotencyConflictError",
    "Image",
    "ImagePart",
    "ImageRef",
    "InvalidToolResultError",
    "Limits",
    "ModelPricing",
    "ModelRefusalError",
    "NoteLimitExceededError",
    "PauseOutcome",
    "PauseStatusMismatchError",
    "PersistenceNotConfiguredError",
    "PriceTable",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderStreamIncompleteError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
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
    "TenantScopeRequiredError",
    "TextPart",
    "ToolCallContext",
    "ToolDecision",
    "ToolOutput",
    "ToolResultContext",
    "ToolResultTransform",
    "UNSCOPED",
    "aheartbeat",
    "heartbeat",
    "run",
    "sweep",
    "tool",
    "tool_pool_stats",
]
