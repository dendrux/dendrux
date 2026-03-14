"""Core types for Dendrite.

These dataclasses define the shapes of data flowing through the agent loop.
Every module in Dendrite speaks this common language.

Data flow:
    User input → Agent → Strategy → LLM → LLMResponse → parse → AgentStep
                                                              ↓
                                                    ToolCall? → execute → ToolResult
                                                    Finish?   → RunResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ulid import ULID


def generate_ulid() -> str:
    """Generate a new ULID string for Dendrite-owned correlation IDs."""
    return str(ULID())


class Role(StrEnum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """Agent wants to execute a tool.

    Carries both a Dendrite-owned ID (stable across pause/resume/replay)
    and the provider's native ID (for building API requests back).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=generate_ulid)
    provider_tool_call_id: str | None = None


@dataclass(frozen=True)
class Message:
    """A single message in the conversation.

    This is the universal format — providers convert to/from their native
    formats internally. Developers and Dendrite internals only deal with Message.

    Role-dependent fields:
        tool_calls: Present on ASSISTANT messages when the LLM called tools
                    (NativeToolCalling strategy). None for text-only turns
                    and all PromptBasedReAct turns.
        call_id:    Present on TOOL messages. References ToolCall.id (Dendrite ULID)
                    from the corresponding ASSISTANT message.
        name:       Present on TOOL messages only. Cached convenience field for
                    debugging/logging — call_id is the authoritative identity.
    """

    role: Role
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role == Role.TOOL:
            if not self.name:
                raise ValueError("TOOL messages require name")
            if not self.call_id:
                raise ValueError("TOOL messages require call_id")
            if self.tool_calls is not None:
                raise ValueError("TOOL messages cannot have tool_calls")
        elif self.role == Role.ASSISTANT:
            if self.call_id is not None:
                raise ValueError("ASSISTANT messages cannot have call_id")
            if self.name is not None:
                raise ValueError("ASSISTANT messages cannot have name")
        else:
            if self.tool_calls is not None:
                raise ValueError(f"{self.role.value.upper()} messages cannot have tool_calls")
            if self.call_id is not None:
                raise ValueError(f"{self.role.value.upper()} messages cannot have call_id")
            if self.name is not None:
                raise ValueError(f"{self.role.value.upper()} messages cannot have name")


@dataclass(frozen=True)
class Finish:
    """Agent is done — has a final answer."""

    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Clarification:
    """Agent needs input from a human before continuing."""

    question: str
    options: list[str] = field(default_factory=list)


Action = ToolCall | Finish | Clarification


@dataclass(frozen=True)
class AgentStep:
    """The output of one ReAct iteration.

    A strategy parses the LLM response into this. The loop consumes it.
    This is the boundary between "talking to the LLM" and "executing actions."
    """

    reasoning: str | None
    action: Action
    raw_response: str | None = None  # Original LLM text (for tracing)
    meta: dict[str, Any] = field(default_factory=dict)  # extra_fields land here


class ToolTarget(StrEnum):
    """Where a tool executes."""

    SERVER = "server"  # Runs on the backend (default)
    CLIENT = "client"  # Shipped to client for execution
    HUMAN = "human"  # Requires human response
    AGENT = "agent"  # Delegated to a sub-agent


@dataclass(frozen=True)
class ToolDef:
    """Definition of a registered tool.

    Created by the @tool decorator. Used by strategies to describe tools
    to the LLM, and by the executor to dispatch tool calls.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for params
    target: ToolTarget = ToolTarget.SERVER
    parallel: bool = True
    priority: int = 0
    max_calls_per_run: int | None = None
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool.

    name is a cached convenience field — call_id is the authoritative
    identity for correlation (call_id → ToolCall.id → ToolCall.name).
    """

    name: str
    call_id: str  # References ToolCall.id (Dendrite-owned)
    payload: str  # Always JSON string — serialized once by execution engine
    success: bool = True
    error: str | None = None
    duration_ms: int = 0


@dataclass
class UsageStats:
    """Token usage from an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider.

    Anthropic, OpenAI, or any other provider all normalize to this.
    Strategies consume this — they never touch provider-specific APIs.
    """

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw: Any = None  # Full provider response for debugging
    usage: UsageStats = field(default_factory=UsageStats)


@dataclass(frozen=True)
class ProviderCapabilities:
    """What an LLM provider can do.

    Layers above check these flags instead of using isinstance.
    Strategy selection, streaming decisions, and tool handling
    all key off capabilities, not provider type.
    """

    supports_native_tools: bool = False
    supports_tool_call_ids: bool = False
    supports_streaming: bool = False
    supports_streaming_tool_deltas: bool = False
    supports_thinking: bool = False
    supports_multimodal: bool = False
    supports_system_prompt: bool = True
    supports_parallel_tool_calls: bool = False
    max_context_tokens: int | None = None


class RunStatus(StrEnum):
    """Status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_CLIENT_TOOL = "waiting_client_tool"
    WAITING_HUMAN_INPUT = "waiting_human_input"
    WAITING_APPROVAL = "waiting_approval"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class RunResult:
    """The final output of an agent run."""

    run_id: str
    status: RunStatus
    answer: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    iteration_count: int = 0
    usage: UsageStats = field(default_factory=UsageStats)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class StreamEventType(StrEnum):
    """Types of events emitted during streaming LLM responses."""

    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    DONE = "done"


@dataclass(frozen=True)
class StreamEvent:
    """A single event from a streaming LLM response.

    Providers yield these during complete_stream(). The loop and transport
    layer consume them — strategies never touch provider-specific streaming APIs.
    """

    type: StreamEventType
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    raw: Any = None


RunInput = str | dict[str, Any]
