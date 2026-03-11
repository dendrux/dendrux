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

# ---------------------------------------------------------------------------
# Messages — what goes into and out of the LLM
# ---------------------------------------------------------------------------


class Role(StrEnum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """A single message in the conversation.

    This is the universal format — strategies convert to/from provider-specific
    formats internally. Developers and Dendrite internals only deal with Message.
    """

    role: Role
    content: str
    name: str | None = None  # Tool name when role=TOOL
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Actions — what the agent decides to do at each step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """Agent wants to execute a tool."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


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


# Union of all possible actions an agent can take
Action = ToolCall | Finish | Clarification


# ---------------------------------------------------------------------------
# AgentStep — one iteration of the agent loop
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool execution types
# ---------------------------------------------------------------------------


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
    """Result of executing a tool."""

    name: str
    result: Any
    success: bool = True
    error: str | None = None
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# LLM types — what the provider returns
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Run types — the final output
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

# RunInput can be a simple string or a dict with structured input
RunInput = str | dict[str, Any]
