"""Core types for Dendrux.

These dataclasses define the shapes of data flowing through the agent loop.
Every module in Dendrux speaks this common language.

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
    """Generate a new ULID string for Dendrux-owned correlation IDs."""
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

    Carries both a Dendrux-owned ID (stable across pause/resume/replay)
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
    formats internally. Developers and Dendrux internals only deal with Message.

    Role-dependent fields:
        tool_calls: Present on ASSISTANT messages when the LLM called tools
                    (NativeToolCalling strategy). None for text-only turns
                    and all PromptBasedReAct turns.
        call_id:    Present on TOOL messages. References ToolCall.id (Dendrux ULID)
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
    """Where a tool executes.

    Only SERVER and CLIENT are implemented. HUMAN and AGENT are reserved
    for future use — the loop currently treats them as CLIENT (pauses).
    """

    SERVER = "server"  # Runs on the backend (default)
    CLIENT = "client"  # Shipped to client for execution
    # Reserved — not yet implemented. Loop treats as CLIENT today.
    HUMAN = "human"
    AGENT = "agent"


@dataclass(frozen=True)
class ToolDef:
    """Definition of a registered tool.

    Created by the @tool decorator. Used by strategies to describe tools
    to the LLM, and by the executor to dispatch tool calls.

    Note: priority is reserved for future use — not enforced by the runtime.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for params
    target: ToolTarget = ToolTarget.SERVER
    parallel: bool = True  # Governs concurrent execution scheduling
    priority: int = 0  # Reserved — not enforced
    max_calls_per_run: int | None = None  # Enforced by ReActLoop — graceful limit
    timeout_seconds: float = 120.0  # Enforced by ReActLoop via asyncio.wait_for
    has_explicit_timeout: bool = False  # True when developer set timeout_seconds explicitly


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool.

    name is a cached convenience field — call_id is the authoritative
    identity for correlation (call_id → ToolCall.id → ToolCall.name).
    """

    name: str
    call_id: str  # References ToolCall.id (Dendrux-owned)
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
    # Adapter-boundary payloads — set by each provider, persisted as opaque JSON.
    # provider_request: the exact kwargs sent to the vendor API (e.g. Anthropic api_kwargs).
    # provider_response: the raw vendor response dict (e.g. response.model_dump()).
    provider_request: dict[str, Any] | None = None
    provider_response: dict[str, Any] | None = None


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
    """Status of an agent run.

    WAITING_APPROVAL is reserved for future use — not implemented.
    """

    PENDING = "pending"
    RUNNING = "running"
    WAITING_CLIENT_TOOL = "waiting_client_tool"
    WAITING_HUMAN_INPUT = "waiting_human_input"
    WAITING_APPROVAL = "waiting_approval"  # Reserved — not yet implemented
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


@dataclass
class PauseState:
    """Everything needed to resume a paused agent run.

    This is execution state, not observability state. It is persisted
    unredacted in pause_data (JSON column) and cleared on finalize.
    Redaction applies only to traces, tool_calls, and agent_runs
    answer/error/input — never to pause_data.
    """

    agent_name: str
    pending_tool_calls: list[ToolCall]
    history: list[Message]
    steps: list[AgentStep]
    iteration: int
    trace_order_offset: int
    usage: UsageStats
    pending_targets: dict[str, str] = field(default_factory=dict)  # tool_call_id → target

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for DB storage.

        Validates that the output is JSON-serializable. Raises TypeError
        if any field (e.g. custom meta values) cannot be serialized.
        """
        import json

        d = {
            "agent_name": self.agent_name,
            "pending_tool_calls": [_tool_call_to_dict(tc) for tc in self.pending_tool_calls],
            "pending_targets": self.pending_targets,
            "history": [_message_to_dict(m) for m in self.history],
            "steps": [_step_to_dict(s) for s in self.steps],
            "iteration": self.iteration,
            "trace_order_offset": self.trace_order_offset,
            "usage": _usage_to_dict(self.usage),
        }
        # Validate JSON safety — catch non-serializable meta values early
        try:
            json.dumps(d)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"PauseState contains non-JSON-serializable data: {e}. "
                f"Ensure all Message.meta and AgentStep.meta values are JSON-safe."
            ) from e
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PauseState:
        """Deserialize from a dict (loaded from DB)."""
        return cls(
            agent_name=data["agent_name"],
            pending_tool_calls=[_tool_call_from_dict(tc) for tc in data["pending_tool_calls"]],
            pending_targets=data.get("pending_targets", {}),
            history=[_message_from_dict(m) for m in data["history"]],
            steps=[_step_from_dict(s) for s in data["steps"]],
            iteration=data["iteration"],
            trace_order_offset=data["trace_order_offset"],
            usage=_usage_from_dict(data["usage"]),
        )


# ------------------------------------------------------------------
# PauseState serialization helpers
# ------------------------------------------------------------------


def _tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    return {
        "name": tc.name,
        "params": tc.params,
        "id": tc.id,
        "provider_tool_call_id": tc.provider_tool_call_id,
    }


def _tool_call_from_dict(d: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=d["name"],
        params=d.get("params", {}),
        id=d["id"],
        provider_tool_call_id=d.get("provider_tool_call_id"),
    )


def _message_to_dict(m: Message) -> dict[str, Any]:
    import json as _json

    d: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.name is not None:
        d["name"] = m.name
    if m.tool_calls is not None:
        d["tool_calls"] = [_tool_call_to_dict(tc) for tc in m.tool_calls]
    if m.call_id is not None:
        d["call_id"] = m.call_id
    if m.meta:
        # Validate meta is JSON-safe; coerce non-serializable values
        safe_meta: dict[str, Any] = {}
        for k, v in m.meta.items():
            try:
                _json.dumps(v)
                safe_meta[k] = v
            except (TypeError, ValueError):
                safe_meta[k] = str(v)
        d["meta"] = safe_meta
    return d


def _message_from_dict(d: dict[str, Any]) -> Message:
    tool_calls = None
    if "tool_calls" in d:
        tool_calls = [_tool_call_from_dict(tc) for tc in d["tool_calls"]]
    return Message(
        role=Role(d["role"]),
        content=d["content"],
        name=d.get("name"),
        tool_calls=tool_calls,
        call_id=d.get("call_id"),
        meta=d.get("meta", {}),
    )


def _action_to_dict(action: Action) -> dict[str, Any]:
    if isinstance(action, ToolCall):
        return {"type": "tool_call", **_tool_call_to_dict(action)}
    if isinstance(action, Finish):
        return {"type": "finish", "answer": action.answer, "meta": action.meta}
    if isinstance(action, Clarification):
        return {"type": "clarification", "question": action.question, "options": action.options}
    raise TypeError(f"Unknown action type: {type(action)}")


def _action_from_dict(d: dict[str, Any]) -> Action:
    t = d["type"]
    if t == "tool_call":
        return _tool_call_from_dict(d)
    if t == "finish":
        return Finish(answer=d["answer"], meta=d.get("meta", {}))
    if t == "clarification":
        return Clarification(question=d["question"], options=d.get("options", []))
    raise ValueError(f"Unknown action type: {t!r}")


def _step_to_dict(s: AgentStep) -> dict[str, Any]:
    import json as _json

    d: dict[str, Any] = {
        "reasoning": s.reasoning,
        "action": _action_to_dict(s.action),
        "raw_response": s.raw_response,
    }
    # Serialize meta — handle known non-JSON types explicitly
    safe_meta: dict[str, Any] = {}
    for k, v in s.meta.items():
        if k == "all_tool_calls":
            safe_meta[k] = [_tool_call_to_dict(tc) for tc in v]
        else:
            # Validate each value is JSON-serializable
            try:
                _json.dumps(v)
                safe_meta[k] = v
            except (TypeError, ValueError):
                # Coerce non-serializable values to string representation
                safe_meta[k] = str(v)
    if safe_meta:
        d["meta"] = safe_meta
    return d


def _step_from_dict(d: dict[str, Any]) -> AgentStep:
    meta = d.get("meta", {})
    if "all_tool_calls" in meta:
        meta = dict(meta)
        meta["all_tool_calls"] = [_tool_call_from_dict(tc) for tc in meta["all_tool_calls"]]
    return AgentStep(
        reasoning=d.get("reasoning"),
        action=_action_from_dict(d["action"]),
        raw_response=d.get("raw_response"),
        meta=meta,
    )


def _usage_to_dict(u: UsageStats) -> dict[str, Any]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
        "cost_usd": u.cost_usd,
    }


def _usage_from_dict(d: dict[str, Any]) -> UsageStats:
    return UsageStats(
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        total_tokens=d.get("total_tokens", 0),
        cost_usd=d.get("cost_usd"),
    )


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
