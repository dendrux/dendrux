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

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine

from ulid import ULID

logger = logging.getLogger(__name__)


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
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for params
    target: ToolTarget = ToolTarget.SERVER
    parallel: bool = True  # Governs concurrent execution scheduling
    max_calls_per_run: int | None = None  # Enforced by ReActLoop — graceful limit
    timeout_seconds: float = 120.0  # Enforced by ReActLoop via asyncio.wait_for
    has_explicit_timeout: bool = False  # True when developer set timeout_seconds explicitly
    meta: dict[str, Any] = field(default_factory=dict)  # MCP source info, annotations, etc.


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


@dataclass(frozen=True)
class Budget:
    """Token budget for advisory spend tracking.

    Non-blocking in v1 — fires governance events (budget.threshold,
    budget.exceeded) but does not pause or stop the run. Developers
    observe events and take action in their own integration.

    Args:
        max_tokens: Advisory token cap (must be > 0). Events fire when
            usage crosses warn_at fractions and when usage reaches
            this value.
        warn_at: Fractions of max_tokens at which budget.threshold
            events fire. Each fraction must be in (0, 1) exclusive.
            Each fraction fires exactly once per run.
            Default: (0.5, 0.75, 0.9).

    Raises:
        ValueError: If max_tokens <= 0 or any warn_at fraction is
            outside (0, 1) exclusive.
    """

    max_tokens: int
    warn_at: tuple[float, ...] = (0.5, 0.75, 0.9)

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"Budget max_tokens must be > 0, got {self.max_tokens}.")
        for f in self.warn_at:
            if not (0 < f < 1):
                raise ValueError(f"Budget warn_at fractions must be in (0, 1) exclusive, got {f}.")


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
    supports_structured_output: bool = False
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


class GovernanceEventType(StrEnum):
    """Governance event types emitted via on_governance_event().

    Use these instead of raw strings for autocomplete and typo prevention:

        from dendrux.types import GovernanceEventType

        if event_type == GovernanceEventType.POLICY_DENIED:
            ...

    StrEnum — compares equal to the string value, so existing code
    using raw strings continues to work without changes.
    """

    # Wave 1: Tool deny
    POLICY_DENIED = "policy.denied"

    # Wave 2: Approval (HITL)
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"

    # Wave 3: Budget
    BUDGET_THRESHOLD = "budget.threshold"
    BUDGET_EXCEEDED = "budget.exceeded"

    # Wave 4: Guardrails
    GUARDRAIL_DETECTED = "guardrail.detected"
    GUARDRAIL_REDACTED = "guardrail.redacted"
    GUARDRAIL_BLOCKED = "guardrail.blocked"

    # Init events: Skills + MCP
    SKILL_REGISTERED = "skill.registered"
    SKILL_DENIED = "skill.denied"
    SKILL_INVOKED = "skill.invoked"
    MCP_CONNECTED = "mcp.connected"
    MCP_ERROR = "mcp.error"


@dataclass
class RunResult:
    """The final output of an agent run."""

    run_id: str
    status: RunStatus
    answer: str | None = None
    output: Any = None
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
    """Types of events emitted during streaming LLM responses.

    TEXT_DELTA: Incremental text token from the LLM.
    TOOL_USE_START: A tool call block has begun (name known, args pending).
    TOOL_USE_END: A tool call is fully assembled and ready to execute.
    DONE: Stream finished. ``raw`` carries the full ``LLMResponse``.
    """

    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
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


# ------------------------------------------------------------------
# Run-level streaming types
# ------------------------------------------------------------------


class RunEventType(StrEnum):
    """Types of events emitted during agent.stream() and agent.resume_stream().

    These span the full multi-turn run, not just one LLM call.

    Lifecycle events (runner-owned):
        RUN_STARTED: First event for new runs. Carries run_id for correlation.
        RUN_RESUMED: First event for resumed runs. Carries run_id.
        RUN_COMPLETED: Terminal. Run finished (success or max_iterations).
        RUN_PAUSED: Terminal. Run paused for client tool or human input.
        RUN_ERROR: Terminal. Run failed. No exception is raised to the consumer.

    LLM output events (translated from provider StreamEvent):
        TEXT_DELTA: Incremental text token from the LLM.
        TOOL_USE_START: A tool call block has begun (name known, args pending).
        TOOL_USE_END: A tool call is fully assembled.

    Execution events (loop-owned):
        TOOL_RESULT: A tool has been executed and its result is available.
    """

    # Lifecycle (runner-owned)
    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    RUN_COMPLETED = "run_completed"
    RUN_PAUSED = "run_paused"
    RUN_ERROR = "run_error"

    # LLM output (translated from provider StreamEvent)
    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_END = "tool_use_end"

    # Execution (loop-owned)
    TOOL_RESULT = "tool_result"


# Terminal event types — after any of these, the stream ends.
_TERMINAL_RUN_EVENTS = frozenset(
    {RunEventType.RUN_COMPLETED, RunEventType.RUN_PAUSED, RunEventType.RUN_ERROR}
)


@dataclass(frozen=True)
class RunEvent:
    """A single event from agent.stream() or agent.resume_stream().

    Carries data relevant to its type:
        RUN_STARTED:   run_id (first event for new runs)
        RUN_RESUMED:   run_id (first event for resumed runs)
        TEXT_DELTA:     text
        TOOL_USE_START: tool_name, tool_call_id
        TOOL_USE_END:   tool_call, tool_name, tool_call_id
        TOOL_RESULT:    tool_call, tool_result
        RUN_COMPLETED:  run_result
        RUN_PAUSED:     run_result (pause state in run_result.meta)
        RUN_ERROR:      run_result, error
    """

    type: RunEventType
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_result: ToolResult | None = None
    run_result: RunResult | None = None
    run_id: str | None = None
    error: str | None = None


class RunStream:
    """Async iterable of RunEvents with lifecycle management.

    Single-use: iterating twice raises RuntimeError.

    Cleanup: ``__aiter__`` returns an async generator wrapper whose
    ``finally`` block runs CAS-guarded cancellation if no terminal event
    was received. The cleanup fires reliably in these cases:

      - The generator is exhausted normally (all events consumed).
      - The consumer calls ``await stream.aclose()`` explicitly.
      - The consumer uses ``async with stream:`` — **recommended**
        for deterministic immediate cleanup on break or exception.

    After ``async for ...: break`` without ``async with``, cleanup
    depends on asyncio's async generator finalization hooks, which
    schedule cleanup when the wrapper is garbage-collected. This is
    **not immediate** and not guaranteed before the event loop ends.
    For production use, prefer ``async with``::

        async with agent.stream("task") as stream:
            async for event in stream:
                if done:
                    break  # __aexit__ guarantees immediate cleanup
    """

    def __init__(
        self,
        run_id: str,
        generator: AsyncGenerator[RunEvent, None],
        cleanup: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        self.run_id = run_id
        self._gen = generator
        self._cleanup = cleanup
        self._terminated = False
        self._started = False
        self._closed = False

    def __aiter__(self) -> AsyncGenerator[RunEvent, None]:
        if self._started:
            raise RuntimeError("RunStream is single-use; cannot iterate twice")
        self._started = True
        return self._wrap()

    async def _wrap(self) -> AsyncGenerator[RunEvent, None]:
        """Async generator wrapper — cleanup runs in the finally block.

        Fires reliably on exhaustion and explicit ``aclose()``.
        After ``break`` without ``async with``, depends on asyncio's
        async gen finalization hooks (non-deterministic timing).
        """
        try:
            async for event in self._gen:
                if event.type in _TERMINAL_RUN_EVENTS:
                    self._terminated = True
                yield event
        finally:
            await self._finalize()

    async def __aenter__(self) -> RunStream:
        """Enter async context — for deterministic immediate cleanup."""
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the stream. Idempotent — safe to call multiple times."""
        await self._finalize()

    async def _finalize(self) -> None:
        """Run cleanup exactly once."""
        if self._closed:
            return
        self._closed = True
        if not self._terminated:
            try:
                await self._cleanup()
            except Exception:
                logger.warning("RunStream cleanup failed", exc_info=True)
        await self._gen.aclose()

    async def text(self) -> AsyncGenerator[str, None]:
        """Convenience filter — yields only text deltas as plain strings."""
        async for event in self:
            if event.type == RunEventType.TEXT_DELTA and event.text:
                yield event.text


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreateRunResult:
    """Outcome of a create_run() call, including idempotency resolution."""

    run_id: str
    outcome: Literal["created", "existing_active", "existing_terminal"]
    status: RunStatus


class RunAlreadyActiveError(RuntimeError):
    """Raised when an idempotency key matches a run that is still executing.

    The caller should use ``agent.resume()`` or poll for completion
    instead of starting a new run.
    """

    def __init__(self, run_id: str, current_status: RunStatus) -> None:
        self.run_id = run_id
        self.current_status = current_status
        super().__init__(
            f"Run {run_id} is already {current_status.value}. "
            f"Use agent.resume() or wait for completion."
        )


class IdempotencyConflictError(RuntimeError):
    """Raised when an idempotency key is reused with different request parameters.

    This indicates a bug in the caller — keys must not be reused across
    different requests. Generate a new key for each distinct request.
    """

    def __init__(self, run_id: str, idempotency_key: str) -> None:
        self.run_id = run_id
        self.idempotency_key = idempotency_key
        super().__init__(
            f"Idempotency key '{idempotency_key}' already used for run {run_id} "
            f"with different input. Keys must not be reused across different requests."
        )


class StructuredOutputValidationError(RuntimeError):
    """Raised when the LLM response fails validation against the requested output_type.

    Carries the raw response text and the validation error for debugging.
    """

    def __init__(self, raw_response: str, output_type_name: str, validation_error: str) -> None:
        self.raw_response = raw_response
        self.output_type_name = output_type_name
        self.validation_error = validation_error
        super().__init__(
            f"Structured output validation failed for {output_type_name}. "
            f"Validation error: {validation_error}\n"
            f"Raw response: {raw_response[:500]}"
        )


def compute_idempotency_fingerprint(
    agent_name: str,
    user_input: str,
    output_type_name: str | None = None,
) -> str:
    """Deterministic SHA-256 fingerprint for idempotency conflict detection.

    Includes agent_name + user_input + output_type (request identity).
    Does NOT include provider kwargs (temperature, max_tokens) — those
    are execution tuning, not request identity.

    output_type changes the return shape contract, so two calls with
    different output_types must be treated as different requests.
    """
    data: dict[str, Any] = {"agent_name": agent_name, "user_input": user_input}
    if output_type_name is not None:
        data["output_type"] = output_type_name
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
