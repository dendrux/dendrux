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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine

from ulid import ULID

logger = logging.getLogger(__name__)

# ``Message.meta`` key naming the model that produced an ASSISTANT message's
# ``reasoning_blocks``. Providers replay the blocks only to that same model.
REASONING_MODEL_META_KEY = "reasoning_model"

# ``Message.meta`` flag on messages replayed from an earlier run via
# ``agent.run(runs=...)``. Providers use it to opt replayed reasoning into
# vendor-side prefix-binding safety nets.
REPLAYED_META_KEY = "dendrux.replayed"


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

    When the model's arguments were not a valid JSON object, ``params`` is
    ``{}``, ``parse_error`` holds a short description of the problem and
    ``raw_arguments`` the original string. The loop rejects such a call
    without running the tool. ``raw_arguments`` is in-memory only and is
    never persisted; persisted records carry ``parse_error`` and the raw
    length instead.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=generate_ulid)
    provider_tool_call_id: str | None = None
    raw_arguments: str | None = field(default=None, repr=False)
    parse_error: str | None = None


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

    Origin envelope (context-blocks foundation):
        kind:       Block identity. Defaults from role (TOOL -> "tool_result",
                    SYSTEM -> "system", else "chat"). Dev-supplied context blocks
                    set their own opaque kind; Dendrux carries it uninterpreted.
        placement:  Cache intent, "stable" | "dynamic". Only meaningful for
                    context-origin messages (drives the stable/dynamic assembly
                    split); ignored for history/scratchpad, which are positioned
                    by the assembly order.
        source:     Optional free-form provenance label (citations/telemetry).
    """

    role: Role
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    kind: str = "chat"
    placement: str = "dynamic"
    source: str | None = None
    # Reasoning the model surfaced on an ASSISTANT turn: ``reasoning`` is the
    # human-readable summary (never raw CoT); ``reasoning_blocks`` are opaque
    # provider artifacts replayed verbatim across tool-call iterations.
    reasoning: str | None = None
    reasoning_blocks: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.placement not in ("stable", "dynamic"):
            raise ValueError(f"placement must be 'stable' or 'dynamic', got {self.placement!r}")
        # Derive kind from role when left at the generic default. A TOOL message
        # is always a tool_result; a SYSTEM message is always system context.
        if self.kind == "chat":
            if self.role == Role.TOOL:
                object.__setattr__(self, "kind", "tool_result")
            elif self.role == Role.SYSTEM:
                object.__setattr__(self, "kind", "system")
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

    ``strict`` opts into OpenAI strict function calling. Normalization
    happens once, at decoration time: when ``strict`` is True, ``parameters``
    already holds the strict-mode schema (every property required, objects
    closed with ``additionalProperties: false``), so every provider sees the
    same stored schema. OpenAI Chat and Responses providers additionally send
    ``"strict": true``; Anthropic ignores the flag and sends the schema as-is.
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
    strict: bool = False


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
    # Runtime annotations persisted on the tool_calls row (e.g. ``substituted``).
    meta: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class ToolResultContext:
    """What a ``tool_result_transform`` sees for one successful tool call.

    ``result`` is the raw Python value the tool returned, before JSON
    serialization. ``meta`` is the tool's ``ToolDef.meta`` — for MCP
    tools it carries ``source_name``, ``namespace``, ``mcp_tool_name``, etc.;
    for local ``@tool`` functions it is empty.
    """

    tool_name: str
    call_id: str
    params: dict[str, Any]
    result: Any
    target: ToolTarget
    meta: dict[str, Any]


ToolResultTransform: TypeAlias = Callable[[ToolResultContext], Any | Awaitable[Any]]
"""Hook that rewrites a server tool's result before it is persisted or sent to the model.

The return value ALWAYS replaces the result — return ``ctx.result`` to leave
it unchanged (``None`` is a legitimate new result, not "unchanged"). Sync and
async callables are both accepted. See ``Agent(tool_result_transform=...)``.
"""


@dataclass(frozen=True)
class ToolCallContext:
    """What a ``before_tool`` policy hook sees for one tool call.

    ``params`` and ``meta`` are deep copies taken for the hook: mutating
    them never changes what executes. ``params`` are deanonymized (real
    values, not guardrail placeholders). ``source`` is ``"local"`` for
    ``@tool`` functions, ``"mcp"`` for MCP tools and ``"skill"`` for the
    built-in skill tools. ``tenant_id`` is the run's tenant (``None`` when
    unset or when the run is not persisted).
    """

    run_id: str
    call_id: str
    tool_name: str
    params: dict[str, Any]
    target: ToolTarget
    source: Literal["local", "mcp", "skill"]
    meta: dict[str, Any]
    iteration: int
    tenant_id: str | None = None


@dataclass(frozen=True)
class ToolDecision:
    """A ``before_tool`` hook's verdict for one tool call.

    Build one with :meth:`allow`, :meth:`deny`, :meth:`require_approval`
    or :meth:`substitute`. Returning ``None`` from the hook means allow.
    """

    kind: Literal["allow", "deny", "require_approval", "substitute"]
    reason: str | None = None
    result: Any = None

    @classmethod
    def allow(cls) -> ToolDecision:
        """Run the tool normally."""
        return cls(kind="allow")

    @classmethod
    def deny(cls, reason: str) -> ToolDecision:
        """Do not run the tool; the model receives ``{"denied": reason}``."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("ToolDecision.deny() requires a non-empty reason string.")
        return cls(kind="deny", reason=reason)

    @classmethod
    def require_approval(cls, reason: str | None = None) -> ToolDecision:
        """Pause the batch for human approval (server-target tools only)."""
        return cls(kind="require_approval", reason=reason)

    @classmethod
    def substitute(cls, result: Any) -> ToolDecision:
        """Do not run the tool; ``result`` (JSON-serializable) is the tool output."""
        return cls(kind="substitute", result=result)


BeforeToolHook: TypeAlias = Callable[
    [ToolCallContext], "ToolDecision | None | Awaitable[ToolDecision | None]"
]
"""Policy hook evaluated before every tool call. See ``Agent(before_tool=...)``."""


@dataclass(frozen=True)
class ApprovalDecision:
    """A human decision for one approval-gated tool call.

    Used with ``Agent.submit_approval(run_id, decisions={call_id: ...})``.
    """

    approved: bool
    reason: str | None = None

    @classmethod
    def approve(cls) -> ApprovalDecision:
        """Run the call."""
        return cls(approved=True)

    @classmethod
    def deny(cls, reason: str | None = None) -> ApprovalDecision:
        """Skip the call; the model receives ``{"denied": reason}``."""
        if reason is not None and not isinstance(reason, str):
            raise ValueError("ApprovalDecision.deny() reason must be a string or None.")
        return cls(approved=False, reason=reason)


@dataclass
class UsageStats:
    """Token usage from an LLM call.

    ``input_tokens`` always means fresh (non-cached) input tokens across
    all providers. OpenAI variants normalize this by subtracting cached
    from prompt_tokens; Anthropic returns it that way natively.

    ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` are
    ``None`` when the provider did not report the field (e.g.
    OpenAI-compatible backends without prompt_tokens_details), or an int
    (possibly 0) when the provider did report it. This distinction lets
    downstream tooling tell "didn't report" apart from "reported zero".

    ``reasoning_tokens`` is the count of internal reasoning/thinking tokens
    the model billed within ``output_tokens`` (Anthropic thinking, OpenAI
    reasoning). ``None`` when the provider did not report it.

    ``cost_usd`` is ``None`` when nothing priced the call: the provider did
    not report a cost and no :class:`~dendrux.pricing.PriceTable` covered the
    model. A run total is ``None`` as soon as any step is unpriced, so a
    partial sum never reads as the whole run's spend. ``cost_source`` says
    where a non-``None`` cost came from: ``"provider"`` (reported on the
    response, e.g. OpenRouter), ``"table"`` (computed from the developer's
    price table), or ``"mixed"`` (a run total whose steps differ).

    ``usage_reported`` is false when token counts were unavailable. Such
    usage cannot be priced by a table, even though its counters default to
    zero. Explicitly constructed usage defaults to reported; providers must
    set this flag to false when the API omits usage.

    ``cost_unknown`` tracks whether an aggregate includes an unpriced call,
    independently of token counts. It survives pause/resume serialization.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_source: str | None = None
    usage_reported: bool = True
    cost_unknown: bool = False


BudgetAction: TypeAlias = Literal["warn", "pause", "cancel"]
_BUDGET_ACTIONS = ("warn", "pause", "cancel")


@dataclass(frozen=True)
class Budget:
    """Token and/or dollar budget for a run and, by default, its delegated children.

    Usage is checked after every LLM response and at the top of every
    iteration. ``budget.threshold`` fires once per ``(metric, fraction)``
    when a ``warn_at`` fraction is first crossed and ``budget.exceeded``
    once per metric when its limit is reached; every payload carries
    ``metric`` (``"tokens"`` or ``"cost_usd"``), ``used`` and ``max``. Fired
    keys survive pause/resume, so a resumed run does not re-fire them.

    ``on_exceed`` decides what happens once a limit is reached:

    - ``"warn"`` (default): events only; the run continues.
    - ``"pause"``: durable pause (``RunStatus.PAUSED``, ``pause_reason="budget"``).
    - ``"cancel"``: ``RunStatus.CANCELLED`` with ``failure_reason="budget_exceeded"``.

    Pause and cancel take effect at the next top-of-iteration checkpoint,
    after the tools of the response that crossed the limit have run, so a
    run can overshoot by up to one iteration. A response that finishes the
    run still completes. Cancellation and requested pauses take precedence
    over a budget pause at the same checkpoint. ``SingleCall`` has no
    checkpoint: pause/cancel degrade to warn there.

    Resuming a budget pause succeeds only if the resuming Agent's limits are
    strictly above the current live usage; otherwise the run pauses again
    immediately with the same reason.

    With a state store, usage is read live from recorded LLM calls; with
    ``include_children=True`` it is the root-inclusive rollup of the run and
    all its delegated descendants. The cost limit counts priced calls only:
    unpriced calls (no provider cost and no ``PriceTable`` entry) emit one
    ``budget.unpriced`` event per run and are otherwise ignored.

    Args:
        max_tokens: Token limit (input + output, > 0), or None.
        warn_at: Fractions of each limit at which ``budget.threshold``
            fires. Each fraction must be in (0, 1) exclusive.
            Default: (0.5, 0.75, 0.9).
        max_cost_usd: Dollar limit (> 0), or None.
        on_exceed: ``"warn"``, ``"pause"`` or ``"cancel"``.
        include_children: Count delegated descendant runs' usage.

    Raises:
        ValueError: If neither limit is set, a limit is not positive, a
            ``warn_at`` fraction is outside (0, 1), or ``on_exceed`` is unknown.
    """

    max_tokens: int | None = None
    warn_at: tuple[float, ...] = (0.5, 0.75, 0.9)
    max_cost_usd: float | None = None
    on_exceed: BudgetAction = "warn"
    include_children: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens is None and self.max_cost_usd is None:
            raise ValueError("Budget needs max_tokens and/or max_cost_usd.")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError(f"Budget max_tokens must be > 0, got {self.max_tokens}.")
        if self.max_cost_usd is not None and (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not (0 < self.max_cost_usd < float("inf"))
        ):
            raise ValueError(f"Budget max_cost_usd must be > 0, got {self.max_cost_usd}.")
        for f in self.warn_at:
            if not (0 < f < 1):
                raise ValueError(f"Budget warn_at fractions must be in (0, 1) exclusive, got {f}.")
        if self.on_exceed not in _BUDGET_ACTIONS:
            raise ValueError(
                f"Budget on_exceed must be one of {_BUDGET_ACTIONS}, got {self.on_exceed!r}."
            )
        if not isinstance(self.include_children, bool):
            raise ValueError("Budget include_children must be a bool.")


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider.

    Anthropic, OpenAI, or any other provider all normalize to this.
    Strategies consume this — they never touch provider-specific APIs.
    """

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw: Any = None  # Full provider response for debugging
    usage: UsageStats = field(default_factory=lambda: UsageStats(usage_reported=False))
    # Adapter-boundary payloads — set by each provider, persisted as opaque JSON.
    # provider_request: the exact kwargs sent to the vendor API (e.g. Anthropic api_kwargs).
    # provider_response: the raw vendor response dict (e.g. response.model_dump()).
    provider_request: dict[str, Any] | None = None
    provider_response: dict[str, Any] | None = None
    # The model actually used for this call. Honors a per-call ``model=`` override;
    # falls back to the provider's configured model when a provider leaves it unset.
    model: str | None = None
    # Reasoning/thinking the model surfaced (summary text — never raw CoT).
    # None unless thinking was enabled and a summary/display was requested.
    reasoning: str | None = None
    # Opaque provider reasoning artifacts to replay verbatim for multi-turn
    # tool use (Anthropic thinking blocks + signature, OpenAI reasoning items).
    # Stored and round-tripped as-is; never interpreted by Dendrux.
    reasoning_blocks: list[Any] | None = None


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
    # Durable cooperative pause requested via ``Agent.request_pause``.
    PAUSED = "paused"
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

    # before_tool policy hook
    POLICY_SUBSTITUTED = "policy.substituted"
    POLICY_HOOK_ERROR = "policy.hook_error"
    POLICY_HOOK_WARNING = "policy.hook_warning"

    # Wave 3: Budget
    BUDGET_THRESHOLD = "budget.threshold"
    BUDGET_EXCEEDED = "budget.exceeded"
    BUDGET_UNPRICED = "budget.unpriced"

    # Wave 4: Guardrails
    GUARDRAIL_DETECTED = "guardrail.detected"
    GUARDRAIL_REDACTED = "guardrail.redacted"
    GUARDRAIL_BLOCKED = "guardrail.blocked"
    GUARDRAIL_UNMAPPED_PLACEHOLDER = "guardrail.unmapped_placeholder"

    # Init events: Skills + MCP
    SKILL_CATALOG_RESOLVED = "skill.catalog_resolved"
    SKILL_ACTIVATED = "skill.activated"
    SKILL_ACTIVATION_FAILED = "skill.activation_failed"
    SKILL_RESOURCE_READ = "skill.resource_read"
    SKILL_SCRIPT_REQUESTED = "skill.script_requested"
    SKILL_SCRIPT_COMPLETED = "skill.script_completed"
    SKILL_SCRIPT_FAILED = "skill.script_failed"
    MCP_CONNECTED = "mcp.connected"
    MCP_ERROR = "mcp.error"
    MCP_TOOLS_SKIPPED = "mcp.tools_skipped"

    # Provider transport: vendor-SDK retry observability
    PROVIDER_RETRY = "provider.retry"

    # Run control: inbox note delivered to the model (payload: note_id only)
    INBOX_DELIVERED = "inbox.delivered"


@dataclass
class RunResult:
    """The outcome of an agent run.

    On success, ``answer`` contains the final answer turn, not all assistant
    text produced across tool calls. ``commentary`` lists, in order, the
    non-empty assistant text of every step that ended in tool calls (progress
    narration such as "Let me look that up"). It is empty for SingleCall and
    for idempotent cached results. Persisted traces tag the same split with
    ``phase`` (``RunStore.get_traces()``).

    On stream interruption via cancel_run(), the partial answer contains
    only the current text buffer, which resets at each TOOL_RESULT. Closing
    a stream early persists that buffer as the run's answer. It can be None
    even if earlier turns produced text. Cancelling the coroutine awaiting run()
    re-raises CancelledError rather than returning a RunResult; it does not
    capture in-flight model text. ``Agent.cancel_run`` from the same process
    instead makes run() return CANCELLED with ``meta["interrupted"]=True`` and
    ``meta["partial_output"]=False``. Paused runs may carry a clarification
    question in ``answer``. ``meta["undelivered_notes"]`` lists inbox note ids
    a finished run never delivered (only present when there are any).
    """

    run_id: str
    status: RunStatus
    answer: str | None = None
    output: Any = None
    steps: list[AgentStep] = field(default_factory=list)
    iteration_count: int = 0
    usage: UsageStats = field(default_factory=UsageStats)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    commentary: list[str] = field(default_factory=list)
    replay: ReplaySummary | None = None


PauseRequestOutcome: TypeAlias = Literal[
    "requested", "already_waiting", "already_paused", "terminal"
]


@dataclass(frozen=True)
class PauseOutcome:
    """Result of :meth:`Agent.request_pause`.

    ``status_before`` / ``status_after`` are the root run's status when the
    request started and after it was recorded. ``affected`` lists every run
    the request touched as ``(run_id, outcome)``, deepest descendants first
    and the root last:

    - ``"requested"``: the run was RUNNING; it pauses at its next checkpoint.
    - ``"already_waiting"``: WAITING_*; already durable, nothing latched.
    - ``"already_paused"``: already PAUSED.
    - ``"terminal"``: already finished; nothing to pause.

    Returning is not confirmation that a run paused: ``run.paused`` in the
    event log (status ``paused``) confirms it.
    """

    run_id: str
    status_before: RunStatus
    status_after: RunStatus
    affected: list[tuple[str, PauseRequestOutcome]] = field(default_factory=list)


@dataclass(frozen=True)
class ReplaySummary:
    """What ``agent.run(runs=...)`` replayed from earlier runs, and what it dropped.

    ``runs_full`` / ``runs_text`` list the run ids replayed in full (text,
    tool calls, results) and as question + answer only. ``reasoning_kept`` /
    ``reasoning_dropped`` list full-replay runs whose stored reasoning blocks
    were sent / withheld (runs with no reasoning appear in neither).
    ``warnings`` are human-readable notes: which run, what was dropped, why.
    """

    runs_full: list[str] = field(default_factory=list)
    runs_text: list[str] = field(default_factory=list)
    reasoning_kept: list[str] = field(default_factory=list)
    reasoning_dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (the shape mirrored into ``agent_runs.meta["replay"]``)."""
        return {
            "runs_full": list(self.runs_full),
            "runs_text": list(self.runs_text),
            "reasoning_kept": list(self.reasoning_kept),
            "reasoning_dropped": list(self.reasoning_dropped),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplaySummary:
        """Rebuild from :meth:`to_dict` output; unknown keys are ignored."""
        return cls(
            runs_full=list(data.get("runs_full") or []),
            runs_text=list(data.get("runs_text") or []),
            reasoning_kept=list(data.get("reasoning_kept") or []),
            reasoning_dropped=list(data.get("reasoning_dropped") or []),
            warnings=list(data.get("warnings") or []),
        )


@dataclass
class PauseState:
    """Everything needed to resume a paused agent run.

    This is execution state, not observability state. It is persisted
    raw in pause_data (JSON column) and cleared on finalize — the DB
    is ground truth across traces, tool_calls, and pause_data alike.
    Guardrail redaction (placeholders) applies only at the LLM-call
    boundary; the bijection lives on agent_runs.pii_mapping for audit
    replay.
    """

    agent_name: str
    pending_tool_calls: list[ToolCall]
    history: list[Message]
    steps: list[AgentStep]
    iteration: int
    trace_order_offset: int
    usage: UsageStats
    pending_targets: dict[str, str] = field(default_factory=dict)  # tool_call_id → target
    # Approval-gated call ids at pause time: {call_id: {"source", "reason"}}.
    # None on non-approval pauses and on pauses persisted before 0.3.0.
    approval_gates: dict[str, dict[str, str | None]] | None = None
    # Why a PAUSED run paused ("requested" or "budget"); None for WAITING_* pauses.
    pause_reason: str | None = None
    # Free-text reason given to ``Agent.request_pause(reason=...)``.
    pause_message: str | None = None
    # Decisions already made for pending calls of a pause that interrupted an
    # approval batch: {call_id: {"approved": bool, "reason": str | None}}.
    # Resume executes them without asking for approval again.
    pending_decisions: dict[str, dict[str, Any]] | None = None
    # Budget events already emitted, as (metric, threshold); threshold 1.0 is
    # "exceeded". Persisted on every pause so a resumed run does not re-fire.
    budget_fired: list[tuple[str, float]] | None = None
    # Whether ``budget.unpriced`` was already emitted for this run.
    budget_unpriced_reported: bool = False
    # Limits a budget pause hit: [{"metric", "used", "max"}]. Budget pauses only.
    budget_exceeded: list[dict[str, Any]] | None = None

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
        if self.approval_gates is not None:
            d["approval_gates"] = self.approval_gates
        if self.pause_reason is not None:
            d["pause_reason"] = self.pause_reason
        if self.pause_message is not None:
            d["pause_message"] = self.pause_message
        if self.pending_decisions is not None:
            d["pending_decisions"] = self.pending_decisions
        if self.budget_fired:
            d["budget_fired"] = [[metric, threshold] for metric, threshold in self.budget_fired]
        if self.budget_unpriced_reported:
            d["budget_unpriced_reported"] = True
        if self.budget_exceeded is not None:
            d["budget_exceeded"] = self.budget_exceeded
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
            approval_gates=data.get("approval_gates"),
            pause_reason=data.get("pause_reason"),
            pause_message=data.get("pause_message"),
            pending_decisions=data.get("pending_decisions"),
            budget_fired=[(str(m), float(t)) for m, t in data.get("budget_fired") or []] or None,
            budget_unpriced_reported=bool(data.get("budget_unpriced_reported", False)),
            budget_exceeded=data.get("budget_exceeded"),
        )


# ------------------------------------------------------------------
# PauseState serialization helpers
# ------------------------------------------------------------------


def tool_call_parse_meta(tc: ToolCall) -> dict[str, Any]:
    """Persistable parse-failure fields of a tool call; empty when it parsed.

    Returns ``parse_error`` and ``raw_length`` only, never the raw text.
    """
    if tc.parse_error is None:
        return {}
    return {
        "parse_error": tc.parse_error,
        "raw_length": len(tc.raw_arguments) if tc.raw_arguments is not None else None,
    }


def _tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    return {
        "name": tc.name,
        "params": tc.params,
        "id": tc.id,
        "provider_tool_call_id": tc.provider_tool_call_id,
        **tool_call_parse_meta(tc),
    }


def _tool_call_from_dict(d: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=d["name"],
        params=d.get("params", {}),
        id=d["id"],
        provider_tool_call_id=d.get("provider_tool_call_id"),
        parse_error=d.get("parse_error"),
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
    # Origin envelope — only persist non-default values (kind for TOOL/SYSTEM is
    # re-derived from role on load, so serialize only when it carries new info).
    if m.kind not in ("chat", "tool_result", "system"):
        d["kind"] = m.kind
    if m.placement != "dynamic":
        d["placement"] = m.placement
    if m.source is not None:
        d["source"] = m.source
    # Reasoning summary + opaque replay blocks — persisted so a run paused
    # mid-tool-loop keeps its reasoning context across resume.
    if m.reasoning is not None:
        d["reasoning"] = m.reasoning
    if m.reasoning_blocks is not None:
        d["reasoning_blocks"] = m.reasoning_blocks
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
        kind=d.get("kind", "chat"),
        placement=d.get("placement", "dynamic"),
        source=d.get("source"),
        reasoning=d.get("reasoning"),
        reasoning_blocks=d.get("reasoning_blocks"),
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
        "cache_read_input_tokens": u.cache_read_input_tokens,
        "cache_creation_input_tokens": u.cache_creation_input_tokens,
        "cost_source": u.cost_source,
        "reasoning_tokens": u.reasoning_tokens,
        "usage_reported": u.usage_reported,
        "cost_unknown": u.cost_unknown,
    }


def _usage_from_dict(d: dict[str, Any]) -> UsageStats:
    return UsageStats(
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        total_tokens=d.get("total_tokens", 0),
        cost_usd=d.get("cost_usd"),
        cache_read_input_tokens=d.get("cache_read_input_tokens"),
        cache_creation_input_tokens=d.get("cache_creation_input_tokens"),
        cost_source=d.get("cost_source"),
        reasoning_tokens=d.get("reasoning_tokens"),
        usage_reported=d.get("usage_reported", True),
        cost_unknown=d.get("cost_unknown", False),
    )


class StreamEventType(StrEnum):
    """Types of events emitted during streaming LLM responses.

    TEXT_DELTA: Incremental text token from the LLM.
    REASONING_DELTA: Incremental reasoning/thinking token (``text`` carries it).
        Only emitted when thinking is enabled; arrives before TEXT_DELTA.
    TOOL_USE_START: A tool call block has begun (name known, args pending).
    TOOL_USE_END: A tool call is fully assembled and ready to execute.
    DONE: Stream finished. ``raw`` carries the full ``LLMResponse``.
    """

    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
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
    # Provider-native answer phase on TEXT_DELTA ("commentary" | "final_answer"),
    # set only when the vendor labels it (OpenAI Responses message ``phase``).
    phase: str | None = None


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
        RUN_CANCELLED: Terminal. Run was cancelled cooperatively.

    LLM output events (translated from provider StreamEvent):
        TEXT_DELTA: Incremental text token from the LLM.
        REASONING_DELTA: Incremental reasoning/thinking token (in ``text``).
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
    RUN_CANCELLED = "run_cancelled"

    # LLM output (translated from provider StreamEvent)
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_END = "tool_use_end"

    # Execution (loop-owned)
    TOOL_RESULT = "tool_result"


# Terminal event types — after any of these, the stream ends.
_TERMINAL_RUN_EVENTS = frozenset(
    {
        RunEventType.RUN_COMPLETED,
        RunEventType.RUN_PAUSED,
        RunEventType.RUN_ERROR,
        RunEventType.RUN_CANCELLED,
    }
)


@dataclass(frozen=True)
class RunEvent:
    """A single event from agent.stream() or agent.resume_stream().

    Carries data relevant to its type:
        RUN_STARTED:     run_id (first event for new runs)
        RUN_RESUMED:     run_id (first event for resumed runs)
        TEXT_DELTA:      text, iteration, phase (when already known)
        REASONING_DELTA: text, iteration
        TOOL_USE_START:  tool_name, tool_call_id, iteration, phase="commentary"
        TOOL_USE_END:    tool_call, tool_name, tool_call_id, iteration, phase="commentary"
        TOOL_RESULT:     tool_call, tool_result, iteration, phase="commentary"
        RUN_COMPLETED:   run_result, phase="final_answer" on success
        RUN_PAUSED:      run_result (pause state in run_result.meta)
        RUN_ERROR:       run_result, error

    ``iteration`` is the loop step (one LLM call) an LLM-output or tool event
    belongs to; it is None on lifecycle events. ``phase`` labels assistant
    text as ``"commentary"`` (narration in a step that ends in tool calls) or
    ``"final_answer"``. A text block usually streams before the tool call
    that makes it commentary, so TEXT_DELTA.phase is often None. Clients
    bucket TEXT_DELTA by ``iteration``, mark a bucket commentary when a tool
    event with that iteration arrives, and treat the last unmarked bucket as
    the answer once RUN_COMPLETED carries ``phase="final_answer"``
    (``run_result.answer`` stays authoritative).
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
    iteration: int | None = None
    phase: str | None = None


class RunStream:
    """Async iterable of RunEvents with lifecycle management.

    Single-use: iterating twice raises RuntimeError.

    With an idempotency key, ``run_id`` is provisional before iteration and
    resolves to the existing run's ID during setup if the key matches.
    Cached outcomes yield one terminal event; ``result`` holds the cached
    summary, and no text or tool events are replayed.

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
        cleanup: Callable[[], Coroutine[Any, Any, RunResult | None]],
    ) -> None:
        self.run_id = run_id
        self._gen = generator
        self._cleanup = cleanup
        self._result: RunResult | None = None
        self._terminated = False
        self._started = False
        self._closed = False

    @property
    def result(self) -> RunResult | None:
        """Terminal result after a terminal event or explicit stream closure."""
        return self._result

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
                    self._result = event.run_result
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
                self._result = await self._cleanup()
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
    history: list[Message] | None = None,
    context: list[Any] | None = None,
    replay: dict[str, Any] | None = None,
    skills: dict[str, Any] | None = None,
) -> str:
    """Deterministic SHA-256 fingerprint for idempotency conflict detection.

    Includes agent_name + user_input + output_type + chat history + per-run
    context (request identity). Does NOT include provider kwargs (temperature,
    max_tokens) — those are execution tuning, not request identity.

    output_type changes the return shape contract, so two calls with
    different output_types must be treated as different requests.

    History is hashed as the canonical list of (role, content) pairs from
    the normalized Message objects — different construction paths producing
    equivalent normalized history hash identically. Context blocks are hashed
    by (kind, placement, source, content) so the same input with different
    context is treated as a different request. ``replay`` carries the
    ``runs`` / ``replay_last`` / ``replay_reasoning`` knobs when ``runs=`` is
    used; omitted otherwise so existing fingerprints are unchanged.
    """
    data: dict[str, Any] = {"agent_name": agent_name, "user_input": user_input}
    if output_type_name is not None:
        data["output_type"] = output_type_name
    if history:
        data["history"] = [{"role": m.role.value, "content": m.content} for m in history]
    if context:
        data["context"] = [
            {
                "kind": b.kind,
                "placement": b.placement,
                "source": b.source,
                "content": b.content,
            }
            for b in context
        ]
    if replay:
        data["replay"] = replay
    if skills is not None:
        data["skills"] = skills
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
