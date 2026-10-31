"""Agent runner — the entry point for executing agents.

Takes an Agent definition and runs it through the loop with an explicit
provider and strategy. This is the top-level API developers interact with.

Sprint 1: caller provides the LLM provider instance, defaults to
NativeToolCalling strategy and ReActLoop. Future sprints add provider
registry (model string → provider resolution), strategy selection from
agent config, and more loop types.

Sprint 2 adds optional state_store for persistence. When provided:
  - Runner owns the run_id (generates it, passes to loop)
  - PersistenceRecorder records traces, tool calls, and usage (fail-closed)
  - finalize_run() is called in try/finally to guarantee persistence
  - Developer's extra_notifier is passed separately (best-effort)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple, get_args, overload

from dendrux import _codes
from dendrux._sentinel import _OMITTED, _OmittedType, _UnsetType
from dendrux.chat import (
    history_has_images,
    normalize_chat_history,
    split_user_content,
    validate_chat_history,
)
from dendrux.context_blocks import (
    ContextBlock,
    fold_context,
    render_band,
    validate_context,
    wire_user_input,
    wire_user_parts,
)
from dendrux.errors import BlobNotFoundError, InvalidImageError, RunFencedError, safe_exception_info
from dendrux.loops._budget import BUDGET_EXCEEDED, BudgetCarry
from dendrux.loops._helpers import (
    notifier_span_link,
    notifier_trace_link,
    notify_message,
    notify_run_failed,
    notify_run_finished,
    notify_run_started,
    order_observations,
    project_message,
    record_message,
    record_run_failed,
    record_run_finished,
    record_run_started,
    record_tool,
)
from dendrux.loops.react import ReActLoop
from dendrux.runtime._entry_checks import resolve_runner_tenant, runner_entry_checks
from dendrux.runtime._tenant_checks import _tenant_scope_chosen
from dendrux.runtime._tracelink import validate_trace_link
from dendrux.runtime.context import (
    DelegationContext,
    StreamContext,
    get_delegation_context,
    get_store_identity,
    reset_delegation_context,
    resolve_child_tenant,
    resolve_parent_link,
    set_delegation_context,
)
from dendrux.runtime.replay import (
    PARTS_KEY,
    REPLAY_META_KEY,
    ReplayLast,
    ReplayPlan,
    ReplayRequest,
    _pair_tool_calls,
    build_seed,
    current_prefix,
    expand_replay,
    has_images,
    image_slot_count,
    initial_image_view,
    load_image_sources,
    load_replay_runs,
    parts_to_dicts,
    rebuild_messages,
    record_run_prefix,
    user_trace_message,
    validate_replay_args,
)
from dendrux.runtime.state import ClaimResult, _initial_loop_state, claimed_store, unclaimed_store
from dendrux.runtime.submit import (
    GOVERNANCE_KEY,
    claim_paused_validated,
    governance_snapshot,
    legacy_approval_denial,
    prepare_resume_results,
    report_governance_change,
    require_pause_id,
    submission_validator,
    validate_result_names,
)
from dendrux.skills._session import META_KEY, SkillSession, continuation, current
from dendrux.skills._session import current as current_skill_session
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    ApprovalDecision,
    ErrorInfo,
    GovernanceEventType,
    IdempotencyConflictError,
    ImagePart,
    ImageRef,
    ImageViewState,
    Message,
    PauseState,
    ReplaySummary,
    Role,
    RunAlreadyActiveError,
    RunEventType,
    RunStatus,
    StopReason,
    TextPart,
    UsageStats,
    _usage_from_dict,
    compute_idempotency_fingerprint,
    content_part_from_dict,
    generate_ulid,
    tool_result_message,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence

    from dendrux.agent import Agent
    from dendrux.chat import ChatMessage
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.media._catalog import BlobEdge
    from dendrux.media._image import UserContent
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import RunRecord, StateStore, TraceRecord
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        ContentPart,
        CreateRunResult,
        RunEvent,
        RunResult,
        RunStream,
        ToolCall,
        ToolResult,
    )

logger = logging.getLogger(__name__)

STOP_REASON_META_KEY = "dendrux.stop_reason"
"""``agent_runs.meta`` key holding a SUCCESS run's final normalized stop reason."""

_STOP_REASONS = frozenset(get_args(StopReason))


class _TraceNotifier:
    """Forward hooks while attaching a tool span only to its executing task."""

    def __init__(self, inner: LoopNotifier) -> None:
        self.inner = inner
        self.tokens: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def on_tool_started(self, run_id: str, call: ToolCall, iteration: int) -> None:
        await self.inner.on_tool_started(run_id, call, iteration)
        ctx = get_delegation_context()
        link = validate_trace_link(notifier_span_link(self.inner, run_id, call.id))
        if ctx is not None and ctx.run_id == run_id and link is not None:
            self.tokens[call.id] = set_delegation_context(replace(ctx, tool_span_link=link))

    async def on_tool_completed(
        self, run_id: str, call: ToolCall, result: ToolResult, iteration: int
    ) -> None:
        try:
            await self.inner.on_tool_completed(run_id, call, result, iteration)
        finally:
            token = self.tokens.pop(call.id, None)
            if token is not None:
                reset_delegation_context(token)


async def _start_trace(
    notifier: LoopNotifier | None,
    store: StateStore | None,
    run_id: str,
    *,
    agent_name: str | None,
    agent_model: str | None,
    parent: DelegationContext | None = None,
    saved: str | None = None,
) -> tuple[str | None, Any]:
    """Start the notifier, persist its valid run link, and retain resume identity."""
    from dendrux.runtime._capabilities import capabilities

    saved = validate_trace_link(saved)
    incoming = saved
    if incoming is None and parent is not None:
        try:
            otel_child = getattr(notifier, "_dendrux_otel", False) is True
        except Exception:
            otel_child = False
        incoming = (
            validate_trace_link(parent.tool_span_link) if otel_child else None
        ) or validate_trace_link(parent.trace_link)
    await notify_run_started(
        notifier, run_id, agent_name=agent_name, agent_model=agent_model, trace_link=incoming
    )
    link = saved or validate_trace_link(notifier_trace_link(notifier, run_id))
    if (
        link is not None
        and saved is None
        and store is not None
        and capabilities(store).fenced_writes
    ):
        await store.set_trace_link(
            run_id, claim_epoch=getattr(store, "claim_epoch", 0), trace_link=link
        )
    return link, _TraceNotifier(notifier) if notifier is not None else None


def _validate_loop_skill_compat(agent: Agent, resolved_loop: Loop) -> None:
    """Reject SingleCall + tools/tool_sources when loop is overridden at run time.

    Skills are allowed on SingleCall — they're delivered inline by
    ``Agent.get_system_prompt`` (inlined skill mode). Tools and tool
    sources are rejected because tool execution is inherently multi-turn
    (the LLM emits a call, the loop runs it, the LLM sees the result),
    and SingleCall is one LLM call by definition. Mirrors the
    construction-time checks in ``Agent._validate``.
    """
    from dendrux.loops.single import SingleCall

    if isinstance(resolved_loop, SingleCall) and agent.tools:
        tool_names = [getattr(fn, "__name__", str(fn)) for fn in agent.tools]
        raise ValueError(
            f"Agent '{agent.name}' has {len(agent.tools)} tool(s) {tool_names} "
            f"but is being run with SingleCall loop. SingleCall agents cannot "
            f"have tools — tool execution requires multi-turn iteration. "
            f"Skills are still supported on SingleCall (delivered inline)."
        )

    if isinstance(resolved_loop, SingleCall) and agent._tool_sources:
        raise ValueError(
            f"Agent '{agent.name}' has MCP tool_sources but is being run with "
            f"SingleCall loop. SingleCall does not support MCP tool discovery."
        )


def _cause_type(error: BaseException) -> str | None:
    """Class name of the wrapped discovery failure, for governance payloads."""
    cause = error.__cause__
    return type(cause).__name__ if cause is not None else None


def _mcp_error_payload(error: _MCPDiscoveryError) -> dict[str, Any]:
    """Payload of the run-fatal mcp.error event, with a stable ``code``."""
    # Deferred: dendrux.mcp needs the optional mcp extra, and discovery only
    # fails when the Agent has MCP tool sources.
    from dendrux.mcp._errors import MCPAuthorizationRequired, error_code_for

    cause = error.__cause__ or error
    payload: dict[str, Any] = {
        "error": str(error)[:500],
        "error_type": _cause_type(error),
        "code": error_code_for(cause),
    }
    if isinstance(cause, MCPAuthorizationRequired):
        payload["reason"] = cause.reason
    return payload


def _snapshot_mcp_meta(agent: Agent) -> dict[str, Any]:
    return {
        "mcp_skipped_sources": deepcopy(agent._mcp_skipped_sources),
        "mcp_skipped_tools": deepcopy(agent._mcp_skipped_tools),
    }


def _attach_mcp_meta(result: RunResult, mcp_meta: dict[str, Any]) -> None:
    result.meta.update(deepcopy(mcp_meta))


def _attach_run_error_meta(exc: Exception, run_id: str, mcp_meta: dict[str, Any]) -> None:
    # Preserve provider exception identity and catch compatibility, including
    # exceptions whose __setattr__ disallows assigning new attributes.
    exc.__dict__.update(run_id=run_id, run_meta=deepcopy(mcp_meta))


class _MCPDiscoveryError(Exception):
    """Wrapper raised only when MCP discovery fails inside _emit_init_events.

    Lets callers distinguish MCP discovery failures from skill emission
    failures so mcp.error is emitted only for actual discovery problems.
    """


async def _emit_init_governance_event(
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    event_type: str,
    data: dict[str, Any],
) -> None:
    """Emit a single init governance event to recorder + notifier."""
    from dendrux.loops._helpers import notify_governance, record_governance

    await record_governance(recorder, run_id, event_type, 0, data)
    await notify_governance(notifier, run_id, event_type, 0, data)


def _local_tool_names(agent: Agent) -> list[str]:
    """Bound code-declared names; discovery/provider catalogs are never authority."""
    from itertools import islice

    from dendrux.tool import get_tool_def

    names = (get_tool_def(fn).name for fn in islice(agent.tools, 1000))
    return sorted({name for name in names if len(name) <= 64 and name.isidentifier()})


async def _emit_declared_tools(
    agent: Agent,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
) -> None:
    await _emit_init_governance_event(
        recorder, notifier, run_id, "tools.declared", {"local_tool_names": _local_tool_names(agent)}
    )


async def _emit_init_events(
    agent: Agent,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    *,
    resolved_loop: Loop | None = None,
    mcp_meta: dict[str, Any] | None = None,
) -> None:
    """Emit skill and MCP init governance events.

    Called inside the runner try block after run.started and delegation
    context is set.

    ``resolved_loop`` is forwarded to ``agent.get_tool_lookups()`` so
    MCP discovery + use_skill injection respect a runtime loop override.

    Skills: reads the bound run session, established before prompt resolution.

    MCP: if agent._tool_sources is non-empty, forces discovery via
    get_tool_lookups(), then groups discovered tools by their
    Agent-visible namespace (falling back to source_name).
    Discovery failures are wrapped in _MCPDiscoveryError so callers
    can distinguish them from skill emission failures.
    """
    await _emit_declared_tools(agent, recorder, notifier, run_id)
    session = current(agent) or SkillSession.prepare(agent, loop=resolved_loop)
    session.run_id = run_id

    async def skill_sink(event: str, data: dict[str, Any], call_id: str | None) -> None:
        await _emit_init_governance_event(recorder, notifier, run_id, event, data)

    session.sink = skill_sink
    await session.start()

    if not agent._tool_sources and mcp_meta is not None:
        mcp_meta["mcp_skipped_sources"] = []
        mcp_meta["mcp_skipped_tools"] = []

    # MCP events — only when tool_sources exist
    if agent._tool_sources:
        try:
            await agent.get_tool_lookups(loop=resolved_loop)  # force discovery
        except Exception as exc:
            raise _MCPDiscoveryError(str(exc)) from exc

        if mcp_meta is not None:
            mcp_meta.update(_snapshot_mcp_meta(agent))

        # Initialize from all sources so zero-tool and failed optional sources
        # still produce explicit, auditable events. Tools are grouped by the
        # Agent-visible namespace (== each tool source's name); an aliased
        # runtime view namespaces its tools differently from its MCP source.
        source_tools: dict[str, list[str]] = {src.name: [] for src in agent._tool_sources}
        for td in agent._discovered_tool_defs or []:
            group = td.meta.get("namespace") or td.meta.get("source_name", "unknown")
            source_tools.setdefault(group, []).append(td.name)

        skipped_sources = {entry["namespace"]: entry for entry in agent._mcp_skipped_sources}
        for source in agent._tool_sources:
            configured_name = getattr(getattr(source, "source", None), "name", None)
            source_name = configured_name if isinstance(configured_name, str) else source.name
            source_error = getattr(source, "last_error", None)
            if isinstance(source_error, str) and source_error:
                skipped_source = skipped_sources.get(source.name, {})
                payload: dict[str, Any] = {
                    "source_name": source_name,
                    "namespace": source.name,
                    "error": source_error,
                    "error_type": skipped_source.get("error_type"),
                    "code": skipped_source.get("code", "unknown"),
                    "failure_mode": getattr(source, "failure_mode", "strict"),
                }
                if "reason" in skipped_source:
                    payload["reason"] = skipped_source["reason"]
                await _emit_init_governance_event(
                    recorder,
                    notifier,
                    run_id,
                    GovernanceEventType.MCP_ERROR,
                    payload,
                )
                continue

            tool_names = source_tools[source.name]
            await _emit_init_governance_event(
                recorder,
                notifier,
                run_id,
                GovernanceEventType.MCP_CONNECTED,
                {
                    "source_name": source_name,
                    "namespace": source.name,
                    "tool_count": len(tool_names),
                    "tool_names": tool_names,
                },
            )

        if agent._mcp_skipped_tools:
            await _emit_init_governance_event(
                recorder,
                notifier,
                run_id,
                GovernanceEventType.MCP_TOOLS_SKIPPED,
                {"skipped": deepcopy(agent._mcp_skipped_tools)},
            )


async def _emit_event(
    state_store: StateStore | None,
    run_id: str,
    event_type: str,
    data: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Record a durable run-level lifecycle event. Fail-closed with retry.

    Runner lifecycle events (run.started, run.paused, run.completed,
    run.error, run.resumed, run.cancelled) are authoritative truth —
    they get the same retry/propagate contract as PersistenceRecorder's
    fail-closed writes. The store allocates the event's sequence index in
    the insert, so every writer of a run (runner, recorder, ``cancel_run``,
    another process) gets the next index in commit order.
    """
    if state_store is None:
        return

    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.durability import retry_critical

    identity: dict[str, Any] = (
        {"event_id": generate_ulid()} if capabilities(state_store).retry_ids else {}
    )

    async def _write() -> None:
        from dendrux.runtime.state import validate_event_index, validate_event_writer

        validate_event_writer(state_store)
        index = await state_store.save_run_event(
            run_id,
            event_type=event_type,
            correlation_id=correlation_id,
            data=data,
            **identity,
        )
        validate_event_index(index)

    await retry_critical(_write, label=f"emit_{event_type}", run_id=run_id)


async def _emit_event_safe(
    state_store: StateStore | None,
    run_id: str,
    event_type: str,
    data: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Best-effort lifecycle event — for use inside streaming error handlers.

    In streaming generators, the error handler itself must not raise
    (the stream contract promises no exception escapes). This variant
    logs and swallows persistence failures instead of propagating.
    Used ONLY in the ``except`` block of streaming generators.
    """
    if state_store is None:
        return
    try:
        await _emit_event(state_store, run_id, event_type, data, correlation_id)
    except RunFencedError:
        raise
    except Exception:
        logger.warning(
            "Failed to record %s for run %s (in error handler)",
            event_type,
            run_id,
            exc_info=safe_exception_info(),
        )


_WAITING_STATUSES = (
    RunStatus.WAITING_CLIENT_TOOL,
    RunStatus.WAITING_HUMAN_INPUT,
    RunStatus.WAITING_APPROVAL,
)
# Every status a loop result can durably pause in.
_PAUSE_STATUSES = (*_WAITING_STATUSES, RunStatus.PAUSED)


async def _attach_undelivered_notes(
    state_store: StateStore, run_id: str, result: RunResult
) -> None:
    """List inbox notes the run ended without delivering in ``meta["undelivered_notes"]``.

    Only set when there are any, so runs that never get notes are unchanged.
    """
    get_notes = getattr(state_store, "get_notes", None)
    if get_notes is None:  # pre-0.3 store: no inbox
        return
    try:
        notes = await get_notes(run_id, undelivered_only=True)
    except Exception:
        logger.warning(
            "Failed to read undelivered notes for run %s", run_id, exc_info=safe_exception_info()
        )
        return
    if notes:
        result.meta["undelivered_notes"] = [n.id for n in notes]


async def _pause_request_message(state_store: StateStore, run_id: str) -> str | None:
    """The ``reason=`` text of the latest ``request_pause`` on this run, if any."""
    from dendrux.runtime.state import PAUSE_REQUEST_META_KEY

    try:
        record = await state_store.get_run(run_id)
    except Exception:
        logger.warning(
            "Failed to read pause request for run %s", run_id, exc_info=safe_exception_info()
        )
        return None
    request = ((record.meta if record else None) or {}).get(PAUSE_REQUEST_META_KEY)
    reason = request.get("reason") if isinstance(request, dict) else None
    return reason if isinstance(reason, str) else None


@asynccontextmanager
async def _closing_loop_stream(
    loop_stream: AsyncGenerator[RunEvent, None], run_id: str
) -> AsyncIterator[AsyncGenerator[RunEvent, None]]:
    """Close the loop generator in this task, never raising out of the close.

    When the consumer abandons the stream, ``GeneratorExit`` is travelling
    through the runner generator. An exception escaping the loop's cleanup
    (a provider's close failing, say) would land in the runner's
    ``except Exception`` and make it *yield* while closing — which Python
    reports to the consumer as ``async generator ignored GeneratorExit``.
    Cleanup failures are logged instead; the run is still finalized by the
    stream's cleanup callback.
    """
    try:
        yield loop_stream
    finally:
        try:
            await loop_stream.aclose()
        except Exception:
            logger.warning(
                "Loop stream cleanup raised for run %s", run_id, exc_info=safe_exception_info()
            )


def _track_partial_text(shared: dict[str, Any], event: RunEvent) -> None:
    """Buffer streamed text so an abandoned stream can persist a partial answer.

    The buffer holds the current LLM call's deltas; it resets when a tool
    result passes, since a new call follows.
    """
    if event.type == RunEventType.TEXT_DELTA:
        shared.setdefault("partial_text", []).append(event.text or "")
    elif event.type == RunEventType.TOOL_RESULT:
        shared.pop("partial_text", None)


def _partial_answer(shared: dict[str, Any]) -> str | None:
    parts = shared.get("partial_text")
    return "".join(parts) if parts else None


def _provider_error_info(exc: Exception) -> ErrorInfo | None:
    from dendrux.errors import ProviderError, _database_failure_in

    return (
        ErrorInfo.from_error(exc)
        if isinstance(exc, ProviderError)
        else _database_error_info(exc, _database_failure_in(exc))
    )


def _database_error_info(exc: Exception, failed: bool) -> ErrorInfo | None:
    from dendrux.errors import PersistenceError

    if not failed:
        return None
    return ErrorInfo(
        category="persistence_error",
        retryable=getattr(exc, "retryable", False) is True,
        safe_message=PersistenceError.safe_message,
        error_type="PersistenceError",
    )


def _safe_error(exc: Exception) -> str:
    info = _provider_error_info(exc)
    return info.safe_message if info else _nonprovider_error(exc)


def _nonprovider_error(exc: Exception) -> str:
    from dendrux.errors import PersistenceError, _database_failure_in

    return PersistenceError.safe_message if _database_failure_in(exc) else str(exc)


def _error_payload(exc: Exception) -> dict[str, Any]:
    info = _provider_error_info(exc)
    return {
        "error": _safe_error(exc)[:500],
        **({"error_info": info.to_dict()} if info else {}),
        **_database_error_code(exc),
    }


def _database_error_code(exc: Exception) -> dict[str, Any]:
    from dendrux.errors import _database_failure_in

    return {"code": "persistence_error"} if _database_failure_in(exc) else {}


def _error_store_kwargs(store: StateStore, exc: Exception) -> dict[str, Any]:
    info = _provider_error_info(exc)
    base = unclaimed_store(store)
    if info and base is not None and _accepts_kwarg(base.finalize_run, "error_info"):
        return {"error_info": info.to_dict()}
    return {}


def _error_info_from_meta(meta: dict[str, Any] | None) -> ErrorInfo | None:
    data = (meta or {}).get("dendrux.error_info")
    return ErrorInfo.from_dict(data) if isinstance(data, dict) else None


def _public_meta(meta: dict[str, Any]) -> dict[str, Any]:
    result = dict(meta)
    result.pop("pause_state", None)
    mapping = result.get("pii_mapping")
    if isinstance(mapping, dict):
        result["pii_mapping"] = {k: v for k, v in mapping.items() if k != "__dendrux_origins__"}
    return result


def _in_memory_pending(result: RunResult) -> Any:
    """Project an ephemeral pause without ever serializing its history."""
    from dendrux.runtime._pending import _KIND_BY_STATUS, _pending_from, approval_calls
    from dendrux.types import Clarification

    state = result.meta.get("pause_state")
    kind = _KIND_BY_STATUS.get(result.status.value)
    if state is None or kind is None:
        return None
    pause = {
        "pause_id": state.pause_id,
        "pending_tool_calls": [
            {"id": c.id, "name": c.name, "params": c.params} for c in state.pending_tool_calls
        ],
        "pending_targets": state.pending_targets,
        "approval_gates": state.approval_gates,
        "pause_reason": state.pause_reason,
        "pause_message": state.pause_message,
        "budget_exceeded": state.budget_exceeded,
    }
    if kind in ("approval", "input"):
        pause["pending_tool_calls"] = approval_calls(pause["pending_tool_calls"], None)
    last = None
    if state.steps and isinstance(state.steps[-1].action, Clarification):
        action = state.steps[-1].action
        last = {
            "action": {
                "type": "clarification",
                "question": action.question,
                "options": action.options,
            }
        }
    return _pending_from(result.run_id, result.status.value, kind, pause, last)


async def _recovery_info(store: StateStore, record: Any) -> Any:
    """Use the same bounded durable projection for direct and read results."""
    from dendrux.types import RecoveryInfo

    if record is None or not getattr(record, "reclaim_attempts", 0):
        return None
    ids, truncated = await store.get_replayed_tool_calls(
        record.id, limit=100, tenant_id=record.tenant_id
    )
    return RecoveryInfo(record.reclaim_attempts, record.last_reclaimed_at, ids, truncated)


async def _attach_persisted_pending(store: StateStore, result: RunResult, record: Any) -> RunResult:
    result.recovery = await _recovery_info(store, record)
    if result.status in _PAUSE_STATUSES and record is not None:
        from dendrux.runtime._pending import _PendingReader

        result.pending = await _PendingReader(store).get_pending(
            result.run_id, tenant_id=getattr(record, "tenant_id", None)
        )
        result.pause_id = result.pending.pause_id if result.pending else None
    return result


def _ensure_pause_generation(
    result: RunResult, provider_kwargs: dict[str, Any] | None = None
) -> None:
    if result.status in _PAUSE_STATUSES:
        pause = result.meta.get("pause_state")
        if pause is not None:
            if pause.pause_id is None:
                pause.pause_id = generate_ulid()
            pause.provider_kwargs = {
                k: v
                for k, v in (provider_kwargs or pause.provider_kwargs).items()
                if k in {"model", "max_tokens", "temperature", "thinking", "effort"}
            }
            result.pause_id = pause.pause_id
            result.pending = _in_memory_pending(result)


def _pause_event_data(result: RunResult, pause_state: PauseState) -> dict[str, Any]:
    """Build the existing pause lifecycle event without changing its shape."""
    paused_data: dict[str, Any] = {
        "status": result.status.value,
        "pause_id": pause_state.pause_id,
        "pending_tool_calls": [
            {
                "id": tc.id,
                "name": tc.name,
                "target": pause_state.pending_targets.get(tc.id),
                "params": tc.params,
            }
            for tc in pause_state.pending_tool_calls
        ],
    }
    if pause_state.pause_reason is not None:
        paused_data["reason"] = pause_state.pause_reason
    if pause_state.pause_message:
        paused_data["message"] = pause_state.pause_message[:500]
    if pause_state.budget_exceeded:
        paused_data["exceeded"] = pause_state.budget_exceeded
    return paused_data


def _terminal_failure_kwargs(state_store: StateStore, result: RunResult) -> dict[str, Any]:
    """Keep guardrail reason support optional for legacy custom stores."""
    failure_kwargs: dict[str, Any] = {}
    base = unclaimed_store(state_store)
    if (
        base is not None
        and result.status == RunStatus.ERROR
        and result.meta.get("failure_reason") in ("guardrail_blocked", "pending_limit")
        and _accepts_kwarg(base.finalize_run, "failure_reason")
    ):
        failure_kwargs["failure_reason"] = result.meta["failure_reason"]
    return failure_kwargs


async def _persist_loop_outcome(
    *,
    state_store: StateStore,
    run_id: str,
    result: RunResult,
    provider_kwargs: dict[str, Any] | None = None,
    recorder: LoopRecorder | None = None,
) -> RunResult:
    """Persist a loop's terminal/pause result and emit the lifecycle event.

    Centralizes pause-vs-finalize handling across initial run, retry,
    resume, and their streaming variants. Implements the cooperative
    cancellation pre-pause checkpoint: if ``cancel_requested`` was set
    during the iteration that just returned WAITING_*, we finalize as
    ``cancelled`` instead of pausing — otherwise the run would settle
    in a paused state holding pause_data the user already asked us to
    abandon.

    Always returns a fresh ``RunResult``; never mutates the input.
    ``meta['_finalize_won']`` records whether the CAS finalize won
    (``True`` for pauses since pause is unconditional).

    ``RunResult.answer`` returned to the caller is deanonymized using
    ``meta['pii_mapping']`` when one is present — the dev sees real values
    ("Nice to meet you, anmol!") instead of placeholders. The DB row keeps
    the raw LLM output so the dashboard's audit view stays placeholder-faithful.
    """
    from dendrux.guardrails._engine import deanonymize_text
    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.state import _capture_finalized_run
    from dendrux.types import RunResult

    recovery = None

    async def _final_recovery(records: Any) -> Any:
        if not capabilities(state_store).recovery:
            return None
        record = records[-1] if records else await state_store.get_run(run_id)
        return await _recovery_info(state_store, record)

    pii_mapping = result.meta.get("pii_mapping") or {}
    user_answer, unmapped_placeholders = deanonymize_text(result.answer, pii_mapping)

    def _wrap(*, status: RunStatus, won: bool) -> RunResult:
        pause_id, pending = (
            (result.pause_id, result.pending) if status in _PAUSE_STATUSES else (None, None)
        )
        return RunResult(
            run_id=run_id,
            status=status,
            answer=user_answer,
            output=result.output,
            error_info=result.error_info,
            error=result.error,
            steps=result.steps,
            iteration_count=result.iteration_count,
            usage=result.usage,
            meta={**_public_meta(result.meta), "_finalize_won": won},
            commentary=[deanonymize_text(c, pii_mapping)[0] or c for c in result.commentary],
            replay=result.replay,
            stop_reason=result.stop_reason if status == RunStatus.SUCCESS else None,
            pause_id=pause_id,
            pending=pending,
            recovery=recovery,
        )

    async def _persisted(status_if_missing: RunStatus) -> RunResult:
        """The persisted winner after a lost CAS (never the loop's proposal)."""
        record = await state_store.get_run(run_id)
        status = RunStatus(record.status) if record is not None else status_if_missing
        persisted = _wrap(status=status, won=False)
        if record is not None:
            persisted.answer = record.answer
            persisted.output = None
            persisted.error = record.error
            persisted.error_info = _error_info_from_meta(record.meta)
        if status not in _PAUSE_STATUSES:
            persisted.meta.pop("pause_state", None)
            await _attach_undelivered_notes(state_store, run_id, persisted)
        return await _attach_persisted_pending(state_store, persisted, record)

    async def _finalize_cancelled(
        allowed: list[str], payload: dict[str, Any], failure_reason: str | None = None
    ) -> RunResult:
        nonlocal recovery
        with _capture_finalized_run(unclaimed_store(state_store), run_id) as records:
            won = await _transition_with_event(
                state_store,
                run_id,
                status=RunStatus.CANCELLED.value,
                allowed_current_statuses=allowed,
                event_type="run.cancelled",
                event_data=payload,
                answer=user_answer,
                iteration_count=result.iteration_count,
                total_usage=result.usage,
                pii_mapping=result.meta.get("pii_mapping"),
                failure_reason=failure_reason,
            )
        if not won:
            return await _persisted(RunStatus.CANCELLED)
        recovery = await _final_recovery(records)
        cancelled = _wrap(status=RunStatus.CANCELLED, won=True)
        cancelled.meta.pop("pause_state", None)
        await _attach_undelivered_notes(state_store, run_id, cancelled)
        return cancelled

    if result.status == RunStatus.CANCELLED and result.meta.get("budget_exceeded"):
        # Budget cancel from the loop's checkpoint: distinct event reason and
        # failure_reason so it never reads as a user cancellation.
        return await _finalize_cancelled(
            [RunStatus.RUNNING.value],
            {"reason": BUDGET_EXCEEDED, "exceeded": result.meta["budget_exceeded"]},
            failure_reason=BUDGET_EXCEEDED,
        )

    if result.status == RunStatus.CANCELLED:
        payload: dict[str, Any] = {"reason": "cancel_requested"}
        if result.meta.get("interrupted"):
            payload["interrupted"] = True
            payload["partial_output"] = bool(result.meta.get("partial_output"))
        return await _finalize_cancelled(
            [s.value for s in _WAITING_STATUSES] + [RunStatus.RUNNING.value], payload
        )

    if result.status in _PAUSE_STATUSES:
        # Pre-pause checkpoint as one CAS: the pause persists only while the
        # row is RUNNING with no pending cancel. Cancellation wins over pause;
        # any durable pause (incl. WAITING_*) satisfies a pause request. The
        # pause and its run.paused event commit together, so a finalizer of
        # paused runs (cancel_run, sweep) always acts after the event.
        await _check_claim(state_store, run_id)
        _ensure_pause_generation(result, provider_kwargs)
        pause_state: PauseState = result.meta["pause_state"]
        # A pause generation belongs to this transition, never to the prior
        # pause from which this execution was resumed.
        pause_state = replace(pause_state, pause_id=generate_ulid())
        result.meta["pause_state"] = pause_state
        if pause_state.pause_reason == "requested" and pause_state.pause_message is None:
            pause_state = replace(
                pause_state, pause_message=await _pause_request_message(state_store, run_id)
            )
            result.meta["pause_state"] = pause_state
        paused_data = _pause_event_data(result, pause_state)
        record = await state_store.get_run(run_id)
        checkpoint = getattr(recorder, "checkpoint_outcome", None)
        if checkpoint is not None:
            pause_state.history[:] = [
                await project_message(recorder, m) for m in pause_state.history
            ]
            await checkpoint(result, provider_kwargs)
        pause_data = {**pause_state.to_dict(), "status": result.status.value}
        if record is not None and GOVERNANCE_KEY in (getattr(record, "meta", None) or {}):
            pause_data[GOVERNANCE_KEY] = (record.meta or {})[GOVERNANCE_KEY]
        paused = await _pause_with_event(
            state_store,
            run_id,
            status=result.status.value,
            pause_data=pause_data,
            event_data=paused_data,
            iteration_count=result.iteration_count,
            pii_mapping=result.meta.get("pii_mapping"),
            total_usage=result.usage,
        )
        if not paused:
            # CAS lost: honor the winner. On a still-RUNNING row only a
            # pending cancel makes the CAS miss, so finalize cancelled.
            record = await state_store.get_run(run_id)
            if record is not None and record.status != RunStatus.RUNNING.value:
                return await _persisted(RunStatus(record.status))
            return await _finalize_cancelled(
                [RunStatus.RUNNING.value], {"reason": "cancel_requested"}
            )
        result.pause_id = pause_state.pause_id
        # Return the persisted state: a finalizer may already have acted on
        # the durable pause (e.g. cancel_run right after it committed).
        record = await state_store.get_run(run_id)
        if record is not None and record.status != result.status.value:
            return await _persisted(RunStatus(record.status))
        return await _attach_persisted_pending(
            state_store, _wrap(status=result.status, won=True), record
        )

    # Terminal: success / max_iterations / error. Cooperative cancellation:
    # a cancel requested after the last checkpoint does not override a run
    # that finished — its work and side effects happened.
    _run_pii = result.meta.get("pii_mapping")
    failure_kwargs = _terminal_failure_kwargs(state_store, result)
    from dendrux.runtime.state import EventWrite

    stop_reason = result.stop_reason if result.status == RunStatus.SUCCESS else None
    event_type = "run.error" if result.status == RunStatus.ERROR else "run.completed"
    event_data = {
        **({"error": result.error} if result.status == RunStatus.ERROR else {}),
        "status": result.status.value,
        "stop_reason": stop_reason,
        "truncated": stop_reason == "max_tokens",
    }
    atomic_final = (
        getattr(state_store, "storage_format", 1) == 2
        and capabilities(state_store).atomic_completion
    )
    with _capture_finalized_run(unclaimed_store(state_store), run_id) as records:
        if atomic_final:
            event = EventWrite(
                generate_ulid(), event_type, result.iteration_count, event_data, None, None
            )
            final_trace = getattr(recorder, "finalization_trace", lambda: None)()
            finalize_won = (
                await state_store.finalize_with_event(
                    run_id,
                    status=result.status.value,
                    allowed_current_statuses=["running"],
                    answer=result.answer,
                    iteration_count=result.iteration_count,
                    total_usage=result.usage,
                    pii_mapping=_run_pii,
                    error=result.error,
                    event_type=event_type,
                    event_data=event_data,
                    trace=final_trace,
                    event=event,
                    **failure_kwargs,
                )
                is not None
            )
        else:
            finalize_won = await state_store.finalize_run(
                run_id,
                status=result.status.value,
                answer=result.answer,
                iteration_count=result.iteration_count,
                total_usage=result.usage,
                expected_current_status="running",
                pii_mapping=_run_pii,
                error=result.error,
                **failure_kwargs,
            )
    if not finalize_won:
        return await _persisted(result.status)
    recovery = await _final_recovery(records)
    if unmapped_placeholders:
        await _emit_event(
            state_store,
            run_id,
            "guardrail.unmapped_placeholder",
            {"placeholders": unmapped_placeholders},
        )
    if stop_reason is not None:
        await _record_stop_reason(state_store, run_id, stop_reason)
    if not atomic_final:
        await _emit_event(state_store, run_id, event_type, event_data)
    finished = _wrap(status=result.status, won=True)
    await _attach_undelivered_notes(state_store, run_id, finished)
    return finished


async def _record_stop_reason(state_store: StateStore, run_id: str, stop_reason: str) -> None:
    """Best-effort ``meta[STOP_REASON_META_KEY]`` write after a won finalize.

    A store without ``update_run_meta`` or a failed write only means an
    idempotent replay of this run reports ``stop_reason=None``.
    """
    update = getattr(state_store, "update_run_meta", None)
    if update is None:
        return
    try:
        await update(run_id, {STOP_REASON_META_KEY: stop_reason})
    except Exception:
        logger.warning(
            "Failed to record stop_reason for run %s", run_id, exc_info=safe_exception_info()
        )


async def _transition_with_event(
    state_store: StateStore,
    run_id: str,
    *,
    status: str,
    allowed_current_statuses: list[str],
    event_type: str,
    event_data: dict[str, Any],
    **finalize_kwargs: Any,
) -> bool:
    """CAS-finalize and record the transition event atomically when the store can.

    Pre-0.3 stores lack ``finalize_with_event``: fall back to the CAS
    followed by a separate event write. Their CAS cannot store a
    ``failure_reason``; the transition event still carries the reason.
    """
    atomic = getattr(state_store, "finalize_with_event", None)
    # Stores predating a keyword (e.g. failure_reason) cannot persist it;
    # the transition and its event (which carries the reason) still happen.
    if finalize_kwargs.get("failure_reason") is None or not _accepts_kwarg(
        atomic or state_store.finalize_run_if_status_in, "failure_reason"
    ):
        finalize_kwargs.pop("failure_reason", None)
    if atomic is not None:
        seq = await atomic(
            run_id,
            status=status,
            allowed_current_statuses=allowed_current_statuses,
            event_type=event_type,
            event_data=event_data,
            **finalize_kwargs,
        )
        return seq is not None
    won = await state_store.finalize_run_if_status_in(
        run_id,
        status=status,
        allowed_current_statuses=allowed_current_statuses,
        **finalize_kwargs,
    )
    if won:
        await _emit_event(state_store, run_id, event_type, event_data)
    return bool(won)


async def _pause_with_event(
    state_store: StateStore,
    run_id: str,
    *,
    status: str,
    pause_data: dict[str, Any],
    event_data: dict[str, Any],
    iteration_count: int | None,
    pii_mapping: dict[str, str] | None,
    total_usage: UsageStats | None = None,
) -> bool:
    """Persist a pause and its ``run.paused`` event; False when cancellation won.

    ``total_usage`` is written as the run's running totals so a paused run
    does not read as zero spend. Pre-0.3 stores lack ``pause_with_event``:
    fall back to the legacy cancel read, unconditional ``pause_run`` (with
    totals only when it accepts them) and a separate event write.
    """
    usage_kwargs: dict[str, Any] = {"total_usage": total_usage} if total_usage else {}
    atomic = getattr(state_store, "pause_with_event", None)
    if not _accepts_kwarg(atomic or state_store.pause_run, "total_usage"):
        usage_kwargs = {}
    if atomic is not None:
        seq = await atomic(
            run_id,
            status=status,
            pause_data=pause_data,
            event_data=event_data,
            iteration_count=iteration_count,
            pii_mapping=pii_mapping,
            **usage_kwargs,
        )
        return seq is not None
    if await state_store.is_cancel_requested(run_id):
        return False
    await state_store.pause_run(
        run_id,
        status=status,
        pause_data=pause_data,
        iteration_count=iteration_count,
        pii_mapping=pii_mapping,
        **usage_kwargs,
    )
    await _emit_event(state_store, run_id, "run.paused", event_data)
    # The read and the pause are separate here: a cancel committed in between
    # left a paused row with cancel_requested set. Finish it now; the caller
    # reloads and returns the persisted winner.
    if await state_store.is_cancel_requested(run_id):
        await _transition_with_event(
            state_store,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=[status],
            event_type="run.cancelled",
            event_data={"reason": "cancel_requested"},
        )
    return True


def _signature_params(fn: Any) -> Any:
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Whether ``fn`` takes keyword ``name`` (explicitly or via ``**kwargs``)."""
    params = _signature_params(fn)
    if params is None:
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _budget_carry_kwargs(loop_method: Any, pause_state: PauseState) -> dict[str, Any]:
    """``budget_carry=`` for a loop that accepts it (built-in loops); else nothing.

    Custom loops written against the pre-0.3 ``Loop`` signature keep working;
    they simply cannot restore fired budget keys.
    """
    params = _signature_params(loop_method)
    if params is None or "budget_carry" not in params:
        return {}
    return {"budget_carry": BudgetCarry.from_pause(pause_state)}


def _carry_budget_fields(new: PauseState, old: PauseState) -> PauseState:
    """Copy fired budget keys from the pause being resumed onto a runner-built pause."""
    return replace(
        new,
        budget_fired=old.budget_fired,
        budget_unpriced_reported=old.budget_unpriced_reported,
        budget_known_cost_usd=old.budget_known_cost_usd,
        budget_unpriced_calls=old.budget_unpriced_calls,
    )


def _carry_image_view(new: PauseState, old: PauseState) -> PauseState:
    """Carry the image view of the pause being resumed onto a runner-built pause.

    A runner-built pause is created between requests, so the view is
    unchanged. A pause predating any image (``image_view is None``) whose new
    history gained images (approved server tools) starts a view in which
    the old history's slots count as sent.
    """
    view = old.image_view
    if view is None:
        if not has_images(new.history):
            return new
        view = ImageViewState(sent_through=image_slot_count(old.history))
    return replace(new, image_view=ImageViewState.from_dict(view.to_dict()))


def _image_runtime(
    agent: Agent, state_store: StateStore | None, tenant_id: str | None, run_id: str | None
) -> ImageRuntime:
    """The per-run :class:`ImageRuntime` (explicit; never a ContextVar)."""
    from dendrux.media._policy import ImagePolicy
    from dendrux.media._runtime import ImageRuntime, tenant_scope_for

    state_store = unclaimed_store(state_store)
    backend = getattr(agent, "_image_backend", None)
    catalog, blob_store, resolver = backend(state_store) if backend else (None, None, None)
    gc = getattr(agent, "_image_gc", None)
    return ImageRuntime(
        collect_garbage=gc(state_store) if gc else None,
        policy=getattr(agent, "image_policy", None) or ImagePolicy(),
        catalog=catalog,
        blob_store=blob_store,
        resolver=resolver,
        tenant_scope=tenant_scope_for(tenant_id),
        run_id=run_id,
    )


def _recorder_image_kwargs(tenant_id: str | None, provider: LLMProvider) -> dict[str, Any]:
    """``tenant_scope=`` (edge scope) and ``provider=`` (image token estimates) for
    ``PersistenceRecorder`` when it takes them (§8)."""
    from dendrux.media._runtime import tenant_scope_for
    from dendrux.runtime.persistence import PersistenceRecorder

    init = PersistenceRecorder.__init__
    out: dict[str, Any] = {}
    if _accepts_kwarg(init, "tenant_scope"):
        out["tenant_scope"] = tenant_scope_for(tenant_id)
    if _accepts_kwarg(init, "provider"):
        out["provider"] = provider
    return out


def _image_loop_kwargs(
    loop_method: Any, runtime: ImageRuntime | None, state: ImageViewState | None
) -> dict[str, Any]:
    """``image_runtime=`` / ``image_view_state=`` for a loop method that accepts them."""
    out: dict[str, Any] = {}
    if runtime is not None and _accepts_kwarg(loop_method, "image_runtime"):
        out["image_runtime"] = runtime
        if state is not None and _accepts_kwarg(loop_method, "image_view_state"):
            out["image_view_state"] = state
    return out


def _require_image_loop(loop: Loop, method: str = "run") -> None:
    """A loop that cannot take ``image_runtime`` cannot run image input."""
    if not _accepts_kwarg(getattr(loop, method, None), "image_runtime"):
        raise TypeError(
            f"{type(loop).__name__}.{method}() does not accept image_runtime, so it "
            "cannot run image input; use ReActLoop or SingleCall, or add the parameter."
        )


def _tool_image_kwargs(fn: Any, runtime: ImageRuntime | None) -> dict[str, Any]:
    if runtime is None or not _accepts_kwarg(fn, "image_runtime"):
        return {}
    return {"image_runtime": runtime}


def _resume_image_view(pause_state: PauseState, history: list[Message]) -> ImageViewState | None:
    """The view a resumed run continues with; fresh (old history = sent) for older pauses."""
    if pause_state.image_view is not None:
        return pause_state.image_view
    if not has_images(history):
        return None
    return ImageViewState(sent_through=image_slot_count(pause_state.history))


async def _write_edges(
    runtime: ImageRuntime | None, run_id: str, edges: Sequence[BlobEdge]
) -> None:
    """Seed / user-trace edges via the catalog (idempotent per owner key)."""
    if not edges:
        return
    if runtime is None or runtime.catalog is None:
        from dendrux.errors import ImagePersistenceUnsupportedError

        raise ImagePersistenceUnsupportedError("state_store")
    await runtime.catalog.add_blob_refs(run_id, runtime.tenant_scope, list(edges))


async def _live_edges(runtime: ImageRuntime | None, edges: list[BlobEdge]) -> list[BlobEdge]:
    """Keep edges whose blob row is ``live`` (purged/missing refs degrade at request time)."""
    if not edges or runtime is None or runtime.catalog is None:
        return edges
    rows = await runtime.catalog.get_current(
        runtime.tenant_scope, list(dict.fromkeys(e.sha256 for e in edges))
    )
    return [e for e in edges if (row := rows.get(e.sha256)) is not None and row.state == "live"]


def _seed_edges(runtime: ImageRuntime, seeded: Sequence[Message]) -> list[BlobEdge]:
    """``seed`` edges: one per image slot of the seeded prefix (ordinal = slot)."""
    from dendrux.media._catalog import BlobEdge

    edges: list[BlobEdge] = []
    slot = 0
    for msg in seeded:
        if msg.role == Role.SYSTEM:
            continue
        for part in msg.images:
            edges.append(BlobEdge(runtime.tenant_scope, part.ref.sha256, "seed", "seed", slot))
            slot += 1
    return edges


async def _start_image_view(
    runtime: ImageRuntime | None,
    store: StateStore | None,
    run_id: str,
    *,
    seeded: Sequence[Message],
    sources: Sequence[tuple[str, str | None, dict[str, Any] | None]],
    single_full: bool,
    input_images: bool,
) -> ImageViewState | None:
    """New-run image view (§5.14) plus seed edges; None for a run without images.

    ``seeded`` is the prefix the run starts with (history=, replayed runs,
    retry rebuild); its slots all count as sent. ``sources`` are the replayed
    / retried runs whose image decisions carry over, in order.
    """
    if not has_images(seeded):
        return ImageViewState() if input_images else None
    if runtime is None:
        raise TypeError(
            "the loop does not accept image_runtime, so it cannot run a seeded prefix with images"
        )
    image_sources = (
        await load_image_sources(store, sources) if store is not None and sources else []
    )
    state = initial_image_view(
        sent_through=image_slot_count(seeded),
        sources=image_sources,
        tenant_scope=runtime.tenant_scope,
        single_full=single_full,
    )
    if store is not None or _memory_catalog(runtime, store) is not None:
        await _write_edges(
            runtime, run_id, await _live_edges(runtime, _seed_edges(runtime, seeded))
        )
    return state


def _memory_catalog(runtime: ImageRuntime | None, store: StateStore | None) -> Any:
    """The in-memory catalog of a run without a database (it tracks run edges)."""
    if store is not None or runtime is None or runtime.catalog is None:
        return None
    return runtime.catalog if hasattr(runtime.catalog, "release_run") else None


async def _register_memory_input(
    runtime: ImageRuntime | None,
    store: StateStore | None,
    run_id: str,
    user_parts: tuple[ContentPart, ...] | None,
) -> None:
    """No database: register the input's edges with the in-memory catalog (GC safety)."""
    if _memory_catalog(runtime, store) is None:
        return
    edges = _input_edge_kwargs(runtime, user_parts).get("blob_refs")
    if edges:
        await _write_edges(runtime, run_id, edges)


def _release_memory_edges(
    runtime: ImageRuntime | None, store: StateStore | None, run_id: str
) -> None:
    """No database: drop the run's in-memory edges once it ends (it cannot pause)."""
    catalog = _memory_catalog(runtime, store)
    if catalog is not None:
        try:
            catalog.release_run(run_id)
        except Exception:
            logger.warning(
                "Failed to release in-memory blob edges of run %s",
                run_id,
                exc_info=safe_exception_info(),
            )


class _EntryImages(NamedTuple):
    """Ingested image input of a new run (before ``create_run``)."""

    runtime: ImageRuntime
    user_parts: tuple[ContentPart, ...] | None
    history: list[Message]


async def _ingest_entry_images(
    agent: Agent,
    provider: LLMProvider,
    model: str,
    *,
    state_store: StateStore | None,
    tenant_id: str | None,
    run_id: str,
    items: tuple[Any, ...],
    history: list[ChatMessage] | None,
) -> _EntryImages:
    """Ingest run input + ``history=`` images before any run row exists (§5.2).

    The input's capability check runs between ingest phase 1 and phase 2, so a
    non-vision model rejects new images before any blob row is written.
    History images are seeded (already sent) and never capability-checked.
    """
    from dendrux.runtime.submit import (
        check_new_image_capability,
        ingest_groups,
        merge_user_items,
        user_items_group,
    )

    runtime = _image_runtime(agent, state_store, tenant_id, run_id)
    turns = [(i, msg.parts) for i, msg in enumerate(history or []) if msg.parts]
    image_count = sum(1 for item in items if not isinstance(item, str))

    async def _capability() -> None:
        await check_new_image_capability(
            provider, model, runtime.policy, image_count=image_count, origin="user"
        )

    # One phase 1 over the input and every history turn before any blob write.
    ingested = await ingest_groups(
        [user_items_group(items), *(user_items_group(parts) for _, parts in turns)],
        runtime=runtime,
        before_store=_capability,
    )
    user_parts = merge_user_items(items, ingested[0]) if items else None
    parts_for = {
        i: merge_user_items(parts, images)
        for (i, parts), images in zip(turns, ingested[1:], strict=True)
    }
    return _EntryImages(runtime, user_parts, normalize_chat_history(history, parts_for))


async def _record_user_trace(
    recorder: LoopRecorder | None, notifier: LoopNotifier | None, run_id: str, user_msg: Message
) -> None:
    """Record the run's own user message (order 0).

    Its image parts (``meta["dendrux.parts"]``) are edged by the recorder in
    the trace transaction (non-live blobs skipped there).
    """
    await record_message(recorder, run_id, user_msg, 0)
    await notify_message(notifier, run_id, user_msg, 0)


_UNSET_DEPTH = _UnsetType()


def _resolve_max_delegation_depth(
    requested: int | None | _UnsetType,
    parent_ctx: DelegationContext | None,
) -> int | None:
    """Resolve effective max_delegation_depth from explicit value + parent context.

    Pure policy helper — no I/O, no persistence, no parent linking.

    Resolution order:
      1. Explicit requested value (even None = unbounded)
      2. Inherit from parent context
      3. Default 10 (root run)

    Tighten-only rule: a child can never loosen the parent's limit.
    Warns (deduplicated per parent run) when an explicit child value is clamped.
    """
    # --- Validation ---
    if (
        not isinstance(requested, _UnsetType)
        and requested is not None
        and (not isinstance(requested, int) or requested < 0)
    ):
        raise ValueError(
            f"max_delegation_depth must be a non-negative integer or None, got {requested!r}"
        )

    # --- Resolution ---
    if not isinstance(requested, _UnsetType):
        effective = requested
    elif parent_ctx is not None:
        effective = parent_ctx.max_delegation_depth
    else:
        effective = 10

    # --- Tighten-only clamp ---
    if parent_ctx is not None and parent_ctx.max_delegation_depth is not None:
        parent_limit = parent_ctx.max_delegation_depth
        if effective is None or (isinstance(effective, int) and effective > parent_limit):
            # Warn only on explicit loosen attempts, deduplicated per parent run.
            if (
                not isinstance(requested, _UnsetType)
                and "depth_clamped" not in parent_ctx.warned_mismatches
            ):
                logger.warning(
                    "max_delegation_depth=%r clamped to %d (inherited parent limit)",
                    effective,
                    parent_limit,
                )
                parent_ctx.warned_mismatches.add("depth_clamped")
            effective = parent_limit

    return effective


@overload
async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    max_delegation_depth: int | None,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunResult: ...


@overload
async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunResult: ...


async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = None,
    context: list[ContextBlock] | None = None,
    activate_skills: Sequence[str] | None = None,
    runs: list[str] | None = None,
    replay_last: ReplayLast = 0,
    replay_reasoning: bool = True,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    tenant_id: str | None | _OmittedType = _OMITTED,
    metadata: dict[str, Any] | None = None,
    extra_notifier: LoopNotifier | None = None,
    max_delegation_depth: int | None | _UnsetType = _UNSET_DEPTH,
    idempotency_key: str | None = None,
    output_type: type | None = None,
    **kwargs: Any,
) -> RunResult:
    """Run an agent to completion.

    This is the primary API for executing a Dendrux agent. It wires
    together the agent definition, LLM provider, strategy, and loop,
    then executes the loop until completion.

    Args:
        agent: Agent definition (model, tools, prompt, limits).
        provider: LLM provider to use for this run.
        user_input: The user's input to process.
        strategy: Communication strategy. Defaults to NativeToolCalling.
        loop: Execution loop. Defaults to ReActLoop.
        state_store: Optional persistence backend. If provided, the run
            is persisted to the database with full traces.
        tenant_id: Optional tenant ID for multi-tenant isolation.
        metadata: Optional developer linking data (thread_id, user_id, etc.).
            Stored in agent_runs.meta — Dendrux stores it, never reads it.
        extra_notifier: Optional additional LoopNotifier for best-effort
            notifications (e.g. ConsoleNotifier).
        max_delegation_depth: Maximum allowed delegation depth for the
            run tree. Default 10. None means unbounded. Propagated to
            all child runs via contextvar. Raises DelegationDepthExceededError
            if a child run would exceed this depth.
        **kwargs: Reserved for future use.

    Returns:
        RunResult with status, answer, steps, and usage stats.

    Usage:
        from dendrux import Agent, tool, run
        from dendrux.llm import AnthropicProvider

        @tool()
        async def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(
            model="claude-sonnet-4-6",
            tools=[add],
            prompt="You are a calculator.",
        )
        provider = AnthropicProvider(api_key="sk-...", model="claude-sonnet-4-6")
        result = await run(agent, provider=provider, user_input="What is 15 + 27?")
        print(result.answer)
    """
    tenant_explicit = _tenant_scope_chosen(tenant_id)
    tenant_id = resolve_runner_tenant(agent, "run", tenant_id, state_store=state_store)
    skill_session = SkillSession.prepare(
        agent, loop=loop or agent.loop or ReActLoop(), explicit=activate_skills
    )
    with skill_session.bind():
        # Validate + resolve depth early — before any side effects.
        # _resolve_max_delegation_depth is a pure policy helper; side effect
        # is limited to dedup warning state on parent_ctx.warned_mismatches.
        parent_ctx = get_delegation_context()
        effective_max_depth = _resolve_max_delegation_depth(max_delegation_depth, parent_ctx)
        tenant_id = resolve_child_tenant(parent_ctx, tenant_id)
        runner_entry_checks(
            agent,
            "run",
            tenant_id=tenant_id,
            metadata=metadata,
            user_input=user_input,
            history=history,
            context=context,
            provider_kwargs=kwargs,
            idempotency_key=idempotency_key,
            provider=provider,
        )

        resolved_strategy = strategy or NativeToolCalling()
        resolved_loop = loop or agent.loop or ReActLoop()
        _validate_loop_skill_compat(agent, resolved_loop)

        # Validate input + chat history before any side effects. Raises
        # TypeError / ValueError on bad input so no DB row is created.
        user_input, input_items = split_user_content(user_input)
        validate_chat_history(history)
        image_input = bool(input_items) or history_has_images(history)
        if image_input:
            _require_image_loop(resolved_loop)
        normalized_history = [] if image_input else normalize_chat_history(history)
        replay_req = validate_replay_args(
            runs, replay_last, replay_reasoning, loop=resolved_loop, context=context
        )
        if replay_req is not None:
            validate_context(context)
            if state_store is None:
                raise ValueError(
                    "runs= requires persistence (database_url or state_store): "
                    "replay reads the earlier runs' traces."
                )
        else:
            validate_context(context)
        replay_runs = (
            await load_replay_runs(state_store, replay_req, tenant_id=tenant_id)
            if replay_req is not None and state_store is not None
            else []
        )
        if _replay_has_images(replay_runs):
            _require_image_loop(resolved_loop)
        replay_plan: ReplayPlan | None = None

        provider_kwargs = dict(kwargs) if kwargs else {}

        run_id = generate_ulid()

        # Images: ingest before the first durable write (§5.2). Typed errors
        # raise here, before any run row or claim exists.
        image_runtime: ImageRuntime | None = None
        user_parts: tuple[ContentPart, ...] | None = None
        if image_input:
            try:
                entry = await _ingest_entry_images(
                    agent,
                    provider,
                    provider_kwargs.get("model") or provider.model,
                    state_store=state_store,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    items=input_items,
                    history=history,
                )
            except (InvalidImageError, BlobNotFoundError) as exc:
                hit = await _idempotent_hit_after_image_error(
                    exc,
                    state_store,
                    idempotency_key,
                    tenant_id=tenant_id,
                    agent_name=agent.name,
                    user_input=user_input,
                    items=input_items,
                    history=history,
                    output_type=output_type,
                    context=context,
                    replay=replay_req.fingerprint() if replay_req else None,
                    skills=skill_session.fingerprint(),
                )
                if hit is None:
                    raise
                if hit.outcome == "existing_terminal":
                    assert state_store is not None  # a hit implies a store
                    return await _build_cached_result(state_store, hit.run_id, output_type)
                raise RunAlreadyActiveError(hit.run_id, hit.status) from None
            image_runtime, user_parts, normalized_history = entry
        elif _accepts_kwarg(resolved_loop.run, "image_runtime"):
            image_runtime = _image_runtime(agent, state_store, tenant_id, run_id)

        # When chat history is provided, the loop expects initial_history to
        # be the COMPLETE message list (loops use initial_history as-is and
        # do not separately append user_input — same contract as resume).
        # The runner will record the new user_input itself (after the recorder
        # is constructed) so seeded prior turns stay out of react_traces.
        # With runs=, the replayed prefix is folded in after expansion.
        if replay_req is None:
            seeded_history = fold_context(
                normalized_history,
                context,
                user_input,
                session=skill_session,
                user_parts=user_parts,
            )
        else:
            seeded_history = None
        recorder: LoopRecorder | None = None

        # --- Delegation context ---
        parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)

        # Resolve system prompt BEFORE create_run() so skill loading
        # failures don't leave a DB row stuck in running.
        _system_prompt = agent.get_system_prompt(loop=resolved_loop)

        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        lifecycle_started = False
        recorder_closed = False
        notifier_closed = False
        result: RunResult | None = None
        task_manager = getattr(agent, "_task_manager", None)
        registered = False
        owned = False

        try:
            try:
                if state_store is not None:
                    # Create the run record before the loop starts
                    # Merge loop type + depth limit into developer metadata
                    run_meta = dict(metadata) if metadata else {}
                    run_meta["dendrux.loop"] = type(resolved_loop).__name__
                    run_meta[GOVERNANCE_KEY] = governance_snapshot(agent)
                    if skill_session.configured:
                        run_meta[META_KEY] = skill_session.manifest()
                    if effective_max_depth is not None:
                        run_meta["dendrux.max_delegation_depth"] = effective_max_depth
                    if output_type is not None:
                        run_meta["dendrux.output_type"] = (
                            f"{output_type.__module__}.{output_type.__qualname__}"
                        )

                    # Compute idempotency fingerprint if key is provided
                    idem_fingerprint: str | None = None
                    if idempotency_key is not None:
                        output_type_name = (
                            f"{output_type.__module__}.{output_type.__qualname__}"
                            if output_type is not None
                            else None
                        )
                        idem_fingerprint = compute_idempotency_fingerprint(
                            agent.name,
                            user_input,
                            output_type_name=output_type_name,
                            history=normalized_history or None,
                            context=context or None,
                            replay=replay_req.fingerprint() if replay_req else None,
                            skills=skill_session.fingerprint(),
                            user_input_parts=user_parts,
                        )

                    seed_kwargs, run_guardrails = await _runtime_seeds(
                        state_store,
                        agent,
                        provider,
                        resolved_loop,
                        resolved_strategy,
                        provider_kwargs,
                        seeded_history,
                        user_input=user_input,
                        run_id=run_id,
                        tenant_id=tenant_id,
                        replay_req=replay_req,
                        replay_runs=replay_runs,
                        normalized_history=normalized_history,
                        context=context,
                        user_parts=user_parts,
                    )
                    create_result = await _create_runtime_run(
                        state_store,
                        resolved_loop,
                        run_id,
                        agent.name,
                        input_data=_input_data(user_input, user_parts),
                        model=provider.model,
                        strategy=type(resolved_strategy).__name__,
                        parent_run_id=parent_run_id,
                        delegation_level=delegation_level,
                        tenant_id=tenant_id,
                        meta=run_meta,
                        idempotency_key=idempotency_key,
                        idempotency_fingerprint=idem_fingerprint,
                        **_input_edge_kwargs(image_runtime, user_parts),
                        **seed_kwargs,
                    )

                    # Handle idempotency outcomes
                    if create_result.outcome == "existing_terminal":
                        return await _build_cached_result(
                            state_store, create_result.run_id, output_type
                        )
                    if create_result.outcome == "existing_active":
                        raise RunAlreadyActiveError(create_result.run_id, create_result.status)

                    # "created" — use the run_id from the result (same as generated)
                    run_id = create_result.run_id
                    state_store = claimed_store(
                        state_store,
                        run_id,
                        ClaimResult(
                            0,
                            create_result.storage_format,
                            _initial_loop_state(_recovery_loop_name(resolved_loop))
                            if create_result.storage_format == 2
                            else None,
                            None,
                        ),
                    )
                    owned = True

                    # The run is durable: from here cancel_run in this process
                    # interrupts this task.
                    if task_manager is not None:
                        task_manager.register_current(run_id)
                        registered = True

                    # Create persistence recorder
                    from dendrux.runtime.persistence import PersistenceRecorder
                    from dendrux.tool import get_tool_def

                    target_lookup = {}
                    for fn in agent.tools:
                        td = get_tool_def(fn)
                        target_lookup[td.name] = td.target
                    recorder = PersistenceRecorder(
                        state_store,
                        run_id,
                        model=provider.model,
                        provider_name=type(provider).__name__,
                        target_lookup=target_lookup,
                        seed_traces=seed_kwargs.get("seed_traces"),
                        guardrail_engine=run_guardrails,
                        **_recorder_image_kwargs(tenant_id, provider),
                    )

                await _emit_event(
                    state_store,
                    run_id,
                    "run.started",
                    {"agent_name": agent.name, "system_prompt": _system_prompt},
                )
                lifecycle_started = True
                await record_run_started(
                    recorder, run_id, agent_name=agent.name, agent_model=provider.model
                )
                trace_link, extra_notifier = await _start_trace(
                    extra_notifier,
                    state_store,
                    run_id,
                    agent_name=agent.name,
                    agent_model=provider.model,
                    parent=parent_ctx,
                )

                # Set delegation context for the duration of this run so nested
                # agent.run() calls inside tools inherit the parent link.
                this_ctx = DelegationContext(
                    run_id=run_id,
                    trace_link=trace_link,
                    delegation_level=delegation_level,
                    persisted=state_store is not None,
                    store_identity=get_store_identity(state_store),
                    max_delegation_depth=effective_max_depth,
                    tenant_id=tenant_id,
                    tenant_explicit=tenant_explicit,
                    state_store=state_store,
                    lease_interval_s=agent.lease_interval.total_seconds(),
                    loop=asyncio.get_running_loop(),
                )
                ctx_token = set_delegation_context(this_ctx)

                # Emit init governance events (skills + MCP) inside the try block
                # so discovery failures flow through the error cleanup path.
                # _MCPDiscoveryError is caught here, mcp.error emitted, then the
                # original cause is re-raised for the outer except Exception.
                try:
                    await _emit_init_events(
                        agent,
                        recorder,
                        extra_notifier,
                        run_id,
                        resolved_loop=resolved_loop,
                        mcp_meta=mcp_meta,
                    )
                except _MCPDiscoveryError as mcp_exc:
                    try:
                        await _emit_init_governance_event(
                            recorder,
                            extra_notifier,
                            run_id,
                            GovernanceEventType.MCP_ERROR,
                            _mcp_error_payload(mcp_exc),
                        )
                    except Exception:
                        logger.warning(
                            "Failed to emit mcp.error event", exc_info=safe_exception_info()
                        )
                    raise mcp_exc.__cause__ or mcp_exc from mcp_exc.__cause__

                replay_plan, seeded_history = await _apply_replay(
                    agent,
                    provider,
                    resolved_loop,
                    provider_kwargs,
                    state_store=state_store,
                    run_id=run_id,
                    replay_req=replay_req,
                    replay_runs=replay_runs,
                    normalized_history=normalized_history,
                    context=context,
                    user_input=user_input,
                    seeded_history=seeded_history,
                    user_parts=user_parts,
                )
                await _register_memory_input(image_runtime, state_store, run_id, user_parts)
                image_view = await _start_image_view(
                    image_runtime,
                    state_store,
                    run_id,
                    seeded=(seeded_history or [])[:-1],
                    sources=_replay_sources(replay_runs),
                    single_full=_single_full_source(replay_plan),
                    input_images=user_parts is not None,
                )

                # When chat history was provided, the loop receives a complete
                # message list as initial_history and skips its own user-message
                # recording. The runner records the new user_input itself so it
                # still lands in react_traces (seeded prior turns are NOT recorded
                # — they live in the dev's chat DB).
                if seeded_history is not None:
                    user_msg = user_trace_message(
                        user_input,
                        seeded_history,
                        render_band(context, "stable"),
                        context,
                        session=skill_session,
                    )
                    await _record_user_trace(recorder, extra_notifier, run_id, user_msg)

                result = await resolved_loop.run(
                    agent=agent,
                    provider=provider,
                    strategy=resolved_strategy,
                    user_input=user_input,
                    run_id=run_id,
                    recorder=recorder,
                    notifier=extra_notifier,
                    initial_history=seeded_history,
                    provider_kwargs=provider_kwargs or None,
                    output_type=output_type,
                    state_store=state_store,
                    **_image_loop_kwargs(resolved_loop.run, image_runtime, image_view),
                )

                if replay_plan is not None:
                    result.replay = replay_plan.summary
                _attach_mcp_meta(result, mcp_meta)
                _ensure_pause_generation(result, provider_kwargs)
                if state_store is not None:
                    result = await _persist_loop_outcome(
                        recorder=recorder,
                        provider_kwargs=provider_kwargs,
                        state_store=state_store,
                        run_id=run_id,
                        result=result,
                    )

                recorder_closed = True
                await record_run_finished(recorder, run_id, result)
                notifier_closed = True
                await notify_run_finished(extra_notifier, run_id, result)

                result.meta = _public_meta(result.meta)
                return result

            except RunFencedError:
                assert state_store is not None
                fenced_result = await _build_cached_result(state_store, run_id)
                if lifecycle_started:
                    finish_recorder = None if recorder_closed else recorder
                    finish_notifier = None if notifier_closed else extra_notifier
                    recorder_closed = notifier_closed = True
                    await _finish_fenced_lifecycle(
                        run_id, fenced_result, finish_recorder, finish_notifier
                    )
                return fenced_result

            except Exception as exc:
                if not lifecycle_started and isinstance(
                    exc, (RunAlreadyActiveError, IdempotencyConflictError)
                ):
                    raise
                _attach_run_error_meta(exc, run_id, mcp_meta)
                # Persist ERROR status before re-raising.
                # Conditional: only if still running (prevents cancel race).
                if state_store is not None and owned:
                    error_won = False
                    try:
                        error_won = await state_store.finalize_run(
                            run_id,
                            status=RunStatus.ERROR.value,
                            error=_safe_error(exc),
                            **_error_store_kwargs(state_store, exc),
                            total_usage=None,
                            expected_current_status="running",
                        )
                    except Exception:
                        logger.error(
                            "Failed to persist ERROR status for run %s",
                            run_id,
                            exc_info=safe_exception_info(),
                        )
                    # Only the CAS winner emits the error event
                    if error_won:
                        await _emit_event(state_store, run_id, "run.error", _error_payload(exc))
                if lifecycle_started:
                    if not recorder_closed:
                        recorder_closed = True
                        await record_run_failed(recorder, run_id, exc)
                    if not notifier_closed:
                        notifier_closed = True
                        await notify_run_failed(extra_notifier, run_id, exc)
                raise

        except asyncio.CancelledError:
            # cancel_run in this process cancels the registered task. Tell
            # that apart from the caller's own task.cancel(): only a
            # cancel_run interrupt that no other cancellation shares returns
            # a CANCELLED result; everything else re-raises as before.
            interrupted = False
            swallow = False
            if registered and task_manager is not None:
                interrupted, others_pending = task_manager.take_interrupt(run_id)
                current_task = asyncio.current_task()
                if interrupted and not others_pending and current_task is not None:
                    swallow = current_task.uncancel() == 0

            # Retain and drain the write task: shielding alone would let run()
            # return while cleanup is still pending after another cancellation.
            cleanup_task = asyncio.create_task(
                _persist_task_cancellation(
                    state_store if owned else None, run_id, interrupted=interrupted
                ),
                name=f"cancel-run:{run_id}",
            )
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            cancelled_result = cleanup_task.result()
            if result is not None and (
                state_store is None
                or (result.status == cancelled_result.status and result.meta.get("_finalize_won"))
            ):
                cancelled_result = result
            _attach_mcp_meta(cancelled_result, mcp_meta)

            # Keep callbacks in the run task so notifier ContextVar tokens remain
            # valid. Mark each sink before awaiting it to prevent duplicate hooks.
            if lifecycle_started:
                if not recorder_closed:
                    recorder_closed = True
                    try:
                        await record_run_finished(recorder, run_id, cancelled_result)
                    except Exception:
                        logger.warning(
                            "Recorder finish failed for run %s",
                            run_id,
                            exc_info=safe_exception_info(),
                        )
                if not notifier_closed:
                    notifier_closed = True
                    await notify_run_finished(extra_notifier, run_id, cancelled_result)
            current_task = asyncio.current_task()
            if swallow and (current_task is None or current_task.cancelling() == 0):
                return cancelled_result
            raise

        finally:
            _release_memory_edges(image_runtime, state_store, run_id)
            if registered and task_manager is not None:
                task_manager.release_current(run_id)
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


def _recovery_loop_name(loop: Loop) -> str | None:
    from dendrux.loops.react import ReActLoop
    from dendrux.loops.single import SingleCall

    if isinstance(loop, ReActLoop):
        return "ReActLoop"
    if isinstance(loop, SingleCall):
        return "SingleCall"
    return None


async def _create_runtime_run(
    store: StateStore, loop: Loop, run_id: str, agent_name: str, **kwargs: Any
) -> CreateRunResult:
    from dendrux.runtime.state import SQLAlchemyStateStore, _creation_loop_scope

    native_store = unclaimed_store(store)
    if isinstance(native_store, SQLAlchemyStateStore):
        with _creation_loop_scope(native_store, run_id, _recovery_loop_name(loop)):
            return await store.create_run(run_id, agent_name, **kwargs)
    return await store.create_run(run_id, agent_name, **kwargs)


async def _runtime_seeds(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    loop: Loop,
    strategy: Strategy,
    provider_kwargs: dict[str, Any],
    history: list[Message] | None,
    *,
    user_input: str,
    run_id: str,
    tenant_id: str | None = None,
    replay_req: ReplayRequest | None = None,
    replay_runs: list[Any] | None = None,
    normalized_history: list[Message] | None = None,
    context: list[ContextBlock] | None = None,
    user_parts: tuple[ContentPart, ...] | None = None,
) -> tuple[dict[str, Any], Any]:
    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.persistence import seed_trace_kwargs

    if not capabilities(store).checkpointing:
        return {}, None
    from dendrux.runtime.state import SQLAlchemyStateStore
    from dendrux.types import _validate_provider_kwargs

    native_store = unclaimed_store(store)
    if (
        isinstance(native_store, SQLAlchemyStateStore)
        and await native_store._creation_storage_format() == 2
    ):
        _validate_provider_kwargs(provider_kwargs, run_id=run_id)
    if replay_req is not None:
        plan = expand_replay(
            replay_runs or [],
            replay_req,
            current=current_prefix(agent, provider, loop, provider_kwargs),
            guardrails=bool(agent.guardrails),
        )
        history = fold_context(
            [*plan.messages, *(normalized_history or [])],
            context,
            user_input,
            session=current_skill_session(agent),
            user_parts=user_parts,
        )
    if not history:
        history = fold_context(
            [], context, user_input, session=current_skill_session(agent), user_parts=user_parts
        )
    if history is None:
        history = [Message(role=Role.USER, content=user_input)]
    from dendrux.media._runtime import tenant_scope_for

    messages = [
        Message(role=Role.SYSTEM, content=agent.get_system_prompt(loop=loop)),
        *(history or []),
    ]
    own_input = (
        user_trace_message(
            user_input,
            history[: max(i for i, m in enumerate(history) if m.role == Role.USER) + 1],
            render_band(context, "stable"),
            context,
            session=current_skill_session(agent),
        )
        if history and any(m.role == Role.USER for m in history)
        else None
    )
    from dendrux.guardrails import GuardrailEngine

    engine = GuardrailEngine(agent.guardrails, run_id=run_id) if agent.guardrails else None
    if engine is not None:
        messages = [
            (await engine.project_incoming_message(m, index=i))[0] for i, m in enumerate(messages)
        ]
    if engine is not None and own_input is not None:
        own_input = (await engine.project_incoming_message(own_input))[0]
        if "wire_content" in own_input.meta:
            own_index = max(i for i, m in enumerate(messages) if m.role == Role.USER)
            own_input = replace(
                own_input, meta={**own_input.meta, "wire_content": messages[own_index].content}
            )
    return {
        **seed_trace_kwargs(
            store, messages, tenant_scope=tenant_scope_for(tenant_id), own_input=own_input
        ),
        "pii_mapping": engine.get_pii_mapping() if engine else None,
    }, engine


async def _apply_replay(
    agent: Agent,
    provider: LLMProvider,
    loop: Loop,
    provider_kwargs: dict[str, Any],
    *,
    state_store: StateStore | None,
    run_id: str,
    replay_req: ReplayRequest | None,
    replay_runs: list[Any],
    normalized_history: list[Message],
    context: list[ContextBlock] | None,
    user_input: str,
    seeded_history: list[Message] | None,
    user_parts: tuple[ContentPart, ...] | None = None,
) -> tuple[ReplayPlan | None, list[Message] | None]:
    """Expand ``runs=`` into the seeded history and record the prefix descriptor.

    Runs after MCP discovery so the tools hash covers the full tool set.
    Without persistence nothing is recorded and nothing is replayed.
    """
    if state_store is None:
        return None, seeded_history
    current = current_prefix(agent, provider, loop, provider_kwargs)
    plan: ReplayPlan | None = None
    if replay_req is not None:
        plan = expand_replay(
            replay_runs, replay_req, current=current, guardrails=bool(agent.guardrails)
        )
        seeded_history = fold_context(
            [*plan.messages, *normalized_history],
            context,
            user_input,
            session=current_skill_session(agent),
            user_parts=user_parts,
        )
    await record_run_prefix(
        state_store,
        run_id,
        current=current,
        seed=build_seed(plan, normalized_history),
        plan=plan,
        seeded_image_slots=image_slot_count((seeded_history or [])[:-1]),
    )
    return plan, seeded_history


def _stream_error_meta(mcp_meta: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    """``RUN_ERROR`` result meta; an image rejection also lists the rejected image ids."""
    from dendrux.errors import ImageRejectedError

    meta = {**deepcopy(mcp_meta), "error_type": type(exc).__name__}
    if isinstance(exc, ImageRejectedError):
        meta["image_rejected"] = list(exc.image_ids)
    return meta


def _input_data(user_input: str, user_parts: tuple[ContentPart, ...] | None) -> dict[str, Any]:
    """``agent_runs.input_data``: the text, plus raw parts for a multimodal input."""
    if user_parts is None:
        return {"input": user_input}
    return {"input": user_input, PARTS_KEY: parts_to_dicts(user_parts)}


def _input_edge_kwargs(
    runtime: ImageRuntime | None, user_parts: tuple[ContentPart, ...] | None
) -> dict[str, Any]:
    """``blob_refs`` for ``create_run`` (``input`` edges), only when there are images."""
    if runtime is None or not user_parts:
        return {}
    from dendrux.runtime.submit import parts_edges

    edges = parts_edges(
        user_parts, tenant_scope=runtime.tenant_scope, owner_kind="input", owner_id="input"
    )
    if not edges:
        return {}
    return {"blob_refs": edges, "blob_refs_scope": runtime.tenant_scope}


def _stored_parts(raw: Any) -> tuple[ContentPart, ...] | None:
    """Parts persisted as dicts, or None when absent or text-only."""
    if not isinstance(raw, list) or not raw:
        return None
    parts = tuple(content_part_from_dict(d) for d in raw)
    return parts if any(isinstance(p, ImagePart) for p in parts) else None


async def _idempotent_hit_after_image_error(
    exc: Exception,
    store: StateStore | None,
    idempotency_key: str | None,
    *,
    tenant_id: str | None,
    agent_name: str,
    user_input: str,
    items: tuple[Any, ...],
    history: list[ChatMessage] | None,
    output_type: type | None,
    context: list[ContextBlock] | None,
    replay: dict[str, Any] | None,
    skills: Any,
) -> Any:
    """The existing run of a keyed request whose image was purged or is gone since.

    Ingest runs before the idempotency lookup, so a retried keyed request whose
    image has since been purged cannot be re-validated. When the key's run
    stored the same input (same text items, same image positions, same
    caller-given refs), the fingerprint is rebuilt from that run's stored
    descriptors and the store's normal idempotency check decides (cached
    terminal result / active / conflict). Image bytes that can no longer be
    read cannot be compared, so a request differing only in which purged
    image it sends is treated as the same request (documented).
    """
    purged = isinstance(exc, InvalidImageError) and exc.reason == "purged"
    if (
        idempotency_key is None
        or store is None
        or not (purged or isinstance(exc, BlobNotFoundError))
    ):
        return None
    if history_has_images(history) or not items:
        return None
    check = getattr(store, "_check_idempotency", None)
    if check is None:
        return None
    try:
        existing = await check(idempotency_key, "", tenant_id=tenant_id)
    except IdempotencyConflictError as found:
        existing_id = found.run_id
    else:
        return existing
    conflict = IdempotencyConflictError(run_id=existing_id, idempotency_key=idempotency_key)
    if existing_id is None:  # the existing run is not disclosed (another tenant)
        raise conflict
    record = await store.get_run(existing_id)
    stored = _stored_parts((record.input_data or {}).get(PARTS_KEY)) if record else None
    if stored is None or len(stored) != len(items):
        raise conflict
    candidate: list[ContentPart] = []
    for item, part in zip(items, stored, strict=True):
        if isinstance(item, str):
            if not isinstance(part, TextPart) or part.text != item:
                raise conflict
            candidate.append(part)
            continue
        if not isinstance(part, ImagePart):
            raise conflict
        ref = getattr(item, "ref", item)
        if isinstance(ref, ImageRef) and ref.sha256 != part.ref.sha256:
            raise conflict
        candidate.append(ImagePart(part.ref, detail=getattr(item, "detail", None)))
    fingerprint = compute_idempotency_fingerprint(
        agent_name,
        user_input,
        output_type_name=(
            f"{output_type.__module__}.{output_type.__qualname__}" if output_type else None
        ),
        history=normalize_chat_history(history) or None,
        context=context or None,
        replay=replay,
        skills=skills,
        user_input_parts=candidate,
    )
    return await check(idempotency_key, fingerprint, tenant_id=tenant_id)


def _replay_sources(replay_runs: Sequence[Any]) -> list[tuple[str, str | None, Any]]:
    """``(run_id, tenant_id, meta)`` of each replayed run, in order."""
    return [(r.record.id, r.record.tenant_id, r.record.meta) for r in replay_runs]


def _replay_has_images(replay_runs: Sequence[Any]) -> bool:
    """Whether any replayed run's input or traces carry image parts."""
    for run in replay_runs:
        if _stored_parts((run.record.input_data or {}).get(PARTS_KEY)) is not None:
            return True
        if any(_stored_parts((t.meta or {}).get(PARTS_KEY)) is not None for t in run.traces):
            return True
    return False


def _single_full_source(plan: ReplayPlan | None) -> bool:
    """Exactly one replayed run, seeded in full (its slots map 1:1)."""
    return plan is not None and len(plan.summary.runs_full) == 1 and not plan.summary.runs_text


async def _persist_task_cancellation(
    state_store: StateStore | None,
    run_id: str,
    *,
    interrupted: bool = False,
) -> RunResult:
    """Finalize a still-RUNNING run as CANCELLED after its task was cancelled.

    ``interrupted`` marks a ``cancel_run`` interrupt of a batch ``run()``:
    the result and the ``run.cancelled`` event carry ``interrupted=True`` and
    ``partial_output=False`` (a batch provider call has no partial text).
    """
    from dendrux.types import RunResult

    interrupt_meta = {"interrupted": True, "partial_output": False} if interrupted else {}
    result = RunResult(
        run_id=run_id,
        status=RunStatus.CANCELLED,
        meta={**interrupt_meta, "dendrux.cancel_persisted": False},
    )
    if state_store is None:
        return result
    try:
        won = await state_store.finalize_run(
            run_id,
            status=RunStatus.CANCELLED.value,
            expected_current_status="running",
        )
        if won:
            result.meta["dendrux.cancel_persisted"] = True
            await _emit_event(
                state_store,
                run_id,
                "run.cancelled",
                {"reason": "cancel_requested", **interrupt_meta}
                if interrupted
                else {"reason": "task_cancelled"},
            )
            await _attach_undelivered_notes(state_store, run_id, result)
        elif await state_store.get_run(run_id) is not None:
            result = await _build_cached_result(state_store, run_id)
            if result.status == RunStatus.CANCELLED:
                result.meta["dendrux.cancel_persisted"] = True
        # No row: cancelled before create_run committed, or while resolving an
        # idempotent hit under the provisional id. Nothing to finalize.
    except Exception:
        _codes.log_code(logger, _codes.CANCEL_PERSIST_FAILED, run_id=run_id)
    return result


async def _raise_resume_claim_failure(
    state_store: StateStore, run_id: str, expected_status: str
) -> None:
    """Raise a descriptive error when resume claim fails.

    Checks the actual run status to give a specific message for swept
    (abandoned/stale) runs instead of the generic "claimed by another caller."
    """
    run = await state_store.get_run(run_id)
    if run is not None and run.status == "error" and run.failure_reason:
        reason = run.failure_reason
        if reason == "abandoned_waiting":
            raise ValueError(
                f"Run '{run_id}' was abandoned (swept as expired waiting run). "
                f"It cannot be resumed."
            )
        if reason in ("stale_running", "never_started"):
            raise ValueError(f"Run '{run_id}' was swept as stale ({reason}). It cannot be resumed.")
    raise ValueError(
        f"Run '{run_id}' is not in status '{expected_status}' — "
        f"cannot resume. It may have been claimed by another caller."
    )


def _cached_output(answer: str | None, output_type: type | None, run_id: str) -> Any:
    """Recover only a configured structured schema, failing closed on old data."""
    # Rebuild typed output from persisted answer if output_type was requested
    output = None
    if output_type is not None and answer:
        from pydantic import ValidationError

        try:
            output = output_type.model_validate_json(answer)  # type: ignore[attr-defined]
        except (ValidationError, Exception):
            # If the persisted answer doesn't validate (schema evolved,
            # corruption, etc.), return None rather than crashing the
            # dedup path. The caller still gets answer as raw text.
            logger.warning(
                "Cached run %s answer failed validation against %s",
                run_id,
                output_type.__name__,
            )

    return output


async def _build_cached_result(
    state_store: StateStore,
    run_id: str,
    output_type: type | None = None,
) -> RunResult:
    """Rebuild a summary RunResult from persisted DB state for idempotent dedup.

    This is a *summary*, not a faithful replay of the original RunResult:
    - steps is always [] (AgentStep history is not persisted in recoverable form)
    - developer meta is omitted; known PII is retained for safe wire projection
    - stop_reason is restored from ``meta[STOP_REASON_META_KEY]`` on SUCCESS
      (None for runs finished before 0.4.1)
    - usage is reconstructed from aggregate columns (total_input_tokens, etc.),
      which may be zero for errored runs where finalize_run got total_usage=None

    When output_type is provided, re-validates the persisted answer against
    the schema to populate result.output. This ensures the idempotent path
    returns the same typed result as a fresh run.

    The contract is: "return the cached final outcome" — status, answer, error,
    iteration count, and aggregate usage. Not "return the identical object."
    """
    from dendrux.types import RunResult

    run = await state_store.get_run(run_id)
    if run is None:
        raise RuntimeError(f"Idempotent run {run_id} not found in database")

    output = _cached_output(run.answer, output_type, run_id)

    result = RunResult(
        run_id=run.id,
        status=RunStatus(run.status),
        answer=run.answer,
        output=output,
        steps=[],
        iteration_count=run.iteration_count,
        usage=(
            _usage_from_dict(run.output_data["usage"])
            if run.output_data and isinstance(run.output_data.get("usage"), dict)
            else UsageStats(
                input_tokens=run.total_input_tokens,
                output_tokens=run.total_output_tokens,
                total_tokens=run.total_input_tokens + run.total_output_tokens,
                cost_usd=run.total_cost_usd,
                reasoning_tokens=run.total_reasoning_tokens,
            )
        ),
        error=run.error,
        error_info=_error_info_from_meta(run.meta),
        replay=_replay_from_meta(run.meta),
        stop_reason=(
            _stop_reason_from_meta(run.meta) if run.status == RunStatus.SUCCESS.value else None
        ),
    )

    mapping = await state_store.get_pii_mapping(run_id)
    if mapping:
        result.meta = _public_meta({"pii_mapping": mapping})
    return await _attach_persisted_pending(state_store, result, run)


def _stop_reason_from_meta(meta: dict[str, Any] | None) -> StopReason | None:
    value = (meta or {}).get(STOP_REASON_META_KEY)
    return value if value in _STOP_REASONS else None


def _replay_from_meta(meta: dict[str, Any] | None) -> ReplaySummary | None:
    data = (meta or {}).get(REPLAY_META_KEY)
    return ReplaySummary.from_dict(data) if isinstance(data, dict) else None


async def retry(
    original_run_id: str,
    *,
    agent: Agent,
    provider: LLMProvider,
    state_store: StateStore,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    tenant_id: str | None | _OmittedType = _OMITTED,
    metadata: dict[str, Any] | None = None,
    extra_notifier: LoopNotifier | None = None,
    **kwargs: Any,
) -> RunResult:
    """Retry a terminal run with approximate prior context.

    Creates a fresh run seeded with the original run's conversation history
    from persisted traces. The LLM sees the prior conversation and can
    continue from context. This is NOT resume (no pause_data, no exact
    state reconstruction) — it's a new run with approximate context.

    Works for any terminal run (error, cancelled, max_iterations, success).
    Source run must have been executed with a retry-capable loop (not
    SingleCall). The retry agent can differ from the original — different
    model, different tools — the only requirement is that the source run
    has persisted traces to seed from.

    Args:
        original_run_id: The terminal run to retry from.
        agent: Agent definition for the retry run. Can differ from the
            original agent (different model, tools, prompt).
        provider: LLM provider for the retry run.
        state_store: Persistence backend (required — retry needs traces).
        strategy: Communication strategy. Defaults to NativeToolCalling.
        loop: Execution loop for the retry. Defaults to ReActLoop.
        tenant_id: Optional tenant ID.
        metadata: Optional developer metadata for the retry run.
        extra_notifier: Optional notifier.
        **kwargs: Forwarded to the LLM provider.

    Returns:
        RunResult from the retry run.

    Raises:
        RunNotFoundError: The source run does not exist or belongs to
            another tenant.
        ValueError: If the source run is not terminal, if the source
            run was SingleCall, or if no traces exist.
    """
    # 1. Validate source run is terminal. A missing run and another tenant's
    # run raise the same RunNotFoundError (B13), so retry is no existence oracle.
    tenant_explicit = _tenant_scope_chosen(tenant_id)
    tenant_id = resolve_runner_tenant(agent, "retry", tenant_id, state_store=state_store)
    from dendrux.errors import RunNotFoundError

    tenant_id = resolve_child_tenant(get_delegation_context(), tenant_id)
    runner_entry_checks(
        agent,
        "retry",
        tenant_id=tenant_id,
        metadata=metadata,
        history=kwargs.get("history"),
        context=kwargs.get("context"),
        provider_kwargs=kwargs,
        provider=provider,
    )
    source_run = await state_store.get_run(original_run_id)
    if source_run is None:
        raise RunNotFoundError(original_run_id)

    if source_run.tenant_id != tenant_id:
        raise RunNotFoundError(original_run_id)
    explicit = (source_run.meta or {}).get(META_KEY, {}).get("explicit", [])
    skill_session = SkillSession.prepare(
        agent, loop=loop or agent.loop or ReActLoop(), explicit=explicit
    )
    with skill_session.bind():
        terminal_statuses = {"success", "error", "cancelled", "max_iterations"}
        if source_run.status not in terminal_statuses:
            raise ValueError(
                f"Run '{original_run_id}' is in status '{source_run.status}' — "
                f"only terminal runs (error, cancelled, max_iterations, success) can be retried."
            )

        # 2. Validate source run was not SingleCall (no iterative history to seed)
        from dendrux.runtime._capabilities import capabilities

        source_loop = (source_run.meta or {}).get("dendrux.loop", "")
        if source_run.storage_format == 2 and capabilities(state_store).checkpointing:
            source_state = await state_store.get_loop_state(original_run_id)
            source_loop = (source_state or {}).get("loop")
        if source_loop in ("SingleCall", "single_call"):
            raise ValueError(
                "SingleCall runs cannot be retried with context — "
                "they have no iterative history to seed. Call agent.run() instead."
            )

        resolved_loop = loop or agent.loop or ReActLoop()
        _validate_loop_skill_compat(agent, resolved_loop)

        # 3. Read traces and reconstruct approximate history. Reasoning replay
        #    blocks only seed a retry on the model that made them.
        from dendrux.runtime.replay import conversation_records

        traces = (
            await conversation_records(state_store, original_run_id, limits=agent._limits)
            if source_run.storage_format == 2 and capabilities(state_store).checkpointing
            else await state_store.get_traces(original_run_id)
        )
        retry_model = (kwargs or {}).get("model") or provider.model
        initial_history = _pair_tool_calls(rebuild_messages(traces, reasoning_for=retry_model))
        if source_run.storage_format == 2:
            initial_history = [m for m in initial_history if m.role != Role.SYSTEM]

        if not initial_history:
            raise ValueError(
                f"Run '{original_run_id}' has no persisted traces. "
                "Cannot retry without prior context."
            )

        # The source's own input follows its persisted seed prefix.
        users = [t for t in traces if t.role == "user"]
        raw_user = next(
            (t for t in users if not (t.meta or {}).get("dendrux.seed_prefix")), users[0]
        )
        user_input = (source_run.input_data or {}).get("input", raw_user.content)
        source_parts = _stored_parts((source_run.input_data or {}).get(PARTS_KEY))

        if explicit:
            own_ordinal = users.index(raw_user)
            first_user = [i for i, m in enumerate(initial_history) if m.role == Role.USER][
                own_ordinal
            ]
            retry_context = [
                ContextBlock(**b) for b in (raw_user.meta or {}).get("dendrux.skills.context", [])
            ]
            first_msg = initial_history[first_user]
            if first_msg.parts is not None and source_parts is not None:
                retry_user = initial_history[first_user] = Message.with_parts(
                    Role.USER,
                    wire_user_parts(source_parts, skill_session.envelopes(), retry_context),
                    placement=first_msg.placement,
                )
            else:
                retry_user = initial_history[first_user] = replace(
                    first_msg,
                    content=wire_user_input(user_input, skill_session.envelopes(), retry_context),
                )

        # 4a. A source run that used runs= re-expands the same runs, same knobs.
        replay_req: ReplayRequest | None = None
        replay_runs: list[Any] = []
        source_replay = (source_run.meta or {}).get(REPLAY_META_KEY)
        if isinstance(source_replay, dict) and source_replay.get("runs"):
            replay_req = validate_replay_args(
                list(source_replay["runs"]),
                source_replay.get("replay_last", 0),
                bool(source_replay.get("replay_reasoning", True)),
                loop=resolved_loop,
                context=None,
            )
            if replay_req is not None:
                replay_runs = await load_replay_runs(state_store, replay_req, tenant_id=tenant_id)

        # 5. Create the retry run
        resolved_strategy = strategy or NativeToolCalling()
        provider_kwargs = dict(kwargs) if kwargs else {}

        run_id = generate_ulid()

        parent_ctx = get_delegation_context()
        parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)
        effective_max_depth = _resolve_max_delegation_depth(_UNSET_DEPTH, parent_ctx)

        _system_prompt = agent.get_system_prompt(loop=resolved_loop)

        run_meta = dict(metadata) if metadata else {}
        run_meta["dendrux.loop"] = type(resolved_loop).__name__
        run_meta[GOVERNANCE_KEY] = governance_snapshot(agent)
        if skill_session.configured:
            run_meta[META_KEY] = skill_session.manifest()
        run_meta["dendrux.retry_of"] = original_run_id

        image_runtime: ImageRuntime | None = None
        if (
            source_parts is not None
            or has_images(initial_history)
            or _replay_has_images(replay_runs)
        ):
            _require_image_loop(resolved_loop)
            image_runtime = _image_runtime(agent, state_store, tenant_id, run_id)
        elif _accepts_kwarg(resolved_loop.run, "image_runtime"):
            image_runtime = _image_runtime(agent, state_store, tenant_id, run_id)
        input_edges: dict[str, Any] = {}
        if source_parts is not None and image_runtime is not None:
            live = await _live_edges(
                image_runtime, _input_edge_kwargs(image_runtime, source_parts).get("blob_refs", [])
            )
            if live:
                input_edges = {"blob_refs": live, "blob_refs_scope": image_runtime.tenant_scope}

        seed_kwargs, run_guardrails = await _runtime_seeds(
            state_store,
            agent,
            provider,
            resolved_loop,
            resolved_strategy,
            provider_kwargs,
            initial_history,
            user_input=user_input,
            run_id=run_id,
            tenant_id=tenant_id,
        )
        create_result = await _create_runtime_run(
            state_store,
            resolved_loop,
            run_id,
            agent.name,
            input_data=_input_data(user_input, source_parts),
            model=provider.model,
            strategy=type(resolved_strategy).__name__,
            parent_run_id=parent_run_id,
            delegation_level=delegation_level,
            tenant_id=tenant_id,
            meta=run_meta,
            retry_of_run_id=original_run_id,
            **seed_kwargs,
            **input_edges,
        )
        run_id = create_result.run_id
        state_store = claimed_store(
            state_store,
            run_id,
            ClaimResult(
                0,
                create_result.storage_format,
                _initial_loop_state(_recovery_loop_name(resolved_loop))
                if create_result.storage_format == 2
                else None,
                None,
            ),
        )

        # Create persistence recorder
        from dendrux.runtime.persistence import PersistenceRecorder
        from dendrux.tool import get_tool_def

        target_lookup = {}
        for fn in agent.tools:
            td = get_tool_def(fn)
            target_lookup[td.name] = td.target
        recorder = PersistenceRecorder(
            state_store,
            run_id,
            model=provider.model,
            provider_name=type(provider).__name__,
            target_lookup=target_lookup,
            seed_traces=seed_kwargs.get("seed_traces"),
            guardrail_engine=run_guardrails,
            **_recorder_image_kwargs(tenant_id, provider),
        )

        await _emit_event(
            state_store,
            run_id,
            "run.started",
            {
                "agent_name": agent.name,
                "system_prompt": _system_prompt,
                "retry_of": original_run_id,
            },
        )
        await record_run_started(
            recorder, run_id, agent_name=agent.name, agent_model=provider.model
        )
        trace_link, extra_notifier = await _start_trace(
            extra_notifier,
            state_store,
            run_id,
            agent_name=agent.name,
            agent_model=provider.model,
            parent=parent_ctx,
        )

        this_ctx = DelegationContext(
            run_id=run_id,
            trace_link=trace_link,
            delegation_level=delegation_level,
            persisted=True,
            store_identity=get_store_identity(state_store),
            max_delegation_depth=effective_max_depth,
            tenant_id=tenant_id,
            tenant_explicit=tenant_explicit,
            state_store=state_store,
            lease_interval_s=agent.lease_interval.total_seconds(),
            loop=asyncio.get_running_loop(),
        )
        ctx_token = set_delegation_context(this_ctx)
        mcp_meta: dict[str, Any] = {}

        try:
            try:
                await _emit_init_events(
                    agent,
                    recorder,
                    extra_notifier,
                    run_id,
                    resolved_loop=resolved_loop,
                    mcp_meta=mcp_meta,
                )
            except _MCPDiscoveryError as mcp_exc:
                try:
                    await _emit_init_governance_event(
                        recorder,
                        extra_notifier,
                        run_id,
                        GovernanceEventType.MCP_ERROR,
                        _mcp_error_payload(mcp_exc),
                    )
                except Exception:
                    logger.warning("Failed to emit mcp.error event", exc_info=safe_exception_info())
                raise mcp_exc.__cause__ or mcp_exc from mcp_exc.__cause__

            current = current_prefix(agent, provider, resolved_loop, provider_kwargs)
            replay_plan: ReplayPlan | None = None
            if replay_req is not None:
                replay_plan = expand_replay(
                    replay_runs, replay_req, current=current, guardrails=bool(agent.guardrails)
                )
                if source_run.storage_format != 2 or not capabilities(state_store).checkpointing:
                    initial_history = [*replay_plan.messages, *initial_history]
            seeded_slots = image_slot_count(initial_history)
            await record_run_prefix(
                state_store,
                run_id,
                current=current,
                seed={"retry_of": original_run_id},
                plan=replay_plan,
                # An explicit-skills retry also traces its rebuilt first user
                # message, so its slot mapping is not a plain offset: unknown.
                seeded_image_slots=None if explicit and seeded_slots else seeded_slots,
            )
            # A retry has no new input: every seeded slot counts as sent.
            image_view = await _start_image_view(
                image_runtime,
                state_store,
                run_id,
                seeded=initial_history,
                sources=[
                    *_replay_sources(replay_runs),
                    (original_run_id, source_run.tenant_id, source_run.meta),
                ],
                single_full=replay_plan is None,
                input_images=False,
            )

            if explicit:
                trace_user = user_trace_message(
                    user_input,
                    [retry_user],
                    render_band(retry_context, "stable"),
                    retry_context,
                    session=skill_session,
                )
                await _record_user_trace(recorder, extra_notifier, run_id, trace_user)

            result = await resolved_loop.run(
                agent=agent,
                provider=provider,
                strategy=resolved_strategy,
                user_input=user_input,
                run_id=run_id,
                recorder=recorder,
                notifier=extra_notifier,
                initial_history=initial_history,
                provider_kwargs=provider_kwargs or None,
                state_store=state_store,
                **_image_loop_kwargs(resolved_loop.run, image_runtime, image_view),
            )

            if replay_plan is not None:
                result.replay = replay_plan.summary
            _attach_mcp_meta(result, mcp_meta)
            result = await _persist_loop_outcome(
                recorder=recorder,
                provider_kwargs=provider_kwargs,
                state_store=state_store,
                run_id=run_id,
                result=result,
            )

            await record_run_finished(recorder, run_id, result)
            await notify_run_finished(extra_notifier, run_id, result)

            return result

        except RunFencedError:
            fenced_result = await _build_cached_result(state_store, run_id)
            await _finish_fenced_lifecycle(run_id, fenced_result, recorder, extra_notifier)
            return fenced_result

        except Exception as exc:
            _attach_run_error_meta(exc, run_id, mcp_meta)
            error_won = False
            try:
                error_won = await state_store.finalize_run(
                    run_id,
                    status=RunStatus.ERROR.value,
                    error=_safe_error(exc),
                    **_error_store_kwargs(state_store, exc),
                    total_usage=None,
                    expected_current_status="running",
                )
            except Exception:
                logger.error(
                    "Failed to persist ERROR status for retry run %s",
                    run_id,
                    exc_info=safe_exception_info(),
                )
            if error_won:
                await _emit_event(state_store, run_id, "run.error", _error_payload(exc))
            await record_run_failed(recorder, run_id, exc)
            await notify_run_failed(extra_notifier, run_id, exc)
            raise

        finally:
            reset_delegation_context(ctx_token)


@overload
def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    state_store_resolver: Callable[[], Any] | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    max_delegation_depth: int | None,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunStream: ...


@overload
def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    state_store_resolver: Callable[[], Any] | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunStream: ...


def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = None,
    context: list[ContextBlock] | None = None,
    activate_skills: Sequence[str] | None = None,
    runs: list[str] | None = None,
    replay_last: ReplayLast = 0,
    replay_reasoning: bool = True,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    state_store_resolver: Callable[[], Any] | None = None,
    tenant_id: str | None | _OmittedType = _OMITTED,
    metadata: dict[str, Any] | None = None,
    extra_notifier: LoopNotifier | None = None,
    max_delegation_depth: int | None | _UnsetType = _UNSET_DEPTH,
    idempotency_key: str | None = None,
    output_type: type | None = None,
    **kwargs: Any,
) -> RunStream:
    """Stream an agent run as RunEvents, returning a RunStream.

    Synchronous — returns immediately with run_id available. All async
    setup (DB row, notifiers) runs lazily on first iteration. With
    ``idempotency_key``, the initial run_id is provisional until that setup
    resolves the key. A cached outcome yields one terminal event, without
    replaying text or tool events. Read its answer from ``stream.result``.

    Accepts either an already-resolved ``state_store`` or an async
    ``state_store_resolver`` callable. The resolver is called once in
    the generator preamble.

    Error semantics: exceptions are caught, persisted, and yielded as
    RUN_ERROR events. No exception is raised to the consumer.

    Lifecycle ownership:
      - Runner emits RUN_STARTED, RUN_ERROR, and handles cancellation cleanup.
      - Loop yields execution outcomes (RUN_COMPLETED, RUN_PAUSED) and
        intermediate events (TEXT_DELTA, TOOL_USE_*, TOOL_RESULT).
      - Runner persists loop outcomes before forwarding them.
    """
    # Validate depth early — synchronous, before generator starts.
    # Full resolution (parent context, clamping) happens inside _generate()
    # because parent_ctx requires async state_store resolution first.
    # But validation of the raw input can fail fast here.
    requested_tenant_id = tenant_id
    tenant_explicit = _tenant_scope_chosen(tenant_id)
    tenant_id = resolve_runner_tenant(agent, "stream", tenant_id, state_store=state_store)
    if (
        not isinstance(max_delegation_depth, _UnsetType)
        and max_delegation_depth is not None
        and (not isinstance(max_delegation_depth, int) or max_delegation_depth < 0)
    ):
        raise ValueError(
            f"max_delegation_depth must be a non-negative integer or None, "
            f"got {max_delegation_depth!r}"
        )

    from dendrux.types import RunEvent, RunResult
    from dendrux.types import RunStream as _RunStream

    # Validate input + chat history synchronously so TypeError / ValueError
    # surfaces to the caller immediately, not as a RUN_ERROR event. Image
    # ingest is async: it runs on first iteration, before create_run, and
    # its errors are raised from the first ``__anext__`` (no run row).
    entry_user_input = user_input  # as passed, for the runner entry checks
    user_input, input_items = split_user_content(user_input)
    validate_chat_history(history)
    image_input = bool(input_items) or history_has_images(history)
    normalized_history = [] if image_input else normalize_chat_history(history)
    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or agent.loop or ReActLoop()
    _validate_loop_skill_compat(agent, resolved_loop)
    if image_input:
        _require_image_loop(resolved_loop, "run_stream")
    replay_req = validate_replay_args(
        runs, replay_last, replay_reasoning, loop=resolved_loop, context=context
    )
    # When chat history is provided, build the complete starting message list
    # (loops use initial_history as-is and do not separately append user_input).
    # The new user_input is recorded by the runner itself inside _generate()
    # so seeded prior turns stay out of react_traces. With runs=, the replayed
    # prefix is folded in after expansion.
    skill_session = SkillSession.prepare(agent, loop=resolved_loop, explicit=activate_skills)
    with skill_session.bind():
        if replay_req is None and not image_input:
            initial_seeded = fold_context(
                normalized_history, context, user_input, session=skill_session
            )
        else:
            validate_context(context)
            initial_seeded = None
    provider_kwargs = dict(kwargs) if kwargs else {}

    run_id = generate_ulid()

    # Shared mutable state — generator populates on first iteration,
    # cleanup reads if the consumer abandons after setup.
    _shared: dict[str, Any] = {"state_store": state_store}

    mcp_meta: dict[str, Any] = {}

    stream_context = StreamContext(skill_session)

    async def _generate() -> AsyncGenerator[RunEvent, None]:
        """Inner generator — lazy async setup, then loop stream with lifecycle.

        The entire body is wrapped in try/except so that setup failures
        (state_store_resolver, create_run, notifier init) are also caught
        and yielded as RUN_ERROR — honoring the "no exception to consumer" contract.
        """
        # store is captured in the closure for the error handler.
        # Starts as the pre-provided value (often None for the lazy path).
        nonlocal run_id, extra_notifier
        store = state_store
        recorder: LoopRecorder | None = None
        seeded_history = initial_seeded
        replay_plan: ReplayPlan | None = None
        image_runtime: ImageRuntime | None = None
        user_parts: tuple[ContentPart, ...] | None = None
        history_messages = normalized_history

        # Entry checks after tenant inheritance, before any I/O. Like image
        # setup they raise (no RUN_ERROR event, no run row, no hooks). A child
        # whose explicit tenant differs from its parent's still fails later, as
        # RUN_ERROR, exactly as before.
        try:
            entry_tenant = resolve_child_tenant(get_delegation_context(), tenant_id)
        except ValueError:
            pass
        else:
            _shared["existing_run"] = True
            runner_entry_checks(
                agent,
                "stream",
                tenant_id=entry_tenant,
                metadata=metadata,
                user_input=entry_user_input,
                history=history,
                context=context,
                provider_kwargs=provider_kwargs,
                idempotency_key=idempotency_key,
                provider=provider,
            )
            _shared["existing_run"] = False

        if image_input:
            # Image setup raises (no RUN_ERROR event, no run row, no hooks).
            _shared["existing_run"] = True
            if store is None and state_store_resolver is not None:
                store = await state_store_resolver()
                resolve_runner_tenant(agent, "stream", requested_tenant_id, state_store=store)
            _shared["state_store"] = store
            try:
                entry = await _ingest_entry_images(
                    agent,
                    provider,
                    provider_kwargs.get("model") or provider.model,
                    state_store=store,
                    tenant_id=resolve_child_tenant(get_delegation_context(), tenant_id),
                    run_id=run_id,
                    items=input_items,
                    history=history,
                )
            except (InvalidImageError, BlobNotFoundError) as exc:
                hit = await _idempotent_hit_after_image_error(
                    exc,
                    store,
                    idempotency_key,
                    tenant_id=resolve_child_tenant(get_delegation_context(), tenant_id),
                    agent_name=agent.name,
                    user_input=user_input,
                    items=input_items,
                    history=history,
                    output_type=output_type,
                    context=context,
                    replay=replay_req.fingerprint() if replay_req else None,
                    skills=skill_session.fingerprint(),
                )
                if hit is None:
                    raise
                if hit.outcome != "existing_terminal":
                    raise RunAlreadyActiveError(hit.run_id, hit.status) from None
                assert store is not None
                cached = await _build_cached_result(store, hit.run_id, output_type)
                run_id = hit.run_id
                stream.run_id = run_id
                yield RunEvent(
                    type={
                        RunStatus.ERROR: RunEventType.RUN_ERROR,
                        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
                    }.get(cached.status, RunEventType.RUN_COMPLETED),
                    run_id=run_id,
                    run_result=cached,
                    error=cached.error if cached.status == RunStatus.ERROR else None,
                    phase="final_answer" if cached.status == RunStatus.SUCCESS else None,
                )
                return
            image_runtime, user_parts, history_messages = entry
            if replay_req is None:
                with skill_session.bind():
                    seeded_history = fold_context(
                        history_messages,
                        context,
                        user_input,
                        session=skill_session,
                        user_parts=user_parts,
                    )
            _shared["existing_run"] = False

        try:
            # 1. Resolve state store lazily (may create DB engine)
            if store is None and state_store_resolver is not None:
                store = await state_store_resolver()
                resolve_runner_tenant(agent, "stream", requested_tenant_id, state_store=store)
            _shared["state_store"] = store

            # --- Delegation context ---
            parent_ctx = get_delegation_context()
            parent_run_id, delegation_level = resolve_parent_link(parent_ctx, store)
            eff_max_depth = _resolve_max_delegation_depth(max_delegation_depth, parent_ctx)
            eff_tenant = resolve_child_tenant(parent_ctx, tenant_id)
            if image_runtime is None and _accepts_kwarg(resolved_loop.run_stream, "image_runtime"):
                image_runtime = _image_runtime(agent, store, eff_tenant, run_id)

            # Resolve system prompt BEFORE create_run() so skill loading
            # failures don't leave a DB row stuck in running.
            _system_prompt = agent.get_system_prompt(loop=resolved_loop)

            if idempotency_key is not None and store is None:
                raise ValueError(
                    "idempotency_key requires persistence (database_url or state_store"
                    "). Idempotency cannot work without durable state."
                )

            replay_runs: list[Any] = []
            if replay_req is not None:
                # Rejected before any row exists: no lifecycle hooks fire.
                _shared["existing_run"] = True
                if store is None:
                    raise ValueError(
                        "runs= requires persistence (database_url or state_store): "
                        "replay reads the earlier runs' traces."
                    )
                replay_runs = await load_replay_runs(store, replay_req, tenant_id=eff_tenant)
                if _replay_has_images(replay_runs):
                    _require_image_loop(resolved_loop, "run_stream")
                _shared["existing_run"] = False

            if store is not None:
                run_meta = dict(metadata) if metadata else {}
                run_meta["dendrux.loop"] = type(resolved_loop).__name__
                run_meta[GOVERNANCE_KEY] = governance_snapshot(agent)
                if skill_session.configured:
                    run_meta[META_KEY] = skill_session.manifest()
                if eff_max_depth is not None:
                    run_meta["dendrux.max_delegation_depth"] = eff_max_depth
                if output_type is not None:
                    run_meta["dendrux.output_type"] = (
                        f"{output_type.__module__}.{output_type.__qualname__}"
                    )
                fingerprint = None
                if idempotency_key is not None:
                    fingerprint = compute_idempotency_fingerprint(
                        agent.name,
                        user_input,
                        output_type_name=(
                            f"{output_type.__module__}.{output_type.__qualname__}"
                            if output_type is not None
                            else None
                        ),
                        history=history_messages or None,
                        context=context or None,
                        replay=replay_req.fingerprint() if replay_req else None,
                        skills=skill_session.fingerprint(),
                        user_input_parts=user_parts,
                    )
                try:
                    seed_kwargs, run_guardrails = await _runtime_seeds(
                        store,
                        agent,
                        provider,
                        resolved_loop,
                        resolved_strategy,
                        provider_kwargs,
                        seeded_history,
                        user_input=user_input,
                        run_id=run_id,
                        tenant_id=eff_tenant,
                        replay_req=replay_req,
                        replay_runs=replay_runs,
                        normalized_history=history_messages,
                        context=context,
                        user_parts=user_parts,
                    )
                    create_result = await _create_runtime_run(
                        store,
                        resolved_loop,
                        run_id,
                        agent.name,
                        input_data=_input_data(user_input, user_parts),
                        model=provider.model,
                        strategy=type(resolved_strategy).__name__,
                        parent_run_id=parent_run_id,
                        delegation_level=delegation_level,
                        tenant_id=eff_tenant,
                        meta=run_meta,
                        idempotency_key=idempotency_key,
                        idempotency_fingerprint=fingerprint,
                        **_input_edge_kwargs(image_runtime, user_parts),
                        **seed_kwargs,
                    )
                except IdempotencyConflictError:
                    # No row was created and no lifecycle opened; the error
                    # handler must not fire failure hooks for a run that
                    # never started (mirrors run(), which re-raises bare).
                    _shared["existing_run"] = True
                    raise
                _shared["existing_run"] = create_result.outcome != "created"
                run_id = create_result.run_id
                stream.run_id = run_id
                if create_result.outcome == "existing_terminal":
                    cached = await _build_cached_result(store, run_id, output_type)
                    event_type = {
                        RunStatus.ERROR: RunEventType.RUN_ERROR,
                        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
                    }.get(cached.status, RunEventType.RUN_COMPLETED)
                    yield RunEvent(
                        type=event_type,
                        run_id=run_id,
                        run_result=cached,
                        error=cached.error if event_type == RunEventType.RUN_ERROR else None,
                        phase="final_answer" if cached.status == RunStatus.SUCCESS else None,
                    )
                    return
                if create_result.outcome == "existing_active":
                    raise RunAlreadyActiveError(run_id, create_result.status)
                store = claimed_store(
                    store,
                    run_id,
                    ClaimResult(
                        0,
                        create_result.storage_format,
                        _initial_loop_state(_recovery_loop_name(resolved_loop))
                        if create_result.storage_format == 2
                        else None,
                        None,
                    ),
                )
                _shared["state_store"] = store
                _shared["claimed"] = True

                from dendrux.runtime.persistence import PersistenceRecorder
                from dendrux.tool import get_tool_def

                target_lookup = {}
                for fn in agent.tools:
                    td = get_tool_def(fn)
                    target_lookup[td.name] = td.target
                recorder = PersistenceRecorder(
                    store,
                    run_id,
                    model=provider.model,
                    provider_name=type(provider).__name__,
                    target_lookup=target_lookup,
                    seed_traces=seed_kwargs.get("seed_traces"),
                    guardrail_engine=run_guardrails,
                    **_recorder_image_kwargs(eff_tenant, provider),
                )

            # 2. Set delegation context for the generator lifetime
            this_ctx = DelegationContext(
                run_id=run_id,
                delegation_level=delegation_level,
                persisted=store is not None,
                store_identity=get_store_identity(store),
                max_delegation_depth=eff_max_depth,
                tenant_id=eff_tenant,
                tenant_explicit=tenant_explicit,
                state_store=store,
                lease_interval_s=agent.lease_interval.total_seconds(),
                loop=asyncio.get_running_loop(),
            )
            stream_context.set_delegation(this_ctx)

            # 3. Emit lifecycle start
            await _emit_event(
                store,
                run_id,
                "run.started",
                {"agent_name": agent.name, "system_prompt": _system_prompt},
            )
            await record_run_started(
                recorder, run_id, agent_name=agent.name, agent_model=provider.model
            )
            trace_link, extra_notifier = await _start_trace(
                extra_notifier,
                store,
                run_id,
                agent_name=agent.name,
                agent_model=provider.model,
                parent=parent_ctx,
            )
            this_ctx = replace(this_ctx, trace_link=trace_link)
            stream_context.set_delegation(this_ctx)
            # Lifecycle pair is now open. Stash recorder/notifier so the
            # cleanup callback can close it with a CANCELLED finish if the
            # consumer abandons the stream before a terminal event.
            _shared["recorder"] = recorder
            _shared["notifier"] = extra_notifier

            yield RunEvent(type=RunEventType.RUN_STARTED, run_id=run_id)

            # Emit init governance events (skills + MCP)
            try:
                await _emit_init_events(
                    agent,
                    recorder,
                    extra_notifier,
                    run_id,
                    resolved_loop=resolved_loop,
                    mcp_meta=mcp_meta,
                )
            except _MCPDiscoveryError as mcp_exc:
                try:
                    await _emit_init_governance_event(
                        recorder,
                        extra_notifier,
                        run_id,
                        GovernanceEventType.MCP_ERROR,
                        _mcp_error_payload(mcp_exc),
                    )
                except Exception:
                    logger.warning("Failed to emit mcp.error event", exc_info=safe_exception_info())
                raise mcp_exc.__cause__ or mcp_exc from mcp_exc.__cause__

            replay_plan, seeded_history = await _apply_replay(
                agent,
                provider,
                resolved_loop,
                provider_kwargs,
                state_store=store,
                run_id=run_id,
                replay_req=replay_req,
                replay_runs=replay_runs,
                normalized_history=history_messages,
                context=context,
                user_input=user_input,
                seeded_history=seeded_history,
                user_parts=user_parts,
            )
            await _register_memory_input(image_runtime, store, run_id, user_parts)
            image_view = await _start_image_view(
                image_runtime,
                store,
                run_id,
                seeded=(seeded_history or [])[:-1],
                sources=_replay_sources(replay_runs),
                single_full=_single_full_source(replay_plan),
                input_images=user_parts is not None,
            )

            # When chat history was provided, the loop receives a complete
            # message list as initial_history and skips its own user-message
            # recording. Record the new user_input here so it still lands in
            # react_traces (seeded prior turns are NOT recorded).
            if seeded_history is not None:
                user_msg = user_trace_message(
                    user_input,
                    seeded_history,
                    render_band(context, "stable"),
                    context,
                    session=skill_session,
                )
                await _record_user_trace(recorder, extra_notifier, run_id, user_msg)

            # 3. Stream the loop. The interrupt signal lets cancel_run reach
            # an in-flight provider stream owned by this process; the closing
            # wrapper guarantees the loop generator is finalized in this task
            # when the consumer abandons the stream.
            interrupt = agent._task_manager.register_interrupt(run_id)
            loop_stream = resolved_loop.run_stream(
                agent=agent,
                provider=provider,
                strategy=resolved_strategy,
                user_input=user_input,
                run_id=run_id,
                recorder=recorder,
                notifier=extra_notifier,
                initial_history=seeded_history,
                provider_kwargs=provider_kwargs or None,
                output_type=output_type,
                state_store=store,
                interrupt=interrupt,
                **_image_loop_kwargs(resolved_loop.run_stream, image_runtime, image_view),
            )
            async with _closing_loop_stream(loop_stream, run_id):
                async for event in loop_stream:
                    _terminal_types = (
                        RunEventType.RUN_COMPLETED,
                        RunEventType.RUN_ERROR,
                        RunEventType.RUN_PAUSED,
                        RunEventType.RUN_CANCELLED,
                    )
                    terminal_result = event.run_result if event.type in _terminal_types else None
                    if terminal_result is not None:
                        _ensure_pause_generation(terminal_result, provider_kwargs)
                        if replay_plan is not None:
                            terminal_result.replay = replay_plan.summary
                        _attach_mcp_meta(terminal_result, mcp_meta)
                        if store is not None:
                            persisted = await _persist_loop_outcome(
                                recorder=recorder,
                                provider_kwargs=provider_kwargs,
                                state_store=store,
                                run_id=run_id,
                                result=terminal_result,
                            )
                            # Pre-pause checkpoint can flip a PAUSED iteration to
                            # CANCELLED. Re-derive the event type from the persisted
                            # status so the consumer sees the correct terminal event.
                            if persisted.status == RunStatus.CANCELLED:
                                event = RunEvent(
                                    type=RunEventType.RUN_CANCELLED, run_result=persisted
                                )
                            else:
                                event = RunEvent(
                                    type=event.type, run_result=persisted, phase=event.phase
                                )
                            await record_run_finished(recorder, run_id, persisted)
                            await notify_run_finished(extra_notifier, run_id, persisted)
                        else:
                            # No persistence: skip _persist_loop_outcome but still
                            # close the lifecycle pair so notifier-side spans
                            # (OTel root span) don't leak.
                            await record_run_finished(recorder, run_id, terminal_result)
                            await notify_run_finished(extra_notifier, run_id, terminal_result)
                        if event.run_result is not None:
                            event.run_result.meta = _public_meta(event.run_result.meta)
                        yield event
                    else:
                        # TEXT_DELTA, TOOL_USE_START, TOOL_USE_END, TOOL_RESULT — pass through
                        _track_partial_text(_shared, event)
                        yield event

        except RunFencedError:
            assert store is not None
            result = await _build_cached_result(store, run_id)
            await _finish_fenced_lifecycle(run_id, result, recorder, extra_notifier)
            yield _fenced_stream_event(run_id, result)

        except Exception as exc:
            # Persist error, yield RUN_ERROR, return cleanly. No re-raise.
            # Covers both setup failures and loop execution errors.
            if store is not None and _shared.get("claimed"):
                error_won = False
                try:
                    error_won = await store.finalize_run(
                        run_id,
                        status=RunStatus.ERROR.value,
                        error=_safe_error(exc),
                        **_error_store_kwargs(store, exc),
                        total_usage=None,
                        expected_current_status="running",
                    )
                except Exception:
                    logger.error(
                        "Failed to persist ERROR status for run %s",
                        run_id,
                        exc_info=safe_exception_info(),
                    )
                if error_won:
                    await _emit_event_safe(store, run_id, "run.error", _error_payload(exc))
            if not _shared.get("existing_run"):
                try:
                    await record_run_failed(recorder, run_id, exc)
                    await notify_run_failed(extra_notifier, run_id, exc)
                except Exception:
                    logger.warning(
                        "Lifecycle hooks failed during stream error handler for run %s",
                        run_id,
                        exc_info=safe_exception_info(),
                    )

            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=RunStatus.ERROR,
                    error=_safe_error(exc),
                    error_info=_provider_error_info(exc),
                    meta=_stream_error_meta(mcp_meta, exc),
                ),
                run_id=run_id,
                error=_safe_error(exc),
            )

        finally:
            _release_memory_edges(image_runtime, store, run_id)
            if not _shared.get("existing_run"):
                agent._task_manager.release_interrupt(run_id)

    async def _cleanup() -> RunResult | None:
        """CAS-guarded cancellation + local lifecycle close.

        Two responsibilities:
          (1) If the run is still RUNNING in the store, finalize it as
              CANCELLED (CAS-guarded so we don't clobber a concurrent
              terminal status set elsewhere).
          (2) If the lifecycle pair was opened locally (recorder/notifier
              stashed by ``_generate`` after ``run_started`` fired), close
              it with a ``run_finished`` event on the LOCAL notifier so
              OTel root spans don't leak. This fires regardless of whether
              the CAS won — if some other caller already finalized the
              run, our local notifier still needs the close event.

        Status resolution for the close event:
          - CAS won → CANCELLED.
          - CAS lost + store available → query store for actual status.
          - No store → CANCELLED (best-effort fallback).
        """
        if not _shared.get("claimed") and _shared.get("state_store") is not None:
            return None
        store = _shared.get("state_store")

        cancel_won = False
        partial_answer = _partial_answer(_shared)
        if store is not None:
            try:
                cancel_won = await store.finalize_run(
                    run_id,
                    status=RunStatus.CANCELLED.value,
                    answer=partial_answer,
                    expected_current_status="running",
                )
                if cancel_won:
                    await _emit_event(
                        store,
                        run_id,
                        "run.cancelled",
                        {"reason": "stream_closed", "partial_output": partial_answer is not None},
                    )
            except Exception:
                logger.error(
                    "Failed to cancel run %s during stream cleanup",
                    run_id,
                    exc_info=safe_exception_info(),
                )

        # Close the lifecycle pair if it was opened.
        if "notifier" in _shared:
            cleanup_recorder = _shared.get("recorder")
            terminal_status = RunStatus.CANCELLED
            cancel_persisted = cancel_won
            if store is not None and not cancel_won:
                try:
                    run_record = await store.get_run(run_id)
                    if run_record is not None:
                        terminal_status = RunStatus(run_record.status)
                        cancel_persisted = terminal_status == RunStatus.CANCELLED
                except Exception:
                    pass  # CANCELLED is a safe fallback
            cancelled_result = RunResult(
                run_id=run_id,
                status=terminal_status,
                meta={**deepcopy(mcp_meta), "dendrux.cancel_persisted": cancel_persisted},
            )
            try:
                await record_run_finished(cleanup_recorder, run_id, cancelled_result)
                await notify_run_finished(_shared.get("notifier"), run_id, cancelled_result)
            except Exception:
                logger.warning(
                    "Lifecycle finish hooks failed during stream cleanup for run %s",
                    run_id,
                    exc_info=safe_exception_info(),
                )

            return cancelled_result
        return None

    stream = _RunStream(
        run_id=run_id,
        generator=stream_context.wrap(_generate()),
        cleanup=lambda: _shielded_cleanup(_cleanup),
    )
    return stream


async def resume(
    run_id: str,
    tool_results: list[ToolResult] | None,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    extra_notifier: LoopNotifier | None = None,
    expected_pause_id: str | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Resume a paused run.

    Three shapes:
      - tool_results provided → client-tool results or approval rejection
      - No args (tool_results=None) → approval approve, or continue a PAUSED run

    Uses an atomic claim to prevent double-resume races.

    Args:
        run_id: The paused run's ID.
        tool_results: Results for the pending tool calls, or None for
            approval-approve (no-arg resume). Each result must have
            a call_id matching one of the pending_tool_calls.
        state_store: Persistence backend (required for resume).
        agent: Agent definition (must match the paused run's agent).
        provider: LLM provider for continuing the run.
        strategy: Strategy override. Defaults to NativeToolCalling.
        loop: Loop override. Defaults to ReActLoop.
        extra_notifier: Optional additional notifier for SSE streaming.
    """
    if tool_results is None:
        # No-arg = approval-approve. Status gate in _resume_core.
        expected = RunStatus.WAITING_APPROVAL.value
    else:
        # Tool results — try WAITING_CLIENT_TOOL first.
        # _resume_core will fall back to WAITING_APPROVAL on claim failure.
        expected = RunStatus.WAITING_CLIENT_TOOL.value

    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        strategy=strategy,
        loop=loop,
        expected_status=expected,
        tool_results=tool_results,
        extra_notifier=extra_notifier,
        expected_pause_id=expected_pause_id,
        _tenant_explicit=_tenant_explicit,
    )


async def resume_with_input(
    run_id: str,
    user_input: UserContent,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    extra_notifier: LoopNotifier | None = None,
    expected_pause_id: str | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Resume a paused run by providing clarification input.

    Only works on runs with status WAITING_HUMAN_INPUT. Appends the
    user's response as a normal USER message and re-enters the loop.

    Args:
        run_id: The paused run's ID.
        user_input: Free-text response to the agent's clarification question.
        state_store: Persistence backend (required for resume).
        agent: Agent definition (must match the paused run's agent).
        provider: LLM provider for continuing the run.
        strategy: Strategy override. Defaults to NativeToolCalling.
        loop: Loop override. Defaults to ReActLoop.
        extra_notifier: Optional additional notifier for SSE streaming.
    """
    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        strategy=strategy,
        loop=loop,
        expected_status=RunStatus.WAITING_HUMAN_INPUT.value,
        user_input=user_input,
        extra_notifier=extra_notifier,
        expected_pause_id=expected_pause_id,
        _tenant_explicit=_tenant_explicit,
    )


class _ResumeContext:
    """Prepared state for re-entering the loop after a resume.

    Built by _prepare_resume(), consumed by both _resume_core() (batch)
    and resume_stream() (streaming). Does NOT own claiming — that happens
    before prepare is called.
    """

    __slots__ = (
        "history",
        "recorder",
        "notifier",
        "pause_state",
        "resolved_loop",
        "resolved_strategy",
    )

    def __init__(
        self,
        *,
        history: list[Message],
        recorder: LoopRecorder,
        notifier: LoopNotifier | None,
        pause_state: PauseState,
        resolved_loop: Loop,
        resolved_strategy: Strategy,
    ) -> None:
        self.history = history
        self.recorder = recorder
        self.notifier = notifier
        self.pause_state = pause_state
        self.resolved_loop = resolved_loop
        self.resolved_strategy = resolved_strategy


@dataclass
class _PauseReads:
    record: RunRecord | None
    checkpoint: dict[str, Any] | None = None
    history_records: list[TraceRecord] | None = None


async def _load_pause_state(
    store: StateStore,
    run_id: str,
    raw: dict[str, Any],
    agent: Agent,
    provider: LLMProvider,
    *,
    reads: _PauseReads | None = None,
) -> PauseState:
    from dendrux.errors import PersistenceError, RunNotReclaimableError
    from dendrux.runtime.replay import rebuild_history
    from dendrux.types import InputValidationError, _usage_from_dict

    pause = PauseState.from_dict(raw)
    if pause.schema_version != 2:
        return pause
    record = reads.record if reads is not None else await store.get_run(run_id)
    if record is None or record.storage_format != 2:
        _codes.log_code(logger, _codes.PAUSE_FORMAT_UNSUPPORTED, run_id=run_id)
        raise PersistenceError(code="pause_format_unsupported")
    checkpoint = (
        getattr(store, "loop_state", None) if getattr(store, "storage_format", None) == 2 else None
    )
    if checkpoint is None:
        checkpoint = await store.get_loop_state(run_id)
    if not checkpoint:
        _codes.log_code(logger, _codes.PAUSE_FORMAT_UNSUPPORTED, run_id=run_id)
        raise PersistenceError(code="pause_format_unsupported")
    try:
        messages = await rebuild_history(
            store,
            run_id,
            limits=agent._limits,
            reasoning_for=pause.provider_kwargs.get("model") or provider.model,
        )
    except RunNotReclaimableError as exc:
        raise InputValidationError("history", "history_too_large") from exc
    # Strategy owns the system prompt; seeds retain it for audit/reconstruction.
    pause.history = [m for m in messages.messages if m.role != Role.SYSTEM]
    if reads is not None:
        reads.history_records = messages.records
        reads.checkpoint = checkpoint
    pause.steps = messages.steps
    pause.usage = _usage_from_dict(checkpoint["usage"])
    pause.iteration = checkpoint["iteration"]
    pause.trace_order_offset = checkpoint["trace_order_offset"]
    pause.provider_kwargs = dict(checkpoint["provider_kwargs"])
    for kind, ref in (
        ("image_view", checkpoint.get("image_view_ref")),
        ("skills_session", checkpoint.get("skills", {}).get("session_ref")),
        ("clarification", checkpoint.get("clarification_ref")),
    ):
        if ref is None:
            continue
        state_trace = await store.get_trace(run_id, ref)
        if state_trace is None:
            raise PersistenceError(code="pause_format_unsupported")
        payload = json.loads(state_trace.content)
        if kind == "image_view":
            pause.image_view = ImageViewState.from_dict(payload)
        elif kind == "skills_session":
            session = current_skill_session(agent)
            if session is not None:
                session.restore_snapshot(payload)
        else:
            pause.pause_message = payload.get("pause_message")
    return pause


async def _prepare_resume(
    run_id: str,
    pause_state: PauseState,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    expected_status: str,
    tool_results: list[ToolResult] | None = None,
    user_input: str | None = None,
    extra_notifier: LoopNotifier | None = None,
    user_parts: tuple[ContentPart, ...] | None = None,
    tenant_id: str | None = None,
    pii_mapping: dict[str, str] | None = None,
    history_records: Any = None,
    pause_data: dict[str, Any] | None = None,
) -> _ResumeContext:
    """Build everything needed to re-enter the loop after a resume.

    Receives an already-loaded PauseState — does NOT load pause data or
    claim the run. Those steps are owned by the caller (_resume_core for
    direct resume, resume_stream for streaming, resume_claimed for
    post-submit_and_claim resume).

    Steps performed:
      1. Record durable run.resumed event (the store allocates its index)
      2. Build resume history (inject tool results or user input)
      3. Create recorder with correct trace offset
      4. Notify recorder of injected messages
    """
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.tool import get_tool_def

    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or agent.loop or ReActLoop()
    _validate_loop_skill_compat(agent, resolved_loop)

    # 1. Record resume event with enriched payload for dashboard
    resume_data: dict[str, Any] = {"resumed_from": expected_status}
    if tool_results is not None:
        submitted: list[dict[str, Any]] = []
        for tr in tool_results:
            entry: dict[str, Any] = {"call_id": tr.call_id, "name": tr.name, "success": tr.success}
            if tr.images:
                entry["image_count"] = len(tr.images)
            submitted.append(entry)
        resume_data["submitted_results"] = submitted
    elif user_input is not None:
        resume_data["user_input"] = user_input
        if user_parts is not None:
            resume_data["image_count"] = sum(1 for p in user_parts if isinstance(p, ImagePart))
    await _emit_event(state_store, run_id, "run.resumed", resume_data)

    # 3. Build resume history
    history = list(pause_state.history)

    if tool_results is not None:
        for tr in tool_results:
            history.append(tool_result_message(tr))
    elif user_parts is not None:
        history.append(Message.with_parts(Role.USER, user_parts))
    elif user_input is not None:
        history.append(Message(role=Role.USER, content=user_input))

    # 4. Create recorder with correct trace offset
    next_index = getattr(state_store, "get_next_trace_index", None)
    if next_index is not None:
        trace_order_offset = await next_index(run_id)
    else:
        # Compatibility for custom stores predating the scalar index read.
        traces = await state_store.get_traces(run_id)
        trace_order_offset = max((t.order_index for t in traces), default=-1) + 1

    if pause_state.schema_version == 2:
        trace_order_offset = max(trace_order_offset, pause_state.trace_order_offset)

    target_lookup = {}
    for fn in agent.tools:
        td = get_tool_def(fn)
        target_lookup[td.name] = td.target
    from dendrux.guardrails import GuardrailEngine

    run_guardrails = (
        GuardrailEngine(agent.guardrails, pii_mapping=pii_mapping, run_id=run_id)
        if pause_state.schema_version == 2 and agent.guardrails
        else None
    )
    recorder = PersistenceRecorder(
        state_store,
        run_id,
        model=provider.model,
        provider_name=type(provider).__name__,
        target_lookup=target_lookup,
        initial_order_index=trace_order_offset,
        guardrail_engine=run_guardrails,
        **_recorder_image_kwargs(tenant_id, provider),
    )

    await recorder.restore_history(limits=agent._limits, records=history_records)
    recorder._resume_pause_data = pause_data

    # NOTE: lifecycle hooks (record_run_started/notify_run_started) and
    # injected-history replay are intentionally NOT done here. The caller
    # fires them inside its own try/except so a failure during replay still
    # pairs with on_run_failed. See _replay_resume_injected_history below.

    await _emit_declared_tools(agent, recorder, extra_notifier, run_id)
    return _ResumeContext(
        history=history,
        recorder=recorder,
        notifier=extra_notifier,
        pause_state=pause_state,
        resolved_loop=resolved_loop,
        resolved_strategy=resolved_strategy,
    )


async def _replay_resume_injected_history(
    ctx: _ResumeContext,
    run_id: str,
    *,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
) -> None:
    """Replay tool-result / user-input messages onto recorder + notifier.

    Split out of _prepare_resume so the caller can invoke it INSIDE its
    own try block, after firing record_run_started. If a recorder write
    fails here, the caller's except handler fires on_run_failed and the
    lifecycle pair stays balanced.

    Recorder calls are fail-closed (exceptions propagate); notifier calls
    are best-effort (swallowed via notify helpers).
    """
    from dendrux.loops._helpers import notify_message as _notify_msg
    from dendrux.loops._helpers import notify_tool as _notify_tool
    from dendrux.loops._helpers import notify_tool_started as _notify_tool_started

    pause_state = ctx.pause_state
    history = ctx.history
    recorder = ctx.recorder
    extra_notifier = ctx.notifier
    consume = getattr(recorder, "consume_resume", None)
    if consume is not None:
        for i in range(len(pause_state.history), len(history)):
            history[i] = await project_message(recorder, history[i])
        await consume(history[len(pause_state.history) :])

    if tool_results is not None:
        pending_by_id = {tc.id: tc for tc in pause_state.pending_tool_calls}
        injected_start = len(pause_state.history)
        for i, tr in enumerate(tool_results):
            msg = history[injected_start + i]
            tc = pending_by_id[tr.call_id]
            # Fire tool_started so OTel can pair a span with on_tool_completed.
            # The tool actually started when the loop dispatched it before pause;
            # the start event is replayed here on resume because that's the only
            # point at which an external observer receives the call_id.
            if pause_state.schema_version != 2:
                await recorder.on_tool_started(run_id, tc, pause_state.iteration)
            await _notify_tool_started(extra_notifier, run_id, tc, pause_state.iteration)
            await recorder.on_message_appended(run_id, msg, pause_state.iteration)
            await record_tool(recorder, run_id, tc, tr, pause_state.iteration)
            await _notify_msg(extra_notifier, run_id, msg, pause_state.iteration)
            await _notify_tool(extra_notifier, run_id, tc, tr, pause_state.iteration)
    elif user_input is not None:
        msg = history[-1]
        await recorder.on_message_appended(run_id, msg, pause_state.iteration)
        await _notify_msg(extra_notifier, run_id, msg, pause_state.iteration)

    order_observations(recorder, history)


class _ApprovalResume(NamedTuple):
    """Outcome of executing an approval batch on resume."""

    # (ToolCall, ToolResult) pairs appended to history, in pending order.
    results: list[tuple[ToolCall, ToolResult]]
    # Client-target calls held in the batch; they never run on the server.
    client_pending: list[ToolCall]
    # ``approval.decided`` payload (scalar decision + per-call outcomes).
    decided: dict[str, Any]
    # Set when a resume barrier stopped execution before a side effect.
    barrier: RunStatus | None
    # Calls not yet resolved when the barrier hit (client calls excluded),
    # as (call, outcome, denial reason) in pending order.
    remaining: tuple[tuple[ToolCall, str, str | None], ...] = ()


async def _resume_pause_barrier(state_store: StateStore, run_id: str) -> RunStatus | None:
    """Cancel-then-pause read of the run's control flags (one query).

    Returns ``CANCELLED`` when cancellation was requested, else ``PAUSED``
    when a pause was requested, else None. Cancellation wins over pause.
    """
    from dendrux.runtime.state import read_run_control

    control = await read_run_control(state_store, run_id)
    if control.cancel_requested:
        return RunStatus.CANCELLED
    if control.pause_requested:
        return RunStatus.PAUSED
    return None


async def _resume_barrier(state_store: StateStore | None, run_id: str) -> RunStatus | None:
    """Cancel-then-pause check before an approved side effect.

    Runs after a resume claim, before each approved/held server call.
    Cancellation wins over pause.
    """
    if state_store is None:
        return None
    return await _resume_pause_barrier(state_store, run_id)


def _approval_outcomes(
    pause_state: PauseState,
    target_lookup: dict[str, Any],
    decisions: dict[str, ApprovalDecision] | None,
) -> list[tuple[ToolCall, str, str | None]]:
    """Classify each pending call: (call, outcome, denial reason) in pending order.

    ``decisions=None`` is the legacy batch approve: every gate approved.
    """
    from dendrux.runtime.submit import DEFAULT_DENIAL_REASON, approval_gates_of, is_client_call

    gates = approval_gates_of(pause_state, target_lookup)
    out: list[tuple[ToolCall, str, str | None]] = []
    for tc in pause_state.pending_tool_calls:
        if is_client_call(pause_state, tc, target_lookup):
            out.append((tc, "client_pending", None))
        elif tc.id not in gates:
            out.append((tc, "not_gated_executed", None))
        else:
            decision = decisions.get(tc.id) if decisions is not None else None
            if decision is None or decision.approved:
                out.append((tc, "approved", None))
            else:
                out.append((tc, "denied", decision.reason or DEFAULT_DENIAL_REASON))
    return out


def _decided_payload(
    run_id: str,
    outcomes: list[tuple[ToolCall, str, str | None]],
    *,
    scalar: str | None = None,
) -> dict[str, Any]:
    """``approval.decided`` payload: scalar decision + every pending call's outcome."""
    if scalar is None:
        gated = [o for _, o, _ in outcomes if o in ("approved", "denied")]
        if all(o == "approved" for o in gated):
            scalar = "approved"
        elif all(o == "denied" for o in gated):
            scalar = "rejected"
        else:
            scalar = "mixed"
    return {
        "decision": scalar,
        "run_id": run_id,
        "decisions": [
            {"call_id": tc.id, "tool_name": tc.name, "decision": outcome, "reason": reason}
            for tc, outcome, reason in outcomes
        ],
    }


async def _record_approval_decided(
    recorder: LoopRecorder,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    data: dict[str, Any],
    *,
    event_id: str | None = None,
) -> None:
    from dendrux.loops._helpers import notify_governance, record_governance
    from dendrux.runtime.persistence import PersistenceRecorder

    if event_id is not None and isinstance(recorder, PersistenceRecorder):
        await recorder._emit_event("approval.decided", iteration, data, event_id=event_id)
    else:
        await record_governance(recorder, run_id, "approval.decided", iteration, data)
    await notify_governance(notifier, run_id, "approval.decided", iteration, data)


def _rejection_decided_payload(
    run_id: str, pause_state: PauseState, tool_results: list[ToolResult]
) -> dict[str, Any]:
    """Legacy batch rejection: every pending call received a synthetic denial."""
    errors = {tr.call_id: tr.error for tr in tool_results}
    outcomes = [(tc, "denied", errors.get(tc.id)) for tc in pause_state.pending_tool_calls]
    return _decided_payload(run_id, outcomes, scalar="rejected")


async def _execute_approved_tools(
    agent: Agent,
    pause_state: PauseState,
    recorder: LoopRecorder,
    notifier: LoopNotifier | None,
    run_id: str,
    *,
    history: list[Message] | None = None,
    resolved_loop: Loop | None = None,
    decisions: dict[str, ApprovalDecision] | None = None,
    state_store: StateStore | None = None,
    image_runtime: ImageRuntime | None = None,
) -> _ApprovalResume:
    """Resolve an approval batch on resume, routing each call by target.

    Shared by _resume_core (batch) and resume_stream (streaming). Walks
    ``pending_tool_calls`` in order:

    - client-target calls are never executed; they are returned in
      ``client_pending`` and the caller pauses the run for client results;
    - denied gated calls get a failed result ``{"denied": reason}``;
    - approved gated calls and held non-gated server calls execute through
      the approved path (no ``before_tool`` re-run: the human decision wins
      and held calls were already evaluated before the pause).

    ``decisions=None`` is the legacy batch approve. Before every server
    execution a resume barrier checks cancel (then pause); on a hit the
    remaining calls are skipped and ``barrier`` is set.

    Tool messages are appended to ``history`` (the live ctx.history list,
    not pause_state.history which is a snapshot). ``resolved_loop`` is
    forwarded to ``agent.get_tool_lookups()`` so a runtime loop override
    stays consistent across resume. Caller emits ``approval.decided`` and
    handles loop re-entry.
    """
    import json

    from dendrux.loops._helpers import notify_message as _ntfy_msg
    from dendrux.loops._helpers import notify_tool as _ntfy_tool
    from dendrux.loops._helpers import notify_tool_started as _ntfy_tool_started
    from dendrux.loops._helpers import record_tool_started as _rec_tool_started
    from dendrux.loops.react import _execute_tool, _process_tool_calls
    from dendrux.skills._session import SkillTool
    from dendrux.tools import enforce_call_limits
    from dendrux.tools._validate import check_tool_call
    from dendrux.types import AgentStep, ToolResult

    target_history = history if history is not None else pause_state.history
    lookups = await agent.get_tool_lookups(loop=resolved_loop)
    configure = getattr(recorder, "configure_tools", None)
    if configure is not None:
        configure(list(lookups.defs.values()))
    outcomes = _approval_outcomes(pause_state, lookups.target, decisions)
    results: list[tuple[ToolCall, ToolResult]] = []
    restored_counts = getattr(recorder, "execution_counts", lambda: None)()
    counts: dict[str, int] = restored_counts if restored_counts is not None else {}
    for message in [] if restored_counts is not None else pause_state.history:
        if (
            message.role == Role.TOOL
            and message.name
            and not message.meta.get("dendrux.not_executed")
        ):
            counts[message.name] = counts.get(message.name, 0) + 1
    client_calls = [
        tc
        for tc, outcome, _ in outcomes
        if outcome == "client_pending" and tc.name not in agent.deny
    ]
    _, limited = enforce_call_limits(client_calls, max_calls=lookups.max_calls, call_counts=counts)
    client_rejections = {tc.id: result for tc, result in limited}
    outcomes = [
        (tc, "policy_denied", "denied_by_policy")
        if outcome == "client_pending" and tc.name in agent.deny
        else (tc, "call_limited", None)
        if tc.id in client_rejections
        else (tc, outcome, reason)
        for tc, outcome, reason in outcomes
    ]
    client_pending = [tc for tc, outcome, _ in outcomes if outcome == "client_pending"]
    decided = _decided_payload(
        run_id,
        [
            (tc, "denied" if outcome in {"policy_denied", "call_limited"} else outcome, reason)
            for tc, outcome, reason in outcomes
        ],
        scalar="approved" if decisions is None else None,
    )
    session = current(agent)
    # An absent executor is a specified error only for a pinned approved script.
    # Inject an execution closure here without advertising the unavailable tool.
    if session is not None and session.configured and session.executor is None:
        for tc in pause_state.pending_tool_calls:
            if tc.name == "run_skill_script" and tc.name not in lookups.fn:
                lookups.fn[tc.name] = SkillTool(session, tc.name)

    iteration = pause_state.iteration

    async def _append(tc: ToolCall, tr: ToolResult, *, executed: bool = True) -> None:
        if executed:
            await record_tool(recorder, run_id, tc, tr, iteration)
            await _ntfy_tool(notifier, run_id, tc, tr, iteration)
        result_msg = tool_result_message(tr)
        if not executed or tr.meta.get("code") == "max_calls":
            result_msg.meta["dendrux.not_executed"] = True
        target_history.append(result_msg)
        result_msg = await record_message(recorder, run_id, result_msg, iteration)
        target_history[-1] = result_msg
        await _ntfy_msg(notifier, run_id, result_msg, iteration)
        results.append((tc, tr))

    for index, (tc, outcome, reason) in enumerate(outcomes):
        if outcome == "client_pending":
            continue
        if outcome == "policy_denied":
            from dendrux.loops._helpers import notify_governance, record_governance

            denial_message = f"Tool '{tc.name}' is denied by policy."
            data = {"tool_name": tc.name, "call_id": tc.id, "reason": "denied_by_policy"}
            await record_governance(
                recorder, run_id, "policy.denied", iteration, data, correlation_id=tc.id
            )
            await notify_governance(
                notifier, run_id, "policy.denied", iteration, data, correlation_id=tc.id
            )
            await _append(
                tc,
                ToolResult(
                    name=tc.name,
                    call_id=tc.id,
                    payload=json.dumps({"denied": denial_message}),
                    success=False,
                    error=denial_message,
                ),
                executed=False,
            )
            continue
        if outcome == "call_limited":
            await _append(tc, client_rejections[tc.id])
            continue
        if outcome == "denied":
            assert reason is not None
            denied = ToolResult(
                name=tc.name,
                call_id=tc.id,
                payload=json.dumps({"denied": reason}),
                success=False,
                error=reason,
            )
            if getattr(state_store, "storage_format", 1) != 2:
                await _rec_tool_started(recorder, run_id, tc, iteration)
            await _ntfy_tool_started(notifier, run_id, tc, iteration)
            await _append(tc, denied)
            continue

        barrier = await _resume_barrier(state_store, run_id)
        if barrier is not None:
            remaining = tuple(o for o in outcomes[index:] if o[1] != "client_pending")
            order_observations(recorder, target_history)
            return _ApprovalResume(results, client_pending, decided, barrier, remaining)

        if isinstance(lookups.fn.get(tc.name), SkillTool):
            outcome_s = await _process_tool_calls(
                step=AgentStep(reasoning=None, action=tc),
                call_counts=counts,
                lookups=lookups,
                strategy=NativeToolCalling(),
                recorder=recorder,
                notifier=notifier,
                run_id=run_id,
                iteration=iteration,
                history=target_history,
                warnings=[],
                deny=agent._deny,
                tool_result_transform=agent.tool_result_transform,
                approved_skills=True,
                **_tool_image_kwargs(_process_tool_calls, image_runtime),
            )
            results.extend(outcome_s.all_results)
            continue
        if tc.name in agent.deny:
            from dendrux.loops._helpers import notify_governance, record_governance

            denial_message = f"Tool '{tc.name}' is denied by policy."
            data = {"tool_name": tc.name, "call_id": tc.id, "reason": "denied_by_policy"}
            await record_governance(
                recorder, run_id, "policy.denied", iteration, data, correlation_id=tc.id
            )
            await notify_governance(
                notifier, run_id, "policy.denied", iteration, data, correlation_id=tc.id
            )
            await _append(
                tc,
                ToolResult(
                    name=tc.name,
                    call_id=tc.id,
                    payload=json.dumps({"denied": denial_message}),
                    success=False,
                    error=denial_message,
                ),
                executed=False,
            )
            continue
        _, rejected = enforce_call_limits([tc], max_calls=lookups.max_calls, call_counts=counts)
        if rejected:
            await _append(tc, rejected[0][1])
            continue

        counts[tc.name] = counts.get(tc.name, 0) + 1
        invalid_result = check_tool_call(tc, lookups.defs.get(tc.name))
        if invalid_result is None:
            await _rec_tool_started(recorder, run_id, tc, iteration)
            await _ntfy_tool_started(notifier, run_id, tc, iteration)
            from dendrux.loops._helpers import lease_scope

            async with lease_scope(recorder):
                tr = await _execute_tool(
                    tc,
                    lookups,
                    agent.tool_result_transform,
                    run_id=run_id,
                    **_tool_image_kwargs(_execute_tool, image_runtime),
                )
        else:
            tr = invalid_result
        await _append(tc, tr)

    order_observations(recorder, target_history)
    return _ApprovalResume(results, client_pending, decided, None)


async def _validated_decisions(
    agent: Agent,
    ctx: _ResumeContext,
    run_id: str,
    pause_state: PauseState,
    decisions: list[tuple[str, ApprovalDecision]] | None,
) -> dict[str, ApprovalDecision] | None:
    """Re-validate claimed decisions against the persisted gate snapshot."""
    if decisions is None:
        return None
    from dendrux.runtime.submit import validate_approval_decisions

    lookups = await agent.get_tool_lookups(loop=ctx.resolved_loop)
    return validate_approval_decisions(run_id, pause_state, decisions, lookups.target)


def _approval_stop_result(
    run_id: str,
    pause_state: PauseState,
    history: list[Message],
    approval: _ApprovalResume,
) -> RunResult | None:
    """Result for an approval resume that must not re-enter the loop.

    Barrier hit → CANCELLED. Held client calls → a fresh WAITING_CLIENT_TOOL
    pause over the updated history. Otherwise None (re-enter the loop).
    """
    from dendrux.loops.react import _commentary, _snapshot_usage
    from dendrux.types import RunResult

    if approval.barrier == RunStatus.PAUSED:
        return _barrier_pause_result(run_id, pause_state, history, approval)
    if approval.barrier is not None:
        return RunResult(
            run_id=run_id,
            status=approval.barrier,
            steps=pause_state.steps,
            commentary=_commentary(pause_state.steps),
            iteration_count=pause_state.iteration,
            usage=pause_state.usage,
        )
    if not approval.client_pending:
        return None
    client_pause = PauseState(
        agent_name=pause_state.agent_name,
        pending_tool_calls=list(approval.client_pending),
        pending_targets={
            tc.id: pause_state.pending_targets.get(tc.id) or "client"
            for tc in approval.client_pending
        },
        history=list(history),
        steps=list(pause_state.steps),
        iteration=pause_state.iteration,
        trace_order_offset=len(history),
        usage=_snapshot_usage(pause_state.usage),
        image_view=pause_state.image_view,
    )
    client_pause = _carry_budget_fields(client_pause, pause_state)
    client_pause = _carry_image_view(client_pause, pause_state)
    return RunResult(
        run_id=run_id,
        status=RunStatus.WAITING_CLIENT_TOOL,
        steps=pause_state.steps,
        commentary=_commentary(pause_state.steps),
        iteration_count=pause_state.iteration,
        usage=pause_state.usage,
        meta={"pause_state": client_pause},
    )


def _barrier_pause_result(
    run_id: str,
    pause_state: PauseState,
    history: list[Message],
    approval: _ApprovalResume,
) -> RunResult:
    """PAUSED result for a pause observed between approved calls.

    Holds every call that has no result yet (the unexecuted remainder plus
    held client calls) in pending order, with the decisions already made for
    the gated ones; resume executes them without asking again.
    """
    from dendrux.loops.react import _commentary, _snapshot_usage
    from dendrux.types import RunResult

    old_gates = pause_state.approval_gates or {}
    gates: dict[str, dict[str, str | None]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for tc, outcome, reason in approval.remaining:
        if outcome in ("approved", "denied"):
            gates[tc.id] = dict(old_gates.get(tc.id) or {"source": "legacy", "reason": None})
            decisions[tc.id] = {"approved": outcome == "approved", "reason": reason}
    client_ids = {tc.id for tc in approval.client_pending}
    open_ids = {tc.id for tc, _, _ in approval.remaining} | client_ids
    pending = [tc for tc in pause_state.pending_tool_calls if tc.id in open_ids]
    targets: dict[str, str] = {}
    for tc in pending:
        target = pause_state.pending_targets.get(tc.id)
        if target is None and tc.id in client_ids:
            target = "client"
        if target is not None:
            targets[tc.id] = target
    pause = PauseState(
        agent_name=pause_state.agent_name,
        pending_tool_calls=pending,
        pending_targets=targets,
        history=list(history),
        steps=list(pause_state.steps),
        iteration=pause_state.iteration,
        trace_order_offset=len(history),
        usage=_snapshot_usage(pause_state.usage),
        approval_gates=gates,
        pause_reason="requested",
        pending_decisions=decisions,
        image_view=pause_state.image_view,
    )
    pause = _carry_budget_fields(pause, pause_state)
    pause = _carry_image_view(pause, pause_state)
    return RunResult(
        run_id=run_id,
        status=RunStatus.PAUSED,
        steps=pause_state.steps,
        commentary=_commentary(pause_state.steps),
        iteration_count=pause_state.iteration,
        usage=pause_state.usage,
        meta={"pause_state": pause},
    )


def _pending_decision_map(pause_state: PauseState) -> dict[str, ApprovalDecision] | None:
    """Decisions persisted with a pause that interrupted an approval batch."""
    if pause_state.pending_decisions is None:
        return None
    return {
        call_id: ApprovalDecision(
            approved=bool(d.get("approved")),
            reason=d.get("reason") if isinstance(d.get("reason"), str) else None,
        )
        for call_id, d in pause_state.pending_decisions.items()
    }


async def _ingest_resume_images(
    agent: Agent,
    provider: LLMProvider,
    state_store: StateStore,
    run_id: str,
    *,
    loop: Loop | None,
    tool_results: list[ToolResult] | None,
    input_items: tuple[Any, ...],
    loop_method: str = "run",
) -> tuple[list[ToolResult] | None, tuple[ContentPart, ...] | None]:
    """Ingest resume-time images (direct ``resume`` / ``resume_stream``) before the claim.

    The run, its pause status and the submitted call ids are checked first,
    so a resume that cannot claim writes no blob.
    """
    from dendrux.errors import RunNotFoundError
    from dendrux.runtime.submit import (
        check_new_image_capability,
        ingest_tool_result_images,
        ingest_user_items,
    )

    _require_image_loop(loop or agent.loop or ReActLoop(), loop_method)
    record = await state_store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    allowed = (
        (RunStatus.WAITING_CLIENT_TOOL.value, RunStatus.WAITING_APPROVAL.value)
        if tool_results is not None
        else (RunStatus.WAITING_HUMAN_INPUT.value,)
    )
    if record.status not in allowed:
        await _raise_resume_claim_failure(state_store, run_id, allowed[0])
    if tool_results is not None:
        raw_pause = await state_store.get_pause_state(run_id)
        if raw_pause is None:
            raise ValueError(f"Run '{run_id}' has no pause state — cannot resume.")
        pending_ids = {tc["id"] for tc in raw_pause.get("pending_tool_calls", [])}
        provided_ids = {tr.call_id for tr in tool_results}
        if provided_ids != pending_ids:
            raise ValueError(
                f"Tool result call_ids {provided_ids} do not match "
                f"pending tool call_ids {pending_ids}."
            )
    runtime = _image_runtime(agent, state_store, record.tenant_id, run_id)
    if input_items:
        count = sum(1 for item in input_items if not isinstance(item, str))

        async def _user_capability() -> None:
            await check_new_image_capability(
                provider, provider.model, runtime.policy, image_count=count, origin="user"
            )

        return tool_results, await ingest_user_items(
            input_items, runtime=runtime, before_store=_user_capability
        )
    assert tool_results is not None
    count = sum(len(tr.images) for tr in tool_results)

    async def _tool_capability() -> None:
        await check_new_image_capability(
            provider, provider.model, runtime.policy, image_count=count, origin="client_tool"
        )

    return (
        await ingest_tool_result_images(
            tool_results, runtime=runtime, before_store=_tool_capability
        ),
        None,
    )


async def _submit_image_runtime(
    agent: Agent, store: StateStore, run_id: str, expected_status: RunStatus
) -> ImageRuntime:
    """Runtime for a submit: status preflight first, so a bad run stores nothing."""
    from dendrux.runtime.submit import preflight_status

    await preflight_status(store, run_id, expected_status)
    record = await store.get_run(run_id)
    return _image_runtime(agent, store, record.tenant_id if record else None, run_id)


async def ingest_submitted_tool_images(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    pause: PauseState,
    results: list[ToolResult],
) -> tuple[list[ToolResult], dict[str, Any]]:
    """Ingest client-tool images before the claim; return results + claim edge kwargs.

    Internal helper for :meth:`Agent.submit_tool_results`. Submission edges
    (``f"{pause.iteration}:{call_id}"``) are written in the claim transaction.
    """
    from dendrux.runtime.submit import (
        check_new_image_capability,
        ingest_tool_result_images,
        tool_result_edges,
    )

    runtime = await _submit_image_runtime(agent, store, run_id, RunStatus.WAITING_CLIENT_TOOL)
    count = sum(len(r.images) for r in results)

    async def _capability() -> None:
        await check_new_image_capability(
            provider, provider.model, runtime.policy, image_count=count, origin="client_tool"
        )

    ingested = await ingest_tool_result_images(results, runtime=runtime, before_store=_capability)
    edges = tool_result_edges(
        ingested, tenant_scope=runtime.tenant_scope, iteration=pause.iteration
    )
    return ingested, {
        "blob_refs": edges,
        "blob_refs_scope": runtime.tenant_scope,
        "validate": _same_pause(run_id, pause.iteration),
    }


def _same_pause(run_id: str, iteration: int) -> Callable[[dict[str, Any]], None]:
    """Claim-time check that the pause is the one whose iteration keyed the submission edges."""

    def _check(pause_data: dict[str, Any]) -> None:
        if pause_data.get("iteration") != iteration:
            from dendrux.errors import RunAlreadyClaimedError

            raise RunAlreadyClaimedError(run_id)

    return _check


async def ingest_submitted_user_parts(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    projection: str,
    items: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ingest ``submit_input`` images before the claim; return submitted data + edge kwargs.

    ``submitted_user_input`` keeps the text projection; the parts go to
    ``submitted_user_parts`` with ``f"{pause.iteration}:input"`` edges.
    """
    from dendrux.runtime.submit import (
        check_new_image_capability,
        ingest_user_items,
        parts_edges,
        raise_for_non_paused_status,
    )

    runtime = await _submit_image_runtime(agent, store, run_id, RunStatus.WAITING_HUMAN_INPUT)
    raw_pause = await store.get_pause_state(run_id)
    if raw_pause is None:
        await raise_for_non_paused_status(store, run_id)
    iteration = int(raw_pause.get("iteration") or 0)
    count = sum(1 for item in items if not isinstance(item, str))

    async def _capability() -> None:
        await check_new_image_capability(
            provider, provider.model, runtime.policy, image_count=count, origin="user"
        )

    parts = await ingest_user_items(items, runtime=runtime, before_store=_capability)
    edges = parts_edges(
        parts,
        tenant_scope=runtime.tenant_scope,
        owner_kind="submission",
        owner_id=f"{iteration}:input",
    )
    submitted = {"submitted_user_input": projection, "submitted_user_parts": parts_to_dicts(parts)}
    return submitted, {
        "blob_refs": edges,
        "blob_refs_scope": runtime.tenant_scope,
        "validate": _same_pause(run_id, iteration),
    }


def _fenced_stream_event(run_id: str, result: RunResult) -> RunEvent:
    from dataclasses import replace

    from dendrux.types import RunEvent, RunEventType

    types = {
        RunStatus.ERROR: RunEventType.RUN_ERROR,
        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
        RunStatus.SUCCESS: RunEventType.RUN_COMPLETED,
        RunStatus.MAX_ITERATIONS: RunEventType.RUN_COMPLETED,
        RunStatus.PAUSED: RunEventType.RUN_PAUSED,
        RunStatus.WAITING_CLIENT_TOOL: RunEventType.RUN_PAUSED,
        RunStatus.WAITING_APPROVAL: RunEventType.RUN_PAUSED,
        RunStatus.WAITING_HUMAN_INPUT: RunEventType.RUN_PAUSED,
    }
    if result.status not in types:
        info = ErrorInfo(
            category="run_fenced",
            retryable=False,
            safe_message="Execution ownership changed; read the run's current state.",
            error_type="RunFencedError",
        )
        return RunEvent(
            type=RunEventType.RUN_ERROR,
            run_id=run_id,
            run_result=replace(result, error_info=info),
            error=info.safe_message,
        )
    return RunEvent(
        type=types[result.status],
        run_id=run_id,
        run_result=result,
        error=result.error if result.status == RunStatus.ERROR else None,
    )


async def _finish_fenced_lifecycle(
    run_id: str,
    result: RunResult,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
) -> None:
    """Close local lifecycle sinks with the persisted winner, without a status write."""
    try:
        await record_run_finished(recorder, run_id, result)
    except Exception:
        _codes.log_code(logger, _codes.FENCED_FINISH_FAILED, run_id=run_id)
    await notify_run_finished(notifier, run_id, result)


async def _check_claim(store: StateStore, run_id: str) -> None:
    getter = getattr(store, "get_run_control", None)
    if getter is not None:
        await getter(run_id)


def _canonicalize_legacy_denials(
    status: str, run_id: str, raw: dict[str, Any], results: list[ToolResult] | None
) -> None:
    if status == RunStatus.WAITING_APPROVAL.value and results is not None:
        results[:] = legacy_approval_denial(run_id, PauseState.from_dict(raw), results)


async def _claim_resume(
    store: StateStore,
    run_id: str,
    raw: dict[str, Any],
    agent: Agent,
    expected: str,
    *,
    expected_pause_id: str | None,
    results: list[ToolResult] | None,
    user_input: Any = None,
    user_parts: tuple[ContentPart, ...] | None = None,
    decisions: list[tuple[str, ApprovalDecision]] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> tuple[StateStore, str, dict[str, Any]]:
    allowed = [expected]
    if results is not None and expected != RunStatus.WAITING_APPROVAL.value:
        allowed.append(RunStatus.WAITING_APPROVAL.value)
    elif results is None and user_input is None and expected == RunStatus.WAITING_APPROVAL.value:
        allowed.append(RunStatus.PAUSED.value)
    validate = submission_validator(
        run_id, raw, agent=agent, expected_pause_id=expected_pause_id, call_results=results
    )
    record = await store.get_run(run_id)
    require_pause_id(
        agent,
        expected_pause_id,
        pause_id=raw.get("pause_id"),
        store=store,
        approval=(
            results is None
            and user_input is None
            and record is not None
            and record.status == RunStatus.WAITING_APPROVAL.value
        ),
        run_id=run_id,
    )
    epoch: ClaimResult | int | bool
    for status in allowed:
        descriptor = None
        _canonicalize_legacy_denials(status, run_id, raw, results)
        if raw.get("schema_version") == 2 and (
            results is not None or user_input is not None or status == "waiting_approval"
        ):
            from dendrux.runtime.submit import (
                serialize_decisions,
                serialize_tool_results,
                submission_descriptor,
            )

            payload: dict[str, Any] = (
                {
                    "submitted_decisions": serialize_decisions(
                        PauseState.from_dict(raw), dict(decisions)
                    )
                }
                if decisions is not None
                else {"submitted_tool_results": serialize_tool_results(results)}
                if results is not None
                else {"submitted_user_input": user_input}
                if user_input is not None
                else {"approved": True}
            )
            if user_parts is not None:
                from dendrux.types import content_part_to_dict

                payload["submitted_user_parts"] = [content_part_to_dict(p) for p in user_parts]
            if status == "waiting_approval" and results is not None:
                payload["approved"] = False
            descriptor = submission_descriptor(
                raw,
                checkpoint if checkpoint is not None else await store.get_loop_state(run_id) or {},
                payload,
                status,
            )
            epoch = await store.submit_and_claim(
                run_id,
                expected_status=status,
                submitted_data=payload,
                submission=descriptor,
                validate=validate,
            )
        else:
            epoch = await claim_paused_validated(
                store,
                run_id,
                expected_status=status,
                validate=validate,
            )
        if epoch:
            view = claimed_store(store, run_id, epoch)
            try:
                await report_governance_change(agent, view, run_id, raw)
            except BaseException:
                await _release_claim(view, run_id, status)
                raise
            if descriptor is not None:
                raw = {
                    **raw,
                    "submission": {
                        **{
                            k: descriptor[k]
                            for k in ("submission_id", "kind", "pause_id", "call_ids")
                        },
                        "payload": payload,
                    },
                }
            return view, status, raw
    await _raise_resume_claim_failure(store, run_id, expected)
    raise AssertionError("unreachable")


async def _release_claim(store: StateStore, run_id: str, status: str) -> None:
    epoch = getattr(store, "claim_epoch", None)
    release = getattr(store, "release_run_claim", None)
    if epoch is not None and release is not None:
        import anyio

        with anyio.CancelScope(shield=True):
            await release(run_id, status=status, claim_epoch=epoch)


async def _shielded_cleanup(cleanup: Callable[[], Any]) -> Any:
    """Drain cleanup under cancellation, then propagate the caller's cancellation."""
    import anyio

    cancellation: asyncio.CancelledError | None = None
    with anyio.CancelScope(shield=True):
        task = asyncio.create_task(cleanup())
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
        try:
            result = task.result()
        except BaseException:
            if cancellation is not None:
                raise cancellation from None
            raise
    if cancellation is not None:
        raise cancellation
    # A level cancellation masked by the scope may never have reached the loop.
    await anyio.lowlevel.checkpoint_if_cancelled()
    return result


async def _resume_core(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    expected_status: str,
    tool_results: list[ToolResult] | None = None,
    user_input: UserContent | None = None,
    extra_notifier: LoopNotifier | None = None,
    _skip_claim: bool = False,
    decisions: list[tuple[str, ApprovalDecision]] | None = None,
    user_parts: tuple[ContentPart, ...] | None = None,
    expected_pause_id: str | None = None,
    _reclaim_pause: PauseState | None = None,
    _reclaim_history: Any = None,
    _tenant_explicit: bool | None = None,
    _raw_pause: dict[str, Any] | None = None,
) -> RunResult:
    """Shared resume logic for tool results and clarification input.

    Args:
        decisions: Per-call approval decisions (call_id, decision) for an
            approval resume; validated against the pause's persisted
            ``approval_gates`` snapshot. None = legacy batch approve.
        extra_notifier: Optional additional LoopNotifier. Composed with
            PersistenceRecorder — recorder handles persistence, notifier
            handles observation side-effects.
        _skip_claim: Internal flag — skip the atomic claim step when the caller
            has already claimed via submit_and_claim(). Used by resume_claimed().
        user_parts: Already-ingested parts of a multimodal user input (the
            claimed path); ``user_input`` is then their text projection.
    """
    input_items: tuple[Any, ...] = ()
    if user_input is not None:
        user_input, input_items = split_user_content(user_input)
    if decisions is not None and (
        tool_results is not None
        or user_input is not None
        or expected_status != RunStatus.WAITING_APPROVAL.value
    ):
        raise ValueError("Approval decisions only resume a run awaiting approval.")
    skill_session = current(agent)
    if skill_session is None or skill_session.run_id != run_id:
        try:
            skill_session = await continuation(agent, state_store, run_id, loop=loop)
        except Exception as exc:
            # Internal claimed-entry callers normally inherit the verified session.
            # If they do not, a failed verification must finalize the owned claim.
            if _skip_claim:
                try:
                    won = await state_store.finalize_run(
                        run_id,
                        status=RunStatus.ERROR.value,
                        error=_safe_error(exc),
                        **_error_store_kwargs(state_store, exc),
                        expected_current_status="running",
                    )
                    if won:
                        await _emit_event(state_store, run_id, "run.error", _error_payload(exc))
                except Exception:
                    logger.error(
                        "Failed to persist claimed resume error for run %s",
                        run_id,
                        exc_info=safe_exception_info(),
                    )
                await notify_run_failed(extra_notifier, run_id, exc)
            raise
    with skill_session.bind():
        run_record = await state_store.get_run(run_id)
        reads = _PauseReads(run_record)
        # 1. Load pause state
        raw_pause = (
            None
            if _reclaim_pause is not None
            else _raw_pause
            if _raw_pause is not None
            else await state_store.get_pause_state(run_id)
        )
        if _reclaim_pause is not None:
            pause_state = _reclaim_pause
        else:
            if raw_pause is None:
                raise ValueError(f"Run '{run_id}' has no pause state — cannot resume.")
            if not _skip_claim:
                require_pause_id(
                    agent,
                    expected_pause_id,
                    pause_id=raw_pause.get("pause_id"),
                    store=state_store,
                    run_id=run_id,
                )
            pause_state = await _load_pause_state(
                state_store, run_id, raw_pause, agent, provider, reads=reads
            )
        from dendrux.runtime._reclaim import needs_resolution

        recovering = _reclaim_pause is not None or await needs_resolution(
            state_store, run_id, pause_state, checkpoint=reads.checkpoint
        )

        # 1a. Load pii_mapping for guardrail continuity across pause/resume
        _saved_pii_mapping = await state_store.get_pii_mapping(run_id)

        # 1b. Verify agent identity — fail closed on mismatch
        if pause_state.agent_name and pause_state.agent_name != agent.name:
            raise ValueError(
                f"Agent name mismatch: run '{run_id}' was paused by "
                f"'{pause_state.agent_name}', but resume called with "
                f"'{agent.name}'. This can cause state corruption."
            )

        if tool_results is not None:
            validate_result_names(run_id, pause_state, tool_results, json_payload=_skip_claim)
            if not _skip_claim:
                tool_results = await prepare_resume_results(
                    agent, state_store, run_id, pause_state, tool_results
                )

        # 2a. Images: ingest before the claim; typed errors leave the run waiting.
        if not _skip_claim and (input_items or any(tr.images for tr in tool_results or ())):
            tool_results, user_parts = await _ingest_resume_images(
                agent,
                provider,
                state_store,
                run_id,
                loop=loop,
                tool_results=tool_results,
                input_items=input_items,
            )

        # Claim validation is repeated inside the row-locking transaction.
        if not _skip_claim:
            assert raw_pause is not None
            state_store, expected_status, raw_pause = await _claim_resume(
                state_store,
                run_id,
                raw_pause,
                agent,
                expected_status,
                expected_pause_id=expected_pause_id,
                results=tool_results,
                user_input=user_input,
                user_parts=user_parts,
                decisions=decisions,
                checkpoint=reads.checkpoint,
            )

        # 4-10. Everything that runs after on_run_started fires must live
        #       inside the try block so on_run_failed can balance the lifecycle
        #       pair on any failure (including injected-history replay errors).
        ctx: _ResumeContext | None = None
        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        agent._task_manager.register_current(run_id)

        try:
            run_tenant = run_record.tenant_id if run_record else None
            # 4-8. Prepare history and notifier (shared with resume_stream).
            #      Does NOT fire any lifecycle hooks — that's the caller's job.
            ctx = await _prepare_resume(
                run_id,
                pause_state,
                state_store=state_store,
                agent=agent,
                provider=provider,
                strategy=strategy,
                loop=loop,
                expected_status=expected_status,
                tool_results=None if recovering else tool_results,
                user_input=None if recovering else user_input,
                extra_notifier=extra_notifier,
                user_parts=user_parts,
                tenant_id=run_tenant,
                pii_mapping=_saved_pii_mapping,
                history_records=(
                    _reclaim_history if _reclaim_history is not None else reads.history_records
                ),
                pause_data=raw_pause,
            )
            image_runtime = _image_runtime(agent, state_store, run_tenant, run_id)

            # 8a. Fire run-level lifecycle hooks. AFTER this point, any failure
            #     must pair with on_run_failed in the except handler.
            await record_run_started(
                ctx.recorder, run_id, agent_name=agent.name, agent_model=provider.model
            )
            trace_link, ctx.notifier = await _start_trace(
                ctx.notifier,
                state_store,
                run_id,
                agent_name=agent.name,
                agent_model=provider.model,
                saved=getattr(run_record, "trace_link", None),
            )

            await agent.get_tool_lookups(loop=ctx.resolved_loop)
            mcp_meta.update(_snapshot_mcp_meta(agent))

            # 8b. tool/user-input history. Inside the try so
            #     a recorder failure here fires on_run_failed.
            if not recovering:
                await _replay_resume_injected_history(
                    ctx, run_id, tool_results=tool_results, user_input=user_input
                )

            # 9. Set delegation context — resumed run is the active parent for any
            #    nested agent.run() calls spawned during this resume cycle.
            #    Read max_delegation_depth from persisted metadata so the depth
            #    guard survives pause/resume cycles.
            resume_delegation_level = run_record.delegation_level if run_record else 0
            resume_max_depth: int | None = None
            if run_record and run_record.meta:
                raw = run_record.meta.get("dendrux.max_delegation_depth")
                if isinstance(raw, int):
                    resume_max_depth = raw
            resume_ctx = DelegationContext(
                run_id=run_id,
                trace_link=trace_link,
                delegation_level=resume_delegation_level,
                persisted=True,
                store_identity=get_store_identity(state_store),
                max_delegation_depth=resume_max_depth,
                tenant_id=run_record.tenant_id if run_record else None,
                tenant_explicit=True if _tenant_explicit is None else _tenant_explicit,
                state_store=state_store,
                lease_interval_s=agent.lease_interval.total_seconds(),
                loop=asyncio.get_running_loop(),
            )
            ctx_token = set_delegation_context(resume_ctx)

            if recovering:
                from dendrux.runtime._reclaim import resolve

                stop = await resolve(agent, state_store, run_id, ctx, image_runtime=image_runtime)
                if stop is not None:
                    stopped = await _persist_loop_outcome(
                        recorder=ctx.recorder,
                        provider_kwargs=ctx.pause_state.provider_kwargs,
                        state_store=state_store,
                        run_id=run_id,
                        result=stop,
                    )
                    await record_run_finished(ctx.recorder, run_id, stopped)
                    await notify_run_finished(ctx.notifier, run_id, stopped)
                    return stopped

            # 10. Re-enter loop (batch)
            # Approval-approve: execute pending tools before loop re-entry.
            # Runs inside try/except so failures properly finalize as ERROR.
            _is_approval_approve = (
                tool_results is None
                and user_input is None
                and expected_status == RunStatus.WAITING_APPROVAL.value
            )
            # A pause that interrupted an approval batch resumes its remainder.
            _is_paused_batch = expected_status == RunStatus.PAUSED.value and bool(
                pause_state.pending_tool_calls
            )
            if not recovering and (_is_approval_approve or _is_paused_batch):
                decision_map = (
                    await _validated_decisions(agent, ctx, run_id, pause_state, decisions)
                    if _is_approval_approve
                    else _pending_decision_map(pause_state)
                )
                approval = await _execute_approved_tools(
                    agent,
                    pause_state,
                    ctx.recorder,
                    ctx.notifier,
                    run_id,
                    history=ctx.history,
                    resolved_loop=ctx.resolved_loop,
                    decisions=decision_map,
                    state_store=state_store,
                    image_runtime=image_runtime,
                )

                if _is_approval_approve:
                    await _record_approval_decided(
                        ctx.recorder, ctx.notifier, run_id, pause_state.iteration, approval.decided
                    )

                stop = _approval_stop_result(run_id, pause_state, ctx.history, approval)
                if stop is not None:
                    _attach_mcp_meta(stop, mcp_meta)
                    stopped = await _persist_loop_outcome(
                        recorder=ctx.recorder,
                        provider_kwargs=ctx.pause_state.provider_kwargs,
                        state_store=state_store,
                        run_id=run_id,
                        result=stop,
                    )
                    await record_run_finished(ctx.recorder, run_id, stopped)
                    await notify_run_finished(ctx.notifier, run_id, stopped)
                    return stopped

            # Approval-reject: tool results already in history from _prepare_resume.
            # Emit governance event only.
            if (
                not recovering
                and tool_results is not None
                and expected_status == RunStatus.WAITING_APPROVAL.value
            ):
                _rejected_data = _rejection_decided_payload(run_id, pause_state, tool_results)
                await _record_approval_decided(
                    ctx.recorder, ctx.notifier, run_id, pause_state.iteration, _rejected_data
                )

            result = await ctx.resolved_loop.run(
                agent=agent,
                provider=provider,
                strategy=ctx.resolved_strategy,
                user_input="",
                run_id=run_id,
                recorder=ctx.recorder,
                notifier=ctx.notifier,
                initial_history=ctx.history,
                initial_steps=ctx.pause_state.steps,
                iteration_offset=ctx.pause_state.iteration,
                initial_usage=ctx.pause_state.usage,
                provider_kwargs=ctx.pause_state.provider_kwargs or None,
                initial_pii_mapping=_saved_pii_mapping,
                state_store=state_store,
                **_budget_carry_kwargs(ctx.resolved_loop.run, ctx.pause_state),
                **_image_loop_kwargs(
                    ctx.resolved_loop.run,
                    image_runtime,
                    _resume_image_view(ctx.pause_state, ctx.history),
                ),
            )

            _attach_mcp_meta(result, mcp_meta)
            result = await _persist_loop_outcome(
                recorder=ctx.recorder,
                provider_kwargs=ctx.pause_state.provider_kwargs,
                state_store=state_store,
                run_id=run_id,
                result=result,
            )

            await record_run_finished(ctx.recorder, run_id, result)
            await notify_run_finished(ctx.notifier, run_id, result)

            return result

        except asyncio.CancelledError:
            interrupted, others_pending = agent._task_manager.take_interrupt(run_id)
            current_task = asyncio.current_task()
            swallow = (
                interrupted
                and not others_pending
                and current_task is not None
                and current_task.uncancel() == 0
            )

            async def persist_cancellation() -> RunResult:
                return await _persist_task_cancellation(
                    state_store, run_id, interrupted=interrupted
                )

            cleanup = asyncio.create_task(persist_cancellation(), name=f"cancel-resume:{run_id}")
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cancelled_result = cleanup.result()
            if ctx is not None:
                try:
                    await record_run_finished(ctx.recorder, run_id, cancelled_result)
                except Exception:
                    logger.warning(
                        "Recorder finish failed for run %s", run_id, exc_info=safe_exception_info()
                    )
                await notify_run_finished(ctx.notifier, run_id, cancelled_result)
            if swallow and (current_task is None or current_task.cancelling() == 0):
                return cancelled_result
            raise

        except RunFencedError:
            fenced_result = await _build_cached_result(state_store, run_id)
            if ctx is not None:
                await _finish_fenced_lifecycle(run_id, fenced_result, ctx.recorder, ctx.notifier)
            return fenced_result

        except Exception as exc:
            _attach_run_error_meta(exc, run_id, mcp_meta)
            error_won = False
            try:
                error_won = await state_store.finalize_run(
                    run_id,
                    status=RunStatus.ERROR.value,
                    error=_safe_error(exc),
                    **_error_store_kwargs(state_store, exc),
                    total_usage=None,
                    expected_current_status="running",
                )
            except Exception:
                logger.error(
                    "Failed to persist ERROR status for run %s",
                    run_id,
                    exc_info=safe_exception_info(),
                )
            if error_won:
                await _emit_event(state_store, run_id, "run.error", _error_payload(exc))
            # Fire on_run_failed only if _prepare_resume returned (ctx exists).
            # If _prepare_resume itself raised, on_run_started never fired, so
            # the lifecycle pair is balanced (both unfired).
            if ctx is not None:
                try:
                    await record_run_failed(ctx.recorder, run_id, exc)
                    await notify_run_failed(ctx.notifier, run_id, exc)
                except Exception:
                    logger.warning(
                        "Lifecycle hooks failed during resume error handler for run %s",
                        run_id,
                        exc_info=safe_exception_info(),
                    )
            raise

        finally:
            agent._task_manager.release_current(run_id)
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


def resume_stream(
    run_id: str,
    *,
    agent: Agent,
    provider: LLMProvider,
    state_store: StateStore | None = None,
    state_store_resolver: Callable[[], Any] | None = None,
    tool_results: list[ToolResult] | None = None,
    user_input: UserContent | None = None,
    extra_notifier: LoopNotifier | None = None,
    expected_pause_id: str | None = None,
    _tenant_explicit: bool | None = None,
) -> RunStream:
    """Stream a resumed run as RunEvents, returning a RunStream.

    Synchronous — returns immediately. All async setup (store resolution,
    pause state load, claim, history build) runs lazily on first iteration.

    Same error contract as run_stream(): exceptions are caught, persisted,
    and yielded as RUN_ERROR events. No exception is raised to the consumer.

    First event is RUN_RESUMED (not RUN_STARTED) carrying the existing run_id.

    Image inputs (``tool_results[i].images`` or a multimodal ``user_input``)
    are ingested before the claim; an image error is raised from the first
    ``__anext__`` and leaves the run waiting.
    """
    from dendrux.types import RunEvent, RunResult
    from dendrux.types import RunStream as _RunStream

    input_items: tuple[Any, ...] = ()
    if user_input is not None:
        user_input, input_items = split_user_content(user_input)
    image_input = bool(input_items) or any(tr.images for tr in tool_results or ())

    _shared: dict[str, Any] = {"state_store": state_store, "ctx": None}

    mcp_meta: dict[str, Any] = {}

    stream_context = StreamContext()

    async def _generate() -> AsyncGenerator[RunEvent, None]:
        store = state_store
        stream_results = tool_results
        user_parts: tuple[ContentPart, ...] | None = None

        results_prepared = False
        if image_input:
            # Image setup raises (the run stays waiting; no RUN_ERROR event).
            if store is None and state_store_resolver is not None:
                store = await state_store_resolver()
            if store is None:
                from dendrux.errors import PersistenceNotConfiguredError

                raise PersistenceNotConfiguredError()
            raw = await store.get_pause_state(run_id)
            if raw is None:
                raise ValueError("Run has no pause state")
            require_pause_id(
                agent, expected_pause_id, pause_id=raw.get("pause_id"), store=store, run_id=run_id
            )
            media_pause = PauseState.from_dict(raw)
            if media_pause.agent_name and media_pause.agent_name != agent.name:
                raise ValueError("Agent name mismatch on resume")
            if stream_results is not None:
                validate_result_names(run_id, media_pause, stream_results, json_payload=False)
                stream_results = await prepare_resume_results(
                    agent, store, run_id, media_pause, stream_results
                )
                results_prepared = True
            stream_results, user_parts = await _ingest_resume_images(
                agent,
                provider,
                store,
                run_id,
                loop=None,
                tool_results=stream_results,
                input_items=input_items,
                loop_method="run_stream",
            )

        try:
            # 1. Resolve state store
            if store is None and state_store_resolver is not None:
                store = await state_store_resolver()
            _shared["state_store"] = store

            if store is None:
                raise ValueError(
                    "resume_stream() requires persistence. "
                    "Pass database_url or state_store to the agent."
                )

            skill_session = await continuation(agent, store, run_id)
            stream_context.set_session(skill_session)

            # 2. Load pause state
            raw_pause = await store.get_pause_state(run_id)
            if raw_pause is None:
                raise ValueError(f"Run '{run_id}' has no pause state — cannot resume.")
            require_pause_id(
                agent,
                expected_pause_id,
                pause_id=raw_pause.get("pause_id"),
                store=store,
                run_id=run_id,
            )
            run_record = await store.get_run(run_id)
            reads = _PauseReads(run_record)
            pause_state = await _load_pause_state(
                store, run_id, raw_pause, agent, provider, reads=reads
            )
            from dendrux.runtime._reclaim import needs_resolution

            recovering = await needs_resolution(
                store, run_id, pause_state, checkpoint=reads.checkpoint
            )

            # 2b. Verify agent identity
            if pause_state.agent_name and pause_state.agent_name != agent.name:
                raise ValueError(
                    f"Agent name mismatch: run '{run_id}' was paused by "
                    f"'{pause_state.agent_name}', but resume called with "
                    f"'{agent.name}'."
                )

            if stream_results is not None:
                validate_result_names(run_id, pause_state, stream_results, json_payload=False)
                if not results_prepared:
                    stream_results = await prepare_resume_results(
                        agent, store, run_id, pause_state, stream_results
                    )

            # 4. Determine expected status and claim
            if stream_results is not None:
                expected = RunStatus.WAITING_CLIENT_TOOL.value
            elif user_input is not None:
                expected = RunStatus.WAITING_HUMAN_INPUT.value
            else:
                # No-arg = approval-approve, or continue a PAUSED run
                expected = RunStatus.WAITING_APPROVAL.value

            store, expected, raw_pause = await _claim_resume(
                store,
                run_id,
                raw_pause,
                agent,
                expected,
                expected_pause_id=expected_pause_id,
                results=stream_results,
                user_input=user_input,
                user_parts=user_parts,
                checkpoint=reads.checkpoint,
            )
            _shared["state_store"] = store
            _shared["claimed"] = True

            # 5. Prepare (shared helper — history, notifier).
            #    Does NOT fire any lifecycle hooks; that's done below inside
            #    the same try block so on_run_failed pairs with any failure
            #    after on_run_started fires.
            run_tenant = run_record.tenant_id if run_record else None
            ctx = await _prepare_resume(
                run_id,
                pause_state,
                state_store=store,
                agent=agent,
                provider=provider,
                expected_status=expected,
                tool_results=None if recovering else stream_results,
                user_input=None if recovering else user_input,
                extra_notifier=extra_notifier,
                user_parts=user_parts,
                tenant_id=run_tenant,
                history_records=reads.history_records,
                pause_data=raw_pause,
            )
            image_runtime = _image_runtime(agent, store, run_tenant, run_id)
            _shared["ctx"] = ctx

            # 5a. Fire run-level lifecycle hooks. After this point, any
            #     failure must pair with on_run_failed in the except handler.
            await record_run_started(
                ctx.recorder, run_id, agent_name=agent.name, agent_model=provider.model
            )
            trace_link, ctx.notifier = await _start_trace(
                ctx.notifier,
                store,
                run_id,
                agent_name=agent.name,
                agent_model=provider.model,
                saved=getattr(run_record, "trace_link", None),
            )
            # Lifecycle pair is now open. Stash recorder/notifier so the
            # cleanup callback can close it with a CANCELLED finish if the
            # consumer abandons the stream before a terminal event.
            _shared["recorder"] = ctx.recorder
            _shared["notifier"] = ctx.notifier

            await agent.get_tool_lookups(loop=ctx.resolved_loop)
            mcp_meta.update(_snapshot_mcp_meta(agent))

            # 5b. tool/user-input history. Inside the try
            #     so a recorder failure here fires on_run_failed.
            if not recovering:
                await _replay_resume_injected_history(
                    ctx, run_id, tool_results=stream_results, user_input=user_input
                )

            # 6. Set delegation context for the generator lifetime.
            #    Read max_delegation_depth from persisted metadata so the
            #    depth guard survives pause/resume cycles.
            resume_level = run_record.delegation_level if run_record else 0
            resume_max_depth: int | None = None
            if run_record and run_record.meta:
                raw = run_record.meta.get("dendrux.max_delegation_depth")
                if isinstance(raw, int):
                    resume_max_depth = raw
            resume_ctx = DelegationContext(
                run_id=run_id,
                trace_link=trace_link,
                delegation_level=resume_level,
                persisted=True,
                store_identity=get_store_identity(store),
                max_delegation_depth=resume_max_depth,
                tenant_id=run_record.tenant_id if run_record else None,
                tenant_explicit=True if _tenant_explicit is None else _tenant_explicit,
                state_store=store,
                lease_interval_s=agent.lease_interval.total_seconds(),
                loop=asyncio.get_running_loop(),
            )
            stream_context.set_delegation(resume_ctx)

            # 7. Emit RUN_RESUMED as first event (contract guarantee)
            yield RunEvent(type=RunEventType.RUN_RESUMED, run_id=run_id)

            if recovering:
                from dendrux.runtime._reclaim import resolve

                stop = await resolve(agent, store, run_id, ctx, image_runtime=image_runtime)
                if stop is not None:
                    stopped = await _persist_loop_outcome(
                        recorder=ctx.recorder,
                        provider_kwargs=ctx.pause_state.provider_kwargs,
                        state_store=store,
                        run_id=run_id,
                        result=stop,
                    )
                    await record_run_finished(ctx.recorder, run_id, stopped)
                    await notify_run_finished(ctx.notifier, run_id, stopped)
                    yield RunEvent(
                        type=RunEventType.RUN_CANCELLED
                        if stopped.status == RunStatus.CANCELLED
                        else RunEventType.RUN_PAUSED,
                        run_result=stopped,
                    )
                    return

            # 8. Approval-execute (inside try/except for proper error handling)
            _is_approval_approve = (
                stream_results is None
                and user_input is None
                and expected == RunStatus.WAITING_APPROVAL.value
            )
            _is_paused_batch = expected == RunStatus.PAUSED.value and bool(
                pause_state.pending_tool_calls
            )
            if not recovering and (_is_approval_approve or _is_paused_batch):
                approval = await _execute_approved_tools(
                    agent,
                    pause_state,
                    ctx.recorder,
                    ctx.notifier,
                    run_id,
                    history=ctx.history,
                    resolved_loop=ctx.resolved_loop,
                    decisions=None if _is_approval_approve else _pending_decision_map(pause_state),
                    state_store=store,
                    image_runtime=image_runtime,
                )
                for tc, tr in approval.results:
                    yield RunEvent(
                        type=RunEventType.TOOL_RESULT,
                        tool_call=tc,
                        tool_result=tr,
                        iteration=pause_state.iteration,
                        phase="commentary",
                    )

                if _is_approval_approve:
                    await _record_approval_decided(
                        ctx.recorder, ctx.notifier, run_id, pause_state.iteration, approval.decided
                    )

                stop = _approval_stop_result(run_id, pause_state, ctx.history, approval)
                if stop is not None:
                    _attach_mcp_meta(stop, mcp_meta)
                    stopped = await _persist_loop_outcome(
                        recorder=ctx.recorder,
                        provider_kwargs=ctx.pause_state.provider_kwargs,
                        state_store=store,
                        run_id=run_id,
                        result=stop,
                    )
                    await record_run_finished(ctx.recorder, run_id, stopped)
                    await notify_run_finished(ctx.notifier, run_id, stopped)
                    yield RunEvent(
                        type=(
                            RunEventType.RUN_CANCELLED
                            if stopped.status == RunStatus.CANCELLED
                            else RunEventType.RUN_PAUSED
                        ),
                        run_result=stopped,
                    )
                    return

            if (
                not recovering
                and stream_results is not None
                and expected == RunStatus.WAITING_APPROVAL.value
            ):
                _rejected_data_s = _rejection_decided_payload(run_id, pause_state, stream_results)
                await _record_approval_decided(
                    ctx.recorder, ctx.notifier, run_id, pause_state.iteration, _rejected_data_s
                )

            # 9. Stream the loop (see run_stream for interrupt + closing notes)
            interrupt = agent._task_manager.register_interrupt(run_id)
            loop_stream = ctx.resolved_loop.run_stream(
                agent=agent,
                provider=provider,
                strategy=ctx.resolved_strategy,
                user_input="",
                run_id=run_id,
                recorder=ctx.recorder,
                notifier=ctx.notifier,
                initial_history=ctx.history,
                initial_steps=ctx.pause_state.steps,
                iteration_offset=ctx.pause_state.iteration,
                initial_usage=ctx.pause_state.usage,
                provider_kwargs=ctx.pause_state.provider_kwargs or None,
                state_store=store,
                interrupt=interrupt,
                **_budget_carry_kwargs(ctx.resolved_loop.run_stream, ctx.pause_state),
                **_image_loop_kwargs(
                    ctx.resolved_loop.run_stream,
                    image_runtime,
                    _resume_image_view(ctx.pause_state, ctx.history),
                ),
            )
            async with _closing_loop_stream(loop_stream, run_id):
                async for event in loop_stream:
                    if (
                        event.type
                        in (
                            RunEventType.RUN_COMPLETED,
                            RunEventType.RUN_ERROR,
                            RunEventType.RUN_PAUSED,
                            RunEventType.RUN_CANCELLED,
                        )
                        and event.run_result
                    ):
                        _attach_mcp_meta(event.run_result, mcp_meta)
                        persisted = await _persist_loop_outcome(
                            recorder=ctx.recorder,
                            provider_kwargs=ctx.pause_state.provider_kwargs,
                            state_store=store,
                            run_id=run_id,
                            result=event.run_result,
                        )
                        if persisted.status == RunStatus.CANCELLED:
                            event = RunEvent(type=RunEventType.RUN_CANCELLED, run_result=persisted)
                        else:
                            event = RunEvent(
                                type=event.type, run_result=persisted, phase=event.phase
                            )
                        await record_run_finished(ctx.recorder, run_id, persisted)
                        await notify_run_finished(ctx.notifier, run_id, persisted)
                        yield event
                    else:
                        _track_partial_text(_shared, event)
                        yield event

        except RunFencedError:
            store = _shared["state_store"]
            result = await _build_cached_result(store, run_id)
            fenced_ctx = _shared.get("ctx")
            if fenced_ctx is not None:
                await _finish_fenced_lifecycle(
                    run_id, result, fenced_ctx.recorder, fenced_ctx.notifier
                )
            yield _fenced_stream_event(run_id, result)

        except Exception as exc:
            store = _shared.get("state_store")
            err_ctx = _shared.get("ctx")
            if store is not None and _shared.get("claimed"):
                error_won = False
                try:
                    error_won = await store.finalize_run(
                        run_id,
                        status=RunStatus.ERROR.value,
                        error=_safe_error(exc),
                        **_error_store_kwargs(store, exc),
                        total_usage=None,
                        expected_current_status="running",
                    )
                except Exception:
                    logger.error(
                        "Failed to persist ERROR status for run %s",
                        run_id,
                        exc_info=safe_exception_info(),
                    )
                if error_won:
                    await _emit_event_safe(
                        store,
                        run_id,
                        "run.error",
                        _error_payload(exc),
                    )
            try:
                if err_ctx is not None:
                    await record_run_failed(err_ctx.recorder, run_id, exc)
                    await notify_run_failed(err_ctx.notifier, run_id, exc)
                else:
                    # Setup failed before _prepare_resume built the recorder.
                    # Fire the notifier-side hook so OTel can surface the error
                    # even when no DB-side recorder exists.
                    await notify_run_failed(extra_notifier, run_id, exc)
            except Exception:
                logger.warning(
                    "Lifecycle hooks failed during resume_stream error handler for run %s",
                    run_id,
                    exc_info=safe_exception_info(),
                )

            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=RunStatus.ERROR,
                    error=_safe_error(exc),
                    error_info=_provider_error_info(exc),
                    meta=_stream_error_meta(mcp_meta, exc),
                ),
                error=_safe_error(exc),
            )

        finally:
            if _shared.get("claimed"):
                agent._task_manager.release_interrupt(run_id)

    async def _cleanup() -> RunResult | None:
        """CAS-guarded cancellation + local lifecycle close.

        Mirrors run_stream._cleanup. The lifecycle close fires whether or
        not the CAS won — if some other caller already finalized the run,
        the local notifier still needs the close event so OTel root spans
        don't leak. Status resolution: CAS won → CANCELLED, CAS lost →
        query store for actual status (falling back to CANCELLED).
        """
        if not _shared.get("claimed"):
            return None
        store = _shared.get("state_store")
        # run.cancelled is recorded only once the resume itself was recorded.
        resumed = _shared.get("ctx") is not None

        cancel_won = False
        partial_answer = _partial_answer(_shared)
        if store is not None:
            try:
                cancel_won = await store.finalize_run(
                    run_id,
                    status=RunStatus.CANCELLED.value,
                    answer=partial_answer,
                    expected_current_status="running",
                )
                if cancel_won and resumed:
                    await _emit_event(
                        store,
                        run_id,
                        "run.cancelled",
                        {"reason": "stream_closed", "partial_output": partial_answer is not None},
                    )
            except Exception:
                logger.error(
                    "Failed to cancel run %s during resume stream cleanup",
                    run_id,
                    exc_info=safe_exception_info(),
                )

        if "notifier" in _shared:
            cleanup_recorder = _shared.get("recorder")
            terminal_status = RunStatus.CANCELLED
            cancel_persisted = cancel_won
            if store is not None and not cancel_won:
                try:
                    run_record = await store.get_run(run_id)
                    if run_record is not None:
                        terminal_status = RunStatus(run_record.status)
                        cancel_persisted = terminal_status == RunStatus.CANCELLED
                except Exception:
                    pass  # CANCELLED is a safe fallback
            cancelled_result = RunResult(
                run_id=run_id,
                status=terminal_status,
                meta={**deepcopy(mcp_meta), "dendrux.cancel_persisted": cancel_persisted},
            )
            try:
                await record_run_finished(cleanup_recorder, run_id, cancelled_result)
                await notify_run_finished(_shared.get("notifier"), run_id, cancelled_result)
            except Exception:
                logger.warning(
                    "Lifecycle finish hooks failed during resume stream cleanup for run %s",
                    run_id,
                    exc_info=safe_exception_info(),
                )

            return cancelled_result
        return None

    return _RunStream(
        run_id=run_id,
        generator=stream_context.wrap(_generate()),
        cleanup=lambda: _shielded_cleanup(_cleanup),
    )


async def _owned_claim_view(
    store: StateStore, run_id: str, claim_epoch: int | None = None
) -> StateStore:
    if claim_epoch is not None:
        return claimed_store(store, run_id, claim_epoch)
    # Legacy internal callers without a token adopt the current epoch best effort.
    if getattr(store, "claim_epoch", None) is None:
        getter = getattr(store, "get_run_control", None)
        if getter is not None:
            control = await getter(run_id)
            if control.claim_epoch is not None:
                return claimed_store(store, run_id, control.claim_epoch)
    return store


async def _validate_owned_pause(
    store: StateStore,
    run_id: str,
    agent: Agent,
    status: str,
    *,
    require_submission: bool = False,
) -> dict[str, Any]:
    from dendrux.runtime.submit import (
        deserialize_decisions,
        deserialize_tool_results,
        validate_approval_decisions,
    )

    try:
        await _check_claim(store, run_id)
        raw = await store.get_pause_state(run_id)
        if raw is None:
            raise ValueError("Claimed run has no pause state")
        submission_validator(run_id, raw, agent=agent)
        from dendrux.runtime.submit import submitted_view

        raw = submitted_view(raw)
        submitted = raw.get("submitted_tool_results")
        if submitted is not None:
            validate_result_names(
                run_id, PauseState.from_dict(raw), deserialize_tool_results(submitted)
            )
        elif raw.get("submitted_decisions") is not None:
            lookups = await agent.get_tool_lookups()
            validate_approval_decisions(
                run_id,
                PauseState.from_dict(raw),
                deserialize_decisions(raw["submitted_decisions"]),
                lookups.target,
            )
        elif require_submission and raw.get("submitted_user_input") is None:
            raise ValueError("Claimed run has no submitted input")
        return raw
    except Exception:
        await _release_claim(store, run_id, status)
        raise


async def resume_claimed(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    extra_notifier: LoopNotifier | None = None,
    pre_claim_status: str | None = None,
    claim_epoch: int | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Resume a run that was already claimed via submit_and_claim().

    Raw-store callers should pass the epoch returned by their claim as
    ``claim_epoch=``. Omitting it retains the legacy best-effort current-epoch
    lookup. SDK submissions pass an already-bound store view.


    Internal helper for the persist-first handoff. NOT a public API.

    Agent submit methods call submit_and_claim() to atomically persist the
    submitted payload and transition WAITING → RUNNING, then call this
    helper to re-enter the loop.

    Validates before proceeding:
      - pause_data exists (run was paused)
      - submitted_tool_results or submitted_user_input present in blob
      - for tool results: call_ids match pending_tool_calls

    Args:
        run_id: The run ID (already in RUNNING status after submit_and_claim).
        state_store: Persistence backend.
        agent: Agent definition.
        provider: LLM provider.
        extra_notifier: Optional additional LoopNotifier.
        pre_claim_status: The pause status the run was in before
            submit_and_claim flipped it to RUNNING. Forwarded to
            ``_resume_core`` as ``expected_status`` so downstream
            governance gates (e.g. ``approval.decided=rejected`` for
            ``submit_approval(approved=False)``) can fire correctly.
            Defaults to ``"running"`` for back-compat.

    Returns:
        RunResult from the resumed loop.

    Raises:
        ValueError: If pause_data is missing, no submitted data found,
            or submitted call_ids don't match pending tool calls.
    """
    from dendrux.runtime.submit import deserialize_tool_results

    try:
        state_store = await _owned_claim_view(state_store, run_id, claim_epoch)
        raw_pause = await _validate_owned_pause(
            state_store,
            run_id,
            agent,
            pre_claim_status or RunStatus.WAITING_CLIENT_TOOL.value,
            require_submission=True,
        )
    except RunFencedError:
        return await _build_cached_result(state_store, run_id)

    # 2. Detect submission type and extract data
    submitted_results = raw_pause.get("submitted_tool_results")
    submitted_input = raw_pause.get("submitted_user_input")

    if submitted_results is not None:
        # 3a. Convert dicts to typed ToolResult objects (images as stored refs)
        tool_results = deserialize_tool_results(submitted_results)

        # 3b. Validate call_ids match pending tool calls
        pending_ids = {tc["id"] for tc in raw_pause.get("pending_tool_calls", [])}
        provided_ids = {tr.call_id for tr in tool_results}
        if provided_ids != pending_ids:
            raise ValueError(
                f"Submitted tool result call_ids {provided_ids} do not match "
                f"pending tool call_ids {pending_ids}."
            )

        return await _resume_core(
            run_id,
            state_store=state_store,
            agent=agent,
            provider=provider,
            expected_status=pre_claim_status or "running",
            tool_results=tool_results,
            extra_notifier=extra_notifier,
            _skip_claim=True,
            _raw_pause=raw_pause,
            _tenant_explicit=_tenant_explicit,
        )

    submitted_decisions = raw_pause.get("submitted_decisions")
    if submitted_decisions is not None:
        from dendrux.runtime.submit import deserialize_decisions

        # Re-validated against the approval_gates snapshot inside _resume_core,
        # where a mismatch finalizes the claimed run as ERROR.
        return await _resume_core(
            run_id,
            state_store=state_store,
            agent=agent,
            provider=provider,
            expected_status=RunStatus.WAITING_APPROVAL.value,
            extra_notifier=extra_notifier,
            _skip_claim=True,
            decisions=deserialize_decisions(submitted_decisions),
            _raw_pause=raw_pause,
            _tenant_explicit=_tenant_explicit,
        )

    if submitted_input is not None:
        return await _resume_core(
            run_id,
            state_store=state_store,
            agent=agent,
            provider=provider,
            expected_status=pre_claim_status or "running",
            user_input=submitted_input,
            extra_notifier=extra_notifier,
            _skip_claim=True,
            user_parts=_stored_parts(raw_pause.get("submitted_user_parts")),
            _raw_pause=raw_pause,
            _tenant_explicit=_tenant_explicit,
        )

    # No submitted data found. Approval-approve flows do not go through
    # resume_claimed — they call resume_approval_approved_claim() below,
    # which takes the explicit WAITING_APPROVAL path. Reaching this point
    # implies a bug in the claim flow.
    raise ValueError(
        f"Run '{run_id}' has pause_data but no submitted_tool_results "
        "or submitted_user_input — nothing to resume with."
    )


async def resume_approval_approved_claim(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    extra_notifier: LoopNotifier | None = None,
    claim_epoch: int | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Resume an approval-approved run that was already claimed.

    Raw-store callers should pass the epoch returned by their claim as
    ``claim_epoch=``. Omitting it retains the legacy best-effort current-epoch
    lookup. SDK submissions pass an already-bound store view.


    Paired with :func:`resume_claimed`: ``resume_claimed`` handles the
    post-claim path for tool results and user input, while this function
    handles approval-approve (no submitted payload — just re-enter the
    loop with the pending batch executed server-side).

    The caller (e.g. :meth:`Agent.submit_approval`) must have already
    transitioned the run to ``RUNNING`` via
    :meth:`StateStore.claim_paused_run` with
    ``expected_status="waiting_approval"``.
    """
    try:
        state_store = await _owned_claim_view(state_store, run_id, claim_epoch)
        raw_pause = await _validate_owned_pause(
            state_store, run_id, agent, RunStatus.WAITING_APPROVAL.value
        )
    except RunFencedError:
        return await _build_cached_result(state_store, run_id)
    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        expected_status=RunStatus.WAITING_APPROVAL.value,
        tool_results=None,
        user_input=None,
        extra_notifier=extra_notifier,
        _skip_claim=True,
        _raw_pause=raw_pause,
        _tenant_explicit=_tenant_explicit,
    )


async def resume_paused_claim(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    extra_notifier: LoopNotifier | None = None,
    claim_epoch: int | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Continue a PAUSED run that :meth:`Agent.submit_resume` already claimed.

    Raw-store callers should pass the epoch returned by their claim as
    ``claim_epoch=``. Omitting it retains the legacy best-effort current-epoch
    lookup. SDK submissions pass an already-bound store view.


    The claim cleared ``pause_requested``. The loop resumes with the full
    history; a pause that interrupted an approval batch first executes the
    batch's remainder with its persisted decisions.
    """
    try:
        state_store = await _owned_claim_view(state_store, run_id, claim_epoch)
        raw_pause = await _validate_owned_pause(state_store, run_id, agent, RunStatus.PAUSED.value)
    except RunFencedError:
        return await _build_cached_result(state_store, run_id)
    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        expected_status=RunStatus.PAUSED.value,
        extra_notifier=extra_notifier,
        _skip_claim=True,
        _raw_pause=raw_pause,
        _tenant_explicit=_tenant_explicit,
    )
