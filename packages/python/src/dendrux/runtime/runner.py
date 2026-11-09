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
from contextlib import aclosing, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, get_args, overload

from dendrux import _codes
from dendrux._lifecycle import (
    CANCEL_FINALIZE_STATUS_VALUES,
    RESUME_INPUT_STATUS_VALUES,
    RESUME_RESULTS_STATUS_VALUES,
    STORAGE_FORMAT_CHECKPOINTED,
    TERMINAL_STATUSES,
    CancelReason,
    DenialReason,
    FailureReason,
    PauseReason,
    StoredEventType,
)
from dendrux._lifecycle import PAUSE_STATUSES as _PAUSE_STATUSES
from dendrux._meta_keys import (
    CANCEL_PERSISTED_META_KEY,
    ERROR_INFO_META_KEY,
    LOOP_META_KEY,
    MAX_DELEGATION_DEPTH_META_KEY,
    NOT_EXECUTED_META_KEY,
    OUTPUT_TYPE_META_KEY,
    RETRY_OF_META_KEY,
    SEED_PREFIX_META_KEY,
    SKILLS_CONTEXT_META_KEY,
)
from dendrux._meta_keys import (
    STOP_REASON_META_KEY as STOP_REASON_META_KEY,
)
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
from dendrux.errors import (
    BlobNotFoundError,
    InvalidImageError,
    RunFencedError,
    RunNotFoundError,
    StoreOutcomeUnknown,
    safe_exception_info,
)
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
from dendrux.loops._policy_context import dispatch_policy, history_call_counts, static_denial
from dendrux.loops.react import ReActLoop
from dendrux.runtime import _reclaim
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
    _load_replay_runs,
    _pair_tool_calls,
    build_seed,
    current_prefix,
    expand_replay,
    has_images,
    image_slot_count,
    initial_image_view,
    load_image_sources,
    parts_to_dicts,
    rebuild_messages,
    record_run_prefix,
    user_trace_message,
    validate_replay_args,
)
from dendrux.runtime.replay import load_replay_runs as load_replay_runs
from dendrux.runtime.state import ClaimResult, _initial_loop_state, claimed_store, unclaimed_store
from dendrux.runtime.stores._limits_fit import ensure_agent_limits_fit
from dendrux.runtime.submit import (
    GOVERNANCE_KEY,
    _claim_failure_mapping,
    _claim_request,
    _claim_submission,
    _claimed_submission_view,
    _release_supported,
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
    AgentStep,
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

_CONTROL_EVENT_LIMIT = 100

ERROR_TEXT_CHARS = 500
_TOOL_NAME_SCAN_LIMIT = 1000
_TOOL_NAME_CHARS = 64
_DEFAULT_RECOVERY_ITERATIONS = 10

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence

    from dendrux.agent import Agent
    from dendrux.chat import ChatMessage
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.media._catalog import BlobEdge
    from dendrux.media._image import UserContent
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import RunRecord, StateStore, TraceRecord
    from dendrux.runtime.tasks import RunTaskManager
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        ContentPart,
        CreateRunResult,
        Limits,
        RunEvent,
        RunResult,
        RunStream,
        ToolCall,
        ToolResult,
    )

logger = logging.getLogger(__name__)


_STOP_REASONS = frozenset(get_args(StopReason))


async def _finalize_running_event(
    store: StateStore,
    run_id: str,
    status: str,
    event_type: str,
    event_data: dict[str, Any],
    *,
    emit_event: bool = True,
    **kwargs: Any,
) -> tuple[bool, bool]:
    atomic = getattr(store, "finalize_with_event", None)
    if atomic is not None and emit_event:
        kwargs.pop("error_info", None)  # Carried by the atomic lifecycle event.
        seq = await atomic(
            run_id,
            status=status,
            allowed_current_statuses=[RunStatus.RUNNING.value],
            event_type=event_type,
            event_data=event_data,
            **kwargs,
        )
        return seq is not None, False
    won = await store.finalize_run(
        run_id, status=status, expected_current_status=RunStatus.RUNNING.value, **kwargs
    )
    return bool(won), bool(won and emit_event)


async def _persist_run_failure(
    store: StateStore, run_id: str, exc: Exception, *, safe_event: bool = False
) -> bool:
    try:
        won, needs_event = await _finalize_running_event(
            store,
            run_id,
            RunStatus.ERROR.value,
            StoredEventType.RUN_ERROR.value,
            _error_payload(exc),
            error=_safe_error(exc),
            **_error_store_kwargs(store, exc),
            total_usage=None,
        )
    except Exception:
        logger.error(
            "Failed to persist ERROR status for run %s", run_id, exc_info=safe_exception_info()
        )
        return False
    if needs_event:
        emit = _emit_event_safe if safe_event else _emit_event
        await emit(store, run_id, StoredEventType.RUN_ERROR.value, _error_payload(exc))
    return bool(won)


async def _stream_run_failure(
    store: StateStore | None,
    shared: dict[str, Any],
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    exc: Exception,
) -> None:
    if store is not None and shared.get("claimed"):
        await _persist_run_failure(store, run_id, exc, safe_event=True)
    if not shared.get("existing_run"):
        try:
            await record_run_failed(recorder, run_id, exc)
            await notify_run_failed(notifier, run_id, exc)
        except Exception:
            logger.warning(
                "Lifecycle hooks failed during stream error handler for run %s",
                run_id,
                exc_info=safe_exception_info(),
            )


def _take_task_interrupt(manager: RunTaskManager | None, run_id: str) -> tuple[bool, bool]:
    if manager is None:
        return False, False
    interrupted, others = manager.take_interrupt(run_id)
    task = asyncio.current_task()
    swallow = interrupted and not others and task is not None and task.uncancel() == 0
    return interrupted, swallow


def _may_swallow_cancellation(swallow: bool) -> bool:
    task = asyncio.current_task()
    return swallow and (task is None or task.cancelling() == 0)


async def _drain_task_cancellation(
    store: StateStore | None, run_id: str, interrupted: bool
) -> RunResult:
    from dendrux.runtime.tasks import _drain_cleanup

    result, _ = await _drain_cleanup(
        _persist_task_cancellation(store, run_id, interrupted=interrupted),
        name=f"cancel-run:{run_id}",
        run_id=run_id,
    )
    return result


@dataclass
class _RunLifecycle:
    run_id: str
    store: StateStore | None
    manager: RunTaskManager | None
    owned: bool = False
    registered: bool = False
    started: bool = False
    recorder_closed: bool = False
    notifier_closed: bool = False
    recorder: LoopRecorder | None = None

    def _claim(self, store: StateStore, loop: Loop, created: CreateRunResult) -> StateStore:
        self.run_id = created.run_id
        state = (
            _initial_loop_state(_recovery_loop_name(loop))
            if created.storage_format == STORAGE_FORMAT_CHECKPOINTED
            else None
        )
        self.store = claimed_store(
            store, self.run_id, ClaimResult(0, created.storage_format, state, None)
        )
        self.owned = True
        if self.manager is not None:
            self.manager.register_current(self.run_id)
            self.registered = True
        return self.store

    async def _finish(
        self, recorder: LoopRecorder | None, notifier: LoopNotifier | None, result: RunResult
    ) -> None:
        self.recorder_closed = True
        await record_run_finished(recorder, self.run_id, result)
        self.notifier_closed = True
        await notify_run_finished(notifier, self.run_id, result)

    async def _failed(
        self,
        exc: Exception,
        meta: dict[str, Any],
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
    ) -> None:
        if not self.started and isinstance(exc, (RunAlreadyActiveError, IdempotencyConflictError)):
            return
        _attach_run_error_meta(exc, self.run_id, meta)
        if self.owned and self.store is not None:
            await _persist_run_failure(self.store, self.run_id, exc)
        if self.started:
            await self._fail_sinks(exc, recorder, notifier)

    async def _fail_sinks(
        self, exc: Exception, recorder: LoopRecorder | None, notifier: LoopNotifier | None
    ) -> None:
        if not self.recorder_closed:
            self.recorder_closed = True
            await record_run_failed(recorder, self.run_id, exc)
        if not self.notifier_closed:
            self.notifier_closed = True
            await notify_run_failed(notifier, self.run_id, exc)

    async def _fenced(
        self, recorder: LoopRecorder | None, notifier: LoopNotifier | None
    ) -> RunResult:
        assert self.store is not None
        result = await _build_cached_result(self.store, self.run_id)
        if self.started:
            open_recorder = None if self.recorder_closed else recorder
            open_notifier = None if self.notifier_closed else notifier
            self.recorder_closed = self.notifier_closed = True
            await _finish_fenced_lifecycle(self.run_id, result, open_recorder, open_notifier)
        return result

    async def _cancel_finish(
        self, recorder: LoopRecorder | None, notifier: LoopNotifier | None, result: RunResult
    ) -> None:
        if not self.recorder_closed:
            self.recorder_closed = True
            try:
                await record_run_finished(recorder, self.run_id, result)
            except Exception:
                logger.warning(
                    "Recorder finish failed for run %s", self.run_id, exc_info=safe_exception_info()
                )
        if not self.notifier_closed:
            self.notifier_closed = True
            await notify_run_finished(notifier, self.run_id, result)

    async def _cancelled(
        self,
        result: RunResult | None,
        meta: dict[str, Any],
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
    ) -> tuple[RunResult, bool]:
        manager = self.manager if self.registered else None
        interrupted, swallow = _take_task_interrupt(manager, self.run_id)
        cancelled = await _drain_task_cancellation(
            self.store if self.owned else None, self.run_id, interrupted
        )
        if result is not None and (
            self.store is None
            or (result.status == cancelled.status and result.meta.get("_finalize_won"))
        ):
            cancelled = result
        _attach_mcp_meta(cancelled, meta)
        if self.started:
            await self._cancel_finish(recorder, notifier, cancelled)
        return cancelled, _may_swallow_cancellation(swallow)

    def _release(self) -> None:
        if self.registered and self.manager is not None:
            self.manager.release_current(self.run_id)


async def _discover_runtime_tools(
    agent: Agent,
    recorder: LoopRecorder | None,
    extra_notifier: LoopNotifier | None,
    run_id: str,
    resolved_loop: Loop,
    mcp_meta: dict[str, Any],
) -> None:
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


def _runtime_recorder(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    tenant_id: str | None,
    seeds: dict[str, Any],
    guardrails: Any,
) -> LoopRecorder:
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.tool import get_tool_def

    definitions = [get_tool_def(fn) for fn in agent.tools]
    recorder = PersistenceRecorder(
        store,
        run_id,
        model=provider.model,
        provider_name=type(provider).__name__,
        target_lookup={td.name: td.target for td in definitions},
        seed_traces=seeds.get("seed_traces"),
        guardrail_engine=guardrails,
        **_recorder_image_kwargs(tenant_id, provider),
    )
    recorder._history_limits = agent._limits
    return recorder


async def _start_new_run(
    lifecycle: _RunLifecycle,
    agent: Agent,
    provider: LLMProvider,
    tenant_id: str | None,
    seeds: dict[str, Any],
    guardrails: Any,
    notifier: LoopNotifier | None,
    parent: DelegationContext | None,
    event_data: dict[str, Any],
    *,
    before_event: Callable[[], None] | None = None,
) -> tuple[Any, LoopNotifier | None]:
    if lifecycle.store is not None:
        lifecycle.recorder = _runtime_recorder(
            lifecycle.store, agent, provider, lifecycle.run_id, tenant_id, seeds, guardrails
        )
    if before_event is not None:
        before_event()
    await _emit_event(
        lifecycle.store, lifecycle.run_id, StoredEventType.RUN_STARTED.value, event_data
    )
    lifecycle.started = True
    await record_run_started(
        lifecycle.recorder, lifecycle.run_id, agent_name=agent.name, agent_model=provider.model
    )
    return await _start_trace(
        notifier,
        lifecycle.store,
        lifecycle.run_id,
        agent_name=agent.name,
        agent_model=provider.model,
        parent=parent,
    )


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
    on_started: Callable[[], None] = lambda: None,
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
    on_started()
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
        "error": str(error)[:ERROR_TEXT_CHARS],
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

    names = (get_tool_def(fn).name for fn in islice(agent.tools, _TOOL_NAME_SCAN_LIMIT))
    return sorted({name for name in names if len(name) <= _TOOL_NAME_CHARS and name.isidentifier()})


async def _emit_declared_tools(
    agent: Agent,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
) -> None:
    await _emit_init_governance_event(
        recorder,
        notifier,
        run_id,
        StoredEventType.TOOLS_DECLARED.value,
        {"local_tool_names": _local_tool_names(agent)},
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
    """Best-effort evidence or failure event; ownership fencing still propagates."""
    if state_store is None:
        return
    try:
        await _emit_event(state_store, run_id, event_type, data, correlation_id)
    except RunFencedError:
        raise
    except Exception:
        logger.warning(
            "Failed to record %s for run %s",
            event_type,
            run_id,
            exc_info=safe_exception_info(),
        )


# Every status a loop result can durably pause in.


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


def _terminal_stream_result(event: RunEvent) -> RunResult | None:
    if event.type in (
        RunEventType.RUN_COMPLETED,
        RunEventType.RUN_ERROR,
        RunEventType.RUN_PAUSED,
        RunEventType.RUN_CANCELLED,
    ):
        return event.run_result
    return None


async def _finish_stream_event(
    event: RunEvent,
    result: RunResult,
    *,
    state_store: StateStore | None,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    provider_kwargs: dict[str, Any] | None,
) -> RunEvent:
    from dendrux.types import RunEvent

    if state_store is not None:
        result = await _persist_loop_outcome(
            recorder=recorder,
            provider_kwargs=provider_kwargs,
            state_store=state_store,
            run_id=run_id,
            result=result,
        )
        # A pre-pause cancellation changes the terminal event delivered to the consumer.
        if result.status == RunStatus.CANCELLED:
            event = RunEvent(type=RunEventType.RUN_CANCELLED, run_result=result)
        else:
            event = RunEvent(type=event.type, run_result=result, phase=event.phase)
    await record_run_finished(recorder, run_id, result)
    await notify_run_finished(notifier, run_id, result)
    if state_store is None:
        result.meta = _public_meta(result.meta)
    return event


def _prepare_stream_result(
    result: RunResult,
    provider_kwargs: dict[str, Any] | None,
    mcp_meta: dict[str, Any],
    replay_plan: ReplayPlan | None,
    new_run: bool,
) -> None:
    if new_run:
        _ensure_pause_generation(result, provider_kwargs)
        if replay_plan is not None:
            result.replay = replay_plan.summary
    _attach_mcp_meta(result, mcp_meta)


async def _stream_loop_events(
    loop_stream: AsyncGenerator[RunEvent, None],
    *,
    state_store: StateStore | None,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    provider_kwargs: dict[str, Any] | None,
    shared: dict[str, Any],
    mcp_meta: dict[str, Any],
    replay_plan: ReplayPlan | None = None,
    new_run: bool = False,
) -> AsyncGenerator[RunEvent, None]:
    async with _closing_loop_stream(loop_stream, run_id):
        async for event in loop_stream:
            result = _terminal_stream_result(event)
            if result is None or (not new_run and not result):
                _track_partial_text(shared, event)
                yield event
                continue
            _prepare_stream_result(result, provider_kwargs, mcp_meta, replay_plan, new_run)
            yield await _finish_stream_event(
                event,
                result,
                state_store=state_store,
                recorder=recorder,
                notifier=notifier,
                run_id=run_id,
                provider_kwargs=provider_kwargs,
            )


async def _resolve_resume_stream_store(
    store: StateStore | None, resolver: Callable[[], Any] | None
) -> StateStore:
    if store is None and resolver is not None:
        store = await resolver()
    if store is None:
        raise ValueError(
            "resume_stream() requires persistence. Pass database_url or state_store to the agent."
        )
    return store


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
        "error": _safe_error(exc)[:ERROR_TEXT_CHARS],
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
    data = (meta or {}).get(ERROR_INFO_META_KEY)
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


async def _recovery_info(store: StateStore, record: Any, *, best_effort: bool = True) -> Any:
    """Project recovery strictly for reads, or best effort after a durable outcome."""
    if record is None or getattr(record, "last_reclaimed_at", None) is None:
        return None
    return await _read_recovery_info(store, record, best_effort=best_effort)


async def _read_recovery_info(store: StateStore, record: Any, *, best_effort: bool) -> Any:
    from dendrux.types import RecoveryInfo

    try:
        ids, truncated = await store.get_replayed_tool_calls(
            record.id, limit=_CONTROL_EVENT_LIMIT, tenant_id=record.tenant_id
        )
    except Exception:
        if not best_effort:
            raise
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=record.id)
        return None
    return RecoveryInfo(record.reclaim_attempts, record.last_reclaimed_at, ids, truncated)


async def _attach_persisted_pending(store: StateStore, result: RunResult, record: Any) -> RunResult:
    result.recovery = await _recovery_info(store, record)
    if result.status in _PAUSE_STATUSES and record is not None:
        await _load_persisted_pending(store, result, record)
    return result


async def _load_persisted_pending(store: StateStore, result: RunResult, record: Any) -> None:
    from dendrux.runtime._pending import _PendingReader

    result.pending = None
    result.pause_id = None
    try:
        result.pending = await _PendingReader(store).get_pending(
            result.run_id, tenant_id=getattr(record, "tenant_id", None)
        )
    except Exception:
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=result.run_id)
    result.pause_id = result.pending.pause_id if result.pending else None


async def _committed_pause_result(
    store: StateStore, result: RunResult, persisted: Callable[[RunStatus], Awaitable[RunResult]]
) -> RunResult:
    try:
        record = await store.get_run(result.run_id)
    except Exception:
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=result.run_id)
        result.pending = None
        result.pause_id = None
        return result
    if record is not None and record.status != result.status.value:
        return await persisted(RunStatus(record.status))
    return await _attach_persisted_pending(store, result, record)


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
        paused_data["message"] = pause_state.pause_message[:ERROR_TEXT_CHARS]
    if pause_state.budget_exceeded:
        paused_data["exceeded"] = pause_state.budget_exceeded
    return paused_data


def _terminal_writer(store: StateStore) -> Any:
    """The method that will write a loop's terminal status on ``store``."""
    return getattr(store, "finalize_with_event", None) or store.finalize_run


def _terminal_failure_reason(result: RunResult) -> str | None:
    """The reason a returned error ended the run, for the reasons a store may keep."""
    if result.status != RunStatus.ERROR:
        return None
    reason = result.meta.get("failure_reason")
    known = (FailureReason.GUARDRAIL_BLOCKED.value, FailureReason.PENDING_LIMIT.value)
    return reason if reason in known else None


def _terminal_failure_kwargs(state_store: StateStore, result: RunResult) -> dict[str, Any]:
    """Keep guardrail reason support optional for legacy custom stores."""
    base = unclaimed_store(state_store)
    reason = _terminal_failure_reason(result)
    if (
        base is None
        or reason is None
        or not _accepts_kwarg(_terminal_writer(base), "failure_reason")
    ):
        return {}
    return {"failure_reason": reason}


def _unstored_failure_reason(result: RunResult, stored: dict[str, Any]) -> dict[str, Any]:
    """The reason for the event, when the store's write cannot keep it on the row."""
    reason = _terminal_failure_reason(result)
    return {"failure_reason": reason} if reason and "failure_reason" not in stored else {}


async def _final_run_recovery(state_store: StateStore, run_id: str, records: Any) -> Any:
    from dendrux.runtime._capabilities import capabilities

    if not capabilities(state_store).recovery:
        return None
    try:
        record = records[-1] if records else await state_store.get_run(run_id)
    except Exception:
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=run_id)
        return None
    return await _recovery_info(state_store, record)


def _wrap_loop_result(
    run_id: str,
    result: RunResult,
    user_answer: str | None,
    pii_mapping: dict[str, Any],
    *,
    status: RunStatus,
    won: bool,
    recovery: Any = None,
) -> RunResult:
    from dendrux.guardrails._engine import deanonymize_text
    from dendrux.types import RunResult

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


async def _persisted_loop_result(
    state_store: StateStore,
    run_id: str,
    result: RunResult,
    user_answer: str | None,
    pii_mapping: dict[str, Any],
    status_if_missing: RunStatus,
) -> RunResult:
    record = await state_store.get_run(run_id)
    status = RunStatus(record.status) if record is not None else status_if_missing
    persisted = _wrap_loop_result(
        run_id, result, user_answer, pii_mapping, status=status, won=False
    )
    if record is not None:
        persisted.answer = record.answer
        persisted.output = None
        persisted.error = record.error
        persisted.error_info = _error_info_from_meta(record.meta)
    if status not in _PAUSE_STATUSES:
        persisted.meta.pop("pause_state", None)
        await _attach_undelivered_notes(state_store, run_id, persisted)
    return await _attach_persisted_pending(state_store, persisted, record)


async def _finalize_loop_cancelled(
    state_store: StateStore,
    run_id: str,
    result: RunResult,
    user_answer: str | None,
    pii_mapping: dict[str, Any],
    allowed: list[str],
    payload: dict[str, Any],
    failure_reason: str | None = None,
) -> RunResult:
    from dendrux.runtime.state import _capture_finalized_run

    with _capture_finalized_run(unclaimed_store(state_store), run_id) as records:
        won = await _transition_with_event(
            state_store,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=allowed,
            event_type=StoredEventType.RUN_CANCELLED.value,
            event_data=payload,
            answer=user_answer,
            iteration_count=result.iteration_count,
            total_usage=result.usage,
            pii_mapping=result.meta.get("pii_mapping"),
            failure_reason=failure_reason,
        )
    if not won:
        return await _persisted_loop_result(
            state_store, run_id, result, user_answer, pii_mapping, RunStatus.CANCELLED
        )
    recovery = await _final_run_recovery(state_store, run_id, records)
    cancelled = _wrap_loop_result(
        run_id,
        result,
        user_answer,
        pii_mapping,
        status=RunStatus.CANCELLED,
        won=True,
        recovery=recovery,
    )
    cancelled.meta.pop("pause_state", None)
    await _attach_undelivered_notes(state_store, run_id, cancelled)
    return cancelled


def _loop_cancel_request(result: RunResult) -> tuple[list[str], dict[str, Any], str | None] | None:
    if result.status != RunStatus.CANCELLED:
        return None
    if result.meta.get("budget_exceeded"):
        return (
            [RunStatus.RUNNING.value],
            {"reason": BUDGET_EXCEEDED, "exceeded": result.meta["budget_exceeded"]},
            BUDGET_EXCEEDED,
        )
    payload: dict[str, Any] = {"reason": CancelReason.CANCEL_REQUESTED.value}
    if result.meta.get("interrupted"):
        payload["interrupted"] = True
        payload["partial_output"] = bool(result.meta.get("partial_output"))
    return list(CANCEL_FINALIZE_STATUS_VALUES), payload, None


def _terminal_loop_event(result: RunResult) -> tuple[StopReason | None, str, dict[str, Any]]:
    stop_reason = result.stop_reason if result.status == RunStatus.SUCCESS else None
    event_type = (
        StoredEventType.RUN_ERROR.value
        if result.status == RunStatus.ERROR
        else StoredEventType.RUN_COMPLETED.value
    )
    event_data = {
        **({"error": result.error} if result.status == RunStatus.ERROR else {}),
        "status": result.status.value,
        "stop_reason": stop_reason,
        "truncated": stop_reason == "max_tokens",
    }
    return stop_reason, event_type, event_data


def _completion_evidence(
    state_store: StateStore,
    result: RunResult,
    recorder: LoopRecorder | None,
    event_type: str,
    event_data: dict[str, Any],
) -> dict[str, Any]:
    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.state import EventWrite

    if getattr(state_store, "storage_format", 1) != STORAGE_FORMAT_CHECKPOINTED:
        return {}
    if not capabilities(state_store).atomic_completion:
        return {}
    return {
        "trace": getattr(recorder, "finalization_trace", lambda: None)(),
        "event": EventWrite(
            generate_ulid(), event_type, result.iteration_count, event_data, None, None
        ),
    }


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
    from dendrux.runtime.state import _capture_finalized_run

    pii_mapping = result.meta.get("pii_mapping") or {}
    user_answer, unmapped_placeholders = deanonymize_text(result.answer, pii_mapping)
    _wrap = partial(_wrap_loop_result, run_id, result, user_answer, pii_mapping)
    _persisted = partial(
        _persisted_loop_result, state_store, run_id, result, user_answer, pii_mapping
    )
    _finalize_cancelled = partial(
        _finalize_loop_cancelled, state_store, run_id, result, user_answer, pii_mapping
    )

    cancel_request = _loop_cancel_request(result)
    if cancel_request is not None:
        return await _finalize_cancelled(*cancel_request)

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
        if (
            pause_state.pause_reason == PauseReason.REQUESTED.value
            and pause_state.pause_message is None
        ):
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
                [RunStatus.RUNNING.value], {"reason": CancelReason.CANCEL_REQUESTED.value}
            )
        result.pause_id = pause_state.pause_id
        # Return the persisted state: a finalizer may already have acted on
        # the durable pause (e.g. cancel_run right after it committed).
        return await _committed_pause_result(
            state_store, _wrap(status=result.status, won=True), _persisted
        )

    # Terminal: success / max_iterations / error. Cooperative cancellation:
    # a cancel requested after the last checkpoint does not override a run
    # that finished — its work and side effects happened.
    _run_pii = result.meta.get("pii_mapping")
    failure_kwargs = _terminal_failure_kwargs(state_store, result)
    stop_reason, event_type, event_data = _terminal_loop_event(result)
    event_data.update(_unstored_failure_reason(result, failure_kwargs))
    atomic_final = getattr(state_store, "finalize_with_event", None) is not None
    if unmapped_placeholders:
        await _emit_event_safe(
            state_store,
            run_id,
            GovernanceEventType.GUARDRAIL_UNMAPPED_PLACEHOLDER.value,
            {"placeholders": unmapped_placeholders},
        )
    with _capture_finalized_run(unclaimed_store(state_store), run_id) as records:
        if atomic_final:
            finalize_won = (
                await state_store.finalize_with_event(
                    run_id,
                    status=result.status.value,
                    allowed_current_statuses=[RunStatus.RUNNING.value],
                    answer=result.answer,
                    iteration_count=result.iteration_count,
                    total_usage=result.usage,
                    pii_mapping=_run_pii,
                    error=result.error,
                    event_type=event_type,
                    event_data=event_data,
                    **_completion_evidence(state_store, result, recorder, event_type, event_data),
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
                expected_current_status=RunStatus.RUNNING.value,
                pii_mapping=_run_pii,
                error=result.error,
                **failure_kwargs,
            )
    if not finalize_won:
        return await _persisted(result.status)
    recovery = await _final_run_recovery(state_store, run_id, records)
    if stop_reason is not None:
        await _record_stop_reason(state_store, run_id, stop_reason)
    if not atomic_final:
        await _emit_event(state_store, run_id, event_type, event_data)
    finished = _wrap(status=result.status, won=True, recovery=recovery)
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
    await _emit_event(state_store, run_id, StoredEventType.RUN_PAUSED.value, event_data)
    # The read and the pause are separate here: a cancel committed in between
    # left a paused row with cancel_requested set. Finish it now; the caller
    # reloads and returns the persisted winner.
    if await state_store.is_cancel_requested(run_id):
        await _transition_with_event(
            state_store,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=[status],
            event_type=StoredEventType.RUN_CANCELLED.value,
            event_data={"reason": CancelReason.CANCEL_REQUESTED.value},
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
        effective = _DEFAULT_RECOVERY_ITERATIONS

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


def _require_idempotency_store(store: StateStore | None, key: str | None) -> None:
    if key is not None and store is None:
        raise ValueError(
            "idempotency_key requires persistence (database_url or state_store). "
            "Idempotency cannot work without durable state."
        )


def _new_run_loop(agent: Agent, loop: Loop | None) -> Loop:
    return loop or agent.loop or ReActLoop()


def _new_run_input(
    user_input: UserContent, history: list[ChatMessage] | None, loop: Loop
) -> tuple[str, tuple[Any, ...], bool, list[Message]]:
    text, items = split_user_content(user_input)
    validate_chat_history(history)
    image_input = bool(items) or history_has_images(history)
    if image_input:
        _require_image_loop(loop)
    return text, items, image_input, [] if image_input else normalize_chat_history(history)


def _new_run_history(
    replay: ReplayRequest | None,
    history: list[Message],
    context: list[ContextBlock] | None,
    user_input: str,
    session: SkillSession,
    parts: tuple[ContentPart, ...] | None,
) -> list[Message] | None:
    if replay is not None:
        return None
    return fold_context(history, context, user_input, session=session, user_parts=parts)


async def _record_seeded_user(
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    user_input: str,
    history: list[Message] | None,
    context: list[ContextBlock] | None,
    session: SkillSession,
) -> None:
    if history is not None:
        message = user_trace_message(
            user_input, history, render_band(context, "stable"), context, session=session
        )
        await _record_user_trace(recorder, notifier, run_id, message)


async def _new_run_image_view(
    runtime: ImageRuntime | None,
    store: StateStore | None,
    run_id: str,
    history: list[Message] | None,
    replay_runs: list[Any],
    replay_plan: ReplayPlan | None,
    parts: tuple[ContentPart, ...] | None,
) -> ImageViewState | None:
    return await _start_image_view(
        runtime,
        store,
        run_id,
        seeded=(history or [])[:-1],
        sources=_replay_sources(replay_runs),
        single_full=_single_full_source(replay_plan),
        input_images=parts is not None,
    )


def _new_run_metadata(
    agent: Agent,
    loop: Loop,
    session: SkillSession,
    metadata: dict[str, Any] | None,
    depth: int | None,
    *,
    user_input: str,
    idempotency_key: str | None = None,
    output_type: type | None = None,
    history: Any = None,
    context: Any = None,
    replay: ReplayRequest | None = None,
    parts: tuple[ContentPart, ...] | None = None,
    retry_of: str | None = None,
    streaming: bool = False,
) -> tuple[dict[str, Any], str | None]:
    result = dict(metadata) if metadata else {}
    result[LOOP_META_KEY] = type(loop).__name__
    result[GOVERNANCE_KEY] = governance_snapshot(agent)
    if session.configured:
        result[META_KEY] = session.manifest()
    if depth is not None:
        result[MAX_DELEGATION_DEPTH_META_KEY] = depth
    output_name = None
    if output_type is not None:
        output_name = f"{output_type.__module__}.{output_type.__qualname__}"
        result[OUTPUT_TYPE_META_KEY] = output_name
    if retry_of is not None:
        result[RETRY_OF_META_KEY] = retry_of
    fingerprint = None
    if idempotency_key is not None:
        fingerprint = compute_idempotency_fingerprint(
            agent.name,
            user_input,
            # Preserve run's truthiness check and stream's presence check.
            output_type_name=output_name if streaming or output_type else None,
            history=history or None,
            context=context or None,
            replay=replay.fingerprint() if replay else None,
            skills=session.fingerprint(),
            user_input_parts=parts,
        )
    return result, fingerprint


async def _load_new_run_replay(
    store: StateStore,
    request: ReplayRequest,
    tenant_id: str | None,
    loop: Loop,
    *,
    limits: Limits,
    method: str = "run",
) -> list[Any]:
    runs = await _load_replay_runs(store, request, tenant_id=tenant_id, limits=limits)
    if _replay_has_images(runs):
        _require_image_loop(loop, method)
    return runs


async def _entry_replay(
    store: StateStore | None,
    tenant_id: str | None,
    runs: list[str] | None,
    last: ReplayLast,
    reasoning: bool,
    loop: Loop,
    context: Any,
    limits: Limits,
) -> tuple[ReplayRequest | None, list[Any]]:
    request = validate_replay_args(runs, last, reasoning, loop=loop, context=context)
    validate_context(context)
    if request is None:
        return None, []
    if store is None:
        raise ValueError(
            "runs= requires persistence (database_url or state_store): "
            "replay reads the earlier runs' traces."
        )
    return request, await _load_new_run_replay(store, request, tenant_id, loop, limits=limits)


async def _entry_media(
    agent: Agent,
    provider: LLMProvider,
    resolved_loop: Loop,
    state_store: StateStore | None,
    tenant_id: str | None,
    run_id: str,
    image_input: bool,
    user_input: str,
    input_items: Any,
    history: Any,
    normalized_history: Any,
    provider_kwargs: dict[str, Any],
    idempotency_key: str | None,
    output_type: type | None,
    context: Any,
    replay_req: ReplayRequest | None,
    skill_session: SkillSession,
) -> tuple[ImageRuntime | None, tuple[ContentPart, ...] | None, Any, RunResult | None]:
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
                return (
                    image_runtime,
                    user_parts,
                    normalized_history,
                    await _build_cached_result(state_store, hit.run_id, output_type),
                )
            raise RunAlreadyActiveError(hit.run_id, hit.status) from None
        image_runtime, user_parts, normalized_history = entry
    elif _accepts_kwarg(resolved_loop.run, "image_runtime"):
        image_runtime = _image_runtime(agent, state_store, tenant_id, run_id)

    return image_runtime, user_parts, normalized_history, None


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
    _require_idempotency_store(state_store, idempotency_key)
    await ensure_agent_limits_fit(agent, state_store)
    skill_session = SkillSession.prepare(
        agent, loop=_new_run_loop(agent, loop), explicit=activate_skills
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
        resolved_loop = _new_run_loop(agent, loop)
        _validate_loop_skill_compat(agent, resolved_loop)

        # Validate input + chat history before any side effects. Raises
        # TypeError / ValueError on bad input so no DB row is created.
        user_input, input_items, image_input, normalized_history = _new_run_input(
            user_input, history, resolved_loop
        )
        replay_req, replay_runs = await _entry_replay(
            state_store,
            tenant_id,
            runs,
            replay_last,
            replay_reasoning,
            resolved_loop,
            context,
            agent._limits,
        )
        replay_plan: ReplayPlan | None = None

        provider_kwargs = dict(kwargs)

        run_id = generate_ulid()

        # Images: ingest before the first durable write (§5.2). Typed errors
        # raise here, before any run row or claim exists.
        image_runtime, user_parts, normalized_history, cached = await _entry_media(
            agent,
            provider,
            resolved_loop,
            state_store,
            tenant_id,
            run_id,
            image_input,
            user_input,
            input_items,
            history,
            normalized_history,
            provider_kwargs,
            idempotency_key,
            output_type,
            context,
            replay_req,
            skill_session,
        )
        if cached is not None:
            return cached

        # When chat history is provided, the loop expects initial_history to
        # be the COMPLETE message list (loops use initial_history as-is and
        # do not separately append user_input — same contract as resume).
        # The runner will record the new user_input itself (after the recorder
        # is constructed) so seeded prior turns stay out of react_traces.
        # With runs=, the replayed prefix is folded in after expansion.
        seeded_history = _new_run_history(
            replay_req, normalized_history, context, user_input, skill_session, user_parts
        )

        # --- Delegation context ---
        parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)

        # Resolve system prompt BEFORE create_run() so skill loading
        # failures don't leave a DB row stuck in running.
        _system_prompt = agent.get_system_prompt(loop=resolved_loop)

        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        result: RunResult | None = None
        lifecycle = _RunLifecycle(run_id, state_store, getattr(agent, "_task_manager", None))

        seed_kwargs: dict[str, Any] = {}
        run_guardrails = None
        try:
            try:
                if state_store is not None:
                    # Create the run record before the loop starts
                    # Merge loop type + depth limit into developer metadata
                    run_meta, idem_fingerprint = _new_run_metadata(
                        agent,
                        resolved_loop,
                        skill_session,
                        metadata,
                        effective_max_depth,
                        user_input=user_input,
                        idempotency_key=idempotency_key,
                        output_type=output_type,
                        history=normalized_history,
                        context=context,
                        replay=replay_req,
                        parts=user_parts,
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

                    run_id = create_result.run_id
                    state_store = lifecycle._claim(state_store, resolved_loop, create_result)

                trace_link, extra_notifier = await _start_new_run(
                    lifecycle,
                    agent,
                    provider,
                    tenant_id,
                    seed_kwargs,
                    run_guardrails,
                    extra_notifier,
                    parent_ctx,
                    {"agent_name": agent.name, "system_prompt": _system_prompt},
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
                await _discover_runtime_tools(
                    agent, lifecycle.recorder, extra_notifier, run_id, resolved_loop, mcp_meta
                )

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
                image_view = await _new_run_image_view(
                    image_runtime,
                    state_store,
                    run_id,
                    seeded_history,
                    replay_runs,
                    replay_plan,
                    user_parts,
                )

                # When chat history was provided, the loop receives a complete
                # message list as initial_history and skips its own user-message
                # recording. The runner records the new user_input itself so it
                # still lands in react_traces (seeded prior turns are NOT recorded
                # — they live in the dev's chat DB).
                await _record_seeded_user(
                    lifecycle.recorder,
                    extra_notifier,
                    run_id,
                    user_input,
                    seeded_history,
                    context,
                    skill_session,
                )

                result = await resolved_loop.run(
                    agent=agent,
                    provider=provider,
                    strategy=resolved_strategy,
                    user_input=user_input,
                    run_id=run_id,
                    recorder=lifecycle.recorder,
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
                        recorder=lifecycle.recorder,
                        provider_kwargs=provider_kwargs,
                        state_store=state_store,
                        run_id=run_id,
                        result=result,
                    )

                await lifecycle._finish(lifecycle.recorder, extra_notifier, result)

                result.meta = _public_meta(result.meta)

                return result

            except RunFencedError:
                return await lifecycle._fenced(lifecycle.recorder, extra_notifier)

            except Exception as exc:
                await lifecycle._failed(exc, mcp_meta, lifecycle.recorder, extra_notifier)
                raise

        except asyncio.CancelledError:
            cancelled_result, swallow = await lifecycle._cancelled(
                result, mcp_meta, lifecycle.recorder, extra_notifier
            )
            if swallow:
                return cancelled_result
            raise

        finally:
            _release_memory_edges(image_runtime, state_store, run_id)
            lifecycle._release()
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
    from dendrux.runtime.durability import _ownership_handoff
    from dendrux.runtime.state import _creation_loop_scope

    async def cancel_created(created: CreateRunResult) -> None:
        if created.outcome == "created":
            await _persist_task_cancellation(
                claimed_store(store, created.run_id, 0), created.run_id
            )

    native_store = unclaimed_store(store)
    with _creation_loop_scope(native_store, run_id, _recovery_loop_name(loop)):
        return await _ownership_handoff(
            store,
            lambda: store.create_run(run_id, agent_name, **kwargs),
            cancel_created,
            run_id=run_id,
        )


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
    from dendrux.types import _validate_provider_kwargs

    native_store = unclaimed_store(store)
    format_reader = getattr(native_store, "_creation_storage_format", None)
    if callable(format_reader) and await format_reader() == 2:
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


async def _read_cancel_winner(
    store: StateStore, run_id: str, *, safe: bool = False
) -> RunRecord | None:
    try:
        return await store.get_run(run_id)
    except Exception:
        if not safe:
            raise
        return None


async def _persist_run_cancellation(
    store: StateStore,
    run_id: str,
    payload: dict[str, Any],
    *,
    on_won: Callable[[], None],
    answer: str | None | _OmittedType = _OMITTED,
    emit_event: bool = True,
    read_on_loss: Callable[[], bool] | None = None,
    safe_read: bool = False,
) -> tuple[bool, RunRecord | None]:
    kwargs: dict[str, Any] = {} if isinstance(answer, _OmittedType) else {"answer": answer}
    won, needs_event = await _finalize_running_event(
        store,
        run_id,
        RunStatus.CANCELLED.value,
        StoredEventType.RUN_CANCELLED.value,
        payload,
        emit_event=emit_event,
        **kwargs,
    )
    if won:
        # Remember the committed CAS even if the separate event write raises.
        on_won()
        if needs_event:
            await _emit_event(store, run_id, StoredEventType.RUN_CANCELLED.value, payload)
        return True, None
    record = None
    if read_on_loss is None or read_on_loss():
        record = await _read_cancel_winner(store, run_id, safe=safe_read)
    return False, record


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
        meta={**interrupt_meta, CANCEL_PERSISTED_META_KEY: False},
    )
    if state_store is None:
        return result
    try:
        won, record = await _persist_run_cancellation(
            state_store,
            run_id,
            {"reason": CancelReason.CANCEL_REQUESTED.value, **interrupt_meta}
            if interrupted
            else {"reason": CancelReason.TASK_CANCELLED.value},
            on_won=lambda: result.meta.update({CANCEL_PERSISTED_META_KEY: True}),
        )
        if won:
            await _attach_undelivered_notes(state_store, run_id, result)
        elif record is not None:
            result = await _build_cached_result(state_store, run_id)
            if result.status == RunStatus.CANCELLED:
                result.meta[CANCEL_PERSISTED_META_KEY] = True
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
    rule = _claim_failure_mapping(run, after_preflight=True)
    raise ValueError(
        rule.resume_message.format(
            run_id=run_id, expected=expected_status, reason=getattr(run, "failure_reason", None)
        )
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


async def _check_retry_source(
    state_store: StateStore, source_run: RunRecord, original_run_id: str
) -> None:

    if source_run.status not in TERMINAL_STATUSES:
        raise ValueError(
            f"Run '{original_run_id}' is in status '{source_run.status}' — "
            f"only terminal runs (error, cancelled, max_iterations, success) can be retried."
        )

    # 2. Validate source run was not SingleCall (no iterative history to seed)
    from dendrux.runtime._capabilities import capabilities

    source_loop = (source_run.meta or {}).get(LOOP_META_KEY, "")
    if (
        source_run.storage_format == STORAGE_FORMAT_CHECKPOINTED
        and capabilities(state_store).checkpointing
    ):
        source_state = await state_store.get_loop_state(original_run_id)
        source_loop = (source_state or {}).get("loop")
    if source_loop in ("SingleCall", "single_call"):
        raise ValueError(
            "SingleCall runs cannot be retried with context — "
            "they have no iterative history to seed. Call agent.run() instead."
        )


async def _retry_history(
    state_store: StateStore,
    source_run: RunRecord,
    agent: Agent,
    provider: LLMProvider,
    kwargs: dict[str, Any],
) -> tuple[Any, list[Message]]:
    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.replay import conversation_records

    traces = (
        await conversation_records(state_store, source_run.id, limits=agent._limits)
        if source_run.storage_format == STORAGE_FORMAT_CHECKPOINTED
        and capabilities(state_store).checkpointing
        else await state_store.get_traces(source_run.id)
    )
    retry_model = (kwargs or {}).get("model") or provider.model
    initial_history = _pair_tool_calls(rebuild_messages(traces, reasoning_for=retry_model))
    if source_run.storage_format == STORAGE_FORMAT_CHECKPOINTED:
        initial_history = [m for m in initial_history if m.role != Role.SYSTEM]

    if not initial_history:
        raise ValueError(
            f"Run '{source_run.id}' has no persisted traces. Cannot retry without prior context."
        )

    return traces, initial_history


def _retry_source_input(source_run: RunRecord, traces: Any) -> tuple[Any, Any, Any, Any]:
    users = [t for t in traces if t.role == "user"]
    raw_user = next((t for t in users if not (t.meta or {}).get(SEED_PREFIX_META_KEY)), users[0])
    user_input = (source_run.input_data or {}).get("input", raw_user.content)
    source_parts = _stored_parts((source_run.input_data or {}).get(PARTS_KEY))

    return user_input, source_parts, raw_user, users


def _refresh_retry_user(
    initial_history: list[Message],
    users: Any,
    raw_user: Any,
    user_input: str,
    source_parts: tuple[ContentPart, ...] | None,
    skill_session: SkillSession,
) -> tuple[list[ContextBlock], Message]:
    own_ordinal = users.index(raw_user)
    first_user = [i for i, m in enumerate(initial_history) if m.role == Role.USER][own_ordinal]
    retry_context = [
        ContextBlock(**b) for b in (raw_user.meta or {}).get(SKILLS_CONTEXT_META_KEY, [])
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

    return retry_context, retry_user


async def _retry_replay(
    source_run: RunRecord,
    resolved_loop: Loop,
    state_store: StateStore,
    tenant_id: str | None,
    limits: Limits,
) -> tuple[ReplayRequest | None, list[Any]]:
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
            replay_runs = await _load_replay_runs(
                state_store, replay_req, tenant_id=tenant_id, limits=limits
            )

    return replay_req, replay_runs


def _retry_media(
    agent: Agent,
    store: StateStore,
    loop: Loop,
    tenant: str | None,
    run_id: str,
    parts: tuple[ContentPart, ...] | None,
    history: list[Message],
    replay: list[Any],
) -> ImageRuntime | None:
    if parts is not None or has_images(history) or _replay_has_images(replay):
        _require_image_loop(loop)
    elif not _accepts_kwarg(loop.run, "image_runtime"):
        return None
    return _image_runtime(agent, store, tenant, run_id)


async def _retry_edges(
    runtime: ImageRuntime | None, parts: tuple[ContentPart, ...] | None
) -> dict[str, Any]:
    if parts is None or runtime is None:
        return {}
    live = await _live_edges(runtime, _input_edge_kwargs(runtime, parts).get("blob_refs", []))
    return {"blob_refs": live, "blob_refs_scope": runtime.tenant_scope} if live else {}


async def _retry_source(
    store: StateStore, run_id: str, tenant_id: str | None
) -> tuple[RunRecord, Any]:
    source = await store.get_run(run_id)
    if source is None:
        raise RunNotFoundError(run_id)
    if source.tenant_id != tenant_id:
        raise RunNotFoundError(run_id)
    return source, (source.meta or {}).get(META_KEY, {}).get("explicit", [])


async def _start_retry_history(
    agent: Agent,
    provider: LLMProvider,
    loop: Loop,
    provider_kwargs: dict[str, Any],
    store: StateStore,
    run_id: str,
    original_run_id: str,
    source: RunRecord,
    replay_req: ReplayRequest | None,
    replay_runs: list[Any],
    history: list[Message],
    image_runtime: ImageRuntime | None,
    explicit: Any,
) -> tuple[ReplayPlan | None, list[Message], ImageViewState | None]:
    from dendrux.runtime._capabilities import capabilities

    current = current_prefix(agent, provider, loop, provider_kwargs)
    replay_plan: ReplayPlan | None = None
    if replay_req is not None:
        replay_plan = expand_replay(
            replay_runs, replay_req, current=current, guardrails=bool(agent.guardrails)
        )
        if (
            source.storage_format != STORAGE_FORMAT_CHECKPOINTED
            or not capabilities(store).checkpointing
        ):
            history = [*replay_plan.messages, *history]
    seeded_slots = image_slot_count(history)
    await record_run_prefix(
        store,
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
        store,
        run_id,
        seeded=history,
        sources=[
            *_replay_sources(replay_runs),
            (original_run_id, source.tenant_id, source.meta),
        ],
        single_full=replay_plan is None,
        input_images=False,
    )
    return replay_plan, history, image_view


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
    await ensure_agent_limits_fit(agent, state_store)

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
    source_run, explicit = await _retry_source(state_store, original_run_id, tenant_id)
    skill_session = SkillSession.prepare(agent, loop=_new_run_loop(agent, loop), explicit=explicit)
    with skill_session.bind():
        await _check_retry_source(state_store, source_run, original_run_id)

        resolved_loop = _new_run_loop(agent, loop)
        _validate_loop_skill_compat(agent, resolved_loop)

        # 3. Read traces and reconstruct approximate history. Reasoning replay
        #    blocks only seed a retry on the model that made them.
        traces, initial_history = await _retry_history(
            state_store, source_run, agent, provider, kwargs
        )

        # The source's own input follows its persisted seed prefix.
        user_input, source_parts, raw_user, users = _retry_source_input(source_run, traces)
        if explicit:
            retry_context, retry_user = _refresh_retry_user(
                initial_history, users, raw_user, user_input, source_parts, skill_session
            )

        # 4a. A source run that used runs= re-expands the same runs, same knobs.
        replay_req, replay_runs = await _retry_replay(
            source_run, resolved_loop, state_store, tenant_id, agent._limits
        )

        # 5. Create the retry run
        resolved_strategy = strategy or NativeToolCalling()
        provider_kwargs = dict(kwargs)

        run_id = generate_ulid()

        parent_ctx = get_delegation_context()
        parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)
        effective_max_depth = _resolve_max_delegation_depth(agent.max_delegation_depth, parent_ctx)

        _system_prompt = agent.get_system_prompt(loop=resolved_loop)

        run_meta, _ = _new_run_metadata(
            agent,
            resolved_loop,
            skill_session,
            metadata,
            effective_max_depth,
            retry_of=original_run_id,
            user_input=user_input,
        )

        image_runtime = _retry_media(
            agent,
            state_store,
            resolved_loop,
            tenant_id,
            run_id,
            source_parts,
            initial_history,
            replay_runs,
        )
        input_edges = await _retry_edges(image_runtime, source_parts)
        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        result: RunResult | None = None
        lifecycle = _RunLifecycle(run_id, state_store, getattr(agent, "_task_manager", None))

        seed_kwargs: dict[str, Any] = {}
        run_guardrails = None
        try:
            try:
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
                state_store = lifecycle._claim(state_store, resolved_loop, create_result)

                trace_link, extra_notifier = await _start_new_run(
                    lifecycle,
                    agent,
                    provider,
                    tenant_id,
                    seed_kwargs,
                    run_guardrails,
                    extra_notifier,
                    parent_ctx,
                    {
                        "agent_name": agent.name,
                        "system_prompt": _system_prompt,
                        "retry_of": original_run_id,
                    },
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

                await _discover_runtime_tools(
                    agent, lifecycle.recorder, extra_notifier, run_id, resolved_loop, mcp_meta
                )

                replay_plan, initial_history, image_view = await _start_retry_history(
                    agent,
                    provider,
                    resolved_loop,
                    provider_kwargs,
                    state_store,
                    run_id,
                    original_run_id,
                    source_run,
                    replay_req,
                    replay_runs,
                    initial_history,
                    image_runtime,
                    explicit,
                )

                if explicit:
                    trace_user = user_trace_message(
                        user_input,
                        [retry_user],
                        render_band(retry_context, "stable"),
                        retry_context,
                        session=skill_session,
                    )
                    await _record_user_trace(lifecycle.recorder, extra_notifier, run_id, trace_user)

                result = await resolved_loop.run(
                    agent=agent,
                    provider=provider,
                    strategy=resolved_strategy,
                    user_input=user_input,
                    run_id=run_id,
                    recorder=lifecycle.recorder,
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
                    recorder=lifecycle.recorder,
                    provider_kwargs=provider_kwargs,
                    state_store=state_store,
                    run_id=run_id,
                    result=result,
                )

                await lifecycle._finish(lifecycle.recorder, extra_notifier, result)

                return result

            except RunFencedError:
                return await lifecycle._fenced(lifecycle.recorder, extra_notifier)

            except Exception as exc:
                await lifecycle._failed(exc, mcp_meta, lifecycle.recorder, extra_notifier)
                raise

        except asyncio.CancelledError:
            cancelled_result, swallow = await lifecycle._cancelled(
                result, mcp_meta, lifecycle.recorder, extra_notifier
            )
            if swallow:
                return cancelled_result
            raise

        finally:
            _release_memory_edges(image_runtime, state_store, run_id)
            lifecycle._release()
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


def _cached_stream_event(cached: RunResult, run_id: str) -> RunEvent:
    from dendrux.types import RunEvent

    event_type = {
        RunStatus.ERROR: RunEventType.RUN_ERROR,
        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
    }.get(cached.status, RunEventType.RUN_COMPLETED)
    return RunEvent(
        type=event_type,
        run_id=run_id,
        run_result=cached,
        error=cached.error if event_type == RunEventType.RUN_ERROR else None,
        phase="final_answer" if cached.status == RunStatus.SUCCESS else None,
    )


def _stream_entry_checks(
    agent: Agent, tenant_id: str | None, shared: dict[str, Any], **checks: Any
) -> None:
    try:
        entry_tenant = resolve_child_tenant(get_delegation_context(), tenant_id)
    except ValueError:
        return
    shared["existing_run"] = True
    runner_entry_checks(agent, "stream", tenant_id=entry_tenant, **checks)
    shared["existing_run"] = False


async def _resolve_stream_store(
    store: StateStore | None,
    resolver: Callable[[], Any] | None,
    agent: Agent,
    tenant_id: str | None | _OmittedType,
) -> StateStore | None:
    if store is None and resolver is not None:
        store = await resolver()
        resolve_runner_tenant(agent, "stream", tenant_id, state_store=store)
    return store


async def _stream_replay_runs(
    request: ReplayRequest | None,
    store: StateStore | None,
    tenant_id: str | None,
    loop: Loop,
    shared: dict[str, Any],
    limits: Limits,
) -> list[Any]:
    if request is None:
        return []
    shared["existing_run"] = True
    if store is None:
        raise ValueError(
            "runs= requires persistence (database_url or state_store): "
            "replay reads the earlier runs' traces."
        )
    runs = await _load_new_run_replay(
        store, request, tenant_id, loop, limits=limits, method="run_stream"
    )
    shared["existing_run"] = False
    return runs


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
    resolved_loop = _new_run_loop(agent, loop)
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
    provider_kwargs = dict(kwargs)

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
        startup = _RunLifecycle(run_id, store, None)
        seed_kwargs: dict[str, Any] = {}
        run_guardrails = None
        seeded_history = initial_seeded
        replay_plan: ReplayPlan | None = None
        image_runtime: ImageRuntime | None = None
        user_parts: tuple[ContentPart, ...] | None = None
        history_messages = normalized_history

        # Entry checks after tenant inheritance, before any I/O. Like image
        # setup they raise (no RUN_ERROR event, no run row, no hooks). A child
        # whose explicit tenant differs from its parent's still fails later, as
        # RUN_ERROR, exactly as before.
        _stream_entry_checks(
            agent,
            tenant_id,
            _shared,
            metadata=metadata,
            user_input=entry_user_input,
            history=history,
            context=context,
            provider_kwargs=provider_kwargs,
            idempotency_key=idempotency_key,
            provider=provider,
        )

        if image_input:
            # Image setup raises (no RUN_ERROR event, no run row, no hooks).
            _shared["existing_run"] = True
            store = await _resolve_stream_store(
                store, state_store_resolver, agent, requested_tenant_id
            )
            await ensure_agent_limits_fit(agent, store)
            _shared["state_store"] = store
            image_runtime, user_parts, history_messages, cached = await _entry_media(
                agent,
                provider,
                resolved_loop,
                store,
                resolve_child_tenant(get_delegation_context(), tenant_id),
                run_id,
                True,
                user_input,
                input_items,
                history,
                history_messages,
                provider_kwargs,
                idempotency_key,
                output_type,
                context,
                replay_req,
                skill_session,
            )
            if cached is not None:
                run_id = cached.run_id
                stream.run_id = run_id
                yield _cached_stream_event(cached, run_id)
                return
            with skill_session.bind():
                seeded_history = _new_run_history(
                    replay_req, history_messages, context, user_input, skill_session, user_parts
                )
            _shared["existing_run"] = False

        try:
            # 1. Resolve state store lazily (may create DB engine)
            store = await _resolve_stream_store(
                store, state_store_resolver, agent, requested_tenant_id
            )
            await ensure_agent_limits_fit(agent, store)
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

            _require_idempotency_store(store, idempotency_key)

            replay_runs = await _stream_replay_runs(
                replay_req, store, eff_tenant, resolved_loop, _shared, agent._limits
            )

            if store is not None:
                run_meta, fingerprint = _new_run_metadata(
                    agent,
                    resolved_loop,
                    skill_session,
                    metadata,
                    eff_max_depth,
                    user_input=user_input,
                    idempotency_key=idempotency_key,
                    output_type=output_type,
                    history=history_messages,
                    context=context,
                    replay=replay_req,
                    parts=user_parts,
                    streaming=True,
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
                    yield _cached_stream_event(cached, run_id)
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
                        if create_result.storage_format == STORAGE_FORMAT_CHECKPOINTED
                        else None,
                        None,
                    ),
                )
                _shared["state_store"] = store
                _shared["claimed"] = True

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
            startup.run_id = run_id
            startup.store = store
            trace_link, extra_notifier = await _start_new_run(
                startup,
                agent,
                provider,
                eff_tenant,
                seed_kwargs,
                run_guardrails,
                extra_notifier,
                parent_ctx,
                {"agent_name": agent.name, "system_prompt": _system_prompt},
                before_event=lambda: stream_context.set_delegation(this_ctx),
            )
            this_ctx = replace(this_ctx, trace_link=trace_link)
            stream_context.set_delegation(this_ctx)
            # Lifecycle pair is now open. Stash recorder/notifier so the
            # cleanup callback can close it with a CANCELLED finish if the
            # consumer abandons the stream before a terminal event.
            _shared["recorder"] = startup.recorder
            _shared["notifier"] = extra_notifier

            yield RunEvent(type=RunEventType.RUN_STARTED, run_id=run_id)

            await _discover_runtime_tools(
                agent, startup.recorder, extra_notifier, run_id, resolved_loop, mcp_meta
            )

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
            image_view = await _new_run_image_view(
                image_runtime, store, run_id, seeded_history, replay_runs, replay_plan, user_parts
            )

            # When chat history was provided, the loop receives a complete
            # message list as initial_history and skips its own user-message
            # recording. Record the new user_input here so it still lands in
            # react_traces (seeded prior turns are NOT recorded).
            await _record_seeded_user(
                startup.recorder,
                extra_notifier,
                run_id,
                user_input,
                seeded_history,
                context,
                skill_session,
            )

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
                recorder=startup.recorder,
                notifier=extra_notifier,
                initial_history=seeded_history,
                provider_kwargs=provider_kwargs or None,
                output_type=output_type,
                state_store=store,
                interrupt=interrupt,
                **_image_loop_kwargs(resolved_loop.run_stream, image_runtime, image_view),
            )
            async with aclosing(
                _stream_loop_events(
                    loop_stream,
                    state_store=store,
                    recorder=startup.recorder,
                    notifier=extra_notifier,
                    run_id=run_id,
                    provider_kwargs=provider_kwargs,
                    shared=_shared,
                    mcp_meta=mcp_meta,
                    replay_plan=replay_plan,
                    new_run=True,
                )
            ) as events:
                async for event in events:
                    yield event

        except RunFencedError:
            assert store is not None
            result = await _build_cached_result(store, run_id)
            await _finish_fenced_lifecycle(run_id, result, startup.recorder, extra_notifier)
            yield _fenced_stream_event(run_id, result)

        except Exception as exc:
            # Persist error, yield RUN_ERROR, return cleanly. No re-raise.
            # Covers both setup failures and loop execution errors.
            await _stream_run_failure(store, _shared, startup.recorder, extra_notifier, run_id, exc)

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
        persisted_row: RunRecord | None | _OmittedType = _OMITTED

        def mark_cancelled() -> None:
            nonlocal cancel_won
            cancel_won = True

        partial_answer = _partial_answer(_shared)
        if store is not None:
            try:
                cancel_won, persisted_row = await _persist_run_cancellation(
                    store,
                    run_id,
                    {
                        "reason": CancelReason.STREAM_CLOSED.value,
                        "partial_output": partial_answer is not None,
                    },
                    on_won=mark_cancelled,
                    answer=partial_answer,
                    emit_event=True,
                    read_on_loss=lambda: "notifier" in _shared,
                    safe_read=True,
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
                    run_record = (
                        await store.get_run(run_id)
                        if isinstance(persisted_row, _OmittedType)
                        else persisted_row
                    )
                    if run_record is not None:
                        terminal_status = RunStatus(run_record.status)
                        cancel_persisted = terminal_status == RunStatus.CANCELLED
                except Exception:
                    pass  # CANCELLED is a safe fallback
            cancelled_result = RunResult(
                run_id=run_id,
                status=terminal_status,
                meta={**deepcopy(mcp_meta), CANCEL_PERSISTED_META_KEY: cancel_persisted},
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
        cleanup=lambda: _shielded_cleanup(_cleanup, run_id=run_id),
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
        "recorder_started",
        "notifier_started",
        "notifier_warnings",
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
        self.recorder_started = False
        self.notifier_started = False
        self.notifier_warnings: list[str] = []

    def _mark_notifier_started(self) -> None:
        self.notifier_started = True


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
    from dendrux.types import InputValidationError

    pause = PauseState.from_dict(raw)
    if pause.schema_version != 2:
        return pause
    record = reads.record if reads is not None else await store.get_run(run_id)
    if record is None or record.storage_format != STORAGE_FORMAT_CHECKPOINTED:
        _codes.log_code(logger, _codes.PAUSE_FORMAT_UNSUPPORTED, run_id=run_id)
        raise PersistenceError(code="pause_format_unsupported")
    checkpoint = (
        getattr(store, "loop_state", None)
        if getattr(store, "storage_format", None) == STORAGE_FORMAT_CHECKPOINTED
        else None
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
        raise InputValidationError("history", FailureReason.HISTORY_TOO_LARGE.value) from exc
    # Strategy owns the system prompt; seeds retain it for audit/reconstruction.
    pause.history = [m for m in messages.messages if m.role != Role.SYSTEM]
    if reads is not None:
        reads.history_records = messages.records
        reads.checkpoint = checkpoint
    await _overlay_pause_checkpoint(pause, store, run_id, agent, checkpoint, messages.steps)
    return pause


async def _overlay_pause_checkpoint(
    pause: PauseState,
    store: StateStore,
    run_id: str,
    agent: Agent,
    checkpoint: dict[str, Any],
    steps: list[AgentStep],
    *,
    recovery: bool = False,
) -> None:
    from dendrux.types import _usage_from_dict

    pause.steps = steps
    # Reclaim tolerates absent counters; resume rejects incomplete checkpoints.
    if recovery:
        pause.iteration = checkpoint.get("iteration", 0)
        pause.trace_order_offset = checkpoint.get("trace_order_offset", 0)
        pause.usage = _usage_from_dict(checkpoint.get("usage", {}))
        pause.provider_kwargs = dict(checkpoint.get("provider_kwargs", {}))
    else:
        pause.usage = _usage_from_dict(checkpoint["usage"])
        pause.iteration = checkpoint["iteration"]
        pause.trace_order_offset = checkpoint["trace_order_offset"]
        pause.provider_kwargs = dict(checkpoint["provider_kwargs"])
    await _restore_pause_refs(
        pause,
        store,
        run_id,
        agent,
        checkpoint,
        missing_code=FailureReason.RECOVERY_STATE_MISSING.value
        if recovery
        else "pause_format_unsupported",
    )


async def _restore_pause_refs(
    pause: PauseState,
    store: StateStore,
    run_id: str,
    agent: Agent,
    checkpoint: dict[str, Any],
    *,
    missing_code: str,
) -> None:
    from dendrux.errors import PersistenceError

    for kind, ref in (
        ("image_view", checkpoint.get("image_view_ref")),
        ("skills_session", checkpoint.get("skills", {}).get("session_ref")),
        ("clarification", checkpoint.get("clarification_ref")),
    ):
        if ref is None:
            continue
        state_trace = await store.get_trace(run_id, ref)
        if state_trace is None:
            raise PersistenceError(code=missing_code)
        payload = json.loads(state_trace.content)
        if kind == "image_view":
            pause.image_view = ImageViewState.from_dict(payload)
        elif kind == "skills_session":
            session = current_skill_session(agent)
            if session is not None:
                session.restore_snapshot(payload)
        else:
            pause.pause_message = payload.get("pause_message")


def _resumed_payload(
    expected_status: str,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    user_parts: tuple[ContentPart, ...] | None,
) -> dict[str, Any]:
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
    return resume_data


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

    await _emit_event(
        state_store,
        run_id,
        StoredEventType.RUN_RESUMED.value,
        _resumed_payload(expected_status, tool_results, user_input, user_parts),
    )

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


def _mark_work(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


async def _consume_injected_history(
    ctx: _ResumeContext, on_consumed: Callable[[], None] | None
) -> None:
    consume = getattr(ctx.recorder, "consume_resume", None)
    if consume is None:
        return
    state = getattr(ctx.recorder, "_loop_state", {})
    had_submission = state.get("submission") is not None
    start = len(ctx.pause_state.history)
    for index in range(start, len(ctx.history)):
        ctx.history[index] = await project_message(ctx.recorder, ctx.history[index])
    await consume(ctx.history[start:])
    if had_submission and getattr(ctx.recorder, "_loop_state", {}).get("submission") is None:
        _mark_work(on_consumed)


async def _replay_tool_result(
    ctx: _ResumeContext,
    run_id: str,
    tc: ToolCall,
    tr: ToolResult,
    msg: Message,
    on_consumed: Callable[[], None] | None,
) -> None:
    from dendrux.loops._helpers import notify_tool as _notify_tool
    from dendrux.loops._helpers import notify_tool_started as _notify_tool_started

    iteration = ctx.pause_state.iteration
    if ctx.pause_state.schema_version != 2:
        await ctx.recorder.on_tool_started(run_id, tc, iteration)
    await _notify_tool_started(ctx.notifier, run_id, tc, iteration)
    await ctx.recorder.on_message_appended(run_id, msg, iteration)
    _mark_work(on_consumed)
    await record_tool(ctx.recorder, run_id, tc, tr, iteration)
    await notify_message(ctx.notifier, run_id, msg, iteration)
    await _notify_tool(ctx.notifier, run_id, tc, tr, iteration)


async def _replay_resume_injected_history(
    ctx: _ResumeContext,
    run_id: str,
    *,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    on_consumed: Callable[[], None] | None = None,
) -> None:
    await _consume_injected_history(ctx, on_consumed)
    if tool_results is not None:
        pending = {tc.id: tc for tc in ctx.pause_state.pending_tool_calls}
        start = len(ctx.pause_state.history)
        for index, result in enumerate(tool_results):
            await _replay_tool_result(
                ctx,
                run_id,
                pending[result.call_id],
                result,
                ctx.history[start + index],
                on_consumed,
            )
    elif user_input is not None:
        msg = ctx.history[-1]
        await ctx.recorder.on_message_appended(run_id, msg, ctx.pause_state.iteration)
        _mark_work(on_consumed)
        await notify_message(ctx.notifier, run_id, msg, ctx.pause_state.iteration)
    order_observations(ctx.recorder, ctx.history)


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
        await recorder._emit_event(
            GovernanceEventType.APPROVAL_DECIDED.value, iteration, data, event_id=event_id
        )
    else:
        await record_governance(
            recorder, run_id, GovernanceEventType.APPROVAL_DECIDED.value, iteration, data
        )
    await notify_governance(
        notifier, run_id, GovernanceEventType.APPROVAL_DECIDED.value, iteration, data
    )


def _rejection_decided_payload(
    run_id: str, pause_state: PauseState, tool_results: list[ToolResult]
) -> dict[str, Any]:
    """Legacy batch rejection: every pending call received a synthetic denial."""
    errors = {tr.call_id: tr.error for tr in tool_results}
    outcomes = [(tc, "denied", errors.get(tc.id)) for tc in pause_state.pending_tool_calls]
    return _decided_payload(run_id, outcomes, scalar="rejected")


def _approval_call_counts(recorder: LoopRecorder, pause_state: PauseState) -> dict[str, int]:
    restored_counts: dict[str, int] | None = getattr(recorder, "execution_counts", lambda: None)()
    if restored_counts is not None:
        return restored_counts
    counts = history_call_counts(pause_state.history)
    # The approved fallback historically excludes empty names, too.
    counts.pop("", None)
    return counts


def _limit_approval_clients(
    agent: Agent, outcomes: Any, lookups: Any, counts: dict[str, int]
) -> tuple[Any, Any]:
    from dendrux.tools import enforce_call_limits

    client_calls = [
        tc
        for tc, outcome, _ in outcomes
        if outcome == "client_pending" and tc.name not in agent.deny
    ]
    _, limited = enforce_call_limits(client_calls, max_calls=lookups.max_calls, call_counts=counts)
    client_rejections = {tc.id: result for tc, result in limited}
    outcomes = [
        (tc, "policy_denied", DenialReason.DENIED_BY_POLICY.value)
        if outcome == "client_pending" and tc.name in agent.deny
        else (tc, "call_limited", None)
        if tc.id in client_rejections
        else (tc, outcome, reason)
        for tc, outcome, reason in outcomes
    ]
    return outcomes, client_rejections


def _inject_approved_skill(agent: Agent, pause_state: PauseState, lookups: Any) -> None:
    from dendrux.skills._session import SkillTool

    session = current(agent)
    # An absent executor is a specified error only for a pinned approved script.
    # Inject an execution closure here without advertising the unavailable tool.
    if session is not None and session.configured and session.executor is None:
        for tc in pause_state.pending_tool_calls:
            if tc.name == "run_skill_script" and tc.name not in lookups.fn:
                lookups.fn[tc.name] = SkillTool(session, tc.name)


async def _approval_policy_denial(
    recorder: LoopRecorder,
    notifier: LoopNotifier | None,
    run_id: str,
    tc: ToolCall,
    iteration: int,
    on_work: Callable[[], None] | None,
) -> ToolResult:
    from dendrux.loops._helpers import notify_governance, record_governance

    result, data = static_denial(tc.name, tc.id)
    await record_governance(
        recorder,
        run_id,
        GovernanceEventType.POLICY_DENIED.value,
        iteration,
        data,
        correlation_id=tc.id,
    )
    _mark_work(on_work)
    await notify_governance(
        notifier,
        run_id,
        GovernanceEventType.POLICY_DENIED.value,
        iteration,
        data,
        correlation_id=tc.id,
    )
    return result


def _approval_dispatch_inputs(
    strategy: Strategy | None, warnings: list[str] | None
) -> tuple[Strategy, list[str]]:
    return (
        strategy if strategy is not None else NativeToolCalling(),
        warnings if warnings is not None else [],
    )


async def _execute_approved_tools(
    agent: Agent,
    pause_state: PauseState,
    recorder: LoopRecorder,
    notifier: LoopNotifier | None,
    run_id: str,
    *,
    history: list[Message] | None = None,
    resolved_loop: Loop | None = None,
    strategy: Strategy | None = None,
    notifier_warnings: list[str] | None = None,
    decisions: dict[str, ApprovalDecision] | None = None,
    state_store: StateStore | None = None,
    image_runtime: ImageRuntime | None = None,
    on_work: Callable[[], None] | None = None,
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
    from dendrux.loops.react import _execute_tool, _format_result, _process_tool_calls
    from dendrux.skills._session import SkillTool
    from dendrux.tools import enforce_call_limits
    from dendrux.tools._validate import check_tool_call
    from dendrux.types import AgentStep, ToolResult

    strategy, notifier_warnings = _approval_dispatch_inputs(strategy, notifier_warnings)
    target_history = history if history is not None else pause_state.history
    lookups = await agent.get_tool_lookups(loop=resolved_loop)
    configure = getattr(recorder, "configure_tools", None)
    if configure is not None:
        configure(list(lookups.defs.values()))
    outcomes = _approval_outcomes(pause_state, lookups.target, decisions)
    results: list[tuple[ToolCall, ToolResult]] = []
    counts = _approval_call_counts(recorder, pause_state)
    outcomes, client_rejections = _limit_approval_clients(agent, outcomes, lookups, counts)
    client_pending = [tc for tc, outcome, _ in outcomes if outcome == "client_pending"]
    decided = _decided_payload(
        run_id,
        [
            (tc, "denied" if outcome in {"policy_denied", "call_limited"} else outcome, reason)
            for tc, outcome, reason in outcomes
        ],
        scalar="approved" if decisions is None else None,
    )
    _inject_approved_skill(agent, pause_state, lookups)

    iteration = pause_state.iteration

    async def _append(tc: ToolCall, tr: ToolResult, *, executed: bool = True) -> None:
        if executed:
            await record_tool(recorder, run_id, tc, tr, iteration)
            _mark_work(on_work)
            await _ntfy_tool(notifier, run_id, tc, tr, iteration, warnings=notifier_warnings)
        result_msg = _format_result(strategy, tr)
        if not executed or tr.meta.get("code") == "max_calls":
            result_msg.meta[NOT_EXECUTED_META_KEY] = True
        target_history.append(result_msg)
        result_msg = await record_message(recorder, run_id, result_msg, iteration)
        _mark_work(on_work)
        target_history[-1] = result_msg
        await _ntfy_msg(notifier, run_id, result_msg, iteration, warnings=notifier_warnings)
        results.append((tc, tr))

    for index, (tc, outcome, reason) in enumerate(outcomes):
        if outcome == "client_pending":
            continue
        if outcome == "policy_denied":
            denied = await _approval_policy_denial(
                recorder, notifier, run_id, tc, iteration, on_work
            )
            await _append(tc, denied, executed=False)
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
            if getattr(state_store, "storage_format", 1) != STORAGE_FORMAT_CHECKPOINTED:
                await _rec_tool_started(recorder, run_id, tc, iteration)
            await _ntfy_tool_started(notifier, run_id, tc, iteration, warnings=notifier_warnings)
            await _append(tc, denied)
            continue

        barrier = await _resume_barrier(state_store, run_id)
        if barrier is not None:
            remaining = tuple(o for o in outcomes[index:] if o[1] != "client_pending")
            order_observations(recorder, target_history)
            return _ApprovalResume(results, client_pending, decided, barrier, remaining)

        if isinstance(lookups.fn.get(tc.name), SkillTool):
            _mark_work(on_work)
            outcome_s = await _process_tool_calls(
                step=AgentStep(reasoning=None, action=tc),
                call_counts=counts,
                lookups=lookups,
                strategy=strategy,
                recorder=recorder,
                notifier=notifier,
                run_id=run_id,
                iteration=iteration,
                history=target_history,
                warnings=notifier_warnings,
                deny=agent._deny,
                tool_result_transform=agent.tool_result_transform,
                tenant_id=None,
                **dispatch_policy(name=tc.name, phase="approved"),
                **_tool_image_kwargs(_process_tool_calls, image_runtime),
            )
            results.extend(outcome_s.all_results)
            continue
        if tc.name in agent.deny:
            denied = await _approval_policy_denial(
                recorder, notifier, run_id, tc, iteration, on_work
            )
            await _append(tc, denied, executed=False)
            continue
        _, rejected = enforce_call_limits([tc], max_calls=lookups.max_calls, call_counts=counts)
        if rejected:
            await _append(tc, rejected[0][1])
            continue

        counts[tc.name] = counts.get(tc.name, 0) + 1
        invalid_result = check_tool_call(tc, lookups.defs.get(tc.name))
        if invalid_result is None:
            await _rec_tool_started(recorder, run_id, tc, iteration)
            _mark_work(on_work)
            await _ntfy_tool_started(notifier, run_id, tc, iteration, warnings=notifier_warnings)
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
        pause_reason=PauseReason.REQUESTED.value,
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
        RESUME_RESULTS_STATUS_VALUES if tool_results is not None else RESUME_INPUT_STATUS_VALUES
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


def _resume_claim_fallbacks(
    expected: str, results: list[ToolResult] | None, user_input: Any
) -> tuple[str, ...]:
    if results is not None and expected != RunStatus.WAITING_APPROVAL.value:
        return (RunStatus.WAITING_APPROVAL.value,)
    if results is None and user_input is None and expected == RunStatus.WAITING_APPROVAL.value:
        return (RunStatus.PAUSED.value,)
    return ()


def _resume_claim_payload(
    raw: dict[str, Any],
    status: str,
    results: list[ToolResult] | None,
    user_input: Any,
    user_parts: tuple[ContentPart, ...] | None,
    decisions: list[tuple[str, ApprovalDecision]] | None,
) -> dict[str, Any] | None:
    if raw.get("schema_version") != 2 or (
        results is None and user_input is None and status != RunStatus.WAITING_APPROVAL.value
    ):
        return None
    from dendrux.runtime.submit import serialize_decisions, serialize_tool_results
    from dendrux.types import content_part_to_dict

    payload: dict[str, Any] = (
        {"submitted_decisions": serialize_decisions(PauseState.from_dict(raw), dict(decisions))}
        if decisions is not None
        else {"submitted_tool_results": serialize_tool_results(results)}
        if results is not None
        else {"submitted_user_input": user_input}
        if user_input is not None
        else {"approved": True}
    )
    if user_parts is not None:
        payload["submitted_user_parts"] = [content_part_to_dict(p) for p in user_parts]
    return payload


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
    on_release: Callable[[StateStore], None] | None = None,
) -> tuple[StateStore, str, dict[str, Any]]:
    request = _claim_request(
        raw,
        expected,
        expected_pause_id,
        fallbacks=_resume_claim_fallbacks(expected, results, user_input),
    )
    validate = submission_validator(
        run_id, raw, agent=agent, expected_pause_id=request.generation, call_results=results
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
    for status in request.statuses:
        _canonicalize_legacy_denials(status, run_id, raw, results)
        payload = _resume_claim_payload(raw, status, results, user_input, user_parts, decisions)
        epoch, descriptor, payload = await _claim_submission(
            store,
            run_id,
            request,
            status,
            validate=validate,
            payload=payload,
            checkpointing=True,
            approval=False
            if status == RunStatus.WAITING_APPROVAL.value and results is not None
            else None,
            checkpoint=checkpoint,
        )
        if epoch:
            view = claimed_store(store, run_id, epoch)
            await _report_claim_governance(agent, view, run_id, raw, status, on_release)
            return view, status, _claimed_submission_view(raw, descriptor, payload)
    await _raise_resume_claim_failure(store, run_id, expected)
    raise AssertionError("unreachable")


async def _report_claim_governance(
    agent: Agent,
    store: StateStore,
    run_id: str,
    raw: dict[str, Any],
    status: str,
    on_release: Callable[[StateStore], None] | None,
) -> None:
    try:
        await report_governance_change(agent, store, run_id, raw)
    except BaseException as exc:
        await _release_claim(store, run_id, status)
        if on_release is not None and isinstance(exc, Exception) and _release_supported(store):
            on_release(store)
        raise


async def _release_claim(store: StateStore, run_id: str, status: str) -> bool:
    import sys

    from dendrux.runtime.submit import _release_claim_result

    epoch = getattr(store, "claim_epoch", True)
    try:
        return await _release_claim_result(store, run_id, status, epoch, sys.exception())
    except asyncio.CancelledError as cancellation:
        await _release_claim_result(store, run_id, status, epoch, cancellation)
        raise


async def _shielded_cleanup(cleanup: Callable[[], Any], *, run_id: str | None = None) -> Any:
    import anyio

    from dendrux.runtime.tasks import _drain_cleanup

    result, cancellation = await _drain_cleanup(cleanup(), run_id=run_id)
    if cancellation is not None:
        raise cancellation
    await anyio.lowlevel.checkpoint_if_cancelled()
    return result


@asynccontextmanager
async def _release_setup_failure(
    store: StateStore, run_id: str, status: str, *, enabled: bool = True
) -> AsyncIterator[None]:
    try:
        yield
    except BaseException as exc:
        if enabled and not isinstance(exc, RunFencedError):
            await _release_claim(store, run_id, status)
        raise


async def _resume_session(
    agent: Agent,
    store: StateStore,
    run_id: str,
    loop: Loop | None,
    user_input: UserContent | None,
    tool_results: list[ToolResult] | None,
    decisions: Any,
    expected_status: str,
) -> tuple[SkillSession, str | None, tuple[Any, ...]]:
    await ensure_agent_limits_fit(agent, store)
    items: tuple[Any, ...] = ()
    if user_input is not None:
        user_input, items = split_user_content(user_input)
    if decisions is not None and (
        tool_results is not None
        or user_input is not None
        or expected_status != RunStatus.WAITING_APPROVAL.value
    ):
        raise ValueError("Approval decisions only resume a run awaiting approval.")
    session = current(agent)
    if session is None or session.run_id != run_id:
        session = await continuation(agent, store, run_id, loop=loop)
    return session, user_input, items


async def _read_resume_pause(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    raw: dict[str, Any] | None,
    reads: _PauseReads,
    skip_claim: bool,
    expected_pause_id: str | None,
    *,
    read_record: bool = False,
) -> PauseState:
    if raw is None:
        raise ValueError(f"Run '{run_id}' has no pause state — cannot resume.")
    if not skip_claim:
        require_pause_id(
            agent, expected_pause_id, pause_id=raw.get("pause_id"), store=store, run_id=run_id
        )
    return await _load_resume_pause(
        store, agent, provider, run_id, raw, reads, read_record=read_record
    )


async def _load_resume_pause(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    raw: dict[str, Any],
    reads: _PauseReads,
    *,
    read_record: bool,
) -> PauseState:
    if read_record:
        reads.record = await store.get_run(run_id)
    return await _load_pause_state(store, run_id, raw, agent, provider, reads=reads)


def _require_pause_agent(pause: PauseState, agent: Agent, run_id: str, streaming: bool) -> None:
    if pause.agent_name and pause.agent_name != agent.name:
        raise ValueError(
            f"Agent name mismatch: run '{run_id}' was paused by "
            f"'{pause.agent_name}', but resume called with '{agent.name}'."
            + ("" if streaming else " This can cause state corruption.")
        )


async def _read_resume_state(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    skip_claim: bool,
    expected_pause_id: str | None,
    supplied_raw: dict[str, Any] | None,
    reclaimed: PauseState | None,
    *,
    streaming: bool = False,
) -> tuple[RunRecord | None, _PauseReads, Any, PauseState, bool, Any]:
    reads = _PauseReads(None if streaming else await store.get_run(run_id))
    raw = None if reclaimed is not None else supplied_raw
    if reclaimed is None and raw is None:
        raw = await store.get_pause_state(run_id)
    pause = (
        reclaimed
        if reclaimed is not None
        else await _read_resume_pause(
            store,
            agent,
            provider,
            run_id,
            raw,
            reads,
            skip_claim,
            expected_pause_id,
            read_record=streaming,
        )
    )
    recovering = reclaimed is not None or await _reclaim.needs_resolution(
        store, run_id, pause, checkpoint=reads.checkpoint
    )
    pii = None if streaming else await store.get_pii_mapping(run_id)
    _require_pause_agent(pause, agent, run_id, streaming)
    return reads.record, reads, raw, pause, recovering, pii


async def _resume_submission(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    pause: PauseState,
    raw: Any,
    reads: _PauseReads,
    loop: Loop | None,
    skip_claim: bool,
    expected_status: str,
    expected_pause_id: str | None,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    user_parts: tuple[ContentPart, ...] | None,
    input_items: tuple[Any, ...],
    decisions: Any,
) -> tuple[StateStore, str, Any, list[ToolResult] | None, tuple[ContentPart, ...] | None]:
    if tool_results is not None:
        validate_result_names(run_id, pause, tool_results)
        if not skip_claim:
            tool_results = await prepare_resume_results(agent, store, run_id, pause, tool_results)
    if not skip_claim and (input_items or any(tr.images for tr in tool_results or ())):
        tool_results, user_parts = await _ingest_resume_images(
            agent,
            provider,
            store,
            run_id,
            loop=loop,
            tool_results=tool_results,
            input_items=input_items,
        )
    if not skip_claim:
        assert raw is not None
        store, expected_status, raw = await _claim_resume(
            store,
            run_id,
            raw,
            agent,
            expected_status,
            expected_pause_id=expected_pause_id,
            results=tool_results,
            user_input=user_input,
            user_parts=user_parts,
            decisions=decisions,
            checkpoint=reads.checkpoint,
        )
    return store, expected_status, raw, tool_results, user_parts


@dataclass
class _ResumeProgress:
    submission: Any = None
    started: bool = False

    def _start(self) -> None:
        self.started = True

    def _can_release(self, ctx: _ResumeContext | None, exc: Exception) -> bool:
        if self.started or isinstance(exc, StoreOutcomeUnknown):
            return False
        return self.submission is None or (
            ctx is not None
            and getattr(ctx.recorder, "_loop_state", {}).get("submission") is not None
        )


def _resume_delegation(
    agent: Agent,
    store: StateStore,
    run_id: str,
    record: RunRecord | None,
    trace_link: Any,
    tenant_explicit: bool | None,
) -> DelegationContext:
    depth = (record.meta or {}).get(MAX_DELEGATION_DEPTH_META_KEY) if record else None
    return DelegationContext(
        run_id=run_id,
        trace_link=trace_link,
        delegation_level=record.delegation_level if record else 0,
        persisted=True,
        store_identity=get_store_identity(store),
        max_delegation_depth=depth if isinstance(depth, int) else None,
        tenant_id=record.tenant_id if record else None,
        tenant_explicit=True if tenant_explicit is None else tenant_explicit,
        state_store=store,
        lease_interval_s=agent.lease_interval.total_seconds(),
        loop=asyncio.get_running_loop(),
    )


def _resume_warning_meta(ctx: _ResumeContext | None, meta: dict[str, Any]) -> dict[str, Any]:
    warnings = getattr(ctx, "notifier_warnings", None)
    if warnings:
        meta["notifier_warnings"] = warnings + meta.get("notifier_warnings", [])
    return meta


async def _resume_loop_stream(
    ctx: _ResumeContext, events: AsyncGenerator[RunEvent, None]
) -> AsyncGenerator[RunEvent, None]:
    async with aclosing(events):
        async for event in events:
            if event.run_result is not None:
                _resume_warning_meta(ctx, event.run_result.meta)
            yield event


async def _finish_resumed_result(
    store: StateStore, run_id: str, ctx: _ResumeContext, result: RunResult
) -> RunResult:
    _resume_warning_meta(ctx, result.meta)
    result = await _persist_loop_outcome(
        recorder=ctx.recorder,
        provider_kwargs=ctx.pause_state.provider_kwargs,
        state_store=store,
        run_id=run_id,
        result=result,
    )
    await record_run_finished(ctx.recorder, run_id, result)
    await notify_run_finished(ctx.notifier, run_id, result)
    return result


async def _resume_approval_batch(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    expected: str,
    results: list[ToolResult] | None,
    user_input: str | None,
    image_runtime: ImageRuntime | None,
    *,
    decisions: Any = None,
    on_work: Callable[[], None] | None = None,
    streaming: bool = False,
) -> tuple[_ApprovalResume | None, bool]:
    pause = ctx.pause_state
    approving = (
        results is None and user_input is None and expected == RunStatus.WAITING_APPROVAL.value
    )
    paused_batch = expected == RunStatus.PAUSED.value and bool(pause.pending_tool_calls)
    if not (approving or paused_batch):
        return None, approving
    decision_map = None
    if not approving:
        decision_map = _pending_decision_map(pause)
    elif not streaming:
        decision_map = await _validated_decisions(agent, ctx, run_id, pause, decisions)
    approval = await _execute_approved_tools(
        agent,
        pause,
        ctx.recorder,
        ctx.notifier,
        run_id,
        history=ctx.history,
        resolved_loop=ctx.resolved_loop,
        strategy=ctx.resolved_strategy,
        notifier_warnings=ctx.notifier_warnings,
        decisions=decision_map,
        state_store=store,
        image_runtime=image_runtime,
        on_work=on_work,
    )
    return approval, approving


async def _finish_approval_batch(
    run_id: str,
    ctx: _ResumeContext,
    expected: str,
    results: list[ToolResult] | None,
    approval: _ApprovalResume | None,
    approving: bool,
) -> RunResult | None:
    pause = ctx.pause_state
    if approval is not None:
        if approving:
            await _record_approval_decided(
                ctx.recorder, ctx.notifier, run_id, pause.iteration, approval.decided
            )
        stop = _approval_stop_result(run_id, pause, ctx.history, approval)
        if stop is not None:
            return stop
    if results is not None and expected == RunStatus.WAITING_APPROVAL.value:
        data = _rejection_decided_payload(run_id, pause, results)
        await _record_approval_decided(ctx.recorder, ctx.notifier, run_id, pause.iteration, data)
    return None


async def _resume_approval_work(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    expected: str,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    decisions: Any,
    image_runtime: ImageRuntime | None,
    on_work: Callable[[], None],
) -> RunResult | None:
    approval, approving = await _resume_approval_batch(
        agent,
        store,
        run_id,
        ctx,
        expected,
        tool_results,
        user_input,
        image_runtime,
        decisions=decisions,
        on_work=on_work,
    )
    return await _finish_approval_batch(run_id, ctx, expected, tool_results, approval, approving)


def _initial_call_counts_kwargs(loop_method: Any, recorder: LoopRecorder) -> dict[str, Any]:
    if not _accepts_kwarg(loop_method, "_initial_call_counts"):
        return {}
    counts = getattr(recorder, "execution_counts", lambda: None)()
    return {"_initial_call_counts": counts}


def _resumed_loop_kwargs(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    image_runtime: ImageRuntime | None,
    loop_method: Any,
) -> dict[str, Any]:
    carry = _budget_carry_kwargs(loop_method, ctx.pause_state)
    images = _image_loop_kwargs(
        loop_method, image_runtime, _resume_image_view(ctx.pause_state, ctx.history)
    )
    return dict(
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
        **_initial_call_counts_kwargs(loop_method, ctx.recorder),
        **carry,
        **images,
    )


async def _run_resumed_loop(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    pii: Any,
    image_runtime: ImageRuntime | None,
    on_work: Callable[[], None],
) -> RunResult:
    kwargs = _resumed_loop_kwargs(
        agent, provider, store, run_id, ctx, image_runtime, ctx.resolved_loop.run
    )
    on_work()
    return await ctx.resolved_loop.run(**kwargs, initial_pii_mapping=pii)


async def _cancel_resumed_run(
    agent: Agent, store: StateStore, run_id: str, ctx: _ResumeContext | None
) -> tuple[RunResult, bool]:
    interrupted, swallow = _take_task_interrupt(agent._task_manager, run_id)
    result = await _drain_task_cancellation(store, run_id, interrupted)
    if ctx is not None:
        try:
            await record_run_finished(ctx.recorder, run_id, result)
        except Exception:
            logger.warning(
                "Recorder finish failed for run %s", run_id, exc_info=safe_exception_info()
            )
        await notify_run_finished(ctx.notifier, run_id, result)
    return result, _may_swallow_cancellation(swallow)


async def _resume_failure_hooks(ctx: _ResumeContext | None, run_id: str, exc: Exception) -> None:
    if ctx is None:
        return
    try:
        await record_run_failed(ctx.recorder, run_id, exc)
        await notify_run_failed(ctx.notifier, run_id, exc)
    except Exception:
        logger.warning(
            "Lifecycle hooks failed during resume error handler for run %s",
            run_id,
            exc_info=safe_exception_info(),
        )


async def _released_resume_status(store: StateStore, run_id: str) -> RunStatus:
    try:
        record = await store.get_run(run_id)
        if record is not None:
            return RunStatus(record.status)
    except Exception:
        logger.warning(
            "Released run status read failed for %s", run_id, exc_info=safe_exception_info()
        )
    return RunStatus.RUNNING


async def _released_resume_hooks(
    store: StateStore, run_id: str, ctx: _ResumeContext | None, exc: Exception
) -> RunStatus:
    from dendrux.types import RunResult

    outcome = await _released_resume_status(store, run_id)
    if ctx is None:
        return outcome
    if outcome == RunStatus.RUNNING:
        await _resume_failure_hooks(ctx, run_id, exc)
        return outcome
    result = RunResult(
        run_id=run_id,
        status=outcome,
        steps=ctx.pause_state.steps,
        iteration_count=ctx.pause_state.iteration,
        usage=ctx.pause_state.usage,
    )
    if ctx.recorder_started:
        try:
            await record_run_finished(ctx.recorder, run_id, result)
        except Exception:
            logger.warning(
                "Recorder finish failed for run %s", run_id, exc_info=safe_exception_info()
            )
    if ctx.notifier_started:
        await notify_run_finished(ctx.notifier, run_id, result)
    return outcome


async def _try_resume_release(
    store: StateStore, run_id: str, status: str, ctx: _ResumeContext | None, exc: Exception
) -> RunStatus:
    released = await _release_claim(store, run_id, status)
    if released:
        return await _released_resume_hooks(store, run_id, ctx, exc)
    await _resume_failure_hooks(ctx, run_id, exc)
    return await _released_resume_status(store, run_id)


def _resume_release_supported(store: StateStore, status: str) -> bool:
    return status == RunStatus.RUNNING.value or _release_supported(store)


async def _handle_resume_failure(
    store: StateStore,
    run_id: str,
    status: str,
    ctx: _ResumeContext | None,
    exc: Exception,
    progress: _ResumeProgress,
    meta: dict[str, Any],
) -> None:
    if progress._can_release(ctx, exc) and _resume_release_supported(store, status):
        await _try_resume_release(store, run_id, status, ctx, exc)
        return
    _attach_run_error_meta(exc, run_id, meta)
    await _persist_run_failure(store, run_id, exc)
    await _resume_failure_hooks(ctx, run_id, exc)


async def _fenced_resume_result(
    store: StateStore, run_id: str, ctx: _ResumeContext | None
) -> RunResult:
    result = await _build_cached_result(store, run_id)
    if ctx is not None:
        await _finish_fenced_lifecycle(run_id, result, ctx.recorder, ctx.notifier)
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
    async with _release_setup_failure(state_store, run_id, expected_status, enabled=_skip_claim):
        skill_session, user_input, input_items = await _resume_session(
            agent, state_store, run_id, loop, user_input, tool_results, decisions, expected_status
        )
    with skill_session.bind():
        async with _release_setup_failure(
            state_store, run_id, expected_status, enabled=_skip_claim
        ):
            (
                run_record,
                reads,
                raw_pause,
                pause_state,
                recovering,
                _saved_pii_mapping,
            ) = await _read_resume_state(
                state_store,
                agent,
                provider,
                run_id,
                _skip_claim,
                expected_pause_id,
                _raw_pause,
                _reclaim_pause,
            )
            (
                state_store,
                expected_status,
                raw_pause,
                tool_results,
                user_parts,
            ) = await _resume_submission(
                state_store,
                agent,
                provider,
                run_id,
                pause_state,
                raw_pause,
                reads,
                loop,
                _skip_claim,
                expected_status,
                expected_pause_id,
                tool_results,
                user_input,
                user_parts,
                input_items,
                decisions,
            )

        # 4-10. Everything that runs after on_run_started fires must live
        #       inside the try block so a closing hook can balance the lifecycle
        #       pair on any failure (including injected-history replay errors).
        ctx: _ResumeContext | None = None
        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        progress = _ResumeProgress()

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
            progress.submission = getattr(ctx.recorder, "_loop_state", {}).get("submission")
            image_runtime = _image_runtime(agent, state_store, run_tenant, run_id)

            # 8a. Fire run-level lifecycle hooks. AFTER this point, any failure
            #     must close the lifecycle pair in the except handler.
            await record_run_started(
                ctx.recorder, run_id, agent_name=agent.name, agent_model=provider.model
            )
            ctx.recorder_started = True
            trace_link, ctx.notifier = await _start_trace(
                ctx.notifier,
                state_store,
                run_id,
                agent_name=agent.name,
                agent_model=provider.model,
                saved=getattr(run_record, "trace_link", None),
                on_started=ctx._mark_notifier_started,
            )

            await agent.get_tool_lookups(loop=ctx.resolved_loop)
            mcp_meta.update(_snapshot_mcp_meta(agent))

            # 8b. tool/user-input history. Inside the try so
            #     a recorder failure here fires on_run_failed.
            if not recovering:
                await _replay_resume_injected_history(
                    ctx,
                    run_id,
                    tool_results=tool_results,
                    user_input=user_input,
                    on_consumed=progress._start,
                )

            ctx_token = set_delegation_context(
                _resume_delegation(
                    agent, state_store, run_id, run_record, trace_link, _tenant_explicit
                )
            )

            if recovering:
                stop = await _reclaim.resolve(
                    agent,
                    state_store,
                    run_id,
                    ctx,
                    image_runtime=image_runtime,
                    tenant_id=run_tenant,
                    on_work=progress._start,
                )
                if stop is not None:
                    return await _finish_resumed_result(state_store, run_id, ctx, stop)

            if not recovering:
                stop = await _resume_approval_work(
                    agent,
                    state_store,
                    run_id,
                    ctx,
                    expected_status,
                    tool_results,
                    user_input,
                    decisions,
                    image_runtime,
                    progress._start,
                )
                if stop is not None:
                    _attach_mcp_meta(stop, mcp_meta)
                    return await _finish_resumed_result(state_store, run_id, ctx, stop)

            result = await _run_resumed_loop(
                agent,
                provider,
                state_store,
                run_id,
                ctx,
                _saved_pii_mapping,
                image_runtime,
                progress._start,
            )
            _attach_mcp_meta(result, mcp_meta)
            return await _finish_resumed_result(state_store, run_id, ctx, result)

        except asyncio.CancelledError:
            cancelled_result, swallow = await _cancel_resumed_run(agent, state_store, run_id, ctx)
            if swallow:
                return cancelled_result
            raise

        except RunFencedError:
            return await _fenced_resume_result(state_store, run_id, ctx)

        except Exception as exc:
            await _handle_resume_failure(
                state_store,
                run_id,
                expected_status,
                ctx,
                exc,
                progress,
                _resume_warning_meta(ctx, mcp_meta),
            )
            raise

        finally:
            agent._task_manager.release_current(run_id)
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


async def _stream_resume_images(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore | None,
    state_store_resolver: Callable[[], Any] | None,
    run_id: str,
    expected_pause_id: str | None,
    stream_results: list[ToolResult] | None,
    input_items: tuple[Any, ...],
) -> tuple[StateStore, list[ToolResult] | None, tuple[ContentPart, ...] | None, bool]:
    results_prepared = False
    # Image setup raises (the run stays waiting; no RUN_ERROR event).
    if store is None and state_store_resolver is not None:
        store = await state_store_resolver()
    if store is None:
        from dendrux.errors import PersistenceNotConfiguredError

        raise PersistenceNotConfiguredError()
    await ensure_agent_limits_fit(agent, store)
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
        validate_result_names(run_id, media_pause, stream_results)
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
    return store, stream_results, user_parts, results_prepared


async def _stream_resume_state(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    expected_pause_id: str | None,
    stream_results: list[ToolResult] | None,
    results_prepared: bool,
    stream_context: StreamContext,
) -> tuple[Any, _PauseReads, Any, PauseState, bool, list[ToolResult] | None]:
    skill_session = await continuation(agent, store, run_id)
    stream_context.set_session(skill_session)

    run_record, reads, raw_pause, pause_state, recovering, _ = await _read_resume_state(
        store,
        agent,
        provider,
        run_id,
        False,
        expected_pause_id,
        None,
        None,
        streaming=True,
    )

    if stream_results is not None:
        validate_result_names(run_id, pause_state, stream_results)
        if not results_prepared:
            stream_results = await prepare_resume_results(
                agent, store, run_id, pause_state, stream_results
            )

    return run_record, reads, raw_pause, pause_state, recovering, stream_results


def _stopped_stream_event(result: RunResult) -> RunEvent:
    from dendrux.types import RunEvent

    return RunEvent(
        type=RunEventType.RUN_CANCELLED
        if result.status == RunStatus.CANCELLED
        else RunEventType.RUN_PAUSED,
        run_result=result,
    )


async def _stream_approval_events(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    expected: str,
    results: list[ToolResult] | None,
    user_input: str | None,
    image_runtime: ImageRuntime | None,
    meta: dict[str, Any],
    on_work: Callable[[], None] | None = None,
) -> AsyncGenerator[RunEvent, None]:
    from dendrux.types import RunEvent

    approval, approving = await _resume_approval_batch(
        agent,
        store,
        run_id,
        ctx,
        expected,
        results,
        user_input,
        image_runtime,
        streaming=True,
        on_work=on_work,
    )
    if approval is not None:
        for tc, tr in approval.results:
            yield RunEvent(
                type=RunEventType.TOOL_RESULT,
                tool_call=tc,
                tool_result=tr,
                iteration=ctx.pause_state.iteration,
                phase="commentary",
            )
    stop = await _finish_approval_batch(run_id, ctx, expected, results, approval, approving)
    if stop is not None:
        _attach_mcp_meta(stop, meta)
        yield _stopped_stream_event(await _finish_resumed_result(store, run_id, ctx, stop))


def _resume_expected_status(results: list[ToolResult] | None, user_input: str | None) -> str:
    if results is not None:
        return RunStatus.WAITING_CLIENT_TOOL.value
    if user_input is not None:
        return RunStatus.WAITING_HUMAN_INPUT.value
    return RunStatus.WAITING_APPROVAL.value


def _mark_stream_recovery_work(shared: dict[str, Any]) -> None:
    shared["work_started"] = True
    shared["progress"]._start()


def _stream_resume_started(shared: dict[str, Any]) -> bool:
    # Work is an independent reason to finalize; it must never be made retryable.
    return bool(shared.get("work_started") or shared.get("claimed"))


async def _stream_resume_failure(
    _shared: dict[str, Any], extra_notifier: LoopNotifier | None, run_id: str, exc: Exception
) -> RunStatus:
    store = _shared.get("state_store")
    err_ctx = _shared.get("ctx")
    progress = _shared.get("progress")
    if (
        store is not None
        and progress is not None
        and progress._can_release(err_ctx, exc)
        and _resume_release_supported(store, _shared["expected_status"])
    ):
        return await _try_resume_release(store, run_id, _shared["expected_status"], err_ctx, exc)
    return await _finish_stream_resume_failure(_shared, store, err_ctx, extra_notifier, run_id, exc)


async def _finish_stream_resume_failure(
    _shared: dict[str, Any],
    store: StateStore | None,
    err_ctx: _ResumeContext | None,
    extra_notifier: LoopNotifier | None,
    run_id: str,
    exc: Exception,
) -> RunStatus:
    if store is not None and _stream_resume_started(_shared):
        await _persist_run_failure(store, run_id, exc, safe_event=True)
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
    prelude = _shared.get("released_prelude")
    if prelude is not None:
        return await _released_resume_status(prelude, run_id)
    return RunStatus.ERROR


def _remember_stream_release(shared: dict[str, Any], store: StateStore) -> None:
    shared["released_prelude"] = store


async def _finish_resume_stream_cleanup(
    store: StateStore | None,
    run_id: str,
    _shared: dict[str, Any],
    mcp_meta: dict[str, Any],
    cancel_won: bool,
    persisted_row: RunRecord | None | _OmittedType,
) -> RunResult:
    from dendrux.types import RunResult

    cleanup_recorder = _shared.get("recorder")
    terminal_status = RunStatus.CANCELLED
    cancel_persisted = cancel_won
    if store is not None and not cancel_won:
        try:
            run_record = (
                await store.get_run(run_id)
                if isinstance(persisted_row, _OmittedType)
                else persisted_row
            )
            if run_record is not None:
                terminal_status = RunStatus(run_record.status)
                cancel_persisted = terminal_status == RunStatus.CANCELLED
        except Exception:
            pass  # CANCELLED is a safe fallback
    cancelled_result = RunResult(
        run_id=run_id,
        status=terminal_status,
        meta={**deepcopy(mcp_meta), CANCEL_PERSISTED_META_KEY: cancel_persisted},
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

    Execution exceptions yield RUN_ERROR events. Before consumption or work,
    cleanup can release the claim; the result reports the observed status and
    the failed attempt's error. Otherwise the run is finalized as ERROR.

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
            store, stream_results, user_parts, results_prepared = await _stream_resume_images(
                agent,
                provider,
                store,
                state_store_resolver,
                run_id,
                expected_pause_id,
                stream_results,
                input_items,
            )

        try:
            store = await _resolve_resume_stream_store(store, state_store_resolver)
            _shared["state_store"] = store

            await ensure_agent_limits_fit(agent, store)

            (
                run_record,
                reads,
                raw_pause,
                pause_state,
                recovering,
                stream_results,
            ) = await _stream_resume_state(
                agent,
                provider,
                store,
                run_id,
                expected_pause_id,
                stream_results,
                results_prepared,
                stream_context,
            )

            # 4. Determine expected status and claim
            expected = _resume_expected_status(stream_results, user_input)

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
                on_release=partial(_remember_stream_release, _shared),
            )
            _shared["state_store"] = store
            _shared["claimed"] = True
            _shared["expected_status"] = expected
            progress = _shared["progress"] = _ResumeProgress()

            # 5. Prepare (shared helper — history, notifier).
            #    Does NOT fire any lifecycle hooks; that's done below inside
            #    the same try block so failure or released-finish hooks pair
            #    with on_run_started.
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
            progress.submission = getattr(ctx.recorder, "_loop_state", {}).get("submission")

            # 5a. Fire run-level lifecycle hooks. The exception handler closes
            #     the pair with failure or the observed released status.
            await record_run_started(
                ctx.recorder, run_id, agent_name=agent.name, agent_model=provider.model
            )
            ctx.recorder_started = True
            trace_link, ctx.notifier = await _start_trace(
                ctx.notifier,
                store,
                run_id,
                agent_name=agent.name,
                agent_model=provider.model,
                saved=getattr(run_record, "trace_link", None),
                on_started=ctx._mark_notifier_started,
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
                    ctx,
                    run_id,
                    tool_results=stream_results,
                    user_input=user_input,
                    on_consumed=progress._start,
                )

            stream_context.set_delegation(
                _resume_delegation(agent, store, run_id, run_record, trace_link, _tenant_explicit)
            )

            # 7. Emit RUN_RESUMED as first event (contract guarantee)
            yield RunEvent(type=RunEventType.RUN_RESUMED, run_id=run_id)

            if recovering:
                stop = await _reclaim.resolve(
                    agent,
                    store,
                    run_id,
                    ctx,
                    image_runtime=image_runtime,
                    tenant_id=run_record.tenant_id if run_record else None,
                    on_work=partial(_mark_stream_recovery_work, _shared),
                )
                if stop is not None:
                    yield _stopped_stream_event(
                        await _finish_resumed_result(store, run_id, ctx, stop)
                    )
                    return

            if not recovering:
                async with aclosing(
                    _stream_approval_events(
                        agent,
                        store,
                        run_id,
                        ctx,
                        expected,
                        stream_results,
                        user_input,
                        image_runtime,
                        mcp_meta,
                        progress._start,
                    )
                ) as approval_events:
                    async for event in approval_events:
                        yield event
                        if event.run_result is not None:
                            return

            # 9. Stream the loop (see run_stream for interrupt + closing notes)
            interrupt = agent._task_manager.register_interrupt(run_id)
            progress._start()
            loop_stream = ctx.resolved_loop.run_stream(
                **_resumed_loop_kwargs(
                    agent, provider, store, run_id, ctx, image_runtime, ctx.resolved_loop.run_stream
                ),
                interrupt=interrupt,
            )
            async with aclosing(
                _stream_loop_events(
                    _resume_loop_stream(ctx, loop_stream),
                    state_store=store,
                    recorder=ctx.recorder,
                    notifier=ctx.notifier,
                    run_id=run_id,
                    provider_kwargs=ctx.pause_state.provider_kwargs,
                    shared=_shared,
                    mcp_meta=mcp_meta,
                )
            ) as events:
                async for event in events:
                    yield event

        except RunFencedError:
            store = _shared["state_store"]
            result = await _fenced_resume_result(store, run_id, _shared.get("ctx"))
            yield _fenced_stream_event(run_id, result)

        except Exception as exc:
            failure_status = await _stream_resume_failure(_shared, extra_notifier, run_id, exc)

            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=failure_status,
                    error=_safe_error(exc),
                    error_info=_provider_error_info(exc),
                    meta=_resume_warning_meta(
                        _shared.get("ctx"), _stream_error_meta(mcp_meta, exc)
                    ),
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
        persisted_row: RunRecord | None | _OmittedType = _OMITTED

        def mark_cancelled() -> None:
            nonlocal cancel_won
            cancel_won = True

        partial_answer = _partial_answer(_shared)
        if store is not None:
            try:
                cancel_won, persisted_row = await _persist_run_cancellation(
                    store,
                    run_id,
                    {
                        "reason": CancelReason.STREAM_CLOSED.value,
                        "partial_output": partial_answer is not None,
                    },
                    on_won=mark_cancelled,
                    answer=partial_answer,
                    emit_event=resumed,
                    read_on_loss=lambda: "notifier" in _shared,
                    safe_read=True,
                )
            except Exception:
                logger.error(
                    "Failed to cancel run %s during resume stream cleanup",
                    run_id,
                    exc_info=safe_exception_info(),
                )

        if "notifier" in _shared:
            return await _finish_resume_stream_cleanup(
                store, run_id, _shared, mcp_meta, cancel_won, persisted_row
            )
        return None

    return _RunStream(
        run_id=run_id,
        generator=stream_context.wrap(_generate()),
        cleanup=lambda: _shielded_cleanup(_cleanup, run_id=run_id),
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

    async with _release_setup_failure(store, run_id, status):
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


async def _resume_claim_status(store: StateStore, run_id: str, prior: str | None) -> str:
    if prior is not None:
        return prior
    record = await store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    return record.status


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
            When omitted, uses the current run-record status for release.
            Pass the original pause status to restore that pause on setup failure.

    Returns:
        RunResult from the resumed loop.

    Raises:
        ValueError: If pause_data is missing, no submitted data found,
            or submitted call_ids don't match pending tool calls.
    """
    from dendrux.runtime.submit import deserialize_tool_results

    try:
        state_store = await _owned_claim_view(state_store, run_id, claim_epoch)
        pre_claim_status = await _resume_claim_status(state_store, run_id, pre_claim_status)
        raw_pause = await _validate_owned_pause(
            state_store,
            run_id,
            agent,
            pre_claim_status,
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

        return await _resume_core(
            run_id,
            state_store=state_store,
            agent=agent,
            provider=provider,
            expected_status=pre_claim_status,
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
        # where a mismatch releases an unconsumed claim; after consumption it
        # finalizes the run as ERROR.
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

    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        expected_status=pre_claim_status,
        user_input=submitted_input,
        extra_notifier=extra_notifier,
        _skip_claim=True,
        user_parts=_stored_parts(raw_pause.get("submitted_user_parts")),
        _raw_pause=raw_pause,
        _tenant_explicit=_tenant_explicit,
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
