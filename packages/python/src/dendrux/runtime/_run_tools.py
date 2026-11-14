"""Discover and announce the governed runtime tool view."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from dendrux._lifecycle import (
    StoredEventType,
)
from dendrux.errors import (
    safe_exception_info,
)
from dendrux.runtime import (
    _run_errors,
    _run_observers,
)
from dendrux.runtime._run_errors import (
    logger,
)
from dendrux.skills._session import SkillSession, current
from dendrux.types import (
    GovernanceEventType,
)

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder


_TOOL_NAME_SCAN_LIMIT = 1000


_TOOL_NAME_CHARS = 64


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


def _mcp_error_payload(error: _MCPDiscoveryError) -> dict[str, Any]:
    """Payload of the run-fatal mcp.error event, with a stable ``code``."""
    # Deferred: dendrux.mcp needs the optional mcp extra, and discovery only
    # fails when the Agent has MCP tool sources.
    from dendrux.mcp._errors import MCPAuthorizationRequired, error_code_for

    cause = error.__cause__ or error
    payload: dict[str, Any] = {
        "error": str(error)[: _run_errors.ERROR_TEXT_CHARS],
        "error_type": _run_observers._cause_type(error),
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


_MCPDiscoveryError.__module__ = "dendrux.runtime.runner"
