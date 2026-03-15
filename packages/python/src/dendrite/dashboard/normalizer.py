"""Timeline normalizer — converts raw DB records into an ordered timeline.

The dashboard's run detail page needs a single ordered list of timeline
nodes. This module assembles that from four data sources:

    run_events   — lifecycle events with durable timestamps
    react_traces — conversation messages (for payload inspection)
    tool_calls   — tool execution details (params, result, duration)
    token_usage  — per-LLM-call token counts (via run_events data)

The normalizer is the ONLY place where these sources are joined.
The API layer calls normalize_timeline() and returns the result.
The React app renders it — no reconstruction, no ad-hoc joins.

Design rules:
    - sequence_index is the ordering key (not timestamps)
    - Pause segments contain all pending tool calls (not one per call)
    - Wait duration = resumed.created_at - paused.created_at
    - Only observable data is included (never pause_data)
    - Redacted content only (traces/tool_calls are already redacted)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

# ------------------------------------------------------------------
# Timeline node types
# ------------------------------------------------------------------


@dataclass
class RunStartedNode:
    """The run began."""

    type: str = "run_started"
    sequence_index: int = 0
    agent_name: str = ""
    timestamp: datetime | None = None


@dataclass
class LLMCallNode:
    """An LLM call completed."""

    type: str = "llm_call"
    sequence_index: int = 0
    iteration: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None
    has_tool_calls: bool = False
    timestamp: datetime | None = None
    # Enriched from traces: the assistant message content
    assistant_text: str | None = None


@dataclass
class ToolCallNode:
    """A tool call completed."""

    type: str = "tool_call"
    sequence_index: int = 0
    iteration: int = 0
    tool_call_id: str = ""
    tool_name: str = ""
    target: str = "server"
    success: bool = True
    duration_ms: int | None = None
    timestamp: datetime | None = None
    # Enriched from tool_calls table
    params: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None


@dataclass
class PendingToolCallInfo:
    """One pending tool call within a pause segment."""

    tool_call_id: str = ""
    tool_name: str = ""
    target: str | None = None


@dataclass
class SubmittedResultInfo:
    """One submitted result within a resume event."""

    call_id: str = ""
    tool_name: str = ""
    success: bool = True


@dataclass
class PauseSegmentNode:
    """Agent paused for client-side tool(s) and was later resumed.

    This is the dashboard's signature element. One segment per
    pause/resume cycle, containing all pending tool calls and
    the submitted results.
    """

    type: str = "pause_segment"
    sequence_index: int = 0
    iteration: int = 0
    pause_status: str = ""
    pending_tool_calls: list[PendingToolCallInfo] = field(default_factory=list)
    paused_at: datetime | None = None
    resumed_at: datetime | None = None
    wait_duration_ms: int | None = None
    submitted_results: list[SubmittedResultInfo] = field(default_factory=list)
    user_input: str | None = None


@dataclass
class FinishNode:
    """The run completed successfully."""

    type: str = "finish"
    sequence_index: int = 0
    status: str = ""
    timestamp: datetime | None = None


@dataclass
class ErrorNode:
    """The run failed."""

    type: str = "error"
    sequence_index: int = 0
    error: str = ""
    timestamp: datetime | None = None


@dataclass
class CancelledNode:
    """The run was cancelled."""

    type: str = "cancelled"
    sequence_index: int = 0
    timestamp: datetime | None = None


# Union of all node types
TimelineNode = (
    RunStartedNode
    | LLMCallNode
    | ToolCallNode
    | PauseSegmentNode
    | FinishNode
    | ErrorNode
    | CancelledNode
)


@dataclass
class RunSummary:
    """Run-level metadata for the detail page header."""

    run_id: str
    agent_name: str
    status: str
    model: str | None = None
    strategy: str | None = None
    input_text: str | None = None
    answer: str | None = None
    error: str | None = None
    iteration_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class NormalizedTimeline:
    """Complete timeline for one run — the API response shape."""

    summary: RunSummary
    nodes: list[TimelineNode]
    system_prompt: str | None = None
    # Traces keyed by iteration for payload inspection
    messages_by_iteration: dict[int, list[dict[str, Any]]] = field(default_factory=dict)


# ------------------------------------------------------------------
# Normalizer
# ------------------------------------------------------------------


def _compute_wait_ms(paused_at: datetime | None, resumed_at: datetime | None) -> int | None:
    """Compute wait duration in ms from pause/resume timestamps."""
    if paused_at is None or resumed_at is None:
        return None
    delta = resumed_at - paused_at
    return int(delta.total_seconds() * 1000)


async def normalize_timeline(
    run_id: str,
    state_store: Any,
) -> NormalizedTimeline | None:
    """Build a normalized timeline for a single run.

    Returns None if the run doesn't exist.

    This is the single entry point for the dashboard API. It:
    1. Loads run summary
    2. Loads run_events (ordered by sequence_index)
    3. Loads traces and tool_calls for enrichment
    4. Merges pause/resume event pairs into PauseSegmentNodes
    5. Enriches tool events with params/results from tool_calls table
    6. Enriches LLM events with assistant text from traces
    7. Returns a single ordered timeline
    """
    # 1. Load run summary
    run_record = await state_store.get_run(run_id)
    if run_record is None:
        return None

    summary = RunSummary(
        run_id=run_record.id,
        agent_name=run_record.agent_name,
        status=run_record.status,
        model=run_record.model,
        strategy=run_record.strategy,
        input_text=run_record.input_data.get("input") if run_record.input_data else None,
        answer=run_record.answer,
        error=run_record.error,
        iteration_count=run_record.iteration_count,
        total_input_tokens=run_record.total_input_tokens,
        total_output_tokens=run_record.total_output_tokens,
        total_cost_usd=run_record.total_cost_usd,
        created_at=run_record.created_at,
        updated_at=run_record.updated_at,
    )

    # 2. Load all data sources
    events = await state_store.get_run_events(run_id)
    traces = await state_store.get_traces(run_id)
    tool_calls = await state_store.get_tool_calls(run_id)

    # 3. Build lookup indexes for enrichment
    # Tool calls by tool_call_id (correlation_id on tool.completed events)
    tool_call_by_id: dict[str, Any] = {}
    for tc in tool_calls:
        tool_call_by_id[tc.tool_call_id] = tc

    # System prompt from run.started event (the strategy layer doesn't
    # persist system messages to traces — they're rebuilt each iteration)
    system_prompt: str | None = None
    for ev in events:
        if ev.event_type == "run.started" and ev.data:
            system_prompt = ev.data.get("system_prompt")
            break

    # Traces by iteration for message inspection
    messages_by_iteration: dict[int, list[dict[str, Any]]] = {}
    for trace in traces:
        iteration = trace.meta.get("iteration", 0) if trace.meta else 0
        msg = {
            "role": trace.role,
            "content": trace.content,
            "order_index": trace.order_index,
            "meta": trace.meta,
            "created_at": str(trace.created_at) if trace.created_at else None,
        }
        messages_by_iteration.setdefault(iteration, []).append(msg)

    # Assistant text by iteration (for LLM call enrichment)
    # Use setdefault to keep the first assistant message per iteration
    assistant_text_by_iter: dict[int, str] = {}
    for trace in traces:
        if trace.role == "assistant":
            iteration = trace.meta.get("iteration", 0) if trace.meta else 0
            assistant_text_by_iter.setdefault(iteration, trace.content)

    # 4. Convert events to timeline nodes, merging pause/resume pairs
    nodes: list[TimelineNode] = []

    i = 0
    while i < len(events):
        event = events[i]
        etype = event.event_type
        data = event.data or {}

        if etype == "run.started":
            nodes.append(
                RunStartedNode(
                    sequence_index=event.sequence_index,
                    agent_name=data.get("agent_name", ""),
                    timestamp=event.created_at,
                )
            )

        elif etype == "llm.completed":
            nodes.append(
                LLMCallNode(
                    sequence_index=event.sequence_index,
                    iteration=event.iteration_index,
                    input_tokens=data.get("input_tokens", 0),
                    output_tokens=data.get("output_tokens", 0),
                    cost_usd=data.get("cost_usd"),
                    model=data.get("model"),
                    has_tool_calls=data.get("has_tool_calls", False),
                    timestamp=event.created_at,
                    assistant_text=assistant_text_by_iter.get(event.iteration_index),
                )
            )

        elif etype == "tool.completed":
            # Enrich from tool_calls table via correlation_id
            tc_record = tool_call_by_id.get(event.correlation_id or "")
            nodes.append(
                ToolCallNode(
                    sequence_index=event.sequence_index,
                    iteration=event.iteration_index,
                    tool_call_id=event.correlation_id or "",
                    tool_name=data.get("tool_name", ""),
                    target=data.get("target", "server"),
                    success=data.get("success", True),
                    duration_ms=data.get("duration_ms"),
                    timestamp=event.created_at,
                    params=tc_record.params if tc_record else None,
                    result=tc_record.result if tc_record else None,
                    error_message=tc_record.error_message if tc_record else None,
                )
            )

        elif etype == "run.paused":
            # Look ahead for matching run.resumed to build a PauseSegmentNode
            pending_calls = [
                PendingToolCallInfo(
                    tool_call_id=tc.get("id", ""),
                    tool_name=tc.get("name", ""),
                    target=tc.get("target"),
                )
                for tc in data.get("pending_tool_calls", [])
            ]

            # Search forward for the matching resume
            resume_event = None
            for j in range(i + 1, len(events)):
                if events[j].event_type == "run.resumed":
                    resume_event = events[j]
                    break

            resume_data = resume_event.data or {} if resume_event else {}
            submitted = [
                SubmittedResultInfo(
                    call_id=r.get("call_id", ""),
                    tool_name=r.get("name", ""),
                    success=r.get("success", True),
                )
                for r in resume_data.get("submitted_results", [])
            ]

            nodes.append(
                PauseSegmentNode(
                    sequence_index=event.sequence_index,
                    iteration=event.iteration_index,
                    pause_status=data.get("status", ""),
                    pending_tool_calls=pending_calls,
                    paused_at=event.created_at,
                    resumed_at=resume_event.created_at if resume_event else None,
                    wait_duration_ms=_compute_wait_ms(
                        event.created_at,
                        resume_event.created_at if resume_event else None,
                    ),
                    submitted_results=submitted,
                    user_input=resume_data.get("user_input"),
                )
            )

        elif etype == "run.resumed":
            # Already consumed by the pause handler above.
            # If we hit one without a preceding pause (shouldn't happen),
            # skip it — the pause segment already captured the data.
            pass

        elif etype == "run.completed":
            nodes.append(
                FinishNode(
                    sequence_index=event.sequence_index,
                    status=data.get("status", "success"),
                    timestamp=event.created_at,
                )
            )

        elif etype == "run.error":
            nodes.append(
                ErrorNode(
                    sequence_index=event.sequence_index,
                    error=data.get("error", ""),
                    timestamp=event.created_at,
                )
            )

        elif etype == "run.cancelled":
            nodes.append(
                CancelledNode(
                    sequence_index=event.sequence_index,
                    timestamp=event.created_at,
                )
            )

        i += 1

    return NormalizedTimeline(
        summary=summary,
        nodes=nodes,
        system_prompt=system_prompt,
        messages_by_iteration=messages_by_iteration,
    )


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------


def timeline_to_dict(timeline: NormalizedTimeline) -> dict[str, Any]:
    """Serialize a NormalizedTimeline to a JSON-safe dict for the API."""
    return {
        "summary": _summary_to_dict(timeline.summary),
        "nodes": [_node_to_dict(n) for n in timeline.nodes],
        "system_prompt": timeline.system_prompt,
        "messages_by_iteration": {str(k): v for k, v in timeline.messages_by_iteration.items()},
    }


def _summary_to_dict(s: RunSummary) -> dict[str, Any]:
    return {
        "run_id": s.run_id,
        "agent_name": s.agent_name,
        "status": s.status,
        "model": s.model,
        "strategy": s.strategy,
        "input_text": s.input_text,
        "answer": s.answer,
        "error": s.error,
        "iteration_count": s.iteration_count,
        "total_input_tokens": s.total_input_tokens,
        "total_output_tokens": s.total_output_tokens,
        "total_cost_usd": s.total_cost_usd,
        "created_at": str(s.created_at) if s.created_at else None,
        "updated_at": str(s.updated_at) if s.updated_at else None,
    }


def _node_to_dict(node: TimelineNode) -> dict[str, Any]:
    """Serialize a timeline node to a JSON-safe dict."""
    if isinstance(node, RunStartedNode):
        return {
            "type": node.type,
            "sequence_index": node.sequence_index,
            "agent_name": node.agent_name,
            "timestamp": str(node.timestamp) if node.timestamp else None,
        }
    if isinstance(node, LLMCallNode):
        return {
            "type": node.type,
            "sequence_index": node.sequence_index,
            "iteration": node.iteration,
            "input_tokens": node.input_tokens,
            "output_tokens": node.output_tokens,
            "cost_usd": node.cost_usd,
            "model": node.model,
            "has_tool_calls": node.has_tool_calls,
            "timestamp": str(node.timestamp) if node.timestamp else None,
            "assistant_text": node.assistant_text,
        }
    if isinstance(node, ToolCallNode):
        return {
            "type": node.type,
            "sequence_index": node.sequence_index,
            "iteration": node.iteration,
            "tool_call_id": node.tool_call_id,
            "tool_name": node.tool_name,
            "target": node.target,
            "success": node.success,
            "duration_ms": node.duration_ms,
            "timestamp": str(node.timestamp) if node.timestamp else None,
            "params": node.params,
            "result": node.result,
            "error_message": node.error_message,
        }
    if isinstance(node, PauseSegmentNode):
        return {
            "type": node.type,
            "sequence_index": node.sequence_index,
            "iteration": node.iteration,
            "pause_status": node.pause_status,
            "pending_tool_calls": [
                {
                    "tool_call_id": tc.tool_call_id,
                    "tool_name": tc.tool_name,
                    "target": tc.target,
                }
                for tc in node.pending_tool_calls
            ],
            "paused_at": str(node.paused_at) if node.paused_at else None,
            "resumed_at": str(node.resumed_at) if node.resumed_at else None,
            "wait_duration_ms": node.wait_duration_ms,
            "submitted_results": [
                {
                    "call_id": r.call_id,
                    "tool_name": r.tool_name,
                    "success": r.success,
                }
                for r in node.submitted_results
            ],
            "user_input": node.user_input,
        }
    if isinstance(node, FinishNode):
        return {
            "type": node.type,
            "sequence_index": node.sequence_index,
            "status": node.status,
            "timestamp": str(node.timestamp) if node.timestamp else None,
        }
    if isinstance(node, ErrorNode):
        return {
            "type": node.type,
            "sequence_index": node.sequence_index,
            "error": node.error,
            "timestamp": str(node.timestamp) if node.timestamp else None,
        }
    if isinstance(node, CancelledNode):
        return {
            "type": node.type,
            "sequence_index": node.sequence_index,
            "timestamp": str(node.timestamp) if node.timestamp else None,
        }
    return {"type": "unknown"}
