"""Shared ReAct pause construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dendrux.types import PauseState, RunResult, ToolCall, ToolTarget, UsageStats

if TYPE_CHECKING:
    from dendrux.loops._budget import BudgetTracker
    from dendrux.loops.react import _RunContext
    from dendrux.types import AgentStep, ImageViewState, Message, RunStatus


def _commentary(steps: list[AgentStep]) -> list[str]:
    """Non-empty assistant text from steps that ended in tool calls, in order."""
    return [
        s.raw_response
        for s in steps
        if isinstance(s.action, ToolCall) and s.raw_response and s.raw_response.strip()
    ]


def _snapshot_usage(usage: UsageStats) -> UsageStats:
    """Create an independent copy of usage stats for pause state."""
    return UsageStats(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost_usd,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cost_source=usage.cost_source,
        usage_reported=usage.usage_reported,
        cost_unknown=usage.cost_unknown,
    )


def _build_pause(
    *,
    agent_name: str,
    pending_calls: list[ToolCall],
    target_lookup: dict[str, ToolTarget],
    history: list[Message],
    steps: list[AgentStep],
    iteration: int,
    usage: UsageStats,
    approval_gates: dict[str, dict[str, str | None]] | None = None,
    budget: BudgetTracker | None = None,
    image_view: ImageViewState | None = None,
) -> PauseState:
    """Build PauseState for client tool, approval or clarification pause."""
    pending_targets = (
        {tc.id: target_lookup.get(tc.name, ToolTarget.SERVER).value for tc in pending_calls}
        if pending_calls
        else {}
    )

    return PauseState(
        agent_name=agent_name,
        pending_tool_calls=pending_calls,
        pending_targets=pending_targets,
        history=list(history),
        steps=list(steps),
        iteration=iteration,
        trace_order_offset=len(history),
        usage=_snapshot_usage(usage),
        approval_gates=approval_gates,
        image_view=image_view,
        **(budget.pause_fields() if budget is not None else {}),
    )


def _pause_result(
    ctx: _RunContext,
    status: RunStatus,
    iteration: int,
    *,
    pending_calls: list[ToolCall],
    approval_gates: dict[str, dict[str, str | None]] | None = None,
    answer: str | None = None,
) -> RunResult:
    pause = _build_pause(
        agent_name=ctx.agent.name,
        pending_calls=pending_calls,
        target_lookup=ctx.lookups.target,
        history=ctx.state.history,
        steps=ctx.state.steps,
        iteration=iteration,
        usage=ctx.state.usage,
        approval_gates=approval_gates,
        budget=ctx.budget,
        image_view=ctx.pause_view(),
    )
    return RunResult(
        run_id=ctx.run_id,
        status=status,
        answer=answer,
        steps=ctx.state.steps,
        commentary=_commentary(ctx.state.steps),
        iteration_count=iteration,
        usage=ctx.state.usage,
        meta={"pause_state": pause, **ctx.meta()},
    )
