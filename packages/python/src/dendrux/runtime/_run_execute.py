"""Execute one fresh non-streaming run in its caller's task."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, overload

import dendrux.runtime.context as _context
import dendrux.strategies.native as _native
from dendrux._sentinel import _OMITTED, _OmittedType, _UnsetType
from dendrux.errors import (
    RunFencedError,
)
from dendrux.runtime import (
    _managed,
    _run_contracts,
    _run_entry,
    _run_errors,
    _run_images,
    _run_observers,
    _run_outcome,
    _run_pending,
    _run_start,
    _run_tools,
)
from dendrux.runtime._entry_checks import resolve_runner_tenant, runner_entry_checks
from dendrux.runtime._managed_cancel import (
    ManagedRootStoppedError,
)
from dendrux.runtime._run_contracts import (
    _UNSET_DEPTH,
)
from dendrux.runtime._tenant_checks import _tenant_scope_chosen
from dendrux.runtime.context import (
    DelegationContext,
    get_store_identity,
    reset_delegation_context,
    resolve_child_tenant,
    set_delegation_context,
    take_managed_invocation,
)
from dendrux.runtime.stores._limits_fit import ensure_agent_limits_fit
from dendrux.skills._session import SkillSession
from dendrux.types import (
    Message,
    Role,
    RunAlreadyActiveError,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

    from dendrux.agent import Agent
    from dendrux.chat import ChatMessage
    from dendrux.context_blocks import (
        ContextBlock,
    )
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier
    from dendrux.media._image import UserContent
    from dendrux.runtime.replay import (
        ReplayLast,
        ReplayPlan,
    )
    from dendrux.runtime.state import StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        RunResult,
    )


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
    result = await _run_once(
        agent,
        provider=provider,
        user_input=user_input,
        history=history,
        context=context,
        activate_skills=activate_skills,
        runs=runs,
        replay_last=replay_last,
        replay_reasoning=replay_reasoning,
        strategy=strategy,
        loop=loop,
        state_store=state_store,
        tenant_id=tenant_id,
        metadata=metadata,
        extra_notifier=extra_notifier,
        max_delegation_depth=max_delegation_depth,
        idempotency_key=idempotency_key,
        output_type=output_type,
        **kwargs,
    )
    # A managed parent exits at its handoff; the driver runs the child in this
    # same task and unwinds terminal outcomes back to the root (0.7).
    return await _managed.drive(result, notifier=extra_notifier)


async def _run_once(
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
    max_delegation_depth: int | None | _UnsetType = _run_contracts._UNSET_DEPTH,
    idempotency_key: str | None = None,
    output_type: type | None = None,
    **kwargs: Any,
) -> RunResult:
    """One run frame: a new run, or a managed child created by its atomic bind."""
    invocation = take_managed_invocation()
    tenant_explicit = _tenant_scope_chosen(tenant_id)
    tenant_id = resolve_runner_tenant(agent, "run", tenant_id, state_store=state_store)
    _run_entry._require_idempotency_store(state_store, idempotency_key)
    await ensure_agent_limits_fit(agent, state_store)
    skill_session = SkillSession.prepare(
        agent, loop=_run_entry._new_run_loop(agent, loop), explicit=activate_skills
    )
    with skill_session.bind():
        # Validate + resolve depth early — before any side effects.
        # _resolve_max_delegation_depth is a pure policy helper; side effect
        # is limited to dedup warning state on parent_ctx.warned_mismatches.
        # A managed child never inherits the caller's delegation context: its
        # parent link, level and depth come from its bound invocation.
        parent_ctx = _managed.parent_context(invocation, _context.get_delegation_context)
        effective_max_depth = _run_entry._resolve_max_delegation_depth(
            max_delegation_depth, parent_ctx
        )
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
        resolved_strategy = strategy or _native.NativeToolCalling()
        resolved_loop = _run_entry._new_run_loop(agent, loop)
        _run_tools._validate_loop_skill_compat(agent, resolved_loop)
        root_limits = await _managed.enter(agent, invocation, state_store, resolved_loop)
        # Validate input + chat history before any side effects. Raises
        # TypeError / ValueError on bad input so no DB row is created.
        user_input, input_items, image_input, normalized_history = _run_entry._new_run_input(
            user_input, history, resolved_loop
        )
        replay_req, replay_runs, replay_omitted = await _run_entry._entry_replay(
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
        # A managed child's id is its parent's bind candidate, minted at the handoff.
        run_id = _managed.new_run_id(invocation)
        from dendrux.runtime._entry_guardrails import _entry_projection

        entry_projection = _entry_projection(agent, run_id)
        _system_prompt = (
            await entry_projection.project(
                Message(Role.SYSTEM, agent.get_system_prompt(loop=resolved_loop)), 0
            )
        ).content
        # Images: ingest before the first durable write (§5.2). Typed errors
        # raise here, before any run row or claim exists.
        image_runtime, user_parts, normalized_history, cached = await _run_images._entry_media(
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
            entry_projection=entry_projection,
        )
        if cached is not None:
            return cached
        # When chat history is provided, the loop expects initial_history to
        # be the COMPLETE message list (loops use initial_history as-is and
        # do not separately append user_input — same contract as resume).
        # The runner will record the new user_input itself (after the recorder
        # is constructed) so seeded prior turns stay out of react_traces.
        # With runs=, the replayed prefix is folded in after expansion.
        (
            seeded_history,
            normalized_history,
            context,
            user_input,
            user_parts,
        ) = await entry_projection.prepare(
            normalized_history, context, user_input, user_parts, skill_session
        )
        seeded_history = _run_entry._entry_seed_history(replay_req, seeded_history)

        # --- Delegation context ---
        parent_run_id, delegation_level = _managed.parent_link(invocation, parent_ctx, state_store)

        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        result: RunResult | None = None
        lifecycle = _run_start._RunLifecycle(
            run_id, state_store, getattr(agent, "_task_manager", None)
        )

        seed_kwargs: dict[str, Any] = {}
        run_guardrails = entry_projection.engine
        try:
            try:
                if state_store is not None:
                    # Create the run record before the loop starts
                    # Merge loop type + depth limit into developer metadata
                    run_meta, idem_fingerprint = _run_entry._new_run_metadata(
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

                    seed_kwargs, run_guardrails = await _run_entry._runtime_seeds(
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
                        replay_omitted=replay_omitted,
                        normalized_history=normalized_history,
                        context=context,
                        user_parts=user_parts,
                        engine=run_guardrails,
                        system_prompt=_system_prompt,
                    )
                    created = await _run_entry._create_projected_run(
                        entry_projection.error is not None,
                        state_store,
                        resolved_loop,
                        run_id,
                        agent.name,
                        input_data=_run_images._input_data(user_input, user_parts),
                        model=provider.model,
                        strategy=type(resolved_strategy).__name__,
                        parent_run_id=parent_run_id,
                        delegation_level=delegation_level,
                        tenant_id=tenant_id,
                        meta=run_meta,
                        idempotency_key=idempotency_key,
                        idempotency_fingerprint=idem_fingerprint,
                        **_run_images._input_edge_kwargs(image_runtime, user_parts),
                        **seed_kwargs,
                        invocation=invocation,
                    )
                    # A child bound by another caller reads as existing_active below.
                    create_result, child_claim = _run_entry._managed_creation(created)

                    # Handle idempotency outcomes
                    if create_result.outcome == "existing_terminal":
                        return await _run_pending._build_cached_result(
                            state_store, create_result.run_id, output_type
                        )
                    if create_result.outcome == "existing_active":
                        raise RunAlreadyActiveError(create_result.run_id, create_result.status)

                    run_id = create_result.run_id
                    state_store = lifecycle._claim(
                        state_store, resolved_loop, create_result, child_claim, root_limits
                    )
                    await _run_entry._verify_projected_creation(
                        entry_projection.error is not None, state_store, run_id
                    )

                trace_link, extra_notifier = await _run_start._start_new_run(
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
                await _run_tools._discover_runtime_tools(
                    agent, lifecycle.recorder, extra_notifier, run_id, resolved_loop, mcp_meta
                )

                replay_plan, seeded_history = await _run_entry._apply_replay(
                    agent,
                    provider,
                    resolved_loop,
                    provider_kwargs,
                    state_store=state_store,
                    run_id=run_id,
                    replay_req=replay_req,
                    replay_runs=replay_runs,
                    replay_omitted=replay_omitted,
                    normalized_history=normalized_history,
                    context=context,
                    user_input=user_input,
                    seeded_history=seeded_history,
                    user_parts=user_parts,
                    system_prompt=_system_prompt,
                )
                await _run_images._register_memory_input(
                    image_runtime, state_store, run_id, user_parts
                )
                image_view = await _run_entry._new_run_image_view(
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
                await _run_entry._record_seeded_user(
                    lifecycle.recorder,
                    extra_notifier,
                    run_id,
                    user_input,
                    seeded_history,
                    context,
                    skill_session,
                )

                result = await _run_entry._run_projected_loop(
                    entry_projection,
                    resolved_loop,
                    user_parts,
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
                    **_run_images._image_loop_kwargs(resolved_loop.run, image_runtime, image_view),
                )

                if replay_plan is not None:
                    result.replay = replay_plan.summary
                _run_observers._attach_mcp_meta(result, mcp_meta)
                _run_pending._ensure_pause_generation(result, provider_kwargs)
                if state_store is not None:
                    result = await _run_outcome._persist_loop_outcome(
                        recorder=lifecycle.recorder,
                        provider_kwargs=provider_kwargs,
                        state_store=state_store,
                        run_id=run_id,
                        result=result,
                    )
                # A parent at a managed handoff exits here without finishing.
                await lifecycle._finish(lifecycle.recorder, extra_notifier, result)

                result.meta = _run_errors._public_meta(result.meta)

                return result

            except (ManagedRootStoppedError, RunFencedError) as stop:
                return await lifecycle._stopped(stop, lifecycle.recorder, extra_notifier)

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
            _run_images._release_memory_edges(image_runtime, state_store, run_id)
            lifecycle._release()
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


run.__module__ = "dendrux.runtime.runner"
