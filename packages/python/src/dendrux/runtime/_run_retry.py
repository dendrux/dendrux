"""Reconstruct a stored source and execute its retry without mutating the source."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NoReturn

import dendrux.runtime.context as _context
import dendrux.runtime.replay as _replay
import dendrux.strategies.native as _native
from dendrux import _codes
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    TERMINAL_STATUSES,
)
from dendrux._meta_keys import (
    LOOP_META_KEY,
    SEED_PREFIX_META_KEY,
    SKILLS_CONTEXT_META_KEY,
)
from dendrux._sentinel import _OMITTED, _OmittedType
from dendrux.context_blocks import (
    ContextBlock,
    render_band,
    wire_user_input,
    wire_user_parts,
)
from dendrux.errors import (
    RunFencedError,
    RunNotFoundError,
)
from dendrux.runtime import (
    _run_contracts,
    _run_entry,
    _run_images,
    _run_observers,
    _run_outcome,
    _run_start,
    _run_tools,
)
from dendrux.runtime._entry_checks import resolve_runner_tenant, runner_entry_checks
from dendrux.runtime._run_errors import (
    logger,
)
from dendrux.runtime._tenant_checks import _tenant_scope_chosen
from dendrux.runtime.context import (
    DelegationContext,
    get_store_identity,
    reset_delegation_context,
    resolve_child_tenant,
    resolve_parent_link,
    set_delegation_context,
)
from dendrux.runtime.replay import (
    PARTS_KEY,
    REPLAY_META_KEY,
    ReplayPlan,
    ReplayRequest,
    _pair_tool_calls,
    current_prefix,
    expand_replay,
    has_images,
    image_slot_count,
    rebuild_messages,
    user_trace_message,
    validate_replay_args,
)
from dendrux.runtime.stores._limits_fit import ensure_agent_limits_fit
from dendrux.skills._session import META_KEY, SkillSession
from dendrux.types import (
    ImageViewState,
    Message,
    Role,
    RunStatus,
    generate_ulid,
)

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import RunRecord, StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        ContentPart,
        Limits,
        RunResult,
    )


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
    initial_history = rebuild_messages(traces, reasoning_for=retry_model)
    _trim_retry_answer(source_run, initial_history)
    initial_history = _pair_tool_calls(initial_history)
    if source_run.storage_format == STORAGE_FORMAT_CHECKPOINTED:
        initial_history = [m for m in initial_history if m.role != Role.SYSTEM]

    if not initial_history:
        raise ValueError(
            f"Run '{source_run.id}' has no persisted traces. Cannot retry without prior context."
        )

    return traces, initial_history


def _trim_retry_answer(source_run: RunRecord, history: list[Message]) -> None:
    if source_run.status == RunStatus.SUCCESS.value and history:
        final = history[-1]
        if final.role == Role.ASSISTANT and not final.tool_calls:
            history.pop()


def _retry_source_input(source_run: RunRecord, traces: Any) -> tuple[Any, Any, Any, Any]:
    users = [t for t in traces if t.role == "user"]
    raw_user = next((t for t in users if not (t.meta or {}).get(SEED_PREFIX_META_KEY)), users[0])
    user_input = (source_run.input_data or {}).get("input", raw_user.content)
    source_parts = _run_images._stored_parts((source_run.input_data or {}).get(PARTS_KEY))

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
    original_run_id: str,
) -> tuple[ReplayRequest | None, list[Any]]:
    """The retried run's ``runs=`` sources, after the blocked-content admission.

    Refuses (R1.8) when the retried run or any of its recorded sources is
    identified as blocked, after their scope/status checks and before any history
    is rebuilt or a run is created. One scoped verdict lookup covers all of them.
    """
    replay_req = _retry_replay_request(source_run, resolved_loop)
    # The verdict and any refusal name the run the caller authorized and asked for.
    retried = replace(source_run, id=original_run_id)
    if replay_req is None:
        from dendrux.runtime._blocked_content import _blocked_sources

        if await _blocked_sources(state_store, [retried], tenant_id=tenant_id):
            _refuse_blocked_retry(original_run_id)
        return None, []
    loaded = await _replay._load_replay_runs(
        state_store, replay_req, tenant_id=tenant_id, limits=limits, extra=(retried,)
    )
    identified = (*loaded.extra_blocked, *loaded.omitted)
    if identified:
        _refuse_blocked_retry(identified[0])
    return replay_req, loaded.runs


def _retry_replay_request(source_run: RunRecord, resolved_loop: Loop) -> ReplayRequest | None:
    replay_req: ReplayRequest | None = None
    source_replay = (source_run.meta or {}).get(REPLAY_META_KEY)
    if isinstance(source_replay, dict) and source_replay.get("runs"):
        replay_req = validate_replay_args(
            list(source_replay["runs"]),
            source_replay.get("replay_last", 0),
            bool(source_replay.get("replay_reasoning", True)),
            loop=resolved_loop,
            context=None,
        )
    return replay_req


def _refuse_blocked_retry(run_id: str) -> NoReturn:
    from dendrux.errors import ReplayRunError

    _codes.log_code(logger, _codes.BLOCKED_CONTENT, run_id=run_id)
    raise ReplayRunError(
        run_id,
        "blocked_content",
        "Source history contains blocked content; start a new run with clean input.",
    )


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
    if parts is not None or has_images(history) or _run_images._replay_has_images(replay):
        _run_images._require_image_loop(loop)
    elif not _run_contracts._accepts_kwarg(loop.run, "image_runtime"):
        return None
    return _run_images._image_runtime(agent, store, tenant, run_id)


async def _retry_edges(
    runtime: ImageRuntime | None, parts: tuple[ContentPart, ...] | None
) -> dict[str, Any]:
    if parts is None or runtime is None:
        return {}
    live = await _run_images._live_edges(
        runtime, _run_images._input_edge_kwargs(runtime, parts).get("blob_refs", [])
    )
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
    await _replay.record_run_prefix(
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
    image_view = await _run_images._start_image_view(
        image_runtime,
        store,
        run_id,
        seeded=history,
        sources=[
            *_run_images._replay_sources(replay_runs),
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
    tenant_id = resolve_child_tenant(_context.get_delegation_context(), tenant_id)
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
    skill_session = SkillSession.prepare(
        agent, loop=_run_entry._new_run_loop(agent, loop), explicit=explicit
    )
    with skill_session.bind():
        await _check_retry_source(state_store, source_run, original_run_id)
        resolved_loop = _run_entry._new_run_loop(agent, loop)
        _run_tools._validate_loop_skill_compat(agent, resolved_loop)
        # 3. Blocked-content admission for the retried run and its runs= sources
        #    (R1.8), then the same runs, same knobs, before any history is rebuilt.
        replay_req, replay_runs = await _retry_replay(
            source_run, resolved_loop, state_store, tenant_id, agent._limits, original_run_id
        )
        # 3a. Read traces and reconstruct approximate history. Reasoning replay
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
        # 5. Create the retry run
        resolved_strategy = strategy or _native.NativeToolCalling()
        provider_kwargs = dict(kwargs)
        run_id = generate_ulid()
        parent_ctx = _context.get_delegation_context()
        parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)
        effective_max_depth = _run_entry._resolve_max_delegation_depth(
            agent.max_delegation_depth, parent_ctx
        )

        _system_prompt = agent.get_system_prompt(loop=resolved_loop)

        run_meta, _ = _run_entry._new_run_metadata(
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
        lifecycle = _run_start._RunLifecycle(
            run_id, state_store, getattr(agent, "_task_manager", None)
        )

        seed_kwargs: dict[str, Any] = {}
        run_guardrails = None
        try:
            try:
                seed_kwargs, run_guardrails = await _run_entry._runtime_seeds(
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
                create_result = await _run_entry._create_runtime_run(
                    state_store,
                    resolved_loop,
                    run_id,
                    agent.name,
                    input_data=_run_images._input_data(user_input, source_parts),
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

                trace_link, extra_notifier = await _run_start._start_new_run(
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

                await _run_tools._discover_runtime_tools(
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
                    await _run_images._record_user_trace(
                        lifecycle.recorder, extra_notifier, run_id, trace_user
                    )

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
                    **_run_images._image_loop_kwargs(resolved_loop.run, image_runtime, image_view),
                )

                if replay_plan is not None:
                    result.replay = replay_plan.summary
                _run_observers._attach_mcp_meta(result, mcp_meta)
                result = await _run_outcome._persist_loop_outcome(
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
            _run_images._release_memory_edges(image_runtime, state_store, run_id)
            lifecycle._release()
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


retry.__module__ = "dendrux.runtime.runner"
