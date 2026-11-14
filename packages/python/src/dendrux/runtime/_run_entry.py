"""Validate and prepare durable fresh-run inputs, replay seeds and creation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

import dendrux.chat as _chat
import dendrux.runtime.replay as _replay
import dendrux.runtime.state as _state
from dendrux import _codes
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    FailureReason,
)
from dendrux._meta_keys import (
    BLOCKED_META_KEY,
    LOOP_META_KEY,
    MAX_DELEGATION_DEPTH_META_KEY,
    OUTPUT_TYPE_META_KEY,
    RETRY_OF_META_KEY,
)
from dendrux._sentinel import _UnsetType
from dendrux.chat import (
    history_has_images,
    normalize_chat_history,
    validate_chat_history,
)
from dendrux.context_blocks import (
    ContextBlock,
    fold_context,
    render_band,
    validate_context,
)
from dendrux.loops.react import ReActLoop
from dendrux.runtime import (
    _run_events,
    _run_images,
)
from dendrux.runtime._run_errors import logger
from dendrux.runtime.replay import (
    LoadedReplay,
    ReplayLast,
    ReplayPlan,
    ReplayRequest,
    build_seed,
    current_prefix,
    expand_replay,
    image_slot_count,
    user_trace_message,
    validate_replay_args,
)
from dendrux.runtime.state import (
    ClaimResult,
    unclaimed_store,
)
from dendrux.runtime.submit import (
    GOVERNANCE_KEY,
    governance_snapshot,
)
from dendrux.skills._session import META_KEY, SkillSession
from dendrux.skills._session import current as current_skill_session
from dendrux.types import (
    GovernanceEventType,
    ImageViewState,
    Message,
    Role,
    RunStatus,
    compute_idempotency_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

    from dendrux.agent import Agent
    from dendrux.chat import ChatMessage
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.media._image import UserContent
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.context import DelegationContext
    from dendrux.runtime.state import StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        ContentPart,
        CreateRunResult,
        Limits,
        RunResult,
    )


__all__ = ["current_skill_session"]

_DEFAULT_RECOVERY_ITERATIONS = 10


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
    text, items = _chat.split_user_content(user_input)
    validate_chat_history(history)
    image_input = bool(items) or history_has_images(history)
    if image_input:
        _run_images._require_image_loop(loop)
    return text, items, image_input, _text_history(history, image_input)


def _text_history(history: list[ChatMessage] | None, images: bool) -> list[Message]:
    return [] if images else normalize_chat_history(history)


def _entry_seed_history(
    replay: ReplayRequest | None, history: list[Message] | None
) -> list[Message] | None:
    return history if replay is None else None


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
        await _run_images._record_user_trace(recorder, notifier, run_id, message)


async def _new_run_image_view(
    runtime: ImageRuntime | None,
    store: StateStore | None,
    run_id: str,
    history: list[Message] | None,
    replay_runs: list[Any],
    replay_plan: ReplayPlan | None,
    parts: tuple[ContentPart, ...] | None,
) -> ImageViewState | None:
    return await _run_images._start_image_view(
        runtime,
        store,
        run_id,
        seeded=(history or [])[:-1],
        sources=_run_images._replay_sources(replay_runs),
        single_full=_run_images._single_full_source(replay_plan),
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
    result = _base_run_metadata(agent, loop, session, metadata, depth)
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


def _base_run_metadata(
    agent: Agent,
    loop: Loop,
    session: SkillSession,
    metadata: dict[str, Any] | None,
    depth: int | None,
) -> dict[str, Any]:
    result = {key: value for key, value in (metadata or {}).items() if key != BLOCKED_META_KEY}
    result[LOOP_META_KEY] = type(loop).__name__
    result[GOVERNANCE_KEY] = governance_snapshot(agent)
    if session.configured:
        result[META_KEY] = session.manifest()
    if depth is not None:
        result[MAX_DELEGATION_DEPTH_META_KEY] = depth
    return result


async def _load_new_run_replay(
    store: StateStore,
    request: ReplayRequest,
    tenant_id: str | None,
    loop: Loop,
    *,
    limits: Limits,
    method: str = "run",
) -> LoadedReplay:
    loaded = await _replay._load_replay_runs(store, request, tenant_id=tenant_id, limits=limits)
    for omitted_id in loaded.omitted:
        _codes.log_code(logger, _codes.BLOCKED_CONTENT, run_id=omitted_id)
    if _run_images._replay_has_images(loaded.runs):
        _run_images._require_image_loop(loop, method)
    return loaded


async def _entry_replay(
    store: StateStore | None,
    tenant_id: str | None,
    runs: list[str] | None,
    last: ReplayLast,
    reasoning: bool,
    loop: Loop,
    context: Any,
    limits: Limits,
) -> tuple[ReplayRequest | None, list[Any], tuple[str, ...]]:
    request = validate_replay_args(runs, last, reasoning, loop=loop, context=context)
    validate_context(context)
    if request is None:
        return None, [], ()
    if store is None:
        raise ValueError(
            "runs= requires persistence (database_url or state_store): "
            "replay reads the earlier runs' traces."
        )
    loaded = await _load_new_run_replay(store, request, tenant_id, loop, limits=limits)
    return request, loaded.runs, loaded.omitted


def _recovery_loop_name(loop: Loop) -> str | None:
    from dendrux.loops.react import ReActLoop
    from dendrux.loops.single import SingleCall

    if isinstance(loop, ReActLoop):
        return "ReActLoop"
    if isinstance(loop, SingleCall):
        return "SingleCall"
    return None


async def _create_projected_run(
    blocked: bool,
    store: StateStore,
    loop: Loop,
    run_id: str,
    agent_name: str,
    *,
    invocation: Any = None,
    **kwargs: Any,
) -> Any:
    from dendrux._meta_keys import BLOCKED_META_KEY
    from dendrux.runtime._blocked_content import _blocked_write, _native_blocked_store

    native = _native_blocked_store(store) is not None
    if blocked and not native:
        kwargs["meta"] = {**(kwargs.get("meta") or {}), BLOCKED_META_KEY: True}
    scope = _blocked_write(store, run_id) if blocked and native else nullcontext()
    with scope:
        if invocation is not None:
            return await _bind_runtime_run(store, loop, run_id, agent_name, invocation, **kwargs)
        return await _create_runtime_run(store, loop, run_id, agent_name, **kwargs)


def _managed_creation(created: Any) -> tuple[CreateRunResult, ClaimResult | None]:
    """A bind result as a creation outcome plus the confirmed creator's token."""
    from dendrux.runtime.state import BindManagedChildResult
    from dendrux.types import CreateRunResult

    if not isinstance(created, BindManagedChildResult):
        return created, None
    outcome: Literal["created", "existing_active"] = (
        "created" if created.child_claim is not None else "existing_active"
    )
    return (
        CreateRunResult(
            created.child_run_id, outcome, RunStatus.RUNNING, STORAGE_FORMAT_CHECKPOINTED
        ),
        created.child_claim,
    )


async def _bind_runtime_run(
    store: StateStore, loop: Loop, run_id: str, agent_name: str, invocation: Any, **kwargs: Any
) -> Any:
    """Create a managed child through the atomic bind, with ordinary ownership handoff."""
    from dendrux.runtime import _managed
    from dendrux.runtime.durability import _ownership_handoff
    from dendrux.runtime.state import _initial_loop_state

    async def cancel_bound(bound: Any) -> None:
        if bound.child_claim is not None:
            await _run_events._persist_task_cancellation(
                _state.claimed_store(store, bound.child_run_id, bound.child_claim),
                bound.child_run_id,
            )

    return await _ownership_handoff(
        store,
        lambda: _managed.bind_child(
            invocation,
            unclaimed_store(store),
            run_id,
            agent_name,
            loop_state=_initial_loop_state(_recovery_loop_name(loop)),
            **kwargs,
        ),
        cancel_bound,
        run_id=run_id,
    )


async def _verify_projected_creation(blocked: bool, store: StateStore, run_id: str) -> None:
    from dendrux.runtime._blocked_content import (
        _native_blocked_store,
        _require_block_mark,
    )

    if blocked and _native_blocked_store(store) is None:
        await _require_block_mark(store, run_id)


async def _run_projected_loop(
    projection: Any, loop: Loop, parts: tuple[ContentPart, ...] | None, **kwargs: Any
) -> RunResult:
    if projection.error is not None:
        return await _blocked_entry_result(
            projection,
            kwargs["recorder"],
            kwargs["notifier"],
            kwargs["run_id"],
            kwargs["user_input"],
            parts,
            kwargs["initial_history"],
        )
    with projection.bind(kwargs["run_id"]):
        return await loop.run(**kwargs)


async def _blocked_entry_result(
    projection: Any,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    user_input: str,
    parts: tuple[ContentPart, ...] | None,
    history: list[Message] | None,
) -> RunResult:
    from dendrux.loops._helpers import (
        guardrail_meta,
        notify_governance,
        record_governance,
    )
    from dendrux.loops.react import _blocked_data
    from dendrux.types import RunResult

    if history is None:
        message = Message(Role.USER, user_input, parts=parts)
        await _run_images._record_user_trace(recorder, notifier, run_id, message)
    data = _blocked_data("incoming", projection.error, projection.findings, projection.location)
    await record_governance(recorder, run_id, GovernanceEventType.GUARDRAIL_BLOCKED.value, 1, data)
    warnings: list[str] = []
    await notify_governance(
        notifier, run_id, GovernanceEventType.GUARDRAIL_BLOCKED.value, 1, data, warnings=warnings
    )
    return RunResult(
        run_id=run_id,
        status=RunStatus.ERROR,
        error=projection.error,
        iteration_count=1,
        meta={
            **guardrail_meta(projection.engine, warnings),
            "failure_reason": FailureReason.GUARDRAIL_BLOCKED.value,
        },
    )


async def _create_runtime_run(
    store: StateStore, loop: Loop, run_id: str, agent_name: str, **kwargs: Any
) -> CreateRunResult:
    from dendrux.runtime.durability import _ownership_handoff
    from dendrux.runtime.state import _creation_loop_scope

    async def cancel_created(created: CreateRunResult) -> None:
        if created.outcome == "created":
            await _run_events._persist_task_cancellation(
                _state.claimed_store(store, created.run_id, 0), created.run_id
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
    replay_omitted: Sequence[str] = (),
    normalized_history: list[Message] | None = None,
    context: list[ContextBlock] | None = None,
    user_parts: tuple[ContentPart, ...] | None = None,
    engine: Any = None,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any], Any]:
    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.persistence import seed_trace_kwargs

    if not capabilities(store).checkpointing:
        return {}, engine
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
            omitted=replay_omitted,
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
        Message(
            role=Role.SYSTEM,
            content=(
                agent.get_system_prompt(loop=loop) if system_prompt is None else system_prompt
            ),
        ),
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

    if engine is None and agent.guardrails:
        engine = GuardrailEngine(agent.guardrails, run_id=run_id)
    messages, own_input = await _project_seed_messages(engine, messages, own_input)
    return {
        **seed_trace_kwargs(
            store, messages, tenant_scope=tenant_scope_for(tenant_id), own_input=own_input
        ),
        "pii_mapping": engine.get_pii_mapping() if engine else None,
    }, engine


async def _project_seed_messages(
    engine: Any, messages: list[Message], own_input: Message | None
) -> tuple[list[Message], Message | None]:
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
    return messages, own_input


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
    replay_omitted: Sequence[str] = (),
    user_parts: tuple[ContentPart, ...] | None = None,
    system_prompt: str | None = None,
) -> tuple[ReplayPlan | None, list[Message] | None]:
    """Expand ``runs=`` into the seeded history and record the prefix descriptor.

    Runs after MCP discovery so the tools hash covers the full tool set.
    Without persistence nothing is recorded and nothing is replayed.
    """
    if state_store is None:
        return None, seeded_history
    current = current_prefix(agent, provider, loop, provider_kwargs)
    if system_prompt is not None:
        from dendrux.runtime.replay import _hash

        current = replace(current, system_hash=_hash(system_prompt))
    plan: ReplayPlan | None = None
    if replay_req is not None:
        plan = expand_replay(
            replay_runs,
            replay_req,
            current=current,
            guardrails=bool(agent.guardrails),
            omitted=replay_omitted,
        )
        seeded_history = fold_context(
            [*plan.messages, *normalized_history],
            context,
            user_input,
            session=current_skill_session(agent),
            user_parts=user_parts,
        )
    await _replay.record_run_prefix(
        state_store,
        run_id,
        current=current,
        seed=build_seed(plan, normalized_history),
        plan=plan,
        seeded_image_slots=image_slot_count((seeded_history or [])[:-1]),
    )
    return plan, seeded_history
