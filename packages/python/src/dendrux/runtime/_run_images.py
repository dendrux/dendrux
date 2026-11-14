"""Ingest and project run images while owning their memory/blob edges."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

from dendrux._lifecycle import (
    RESUME_INPUT_STATUS_VALUES,
    RESUME_RESULTS_STATUS_VALUES,
)
from dendrux.chat import (
    history_has_images,
    normalize_chat_history,
)
from dendrux.errors import (
    BlobNotFoundError,
    InvalidImageError,
    safe_exception_info,
)
from dendrux.loops._helpers import (
    notify_message,
    record_message,
)
from dendrux.loops.react import ReActLoop
from dendrux.runtime import (
    _run_contracts,
    _run_pending,
)
from dendrux.runtime._run_errors import (
    logger,
)
from dendrux.runtime.replay import (
    PARTS_KEY,
    ReplayPlan,
    ReplayRequest,
    has_images,
    image_slot_count,
    initial_image_view,
    load_image_sources,
    parts_to_dicts,
)
from dendrux.runtime.state import (
    unclaimed_store,
)
from dendrux.types import (
    IdempotencyConflictError,
    ImagePart,
    ImageRef,
    ImageViewState,
    Message,
    PauseState,
    Role,
    RunAlreadyActiveError,
    RunStatus,
    TextPart,
    compute_idempotency_fingerprint,
    content_part_from_dict,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from dendrux.agent import Agent
    from dendrux.chat import ChatMessage
    from dendrux.context_blocks import ContextBlock
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.media._catalog import BlobEdge
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime._entry_guardrails import _EntryProjection
    from dendrux.runtime.state import StateStore
    from dendrux.skills._session import SkillSession
    from dendrux.types import (
        ContentPart,
        RunResult,
        ToolResult,
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
    if _run_contracts._accepts_kwarg(init, "tenant_scope"):
        out["tenant_scope"] = tenant_scope_for(tenant_id)
    if _run_contracts._accepts_kwarg(init, "provider"):
        out["provider"] = provider
    return out


def _image_loop_kwargs(
    loop_method: Any, runtime: ImageRuntime | None, state: ImageViewState | None
) -> dict[str, Any]:
    """``image_runtime=`` / ``image_view_state=`` for a loop method that accepts them."""
    out: dict[str, Any] = {}
    if runtime is not None and _run_contracts._accepts_kwarg(loop_method, "image_runtime"):
        out["image_runtime"] = runtime
        if state is not None and _run_contracts._accepts_kwarg(loop_method, "image_view_state"):
            out["image_view_state"] = state
    return out


def _require_image_loop(loop: Loop, method: str = "run") -> None:
    """A loop that cannot take ``image_runtime`` cannot run image input."""
    if not _run_contracts._accepts_kwarg(getattr(loop, method, None), "image_runtime"):
        raise TypeError(
            f"{type(loop).__name__}.{method}() does not accept image_runtime, so it "
            "cannot run image input; use ReActLoop or SingleCall, or add the parameter."
        )


def _tool_image_kwargs(fn: Any, runtime: ImageRuntime | None) -> dict[str, Any]:
    if runtime is None or not _run_contracts._accepts_kwarg(fn, "image_runtime"):
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
    user_msg = await record_message(recorder, run_id, user_msg, 0)
    await notify_message(notifier, run_id, user_msg, 0)


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
    *,
    entry_projection: _EntryProjection | None = None,
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
                entry_projection=entry_projection,
                skill_session=skill_session,
            )
            if hit is None:
                raise
            if hit.outcome == "existing_terminal":
                assert state_store is not None  # a hit implies a store
                return (
                    image_runtime,
                    user_parts,
                    normalized_history,
                    await _run_pending._build_cached_result(state_store, hit.run_id, output_type),
                )
            raise RunAlreadyActiveError(hit.run_id, hit.status) from None
        image_runtime, user_parts, normalized_history = entry
    elif _run_contracts._accepts_kwarg(resolved_loop.run, "image_runtime"):
        image_runtime = _image_runtime(agent, state_store, tenant_id, run_id)

    return image_runtime, user_parts, normalized_history, None


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
    entry_projection: _EntryProjection | None = None,
    skill_session: SkillSession | None = None,
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
    candidate = _image_retry_parts(items, stored, conflict)
    normalized_history = normalize_chat_history(history)
    if entry_projection is not None:
        (
            _,
            normalized_history,
            context,
            user_input,
            projected_parts,
        ) = await entry_projection.prepare(
            normalized_history, context, user_input, tuple(candidate), skill_session
        )
        if projected_parts is None:
            raise conflict
        candidate = list(projected_parts)
    if any(
        isinstance(part, TextPart) and part != previous
        for part, previous in zip(candidate, stored, strict=True)
    ):
        raise conflict
    fingerprint = compute_idempotency_fingerprint(
        agent_name,
        user_input,
        output_type_name=(
            f"{output_type.__module__}.{output_type.__qualname__}" if output_type else None
        ),
        history=normalized_history or None,
        context=context or None,
        replay=replay,
        skills=skills,
        user_input_parts=candidate,
    )
    return await check(idempotency_key, fingerprint, tenant_id=tenant_id)


def _image_retry_parts(
    items: tuple[Any, ...], stored: tuple[ContentPart, ...], conflict: IdempotencyConflictError
) -> list[ContentPart]:
    candidate: list[ContentPart] = []
    for item, part in zip(items, stored, strict=True):
        if isinstance(item, str):
            if not isinstance(part, TextPart):
                raise conflict
            candidate.append(TextPart(item))
            continue
        if not isinstance(part, ImagePart):
            raise conflict
        ref = getattr(item, "ref", item)
        if isinstance(ref, ImageRef) and ref.sha256 != part.ref.sha256:
            raise conflict
        candidate.append(ImagePart(part.ref, detail=getattr(item, "detail", None)))
    return candidate


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
        await _run_pending._raise_resume_claim_failure(state_store, run_id, allowed[0])
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


_EntryImages.__module__ = "dendrux.runtime.runner"
ingest_submitted_tool_images.__module__ = "dendrux.runtime.runner"
ingest_submitted_user_parts.__module__ = "dendrux.runtime.runner"
