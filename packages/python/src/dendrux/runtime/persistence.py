"""PersistenceRecorder — authoritative evidence recording.

Internal module. Implements LoopRecorder by writing each event to the
StateStore. The runner creates this when persistence is active.

Write classification:
  - Fail-closed (retry with backoff, then propagate): save_trace, save_tool_call, save_run_event
  - Best-effort (bounded transient retry, then swallowed): save_usage, save_llm_interaction
  - Best-effort (exceptions swallowed): touch_progress

Images (0.4.0): parts messages persist ``meta["dendrux.parts"]`` on traces
and ``parts`` in ``semantic_request.messages``; tool results with images
persist ``meta["dendrux.images"]``. Every image a record references gets a
blob edge written in the record's transaction, scoped to ``tenant_scope``.
Text-only records are byte-identical to v0.3.0 (no new keys, no new kwargs).

Lifecycle hooks (``on_run_started``, ``on_run_finished``, ``on_run_failed``,
``on_llm_call_started``, ``on_llm_call_failed``, ``on_tool_started``) are
inherited as no-ops from :class:`BaseRecorder`. The runner writes the
corresponding ``run.started`` / ``run.error`` / etc events directly via
``_emit_event`` to avoid duplicate DB writes — this transitional state
keeps the audit trail unchanged while still exposing the hooks on the
LoopRecorder protocol so third-party recorders can subscribe.
"""

from __future__ import annotations

import inspect
import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux.errors import ImagePersistenceUnsupportedError, RunFencedError, safe_exception_info
from dendrux.loops.base import BaseRecorder
from dendrux.runtime._blob_catalog import edges_for_parts, scrub_inline_images
from dendrux.runtime._capabilities import capabilities
from dendrux.runtime.durability import retry_critical, retry_transient_db
from dendrux.types import (
    ImagePart,
    ImageRef,
    content_part_to_dict,
    generate_ulid,
    tool_call_parse_meta,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractAsyncContextManager

    from dendrux.media import BlobEdge
    from dendrux.runtime.state import StateStore
    from dendrux.types import LLMResponse, Message, ToolCall, ToolDef, ToolResult, ToolTarget

PARTS_META_KEY = "dendrux.parts"
IMAGES_META_KEY = "dendrux.images"
IMAGE_NOTES_META_KEY = "dendrux.image_notes"

logger = logging.getLogger(__name__)


def _tool_call_evidence(tc: ToolCall) -> dict[str, Any]:
    """Persisted shape of a tool call; never includes the raw argument text."""
    return {
        "id": tc.id,
        "provider_tool_call_id": tc.provider_tool_call_id,
        "name": tc.name,
        "params": tc.params,
        **tool_call_parse_meta(tc),
    }


def _serialize_message(m: Message) -> dict[str, Any]:
    """Full-fidelity serialization of a Message for the evidence layer."""
    d: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.tool_calls is not None:
        d["tool_calls"] = [_tool_call_evidence(tc) for tc in m.tool_calls]
    if m.call_id is not None:
        d["call_id"] = m.call_id
    if m.name is not None:
        d["name"] = m.name
    if m.meta:
        d["meta"] = m.meta
    if m.parts is not None:
        d["parts"] = [content_part_to_dict(p) for p in m.parts]
    return d


def image_dict(image: object) -> dict[str, Any]:
    """Persisted descriptor of an ingested tool-result image (``ImagePart`` / ``ImageRef``)."""
    if isinstance(image, ImagePart):
        return content_part_to_dict(image)
    if isinstance(image, ImageRef):
        return image.to_dict()
    raise TypeError(
        f"tool result images must be ingested before they are recorded, got {type(image).__name__}"
    )


# image_capability_key() api_surface -> estimate_image_tokens provider family.
_SURFACE_FAMILY = {
    "anthropic": "anthropic",
    "chat": "openai",
    "responses": "openai",
    "openrouter": "openrouter",
}


def provider_family(provider: Any, model: str | None) -> str | None:
    """Token-estimate family of ``provider`` (from its image capability key), or None.

    Uses ``provider.image_capability_key(model)[0]`` (the API surface the
    built-in adapters report); a custom provider without a known surface
    gives None, so estimates stay unknown rather than guessed.
    """
    key_fn = getattr(provider, "image_capability_key", None)
    if key_fn is None or not model:
        return None
    try:
        surface = key_fn(model)[0]
    except Exception:
        return None
    return _SURFACE_FAMILY.get(surface) if isinstance(surface, str) else None


def image_stats(
    messages: Sequence[Message], *, provider: Any = None, model: str | None
) -> dict[str, Any] | None:
    """``{"count", "bytes", "estimated_tokens"}`` over the visible images of a request.

    Visible images are the ``ImagePart``s left in the request view (placeholders
    are text). ``estimated_tokens`` is None when any estimate is unknown.
    Returns None for a request without visible images.
    """
    from dendrux.media import estimate_image_tokens

    parts = [
        p for m in messages if m.parts is not None for p in m.parts if isinstance(p, ImagePart)
    ]
    if not parts:
        return None
    family = provider_family(provider, model)
    total: int | None = 0
    for p in parts:
        est = (
            estimate_image_tokens(p.ref, provider_family=family, model=model, detail=p.detail)
            if family is not None and model
            else None
        )
        total = None if est is None or total is None else total + est
    return {
        "count": len(parts),
        "bytes": sum(p.ref.byte_length for p in parts),
        "estimated_tokens": total,
    }


def message_trace(
    message: Message,
    order: int,
    *,
    kind: str = "turn",
    tenant_scope: str | None = None,
) -> Any:
    """A lossless conversation envelope, with content stored only once."""
    from dendrux.runtime.state import TraceWrite
    from dendrux.types import _message_to_dict

    envelope = _message_to_dict(message)
    envelope.pop("content", None)
    meta = dict(message.meta)
    meta["dendrux.message"] = envelope
    if message.tool_calls:
        meta["tool_calls"] = [_tool_call_evidence(tc) for tc in message.tool_calls]
    if message.reasoning_blocks:
        meta["reasoning_blocks"] = message.reasoning_blocks
    if message.call_id:
        meta["call_id"] = message.call_id
    if message.name:
        meta["tool_name"] = message.name
    if message.parts is not None:
        meta[PARTS_META_KEY] = [content_part_to_dict(p) for p in message.parts]
    edges = (
        edges_for_parts(tenant_scope, "trace", str(order), meta.get(PARTS_META_KEY) or [])
        if tenant_scope
        else []
    )
    return TraceWrite(
        generate_ulid(),
        message.role.value,
        message.content,
        order,
        trace_kind=kind,
        meta=meta,
        blob_refs=edges or None,
        blob_refs_scope=tenant_scope if edges else None,
    )


def seed_trace_kwargs(
    store: StateStore,
    messages: list[Message],
    *,
    tenant_scope: str | None = None,
    own_input: Message | None = None,
) -> dict[str, Any]:
    """Only declared checkpoint stores receive the additive create argument."""
    if not capabilities(store).checkpointing:
        return {}
    traces = [
        message_trace(m, i, kind="seed", tenant_scope=tenant_scope) for i, m in enumerate(messages)
    ]
    own_index = max((i for i, m in enumerate(messages) if m.role.value == "user"), default=-1)
    for i, trace in enumerate(traces):
        trace.meta["dendrux.seed_prefix"] = i < own_index
        if i == own_index and own_input is not None:
            traces[i] = replace(
                trace, content=own_input.content, meta={**trace.meta, **own_input.meta}
            )
    return {"seed_traces": traces}


def _view_identity(entry: dict[str, Any]) -> str:
    meta = entry.get("meta") or {}
    return json.dumps(
        {
            "role": entry["role"],
            "content": meta.get("wire_content", entry["content"]),
            "message": meta.get("dendrux.message", meta),
        },
        sort_keys=True,
    )


class PersistenceRecorder(BaseRecorder):
    """Authoritative evidence recorder — writes loop events to StateStore.

    Fail-closed writes (exceptions propagate to caller):
      - save_trace: what the agent saw and said
      - save_tool_call: proof of side effects
      - save_run_event: lifecycle audit trail

    Best-effort writes (exceptions swallowed):
      - save_usage: cost tracking
      - save_llm_interaction: full forensics
      - touch_progress: operational freshness for sweep

    The loop calls record_* helpers which let exceptions from this class
    propagate. If a fail-closed write fails, the run stops.

    ``tenant_scope`` (0.4.0) scopes every blob edge this recorder writes;
    ``None`` is valid only for runs without images. ``provider`` (the run's
    LLM provider) lets ``semantic_request.image_stats`` estimate image tokens
    from its ``image_capability_key``; without it the estimate is ``None``.
    """

    def __init__(
        self,
        state_store: StateStore,
        run_id: str,
        *,
        model: str | None = None,
        provider_name: str | None = None,
        target_lookup: dict[str, ToolTarget] | None = None,
        initial_order_index: int = 0,
        tenant_scope: str | None = None,
        provider: Any | None = None,
        seed_traces: Any = None,
    ) -> None:
        self._store = state_store
        self._tenant_scope = tenant_scope
        self._provider = provider
        self._run_id = run_id
        self._model = model
        self._provider_name = provider_name
        self._target_lookup = target_lookup or {}
        self._format2 = (
            getattr(state_store, "storage_format", 1) == 2
            and capabilities(state_store).checkpointing
        )
        self._order_index = (
            len(seed_traces) if self._format2 and seed_traces else initial_order_index
        )
        self._loop_source: Any = None
        self._loop_state: dict[str, Any] = deepcopy(getattr(state_store, "loop_state", None) or {})
        self._known_history = (
            [{"role": t.role, "content": t.content, "meta": t.meta} for t in seed_traces or []]
            if self._format2
            else []
        )
        self._state_payloads: dict[str, str] = {}
        self._seed_input_pending = any(t.role == "user" for t in seed_traces or [])
        self._durable_state = deepcopy(self._loop_state)
        self._started_calls: set[str] = set()
        self._completed_observations: set[str] = set()
        self._tool_idempotent: dict[str, bool] = {}
        self._result_formatter: Any = None
        self._consumed_user_count = 0
        self._final_trace: Any = None

    async def restore_history(self, *, limits: Any) -> None:
        if not self._format2:
            return
        from dendrux.runtime.replay import conversation_records

        records = await conversation_records(self._store, self._run_id, limits=limits)
        self._known_history = [
            {"role": t.role, "content": t.content, "meta": t.meta} for t in records
        ]

    def bind_loop(self, source: Any, budget: Any) -> None:
        """Bind live loop data without making loops depend on runtime internals."""
        if not self._format2:
            return
        self._loop_source = source
        live = source()
        self._tool_idempotent = live.get("idempotent", {})
        self._result_formatter = live.get("format_result")
        for target, saved_value in (
            (live["tool_counts"], self._loop_state.get("tool_counts")),
            (live["pii_counters"], self._loop_state.get("pii", {}).get("counters")),
        ):
            if saved_value is not None:
                target.clear()
                target.update(saved_value)
        saved = self._loop_state.get("budget")
        if saved:
            from dendrux.loops._budget import BudgetState

            resumed = budget.snapshot()
            restored = BudgetState(**saved)
            if resumed.gate:
                restored = replace(
                    restored,
                    gate=resumed.gate,
                    accepted_unpriced_baseline=max(
                        restored.accepted_unpriced_baseline,
                        resumed.accepted_unpriced_baseline,
                    ),
                )
            budget.restore(restored)

    async def _state_ref(self, kind: str, payload: dict[str, Any]) -> str | None:
        from dendrux.runtime.state import TraceWrite

        content = json.dumps(
            {"kind": kind, "v": 1, **payload}, ensure_ascii=False, separators=(",", ":")
        )
        field = "session_ref" if kind == "skills_session" else kind + "_ref"
        refs = (
            self._loop_state.setdefault("skills", {})
            if kind == "skills_session"
            else self._loop_state
        )
        if self._state_payloads.get(kind) == content:
            return refs.get(field)
        trace = TraceWrite(
            generate_ulid(),
            "state",
            content,
            self._order_index,
            trace_kind="state",
            meta={"state_kind": kind},
        )
        self._order_index += 1
        refs[field] = trace.trace_id
        self._state_payloads[kind] = content

        async def write_state() -> None:
            await self._store.save_trace(
                self._run_id,
                "state",
                content,
                order_index=trace.order_index,
                meta=trace.meta,
                trace_id=trace.trace_id,
                trace_kind="state",
            )

        await retry_critical(write_state, label="state_trace", run_id=self._run_id)
        return trace.trace_id

    async def _snapshot(self, iteration: int) -> None:
        from dendrux.skills._session import current
        from dendrux.types import _usage_to_dict

        if self._loop_source is None:
            return
        data = self._loop_source()
        budget = asdict(data["budget"].snapshot())
        usage = _usage_to_dict(data["usage"])
        usage.update(
            priced_subtotal_usd=budget["known_cost_usd"], unpriced_calls=budget["unpriced_calls"]
        )
        self._loop_state.update(
            v=1,
            loop="react",
            iteration=iteration,
            steps_count=len(data["steps"]),
            usage=usage,
            budget=budget,
            tool_counts={
                k: v for k, v in data["tool_counts"].items() if k in data["declared_tools"]
            },
            pii={
                "mapping_version": len(data["pii_mapping"] or {}),
                "counters": dict(data["pii_counters"]),
            },
            provider_kwargs=dict(data["provider_kwargs"]),
        )
        for key, default in (
            ("pending", []),
            ("submission", None),
            ("image_view_ref", None),
            ("clarification_ref", None),
            ("phase", "model_next"),
        ):
            self._loop_state.setdefault(key, default)
        self._loop_state.setdefault("skills", {"session_ref": None, "pins": {}})
        session = current()
        if session is not None and session.configured:
            self._loop_state["skills"]["pins"] = {
                s.name: s.digest for s in session.available.values()
            }
            await self._state_ref("skills_session", session.snapshot())
        image_view = data["image_view"].to_dict()
        if data["has_images"] or self._loop_state.get("image_view_ref"):
            await self._state_ref("image_view", image_view)

    async def _checkpoint(self, trace: Any = None) -> None:
        self._loop_state["trace_order_offset"] = self._order_index
        mapping = self._loop_source()["pii_mapping"] if self._loop_source else None
        snapshot = deepcopy(self._loop_state)

        async def write() -> None:
            await self._store.checkpoint_step(
                self._run_id,
                claim_epoch=getattr(self._store, "claim_epoch", 0),
                loop_state=snapshot,
                pii_mapping=mapping,
                trace=trace,
            )

        await retry_critical(write, label="checkpoint_step", run_id=self._run_id)
        self._durable_state = snapshot

    def finalization_trace(self) -> Any:
        """Return the caller-owned final trace for the atomic terminal transaction."""
        return self._final_trace

    def execution_counts(self) -> dict[str, int] | None:
        """Return checkpoint counters even when history has been compacted."""
        return dict(self._loop_state.get("tool_counts", {})) if self._format2 else None

    def configure_tools(self, definitions: list[ToolDef]) -> None:
        """Load trusted execution hints before a direct approval-resume dispatch."""
        self._tool_idempotent = {d.name: d.idempotent for d in definitions}

    async def consume_resume(self, messages: list[Message]) -> None:
        """Commit submitted observations or an approval decision exactly once."""
        from dendrux.errors import PersistenceError
        from dendrux.runtime.state import LoopStatePatch, _apply_loop_patch

        descriptor = self._loop_state.get("submission")
        if not self._format2 or descriptor is None:
            return
        raw = await self._store.get_pause_state(self._run_id) or {}
        payload = raw.get("submitted", {})
        pending = {p["call_id"]: p for p in self._loop_state.get("pending", [])}
        phases = {}
        if descriptor["kind"] == "approval":
            decisions = {
                d["call_id"]: d["approved"] for d in payload.get("submitted_decisions", [])
            }
            for call_id in descriptor["call_ids"]:
                call = pending.get(call_id)
                if call is None or call["phase"] != "gated":
                    continue
                if call.get("decision_id") != f"{descriptor['pause_id']}:{call_id}":
                    logger.warning(
                        "Approval generation mismatch",
                        extra={"run_id": self._run_id, "code": "submission_generation"},
                    )
                    raise PersistenceError(code="submission_generation")
                phases[call_id] = (
                    "approved"
                    if decisions.get(call_id, payload.get("approved")) is True
                    else "rejected"
                )
        traces = []
        removed = []
        for i, message in enumerate(messages):
            call = pending.get(message.call_id or "")
            order = call["order_index"] if call else descriptor["reserved_order_index"] + i
            traces.append(message_trace(message, order, tenant_scope=self._tenant_scope))
            self._order_index = max(self._order_index, order + 1)
            if message.call_id is not None:
                removed.append(message.call_id)
            else:
                self._consumed_user_count += 1
        patch = LoopStatePatch(
            remove_pending=tuple(removed),
            set_call_phase=phases,
            set_phase="dispatch" if len(pending) > len(removed) else "model_next",
            set_fields={"trace_order_offset": self._order_index},
        )

        async def consume() -> None:
            await self._store.consume_submission(
                self._run_id,
                claim_epoch=getattr(self._store, "claim_epoch", 0),
                submission_id=descriptor["submission_id"],
                traces=traces,
                patch=patch,
            )

        await retry_critical(consume, label="consume_submission", run_id=self._run_id)
        self._loop_state = _apply_loop_patch(self._loop_state, patch)
        self._loop_state["submission"] = None
        self._completed_observations.update(removed)
        self._known_history.extend(
            {"role": t.role, "content": t.content, "meta": t.meta} for t in traces
        )
        self._durable_state = deepcopy(self._loop_state)

    def order_observations(self, history: list[Message]) -> None:
        """Use reserved call order for each format-2 batch, including resumed batches."""
        if not self._format2:
            return
        order: dict[str, int] = {}
        start = 0
        for i, message in enumerate([*history, None]):
            if message is not None and message.role.value == "tool":
                continue
            history[start:i] = sorted(
                history[start:i], key=lambda m: order.get(m.call_id or "", len(order))
            )
            if message is not None and message.tool_calls:
                order = {tc.id: n for n, tc in enumerate(message.tool_calls)}
            start = i + 1

    async def checkpoint_view(self, messages: list[Message], iteration: int) -> None:
        """Persist a changed prefix before its next model call, including edits."""
        if not self._format2:
            return
        from dendrux.runtime.state import TraceWrite

        await self._snapshot(iteration - 1)
        view = [
            {"role": t.role, "content": t.content, "meta": t.meta}
            for t in (message_trace(m, 0) for m in messages)
        ]
        self._loop_state["phase"] = "model_next"
        previous = {_view_identity(entry): entry for entry in self._known_history}
        for entry in view:
            old = previous.get(_view_identity(entry))
            if old is not None:
                entry["meta"] = {**(old.get("meta") or {}), **entry["meta"]}
        if [_view_identity(e) for e in view] != [_view_identity(e) for e in self._known_history]:
            trace = TraceWrite(
                generate_ulid(),
                "rewrite",
                json.dumps(view, ensure_ascii=False),
                self._order_index,
                trace_kind="rewrite",
            )
            self._order_index += 1
            await self._checkpoint(trace)
            self._known_history = view
        elif self._loop_state != self._durable_state:
            await self._checkpoint()

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None:
        if self._format2 and self._loop_source is not None:
            await self._snapshot(iteration - 1)
            if self._loop_state != self._durable_state:
                await self._checkpoint()

    async def checkpoint_outcome(self, result: Any, provider_kwargs: Any) -> None:
        """Store pause references and the last execution boundary before releasing ownership."""
        if not self._format2:
            return
        await self._snapshot(result.iteration_count)
        pause = result.meta.get("pause_state")
        if pause is not None:
            from dendrux.types import Message, Role

            if self._final_trace is not None:
                await self._checkpoint(self._final_trace)
                self._final_trace = None

            prefix = [
                Message(role=Role.SYSTEM, content=e["content"])
                for e in self._known_history
                if e["role"] == "system"
            ]
            await self.checkpoint_view([*prefix, *pause.history], result.iteration_count + 1)
            pause.schema_version = 2
            pause.loop_state_ref = True
            pause.provider_kwargs = dict(provider_kwargs or pause.provider_kwargs)
            if pause.image_view is not None:
                await self._state_ref("image_view", pause.image_view.to_dict())
            pause.image_view_ref = self._loop_state.get("image_view_ref")
            self._loop_state["phase"] = {
                "waiting_approval": "awaiting_approval",
                "waiting_client_tool": "awaiting_client",
                "waiting_human_input": "awaiting_input",
            }.get(result.status.value, "model_next")
            gates = pause.approval_gates or {}
            remaining = {tc.id for tc in pause.pending_tool_calls}
            self._loop_state["pending"] = [
                {
                    **p,
                    "phase": p["phase"]
                    if p["phase"] in ("approved", "rejected")
                    and p["call_id"] in (pause.pending_decisions or {})
                    else "gated"
                    if p["call_id"] in gates
                    else "client"
                    if p.get("target") == "client"
                    else p["phase"],
                    "decision_id": p.get("decision_id")
                    if p["phase"] in ("approved", "rejected")
                    and p["call_id"] in (pause.pending_decisions or {})
                    else f"{pause.pause_id}:{p['call_id']}"
                    if p["call_id"] in gates
                    else p.get("decision_id"),
                }
                for p in self._loop_state.get("pending", [])
                if p["call_id"] in remaining
            ]
            if result.status.value == "waiting_human_input":
                await self._state_ref(
                    "clarification",
                    {
                        "v": 1,
                        "question": result.answer or "",
                        "pause_message": pause.pause_message,
                    },
                )
        else:
            self._loop_state["phase"] = "finalizing"
        await self._checkpoint()

    def lease_scope(self) -> AbstractAsyncContextManager[None]:
        """Provide the loop's optional renewal seam from the runtime layer."""
        from dendrux.runtime.context import lease_scope

        return lease_scope()

    def _edge_kwargs(self, edges: list[BlobEdge]) -> dict[str, Any]:
        """``blob_refs`` kwargs for a store write, only when there are edges."""
        if not edges:
            return {}
        if self._tenant_scope is None:
            raise ImagePersistenceUnsupportedError(
                "state_store", store_type=type(self._store).__name__
            )
        return {"blob_refs": edges, "blob_refs_scope": self._tenant_scope}

    def _edges(self, kind: str, owner_id: str, dicts: Sequence[dict[str, Any]]) -> list[BlobEdge]:
        if not dicts:
            return []
        return edges_for_parts(self._tenant_scope or "-", kind, owner_id, dicts)

    async def _emit_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Record a durable run event. FAIL-CLOSED with retry.

        The store allocates the sequence index in the insert.
        """

        identity: dict[str, Any] = (
            {"event_id": generate_ulid()} if capabilities(self._store).retry_ids else {}
        )

        async def _write() -> None:
            from dendrux.runtime.state import validate_event_index, validate_event_writer

            validate_event_writer(self._store)
            index = await self._store.save_run_event(
                self._run_id,
                event_type=event_type,
                iteration_index=iteration,
                correlation_id=correlation_id,
                data=data,
                **identity,
            )
            validate_event_index(index)

        await retry_critical(_write, label="save_run_event", run_id=self._run_id)

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None:
        """Persist a message to react_traces. FAIL-CLOSED."""
        if self._format2:
            if message.role.value == "user" and self._consumed_user_count:
                self._consumed_user_count -= 1
                return
            if message.call_id in self._completed_observations:
                self._completed_observations.remove(message.call_id)
                return
            # The runner's seed already includes its input. Resume injections
            # occur before bind_loop and still need their own trace.
            if self._seed_input_pending and message.role.value == "user":
                self._seed_input_pending = False
                return
            await self._snapshot(iteration)
            pending = self._loop_state.setdefault("pending", [])
            reserved = next((p for p in pending if p["call_id"] == message.call_id), None)
            order = reserved["order_index"] if reserved is not None else self._order_index
            trace = message_trace(message, order, tenant_scope=self._tenant_scope)
            if message.role.value == "assistant" and self._loop_source:
                from dendrux.types import _step_to_dict

                steps = self._loop_source()["steps"]
                if steps:
                    trace.meta["dendrux.step"] = _step_to_dict(steps[-1])
            if reserved is None:
                self._order_index += 1
            else:
                self._loop_state["pending"] = [
                    p for p in pending if p["call_id"] != message.call_id
                ]
                self._loop_state["phase"] = (
                    "dispatch" if self._loop_state["pending"] else "model_next"
                )
            if message.tool_calls:
                data = self._loop_source() if self._loop_source else {}
                targets = data.get("tool_targets", {})
                gates = data.get("approval_tools", ())
                denied = data.get("denied_tools", ())
                self._loop_state["pending"] = [
                    {
                        "call_id": tc.id,
                        "tool": tc.name,
                        "phase": (
                            "invalid"
                            if tc.parse_error or tc.name not in targets or tc.name in denied
                            else "gated"
                            if tc.name in gates
                            else "client"
                            if targets.get(tc.name) == "client"
                            else "dispatchable"
                        ),
                        "decision_id": None,
                        "order_index": self._order_index + i,
                        "target": targets.get(tc.name, "server"),
                    }
                    for i, tc in enumerate(message.tool_calls)
                ]
                self._order_index += len(message.tool_calls)
                self._loop_state["phase"] = "dispatch"
            elif message.role.value == "assistant":
                from dendrux.types import Clarification

                steps = self._loop_source()["steps"] if self._loop_source else []
                if steps and isinstance(steps[-1].action, Clarification):
                    await self._state_ref(
                        "clarification",
                        {"v": 1, "question": steps[-1].action.question, "pause_message": None},
                    )
                    self._loop_state["phase"] = "awaiting_input"
                elif capabilities(self._store).atomic_completion:
                    # A kill before the final transaction leaves model_next:
                    # no committed final answer exists to finalize from yet.
                    self._loop_state["phase"] = "model_next"
                    self._final_trace = trace
                    await self._checkpoint()
                    return
                else:
                    self._loop_state["phase"] = "finalizing"
            await self._checkpoint(trace)
            self._known_history.append(
                {
                    "role": trace.role,
                    "content": trace.content,
                    "meta": {k: v for k, v in trace.meta.items() if k != "dendrux.step"},
                }
            )
            return
        meta = dict(message.meta) if message.meta else {}
        meta["iteration"] = iteration

        if message.tool_calls:
            meta["tool_calls"] = [_tool_call_evidence(tc) for tc in message.tool_calls]

        if message.call_id:
            meta["call_id"] = message.call_id
        if message.name:
            meta["tool_name"] = message.name
        if message.reasoning_blocks:
            # Opaque replay blocks (producing model is already in message.meta)
            # so retry() can rebuild a tool loop with its reasoning intact.
            meta["reasoning_blocks"] = message.reasoning_blocks
        if message.parts is not None and PARTS_META_KEY not in meta:
            meta[PARTS_META_KEY] = [content_part_to_dict(p) for p in message.parts]

        content = message.content
        order_idx = self._order_index
        edge_kwargs = self._edge_kwargs(
            self._edges("trace", str(order_idx), meta.get(PARTS_META_KEY) or [])
        )

        identity: dict[str, Any] = (
            {"trace_id": generate_ulid()} if capabilities(self._store).retry_ids else {}
        )

        # FAIL-CLOSED with retry
        async def _write() -> None:
            await self._store.save_trace(
                self._run_id,
                message.role.value,
                content,
                order_index=order_idx,
                meta=meta,
                **edge_kwargs,
                **identity,
            )

        await retry_critical(_write, label="save_trace", run_id=self._run_id)
        self._order_index += 1

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        """Persist LLM interaction + usage + event."""
        semantic_request: dict[str, Any] | None = None
        has_parts = False
        if semantic_messages is not None:
            semantic_request = {
                "messages": [_serialize_message(m) for m in semantic_messages],
            }
            has_parts = any(m.parts is not None for m in semantic_messages)
            if semantic_tools is not None:
                semantic_request["tools"] = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "target": getattr(t, "target", "server"),
                        "meta": t.meta or None,
                    }
                    for t in semantic_tools
                ]
            if has_parts:
                stats = image_stats(
                    semantic_messages,
                    provider=self._provider,
                    model=response.model or self._model,
                )
                if stats is not None:
                    semantic_request["image_stats"] = stats

        semantic_response: dict[str, Any] | None = None
        if (
            response.text is not None
            or response.tool_calls is not None
            or response.reasoning is not None
            or response.stop_reason is not None
        ):
            semantic_response = {}
            if response.reasoning is not None:
                semantic_response["reasoning"] = response.reasoning
            if response.text is not None:
                semantic_response["text"] = response.text
            if response.tool_calls is not None:
                semantic_response["tool_calls"] = [
                    _tool_call_evidence(tc) for tc in response.tool_calls
                ]
            semantic_response["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost_usd": response.usage.cost_usd,
                "cost_source": response.usage.cost_source,
                "usage_reported": response.usage.usage_reported,
                "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
                "reasoning_tokens": response.usage.reasoning_tokens,
            }
            semantic_response["stop_reason"] = response.stop_reason

        # Client-minted before either write so the llm_interactions row and
        # the llm.completed event share one id — a stable 1:1 join key even
        # if the best-effort interaction save below fails.
        interaction_id = generate_ulid()

        # BEST-EFFORT: llm_interactions (full forensics)
        try:

            async def _write_interaction() -> None:
                await self._store.save_llm_interaction(
                    self._run_id,
                    iteration_index=iteration,
                    usage=response.usage,
                    model=response.model or self._model,
                    provider=self._provider_name,
                    duration_ms=duration_ms,
                    semantic_request=semantic_request,
                    semantic_response=semantic_response,
                    provider_request=(
                        scrub_inline_images(response.provider_request)
                        if has_parts
                        else response.provider_request
                    ),
                    provider_response=response.provider_response,
                    guardrail_findings=guardrail_findings,
                    interaction_id=interaction_id,
                )

            await retry_transient_db(
                _write_interaction, label="save_llm_interaction", run_id=self._run_id
            )
        except RunFencedError:
            raise
        except Exception:
            logger.warning(
                "Failed to save llm_interaction for run %s iteration %d",
                self._run_id,
                iteration,
                exc_info=safe_exception_info(),
            )

        # Mint once per logical call, outside the bounded retry loop.
        usage_id = generate_ulid()
        usage_params = inspect.signature(self._store.save_usage).parameters
        idempotent_usage = "usage_id" in usage_params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in usage_params.values()
        )
        # BEST-EFFORT: token_usage (cost tracking)
        try:
            usage_kwargs: dict[str, Any] = {"usage_id": usage_id} if idempotent_usage else {}

            async def _write_usage() -> None:
                await self._store.save_usage(
                    self._run_id,
                    **usage_kwargs,
                    iteration_index=iteration,
                    usage=response.usage,
                    model=response.model or self._model,
                    provider=self._provider_name,
                    meta=(
                        {"cost_source": response.usage.cost_source}
                        if response.usage.cost_source is not None
                        else None
                    ),
                )

            if idempotent_usage:
                await retry_transient_db(_write_usage, label="save_usage", run_id=self._run_id)
            else:
                # A legacy store cannot safely replay an ambiguous commit.
                await _write_usage()
        except RunFencedError:
            raise
        except Exception:
            logger.warning(
                "Failed to save usage for run %s iteration %d",
                self._run_id,
                iteration,
                exc_info=safe_exception_info(),
            )

        # FAIL-CLOSED: run event (lifecycle audit trail). correlation_id joins
        # 1:1 to LLMCall.id, mirroring tool.completed -> ToolInvocation.tool_call_id.
        await self._emit_event(
            "llm.completed",
            iteration,
            {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
                "reasoning_tokens": response.usage.reasoning_tokens,
                "cost_usd": response.usage.cost_usd,
                "cost_source": response.usage.cost_source,
                "usage_reported": response.usage.usage_reported,
                "model": response.model or self._model,
                "has_tool_calls": bool(response.tool_calls),
                "phase": "commentary" if response.tool_calls else "final_answer",
            },
            correlation_id=interaction_id,
        )

        # BEST-EFFORT: touch progress for sweep
        try:
            await self._store.touch_progress(self._run_id)
        except RunFencedError:
            raise
        except Exception:
            logger.warning(
                "Failed to touch progress for run %s", self._run_id, exc_info=safe_exception_info()
            )

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        """Persist an execution intent before an actual server dispatch."""
        if self._format2:
            from dendrux.runtime.state import EventWrite

            pending = next(
                (p for p in self._loop_state.get("pending", []) if p["call_id"] == tool_call.id),
                None,
            )
            if pending is None or pending["phase"] not in ("dispatchable", "approved"):
                from dendrux.errors import PersistenceError

                code = (
                    "submission_missing"
                    if pending and pending["phase"] == "gated"
                    else "persistence_error"
                )
                logger.warning(
                    "Execution has no dispatchable intent", extra={"run_id": run_id, "code": code}
                )
                raise PersistenceError(code=code)
            if pending.get("target") == "client":
                return
            identity = generate_ulid()
            idempotent = self._tool_idempotent.get(tool_call.name, False)
            event = EventWrite(
                generate_ulid(),
                "tool.started",
                iteration,
                {"tool_name": tool_call.name, "call_id": tool_call.id, "idempotent": idempotent},
                tool_call.id,
                None,
            )

            async def write_intent() -> None:
                await self._store.start_tool_call(
                    run_id,
                    claim_epoch=getattr(self._store, "claim_epoch", 0),
                    row_id=identity,
                    tool_call_id=tool_call.id,
                    provider_tool_call_id=tool_call.provider_tool_call_id,
                    tool_name=tool_call.name,
                    target="server",
                    params=tool_call.params or None,
                    iteration_index=iteration,
                    idempotent=idempotent,
                    event=event,
                )

            await retry_critical(write_intent, label="start_tool_call", run_id=run_id)
            self._started_calls.add(tool_call.id)
            return
        try:
            await self._store.touch_progress(self._run_id)
        except RunFencedError:
            raise
        except Exception:
            logger.warning(
                "Failed to touch progress for run %s", self._run_id, exc_info=safe_exception_info()
            )

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        """Persist safe failure/deadline evidence without inventing billed usage."""
        from dendrux.errors import ProviderError
        from dendrux.types import ErrorInfo

        if not isinstance(error, ProviderError):
            return
        info = ErrorInfo.from_error(error)
        try:
            await self._emit_event(
                "llm.failed",
                iteration,
                {
                    "error_info": info.to_dict(),
                    "timeout_reason": info.timeout_reason,
                    "duration_ms": duration_ms,
                },
            )
        except RunFencedError:
            raise
        except Exception:
            logger.warning(
                "Failed-call evidence unavailable: run_id=%s code=llm_failed_write", self._run_id
            )

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        """Persist tool call record + event."""
        params = tool_call.params if tool_call.params else None
        target = self._target_lookup.get(tool_call.name, "server")
        meta = dict(tool_result.meta) if tool_result.meta else {}
        images = [image_dict(i) for i in tool_result.images]
        if images:
            meta[IMAGES_META_KEY] = images
        if tool_result.notes:
            meta[IMAGE_NOTES_META_KEY] = list(tool_result.notes)
        extra: dict[str, Any] = {"meta": meta} if meta else {}
        extra.update(self._edge_kwargs(self._edges("tool_call", tool_call.id, images)))

        if self._format2 and tool_call.id in self._started_calls:
            from dendrux.runtime.state import EventWrite, LoopStatePatch
            from dendrux.types import tool_result_message

            pending = next(p for p in self._loop_state["pending"] if p["call_id"] == tool_call.id)
            message = (self._result_formatter or tool_result_message)(tool_result)
            observation = message_trace(
                message, pending["order_index"], tenant_scope=self._tenant_scope
            )
            event = EventWrite(
                generate_ulid(),
                "tool.completed",
                iteration,
                {
                    "tool_name": tool_call.name,
                    "target": target,
                    "success": tool_result.success,
                    "duration_ms": tool_result.duration_ms,
                    **{
                        k: tool_result.meta[k]
                        for k in ("code", "zombie_threads")
                        if k in tool_result.meta
                    },
                    **({"image_count": len(images)} if images else {}),
                },
                tool_call.id,
                None,
            )
            patch = LoopStatePatch(remove_pending=(tool_call.id,), set_phase="model_next")

            async def complete() -> None:
                await self._store.complete_tool_call(
                    run_id,
                    claim_epoch=getattr(self._store, "claim_epoch", 0),
                    tool_call_id=tool_call.id,
                    outcome=(
                        "unknown"
                        if tool_result.meta.get("code")
                        in ("outcome_unknown", "timeout_outcome_unknown")
                        else "completed"
                        if tool_result.success
                        else "failed"
                    ),
                    result_payload=tool_result.payload,
                    success=tool_result.success,
                    duration_ms=tool_result.duration_ms,
                    error_message=tool_result.error,
                    replayed=False,
                    observation=observation,
                    event=event,
                    loop_state_patch=patch,
                    **extra,
                )

            await retry_critical(complete, label="complete_tool_call", run_id=run_id)
            self._loop_state["pending"] = [
                p for p in self._loop_state["pending"] if p["call_id"] != tool_call.id
            ]
            self._loop_state["phase"] = "dispatch" if self._loop_state["pending"] else "model_next"
            counts = self._loop_state.setdefault("tool_counts", {})
            counts[tool_call.name] = counts.get(tool_call.name, 0) + 1
            self._completed_observations.add(tool_call.id)
            self._known_history.append(
                {"role": observation.role, "content": observation.content, "meta": observation.meta}
            )
            return

        if capabilities(self._store).retry_ids:
            extra["row_id"] = generate_ulid()

        # FAIL-CLOSED with retry: save_tool_call (proof of side effects)
        async def _write_tool() -> None:
            await self._store.save_tool_call(
                self._run_id,
                tool_call_id=tool_call.id,
                provider_tool_call_id=tool_call.provider_tool_call_id,
                tool_name=tool_call.name,
                target=target,
                params=params,
                result_payload=tool_result.payload,
                success=tool_result.success,
                duration_ms=tool_result.duration_ms,
                iteration_index=iteration,
                error_message=tool_result.error,
                **extra,
            )

        await retry_critical(_write_tool, label="save_tool_call", run_id=self._run_id)

        # FAIL-CLOSED: run event (lifecycle audit trail)
        await self._emit_event(
            "tool.completed",
            iteration,
            {
                "tool_name": tool_call.name,
                "target": target,
                "success": tool_result.success,
                "duration_ms": tool_result.duration_ms,
                **{
                    k: tool_result.meta[k]
                    for k in ("code", "zombie_threads")
                    if k in tool_result.meta
                },
                **tool_call_parse_meta(tool_call),
                **({"image_count": len(images)} if images else {}),
            },
            correlation_id=tool_call.id,
        )

        # BEST-EFFORT: touch progress for sweep
        try:
            await self._store.touch_progress(self._run_id)
        except RunFencedError:
            raise
        except Exception:
            logger.warning(
                "Failed to touch progress for run %s", self._run_id, exc_info=safe_exception_info()
            )

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        """Record a governance event. FAIL-CLOSED with retry."""
        if self._format2 and event_type == "approval.requested":
            for pending in self._loop_state.get("pending", []):
                if pending["call_id"] == correlation_id:
                    pending["phase"] = "gated"
                    self._loop_state["phase"] = "awaiting_approval"
                    # Governance runs before server dispatch; no completion is
                    # outstanding while this checkpoint replaces loop_state.
                    await self._checkpoint()
                    break
        await self._emit_event(
            event_type,
            iteration,
            data=data,
            correlation_id=correlation_id,
        )


@dataclass(frozen=True)
class ResolutionAction:
    """A recovery decision for an unresolved reserved observation.

    Only pending calls are considered: atomic completion removes a call from
    pending in the same transaction as its observation. Dispatch actions must
    still go through ordinary validation and governance in the reclaim runner.
    """

    kind: str
    call_id: str | None = None
    order_index: int | None = None


def resolve_pending(loop_state: dict[str, Any], tool_rows: Sequence[Any]) -> list[ResolutionAction]:
    """Decide the next recovery actions without executing or mutating anything."""
    if loop_state.get("submission") is not None:
        return [ResolutionAction("consume_submission")]

    def read(row: Any, name: str, default: Any = None) -> Any:
        return row.get(name, default) if isinstance(row, dict) else getattr(row, name, default)

    rows = {read(row, "tool_call_id"): row for row in tool_rows}
    actions = []
    for pending in loop_state.get("pending", []):
        call_id, phase = pending["call_id"], pending["phase"]
        row = rows.get(call_id)
        kind: str | None
        if phase == "client":
            kind = "wait_client"
        elif row is None:
            kind = {
                "gated": "approval",
                "rejected": "resolve",
                "invalid": "resolve",
                "approved": "dispatch",
                "dispatchable": "dispatch",
            }.get(phase)
            if kind is None:
                raise ValueError("Unsupported pending-call phase")
        elif read(row, "outcome") == "started":
            kind = "replay" if read(row, "idempotent", False) else "approval_replay"
        elif read(row, "outcome") in ("completed", "failed", "unknown"):
            kind = "observe"
        else:
            raise ValueError("Unsupported tool-call outcome")
        actions.append(ResolutionAction(kind, call_id, pending["order_index"]))
    return actions
