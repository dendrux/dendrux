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

import asyncio
import hashlib
import inspect
import json
import logging
from bisect import insort
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED, GovernanceEventType, StoredEventType
from dendrux._meta_keys import (
    IMAGE_NOTES_META_KEY as IMAGE_NOTES_META_KEY,
)
from dendrux._meta_keys import (
    IMAGES_META_KEY as IMAGES_META_KEY,
)
from dendrux._meta_keys import (
    MESSAGE_META_KEY,
    NOT_EXECUTED_META_KEY,
    OBSERVATION_META_KEY,
    SEED_PREFIX_META_KEY,
    STEP_META_KEY,
)
from dendrux._meta_keys import (
    PARTS_META_KEY as PARTS_META_KEY,
)
from dendrux.errors import ImagePersistenceUnsupportedError, RunFencedError, safe_exception_info
from dendrux.loops.base import BaseRecorder
from dendrux.runtime import _managed_budget, _managed_cancel
from dendrux.runtime._blob_catalog import edges_for_parts, scrub_inline_images
from dendrux.runtime._cancel_guard import capture
from dendrux.runtime._capabilities import capabilities
from dendrux.runtime._clarification import DEFAULT_PAUSE_SUMMARY_BYTES, clarification_summary
from dendrux.runtime.durability import retry_critical, retry_transient_db
from dendrux.types import (
    ImagePart,
    ImageRef,
    RunStatus,
    content_part_to_dict,
    generate_ulid,
    tool_call_parse_meta,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from contextlib import AbstractAsyncContextManager

    from dendrux.guardrails import GuardrailEngine
    from dendrux.media import BlobEdge
    from dendrux.runtime.state import RunControl, StateStore, TraceWrite
    from dendrux.types import (
        AgentStep,
        Limits,
        LLMResponse,
        Message,
        ToolCall,
        ToolDef,
        ToolResult,
        ToolTarget,
    )


logger = logging.getLogger(__name__)


def _paused_call(call: dict[str, Any], pause: Any) -> dict[str, Any]:
    """Retain decisions already made and assign the remaining pause gates."""
    call = {**call, "decision_id": call.get("decision_id")}
    call_id = call["call_id"]
    decided = call["phase"] in ("approved", "rejected") and call_id in (
        pause.pending_decisions or {}
    )
    if decided:
        return call
    if call_id in (pause.approval_gates or {}):
        call.update(phase="gated", decision_id=f"{pause.pause_id}:{call_id}")
    elif call.get("target") == "client":
        call["phase"] = "client"
    return call


def _pause_pending(pending: list[dict[str, Any]], pause: Any) -> list[dict[str, Any]]:
    remaining = {tc.id for tc in pause.pending_tool_calls}
    return [_paused_call(call, pause) for call in pending if call["call_id"] in remaining]


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
    return _message_trace(message, order, generate_ulid(), kind=kind, tenant_scope=tenant_scope)


def _message_trace(
    message: Message,
    order: int,
    trace_id: str,
    *,
    kind: str = "turn",
    tenant_scope: str | None = None,
) -> Any:
    from dendrux.runtime.state import TraceWrite
    from dendrux.types import _message_to_dict

    envelope = _message_to_dict(message)
    envelope.pop("content", None)
    meta = dict(message.meta)
    meta[MESSAGE_META_KEY] = envelope
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
        trace_id,
        message.role.value,
        message.content,
        order,
        trace_kind=kind,
        meta=meta,
        blob_refs=edges or None,
        blob_refs_scope=tenant_scope if edges else None,
    )


@dataclass(frozen=True)
class _MessagePlan:
    trace: Any
    pending: list[dict[str, Any]]
    phase: str | None
    order_delta: int
    final_trace: bool = False
    clarification: dict[str, Any] | None = None
    phase_after_state: str | None = None


def _planned_call(
    tc: ToolCall,
    order: int,
    targets: dict[str, Any],
    gates: Collection[str],
    denied: Collection[str],
) -> dict[str, Any]:
    if tc.parse_error or tc.name not in targets or tc.name in denied:
        phase = "invalid"
    elif tc.name in gates:
        phase = "gated"
    elif targets.get(tc.name) == "client":
        phase = "client"
    else:
        phase = "dispatchable"
    return {
        "call_id": tc.id,
        "tool": tc.name,
        "phase": phase,
        "decision_id": None,
        "order_index": order,
        "target": targets.get(tc.name, "server"),
    }


def _plan_assistant(
    plan: _MessagePlan,
    last_step: AgentStep | None,
    atomic_completion: bool,
    pause_summary_bytes: int = DEFAULT_PAUSE_SUMMARY_BYTES,
    run_id: str | None = None,
    clarification: dict[str, Any] | None = None,
) -> _MessagePlan:
    from dendrux.types import Clarification

    if last_step is not None and isinstance(last_step.action, Clarification):
        phase = "awaiting_input"
        return replace(
            plan,
            clarification={
                "v": 1,
                **(
                    clarification
                    if clarification is not None
                    else clarification_summary(
                        last_step.action.question,
                        last_step.action.options,
                        max_bytes=pause_summary_bytes,
                        run_id=run_id,
                    )
                ),
                "pause_message": None,
            },
            phase_after_state=phase,
        )
    # A kill before the final transaction must leave model_next: there is
    # no committed final answer to finalize from yet.
    phase = "model_next" if atomic_completion else "finalizing"
    return replace(plan, phase=phase, final_trace=atomic_completion)


def _message_position(
    call_id: str | None,
    pending: list[dict[str, Any]],
    phase: str | None,
    order_index: int,
    rejected: bool = False,
) -> tuple[int, list[dict[str, Any]], str | None, int]:
    reserved = next((p for p in pending if p["call_id"] == call_id), None)
    if reserved is None:
        return order_index, pending, phase, 1
    if rejected:
        ordered = sorted(pending, key=lambda p: p["order_index"])
        slots = [p["order_index"] for p in ordered]
        reordered = [reserved, *[p for p in ordered if p is not reserved]]
        for call, slot in zip(reordered, slots, strict=True):
            call["order_index"] = slot
    pending = sorted(
        (p for p in pending if p["call_id"] != call_id), key=lambda p: p["order_index"]
    )
    phase = "dispatch" if pending else "model_next"
    return reserved["order_index"], pending, phase, 0


def _plan_message(
    message: Message,
    *,
    trace_id: str,
    pending: list[dict[str, Any]],
    phase: str | None,
    order_index: int,
    tenant_scope: str | None,
    last_step: AgentStep | None,
    targets: dict[str, Any],
    gates: Collection[str],
    denied: Collection[str],
    atomic_completion: bool,
    pause_summary_bytes: int = DEFAULT_PAUSE_SUMMARY_BYTES,
    run_id: str | None = None,
    clarification: dict[str, Any] | None = None,
) -> _MessagePlan:
    from dendrux.types import _step_to_dict

    order, pending, phase, delta = _message_position(
        message.call_id, pending, phase, order_index, bool(message.meta.get(NOT_EXECUTED_META_KEY))
    )
    trace = _message_trace(message, order, trace_id, tenant_scope=tenant_scope)
    if message.role.value == "assistant" and last_step is not None:
        trace.meta[STEP_META_KEY] = _step_to_dict(last_step)
    if message.tool_calls:
        pending = [
            _planned_call(tc, order_index + delta + i, targets, gates, denied)
            for i, tc in enumerate(message.tool_calls)
        ]
        slots = sorted(p["order_index"] for p in pending)
        pending.sort(
            key=lambda p: 0 if p["phase"] == "invalid" else 2 if p["phase"] == "client" else 1
        )
        for call, slot in zip(pending, slots, strict=True):
            call["order_index"] = slot
        delta += len(message.tool_calls)
        phase = "dispatch"
    plan = _MessagePlan(trace, pending, phase, delta)
    if message.role.value == "assistant" and not message.tool_calls:
        return _plan_assistant(
            plan, last_step, atomic_completion, pause_summary_bytes, run_id, clarification
        )
    return plan


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
        trace.meta[SEED_PREFIX_META_KEY] = i < own_index
        trace.meta["iteration"] = 0
        if i == own_index and own_input is not None:
            traces[i] = replace(
                trace, content=own_input.content, meta={**trace.meta, **own_input.meta}
            )
    return {"seed_traces": traces}


def _view_identity(entry: dict[str, Any]) -> str:
    meta = entry.get("meta") or {}
    value = json.dumps(
        {
            "role": entry["role"],
            "content": meta.get("wire_content", entry["content"]),
            "message": meta.get(MESSAGE_META_KEY, meta),
        },
        sort_keys=True,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PersistenceRecorder(_managed_cancel.CancelAdmission, BaseRecorder):
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
        guardrail_engine: GuardrailEngine | None = None,
    ) -> None:
        self._store = state_store
        self._cancel_guard = capture()
        self._store_calls = self._cancel_guard.store_calls(state_store)
        self._tenant_scope = tenant_scope
        self._provider = provider
        self._run_id = run_id
        self._model = model
        self._provider_name = provider_name
        self._target_lookup = target_lookup or {}
        self._format2 = (
            getattr(state_store, "storage_format", 1) == STORAGE_FORMAT_CHECKPOINTED
            and capabilities(state_store).checkpointing
        )
        self._order_index = (
            len(seed_traces) if self._format2 and seed_traces else initial_order_index
        )
        self.guardrail_engine = guardrail_engine if self._format2 else None
        self._projected_observations: dict[str, Message] = {}
        self._observation_lock = asyncio.Lock()
        self._loop_source: Any = None
        self._loop_state: dict[str, Any] = deepcopy(getattr(state_store, "loop_state", None) or {})
        self._known_history: list[tuple[int, str]] = []
        self._observation_order: dict[str, int] = {}
        self._system_messages: list[Message] = []
        for trace in seed_traces or [] if self._format2 else []:
            self._remember(trace)
        self._state_payloads: dict[str, str] = {}
        self._seed_input_pending = any(t.role == "user" for t in seed_traces or [])
        self._durable_state = deepcopy(self._loop_state)
        self._started_calls: set[str] = set()
        self._replay_calls: set[str] = set()
        self._replayed_calls: set[str] = set()
        self._completed_observations: set[str] = set()
        self._tool_idempotent: dict[str, bool] = {}
        self._result_formatter: Any = None
        self._consumed_user_count = 0
        self._final_trace: Any = None
        self._resume_pause_data: dict[str, Any] | None = None
        self._history_limits: Limits | None = None
        self._clarification: dict[str, Any] | None = None

    def _blocked_evidence(self, run_id: str) -> AbstractAsyncContextManager[None]:
        from dendrux.runtime._blocked_content import _blocked_evidence

        return _blocked_evidence(self._store, run_id)

    @property
    def _pause_summary_bytes(self) -> int:
        return (
            self._history_limits.max_pause_summary_bytes
            if self._history_limits is not None
            else DEFAULT_PAUSE_SUMMARY_BYTES
        )

    async def restore_history(self, *, limits: Any, records: Any = None) -> None:
        self._history_limits = limits
        if not self._format2:
            return
        from dendrux.runtime.replay import conversation_records

        if records is None:
            records = await conversation_records(self._store_calls, self._run_id, limits=limits)
        self._known_history.clear()
        self._system_messages.clear()
        for trace in records:
            self._remember(trace, position=trace.order_index)

    def _remember(self, trace: Any, *, position: int | None = None) -> None:
        from dendrux.types import Message, Role

        insort(
            self._known_history,
            (
                trace.order_index * 1000 if position is None else position,
                _view_identity({"role": trace.role, "content": trace.content, "meta": trace.meta}),
            ),
        )
        self._remember_observation(trace, position)
        if trace.role == "system":
            self._system_messages.append(Message(role=Role.SYSTEM, content=trace.content))

    def _remember_observation(self, trace: Any, position: int | None) -> None:
        call_id = (trace.meta or {}).get("call_id")
        if call_id:
            self._observation_order[call_id] = (
                trace.order_index * 1000 if position is None else position
            )

    def _mapping(self) -> dict[str, str] | None:
        if self.guardrail_engine is not None:
            return self.guardrail_engine.get_pii_mapping()
        return self._loop_source()["pii_mapping"] if self._loop_source else None

    def _pii_patch(self) -> dict[str, Any]:
        if self.guardrail_engine is None:
            return {}
        return {
            "pii": {
                "mapping_version": len(self._mapping() or {}),
                "counters": dict(self.guardrail_engine._counters),
            }
        }

    def bind_loop(self, source: Any, budget: Any) -> None:
        """Bind live loop data without making loops depend on runtime internals."""
        if not self._format2:
            return
        self._loop_source = source
        live = source()
        from dendrux.types import Limits, _validate_provider_kwargs

        self._history_limits = live.get("limits", self._history_limits) or Limits()

        _validate_provider_kwargs(live.get("provider_kwargs", {}), run_id=self._run_id)
        self._tool_idempotent = live.get("idempotent", {})
        self._result_formatter = live.get("format_result")
        for target, saved_value in (
            (live["tool_counts"], self._loop_state.get("tool_counts")),
            (live["pii_counters"], self._loop_state.get("pii", {}).get("counters")),
        ):
            if saved_value is not None:
                if target is live["pii_counters"]:
                    for key, value in saved_value.items():
                        target[key] = max(target.get(key, 0), value)
                else:
                    target.clear()
                    target.update(saved_value)
        _managed_budget.bind(self, budget)
        self._restore_budget(budget)

    def _restore_budget(self, budget: Any) -> None:
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
            {"kind": kind, "v": 1, **payload},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        field = "session_ref" if kind == "skills_session" else kind + "_ref"
        refs = (
            self._loop_state.setdefault("skills", {})
            if kind == "skills_session"
            else self._loop_state
        )
        if kind not in self._state_payloads and refs.get(field):
            from dendrux.errors import PersistenceError

            saved = await self._store_calls.get_trace(self._run_id, refs[field])
            try:
                if saved is None or saved.trace_kind != "state" or saved.role != "state":
                    raise ValueError("Missing state evidence")
                value = json.loads(saved.content)
                if (
                    not isinstance(value, dict)
                    or value.get("kind") != kind
                    or type(value.get("v")) is not int
                    or value["v"] != 1
                ):
                    raise ValueError("Unsupported state evidence")
                self._state_payloads[kind] = json.dumps(
                    value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
            except (TypeError, ValueError):
                raise PersistenceError(code="pause_format_unsupported") from None
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
            await self._store_calls.save_trace(
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
            loop=data.get("loop", "react"),
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
        mapping = self._mapping()
        snapshot = deepcopy(self._loop_state)

        async def write() -> None:
            await self._store_calls.checkpoint_step(
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

    def _submitted_client_counts(
        self, pending: dict[str, Any], removed: list[str]
    ) -> dict[str, int]:
        counts = dict(self._loop_state.get("tool_counts", {}))
        for call_id in set(removed):
            call = pending.get(call_id)
            if call is not None and call["phase"] == "client":
                name = call["tool"]
                counts[name] = counts.get(name, 0) + 1
        return counts

    def _submission_traces(
        self, messages: list[Message], descriptor: dict[str, Any], pending: dict[str, Any]
    ) -> tuple[list[Any], list[str]]:
        traces = []
        removed = []
        for i, message in enumerate(messages):
            call = pending.get(message.call_id or "")
            order = call["order_index"] if call else descriptor["reserved_order_index"] + i
            traces.append(message_trace(message, order, tenant_scope=self._tenant_scope))
            traces[-1].meta["iteration"] = int(self._loop_state.get("iteration", 0))
            self._order_index = max(self._order_index, order + 1)
            if message.call_id is not None:
                removed.append(message.call_id)
            else:
                self._consumed_user_count += 1
        return traces, removed

    async def consume_resume(self, messages: list[Message]) -> None:
        """Commit submitted observations or an approval decision exactly once."""
        from dendrux.errors import PersistenceError
        from dendrux.runtime.state import LoopStatePatch, _apply_loop_patch

        descriptor = self._loop_state.get("submission")
        if not self._format2 or descriptor is None:
            return
        raw = self._resume_pause_data
        if raw is None:
            raw = await self._store_calls.get_pause_state(self._run_id) or {}
        from dendrux.runtime.submit import _submission_payload

        payload = _submission_payload(raw)
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
                    _codes.log_code(logger, _codes.SUBMISSION_GENERATION, run_id=self._run_id)
                    raise PersistenceError(code="submission_generation")
                phases[call_id] = (
                    "approved"
                    if decisions.get(call_id, payload.get("approved")) is True
                    else "rejected"
                )
        traces, removed = self._submission_traces(messages, descriptor, pending)
        patch = LoopStatePatch(
            remove_pending=tuple(removed),
            set_call_phase=phases,
            set_phase="dispatch" if len(pending) > len(removed) else "model_next",
            set_fields={
                "trace_order_offset": self._order_index,
                "tool_counts": self._submitted_client_counts(pending, removed),
                **self._pii_patch(),
            },
        )

        async def consume() -> None:
            await self._store_calls.consume_submission(
                self._run_id,
                claim_epoch=getattr(self._store, "claim_epoch", 0),
                submission_id=descriptor["submission_id"],
                traces=traces,
                patch=patch,
                **({"pii_mapping": self._mapping()} if self.guardrail_engine else {}),
            )

        await retry_critical(consume, label="consume_submission", run_id=self._run_id)
        self._resume_pause_data = None
        self._loop_state = _apply_loop_patch(self._loop_state, patch)
        self._loop_state["submission"] = None
        self._completed_observations.update(removed)
        for trace in traces:
            self._remember(trace)
        self._durable_state = deepcopy(self._loop_state)

    async def mark_managed_refusal(self, call_ids: list[str]) -> None:
        """Durably refuse a whole unsupported managed batch before any single refusal (0.7).

        Recovery refuses every listed call still pending, even when no managed
        call remains, so a partial refusal never lets a sibling's effect run.
        """
        if not self._format2:
            return
        self._loop_state["managed_refused"] = list(call_ids)
        await self._checkpoint()

    def order_observations(self, history: list[Message]) -> None:
        """Use the durable observation slots, including decisions made after resume."""
        if not self._format2:
            return
        start = 0
        for i, message in enumerate([*history, None]):
            if message is not None and message.role.value == "tool":
                continue
            history[start:i] = sorted(
                history[start:i], key=lambda m: self._observation_order.get(m.call_id or "", 0)
            )
            start = i + 1

    async def checkpoint_view(self, messages: list[Message], iteration: int) -> None:
        """Persist a changed prefix before its next model call, including edits."""
        if not self._format2:
            return
        await self._snapshot(iteration - 1)
        view = [
            {"role": t.role, "content": t.content, "meta": t.meta}
            for t in (message_trace(m, 0) for m in messages)
        ]
        self._loop_state["phase"] = "model_next"
        digests = [_view_identity(entry) for entry in view]
        if digests != [digest for _, digest in self._known_history]:
            trace = await self._edited_view(view, digests)
            self._order_index += 1
            self._rebase_pending(trace.order_index)
            await self._checkpoint(trace)
            self._known_history = [
                (trace.order_index * 1000 + i, digest) for i, digest in enumerate(digests)
            ]
            self._observation_order = {
                m.call_id: trace.order_index * 1000 + i
                for i, m in enumerate(messages)
                if m.call_id is not None
            }
            self._system_messages = [m for m in messages if m.role.value == "system"]
        elif self._loop_state != self._durable_state:
            await self._checkpoint()

    def _rebase_pending(self, rewrite_order: int) -> None:
        pending = sorted(self._loop_state.get("pending", []), key=lambda p: p["order_index"])
        for i, call in enumerate(pending):
            call["order_index"] = rewrite_order + 1 + i
        self._order_index = max(self._order_index, rewrite_order + 1 + len(pending))

    async def _edited_view(self, view: list[dict[str, Any]], digests: list[str]) -> TraceWrite:
        from dendrux.runtime.replay import conversation_records
        from dendrux.runtime.state import TraceWrite

        # Only a genuine edit needs the old evidence metadata. The normal
        # boundary compares compact digests without reading history again.
        previous = {
            _view_identity({"role": t.role, "content": t.content, "meta": t.meta}): t.meta
            for t in await conversation_records(
                self._store_calls, self._run_id, limits=self._history_limits
            )
        }
        for entry, digest in zip(view, digests, strict=True):
            entry["meta"] = {**(previous.get(digest) or {}), **entry["meta"]}
        return TraceWrite(
            generate_ulid(),
            "rewrite",
            json.dumps(view, ensure_ascii=False),
            self._order_index,
            trace_kind="rewrite",
        )

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
        # Ownership can still change between this read and the network send.
        await self._admission_control(run_id)
        await self._cancel_guard.check(run_id)

    async def checkpoint_outcome(self, result: Any, provider_kwargs: Any) -> None:
        """Store pause references and the last execution boundary before releasing ownership."""
        if not self._format2:
            return
        await self._snapshot(result.iteration_count)
        pause = result.meta.get("pause_state")
        if pause is not None:
            if self._final_trace is not None:
                await self._checkpoint(self._final_trace)
                self._final_trace = None

            await self.checkpoint_view(
                [*self._system_messages, *pause.history], result.iteration_count + 1
            )
            pause.schema_version = 2
            pause.loop_state_ref = True
            pause.provider_kwargs = dict(provider_kwargs or pause.provider_kwargs)
            if pause.image_view is not None:
                await self._state_ref("image_view", pause.image_view.to_dict())
            pause.image_view_ref = self._loop_state.get("image_view_ref")
            self._loop_state["phase"] = {
                RunStatus.WAITING_APPROVAL.value: "awaiting_approval",
                RunStatus.WAITING_CLIENT_TOOL.value: "awaiting_client",
                RunStatus.WAITING_HUMAN_INPUT.value: "awaiting_input",
            }.get(result.status.value, "model_next")
            self._loop_state["pending"] = _pause_pending(self._loop_state.get("pending", []), pause)
            if result.status.value == RunStatus.WAITING_HUMAN_INPUT.value:
                await self._checkpoint_clarification(result, pause)
        else:
            self._loop_state["phase"] = "finalizing"
        await self._checkpoint()

    def prepare_clarification(self, response: Any, step: Any) -> tuple[Any, Any]:
        """Bound a format-2 question before it becomes history needed to resume."""
        from dendrux.types import Clarification

        if not self._format2 or not isinstance(step.action, Clarification):
            return response, step
        action = step.action
        summary = clarification_summary(
            action.question,
            action.options,
            max_bytes=self._pause_summary_bytes,
            run_id=self._run_id,
        )
        self._clarification = summary
        if summary == {"question": action.question, "options": action.options}:
            return response, step
        bounded = Clarification(question=summary["question"], options=summary["options"])
        if bounded.question != action.question:
            return replace(response, text=bounded.question), replace(
                step, action=bounded, raw_response=None
            )
        return response, replace(step, action=bounded)

    async def _checkpoint_clarification(self, result: Any, pause: Any) -> None:
        from dendrux.types import Clarification

        action = pause.steps[-1].action if pause.steps else None
        options = (
            action.options
            if isinstance(action, Clarification)
            else (pause.clarification or {}).get("options", [])
        )
        pause.clarification = (
            self._clarification
            if self._clarification is not None
            else clarification_summary(
                result.answer or "",
                options,
                max_bytes=self._pause_summary_bytes,
                run_id=self._run_id,
            )
        )
        await self._state_ref(
            "clarification",
            {
                "v": 1,
                **pause.clarification,
                "pause_message": pause.pause_message,
            },
        )

    def check_lease(self) -> None:
        from dendrux.runtime.context import get_delegation_context

        ctx = get_delegation_context()
        if ctx is not None and ctx._lease_errors[0] is not None:
            raise ctx._lease_errors[0]

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
        *,
        event_id: str | None = None,
    ) -> None:
        """Record a durable run event. FAIL-CLOSED with retry.

        The store allocates the sequence index in the insert.
        """

        identity: dict[str, Any] = (
            {"event_id": event_id or generate_ulid()} if capabilities(self._store).retry_ids else {}
        )

        async def _write() -> None:
            from dendrux.runtime.state import validate_event_index, validate_event_writer

            validate_event_writer(self._store)
            index = await self._store_calls.save_run_event(
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
        if not self._format2:
            await self._append_legacy_message(message, iteration)
            return
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
        data = (
            self._loop_source()
            if self._loop_source and (message.role.value == "assistant" or message.tool_calls)
            else {}
        )
        steps = data.get("steps", [])
        plan = _plan_message(
            message,
            trace_id=generate_ulid(),
            pending=deepcopy(self._loop_state.setdefault("pending", [])),
            phase=self._loop_state.get("phase"),
            order_index=self._order_index,
            tenant_scope=self._tenant_scope,
            last_step=steps[-1] if steps else None,
            targets=data.get("tool_targets", {}),
            gates=data.get("approval_tools", ()),
            denied=data.get("denied_tools", ()),
            atomic_completion=capabilities(self._store).atomic_completion,
            pause_summary_bytes=self._pause_summary_bytes,
            run_id=self._run_id,
            clarification=self._clarification,
        )
        plan.trace.meta["iteration"] = iteration
        await self._apply_message_plan(plan)

    async def _apply_message_plan(self, plan: _MessagePlan) -> None:
        self._order_index += plan.order_delta
        self._loop_state["pending"] = plan.pending
        if plan.phase is not None:
            self._loop_state["phase"] = plan.phase
        if plan.clarification is not None:
            # Message order N is reserved before this separate state write
            # consumes N+1; awaiting_input applies only after it succeeds.
            await self._state_ref("clarification", plan.clarification)
            self._loop_state["phase"] = plan.phase_after_state
        if plan.final_trace:
            self._final_trace = plan.trace
            await self._checkpoint()
            return
        await self._checkpoint(plan.trace)
        self._remember(plan.trace)

    async def _append_legacy_message(self, message: Message, iteration: int) -> None:
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
            await self._store_calls.save_trace(
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
                await self._store_calls.save_llm_interaction(
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

            await _managed_budget.record_interaction(self, _write_interaction)
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
                await self._store_calls.save_usage(
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
            StoredEventType.LLM_COMPLETED.value,
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

        if self._format2:
            return  # The preceding fenced event already renewed progress.

        # BEST-EFFORT: touch progress for sweep
        try:
            await self._store_calls.touch_progress(self._run_id)
        except RunFencedError:
            raise
        except Exception:
            logger.warning(
                "Failed to touch progress for run %s", self._run_id, exc_info=safe_exception_info()
            )

    async def resolve_observation(self, call_id: str, message: Message | None) -> None:
        """Resolve a stored result at its reserved position without another effect."""
        pending = next(p for p in self._loop_state["pending"] if p["call_id"] == call_id)
        trace = (
            replace(
                message_trace(message, pending["order_index"], tenant_scope=self._tenant_scope),
                advances_progress=False,
            )
            if message is not None
            else None
        )
        self._loop_state["pending"] = [
            p for p in self._loop_state["pending"] if p["call_id"] != call_id
        ]
        self._loop_state["phase"] = "dispatch" if self._loop_state["pending"] else "model_next"
        await self._checkpoint(trace)
        self._completed_observations.add(call_id)
        if trace is not None:
            self._remember(trace)

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        """Persist an execution intent before an actual server dispatch."""
        from dendrux.errors import PersistenceError
        from dendrux.runtime._blocked_content import _dispatch_write, _DispatchUndecided

        await self._check_custom_dispatch(run_id, await self._managed_checkpoint(run_id))
        undecided = False
        with _dispatch_write(self._store, run_id):
            try:
                await self._write_tool_intent(run_id, tool_call, iteration)
            except _DispatchUndecided:
                # A refused dispatch write whose cause could not be read: refuse safely.
                undecided = True
            if not undecided:
                await self._cancel_guard.check(run_id)
        if undecided:
            raise PersistenceError()

    async def _check_custom_dispatch(self, run_id: str, control: RunControl | None = None) -> None:
        from dendrux.guardrails._engine import _entry_engine
        from dendrux.loops._helpers import _check_blocked_run
        from dendrux.runtime._blocked_content import _native_blocked_store

        native = _native_blocked_store(self._store) is not None
        guarded = self.guardrail_engine is not None or _entry_engine(run_id) is not None
        if not native and guarded and hasattr(self._store, "add_note"):
            # Custom stores have no mark predicate on the dispatch write: a guarded
            # run reads the mark when notes are pending. Guardrail-free runs add no read.
            control = await self._admission_control(run_id, control)
            if control.has_pending_notes:
                await _check_blocked_run(self._store, run_id)

    async def _write_tool_intent(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        if self._format2:
            await self._start_checkpointed_tool(run_id, tool_call, iteration)
        else:
            await self._touch_tool_progress(run_id)

    async def _touch_tool_progress(self, run_id: str) -> None:
        from dendrux.loops._helpers import _BlockedObservationError
        from dendrux.runtime._blocked_content import _DispatchUndecided

        try:
            # Legacy progress writes stay best-effort. A write the native mark predicate
            # refused stops the dispatch: a proven mark, or an unexplained refusal.
            await self._store_calls.touch_progress(run_id)
        except (RunFencedError, _BlockedObservationError, _DispatchUndecided):
            raise
        except Exception:
            logger.warning(
                "Failed to touch progress for run %s",
                run_id,
                exc_info=safe_exception_info(),
            )

    async def _start_checkpointed_tool(
        self, run_id: str, tool_call: ToolCall, iteration: int
    ) -> None:
        pending = next(
            (p for p in self._loop_state.get("pending", []) if p["call_id"] == tool_call.id),
            None,
        )
        if (
            pending is not None
            and pending["phase"] == "gated"
            and tool_call.id in self._replay_calls
            and self._tool_idempotent.get(tool_call.name, False)
        ):
            # A requested pause carries recovery identity even when replay
            # itself needs no approval. Ordinary gates run before this hook.
            pending["phase"] = "dispatchable"
            pending["decision_id"] = None
            await self._checkpoint()
        if pending is None or pending["phase"] not in ("dispatchable", "approved"):
            from dendrux.errors import PersistenceError

            code = (
                "submission_missing"
                if pending and pending["phase"] == "gated"
                else "persistence_error"
            )
            _codes.log_code(logger, _codes.REGISTRY[code], run_id=run_id)
            raise PersistenceError(code=code)
        if pending.get("target") == "client":
            return
        if pending["phase"] == "approved":
            # One durable decision authorizes one attempt. A second crash
            # after this boundary must ask again for a non-idempotent effect.
            pending["phase"] = "dispatchable"
            pending["decision_id"] = None
            await self._checkpoint()
        await self._record_tool_intent(run_id, tool_call, iteration)

    async def _record_tool_intent(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        from dendrux.runtime.state import EventWrite

        if tool_call.id in self._replay_calls:
            event_id = generate_ulid()

            async def replay_event() -> None:
                await self._store_calls.save_run_event(
                    run_id,
                    event_type=StoredEventType.TOOL_REPLAYED.value,
                    iteration_index=iteration,
                    data={"call_id": tool_call.id},
                    correlation_id=tool_call.id,
                    event_id=event_id,
                )

            await retry_critical(replay_event, label="tool_replayed", run_id=run_id)
            _codes.log_code(logger, _codes.TOOL_REPLAYED, run_id=run_id)
            self._replayed_calls.add(tool_call.id)
            self._started_calls.add(tool_call.id)
            return
        identity = generate_ulid()
        idempotent = self._tool_idempotent.get(tool_call.name, False)
        event = EventWrite(
            generate_ulid(),
            StoredEventType.TOOL_STARTED.value,
            iteration,
            {"tool_name": tool_call.name, "call_id": tool_call.id, "idempotent": idempotent},
            tool_call.id,
            None,
        )

        async def write_intent() -> None:
            await self._store_calls.start_tool_call(
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
                StoredEventType.LLM_FAILED.value,
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
            _codes.log_code(logger, _codes.LLM_FAILED_WRITE, run_id=self._run_id)

    def format_observation(self, result: ToolResult) -> Message:
        from dendrux.types import tool_result_message

        return (self._result_formatter or tool_result_message)(result)

    def take_observation(self, call_id: str | None) -> Message | None:
        return self._projected_observations.pop(call_id, None) if call_id else None

    def observation_scope(self) -> AbstractAsyncContextManager[None]:
        """Serialize loop projection and its atomic observation/mapping write."""
        return self._observation_lock

    async def on_projected_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
        observation: Message,
    ) -> None:
        # The loop supplies provider form. Serialize full mapping snapshots
        # with their writes; the recorder never invokes the guardrail engine.
        await self._complete_observation(
            run_id, tool_call, tool_result, iteration, observation=observation
        )
        self._projected_observations[tool_call.id] = observation

    async def on_tool_completed(
        self, run_id: str, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        """Commit an observation with the mapping that produced its placeholders."""
        if self.guardrail_engine is not None:
            async with self._observation_lock:
                await self._complete_observation(run_id, tool_call, tool_result, iteration)
        else:
            await self._complete_observation(run_id, tool_call, tool_result, iteration)

    async def _complete_observation(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
        *,
        observation: Message | None = None,
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

            pending = next(p for p in self._loop_state["pending"] if p["call_id"] == tool_call.id)
            message = (
                observation if observation is not None else self.format_observation(tool_result)
            )
            observation_trace = message_trace(
                message, pending["order_index"], tenant_scope=self._tenant_scope
            )
            observation_trace.meta["iteration"] = iteration
            event = EventWrite(
                generate_ulid(),
                StoredEventType.TOOL_COMPLETED.value,
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
            patch = LoopStatePatch(
                remove_pending=(tool_call.id,), set_phase="model_next", set_fields=self._pii_patch()
            )
            if self.guardrail_engine is not None:
                extra["pii_mapping"] = self._mapping()

            async def complete() -> None:
                await self._store_calls.complete_tool_call(
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
                    replayed=tool_call.id in self._replayed_calls,
                    observation=observation_trace,
                    event=event,
                    loop_state_patch=patch,
                    **extra,
                )

            await retry_critical(complete, label="complete_tool_call", run_id=run_id)
            self._loop_state.update(patch.set_fields)
            self._loop_state["pending"] = [
                p for p in self._loop_state["pending"] if p["call_id"] != tool_call.id
            ]
            self._loop_state["phase"] = "dispatch" if self._loop_state["pending"] else "model_next"
            counts = self._loop_state.setdefault("tool_counts", {})
            counts[tool_call.name] = counts.get(tool_call.name, 0) + 1
            self._completed_observations.add(tool_call.id)
            self._remember(observation_trace)
            return

        if capabilities(self._store).retry_ids:
            extra["row_id"] = generate_ulid()

        if self._format2:
            # No committed intent: this call was rejected before its effect ran.
            extra["outcome"] = "failed"
            from dendrux.types import _message_to_dict

            message = (
                observation if observation is not None else self.format_observation(tool_result)
            )
            extra["meta"] = {**meta, OBSERVATION_META_KEY: _message_to_dict(message)}

        # FAIL-CLOSED with retry: save_tool_call (proof of side effects)
        async def _write_tool() -> None:
            await self._store_calls.save_tool_call(
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
            StoredEventType.TOOL_COMPLETED.value,
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

        if self._format2:
            return  # The preceding fenced event already renewed progress.

        # BEST-EFFORT: touch progress for sweep
        try:
            await self._store_calls.touch_progress(self._run_id)
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
        if self._format2 and event_type == GovernanceEventType.APPROVAL_REQUESTED.value:
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


def resolve_pending(
    loop_state: dict[str, Any],
    tool_rows: Sequence[Any],
    *,
    current_idempotent: dict[str, bool] | None = None,
) -> list[ResolutionAction]:
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
            kind = (
                "resolve"
                if phase == "rejected"
                else "replay"
                if phase == "approved"
                or (
                    read(row, "idempotent", False)
                    and (
                        current_idempotent is None or current_idempotent.get(pending["tool"], False)
                    )
                )
                else "approval_replay"
            )
        elif read(row, "outcome") in ("completed", "failed", "unknown"):
            kind = "observe"
        else:
            raise ValueError("Unsupported tool-call outcome")
        actions.append(ResolutionAction(kind, call_id, pending["order_index"]))
    return actions


async def adopt_managed_result(
    store: StateStore,
    run_id: str,
    *,
    claim: Any,
    call_id: str,
    report: ToolResult,
    child_run_id: str,
    agent_path: Sequence[str],
    observation: Message | None = None,
    pii_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Complete a parent's managed tool intent with its accepted report (0.7).

    One ``complete_tool_call`` transaction under the adoption claim's epoch:
    started intent to completed or failed (its binding metadata kept), the
    reserved observation, the outcome event, the call removed from pending and
    the outgoing binding and submission marker cleared. Ordinary
    ``consume_submission`` is never also called for this observation. A retry
    after a lost acknowledgement finds the completed intent and adds nothing.
    Returns the checkpoint after the change.
    """
    from dendrux.errors import ManagedStateError
    from dendrux.runtime.state import EventWrite, LoopStatePatch, _apply_loop_patch
    from dendrux.types import tool_result_message

    state = dict(claim.loop_state or {})
    binding = state.get("managed_wait") or {}
    pending = next((p for p in state.get("pending", []) if p.get("call_id") == call_id), None)
    if binding.get("tool_call_id") != call_id or binding.get("child_run_id") != child_run_id:
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    if pending is None:
        raise ManagedStateError(reason="pending_missing", run_id=run_id)
    iteration = int(state.get("iteration", 0))
    trace = message_trace(observation or tool_result_message(report), pending["order_index"])
    trace.meta["iteration"] = iteration
    outcome = "completed" if report.success else "failed"
    event = EventWrite(
        generate_ulid(),
        StoredEventType.TOOL_COMPLETED.value,
        iteration,
        {
            "tool_name": report.name,
            "target": "agent",
            "success": report.success,
            "duration_ms": report.duration_ms,
            "child_run_id": child_run_id,
            "agent_path": list(agent_path),
            "managed_outcome": _managed_cancel.outcome_code(report),
            **({"code": report.meta["code"]} if "code" in report.meta else {}),
        },
        call_id,
        None,
    )
    patch = LoopStatePatch(
        remove_pending=(call_id,),
        set_phase="model_next",
        set_fields={"managed_wait": None, "submission": None, "managed_admitted": None},
    )

    async def complete() -> None:
        await store.complete_tool_call(
            run_id,
            claim_epoch=claim.claim_epoch,
            tool_call_id=call_id,
            outcome=outcome,
            result_payload=report.payload,
            success=report.success,
            duration_ms=report.duration_ms,
            error_message=report.error,
            replayed=False,
            observation=trace,
            event=event,
            loop_state_patch=patch,
            meta=None,
            pii_mapping=pii_mapping,
        )

    await retry_critical(complete, label="complete_tool_call", run_id=run_id)
    return _apply_loop_patch(state, patch)
