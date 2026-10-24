"""PersistenceRecorder — authoritative evidence recording.

Internal module. Implements LoopRecorder by writing each event to the
StateStore. The runner creates this when persistence is active.

Write classification:
  - Fail-closed (retry with backoff, then propagate): save_trace, save_tool_call, save_run_event
  - Best-effort (exceptions swallowed): save_usage, save_llm_interaction, touch_progress

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

import logging
from typing import TYPE_CHECKING, Any

from dendrux.errors import ImagePersistenceUnsupportedError
from dendrux.loops.base import BaseRecorder
from dendrux.runtime._blob_catalog import edges_for_parts, scrub_inline_images
from dendrux.runtime.durability import retry_critical
from dendrux.types import (
    ImagePart,
    ImageRef,
    content_part_to_dict,
    generate_ulid,
    tool_call_parse_meta,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    ) -> None:
        self._store = state_store
        self._tenant_scope = tenant_scope
        self._provider = provider
        self._run_id = run_id
        self._model = model
        self._provider_name = provider_name
        self._target_lookup = target_lookup or {}
        self._order_index = initial_order_index

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

        async def _write() -> None:
            await self._store.save_run_event(
                self._run_id,
                event_type=event_type,
                iteration_index=iteration,
                correlation_id=correlation_id,
                data=data,
            )

        await retry_critical(_write, label="save_run_event", run_id=self._run_id)

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None:
        """Persist a message to react_traces. FAIL-CLOSED."""
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

        # FAIL-CLOSED with retry
        async def _write() -> None:
            await self._store.save_trace(
                self._run_id,
                message.role.value,
                content,
                order_index=order_idx,
                meta=meta,
                **edge_kwargs,
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
        except Exception:
            logger.warning(
                "Failed to save llm_interaction for run %s iteration %d",
                self._run_id,
                iteration,
                exc_info=True,
            )

        # BEST-EFFORT: token_usage (cost tracking)
        try:
            await self._store.save_usage(
                self._run_id,
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
        except Exception:
            logger.warning(
                "Failed to save usage for run %s iteration %d",
                self._run_id,
                iteration,
                exc_info=True,
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
        except Exception:
            logger.warning("Failed to touch progress for run %s", self._run_id, exc_info=True)

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        """Touch ``last_progress_at`` at dispatch so a long tool starts fresh. BEST-EFFORT."""
        try:
            await self._store.touch_progress(self._run_id)
        except Exception:
            logger.warning("Failed to touch progress for run %s", self._run_id, exc_info=True)

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
                **tool_call_parse_meta(tool_call),
                **({"image_count": len(images)} if images else {}),
            },
            correlation_id=tool_call.id,
        )

        # BEST-EFFORT: touch progress for sweep
        try:
            await self._store.touch_progress(self._run_id)
        except Exception:
            logger.warning("Failed to touch progress for run %s", self._run_id, exc_info=True)

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
        await self._emit_event(
            event_type,
            iteration,
            data=data,
            correlation_id=correlation_id,
        )
