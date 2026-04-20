"""PersistenceRecorder — authoritative evidence recording.

Internal module. Implements LoopRecorder by writing each event to the
StateStore. The runner creates this when persistence is active.

Write classification:
  - Fail-closed (retry with backoff, then propagate): save_trace, save_tool_call, save_run_event
  - Best-effort (exceptions swallowed): save_usage, save_llm_interaction, touch_progress
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux.loops.base import LoopRecorder
from dendrux.runtime.durability import retry_critical

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore
    from dendrux.types import LLMResponse, Message, ToolCall, ToolDef, ToolResult, ToolTarget

logger = logging.getLogger(__name__)


def _serialize_message(m: Message) -> dict[str, Any]:
    """Full-fidelity serialization of a Message for the evidence layer."""
    d: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.tool_calls is not None:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "provider_tool_call_id": tc.provider_tool_call_id,
                "name": tc.name,
                "params": tc.params,
            }
            for tc in m.tool_calls
        ]
    if m.call_id is not None:
        d["call_id"] = m.call_id
    if m.name is not None:
        d["name"] = m.name
    if m.meta:
        d["meta"] = m.meta
    return d


class PersistenceRecorder(LoopRecorder):
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
        event_sequencer: Any | None = None,
    ) -> None:
        self._store = state_store
        self._run_id = run_id
        self._model = model
        self._provider_name = provider_name
        self._target_lookup = target_lookup or {}
        self._order_index = initial_order_index
        self._event_sequencer = event_sequencer

    async def _emit_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Record a durable run event. FAIL-CLOSED with retry."""
        seq = self._event_sequencer.next() if self._event_sequencer else 0

        async def _write() -> None:
            await self._store.save_run_event(
                self._run_id,
                event_type=event_type,
                sequence_index=seq,
                iteration_index=iteration,
                correlation_id=correlation_id,
                data=data,
            )

        await retry_critical(_write, label="save_run_event", run_id=self._run_id)

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        """Persist a message to react_traces. FAIL-CLOSED."""
        meta = dict(message.meta) if message.meta else {}
        meta["iteration"] = iteration

        if message.tool_calls:
            meta["tool_calls"] = [
                {
                    "id": tc.id,
                    "provider_tool_call_id": tc.provider_tool_call_id,
                    "name": tc.name,
                    "params": tc.params,
                }
                for tc in message.tool_calls
            ]

        if message.call_id:
            meta["call_id"] = message.call_id
        if message.name:
            meta["tool_name"] = message.name

        content = message.content
        order_idx = self._order_index

        # FAIL-CLOSED with retry
        async def _write() -> None:
            await self._store.save_trace(
                self._run_id,
                message.role.value,
                content,
                order_index=order_idx,
                meta=meta,
            )

        await retry_critical(_write, label="save_trace", run_id=self._run_id)
        self._order_index += 1

    async def on_llm_call_completed(
        self,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        """Persist LLM interaction + usage + event."""
        semantic_request = None
        if semantic_messages is not None:
            semantic_request = {
                "messages": [_serialize_message(m) for m in semantic_messages],
            }
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

        semantic_response: dict[str, Any] | None = None
        if response.text is not None or response.tool_calls is not None:
            semantic_response = {}
            if response.text is not None:
                semantic_response["text"] = response.text
            if response.tool_calls is not None:
                semantic_response["tool_calls"] = [
                    {
                        "id": tc.id,
                        "provider_tool_call_id": tc.provider_tool_call_id,
                        "name": tc.name,
                        "params": tc.params,
                    }
                    for tc in response.tool_calls
                ]
            semantic_response["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost_usd": response.usage.cost_usd,
                "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
            }

        # BEST-EFFORT: llm_interactions (full forensics)
        try:
            await self._store.save_llm_interaction(
                self._run_id,
                iteration_index=iteration,
                usage=response.usage,
                model=self._model,
                provider=self._provider_name,
                duration_ms=duration_ms,
                semantic_request=semantic_request,
                semantic_response=semantic_response,
                provider_request=response.provider_request,
                provider_response=response.provider_response,
                guardrail_findings=guardrail_findings,
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
                model=self._model,
                provider=self._provider_name,
            )
        except Exception:
            logger.warning(
                "Failed to save usage for run %s iteration %d",
                self._run_id,
                iteration,
                exc_info=True,
            )

        # FAIL-CLOSED: run event (lifecycle audit trail)
        await self._emit_event(
            "llm.completed",
            iteration,
            {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
                "cost_usd": response.usage.cost_usd,
                "model": self._model,
                "has_tool_calls": bool(response.tool_calls),
            },
        )

        # BEST-EFFORT: touch progress for sweep
        try:
            await self._store.touch_progress(self._run_id)
        except Exception:
            logger.warning("Failed to touch progress for run %s", self._run_id, exc_info=True)

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        """Persist tool call record + event."""
        params = tool_call.params if tool_call.params else None
        target = self._target_lookup.get(tool_call.name, "server")

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
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        """Record a governance event. FAIL-CLOSED with retry."""
        await self._emit_event(
            event_type,
            iteration,
            data=data,
            correlation_id=correlation_id,
        )
