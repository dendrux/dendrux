"""PersistenceObserver — bridges the loop's observer hooks to the StateStore.

Implements LoopObserver by writing each event to the StateStore.
The runner creates this and passes it to the loop when persistence is active.

Failure policy: log and continue. Observer failures must not kill agent runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux.loops.base import LoopObserver

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrux.runtime.state import StateStore
    from dendrux.types import LLMResponse, Message, ToolCall, ToolDef, ToolResult, ToolTarget

logger = logging.getLogger(__name__)


def _identity(s: str) -> str:
    return s


def _redact_value(
    v: Any, redact: Callable[[str], str], _stack: set[int] | None = None
) -> Any:
    """Recursively apply a string redactor to all string values.

    Uses an active recursion stack (not a global visited set) so that
    shared sub-objects are redacted correctly on every path, while true
    cycles are replaced with a JSON-safe placeholder.
    """
    if isinstance(v, str):
        return redact(v)
    if isinstance(v, (dict, list)):
        obj_id = id(v)
        if _stack is None:
            _stack = set()
        if obj_id in _stack:
            return "[circular]"
        _stack.add(obj_id)
        try:
            if isinstance(v, dict):
                return {k: _redact_value(val, redact, _stack) for k, val in v.items()}
            return [_redact_value(item, redact, _stack) for item in v]
        finally:
            _stack.discard(obj_id)
    return v


def _redact_dict(d: dict[str, Any], redact: Callable[[str], str]) -> dict[str, Any]:
    """Recursively apply a string redactor to all string values in a dict."""
    return {k: _redact_value(v, redact) for k, v in d.items()}


def _serialize_message(m: Message) -> dict[str, Any]:
    """Full-fidelity serialization of a Message for the evidence layer.

    Preserves role, content, tool_calls (with IDs and params), call_id,
    name, and meta — everything needed to reconstruct the conversation.
    """
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


class PersistenceObserver(LoopObserver):
    """Writes loop events to a StateStore for persistence.

    Tracks order_index internally to ensure react_traces are ordered
    correctly within a single run. Also records durable run_events
    for dashboard timeline reconstruction.

    Args:
        redact: Optional callable that receives a string and returns a
            scrubbed/sanitized string. Applied to trace content, tool
            params (string values), tool result payloads, and error
            messages before persistence. The function receives plain
            strings and must return plain strings (not JSON).
    """

    def __init__(
        self,
        state_store: StateStore,
        run_id: str,
        *,
        model: str | None = None,
        provider_name: str | None = None,
        target_lookup: dict[str, ToolTarget] | None = None,
        redact: Callable[[str], str] | None = None,
        initial_order_index: int = 0,
        event_sequencer: Any | None = None,
    ) -> None:
        self._store = state_store
        self._run_id = run_id
        self._model = model
        self._provider_name = provider_name
        self._target_lookup = target_lookup or {}
        self._redact = redact or _identity
        self._order_index = initial_order_index
        # Shared sequencer from runner — guarantees globally monotonic
        # sequence_index across runner-level and observer-level events.
        self._event_sequencer = event_sequencer

    async def _emit_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Record a durable run event. Failures are logged, never fatal."""
        seq = self._event_sequencer.next() if self._event_sequencer else 0
        try:
            await self._store.save_run_event(
                self._run_id,
                event_type=event_type,
                sequence_index=seq,
                iteration_index=iteration,
                correlation_id=correlation_id,
                data=data,
            )
        except Exception:
            logger.warning(
                "Failed to record run event %s for run %s",
                event_type,
                self._run_id,
                exc_info=True,
            )

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        """Persist a message to react_traces."""
        # Build meta with tool_calls info for assistant messages
        meta = dict(message.meta) if message.meta else {}
        meta["iteration"] = iteration

        if message.tool_calls:
            meta["tool_calls"] = [
                {
                    "id": tc.id,
                    "provider_tool_call_id": tc.provider_tool_call_id,
                    "name": tc.name,
                    "params": _redact_dict(tc.params, self._redact) if tc.params else tc.params,
                }
                for tc in message.tool_calls
            ]

        if message.call_id:
            meta["call_id"] = message.call_id
        if message.name:
            meta["tool_name"] = message.name

        # Apply redaction before persistence
        content = self._redact(message.content)

        await self._store.save_trace(
            self._run_id,
            message.role.value,
            content,
            order_index=self._order_index,
            meta=meta,
        )
        self._order_index += 1

    async def on_llm_call_completed(
        self,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Persist LLM interaction (primary) + token usage (legacy) and record event."""
        # Build semantic payloads — full-fidelity serialization of Dendrux types.
        # These are the authoritative evidence records for debugging and audit.
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
            }

        # Primary write: llm_interactions table
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
            )
        except Exception:
            logger.warning(
                "Failed to save llm_interaction for run %s iteration %d",
                self._run_id,
                iteration,
                exc_info=True,
            )

        # Legacy dual-write: token_usage table (backcompat)
        await self._store.save_usage(
            self._run_id,
            iteration_index=iteration,
            usage=response.usage,
            model=self._model,
            provider=self._provider_name,
        )

        # Durable event for dashboard timeline
        await self._emit_event(
            "llm.completed",
            iteration,
            {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": response.usage.cost_usd,
                "model": self._model,
                "has_tool_calls": bool(response.tool_calls),
            },
        )

        # Touch forward-progress signal for stale-run detection
        try:
            await self._store.touch_progress(self._run_id)
        except Exception:
            logger.warning(
                "Failed to touch progress for run %s", self._run_id, exc_info=True
            )

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        """Persist a tool call record and record tool.completed event."""
        # Redact params (string values only) before persistence
        params = tool_call.params if tool_call.params else None
        if params is not None:
            params = _redact_dict(params, self._redact)

        # Redact tool result payload and error message
        redacted_payload = self._redact(tool_result.payload)
        redacted_error = self._redact(tool_result.error) if tool_result.error else None

        target = self._target_lookup.get(tool_call.name, "server")

        await self._store.save_tool_call(
            self._run_id,
            tool_call_id=tool_call.id,
            provider_tool_call_id=tool_call.provider_tool_call_id,
            tool_name=tool_call.name,
            target=target,
            params=params,
            result_payload=redacted_payload,
            success=tool_result.success,
            duration_ms=tool_result.duration_ms,
            iteration_index=iteration,
            error_message=redacted_error,
        )

        # Durable event for dashboard timeline
        # correlation_id links this event to the tool_call_id for tracing
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

        # Touch forward-progress signal for stale-run detection
        try:
            await self._store.touch_progress(self._run_id)
        except Exception:
            logger.warning(
                "Failed to touch progress for run %s", self._run_id, exc_info=True
            )
