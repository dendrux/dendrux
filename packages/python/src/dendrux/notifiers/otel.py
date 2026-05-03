"""OpenTelemetry notifier for Dendrux.

Emits spans on the developer's existing OTel tracer following the
GenAI semantic conventions. Dendrux runs become visible in whichever
OTel backend the host application already exports to — no extra
setup, no exporter ownership.

Three-level span tree::

    invoke_agent <agent_name>          (run-level root)
    ├── chat <model>                   (per LLM call)
    └── execute_tool <tool_name>       (per tool call)

If a span is already active when the run starts (e.g. a FastAPI
request span), the ``invoke_agent`` span attaches as its child
automatically via OTel context propagation.

Failure policy: this notifier is fail-open. Any exception raised by
the OTel SDK or tracer is swallowed and logged as a warning — runs
must never die because observability is broken.

By default no prompt content, completion text, or tool arguments are
captured. Use ``include_messages=True`` / ``include_tool_params=True``
to opt in (this bypasses Dendrux's PII guardrail redaction — only flip
in trusted environments).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from dendrux.loops.base import BaseNotifier

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, RunResult, ToolCall, ToolDef, ToolResult

try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.trace import Span, Status, StatusCode, Tracer

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    _OTEL_AVAILABLE = False
    Span = Any  # type: ignore[assignment,misc]
    Tracer = Any  # type: ignore[assignment,misc]


_LOG = logging.getLogger(__name__)

_FRAMEWORK = "dendrux"
_OP_INVOKE = "invoke_agent"
_OP_CHAT = "chat"
_OP_TOOL = "execute_tool"


class OpenTelemetryNotifier(BaseNotifier):
    """LoopNotifier that emits OpenTelemetry spans for Dendrux runs.

    Args:
        tracer: An OTel ``Tracer`` instance. If ``None``, a tracer is
            obtained from the global provider via ``trace.get_tracer("dendrux")``.
            Pass an explicit tracer when you want to control the
            instrumentation scope name or version.
        include_tool_params: When ``True``, serialize tool call ``params``
            into ``dendrux.tool.params`` as a JSON string. Bypasses
            guardrail redaction. Defaults to ``False``.
        include_messages: When ``True``, attach the LLM's response text as
            ``gen_ai.completion`` on chat spans. **V1 scope: completion
            text only — prompt/messages are not captured** (capturing them
            requires JSON-serializing multimodal/tool-call content; deferred
            until requested). Bypasses guardrail redaction. Defaults to ``False``.

    Example::

        from dendrux.notifiers.otel import OpenTelemetryNotifier

        result = await agent.run(
            "summarize this PDF",
            notifier=OpenTelemetryNotifier(),
        )
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        *,
        include_tool_params: bool = False,
        include_messages: bool = False,
    ) -> None:
        if not _OTEL_AVAILABLE:
            msg = (
                "OpenTelemetryNotifier requires the OpenTelemetry API. "
                "Install it with: pip install dendrux[otel]"
            )
            raise ImportError(msg)

        self._tracer: Tracer = tracer if tracer is not None else trace.get_tracer(_FRAMEWORK)
        self._include_tool_params = include_tool_params
        self._include_messages = include_messages

        # Per-run span tracking. Concurrent runs are disambiguated by run_id;
        # we never rely on contextvars for our own bookkeeping. Tool spans
        # are keyed by (run_id, tool_call.id) so a misbehaving caller that
        # reuses tool_call.id values across concurrent runs cannot collide.
        self._run_spans: dict[str, Span] = {}
        self._llm_spans: dict[tuple[str, int], Span] = {}
        self._tool_spans: dict[tuple[str, str], Span] = {}
        self._run_models: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Run-level (root span)
    # ------------------------------------------------------------------

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
    ) -> None:
        try:
            span_name = f"{_OP_INVOKE} {agent_name}" if agent_name else _OP_INVOKE
            attributes: dict[str, Any] = {
                "gen_ai.operation.name": _OP_INVOKE,
                "dendrux.framework": _FRAMEWORK,
                "dendrux.run.id": run_id,
            }
            if agent_name:
                attributes["gen_ai.agent.name"] = agent_name
            if agent_model:
                # Stashed for child chat spans; per GenAI semconv, model lives
                # on the LLM call span, not the agent operation span.
                self._run_models[run_id] = agent_model

            span = self._tracer.start_span(span_name, attributes=attributes)
            self._run_spans[run_id] = span
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_started failed: %s", exc)

    async def on_run_finished(self, run_id: str, result: RunResult) -> None:
        # Close any in-flight child spans first — streaming cancellation can
        # bypass on_llm_call_completed/failed (GeneratorExit/CancelledError
        # are BaseException, not Exception). Without this, abandoned streams
        # would leak chat/tool spans even though the run span closes cleanly.
        self._close_orphan_children(run_id, terminal_status=result.status.value)

        span = self._run_spans.pop(run_id, None)
        self._run_models.pop(run_id, None)
        if span is None:
            return
        try:
            span.set_attribute("dendrux.run.status", result.status.value)
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_finished failed: %s", exc)

    async def on_run_failed(
        self,
        run_id: str,
        error: BaseException,
        *,
        iteration: int | None = None,
    ) -> None:
        # Same orphan-cleanup contract as on_run_finished.
        self._close_orphan_children(run_id, terminal_status="error", error=error)

        span = self._run_spans.pop(run_id, None)
        self._run_models.pop(run_id, None)
        if span is None:
            return
        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            if iteration is not None:
                span.set_attribute("dendrux.run.iteration", int(iteration))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_failed failed: %s", exc)

    # ------------------------------------------------------------------
    # LLM call (chat span)
    # ------------------------------------------------------------------

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None:
        try:
            model = self._run_models.get(run_id)
            span_name = f"{_OP_CHAT} {model}" if model else _OP_CHAT
            attributes: dict[str, Any] = {
                "gen_ai.operation.name": _OP_CHAT,
                "dendrux.run.id": run_id,
                "dendrux.run.iteration": iteration,
            }
            if model:
                attributes["gen_ai.request.model"] = model

            span = self._tracer.start_span(
                span_name,
                context=self._child_context(run_id),
                attributes=attributes,
            )
            self._llm_spans[(run_id, iteration)] = span
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_llm_call_started failed: %s", exc)

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
        span = self._llm_spans.pop((run_id, iteration), None)
        if span is None:
            return
        try:
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)

            if duration_ms is not None:
                span.set_attribute("dendrux.llm.duration_ms", int(duration_ms))

            if guardrail_findings:
                span.set_attribute("dendrux.guardrails.hit_count", len(guardrail_findings))

            if self._include_messages and response.text:
                span.set_attribute("gen_ai.completion", response.text)

            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_llm_call_completed failed: %s", exc)

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        span = self._llm_spans.pop((run_id, iteration), None)
        if span is None:
            return
        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            if duration_ms is not None:
                span.set_attribute("dendrux.llm.duration_ms", int(duration_ms))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_llm_call_failed failed: %s", exc)

    # ------------------------------------------------------------------
    # Tool call (execute_tool span)
    # ------------------------------------------------------------------

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        try:
            span_name = f"{_OP_TOOL} {tool_call.name}"
            attributes: dict[str, Any] = {
                "gen_ai.operation.name": _OP_TOOL,
                "dendrux.run.id": run_id,
                "dendrux.run.iteration": iteration,
                "dendrux.tool.name": tool_call.name,
                "dendrux.tool.call_id": tool_call.id,
            }
            if self._include_tool_params and tool_call.params is not None:
                attributes["dendrux.tool.params"] = json.dumps(tool_call.params, default=str)

            span = self._tracer.start_span(
                span_name,
                context=self._child_context(run_id),
                attributes=attributes,
            )
            self._tool_spans[(run_id, tool_call.id)] = span
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_tool_started failed: %s", exc)

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        span = self._tool_spans.pop((run_id, tool_call.id), None)
        if span is None:
            return
        try:
            span.set_attribute("dendrux.tool.success", tool_result.success)
            span.set_attribute("dendrux.tool.duration_ms", tool_result.duration_ms)

            if tool_result.success:
                span.set_status(Status(StatusCode.OK))
            else:
                err_msg = tool_result.error or "tool returned success=false"
                span.set_status(Status(StatusCode.ERROR, err_msg))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_tool_completed failed: %s", exc)

    # ------------------------------------------------------------------
    # Governance events (span events on the run-level span)
    # ------------------------------------------------------------------

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        span = self._run_spans.get(run_id)
        if span is None:
            return
        try:
            attrs: dict[str, Any] = {"dendrux.run.iteration": iteration}
            if correlation_id:
                attrs["dendrux.correlation_id"] = correlation_id
            for k, v in data.items():
                attrs[f"dendrux.governance.{k}"] = _to_attr_value(v)
            span.add_event(f"governance.{event_type}", attributes=attrs)
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_governance_event failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _close_orphan_children(
        self,
        run_id: str,
        *,
        terminal_status: str,
        error: BaseException | None = None,
    ) -> None:
        """Close any chat/tool spans still open for ``run_id``.

        Stream cancellation (GeneratorExit / asyncio.CancelledError) is a
        ``BaseException`` and bypasses the loop's ``except Exception`` paths,
        so ``on_llm_call_completed`` / ``on_llm_call_failed`` may never fire
        for the in-flight LLM call. Without sweeping these on the run-level
        terminal hook, the chat/tool spans leak and Jaeger/Honeycomb show
        them as never-ending operations.

        Each orphan span is closed with status ERROR and a span attribute
        ``dendrux.span.orphan_close_reason`` so operators can spot them.
        Best-effort: any exception per-span is logged and skipped.
        """
        orphan_keys_llm = [k for k in self._llm_spans if k[0] == run_id]
        orphan_keys_tool = [k for k in self._tool_spans if k[0] == run_id]

        if not orphan_keys_llm and not orphan_keys_tool:
            return

        reason = f"run terminated as {terminal_status} before child completed"
        for llm_key in orphan_keys_llm:
            span = self._llm_spans.pop(llm_key, None)
            if span is None:
                continue
            try:
                span.set_attribute("dendrux.span.orphan_close_reason", reason)
                if error is not None:
                    span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, reason))
                span.end()
            except Exception as exc:  # noqa: BLE001 - fail open
                _LOG.warning("orphan chat span close failed: %s", exc)

        for tool_key in orphan_keys_tool:
            span = self._tool_spans.pop(tool_key, None)
            if span is None:
                continue
            try:
                span.set_attribute("dendrux.span.orphan_close_reason", reason)
                span.set_attribute("dendrux.tool.success", False)
                span.set_status(Status(StatusCode.ERROR, reason))
                span.end()
            except Exception as exc:  # noqa: BLE001 - fail open
                _LOG.warning("orphan tool span close failed: %s", exc)

    def _child_context(self, run_id: str) -> Any:
        """Return an OTel context with the run span set as parent.

        If the run span is unknown (e.g. a stray emission after cleanup
        or a lifecycle ordering bug), return an empty ``Context()`` so
        the resulting span becomes a new root in its own trace rather
        than silently parenting under whatever happens to be active —
        which under concurrent runs would mean cross-contaminating
        another run's trace.
        """
        parent = self._run_spans.get(run_id)
        if parent is None:
            return Context()
        return trace.set_span_in_context(parent)

    def __repr__(self) -> str:
        return (
            f"OpenTelemetryNotifier(active_runs={len(self._run_spans)}, "
            f"include_tool_params={self._include_tool_params}, "
            f"include_messages={self._include_messages})"
        )


def _to_attr_value(v: Any) -> Any:
    """Coerce a value to an OTel-acceptable attribute value.

    OTel attribute values are constrained to scalars and homogeneous
    sequences of scalars. Anything else (dicts, mixed lists, dataclasses)
    is JSON-stringified so it survives without dropping data silently.
    """
    if isinstance(v, (str, bool, int, float)):
        return v
    try:
        return json.dumps(v, default=str)
    except (TypeError, ValueError):
        return repr(v)
