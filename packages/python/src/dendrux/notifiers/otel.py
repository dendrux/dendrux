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
import sys
import traceback
from itertools import islice
from typing import TYPE_CHECKING, Any

from dendrux.errors import provider_error_in
from dendrux.llm._provider_codes import map_provider_code
from dendrux.loops.base import BaseNotifier
from dendrux.runtime._tracelink import validate_trace_link
from dendrux.types import ErrorInfo

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, RunResult, ToolCall, ToolDef, ToolResult

try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.trace import Span, Status, StatusCode, Tracer
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

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

    def trace_link(self, run_id: str) -> str | None:
        """Return the active run span as a persistable W3C traceparent."""
        return self._span_link(self._run_spans.get(run_id))

    def current_span_link(self, run_id: str, tool_call_id: str) -> str | None:
        """Return only the active span for this run and tool call."""
        return self._span_link(self._tool_spans.get((run_id, tool_call_id)))

    @staticmethod
    def _span_link(span: Span | None) -> str | None:
        if span is None:
            return None
        try:
            carrier: dict[str, str] = {}
            TraceContextTextMapPropagator().inject(
                carrier, context=trace.set_span_in_context(span, Context())
            )
            return validate_trace_link(carrier.get("traceparent"))
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier span link failed: %s", type(exc).__name__)
            return None

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
        trace_link: str | None = None,
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

            link = validate_trace_link(trace_link)
            parent = (
                TraceContextTextMapPropagator().extract({"traceparent": link}, context=Context())
                if link is not None
                else None
            )
            span = self._tracer.start_span(span_name, context=parent, attributes=attributes)
            self._run_spans[run_id] = span
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_started failed: %s", type(exc).__name__)

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
            description = None
            if result.error_info is not None:
                description = _set_error_info(span, result.error_info)
            if result.status.value in {"error", "max_iterations", "cancelled"}:
                span.set_status(Status(StatusCode.ERROR, description))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_finished failed: %s", type(exc).__name__)

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
            _set_span_error(span, error)
            if iteration is not None:
                span.set_attribute("dendrux.run.iteration", int(iteration))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_failed failed: %s", type(exc).__name__)

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
            _LOG.warning("OpenTelemetryNotifier.on_llm_call_started failed: %s", type(exc).__name__)

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

            if semantic_messages:
                self._set_image_attributes(span, run_id, semantic_messages)

            if self._include_messages and response.text:
                span.set_attribute("gen_ai.completion", response.text)

            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning(
                "OpenTelemetryNotifier.on_llm_call_completed failed: %s", type(exc).__name__
            )

    def _set_image_attributes(self, span: Any, run_id: str, messages: list[Message]) -> None:
        """``dendrux.images.count`` / ``.estimated_tokens`` for visible images (ids only)."""
        images = [p for m in messages if m.parts for p in m.images]
        if not images:
            return
        span.set_attribute("dendrux.images.count", len(images))
        model = self._run_models.get(run_id)
        family = _image_family(model)
        if model is None or family is None:
            return
        from dendrux.media._tokens import estimate_image_tokens

        total = 0
        for part in images:
            tokens = estimate_image_tokens(
                part.ref, provider_family=family, model=model, detail=part.detail
            )
            if tokens is None:
                return
            total += tokens
        span.set_attribute("dendrux.images.estimated_tokens", total)

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
            _set_span_error(span, error)
            if duration_ms is not None:
                span.set_attribute("dendrux.llm.duration_ms", int(duration_ms))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_llm_call_failed failed: %s", type(exc).__name__)

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
            _LOG.warning("OpenTelemetryNotifier.on_tool_started failed: %s", type(exc).__name__)

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
                span.set_status(Status(StatusCode.ERROR))
                span.add_event(
                    "exception",
                    attributes={
                        "exception.type": type(tool_result).__name__,
                        "exception.stacktrace": _frame_stack(traceback.walk_stack(sys._getframe())),
                    },
                )
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_tool_completed failed: %s", type(exc).__name__)

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
            _LOG.warning("OpenTelemetryNotifier.on_governance_event failed: %s", type(exc).__name__)

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
                    _set_span_error(span, error)
                else:
                    span.set_status(Status(StatusCode.ERROR, reason))
                span.end()
            except Exception as exc:  # noqa: BLE001 - fail open
                _LOG.warning("orphan chat span close failed: %s", type(exc).__name__)

        for tool_key in orphan_keys_tool:
            span = self._tool_spans.pop(tool_key, None)
            if span is None:
                continue
            try:
                span.set_attribute("dendrux.span.orphan_close_reason", reason)
                span.set_attribute("dendrux.tool.success", False)
                if error is not None:
                    _set_span_error(span, error)
                else:
                    span.set_status(Status(StatusCode.ERROR, reason))
                span.end()
            except Exception as exc:  # noqa: BLE001 - fail open
                _LOG.warning("orphan tool span close failed: %s", type(exc).__name__)

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


def _image_family(model: str | None) -> str | None:
    """Provider family for image token estimates, from the model id."""
    if not model:
        return None
    if "/" in model:
        return "openrouter"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "openai"
    return None


def _set_error_info(span: Any, info: ErrorInfo) -> str:
    projected = info.to_dict()
    mapped = map_provider_code(info.provider, info.provider_code)
    projected["provider_code"] = mapped
    if mapped != info.provider_code:
        status = info.status_code if info.status_code is not None else "n/a"
        projected["safe_message"] = (
            f"{info.provider or 'provider'} {info.category} (HTTP {status}, code {mapped or 'n/a'})"
        )
    fields = {
        f"dendrux.error.{key}": value for key, value in projected.items() if value is not None
    }
    for key, value in fields.items():
        span.set_attribute(key, value)
    span.add_event("provider.error", attributes=fields)
    span.set_status(Status(StatusCode.ERROR, projected["safe_message"]))
    return str(projected["safe_message"])


def _set_span_error(span: Any, error: BaseException) -> None:
    provider_error = provider_error_in(error)
    if provider_error is not None:
        _set_error_info(span, ErrorInfo.from_error(provider_error))
    if provider_error is not error:
        span.add_event(
            "exception",
            attributes={
                "exception.type": type(error).__name__,
                "exception.stacktrace": _frame_stack(traceback.walk_tb(error.__traceback__)),
            },
        )
    if provider_error is None:
        span.set_status(Status(StatusCode.ERROR))


def _frame_stack(frames: Any) -> str:
    return "".join(
        f"  File {frame.f_code.co_filename!r}, line {line}, in {frame.f_code.co_name}\n"
        for frame, line in islice(frames, 32)
    )
