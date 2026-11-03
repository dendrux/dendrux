"""Provider-call bookkeeping shared by the ReAct and SingleCall loops.

The failure record, the image-retry decision and the priced response are the
same in every loop and mode; only what a stop means differs (ReAct pauses or
cancels, SingleCall honours cancel only), so callers keep that part.

``stream_call_events`` is the streamed provider call of both loops. It is an
async generator the loop drives itself (``aclosing`` + ``EventRelay``), so
every provider pull, close and thrown exception happens in the consumer's
task and context; nothing here starts a task.

This module sits above ``_images`` (which imports ``_helpers``), so these
helpers cannot live in ``_helpers`` without an import cycle.
"""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any

from dendrux.llm._calls import stream_call
from dendrux.llm._evidence import persistable_response
from dendrux.llm._retry_telemetry import telemetry_stream
from dendrux.loops._helpers import (
    StreamInterruptedError,
    governance_sinks,
    interruptible,
    leased_stream,
    notify_llm_failed,
    notify_llm_started,
    price_response,
    record_llm_failed,
    record_llm_started,
)
from dendrux.loops._images import control_stop, release_wire
from dendrux.types import StreamEventType

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from dendrux.agent import Agent
    from dendrux.llm._retry_telemetry import TelemetryStream
    from dendrux.llm.base import LLMProvider
    from dendrux.loops._helpers import _LeasedStream
    from dendrux.loops._images import ImageCallSite, RetryDecision
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.media._prepared import PreparedRequest
    from dendrux.types import LLMResponse, Message, RunEvent, StreamEvent, ToolDef


def drop_frame_locals(exc: BaseException) -> None:
    """Clear finished frames' locals in ``exc``'s traceback (they may hold wire bytes)."""
    traceback.clear_frames(exc.__traceback__)


class ProviderCallSite:
    """One run's provider calls: failure records, the retry decision, the priced response."""

    __slots__ = ("agent", "images", "notifier", "pkw", "provider", "recorder", "run_id", "warnings")

    def __init__(
        self,
        images: ImageCallSite,
        *,
        agent: Agent,
        provider: LLMProvider,
        pkw: dict[str, Any],
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
        run_id: str,
        warnings: list[str],
    ) -> None:
        self.images = images
        self.agent = agent
        self.provider = provider
        self.pkw = pkw
        self.recorder = recorder
        self.notifier = notifier
        self.run_id = run_id
        self.warnings = warnings

    async def started(
        self, prepared: PreparedRequest, tools: list[ToolDef] | None, iteration: int
    ) -> None:
        """``llm.started`` for a provider call about to begin: record, then notify."""
        await record_llm_started(
            self.recorder,
            self.run_id,
            iteration,
            semantic_messages=prepared.semantic,
            semantic_tools=tools,
        )
        await notify_llm_started(
            self.notifier,
            self.run_id,
            iteration,
            self.warnings,
            semantic_messages=prepared.semantic,
            semantic_tools=tools,
        )

    async def record_failure(self, exc: BaseException, t0: float, iteration: int) -> None:
        """``llm.failed`` for a provider call that raised: record, then notify."""
        duration_ms = int((time.monotonic() - t0) * 1000)
        await record_llm_failed(self.recorder, self.run_id, iteration, exc, duration_ms=duration_ms)
        await notify_llm_failed(
            self.notifier, self.run_id, iteration, exc, self.warnings, duration_ms=duration_ms
        )

    async def decide(
        self,
        failure: Exception,
        prepared: PreparedRequest,
        iteration: int,
        *,
        retried: bool,
        visible_deltas: int,
        control_reader: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, BaseException | None, RetryDecision]:
        """What a failed call does next; call it outside the ``except`` block.

        Returns ``(stop, error, decision)``. ``stop`` is the cancel or pause
        control seen before a retry (the caller ends the run its own way).
        Otherwise ``error`` is what to raise: ``failure`` itself (its frames'
        locals cleared) or the decision's fresh, cause-free error. With
        neither, the caller retries with ``decision.tombstones``. The caller
        drops its own ``failure`` before raising, so the raised error's
        traceback never keeps the provider failure alive.
        """
        decision = await self.images.on_failure(
            failure,
            prepared,
            iteration=iteration,
            retried=retried,
            visible_deltas=visible_deltas,
            control_reader=control_reader,
        )
        stop = control_stop(decision)
        if stop is not None:
            return stop, None, decision
        if decision.action == "propagate":
            drop_frame_locals(failure)
            return None, failure, decision
        if decision.action == "raise":
            assert decision.error is not None
            return None, decision.error, decision
        return None, None, decision

    def settle(
        self, prepared: PreparedRequest, response: LLMResponse, t0: float
    ) -> tuple[PreparedRequest, int, LLMResponse]:
        """After the provider returns: commit the image view, time the call, price it."""
        prepared = self.images.mark_sent(prepared)
        duration_ms = int((time.monotonic() - t0) * 1000)
        response = persistable_response(response)
        response = price_response(
            response,
            pricing=self.agent.pricing,
            default_model=self.provider.model,
            model=self.pkw.get("model"),
        )
        return prepared, duration_ms, response


class StreamCall:
    """One streamed provider call of an iteration, and its one image retry.

    ``stream_call_events`` drives it. When the events end the loop reads the
    outcome: ``stop`` (a cancel or pause seen before a retry), ``interrupted``
    (with ``partial_text``), or else ``response`` with ``prepared`` and ``t0``
    for :meth:`ProviderCallSite.settle`. ``translate`` turns a provider event
    into the loop's event (None to skip it); ``check`` validates the response
    once the stream ends, inside the failure handling.
    """

    __slots__ = (
        "active",
        "cache_key_prefix",
        "calls",
        "check",
        "control_reader",
        "failure",
        "interrupt",
        "interrupted",
        "iteration",
        "messages",
        "partial_text",
        "prepared",
        "provider_stream",
        "response",
        "retried",
        "stop",
        "t0",
        "tools",
        "translate",
        "visible",
    )

    active: _LeasedStream
    provider_stream: TelemetryStream

    def __init__(
        self,
        calls: ProviderCallSite,
        prepared: PreparedRequest,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None,
        iteration: int,
        interrupt: asyncio.Event | None,
        control_reader: Callable[[], Awaitable[Any]],
        cache_key_prefix: str | None,
        translate: Callable[[StreamEvent], RunEvent | None],
        check: Callable[[LLMResponse | None, int], None],
    ) -> None:
        self.calls = calls
        self.prepared = prepared
        self.messages = messages
        self.tools = tools
        self.iteration = iteration
        self.interrupt = interrupt
        self.control_reader = control_reader
        self.cache_key_prefix = cache_key_prefix
        self.translate = translate
        self.check = check
        self.response: LLMResponse | None = None
        self.t0 = 0.0
        self.partial_text: list[str] = []
        self.interrupted = False
        self.retried = False
        self.stop: Any = None
        self.failure: Exception | None = None
        self.visible = 0

    async def open(self) -> None:
        """``llm.started``, the clock, then a provider stream bound to the run's telemetry."""
        calls = self.calls
        await calls.started(self.prepared, self.tools, self.iteration)
        self.t0 = time.monotonic()
        self.provider_stream = telemetry_stream(
            stream_call(
                calls.provider,
                self.prepared.wire,
                tools=self.tools,
                run_id=calls.run_id,
                cache_key_prefix=self.cache_key_prefix,
                **calls.pkw,
            ),
            run_id=calls.run_id,
            iteration=self.iteration,
            sinks=governance_sinks(calls.recorder, calls.notifier, calls.run_id),
        )
        # Externally visible deltas of this provider call (§5.7 guard).
        self.visible = 0
        self.active = leased_stream(
            interruptible(self.provider_stream, self.interrupt), calls.recorder
        )

    def take(self, event: StreamEvent) -> RunEvent | None:
        """The loop event for ``event``; DONE keeps the response and yields nothing."""
        if event.type == StreamEventType.DONE:
            self.response = event.raw
            return None
        run_event = self.translate(event)
        if run_event is not None:
            if event.type == StreamEventType.TEXT_DELTA:
                self.partial_text.append(event.text or "")
            self.visible += 1
        return run_event

    async def failed(self, exc: Exception) -> bool:
        """Record a failed attempt; True when ``exc`` must propagate unchanged.

        An interrupt is checked first: it ends the call even without images.
        Otherwise a request with image slots keeps the failure for
        :meth:`retry` and drops its wire bytes.
        """
        await self.calls.record_failure(exc, self.t0, self.iteration)
        if isinstance(exc, StreamInterruptedError):
            self.interrupted = True
            return False
        if not self.prepared.slots:
            return True
        self.failure = exc
        self.prepared = release_wire(self.prepared)
        return False

    async def retry(self) -> bool:
        """After an attempt: True when a retry stream is open, False when the call is over.

        Raises the decision's error. The failure is dropped before raising, so
        no frame keeps the provider failure alive.
        """
        failure, self.failure = self.failure, None
        if failure is None:
            return False
        stop, error, decision = await self.calls.decide(
            failure,
            self.prepared,
            self.iteration,
            retried=self.retried,
            visible_deltas=self.visible,
            control_reader=self.control_reader,
        )
        del failure
        if stop is not None:
            self.stop = stop
            return False
        if error is not None:
            raise error
        # Retry in place (N-15.2): close the first stream, keep the telemetry
        # context, open a new stream for the new view.
        self.retried = True
        await self.provider_stream.aclose()
        self.prepared = await self.calls.images.prepare(
            self.messages, iteration=self.iteration, force_tombstones=decision.tombstones
        )
        await self.open()
        return True

    async def close(self) -> None:
        """Close the active stream, then the provider stream (both idempotent)."""
        await self.active.aclose()
        await self.provider_stream.aclose()


async def stream_call_events(call: StreamCall) -> AsyncGenerator[RunEvent, None]:
    """Stream one provider call as the loop's events, retrying an image failure once.

    The caller drives this generator in its own task and closes it with
    ``aclosing``; it never starts a task. Every exit, including a close at a
    yielded delta, closes the active stream and then the provider stream.
    """
    await call.open()
    try:
        while True:
            try:
                async for event in call.active:
                    run_event = call.take(event)
                    if run_event is not None:
                        yield run_event
                # Validate inside the same try so a contract-violating provider
                # stream still pairs with on_llm_call_failed.
                call.check(call.response, call.iteration)
            except Exception as exc:
                if await call.failed(exc):
                    raise
            if not await call.retry():
                return
    finally:
        await call.close()


class EventRelay:
    """Re-yield ``events`` from a loop generator, passing thrown exceptions back in.

    A plain ``async for`` would raise an exception thrown into the loop at a
    re-yielded delta in the loop itself; the relay hands it to the call via
    ``athrow``, so it is recorded as the failed call it was before the call
    moved out of the loop. Not a generator: closing stays with ``aclosing``.
    """

    __slots__ = ("events", "thrown")

    def __init__(self, events: AsyncGenerator[RunEvent, None]) -> None:
        self.events = events
        self.thrown: Exception | None = None

    def throw(self, exc: Exception) -> None:
        """Pass ``exc`` into the call at the next :meth:`next`."""
        self.thrown = exc

    async def next(self) -> RunEvent | None:
        """The next event, or None once the call's events have ended."""
        try:
            if self.thrown is None:
                return await anext(self.events)
            return await self.events.athrow(self._take())
        except StopAsyncIteration:
            return None

    def _take(self) -> Exception:
        thrown, self.thrown = self.thrown, None
        assert thrown is not None
        return thrown
