"""Pin the streamed provider-call loop of ReAct and SingleCall, captured before H7b.

One probe provider records, per provider call, when its generator opens, is
pulled and is closed, with the telemetry binding (run id, iteration, sinks)
it sees and whether it ran in the consumer's task. A probe recorder adds the
LLM lifecycle records and the lease scope, and ``TelemetryStream.aclose``
calls are traced (``~`` marks a call on an already-closed stream). The
consumer adds every event it receives. Each cell compares the whole trace.

Cells: text deltas; a tool call on stream; a provider error before the first
event (retried once, and failing again after the retry); a generic error with
no images; an error after a visible delta (never retried); an image rejection
during the stream; an interrupt mid-stream (partial answer); cancel and pause
seen between the failed call and its retry; the consumer closing the stream
at a delta, and the same with a provider whose close raises (the lease is
still released, so the active stream closes before the provider stream).
Error cells also check that no traceback frame still holds the provider
failure or the image-bearing wire request.
"""

from __future__ import annotations

import asyncio
import dataclasses
from contextlib import aclosing, asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from dendrux import Agent, tool
from dendrux.errors import ImageRejectedError, UnsupportedModalityError
from dendrux.llm import _retry_telemetry
from dendrux.llm._retry_telemetry import TelemetryStream
from dendrux.llm.base import LLMProvider
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    ImageViewState,
    LLMResponse,
    RunEvent,
    RunEventType,
    StreamEvent,
    StreamEventType,
    ToolCall,
)
from tests.unit.test_loop_images import (
    ControlStore,
    ImageLLM,
    _history,
    make_png,
    make_runtime,
    rejected,
    shot,
    store_image,
    user_msg,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def text(t: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.TEXT_DELTA, text=t)


def reasoning(t: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.REASONING_DELTA, text=t)


def done(response: LLMResponse) -> StreamEvent:
    return StreamEvent(type=StreamEventType.DONE, raw=response)


ADD = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="p1")


class Script:
    def __init__(
        self,
        *events: StreamEvent,
        error: BaseException | None = None,
        block: bool = False,
        close_error: bool = False,
    ) -> None:
        self.events = list(events)
        self.error = error
        self.block = block
        self.close_error = close_error


class Trace:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.consumer: asyncio.Task[Any] | None = None

    def mark(self, label: str) -> None:
        if _retry_telemetry._run_id.get() is not None:
            label += f" t{_retry_telemetry._iteration.get()}"
            label += "s" if _retry_telemetry._sinks.get() is not None else ""
        if asyncio.current_task() is not self.consumer:
            label += " @other"
        self.lines.append(label)


class Probe(ImageLLM):
    """Each ``complete_stream`` is a real async generator that records its life."""

    def __init__(self, scripts: list[Script], trace: Trace, **kw: Any) -> None:
        super().__init__([], **kw)
        self.scripts = list(scripts)
        self.trace = trace
        self.wires: list[Any] = []
        self.opened = 0

    def complete_stream(self, messages: Any, tools: Any = None, **kw: Any) -> Any:  # type: ignore[override]
        self.wires.append(messages)
        self.opened += 1
        return self._stream(self.opened, self.scripts.pop(0))

    async def _stream(self, n: int, script: Script) -> AsyncGenerator[StreamEvent, None]:
        self.trace.mark(f"open{n}")
        try:
            for event in script.events:
                await asyncio.sleep(0)
                self.trace.mark(f"pull{n}")
                yield event
            await asyncio.sleep(0)
            if script.block:
                await asyncio.Event().wait()
            if script.error is not None:
                raise script.error
        finally:
            self.trace.mark(f"close{n}")
            if script.close_error:
                raise RuntimeError("probe close failed")


class ProbeRecorder(BaseRecorder):
    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    @asynccontextmanager
    async def lease_scope(self) -> Any:
        self.trace.mark("lease+")
        try:
            yield
        finally:
            self.trace.mark("lease-")

    async def on_llm_call_started(self, run_id: str, iteration: int, **kw: Any) -> None:
        self.trace.lines.append(f"rec.started{iteration}")

    async def on_llm_call_failed(
        self, run_id: str, iteration: int, error: BaseException, **kw: Any
    ) -> None:
        self.trace.lines.append(f"rec.failed{iteration}:{type(error).__name__}")

    async def on_llm_call_completed(
        self, run_id: str, response: LLMResponse, iteration: int, **kw: Any
    ) -> None:
        self.trace.lines.append(f"rec.completed{iteration}")

    async def on_message_appended(self, run_id: str, message: Any, iteration: int) -> None:
        self.trace.lines.append(f"rec.msg{iteration}:{message.role.value}")

    async def on_tool_started(self, run_id: str, tool_call: Any, iteration: int) -> None:
        self.trace.lines.append(f"rec.tool+{tool_call.name}")

    async def on_tool_completed(
        self, run_id: str, tool_call: Any, tool_result: Any, iteration: int
    ) -> None:
        self.trace.lines.append(f"rec.tool-{tool_call.name}")

    async def on_governance_event(
        self, run_id: str, event_type: str, iteration: int, data: Any, **kw: Any
    ) -> None:
        self.trace.lines.append(f"gov:{event_type}")


@pytest.fixture
def trace(monkeypatch: pytest.MonkeyPatch) -> Trace:
    """A trace that also sees every ``TelemetryStream`` close, numbered by creation."""
    tr = Trace()
    original_init, original_close = TelemetryStream.__init__, TelemetryStream.aclose
    count = 0

    def init(self: Any, *args: Any, **kw: Any) -> None:
        nonlocal count
        count += 1
        self._probe_n = count
        original_init(self, *args, **kw)

    async def aclose(self: Any) -> None:
        tr.mark(f"tel.aclose{self._probe_n}{'~' if self._closed else ''}")
        await original_close(self)

    monkeypatch.setattr(TelemetryStream, "__init__", init)
    monkeypatch.setattr(TelemetryStream, "aclose", aclose)
    return tr


def _event_line(event: RunEvent) -> str:
    result = event.run_result
    if result is not None:
        return (
            f"END {event.type.value} {result.status.value} it={result.iteration_count} "
            f"answer={result.answer!r} phase={event.phase}"
        )
    detail = event.text if event.text is not None else event.tool_name
    if detail is None and event.tool_call is not None:
        detail = event.tool_call.name
    return f"EV {event.type.value} {detail!r} it={event.iteration} phase={event.phase}"


async def _drive(
    gen: AsyncGenerator[RunEvent, None],
    tr: Trace,
    *,
    stop_at: Callable[[RunEvent], bool] | None = None,
    interrupt: asyncio.Event | None = None,
) -> BaseException | None:
    """Consume ``gen`` in this task; optionally close at an event or fire ``interrupt``."""
    tr.consumer = asyncio.current_task()
    try:
        async with aclosing(gen):
            async for event in gen:
                tr.lines.append(_event_line(event))
                if interrupt is not None and event.type == RunEventType.TEXT_DELTA:
                    interrupt.set()
                if stop_at is not None and stop_at(event):
                    tr.lines.append("CONSUMER CLOSE")
                    break
    except Exception as exc:  # noqa: BLE001 — the trace records what escaped
        tr.lines.append(f"RAISED {type(exc).__name__}")
        return exc
    return None


async def _throw_at_first_delta(
    gen: AsyncGenerator[RunEvent, None], tr: Trace, thrown: BaseException
) -> BaseException | None:
    # ``thrown`` stays bound here; ``_holders`` only scans dendrux frames.
    """Consume ``gen``; at the first text delta, throw ``thrown`` into it (``athrow``)."""
    tr.consumer = asyncio.current_task()
    pending = [thrown]
    try:
        async with aclosing(gen):
            event = await anext(gen)
            while True:
                tr.lines.append(_event_line(event))
                if event.type == RunEventType.TEXT_DELTA and pending:
                    tr.lines.append("CONSUMER THROW")
                    event = await gen.athrow(pending.pop())
                else:
                    event = await anext(gen)
    except StopAsyncIteration:
        return None
    except Exception as exc:  # noqa: BLE001 — the trace records what escaped
        tr.lines.append(f"RAISED {type(exc).__name__}")
        return exc


def _first_delta(event: RunEvent) -> bool:
    return event.type == RunEventType.TEXT_DELTA


def _holders(err: BaseException, targets: list[Any]) -> list[str]:
    """Library traceback frames whose locals are, or directly hold, one of ``targets``."""
    found: list[str] = []
    tb = err.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        # By module, not path: a checkout path may itself contain "dendrux".
        if not frame.f_globals.get("__name__", "").startswith("dendrux."):
            tb = tb.tb_next
            continue
        for name, value in dict(frame.f_locals).items():
            if value is err:
                continue
            if _holds(value, targets):
                found.append(f"{frame.f_code.co_name}.{name}")
        tb = tb.tb_next
    return found


def _holds(value: Any, targets: list[Any], depth: int = 2) -> bool:
    """``value`` is a target, or holds one as an item or attribute, ``depth`` levels down.

    Two levels reach a wire request behind a call object (``call.prepared.wire``).
    The probe provider keeps every wire it received on purpose, so providers are
    not searched.
    """
    if any(value is t for t in targets):
        return True
    if depth == 0 or isinstance(value, LLMProvider):
        return False
    return any(_holds(item, targets, depth - 1) for item in _members(value))


def _members(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (type, str, bytes, int, float)) or callable(value):
        return []
    names = list(getattr(value, "__dict__", {}))
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        names.extend([slots] if isinstance(slots, str) else slots)
    if dataclasses.is_dataclass(value):
        names.extend(f.name for f in dataclasses.fields(value))
    return [getattr(value, name, None) for name in names]


# --------------------------------------------------------------------------- ReAct


async def _react(
    tr: Trace,
    scripts: list[Script],
    *,
    images: bool = False,
    store: Any = None,
    interrupt: asyncio.Event | None = None,
    stop_at: Callable[[RunEvent], bool] | None = None,
    throw: BaseException | None = None,
) -> tuple[Probe, BaseException | None]:
    rt = make_runtime() if images else None
    history = await _history(rt, tools=2, users=0) if rt is not None else None
    probe = Probe(scripts, tr)
    agent = Agent(provider=probe, tools=[shot] if images else [add], prompt="p")
    gen = ReActLoop().run_stream(
        agent=agent,
        provider=probe,
        strategy=NativeToolCalling(),
        user_input="go",
        initial_history=history,
        recorder=ProbeRecorder(tr),
        image_runtime=rt,
        state_store=store,
        interrupt=interrupt,
    )
    if throw is not None:
        return probe, await _throw_at_first_delta(gen, tr, throw)
    err = await _drive(gen, tr, stop_at=stop_at, interrupt=interrupt)
    return probe, err


_R_OPEN = ["rec.msg0:user", "rec.started1"]


async def test_react_text_deltas(trace: Trace) -> None:
    scripts = [Script(reasoning("r"), text("a"), text("b"), done(LLMResponse(text="ab")))]
    _, err = await _react(trace, scripts)
    assert err is None
    assert trace.lines == REACT_TEXT


async def test_react_tool_call_on_stream(trace: Trace) -> None:
    first = LLMResponse(text="c", tool_calls=[ADD])
    scripts = [
        Script(
            text("c"),
            StreamEvent(type=StreamEventType.TOOL_USE_START, tool_name="add", tool_call_id="p1"),
            StreamEvent(
                type=StreamEventType.TOOL_USE_END, tool_call=ADD, tool_name="add", tool_call_id="p1"
            ),
            text("after"),
            done(first),
        ),
        Script(text("3"), done(LLMResponse(text="3"))),
    ]
    _, err = await _react(trace, scripts)
    assert err is None
    assert trace.lines == REACT_TOOL


async def test_react_error_before_first_event_retries_once(trace: Trace) -> None:
    scripts = [Script(error=rejected()), Script(text("ok"), done(LLMResponse(text="ok")))]
    _, err = await _react(trace, scripts, images=True)
    assert err is None
    assert trace.lines == REACT_RETRY


async def test_react_second_failure_raises_and_holds_nothing(trace: Trace) -> None:
    first, second = rejected(), rejected()
    probe, err = await _react(trace, [Script(error=first), Script(error=second)], images=True)
    assert isinstance(err, ImageRejectedError) and err.retried is True
    assert err.__context__ is None and err.__cause__ is None
    assert _holders(err, [first, second, *probe.wires]) == []
    assert trace.lines == REACT_RETRY_FAILS


async def test_react_generic_error_is_not_retried(trace: Trace) -> None:
    boom = RuntimeError("boom")
    _, err = await _react(trace, [Script(error=boom)])
    assert err is boom
    assert trace.lines == REACT_GENERIC


async def test_react_error_after_a_delta_is_not_retried(trace: Trace) -> None:
    first = rejected()
    probe, err = await _react(trace, [Script(text("a"), error=first)], images=True)
    assert isinstance(err, ImageRejectedError) and err is not first and err.retried is False
    assert _holders(err, [first, *probe.wires]) == []
    assert trace.lines == REACT_AFTER_DELTA


async def test_react_interrupt_mid_stream_keeps_partial_answer(trace: Trace) -> None:
    _, err = await _react(
        trace, [Script(text("par"), text("tial"), block=True)], interrupt=asyncio.Event()
    )
    assert err is None
    assert trace.lines == REACT_INTERRUPT


@pytest.mark.parametrize("control", ["cancel", "pause"])
async def test_react_stop_between_retries(trace: Trace, control: str) -> None:
    store = ControlStore()
    # Read 1 is the top-of-iteration gate; read 2 is the one before the retry.
    setattr(store, f"{control}_after", 1)
    _, err = await _react(trace, [Script(error=rejected())], images=True, store=store)
    assert err is None and store.reads == 2
    assert trace.lines == (REACT_STOP_CANCEL if control == "cancel" else REACT_STOP_PAUSE)


async def test_react_consumer_close_at_a_delta(trace: Trace) -> None:
    scripts = [Script(text("a"), text("b"), done(LLMResponse(text="ab")))]
    _, err = await _react(trace, scripts, stop_at=_first_delta)
    assert err is None
    assert trace.lines == REACT_CLOSE


async def test_react_close_failure_still_releases_the_lease(trace: Trace) -> None:
    scripts = [Script(text("a"), text("b"), done(LLMResponse(text="ab")), close_error=True)]
    _, err = await _react(trace, scripts, stop_at=_first_delta)
    assert isinstance(err, RuntimeError) and str(err) == "probe close failed"
    assert trace.lines == REACT_CLOSE_FAILS


async def test_react_exception_thrown_at_a_delta_is_a_failed_call(trace: Trace) -> None:
    thrown = RuntimeError("thrown")
    scripts = [Script(text("a"), text("b"), done(LLMResponse(text="ab")))]
    _, err = await _react(trace, scripts, throw=thrown)
    assert err is thrown
    assert trace.lines == REACT_THROWN


async def test_react_image_error_thrown_at_a_delta_raises_fresh(trace: Trace) -> None:
    thrown = rejected()
    scripts = [Script(text("a"), done(LLMResponse(text="a")))]
    probe, err = await _react(trace, scripts, images=True, throw=thrown)
    assert isinstance(err, ImageRejectedError) and err is not thrown and err.retried is False
    assert err.__context__ is None and err.__cause__ is None
    assert _holders(err, [thrown, *probe.wires]) == []
    assert trace.lines == REACT_THROWN_IMAGE


# --------------------------------------------------------------------------- SingleCall


def _unsupported() -> UnsupportedModalityError:
    return UnsupportedModalityError(provider="x", model="claude-test", source="provider_response")


async def _single(
    tr: Trace,
    scripts: list[Script],
    *,
    images: bool = False,
    store: Any = None,
    interrupt: asyncio.Event | None = None,
    stop_at: Callable[[RunEvent], bool] | None = None,
    view: ImageViewState | None = None,
    throw: BaseException | None = None,
) -> tuple[Probe, BaseException | None]:
    rt = make_runtime() if images else None
    history = None
    if rt is not None:
        part = await store_image(rt, make_png(seed=7300))
        history = [user_msg("q", part)]
    probe = Probe(scripts, tr, support="unknown")
    agent = Agent(provider=probe, prompt="p", loop=SingleCall())
    gen = SingleCall().run_stream(
        agent=agent,
        provider=probe,
        strategy=NativeToolCalling(),
        user_input="x",
        initial_history=history,
        recorder=ProbeRecorder(tr),
        image_runtime=rt,
        image_view_state=view if view is not None else ImageViewState(sent_through=1),
        state_store=store,
        interrupt=interrupt,
    )
    if throw is not None:
        return probe, await _throw_at_first_delta(gen, tr, throw)
    err = await _drive(gen, tr, stop_at=stop_at, interrupt=interrupt)
    return probe, err


async def test_single_text_deltas(trace: Trace) -> None:
    scripts = [Script(reasoning("r"), text("a"), text("b"), done(LLMResponse(text="ab")))]
    _, err = await _single(trace, scripts)
    assert err is None
    assert trace.lines == SINGLE_TEXT


async def test_single_tool_event_on_stream_raises(trace: Trace) -> None:
    scripts = [
        Script(
            text("a"),
            StreamEvent(type=StreamEventType.TOOL_USE_START, tool_name="add", tool_call_id="p1"),
            done(LLMResponse(text="a")),
        )
    ]
    _, err = await _single(trace, scripts)
    assert isinstance(err, RuntimeError) and "unexpected tool_use_start" in str(err)
    assert trace.lines == SINGLE_TOOL


async def test_single_error_before_first_event_retries_once(trace: Trace) -> None:
    scripts = [Script(error=_unsupported()), Script(text("ok"), done(LLMResponse(text="ok")))]
    _, err = await _single(trace, scripts, images=True)
    assert err is None
    assert trace.lines == SINGLE_RETRY


async def test_single_second_failure_propagates_and_holds_nothing(trace: Trace) -> None:
    first, second = _unsupported(), _unsupported()
    probe, err = await _single(trace, [Script(error=first), Script(error=second)], images=True)
    assert err is second
    assert _holders(err, [first, *probe.wires]) == []
    assert trace.lines == SINGLE_RETRY_FAILS


async def test_single_generic_error_is_not_retried(trace: Trace) -> None:
    boom = RuntimeError("boom")
    _, err = await _single(trace, [Script(error=boom)])
    assert err is boom
    assert trace.lines == SINGLE_GENERIC


async def test_single_error_after_a_delta_is_not_retried(trace: Trace) -> None:
    first = _unsupported()
    probe, err = await _single(trace, [Script(text("a"), error=first)], images=True)
    assert err is first
    assert _holders(err, probe.wires) == []
    assert trace.lines == SINGLE_AFTER_DELTA


async def test_single_image_rejection_mid_stream_raises_fresh(trace: Trace) -> None:
    first = rejected()
    probe, err = await _single(trace, [Script(error=first)], images=True, view=ImageViewState())
    assert isinstance(err, ImageRejectedError) and err is not first
    assert err.__context__ is None and err.__cause__ is None
    assert _holders(err, [first, *probe.wires]) == []
    assert trace.lines == SINGLE_REJECTED


async def test_single_interrupt_mid_stream_keeps_partial_answer(trace: Trace) -> None:
    _, err = await _single(
        trace, [Script(text("par"), text("tial"), block=True)], interrupt=asyncio.Event()
    )
    assert err is None
    assert trace.lines == SINGLE_INTERRUPT


@pytest.mark.parametrize("control", ["cancel", "pause"])
async def test_single_stop_between_retries(trace: Trace, control: str) -> None:
    store = ControlStore()
    setattr(store, f"{control}_after", 0)
    scripts = [Script(error=_unsupported()), Script(text("ok"), done(LLMResponse(text="ok")))]
    _, err = await _single(trace, scripts, images=True, store=store)
    assert err is None and store.reads == 1
    assert trace.lines == (SINGLE_STOP_CANCEL if control == "cancel" else SINGLE_RETRY)


async def test_single_consumer_close_at_a_delta(trace: Trace) -> None:
    scripts = [Script(text("a"), text("b"), done(LLMResponse(text="ab")))]
    _, err = await _single(trace, scripts, stop_at=_first_delta)
    assert err is None
    assert trace.lines == SINGLE_CLOSE


async def test_single_close_failure_still_releases_the_lease(trace: Trace) -> None:
    scripts = [Script(text("a"), text("b"), done(LLMResponse(text="ab")), close_error=True)]
    _, err = await _single(trace, scripts, stop_at=_first_delta)
    assert isinstance(err, RuntimeError) and str(err) == "probe close failed"
    assert trace.lines == SINGLE_CLOSE_FAILS


async def test_single_exception_thrown_at_a_delta_is_a_failed_call(trace: Trace) -> None:
    thrown = RuntimeError("thrown")
    scripts = [Script(text("a"), text("b"), done(LLMResponse(text="ab")))]
    _, err = await _single(trace, scripts, throw=thrown)
    assert err is thrown
    assert trace.lines == SINGLE_THROWN


async def _completion_record_fails(self: Any, *args: Any, **kw: Any) -> None:
    raise RuntimeError("completion record unavailable")


def _sent_image_bytes(wires: list[Any]) -> bool:
    return any(getattr(part, "data", None) for w in wires for m in w for part in m.parts or [])


@pytest.mark.parametrize("loop", ["react", "single"])
async def test_success_then_record_error_holds_no_wire(
    trace: Trace, monkeypatch: pytest.MonkeyPatch, loop: str
) -> None:
    """An image call succeeds, then recording it fails: no frame keeps the image request."""
    monkeypatch.setattr(ProbeRecorder, "on_llm_call_completed", _completion_record_fails)
    drive = _react if loop == "react" else _single
    scripts = [Script(text("ok"), done(LLMResponse(text="ok")))]
    probe, err = await drive(trace, scripts, images=True)
    assert isinstance(err, RuntimeError) and str(err) == "completion record unavailable"
    assert _sent_image_bytes(probe.wires)
    assert _holders(err, probe.wires) == []


# --------------------------------------------------------------------------- expected traces
# Captured from the code before H7b (2320646d).

REACT_TEXT = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV reasoning_delta 'r' it=1 phase=None",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=None",
    "pull1 t1s",
    "EV text_delta 'b' it=1 phase=None",
    "pull1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "tel.aclose1~",
    "lease-",
    "tel.aclose1~",
    "rec.completed1",
    "rec.msg1:assistant",
    "END run_completed success it=1 answer='ab' phase=final_answer",
]
REACT_TOOL = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'c' it=1 phase=None",
    "pull1 t1s",
    "EV tool_use_start 'add' it=1 phase=commentary",
    "pull1 t1s",
    "EV tool_use_end 'add' it=1 phase=commentary",
    "pull1 t1s",
    "EV text_delta 'after' it=1 phase=commentary",
    "pull1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "tel.aclose1~",
    "lease-",
    "tel.aclose1~",
    "rec.completed1",
    "rec.msg1:assistant",
    "rec.tool+add",
    "lease+",
    "lease-",
    "rec.tool-add",
    "rec.msg1:tool",
    "EV tool_result 'add' it=1 phase=commentary",
    "rec.started2",
    "lease+",
    "open2 t2s",
    "pull2 t2s",
    "EV text_delta '3' it=2 phase=None",
    "pull2 t2s",
    "close2 t2s",
    "tel.aclose2",
    "tel.aclose2~",
    "lease-",
    "tel.aclose2~",
    "rec.completed2",
    "rec.msg2:assistant",
    "END run_completed success it=2 answer='3' phase=final_answer",
]
REACT_RETRY = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:ImageRejectedError",
    "gov:image.rejected",
    "gov:image.rejected",
    "tel.aclose1~",
    "gov:image.degraded",
    "gov:image.state",
    "rec.started1",
    "lease+",
    "open2 t1s",
    "pull2 t1s",
    "EV text_delta 'ok' it=1 phase=None",
    "pull2 t1s",
    "close2 t1s",
    "tel.aclose2",
    "tel.aclose2~",
    "lease-",
    "tel.aclose2~",
    "rec.completed1",
    "rec.msg1:assistant",
    "END run_completed success it=1 answer='ok' phase=final_answer",
]
REACT_RETRY_FAILS = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:ImageRejectedError",
    "gov:image.rejected",
    "gov:image.rejected",
    "tel.aclose1~",
    "gov:image.degraded",
    "gov:image.state",
    "rec.started1",
    "lease+",
    "open2 t1s",
    "close2 t1s",
    "tel.aclose2",
    "lease-",
    "rec.failed1:ImageRejectedError",
    "tel.aclose2~",
    "RAISED ImageRejectedError",
]
REACT_GENERIC = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:RuntimeError",
    "tel.aclose1~",
    "RAISED RuntimeError",
]
REACT_AFTER_DELTA = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=None",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:ImageRejectedError",
    "gov:image.rejected",
    "gov:image.rejected",
    "tel.aclose1~",
    "RAISED ImageRejectedError",
]
REACT_INTERRUPT = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'par' it=1 phase=None",
    "pull1 t1s",
    "EV text_delta 'tial' it=1 phase=None",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:StreamInterruptedError",
    "tel.aclose1~",
    "END run_cancelled cancelled it=1 answer='partial' phase=None",
]
REACT_STOP_CANCEL = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:ImageRejectedError",
    "gov:image.rejected",
    "gov:image.rejected",
    "tel.aclose1~",
    "END run_cancelled cancelled it=0 answer=None phase=None",
]
REACT_STOP_PAUSE = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:ImageRejectedError",
    "gov:image.rejected",
    "gov:image.rejected",
    "tel.aclose1~",
    "END run_paused paused it=0 answer=None phase=None",
]
REACT_CLOSE = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=None",
    "CONSUMER CLOSE",
    "tel.aclose1",
    "close1 t1s",
    "lease-",
    "tel.aclose1~",
]
REACT_CLOSE_FAILS = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=None",
    "CONSUMER CLOSE",
    "tel.aclose1",
    "close1 t1s",
    "lease-",
    "RAISED RuntimeError",
]
SINGLE_TEXT = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV reasoning_delta 'r' it=1 phase=None",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=final_answer",
    "pull1 t1s",
    "EV text_delta 'b' it=1 phase=final_answer",
    "pull1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "tel.aclose1~",
    "lease-",
    "tel.aclose1~",
    "rec.completed1",
    "rec.msg1:assistant",
    "END run_completed success it=1 answer='ab' phase=final_answer",
]
SINGLE_TOOL = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=final_answer",
    "pull1 t1s",
    "rec.failed1:RuntimeError",
    "tel.aclose1",
    "close1 t1s",
    "lease-",
    "tel.aclose1~",
    "RAISED RuntimeError",
]
SINGLE_RETRY = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:UnsupportedModalityError",
    "tel.aclose1~",
    "gov:image.degraded",
    "gov:image.state",
    "rec.started1",
    "lease+",
    "open2 t1s",
    "pull2 t1s",
    "EV text_delta 'ok' it=1 phase=final_answer",
    "pull2 t1s",
    "close2 t1s",
    "tel.aclose2",
    "tel.aclose2~",
    "lease-",
    "tel.aclose2~",
    "rec.completed1",
    "rec.msg1:assistant",
    "END run_completed success it=1 answer='ok' phase=final_answer",
]
SINGLE_RETRY_FAILS = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:UnsupportedModalityError",
    "tel.aclose1~",
    "gov:image.degraded",
    "gov:image.state",
    "rec.started1",
    "lease+",
    "open2 t1s",
    "close2 t1s",
    "tel.aclose2",
    "lease-",
    "rec.failed1:UnsupportedModalityError",
    "tel.aclose2~",
    "RAISED UnsupportedModalityError",
]
SINGLE_GENERIC = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:RuntimeError",
    "tel.aclose1~",
    "RAISED RuntimeError",
]
SINGLE_AFTER_DELTA = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=final_answer",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:UnsupportedModalityError",
    "tel.aclose1~",
    "RAISED UnsupportedModalityError",
]
SINGLE_REJECTED = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:ImageRejectedError",
    "gov:image.rejected",
    "tel.aclose1~",
    "RAISED ImageRejectedError",
]
SINGLE_INTERRUPT = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'par' it=1 phase=final_answer",
    "pull1 t1s",
    "EV text_delta 'tial' it=1 phase=final_answer",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:StreamInterruptedError",
    "tel.aclose1~",
    "END run_cancelled cancelled it=1 answer='partial' phase=None",
]
SINGLE_STOP_CANCEL = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "close1 t1s",
    "tel.aclose1",
    "lease-",
    "rec.failed1:UnsupportedModalityError",
    "tel.aclose1~",
    "END run_cancelled cancelled it=0 answer=None phase=None",
]
SINGLE_CLOSE = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=final_answer",
    "CONSUMER CLOSE",
    "tel.aclose1",
    "close1 t1s",
    "lease-",
    "tel.aclose1~",
]
SINGLE_CLOSE_FAILS = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=final_answer",
    "CONSUMER CLOSE",
    "tel.aclose1",
    "close1 t1s",
    "lease-",
    "RAISED RuntimeError",
]
REACT_THROWN = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=None",
    "CONSUMER THROW",
    "rec.failed1:RuntimeError",
    "tel.aclose1",
    "close1 t1s",
    "lease-",
    "tel.aclose1~",
    "RAISED RuntimeError",
]
REACT_THROWN_IMAGE = [
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=None",
    "CONSUMER THROW",
    "rec.failed1:ImageRejectedError",
    "gov:image.rejected",
    "gov:image.rejected",
    "tel.aclose1",
    "close1 t1s",
    "lease-",
    "tel.aclose1~",
    "RAISED ImageRejectedError",
]
SINGLE_THROWN = [
    "rec.msg0:user",
    "rec.started1",
    "lease+",
    "open1 t1s",
    "pull1 t1s",
    "EV text_delta 'a' it=1 phase=final_answer",
    "CONSUMER THROW",
    "rec.failed1:RuntimeError",
    "tel.aclose1",
    "close1 t1s",
    "lease-",
    "tel.aclose1~",
    "RAISED RuntimeError",
]
