"""Pin the provider-failure and image-retry ladder of ReAct and SingleCall, batch and stream.

Captured before H7 moved the ladder into shared helpers. Covers: the
``llm.failed`` record and notify order, SingleCall's cancel-only control read
(a pause is ignored, a cancel ends the run at iteration 0), and the
fresh-error ("raise") path, whose raised error carries no context and whose
loop frame no longer holds the provider failure.
"""

from __future__ import annotations

from typing import Any

import pytest

from dendrux import Agent
from dendrux.errors import ImageRejectedError, UnsupportedModalityError
from dendrux.loops.base import BaseNotifier, BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import ImageViewState, LLMResponse, RunEventType, RunStatus
from tests.unit.test_loop_images import (
    ControlStore,
    ImageLLM,
    StreamScript,
    _history,
    make_png,
    make_runtime,
    rejected,
    shot,
    store_image,
    user_msg,
)


def _spies(log: list[tuple[Any, ...]]) -> tuple[BaseRecorder, BaseNotifier]:
    class Rec(BaseRecorder):
        async def on_llm_call_started(self, run_id, iteration, **kw):
            log.append(("rec.started", iteration))

        async def on_llm_call_failed(self, run_id, iteration, error, *, duration_ms=None):
            log.append(("rec.failed", iteration, type(error).__name__, type(duration_ms)))

        async def on_llm_call_completed(self, run_id, response, iteration, **kw):
            log.append(("rec.completed", iteration, response.text))

    class Not(BaseNotifier):
        async def on_llm_call_started(self, run_id, iteration, **kw):
            log.append(("not.started", iteration))

        async def on_llm_call_failed(self, run_id, iteration, error, *, duration_ms=None):
            log.append(("not.failed", iteration, type(error).__name__, type(duration_ms)))

        async def on_llm_call_completed(self, run_id, response, iteration, **kw):
            log.append(("not.completed", iteration, response.text))

    return Rec(), Not()


def _unsupported() -> UnsupportedModalityError:
    return UnsupportedModalityError(provider="x", model="claude-test", source="provider_response")


async def _single(
    llm: ImageLLM, *, stream: bool, log: list[Any], store: Any = None, **kw: Any
) -> Any:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=7100))
    recorder, notifier = _spies(log)
    kwargs: dict[str, Any] = {
        "agent": Agent(provider=llm, prompt="p", loop=SingleCall()),
        "provider": llm,
        "strategy": NativeToolCalling(),
        "user_input": "x",
        "initial_history": [user_msg("q", part)],
        "image_runtime": rt,
        "image_view_state": ImageViewState(sent_through=1),
        "recorder": recorder,
        "notifier": notifier,
        "state_store": store,
        **kw,
    }
    if not stream:
        return await SingleCall().run(**kwargs)
    events = [e async for e in SingleCall().run_stream(**kwargs)]
    return events[-1]


_RETRY_LOG = [
    ("rec.started", 1),
    ("not.started", 1),
    ("rec.failed", 1, "UnsupportedModalityError", int),
    ("not.failed", 1, "UnsupportedModalityError", int),
    ("rec.started", 1),
    ("not.started", 1),
    ("rec.completed", 1, "ok"),
    ("not.completed", 1, "ok"),
]


@pytest.mark.parametrize("stream", [False, True])
async def test_single_retry_records_failure_then_retries(stream: bool) -> None:
    llm_script: list[Any] = [StreamScript(error=_unsupported()) if stream else _unsupported()]
    llm = ImageLLM([*llm_script, LLMResponse(text="ok")], support="unknown")
    log: list[Any] = []
    out = await _single(llm, stream=stream, log=log)
    result = out.run_result if stream else out
    assert result.status == RunStatus.SUCCESS and result.answer == "ok"
    assert log == _RETRY_LOG


@pytest.mark.parametrize("stream", [False, True])
async def test_single_cancel_before_retry_ends_at_iteration_zero(stream: bool) -> None:
    llm_script: list[Any] = [StreamScript(error=_unsupported()) if stream else _unsupported()]
    llm = ImageLLM([*llm_script, LLMResponse(text="never")], support="unknown")
    store = ControlStore()
    store.cancel_after = 0
    log: list[Any] = []
    out = await _single(llm, stream=stream, log=log, store=store)
    if stream:
        assert out.type == RunEventType.RUN_CANCELLED
        out = out.run_result
    assert out.status == RunStatus.CANCELLED
    assert out.iteration_count == 0 and out.steps == [] and out.meta == {}
    assert out.usage.total_tokens == 0 and len(llm.calls) == 1
    assert log == _RETRY_LOG[:4] and store.reads == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_single_ignores_a_pause_before_retry(stream: bool) -> None:
    llm_script: list[Any] = [StreamScript(error=_unsupported()) if stream else _unsupported()]
    llm = ImageLLM([*llm_script, LLMResponse(text="ok")], support="unknown")
    store = ControlStore()
    store.pause_after = 0
    log: list[Any] = []
    out = await _single(llm, stream=stream, log=log, store=store)
    result = out.run_result if stream else out
    assert result.status == RunStatus.SUCCESS and store.reads == 1
    assert log == _RETRY_LOG


def _loop_frames_hold_failure(exc: BaseException, owners: tuple[str, ...]) -> list[str]:
    """Loop frames in ``exc``'s traceback still holding a provider failure other than ``exc``."""
    held = []
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        failure = frame.f_locals.get("failure") if frame.f_code.co_name in owners else None
        if isinstance(failure, BaseException) and failure is not exc:
            held.append(frame.f_code.co_name)
        tb = tb.tb_next
    return held


async def test_react_batch_fresh_error_drops_the_failure() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=2, users=0)
    llm = ImageLLM([rejected(), rejected()])
    agent = Agent(provider=llm, tools=[shot], prompt="p")
    with pytest.raises(ImageRejectedError) as err:
        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            initial_history=history,
            image_runtime=rt,
        )
    assert err.value.retried is True
    assert err.value.__context__ is None and err.value.__cause__ is None
    assert _loop_frames_hold_failure(err.value, ("run",)) == []


async def test_react_stream_fresh_error_drops_the_failure() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=2, users=0)
    llm = ImageLLM([StreamScript(error=rejected()), StreamScript(error=rejected())])
    agent = Agent(provider=llm, tools=[shot], prompt="p")
    with pytest.raises(ImageRejectedError) as err:
        async for _ in ReActLoop().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            initial_history=history,
            image_runtime=rt,
        ):
            pass
    assert err.value.retried is True
    assert err.value.__context__ is None and err.value.__cause__ is None
    assert _loop_frames_hold_failure(err.value, ("run_stream",)) == []


@pytest.mark.parametrize("stream", [False, True])
async def test_single_failure_after_retry_propagates_unchanged(stream: bool) -> None:
    second_error = _unsupported()
    first: Any = StreamScript(error=_unsupported()) if stream else _unsupported()
    second: Any = StreamScript(error=second_error) if stream else second_error
    llm = ImageLLM([first, second], support="unknown")
    log: list[Any] = []
    with pytest.raises(UnsupportedModalityError) as err:
        await _single(llm, stream=stream, log=log)
    assert err.value is second_error and len(llm.calls) == 2
    assert log == [*_RETRY_LOG[:6], *_RETRY_LOG[2:4]]
    owners = ("run_stream",) if stream else ("run",)
    assert _loop_frames_hold_failure(err.value, owners) == []


@pytest.mark.parametrize("stream", [False, True])
async def test_single_user_image_rejection_raises_without_context(stream: bool) -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=7200))
    original = rejected(part.image_id, located=True)
    llm = ImageLLM([StreamScript(error=original) if stream else original])
    kwargs: dict[str, Any] = {
        "agent": Agent(provider=llm, prompt="p", loop=SingleCall()),
        "provider": llm,
        "strategy": NativeToolCalling(),
        "user_input": "x",
        "initial_history": [user_msg("what", part)],
        "image_runtime": rt,
    }
    with pytest.raises(ImageRejectedError) as err:
        if stream:
            async for _ in SingleCall().run_stream(**kwargs):
                pass
        else:
            await SingleCall().run(**kwargs)
    assert err.value is not original and err.value.retried is False
    assert err.value.__context__ is None and err.value.__cause__ is None
    owners = ("run_stream",) if stream else ("run",)
    assert _loop_frames_hold_failure(err.value, owners) == []
