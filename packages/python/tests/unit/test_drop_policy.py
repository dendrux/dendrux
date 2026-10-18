"""Track D1: opt-in drop-old-images view (§5.10, U-08)."""

from __future__ import annotations

from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.errors import ImageLimitError
from dendrux.loops._images import apply_delta
from dendrux.loops.react import ReActLoop
from dendrux.media import ImageDropPolicy, ImagePolicy
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import ImageViewState, LLMResponse, Message, Role, RunStatus
from tests.unit.test_loop_images import (
    ImageLLM,
    RecordingRecorder,
    assistant,
    make_png,
    make_runtime,
    prep,
    store_image,
    tool_msg,
    user_msg,
)


def drop_runtime(keep: int = 2, interval: int = 3, scope: str = "tool_results", **kw: Any) -> Any:
    return make_runtime(
        ImagePolicy(drop=ImageDropPolicy(keep_recent=keep, interval=interval, scope=scope), **kw)
    )


async def tool_history(rt: Any, n: int, *, close: bool = True, seed: int = 0) -> list[Message]:
    msgs: list[Message] = [Message(role=Role.USER, content="go")]
    for i in range(n):
        msgs.append(assistant(call_ids=[f"c{i}"]))
        part = await store_image(rt, make_png(seed=seed + i))
        msgs.append(tool_msg("shot", f"c{i}", part))
    if close:
        msgs.append(assistant("thinking"))
    return msgs


def visible(out: Any) -> list[int]:
    return [s.slot for s in out.slots if s.visible]


async def test_hysteresis_advances_only_past_trigger() -> None:
    rt = drop_runtime(keep=2, interval=3)  # visible eligible stays within [2, 4]
    state = ImageViewState()
    history = await tool_history(rt, 4)
    out = await prep(history, rt, state=state)
    assert visible(out) == [0, 1, 2, 3] and out.delta.dropped_through is None
    history = await tool_history(rt, 5)
    out = await prep(history, rt, state=state)
    assert visible(out) == [3, 4]
    assert out.delta.dropped_through == 3 and out.delta.epoch == 1
    (changed,) = [e.data for e in out.events if e.event_type == "image.view_changed"]
    assert changed == {"epoch": 1, "dropped_through": 3, "visible": 2}
    (degraded,) = [e.data for e in out.events if e.event_type == "image.degraded"]
    assert degraded == {"reason": "drop", "count": 3, "slots": [0, 1, 2]}
    apply_delta(state, out.delta)
    # Two more images: 4 visible eligible, no new advance; old drops are sticky.
    history = await tool_history(rt, 7)
    out2 = await prep(history, rt, state=state)
    assert visible(out2) == [3, 4, 5, 6]
    assert out2.delta.dropped_through is None and out2.events == ()
    assert [s.placeholder for s in out2.slots[:3]] == ["drop"] * 3
    # The prefix bytes up to the last dropped slot are identical across requests.
    assert out2.semantic[:7] == out.semantic[:7]


async def test_current_turn_never_dropped() -> None:
    rt = drop_runtime(keep=1, interval=1)
    history = await tool_history(rt, 3, close=False)  # last tool message is the current turn
    out = await prep(history, rt)
    assert visible(out) == [1, 2]
    assert out.delta.dropped_through == 1


async def test_scope_tool_results_keeps_user_images() -> None:
    rt = drop_runtime(keep=1, interval=1)
    user_part = await store_image(rt, make_png(seed=900))
    history = await tool_history(rt, 3)
    history[0] = user_msg("look", user_part)
    out = await prep(history, rt)
    assert out.slots[0].visible  # user image is not eligible
    assert visible(out) == [0, 3]
    rt_all = drop_runtime(keep=1, interval=1, scope="all")
    rt_all.catalog.rows.update(rt.catalog.rows)  # type: ignore[union-attr]
    rt_all.catalog.data.update(rt.catalog.data)  # type: ignore[union-attr]
    out_all = await prep(history, rt_all)
    assert visible(out_all) == [3]


async def test_dropped_through_never_decreases() -> None:
    rt = drop_runtime(keep=1, interval=1)
    state = ImageViewState(dropped_through=5, epoch=2)
    history = await tool_history(rt, 3)
    out = await prep(history, rt, state=state)
    assert out.delta.dropped_through is None
    apply_delta(state, out.delta)
    assert state.dropped_through == 5 and state.epoch == 2


async def test_forced_prune_on_request_count() -> None:
    # 2 user images (never eligible) + 10 tool images = 12 > cap 11, while the
    # 10 eligible stay under the hysteresis trigger (11): only the cap prunes.
    rt = drop_runtime(keep=2, interval=10, max_images_per_request=11)
    history = await tool_history(rt, 10)
    users = [await store_image(rt, make_png(seed=950 + i)) for i in range(2)]
    history[0] = user_msg("look", *users)
    out = await prep(history, rt)
    assert out.stats.count == 4
    assert out.delta.dropped_through == 10 and out.delta.epoch == 1
    assert visible(out) == [0, 1, 10, 11]


async def test_forced_prune_beats_cap_and_without_policy_raises() -> None:
    rt = drop_runtime(keep=2, interval=10, max_images_per_request=12)
    history = await tool_history(rt, 5)
    tight = drop_runtime(keep=2, interval=10, max_request_image_bytes=200)
    tight.catalog.rows.update(rt.catalog.rows)  # type: ignore[union-attr]
    tight.catalog.data.update(rt.catalog.data)  # type: ignore[union-attr]
    out = await prep(history, tight)
    assert out.stats.count == 2 and out.delta.dropped_through == 3
    plain = make_runtime(ImagePolicy(max_request_image_bytes=200), catalog=rt.catalog)  # type: ignore[arg-type]
    with pytest.raises(ImageLimitError) as err:
        await prep(history, plain)
    assert err.value.scope == "request_bytes"


async def test_drop_with_thinking_warns_once() -> None:
    rt = drop_runtime()
    state = ImageViewState()
    history = await tool_history(rt, 1)
    out = await prep(history, rt, ImageLLM([], thinking=True), state=state)
    assert [e.data for e in out.events if e.event_type == "image.warning"] == [
        {"warning": "drop_with_thinking"}
    ]
    apply_delta(state, out.delta)
    out2 = await prep(history, rt, ImageLLM([], thinking=True), state=state)
    assert not [e for e in out2.events if e.event_type == "image.warning"]


async def test_loop_drop_view_state_persists_in_pause() -> None:
    """Run-level: three tool turns with keep=1/interval=1, then a pause carries the view."""
    from dendrux.media import Image, ToolOutput
    from dendrux.tool import tool
    from dendrux.types import ToolCall, ToolTarget

    counter = {"n": 0}

    @tool()
    async def shot() -> Any:
        """Shot."""
        counter["n"] += 1
        return ToolOutput("ok", images=[Image.from_bytes(make_png(seed=700 + counter["n"]))])

    @tool(target=ToolTarget.CLIENT)
    async def ask() -> str:
        """Client."""
        return ""

    def call(name: str, i: int) -> LLMResponse:
        return LLMResponse(
            tool_calls=[ToolCall(name=name, params={}, provider_tool_call_id=f"p{i}")]
        )

    llm = ImageLLM([call("shot", 1), call("shot", 2), call("shot", 3), call("ask", 4)])
    rt = drop_runtime(keep=1, interval=1)
    rec = RecordingRecorder()
    agent = Agent(provider=llm, tools=[shot, ask], prompt="p")
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        recorder=rec,
        image_runtime=rt,
    )
    assert result.status == RunStatus.WAITING_CLIENT_TOOL
    view = result.meta["pause_state"].image_view
    assert view.dropped_through == 1 and view.epoch == 1 and view.sent_through == 3
    assert view.placeholders == {0: "drop"}
    assert [len(m.images) for m in llm.calls[3] if m.role == Role.TOOL] == [0, 1, 1]
    assert len(rec.of("image.view_changed")) == 1
