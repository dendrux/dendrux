"""The side-effect counters every fan-out test asserts on (0.8 spec §16)."""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import sys
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest

from dendrux.tool import get_tool_def, tool
from dendrux.types import LLMResponse, StreamEventType, ToolResultContext, ToolTarget
from tests._helpers.fanout_effects import CountingProvider, EffectCounts, FanoutEffects
from tests._helpers.gated_provider import GatedStreamLLM
from tests._helpers.scripted_provider import ScriptedLLM


def _ctx(call_id: str, result: Any = "raw", tool_name: str = "researcher") -> ToolResultContext:
    return ToolResultContext(
        tool_name=tool_name,
        call_id=call_id,
        params={},
        result=result,
        target=ToolTarget.SERVER,
        meta={},
    )


# --- tool counters: per (child, tool) ------------------------------------------


def test_a_new_recorder_has_no_effects() -> None:
    counts = FanoutEffects().counts()
    assert counts == EffectCounts(tools={}, provider={}, transforms={})
    assert counts.tool("A", "save_note") == 0
    assert counts.provider_calls("A") == 0
    assert counts.transform_calls() == 0


def test_tool_effects_are_counted_per_child_and_tool() -> None:
    effects = FanoutEffects()
    effects.record_tool("A", "save_note")
    effects.record_tool("A", "save_note")
    effects.record_tool("A", "search")
    effects.record_tool("B", "save_note")
    counts = effects.counts()
    assert counts.tools == {("A", "save_note"): 2, ("A", "search"): 1, ("B", "save_note"): 1}
    assert counts.tool("A", "save_note") == 2
    assert counts.tool("B", "search") == 0


async def test_counting_wraps_an_async_tool_and_keeps_its_shape() -> None:
    effects = FanoutEffects()

    @effects.counting("A")
    async def save_note(text: str, urgent: bool = False) -> str:
        """Save a note."""
        return f"saved {text}"

    assert inspect.iscoroutinefunction(save_note)
    assert save_note.__name__ == "save_note"
    assert list(inspect.signature(save_note).parameters) == ["text", "urgent"]
    assert await save_note("x") == "saved x"
    assert await save_note(text="y", urgent=True) == "saved y"
    assert effects.counts().tools == {("A", "save_note"): 2}


def test_counting_wraps_a_sync_tool_and_accepts_an_explicit_name() -> None:
    effects = FanoutEffects()

    @effects.counting("B", name="lookup")
    def find(key: str) -> str:
        return key.upper()

    assert not inspect.iscoroutinefunction(find)
    assert find("k") == "K"
    assert effects.counts().tools == {("B", "lookup"): 1}


def test_a_tool_that_raises_still_counts_its_attempt() -> None:
    effects = FanoutEffects()

    @effects.counting("A")
    def boom() -> None:
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        boom()
    assert effects.counts().tool("A", "boom") == 1


def test_a_counting_tool_registers_as_a_dendrux_tool_with_the_same_schema() -> None:
    effects = FanoutEffects()

    async def plain(text: str, count: int = 1) -> str:
        """Doc."""
        return text * count

    counted = tool()(effects.counting("A", name="plain")(plain))
    expected = get_tool_def(tool()(plain))
    assert get_tool_def(counted).name == "plain"
    assert get_tool_def(counted).parameters == expected.parameters
    assert get_tool_def(counted).description == "Doc."


# --- provider counters -----------------------------------------------------------


async def test_provider_calls_are_counted_per_label_and_delegated() -> None:
    effects = FanoutEffects()
    inner_a = ScriptedLLM([LLMResponse(text="one"), LLMResponse(text="two")])
    a = effects.provider("A", inner_a)
    b = effects.provider("B", ScriptedLLM([LLMResponse(text="b")]))
    assert isinstance(a, CountingProvider)
    assert (await a.complete(["m1"])).text == "one"
    assert (await a.complete(["m2"], None, run_id="r")).text == "two"
    assert (await b.complete(["m"])).text == "b"
    counts = effects.counts()
    assert counts.provider == {"A": 2, "B": 1}
    assert counts.provider_calls("A") == 2
    assert inner_a.calls == 2
    assert inner_a.messages == [["m1"], ["m2"]]


def test_the_wrapper_reports_the_inner_providers_identity() -> None:
    inner = ScriptedLLM([])
    wrapped = FanoutEffects().provider("A", inner)
    assert wrapped.model == "scripted"
    assert wrapped.capabilities is inner.capabilities
    assert wrapped.tool_result_images == inner.tool_result_images
    assert wrapped.inner is inner
    assert wrapped.reports_cost is inner.reports_cost
    assert wrapped.thinking_enabled is inner.thinking_enabled


async def test_a_provider_call_that_fails_or_is_cancelled_still_counts() -> None:
    effects = FanoutEffects()
    failing = effects.provider("A", ScriptedLLM([]))
    with pytest.raises(IndexError):
        await failing.complete([])
    gated_inner = ScriptedLLM([LLMResponse(text="x")], gate_at=0)
    gated = effects.provider("B", gated_inner)
    task = asyncio.ensure_future(gated.complete([]))
    await gated_inner.entered.wait()
    assert effects.counts().provider == {"A": 1, "B": 1}
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gated_inner.cancelled
    assert effects.counts().provider == {"A": 1, "B": 1}


async def test_a_streamed_call_counts_once_when_it_starts_and_closes_the_inner_stream() -> None:
    effects = FanoutEffects()
    inner = GatedStreamLLM()
    wrapped = effects.provider("A", inner)
    stream = wrapped.complete_stream([])
    assert effects.counts().provider_calls("A") == 0  # not started until first advanced
    async with aclosing(stream) as events:
        first = await anext(events)
        assert first.type is StreamEventType.TEXT_DELTA
        assert effects.counts().provider_calls("A") == 1
    assert inner.closed
    assert effects.counts().provider_calls("A") == 1


async def test_a_streamed_call_runs_in_the_callers_task() -> None:
    seen: list[Any] = []

    class Probe(ScriptedLLM):
        async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
            seen.append(asyncio.current_task())
            return await super().complete(messages, tools, **kwargs)

    wrapped = FanoutEffects().provider("A", Probe([LLMResponse(text="x")]))
    events = [event async for event in wrapped.complete_stream([])]
    assert events[-1].type is StreamEventType.DONE
    assert seen == [asyncio.current_task()]


async def test_closing_the_wrapper_closes_the_inner_provider() -> None:
    closed: list[bool] = []

    class Closable(ScriptedLLM):
        async def close(self) -> None:
            closed.append(True)

    await FanoutEffects().provider("A", Closable([])).close()
    assert closed == [True]


# --- transform counters ------------------------------------------------------------


def test_the_counting_transform_counts_each_invocation_per_call() -> None:
    effects = FanoutEffects()
    transform = effects.transform()
    assert transform(_ctx("c1", result={"a": 1})) == {"a": 1}
    assert transform(_ctx("c1")) == "raw"
    assert transform(_ctx("c2", tool_name="classify")) == "raw"
    counts = effects.counts()
    assert counts.transforms == {("researcher", "c1"): 2, ("classify", "c2"): 1}
    assert counts.transform_calls() == 3
    assert counts.transform_calls("c1") == 2
    assert counts.transform_calls("c9") == 0


async def test_the_counting_transform_wraps_sync_and_async_transforms() -> None:
    effects = FanoutEffects()

    async def shout(ctx: ToolResultContext) -> str:
        return str(ctx.result).upper()

    assert effects.transform(lambda ctx: f"<{ctx.result}>")(_ctx("c1")) == "<raw>"
    assert await effects.transform(shout)(_ctx("c2")) == "RAW"
    assert effects.counts().transform_calls() == 2


def test_a_transform_that_raises_still_counts() -> None:
    effects = FanoutEffects()

    def bad(ctx: ToolResultContext) -> Any:
        raise ValueError("no")

    with pytest.raises(ValueError, match="no"):
        effects.transform(bad)(_ctx("c1"))
    assert effects.counts().transform_calls("c1") == 1


# --- snapshots and the "finished child is untouched" assertion ----------------------


def test_counts_are_an_immutable_snapshot() -> None:
    effects = FanoutEffects()
    effects.record_tool("A", "t")
    before = effects.counts()
    effects.record_tool("A", "t")
    assert before.tool("A", "t") == 1
    assert effects.counts().tool("A", "t") == 2
    with pytest.raises(TypeError):
        before.tools[("A", "t")] = 5  # type: ignore[index]


def test_for_child_keeps_only_that_childs_tool_and_provider_counts() -> None:
    effects = FanoutEffects()
    effects.record_tool("A", "t")
    effects.record_tool("B", "t")
    effects.record_provider("A")
    effects.record_provider("B")
    effects.transform()(_ctx("c1"))
    assert effects.counts().for_child("A") == EffectCounts(
        tools={("A", "t"): 1}, provider={"A": 1}, transforms={}
    )


def test_assert_unchanged_passes_when_a_finished_child_was_left_alone() -> None:
    effects = FanoutEffects()
    effects.record_tool("A", "t")
    effects.record_provider("A")
    before = effects.counts()
    effects.record_tool("B", "t")  # a sibling keeps working
    effects.record_provider("B")
    effects.assert_unchanged(before, "A")


@pytest.mark.parametrize("kind", ["tool", "new tool", "provider"])
def test_assert_unchanged_fails_when_a_finished_child_ran_again(kind: str) -> None:
    effects = FanoutEffects()
    effects.record_tool("A", "t")
    effects.record_provider("A")
    before = effects.counts()
    if kind == "tool":
        effects.record_tool("A", "t")
    elif kind == "new tool":
        effects.record_tool("A", "other")
    else:
        effects.record_provider("A")
    with pytest.raises(AssertionError, match="A"):
        effects.assert_unchanged(before, "A")


# --- external (file) counters survive a killed process -------------------------------


def test_memory_counters_do_not_share_state_between_recorders() -> None:
    first, second = FanoutEffects(), FanoutEffects()
    first.record_tool("A", "t")
    assert second.counts().tools == {}


def test_file_counters_are_shared_by_every_recorder_on_the_same_path(tmp_path: Path) -> None:
    path = tmp_path / "effects.log"
    first, second = FanoutEffects(path), FanoutEffects(path)
    first.record_tool("A", "save\nnote")
    second.record_provider("A")
    second.transform()(_ctx("c1"))
    assert first.counts() == second.counts()
    assert first.counts() == EffectCounts(
        tools={("A", "save\nnote"): 1}, provider={"A": 1}, transforms={("researcher", "c1"): 1}
    )


def test_a_missing_file_reads_as_no_effects(tmp_path: Path) -> None:
    assert FanoutEffects(tmp_path / "never-written.log").counts().tools == {}


def test_file_counters_survive_os_exit_in_another_process(tmp_path: Path) -> None:
    path = tmp_path / "effects.log"
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from tests._helpers.fanout_effects import FanoutEffects\n"
        "effects = FanoutEffects(Path(sys.argv[1]))\n"
        "effects.record_tool('A', 'save_note')\n"
        "effects.record_provider('A')\n"
        "os._exit(9)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(path)], env=env, capture_output=True, text=True
    )
    assert done.returncode == 9, done.stderr
    survivor = FanoutEffects(path)
    survivor.record_tool("B", "save_note")
    assert survivor.counts() == EffectCounts(
        tools={("A", "save_note"): 1, ("B", "save_note"): 1}, provider={"A": 1}, transforms={}
    )


def test_a_torn_last_line_is_reported_not_miscounted(tmp_path: Path) -> None:
    path = tmp_path / "effects.log"
    FanoutEffects(path).record_tool("A", "t")
    with path.open("a") as handle:
        handle.write('["tool", "A"')
    with pytest.raises(ValueError, match="effect log"):
        FanoutEffects(path).counts()


# --- no dependency on unbuilt 0.8 modules ---------------------------------------------


def test_the_helper_imports_no_fanout_runtime_module() -> None:
    import ast

    import tests._helpers.fanout_effects as helper

    imported = {
        node.module
        for node in ast.walk(ast.parse(Path(helper.__file__).read_text()))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert {m for m in imported if m.startswith("dendrux")} <= {"dendrux.llm.base", "dendrux.types"}
