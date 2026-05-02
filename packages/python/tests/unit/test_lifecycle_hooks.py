"""Tests for lifecycle hooks on LoopRecorder and LoopNotifier.

Covers PR A — adds run_started / run_finished / run_failed (runner) plus
llm_call_started / llm_call_failed and tool_started (loop). All hooks
gain a leading positional `run_id`. BaseNotifier and BaseRecorder
provide concrete no-op defaults so subclasses override only what they
care about.
"""

from __future__ import annotations

from typing import Any

import pytest

from dendrux import Agent, run, tool
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import (
    BaseNotifier,
    BaseRecorder,
    LoopNotifier,
    LoopRecorder,
)
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    LLMResponse,
    RunEventType,
    RunStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class CapturingNotifier(BaseNotifier):
    """Captures every hook call with positional + kwargs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
    ) -> None:
        self.calls.append(
            ("on_run_started", (run_id,), {"agent_name": agent_name, "agent_model": agent_model})
        )

    async def on_run_finished(self, run_id: str, result: Any) -> None:
        self.calls.append(("on_run_finished", (run_id, result), {}))

    async def on_run_failed(
        self, run_id: str, error: BaseException, *, iteration: int | None = None
    ) -> None:
        self.calls.append(("on_run_failed", (run_id, error), {"iteration": iteration}))

    async def on_message_appended(self, run_id: str, message: Any, iteration: int) -> None:
        self.calls.append(("on_message_appended", (run_id, message, iteration), {}))

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: Any = None,
        semantic_tools: Any = None,
    ) -> None:
        self.calls.append(("on_llm_call_started", (run_id, iteration), {}))

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: Any,
        iteration: int,
        *,
        semantic_messages: Any = None,
        semantic_tools: Any = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        self.calls.append(("on_llm_call_completed", (run_id, iteration), {}))

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        self.calls.append(("on_llm_call_failed", (run_id, iteration, error), {}))

    async def on_tool_started(self, run_id: str, tool_call: Any, iteration: int) -> None:
        self.calls.append(("on_tool_started", (run_id, tool_call.name, iteration), {}))

    async def on_tool_completed(
        self, run_id: str, tool_call: Any, tool_result: Any, iteration: int
    ) -> None:
        self.calls.append(("on_tool_completed", (run_id, tool_call.name, iteration), {}))

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        self.calls.append(("on_governance_event", (run_id, event_type, iteration), {}))

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def run_ids(self) -> set[str]:
        return {c[1][0] for c in self.calls if c[1]}


# ------------------------------------------------------------------
# Base classes
# ------------------------------------------------------------------


class TestBaseClasses:
    """BaseNotifier and BaseRecorder are concrete no-op defaults."""

    def test_base_notifier_satisfies_protocol(self) -> None:
        """BaseNotifier instances are structurally LoopNotifier."""
        n = BaseNotifier()
        assert isinstance(n, LoopNotifier)

    def test_base_recorder_satisfies_protocol(self) -> None:
        """BaseRecorder instances are structurally LoopRecorder."""
        r = BaseRecorder()
        assert isinstance(r, LoopRecorder)

    async def test_base_notifier_methods_are_noop(self) -> None:
        """All BaseNotifier methods return None and never raise."""
        n = BaseNotifier()
        await n.on_run_started("r1", agent_name="a", agent_model="m")
        await n.on_run_finished("r1", None)
        await n.on_run_failed("r1", RuntimeError("x"), iteration=0)
        await n.on_message_appended("r1", None, 0)
        await n.on_llm_call_started("r1", 0)
        await n.on_llm_call_completed("r1", None, 0)
        await n.on_llm_call_failed("r1", 0, RuntimeError("x"))
        await n.on_tool_started("r1", None, 0)
        await n.on_tool_completed("r1", None, None, 0)
        await n.on_governance_event("r1", "x", 0, {})

    async def test_base_recorder_methods_are_noop(self) -> None:
        """All BaseRecorder methods return None and never raise."""
        r = BaseRecorder()
        await r.on_run_started("r1", agent_name="a", agent_model="m")
        await r.on_run_finished("r1", None)
        await r.on_run_failed("r1", RuntimeError("x"), iteration=0)
        await r.on_message_appended("r1", None, 0)
        await r.on_llm_call_started("r1", 0)
        await r.on_llm_call_completed("r1", None, 0)
        await r.on_llm_call_failed("r1", 0, RuntimeError("x"))
        await r.on_tool_started("r1", None, 0)
        await r.on_tool_completed("r1", None, None, 0)
        await r.on_governance_event("r1", "x", 0, {})


# ------------------------------------------------------------------
# Runner — run_started / run_finished / run_failed
# ------------------------------------------------------------------


class TestRunnerLifecycleHooks:
    """Runner fires on_run_started before loop, on_run_finished after."""

    async def test_run_started_fires_once_at_top(self) -> None:
        n = CapturingNotifier()
        llm = MockLLM([LLMResponse(text="hi")])
        agent = Agent(name="alice", prompt="Test.", tools=[add])

        result = await run(agent, provider=llm, user_input="Hi", extra_notifier=n)

        assert result.status == RunStatus.SUCCESS
        starts = [c for c in n.calls if c[0] == "on_run_started"]
        assert len(starts) == 1
        # First positional arg is run_id (non-empty string)
        assert isinstance(starts[0][1][0], str) and starts[0][1][0]
        assert starts[0][2]["agent_name"] == "alice"
        # Provider model lands on agent_model — MockLLM exposes a model attr
        assert starts[0][2]["agent_model"] == llm.model

    async def test_run_finished_fires_with_result(self) -> None:
        n = CapturingNotifier()
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[add])

        result = await run(agent, provider=llm, user_input="Hi", extra_notifier=n)

        finishes = [c for c in n.calls if c[0] == "on_run_finished"]
        assert len(finishes) == 1
        assert finishes[0][1][1] is result

    async def test_run_started_precedes_run_finished(self) -> None:
        n = CapturingNotifier()
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="Hi", extra_notifier=n)

        names = n.names()
        assert names.index("on_run_started") < names.index("on_run_finished")

    async def test_run_failed_fires_on_unhandled_exception(self) -> None:
        from dendrux.llm.base import LLMProvider

        class BoomLLM(LLMProvider):
            @property
            def model(self) -> str:
                return "boom-model"

            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("boom")

            async def complete_stream(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("boom")

        n = CapturingNotifier()
        agent = Agent(prompt="Test.", tools=[add])

        with pytest.raises(RuntimeError, match="boom"):
            await run(agent, provider=BoomLLM(), user_input="Hi", extra_notifier=n)

        failures = [c for c in n.calls if c[0] == "on_run_failed"]
        assert len(failures) == 1
        assert isinstance(failures[0][1][1], RuntimeError)
        # And on_run_finished must NOT fire for an errored run
        assert not any(c[0] == "on_run_finished" for c in n.calls)

    async def test_run_id_is_consistent_across_all_hooks(self) -> None:
        n = CapturingNotifier()
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="Hi", extra_notifier=n)

        assert len(n.run_ids()) == 1


# ------------------------------------------------------------------
# ReActLoop — llm_call_started / llm_call_failed / tool_started
# ------------------------------------------------------------------


class TestReActLifecycleHooks:
    """ReActLoop fires lifecycle hooks around LLM calls and tool calls."""

    async def test_llm_started_precedes_llm_completed(self) -> None:
        n = CapturingNotifier()
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[add])

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hi",
            run_id="run-react-1",
            notifier=n,
        )

        names = n.names()
        assert "on_llm_call_started" in names
        assert "on_llm_call_completed" in names
        assert names.index("on_llm_call_started") < names.index("on_llm_call_completed")

    async def test_tool_started_precedes_tool_completed(self) -> None:
        n = CapturingNotifier()
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="3"),
            ]
        )
        agent = Agent(prompt="Test.", tools=[add])

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
            run_id="run-react-2",
            notifier=n,
        )

        names = n.names()
        assert "on_tool_started" in names
        assert "on_tool_completed" in names
        assert names.index("on_tool_started") < names.index("on_tool_completed")

    async def test_llm_failed_fires_when_provider_raises(self) -> None:
        from dendrux.llm.base import LLMProvider

        class BoomLLM(LLMProvider):
            @property
            def model(self) -> str:
                return "boom-model"

            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("provider crashed")

            async def complete_stream(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("provider crashed")

        n = CapturingNotifier()
        agent = Agent(prompt="Test.", tools=[add])

        with pytest.raises(RuntimeError, match="provider crashed"):
            await ReActLoop().run(
                agent=agent,
                provider=BoomLLM(),
                strategy=NativeToolCalling(),
                user_input="Hi",
                run_id="run-react-3",
                notifier=n,
            )

        failures = [c for c in n.calls if c[0] == "on_llm_call_failed"]
        assert len(failures) == 1
        assert failures[0][1][0] == "run-react-3"
        assert isinstance(failures[0][1][2], RuntimeError)
        # llm_started fired, but completed did NOT
        assert "on_llm_call_started" in n.names()
        assert "on_llm_call_completed" not in n.names()

    async def test_run_id_propagates_to_loop_hooks(self) -> None:
        n = CapturingNotifier()
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="3"),
            ]
        )
        agent = Agent(prompt="Test.", tools=[add])

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hi",
            run_id="run-react-4",
            notifier=n,
        )

        # Every captured call's first positional arg should be the run_id
        for name, args, _ in n.calls:
            assert args[0] == "run-react-4", f"{name} got run_id={args[0]!r}"


# ------------------------------------------------------------------
# SingleCall — llm_call_started / llm_call_failed
# ------------------------------------------------------------------


class TestSingleCallLifecycleHooks:
    """SingleCall fires LLM lifecycle hooks too."""

    async def test_llm_started_precedes_llm_completed(self) -> None:
        n = CapturingNotifier()
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", loop=SingleCall())  # no tools — SingleCall happy path

        await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hi",
            run_id="run-single-1",
            notifier=n,
        )

        names = n.names()
        assert names.index("on_llm_call_started") < names.index("on_llm_call_completed")

    async def test_llm_failed_fires_when_provider_raises(self) -> None:
        from dendrux.llm.base import LLMProvider

        class BoomLLM(LLMProvider):
            @property
            def model(self) -> str:
                return "boom-model"

            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("crashed")

            async def complete_stream(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("crashed")

        n = CapturingNotifier()
        agent = Agent(prompt="Test.", loop=SingleCall())

        with pytest.raises(RuntimeError, match="crashed"):
            await SingleCall().run(
                agent=agent,
                provider=BoomLLM(),
                strategy=NativeToolCalling(),
                user_input="Hi",
                run_id="run-single-2",
                notifier=n,
            )

        failures = [c for c in n.calls if c[0] == "on_llm_call_failed"]
        assert len(failures) == 1
        assert failures[0][1][0] == "run-single-2"


# ------------------------------------------------------------------
# Streaming LLM lifecycle — protocol-violation paths
# ------------------------------------------------------------------


class _ProviderStream:
    """Minimal complete_stream return — async-iterable + aclose()."""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events
        self.closed = False

    def __aiter__(self) -> Any:
        return self._gen()

    async def _gen(self) -> Any:
        for e in self._events:
            yield e

    async def aclose(self) -> None:
        self.closed = True


def _streaming_provider(events: list[StreamEvent]) -> Any:
    """Build an LLMProvider whose complete_stream yields ``events``."""
    from dendrux.llm.base import LLMProvider

    class _P(LLMProvider):
        @property
        def model(self) -> str:
            return "fake-stream"

        async def complete(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        def complete_stream(self, *args: Any, **kwargs: Any) -> Any:
            return _ProviderStream(list(events))

    return _P()


class TestStreamingLLMLifecycle:
    """Streaming provider that violates contract still pairs started/failed."""

    async def test_react_stream_no_done_event_fires_llm_failed(self) -> None:
        """Provider stream ends without DONE → llm_failed must fire so notifier
        spans opened by on_llm_call_started can be closed."""
        n = CapturingNotifier()
        provider = _streaming_provider([StreamEvent(type=StreamEventType.TEXT_DELTA, text="hi")])
        agent = Agent(prompt="Test.", tools=[add])

        events: list[Any] = []
        with pytest.raises(RuntimeError, match="ended without DONE"):
            async for ev in ReActLoop().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="Hi",
                run_id="run-stream-react-1",
                notifier=n,
            ):
                events.append(ev)

        names = n.names()
        assert "on_llm_call_started" in names
        assert "on_llm_call_failed" in names, (
            "Stream ended without DONE — llm_failed must fire to close any span "
            "opened on llm_started. Currently the validation RuntimeError is raised "
            "outside the try/except that emits llm_failed."
        )
        assert "on_llm_call_completed" not in names

    async def test_single_call_stream_no_done_event_fires_llm_failed(self) -> None:
        n = CapturingNotifier()
        provider = _streaming_provider([StreamEvent(type=StreamEventType.TEXT_DELTA, text="hi")])
        agent = Agent(prompt="Test.", loop=SingleCall())

        events: list[Any] = []
        with pytest.raises(RuntimeError, match="ended without DONE"):
            async for ev in SingleCall().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="Hi",
                run_id="run-stream-single-1",
                notifier=n,
            ):
                events.append(ev)

        names = n.names()
        assert "on_llm_call_started" in names
        assert "on_llm_call_failed" in names
        assert "on_llm_call_completed" not in names

    async def test_single_call_stream_unexpected_tool_calls_fires_llm_failed(self) -> None:
        """SingleCall validates llm_response.tool_calls AFTER stream close.
        That validation RuntimeError must also pair with llm_failed."""
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        done_response = LLMResponse(text="x", tool_calls=[tc])
        provider = _streaming_provider([StreamEvent(type=StreamEventType.DONE, raw=done_response)])

        n = CapturingNotifier()
        agent = Agent(prompt="Test.", loop=SingleCall())

        events: list[Any] = []
        with pytest.raises(RuntimeError, match="unexpected tool_calls"):
            async for ev in SingleCall().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="Hi",
                run_id="run-stream-single-2",
                notifier=n,
            ):
                events.append(ev)

        names = n.names()
        assert "on_llm_call_started" in names
        assert "on_llm_call_failed" in names
        assert "on_llm_call_completed" not in names


# ------------------------------------------------------------------
# Non-persisted run_stream — lifecycle hooks must still fire
# ------------------------------------------------------------------


class TestNonPersistedStreamLifecycle:
    """When users stream without a state store, on_run_finished must still
    fire so notifier-side spans (OTel root span) close cleanly."""

    async def test_run_stream_without_persistence_fires_run_finished(self) -> None:
        from dendrux.runtime.runner import run_stream as runner_run_stream

        n = CapturingNotifier()
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        events: list[Any] = []
        async for ev in runner_run_stream(
            agent,
            provider=llm,
            user_input="Hi",
            extra_notifier=n,
        ):
            events.append(ev)

        terminal_types = {
            e.type
            for e in events
            if e.type
            in (
                RunEventType.RUN_COMPLETED,
                RunEventType.RUN_PAUSED,
                RunEventType.RUN_CANCELLED,
            )
        }
        assert RunEventType.RUN_COMPLETED in terminal_types

        names = n.names()
        assert "on_run_started" in names
        assert "on_run_finished" in names, (
            "run_stream() without persistence must still emit on_run_finished — "
            "OTel notifiers need it to close the root span. Currently the hook "
            "is gated by `store is not None`."
        )
