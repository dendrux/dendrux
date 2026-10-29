"""0.5.0 contracts: Agent and read primitives, and the inert entry-check seams.

Every public entry point calls ``runtime._entry_checks.entry_checks`` first,
which dispatches to ``_tenant_checks`` and ``_limit_checks``; the runner calls
``run_meta_checks`` where it merges developer metadata. These tests pin that
each seam fires exactly once per entry point, before any I/O, and that the
omitted-tenant default resolves to the 0.4.1 behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dendrux import Agent
from dendrux._sentinel import _OMITTED
from dendrux.llm.mock import MockLLM
from dendrux.runtime import _limit_checks, _tenant_checks
from dendrux.runtime.state import UNSCOPED
from dendrux.types import Limits, LLMResponse, RunStatus

if TYPE_CHECKING:
    from pathlib import Path


class _StopError(Exception):
    """Raised by the patched limit check so an entry point stops before any I/O."""


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    record: dict[str, list[Any]] = {"resolve": [], "tenant": [], "limit": []}
    resolve = _tenant_checks.resolve_tenant

    def _resolve(agent: Any, entry: Any, tenant_id: Any, *, default: Any) -> Any:
        record["resolve"].append((entry, tenant_id, default))
        return resolve(agent, entry, tenant_id, default=default)

    def _tenant(agent: Any, entry: Any, *, tenant_id: Any, **fields: Any) -> None:
        record["tenant"].append((entry, tenant_id, fields))

    def _limit(agent: Any, entry: Any, *, tenant_id: Any, **fields: Any) -> None:
        record["limit"].append((entry, tenant_id, fields))
        raise _StopError

    monkeypatch.setattr(_tenant_checks, "resolve_tenant", _resolve)
    monkeypatch.setattr(_tenant_checks, "check_entry", _tenant)
    monkeypatch.setattr(_limit_checks, "check_entry", _limit)
    return record


def _agent() -> Agent:
    # No persistence: an entry point that got past the seam would fail on I/O.
    return Agent(provider=MockLLM([]), prompt="p")


_ASYNC_ENTRIES: list[tuple[str, Any, Any]] = [
    ("run", lambda a: a.run("hi", metadata={"k": 1}), None),
    ("retry", lambda a: a.retry("00000000000000000000000304", metadata={"k": 1}), None),
    ("resume", lambda a: a.resume("00000000000000000000000304"), UNSCOPED),
    (
        "submit_tool_results",
        lambda a: a.submit_tool_results("00000000000000000000000304", []),
        UNSCOPED,
    ),
    ("submit_input", lambda a: a.submit_input("00000000000000000000000304", "answer"), UNSCOPED),
    (
        "submit_approval",
        lambda a: a.submit_approval("00000000000000000000000304", approved=True),
        UNSCOPED,
    ),
    ("submit_resume", lambda a: a.submit_resume("00000000000000000000000304"), UNSCOPED),
    ("cancel_run", lambda a: a.cancel_run("00000000000000000000000304"), UNSCOPED),
    ("request_pause", lambda a: a.request_pause("00000000000000000000000304"), UNSCOPED),
    (
        "submit_note",
        lambda a: a.submit_note("00000000000000000000000304", "note", metadata={"m": 1}),
        UNSCOPED,
    ),
]
_SYNC_ENTRIES: list[tuple[str, Any, Any]] = [
    ("stream", lambda a: a.stream("hi", metadata={"k": 1}), None),
    ("resume_stream", lambda a: a.resume_stream("00000000000000000000000304"), UNSCOPED),
]


def _assert_called_once(calls: dict[str, list[Any]], entry: str, default: Any) -> None:
    assert [c[0] for c in calls["resolve"]] == [entry]
    assert calls["resolve"][0][1] is _OMITTED
    assert calls["resolve"][0][2] is default
    assert [c[0] for c in calls["tenant"]] == [entry]
    assert [c[0] for c in calls["limit"]] == [entry]
    assert calls["tenant"][0][1] is default
    assert calls["limit"][0][1] is default


class TestEntryChecksFireOncePerEntryPoint:
    @pytest.mark.parametrize(("entry", "call", "default"), _ASYNC_ENTRIES)
    async def test_async_entry(
        self, calls: dict[str, list[Any]], entry: str, call: Any, default: Any
    ) -> None:
        with pytest.raises(_StopError):
            await call(_agent())
        _assert_called_once(calls, entry, default)

    @pytest.mark.parametrize(("entry", "call", "default"), _SYNC_ENTRIES)
    def test_sync_entry_checks_at_call_time(
        self, calls: dict[str, list[Any]], entry: str, call: Any, default: Any
    ) -> None:
        with pytest.raises(_StopError):
            call(_agent())
        _assert_called_once(calls, entry, default)

    async def test_fields_passed_to_both_modules(self, calls: dict[str, list[Any]]) -> None:
        with pytest.raises(_StopError):
            await _agent().submit_note(
                "00000000000000000000000304", "note", metadata={"m": 1}, tenant_id="t"
            )
        fields = {
            "run_id": "00000000000000000000000304",
            "note_text": "note",
            "note_metadata": {"m": 1},
        }
        assert calls["tenant"] == [("submit_note", "t", fields)]
        assert calls["limit"] == [("submit_note", "t", fields)]

    async def test_run_fields(self, calls: dict[str, list[Any]]) -> None:
        with pytest.raises(_StopError):
            await _agent().run("hi", metadata={"k": 1}, model="m-2", temperature=0)
        _, _, fields = calls["tenant"][0]
        assert fields == {
            "metadata": {"k": 1},
            "user_input": "hi",
            "history": None,
            "context": None,
            "provider_kwargs": {"model": "m-2", "temperature": 0},
            "idempotency_key": None,
        }

    async def test_submit_approval_fields(self, calls: dict[str, list[Any]]) -> None:
        with pytest.raises(_StopError):
            await _agent().submit_approval(
                "00000000000000000000000304", approved=False, rejection_reason="no"
            )
        _, _, fields = calls["tenant"][0]
        assert fields == {
            "run_id": "00000000000000000000000304",
            "rejection_reason": "no",
            "decisions": None,
        }


class TestTenantResolution:
    @pytest.mark.parametrize("explicit", ["t1", None, UNSCOPED])
    async def test_explicit_tenant_passes_through(
        self, calls: dict[str, list[Any]], explicit: Any
    ) -> None:
        with pytest.raises(_StopError):
            await _agent().cancel_run("00000000000000000000000304", tenant_id=explicit)
        assert calls["tenant"][0][1] is explicit

    def test_resolve_tenant_defaults(self) -> None:
        agent = _agent()
        assert _tenant_checks.resolve_tenant(agent, "run", _OMITTED, default=None) is None
        assert (
            _tenant_checks.resolve_tenant(agent, "submit_input", _OMITTED, default=UNSCOPED)
            is UNSCOPED
        )
        assert _tenant_checks.resolve_tenant(agent, "run", "t", default=None) == "t"

    def test_inert_checks_return_none(self) -> None:
        agent = _agent()
        assert _tenant_checks.check_entry(agent, "run", tenant_id=None, metadata={}) is None
        assert _limit_checks.check_entry(agent, "run", tenant_id=None, user_input="x") is None
        for module in (_tenant_checks, _limit_checks):
            assert module.check_runner_entry(agent, "run", tenant_id=None, metadata=None) is None


@pytest.fixture
def runner_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, Any, Any, dict[str, Any]]]:
    record: list[tuple[str, str, Any, Any, dict[str, Any]]] = []

    def _make(module: str) -> Any:
        def _check(agent: Any, entry: Any, *, tenant_id: Any, metadata: Any, **fields: Any) -> None:
            record.append((module, entry, tenant_id, metadata, fields))

        return _check

    monkeypatch.setattr(_tenant_checks, "check_runner_entry", _make("tenant"))
    monkeypatch.setattr(_limit_checks, "check_runner_entry", _make("limit"))
    return record


def _modules(record: list[tuple[str, str, Any, Any, dict[str, Any]]]) -> list[tuple[Any, ...]]:
    return [r[:4] for r in record]


class TestRunnerEntrySeam:
    async def test_run_stream_retry_each_fire_once(
        self, tmp_path: Path, runner_calls: list[tuple[str, str, Any, Any, dict[str, Any]]]
    ) -> None:
        agent = Agent(
            provider=MockLLM([LLMResponse(text="a"), LLMResponse(text="b"), LLMResponse(text="c")]),
            prompt="p",
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'seams.db'}",
        )
        try:
            first = await agent.run("hi", tenant_id="t", metadata={"k": 1})
            assert first.status == RunStatus.SUCCESS
            assert _modules(runner_calls) == [
                ("tenant", "run", "t", {"k": 1}),
                ("limit", "run", "t", {"k": 1}),
            ]
            assert runner_calls[0][4] == {
                "user_input": "hi",
                "history": None,
                "context": None,
                "provider_kwargs": {},
                "idempotency_key": None,
                "provider": agent.provider,
            }
            runner_calls.clear()

            async with agent.stream("again", tenant_id="t", metadata={"s": 2}) as stream:
                async for _ in stream:
                    pass
            assert _modules(runner_calls) == [
                ("tenant", "stream", "t", {"s": 2}),
                ("limit", "stream", "t", {"s": 2}),
            ]
            assert runner_calls[0][4] == {
                "user_input": "again",
                "history": None,
                "context": None,
                "provider_kwargs": {},
                "idempotency_key": None,
                "provider": agent.provider,
            }
            runner_calls.clear()

            retried = await agent.retry(first.run_id, tenant_id="t", metadata={"r": 3})
            assert retried.status == RunStatus.SUCCESS
            assert _modules(runner_calls) == [
                ("tenant", "retry", "t", {"r": 3}),
                ("limit", "retry", "t", {"r": 3}),
            ]
        finally:
            await agent.close()

    async def test_fires_without_a_store(
        self, runner_calls: list[tuple[str, str, Any, Any, dict[str, Any]]]
    ) -> None:
        agent = Agent(provider=MockLLM([LLMResponse(text="a"), LLMResponse(text="b")]), prompt="p")
        assert (await agent.run("hi", metadata={"k": 1})).status == RunStatus.SUCCESS
        async with agent.stream("hi", metadata={"k": 2}) as stream:
            async for _ in stream:
                pass
        assert _modules(runner_calls) == [
            ("tenant", "run", None, {"k": 1}),
            ("limit", "run", None, {"k": 1}),
            ("tenant", "stream", None, {"k": 2}),
            ("limit", "stream", None, {"k": 2}),
        ]

    async def test_delegated_child_sees_the_inherited_tenant(
        self, runner_calls: list[tuple[str, str, Any, Any, dict[str, Any]]]
    ) -> None:
        from dendrux.runtime.context import (
            DelegationContext,
            reset_delegation_context,
            set_delegation_context,
        )

        child = Agent(provider=MockLLM([LLMResponse(text="a")]), prompt="p")
        token = set_delegation_context(DelegationContext(run_id="parent", tenant_id="t-parent"))
        try:
            assert (await child.run("hi")).status == RunStatus.SUCCESS
        finally:
            reset_delegation_context(token)
        assert _modules(runner_calls)[0] == ("tenant", "run", "t-parent", None)

    @pytest.mark.parametrize("entry", ["run", "stream"])
    async def test_direct_runner_callers_are_checked_before_any_io(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: str
    ) -> None:
        from dendrux import run as runner_run
        from dendrux.runtime.runner import run_stream

        def _stop(*args: Any, **kwargs: Any) -> None:
            raise _StopError

        monkeypatch.setattr(_tenant_checks, "check_runner_entry", _stop)
        agent = Agent(provider=MockLLM([]), prompt="p")
        store = _SpyStore()
        with pytest.raises(_StopError):
            if entry == "run":
                await runner_run(agent, provider=agent.provider, user_input="hi", state_store=store)  # type: ignore[arg-type]
            else:
                stream = run_stream(
                    agent, provider=agent.provider, user_input="hi", state_store=store
                )  # type: ignore[arg-type]
                async for _ in stream:
                    pass
        assert store.calls == []


class _SpyStore:
    """Records every attribute access as a store call; nothing may be touched."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.calls.append(name)
        raise AttributeError(name)


class TestBudgetConfigSeam:
    def test_called_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dendrux import pricing

        seen: list[Any] = []
        monkeypatch.setattr(
            pricing, "check_budget_config", lambda agent, *, model=None: seen.append(model)
        )
        _agent()
        assert seen == [None]

    @pytest.mark.parametrize(
        ("entry", "call"),
        [
            ("run", lambda a: a.run("hi", model="m-2")),
            ("retry", lambda a: a.retry("00000000000000000000000304", model="m-2")),
        ],
    )
    async def test_called_at_new_run_entries_with_requested_model(
        self, calls: dict[str, list[Any]], monkeypatch: pytest.MonkeyPatch, entry: str, call: Any
    ) -> None:
        from dendrux import pricing

        agent = _agent()
        seen: list[Any] = []
        monkeypatch.setattr(
            pricing, "check_budget_config", lambda agent, *, model=None: seen.append(model)
        )
        # The patched limit check stops the call first, so lift it for this test.
        monkeypatch.setattr(_limit_checks, "check_entry", lambda *a, **k: None)
        with pytest.raises(Exception):  # noqa: B017 - fails later, on missing persistence/run
            await call(agent)
        assert seen == (["m-2", "m-2"] if entry == "run" else ["m-2"])

    def test_stream_entry_uses_provider_model_when_not_overridden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dendrux import pricing

        agent = _agent()
        seen: list[Any] = []
        monkeypatch.setattr(
            pricing, "check_budget_config", lambda agent, *, model=None: seen.append(model)
        )
        agent.stream("hi")
        assert seen == [None]

    @pytest.mark.parametrize("entry", ["resume", "cancel_run"])
    async def test_not_called_for_continuations(
        self, monkeypatch: pytest.MonkeyPatch, entry: str
    ) -> None:
        from dendrux import pricing

        agent = _agent()
        seen: list[Any] = []
        monkeypatch.setattr(
            pricing, "check_budget_config", lambda agent, *, model=None: seen.append(model)
        )
        with pytest.raises(Exception):  # noqa: B017 - no persistence configured
            await getattr(agent, entry)("00000000000000000000000304")
        assert seen == []


class TestNoteCapPlumbing:
    async def test_submit_note_passes_max_notes(self) -> None:
        agent = Agent(provider=MockLLM([]), prompt="p", limits=Limits(max_notes_per_run=7))
        seen: dict[str, Any] = {}

        class _Store:
            async def add_note(self, run_id: str, **kwargs: Any) -> bool:
                seen.update(kwargs)
                return True

        agent._state_store = _Store()  # type: ignore[assignment]
        await agent.submit_note("00000000000000000000000304", "note")
        assert seen["max_notes"] == 7

    async def test_store_without_max_notes_keeps_working(self) -> None:
        agent = _agent()
        seen: dict[str, Any] = {}

        class _LegacyStore:
            async def add_note(
                self, run_id: str, *, note_id: str, text: str, metadata: Any, tenant_id: Any
            ) -> bool:
                seen["called"] = True
                return True

        agent._state_store = _LegacyStore()  # type: ignore[assignment]
        await agent.submit_note("00000000000000000000000304", "note")
        assert seen == {"called": True}

    async def test_default_cap_is_1000(self) -> None:
        agent = _agent()
        seen: dict[str, Any] = {}

        class _Store:
            async def add_note(self, run_id: str, **kwargs: Any) -> bool:
                seen.update(kwargs)
                return True

        agent._state_store = _Store()  # type: ignore[assignment]
        await agent.submit_note("00000000000000000000000304", "note")
        assert seen["max_notes"] == 1000


class TestAgentKwargs:
    def test_defaults(self) -> None:
        agent = _agent()
        assert agent._strict_tenancy is False
        assert agent._strict_tenancy_chosen is False
        assert agent._limits == Limits()
        assert agent._tool_threads == 8
        assert agent._on_governance_change == "warn"
        assert agent._on_governance_change_chosen is False
        from dendrux.tools._pool import ToolCallLimiter

        assert isinstance(agent._tool_call_limiter, ToolCallLimiter)
        assert agent._tool_call_limiter.slots == 8

    def test_explicit_defaults_count_as_chosen(self) -> None:
        agent = Agent(
            provider=MockLLM([]), prompt="p", strict_tenancy=False, on_governance_change="warn"
        )
        assert (agent._strict_tenancy, agent._strict_tenancy_chosen) == (False, True)
        assert (agent._on_governance_change, agent._on_governance_change_chosen) == ("warn", True)

    def test_stored(self) -> None:
        limits = Limits(max_input_bytes=10)
        agent = Agent(
            provider=MockLLM([]),
            prompt="p",
            strict_tenancy=True,
            limits=limits,
            tool_threads=2,
            on_governance_change="raise",
        )
        assert agent._strict_tenancy is True
        assert agent._limits is limits
        assert agent._tool_threads == 2
        assert agent._on_governance_change == "raise"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"strict_tenancy": 1}, "strict_tenancy"),
            ({"strict_tenancy": None}, "strict_tenancy"),
            ({"limits": {"max_input_bytes": 1}}, "limits"),
            ({"tool_threads": 0}, "tool_threads"),
            ({"tool_threads": True}, "tool_threads"),
            ({"tool_threads": 2.0}, "tool_threads"),
            ({"on_governance_change": "ignore"}, "on_governance_change"),
        ],
    )
    def test_validation(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Agent(provider=MockLLM([]), prompt="p", **kwargs)

    async def test_call_limiter_threaded_into_tool_lookups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dendrux.tools import _lookups

        sentinel = object()
        seen: list[int] = []

        def _make(tool_threads: int) -> object:
            seen.append(tool_threads)
            return sentinel

        monkeypatch.setattr(_lookups, "make_call_limiter", _make)
        agent = Agent(provider=MockLLM([]), prompt="p", tool_threads=3)
        assert seen == [3]
        lookups = await agent.get_tool_lookups()
        assert lookups.call_limiter is sentinel
