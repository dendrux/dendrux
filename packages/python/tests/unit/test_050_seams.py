"""0.5.0 contracts: Agent and read-router kwargs, and the inert entry-check seams.

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
from dendrux.http import make_read_router
from dendrux.http.read_router import _tenant_dependency
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
    ("retry", lambda a: a.retry("r1", metadata={"k": 1}), None),
    ("resume", lambda a: a.resume("r1"), UNSCOPED),
    ("submit_tool_results", lambda a: a.submit_tool_results("r1", []), UNSCOPED),
    ("submit_input", lambda a: a.submit_input("r1", "answer"), UNSCOPED),
    ("submit_approval", lambda a: a.submit_approval("r1", approved=True), UNSCOPED),
    ("submit_resume", lambda a: a.submit_resume("r1"), UNSCOPED),
    ("cancel_run", lambda a: a.cancel_run("r1"), UNSCOPED),
    ("request_pause", lambda a: a.request_pause("r1"), UNSCOPED),
    ("submit_note", lambda a: a.submit_note("r1", "note", metadata={"m": 1}), UNSCOPED),
]
_SYNC_ENTRIES: list[tuple[str, Any, Any]] = [
    ("stream", lambda a: a.stream("hi", metadata={"k": 1}), None),
    ("resume_stream", lambda a: a.resume_stream("r1"), UNSCOPED),
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
            await _agent().submit_note("r1", "note", metadata={"m": 1}, tenant_id="t")
        fields = {"run_id": "r1", "note_text": "note", "note_metadata": {"m": 1}}
        assert calls["tenant"] == [("submit_note", "t", fields)]
        assert calls["limit"] == [("submit_note", "t", fields)]

    async def test_run_fields(self, calls: dict[str, list[Any]]) -> None:
        with pytest.raises(_StopError):
            await _agent().run("hi", metadata={"k": 1})
        _, _, fields = calls["tenant"][0]
        assert fields == {
            "metadata": {"k": 1},
            "user_input": "hi",
            "history": None,
            "context": None,
        }


class TestTenantResolution:
    @pytest.mark.parametrize("explicit", ["t1", None, UNSCOPED])
    async def test_explicit_tenant_passes_through(
        self, calls: dict[str, list[Any]], explicit: Any
    ) -> None:
        with pytest.raises(_StopError):
            await _agent().cancel_run("r1", tenant_id=explicit)
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
        assert (
            _tenant_checks.check_run_metadata(agent, "run", tenant_id=None, metadata=None) is None
        )
        assert _limit_checks.check_run_metadata(agent, "run", tenant_id=None, metadata=None) is None


@pytest.fixture
def meta_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, Any, Any]]:
    record: list[tuple[str, str, Any, Any]] = []

    def _make(module: str) -> Any:
        def _check(agent: Any, entry: Any, *, tenant_id: Any, metadata: Any) -> None:
            record.append((module, entry, tenant_id, metadata))

        return _check

    monkeypatch.setattr(_tenant_checks, "check_run_metadata", _make("tenant"))
    monkeypatch.setattr(_limit_checks, "check_run_metadata", _make("limit"))
    return record


class TestRunMetaMergeSeam:
    async def test_run_stream_retry_each_fire_once(
        self, tmp_path: Path, meta_calls: list[tuple[str, str, Any, Any]]
    ) -> None:
        agent = Agent(
            provider=MockLLM([LLMResponse(text="a"), LLMResponse(text="b"), LLMResponse(text="c")]),
            prompt="p",
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'seams.db'}",
        )
        try:
            first = await agent.run("hi", tenant_id="t", metadata={"k": 1})
            assert first.status == RunStatus.SUCCESS
            assert meta_calls == [
                ("tenant", "run", "t", {"k": 1}),
                ("limit", "run", "t", {"k": 1}),
            ]
            meta_calls.clear()

            async with agent.stream("again", tenant_id="t", metadata={"s": 2}) as stream:
                async for _ in stream:
                    pass
            assert meta_calls == [
                ("tenant", "stream", "t", {"s": 2}),
                ("limit", "stream", "t", {"s": 2}),
            ]
            meta_calls.clear()

            retried = await agent.retry(first.run_id, tenant_id="t", metadata={"r": 3})
            assert retried.status == RunStatus.SUCCESS
            assert meta_calls == [
                ("tenant", "retry", "t", {"r": 3}),
                ("limit", "retry", "t", {"r": 3}),
            ]
        finally:
            await agent.close()

    async def test_no_store_no_merge(self, meta_calls: list[tuple[str, str, Any, Any]]) -> None:
        agent = Agent(provider=MockLLM([LLMResponse(text="a")]), prompt="p")
        assert (await agent.run("hi", metadata={"k": 1})).status == RunStatus.SUCCESS
        assert meta_calls == []


class TestAgentKwargs:
    def test_defaults(self) -> None:
        agent = _agent()
        assert agent._strict_tenancy is False
        assert agent._limits is None
        assert agent._tool_threads == 8
        assert agent._on_governance_change == "warn"
        assert agent._tool_call_limiter is None

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


class _FakeStore:
    pass


class TestReadRouterKwargs:
    def test_tenant_resolver_is_required(self) -> None:
        with pytest.raises(TypeError, match="tenant_resolver"):
            make_read_router(store=_FakeStore(), authorize=lambda: None)  # type: ignore[call-arg, arg-type]

    def test_accepts_unscoped_and_callables(self) -> None:
        make_read_router(store=_FakeStore(), authorize=lambda: None, tenant_resolver=UNSCOPED)  # type: ignore[arg-type]
        make_read_router(
            store=_FakeStore(),  # type: ignore[arg-type]
            authorize=lambda: None,
            tenant_resolver=lambda: "t",
            include_payloads=True,
            sse_max_connections=10,
            sse_max_duration_s=60,
        )

    def test_unscoped_sentinel_becomes_a_constant_dependency(self) -> None:
        dep = _tenant_dependency(UNSCOPED)
        assert callable(dep)
        assert dep() is UNSCOPED

        def resolver() -> str:
            return "t"

        assert _tenant_dependency(resolver) is resolver

    @pytest.mark.parametrize(
        ("kwargs", "exc"),
        [
            ({"tenant_resolver": "tenant-a"}, TypeError),
            ({"tenant_resolver": None}, TypeError),
            ({"tenant_resolver": UNSCOPED, "include_payloads": 1}, TypeError),
            ({"tenant_resolver": UNSCOPED, "sse_max_connections": 0}, ValueError),
            ({"tenant_resolver": UNSCOPED, "sse_max_connections": True}, ValueError),
            ({"tenant_resolver": UNSCOPED, "sse_max_duration_s": 0}, ValueError),
            ({"tenant_resolver": UNSCOPED, "sse_max_duration_s": float("inf")}, ValueError),
        ],
    )
    def test_validation(self, kwargs: dict[str, Any], exc: type[Exception]) -> None:
        with pytest.raises(exc):
            make_read_router(store=_FakeStore(), authorize=lambda: None, **kwargs)  # type: ignore[arg-type]
