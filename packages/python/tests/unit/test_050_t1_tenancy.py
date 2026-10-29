"""0.5.0: entry-point tenancy and run-metadata checks (``runtime/_tenant_checks.py``).

Strict tenancy, ``tenant_id=""`` rejection, reserved ``dendrux.``
metadata keys (B8) and the metadata size cap, on every entry point,
before any I/O. Delegated children inherit the parent's tenant and never
raise under strict tenancy.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

import dendrux
from dendrux import Agent, tool
from dendrux.db.models import Base
from dendrux.errors import TenantScopeRequiredError
from dendrux.llm.mock import MockLLM
from dendrux.runtime import _tenant_checks
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.state import UNSCOPED, SQLAlchemyStateStore
from dendrux.types import Limits, LLMResponse, ToolCall


class _SpyStore(SQLAlchemyStateStore):
    """Records every store call so a test can assert "nothing was read or written"."""

    def __init__(self, engine: Any) -> None:
        super().__init__(engine)
        self.calls: list[str] = []

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if not name.startswith("_") and callable(attr) and name != "calls":
            object.__getattribute__(self, "calls").append(name)
        return attr


@pytest.fixture
async def store(tmp_path: Any) -> Any:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 't1.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    spy = _SpyStore(engine)
    yield spy
    await engine.dispose()


def _agent(store: Any = None, **kw: Any) -> Agent:
    return Agent(
        provider=MockLLM([LLMResponse(text="ok")] * 5), prompt="p", state_store=store, **kw
    )


_ASYNC_ENTRIES: list[tuple[str, Any]] = [
    ("run", lambda a, **k: a.run("hi", **k)),
    ("retry", lambda a, **k: a.retry("00000000000000000000000105", **k)),
    ("resume", lambda a, **k: a.resume("00000000000000000000000105", **k)),
    (
        "submit_tool_results",
        lambda a, **k: a.submit_tool_results("00000000000000000000000105", [], **k),
    ),
    ("submit_input", lambda a, **k: a.submit_input("00000000000000000000000105", "answer", **k)),
    (
        "submit_approval",
        lambda a, **k: a.submit_approval("00000000000000000000000105", approved=True, **k),
    ),
    ("submit_resume", lambda a, **k: a.submit_resume("00000000000000000000000105", **k)),
    ("cancel_run", lambda a, **k: a.cancel_run("00000000000000000000000105", **k)),
    ("request_pause", lambda a, **k: a.request_pause("00000000000000000000000105", **k)),
    ("submit_note", lambda a, **k: a.submit_note("00000000000000000000000105", "note", **k)),
]
_SYNC_ENTRIES: list[tuple[str, Any]] = [
    ("stream", lambda a, **k: a.stream("hi", **k)),
    ("resume_stream", lambda a, **k: a.resume_stream("00000000000000000000000105", **k)),
]


# ---------------------------------------------------------------------------
# Strict tenancy
# ---------------------------------------------------------------------------


class TestStrictTenancy:
    @pytest.mark.parametrize(("entry", "call"), _ASYNC_ENTRIES)
    async def test_omitted_tenant_raises_before_io(self, store: Any, entry: str, call: Any) -> None:
        agent = _agent(store, strict_tenancy=True)
        with pytest.raises(TenantScopeRequiredError) as caught:
            await call(agent)
        assert caught.value.operation == entry
        assert isinstance(caught.value, ValueError)
        assert store.calls == []

    @pytest.mark.parametrize(("entry", "call"), _SYNC_ENTRIES)
    def test_omitted_tenant_raises_at_call_time_for_streams(
        self, store: Any, entry: str, call: Any
    ) -> None:
        agent = _agent(store, strict_tenancy=True)
        with pytest.raises(TenantScopeRequiredError):
            call(agent)
        assert store.calls == []

    async def test_explicit_none_and_string_are_accepted(self, store: Any) -> None:
        agent = _agent(store, strict_tenancy=True)
        assert (await agent.run("hi", tenant_id=None)).answer == "ok"
        result = await agent.run("hi", tenant_id="acme")
        assert (await store.get_run(result.run_id)).tenant_id == "acme"

    async def test_not_strict_keeps_041_defaults(self, store: Any) -> None:
        agent = _agent(store, strict_tenancy=False)
        result = await agent.run("hi")
        assert (await store.get_run(result.run_id)).tenant_id is None

    async def test_delegated_child_inherits_parent_tenant(self, store: Any) -> None:
        seen: dict[str, Any] = {}

        @tool()
        async def spawn() -> str:
            """Run a strict child agent without a tenant."""
            child = Agent(
                provider=MockLLM([LLMResponse(text="child done")]),
                prompt="child",
                state_store=store,
                strict_tenancy=True,
            )
            result = await child.run("go")
            seen["child"] = result.run_id
            seen["parent_ctx"] = get_delegation_context()
            return "spawned"

        parent = Agent(
            provider=MockLLM(
                [
                    LLMResponse(tool_calls=[ToolCall(name="spawn", params={})]),
                    LLMResponse(text="ok"),
                ]
            ),
            prompt="parent",
            tools=[spawn],
            state_store=store,
            strict_tenancy=True,
        )
        result = await parent.run("go", tenant_id="acme")
        assert result.answer == "ok"
        child = await store.get_run(seen["child"])
        assert child.tenant_id == "acme"
        assert child.parent_run_id == result.run_id

    def test_delegated_non_run_entry_is_scoped_to_parent_tenant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Ctx:
            tenant_id = "acme"

        monkeypatch.setattr(_tenant_checks, "get_delegation_context", lambda: _Ctx())
        agent = _agent(strict_tenancy=True)
        from dendrux._sentinel import _OMITTED

        resolved = _tenant_checks.resolve_tenant(agent, "submit_input", _OMITTED, default=UNSCOPED)
        assert resolved == "acme"
        new_run = _tenant_checks.resolve_tenant(agent, "run", _OMITTED, default=None)
        assert new_run is None  # the runner's resolve_child_tenant inherits "acme"

    async def test_strict_logs_a_stable_code(
        self, store: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = _agent(store, strict_tenancy=True)
        with pytest.raises(TenantScopeRequiredError):
            await agent.cancel_run("00000000000000000000000105")
        assert any(getattr(r, "code", None) == "tenant_scope_required" for r in caplog.records)


class TestStrictDefaultFutureWarning:
    async def test_future_warning_points_at_the_caller_and_dedups(self, store: Any) -> None:
        agent = _agent(store)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                await agent.run("hi")  # one location: shown once
            await agent.run("hi")  # another location: shown again
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(future) == 2
        assert all(w.filename == __file__ for w in future)
        assert "strict_tenancy" in str(future[0].message)

    def test_visible_under_default_filters(self) -> None:
        from dendrux._sentinel import _OMITTED

        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()  # Python's defaults: FutureWarning is shown
            warnings.simplefilter("default", FutureWarning)
            _tenant_checks.resolve_tenant(_agent(), "run", _OMITTED, default=None)
        assert [w.category for w in caught] == [FutureWarning]

    @pytest.mark.parametrize("chosen", [True, False])
    def test_no_warning_when_chosen(self, chosen: bool) -> None:
        from dendrux._sentinel import _OMITTED

        agent = _agent(strict_tenancy=chosen)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if chosen:
                with pytest.raises(TenantScopeRequiredError):
                    _tenant_checks.resolve_tenant(agent, "run", _OMITTED, default=None)
            else:
                _tenant_checks.resolve_tenant(agent, "run", _OMITTED, default=None)
        assert not [w for w in caught if issubclass(w.category, FutureWarning)]

    def test_no_warning_for_an_explicit_tenant(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _tenant_checks.resolve_tenant(_agent(), "run", "acme", default=None)
        assert not caught


# ---------------------------------------------------------------------------
# tenant_id="" and bad scope types
# ---------------------------------------------------------------------------


class TestEmptyTenant:
    @pytest.mark.parametrize(("entry", "call"), _ASYNC_ENTRIES)
    async def test_empty_string_rejected_before_io(self, store: Any, entry: str, call: Any) -> None:
        with pytest.raises(ValueError, match="tenant_id=''"):
            await call(_agent(store), tenant_id="")
        assert store.calls == []

    @pytest.mark.parametrize(("entry", "call"), _SYNC_ENTRIES)
    def test_empty_string_rejected_for_streams(self, store: Any, entry: str, call: Any) -> None:
        with pytest.raises(ValueError, match="tenant_id=''"):
            call(_agent(store), tenant_id="")

    async def test_direct_runner_call_rejects_empty_tenant(self, store: Any) -> None:
        with pytest.raises(ValueError, match="tenant_id=''"):
            await dendrux.run(
                _agent(), provider=MockLLM([]), user_input="hi", tenant_id="", state_store=store
            )
        assert store.calls == []

    async def test_unscoped_is_not_a_tenant_for_a_new_run(self, store: Any) -> None:
        with pytest.raises(ValueError, match="tenant_id must be"):
            await _agent(store).run("hi", tenant_id=UNSCOPED)  # type: ignore[arg-type]

    async def test_non_string_tenant_rejected(self, store: Any) -> None:
        with pytest.raises(ValueError, match="tenant_id must be"):
            await _agent(store).cancel_run("00000000000000000000000105", tenant_id=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# B8 reserved metadata keys and the metadata size cap
# ---------------------------------------------------------------------------


class TestRunMetadata:
    @pytest.mark.parametrize(
        "call",
        [
            lambda a, m: a.run("hi", metadata=m),
            lambda a, m: a.retry("00000000000000000000000105", metadata=m),
        ],
    )
    async def test_reserved_key_rejected(self, store: Any, call: Any) -> None:
        with pytest.raises(ValueError, match="reserved"):
            await call(_agent(store), {"dendrux.stop_reason": "forged", "app": 1})
        assert store.calls == []

    def test_reserved_key_rejected_for_stream(self, store: Any) -> None:
        with pytest.raises(ValueError, match="reserved"):
            _agent(store).stream("hi", metadata={"dendrux.replay": {}})

    async def test_direct_runner_call_rejects_reserved_key(self, store: Any) -> None:
        with pytest.raises(ValueError, match="reserved"):
            await dendrux.run(
                _agent(),
                provider=MockLLM([]),
                user_input="hi",
                metadata={"dendrux.x": 1},
                state_store=store,
            )
        assert store.calls == []

    async def test_similar_non_reserved_keys_are_fine(self, store: Any) -> None:
        agent = _agent(store)
        result = await agent.run("hi", metadata={"dendrux": 1, "x.dendrux.y": 2, "Dendrux_": 3})
        meta = (await store.get_run(result.run_id)).meta
        assert meta["dendrux"] == 1

    async def test_metadata_over_the_cap_rejected(self, store: Any, caplog: Any) -> None:
        agent = _agent(store, limits=Limits(max_metadata_bytes=64))
        with pytest.raises(ValueError, match="64-byte limit"):
            await agent.run("hi", metadata={"blob": "x" * 100})
        assert store.calls == []
        assert any(getattr(r, "code", None) == "metadata_too_large" for r in caplog.records)

    async def test_metadata_at_the_cap_accepted(self, store: Any) -> None:
        # {"b":"xxx..."} -> 8 bytes of framing + the value.
        agent = _agent(store, limits=Limits(max_metadata_bytes=64))
        assert (await agent.run("hi", metadata={"b": "x" * 56})).answer == "ok"

    async def test_size_counts_utf8_bytes(self, store: Any) -> None:
        agent = _agent(store, limits=Limits(max_metadata_bytes=64))
        with pytest.raises(ValueError, match="limit"):
            await agent.run("hi", metadata={"b": "é" * 40})  # 80 bytes of UTF-8

    async def test_default_cap_is_64_kib(self, store: Any) -> None:
        with pytest.raises(ValueError, match="65536-byte limit"):
            await _agent(store).run("hi", metadata={"b": "x" * 70_000})

    async def test_note_metadata_capped(self, store: Any) -> None:
        agent = _agent(store, limits=Limits(max_metadata_bytes=32))
        with pytest.raises(ValueError, match="note metadata"):
            await agent.submit_note("00000000000000000000000105", "hi", metadata={"b": "x" * 64})
        assert store.calls == []

    async def test_non_dict_metadata_rejected(self, store: Any) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            await _agent(store).run("hi", metadata=["a"])  # type: ignore[arg-type]

    def test_check_run_metadata_helper(self) -> None:
        _tenant_checks.check_run_metadata(None, limit=10)
        _tenant_checks.check_run_metadata({}, limit=10)
        with pytest.raises(ValueError, match="structure_limit"):
            cyclic: dict[str, Any] = {}
            cyclic["self"] = cyclic
            _tenant_checks.check_run_metadata(cyclic, limit=1000)
