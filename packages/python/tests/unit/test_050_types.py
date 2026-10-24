"""0.5.0 contracts: new types and fields, with defaults that keep 0.4.1 behaviour."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from dendrux import tool
from dendrux.errors import ProviderRateLimitError
from dendrux.guardrails import PII
from dendrux.llm.base import LLMProvider
from dendrux.llm.mock import MockLLM
from dendrux.store import PausePair, RunDetail, RunSummary
from dendrux.tool import get_tool_def
from dendrux.tools import ToolLookups, build_tool_lookups
from dendrux.types import (
    Budget,
    ErrorInfo,
    Limits,
    PauseState,
    RunResult,
    RunStatus,
    ToolDef,
    UsageStats,
)


class TestErrorInfo:
    def test_from_error_reads_only_safe_fields(self) -> None:
        exc = ProviderRateLimitError(
            provider="openai",
            model="gpt-x",
            status_code=429,
            provider_code="rate_limit_exceeded",
            message="vendor text with secrets",
        )
        info = ErrorInfo.from_error(exc)
        assert info == ErrorInfo(
            category="rate_limit",
            retryable=True,
            safe_message="openai rate_limit (HTTP 429, code rate_limit_exceeded)",
            error_type="ProviderRateLimitError",
            provider="openai",
            model="gpt-x",
            status_code=429,
            provider_code="rate_limit_exceeded",
        )
        assert "secrets" not in json.dumps(info.to_dict())

    def test_dict_roundtrip_and_tolerant_reader(self) -> None:
        info = ErrorInfo(category="timeout", retryable=False, safe_message="p timeout (...)")
        assert ErrorInfo.from_dict(info.to_dict()) == info
        assert ErrorInfo.from_dict({**info.to_dict(), "future_key": 1}) == info
        assert ErrorInfo.from_dict({}) == ErrorInfo(
            category="unknown", retryable=False, safe_message=""
        )

    def test_frozen(self) -> None:
        info = ErrorInfo(category="auth", retryable=False, safe_message="x")
        with pytest.raises(AttributeError):
            info.category = "timeout"  # type: ignore[misc]


class TestLimits:
    def test_defaults(self) -> None:
        limits = Limits()
        assert limits.max_tool_result_bytes == 1024 * 1024
        assert limits.max_input_bytes == 1024 * 1024
        assert limits.max_metadata_bytes == 64 * 1024
        assert limits.max_notes_per_run == 1000

    @pytest.mark.parametrize(
        "field",
        ["max_tool_result_bytes", "max_input_bytes", "max_metadata_bytes", "max_notes_per_run"],
    )
    @pytest.mark.parametrize("value", [0, -1, 1.5, True, None, "10"])
    def test_rejects_non_positive_ints(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field):
            Limits(**{field: value})  # type: ignore[arg-type]


class TestRunResultFields:
    def test_new_fields_default_none(self) -> None:
        result = RunResult(run_id="r", status=RunStatus.SUCCESS)
        assert result.error_info is None
        assert result.pause_id is None

    def test_positional_construction_unchanged(self) -> None:
        result = RunResult("r", RunStatus.ERROR, "a", None, [], 2, UsageStats(), "boom")
        assert (result.answer, result.iteration_count, result.error) == ("a", 2, "boom")


def _pause_state(**kwargs: object) -> PauseState:
    return PauseState(
        agent_name="a",
        pending_tool_calls=[],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=UsageStats(),
        **kwargs,  # type: ignore[arg-type]
    )


class TestPauseStateFields:
    def test_defaults_are_none_and_not_serialized(self) -> None:
        state = _pause_state()
        assert state.pause_id is None
        assert state.schema_version is None
        d = state.to_dict()
        assert "pause_id" not in d
        assert "schema_version" not in d

    def test_roundtrip_when_set(self) -> None:
        state = _pause_state(pause_id="01JPAUSE", schema_version=2)
        d = state.to_dict()
        assert d["pause_id"] == "01JPAUSE"
        assert d["schema_version"] == 2
        back = PauseState.from_dict(d)
        assert (back.pause_id, back.schema_version) == ("01JPAUSE", 2)

    def test_reader_tolerates_missing_and_unknown_keys(self) -> None:
        d = _pause_state().to_dict()
        d["written_by_a_newer_version"] = {"x": 1}
        back = PauseState.from_dict(d)
        assert back.pause_id is None
        assert back.schema_version is None


class TestBudgetCountCacheTokens:
    def test_default_is_not_chosen(self) -> None:
        assert Budget(max_tokens=10).count_cache_tokens is None

    @pytest.mark.parametrize("value", [True, False])
    def test_explicit_choice(self, value: bool) -> None:
        assert Budget(max_tokens=10, count_cache_tokens=value).count_cache_tokens is value

    def test_rejects_non_bool(self) -> None:
        with pytest.raises(ValueError, match="count_cache_tokens"):
            Budget(max_tokens=10, count_cache_tokens=1)  # type: ignore[arg-type]

    def test_positional_order_unchanged(self) -> None:
        b = Budget(100, (0.5,), 1.0, "pause", False)
        assert (b.max_tokens, b.max_cost_usd, b.on_exceed, b.include_children) == (
            100,
            1.0,
            "pause",
            False,
        )


class TestReceivesPii:
    def test_default_false(self) -> None:
        @tool()
        def lookup(q: str) -> str:
            """Look up."""
            return q

        assert get_tool_def(lookup).receives_pii is False
        assert ToolDef(name="x", description="", parameters={}).receives_pii is False

    def test_true(self) -> None:
        @tool(receives_pii=True)
        def send(to: str) -> str:
            """Send."""
            return to

        assert get_tool_def(send).receives_pii is True

    def test_entity_set_is_frozen(self) -> None:
        @tool(receives_pii={"EMAIL", "PHONE"})
        def send(to: str) -> str:
            """Send."""
            return to

        assert get_tool_def(send).receives_pii == frozenset({"EMAIL", "PHONE"})

    @pytest.mark.parametrize("bad", ["EMAIL", 1, None, {""}, {1}])
    def test_rejects_invalid(self, bad: object) -> None:
        with pytest.raises(ValueError, match="receives_pii"):

            @tool(receives_pii=bad)  # type: ignore[arg-type]
            def send(to: str) -> str:
                """Send."""
                return to

    def test_schema_unchanged(self) -> None:
        @tool()
        def a(x: int) -> int:
            """A."""
            return x

        @tool(receives_pii=True)
        def b(x: int) -> int:
            """A."""
            return x

        assert get_tool_def(a).parameters == get_tool_def(b).parameters


class TestPiiRestore:
    def test_default_all(self) -> None:
        assert PII().restore == "all"

    def test_allowlisted(self) -> None:
        assert PII(restore="allowlisted").restore == "allowlisted"

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="restore"):
            PII(restore="some")  # type: ignore[arg-type]


class TestReportsCost:
    def test_default_false(self) -> None:
        assert MockLLM([]).reports_cost is False
        assert isinstance(LLMProvider.reports_cost, property)


class TestToolLookupsCallLimiter:
    def test_default_none(self) -> None:
        lookups = build_tool_lookups([])
        assert lookups.call_limiter is None
        assert ToolLookups({}, {}, {}, {}, {}, {}, {}, {}).call_limiter is None

    def test_carried_through(self) -> None:
        sentinel = object()
        assert build_tool_lookups([], call_limiter=sentinel).call_limiter is sentinel


_NOW = dt.datetime(2026, 10, 24, tzinfo=dt.UTC)


class TestStoreDataclassFields:
    def test_run_summary_positional_and_default(self) -> None:
        s = RunSummary("r", "a", "success", _NOW, _NOW, 1, 2, 3, 0, 0, None, "m", None, 0)
        assert s.tenant_id is None
        assert s.total_reasoning_tokens == 0
        assert RunSummary(
            "r", "a", "success", _NOW, _NOW, 1, 2, 3, 0, 0, None, "m", None, 0, tenant_id="t"
        ).tenant_id == ("t")

    def test_run_detail_positional_and_defaults(self) -> None:
        d = RunDetail(
            "r", "a", "error", _NOW, _NOW, 1, 2, 3, 0, 0, None, "m", None, None, 0, None, None,
            "boom", None,
        )  # fmt: skip
        assert d.tenant_id is None
        assert d.error_info is None
        assert d.pause_requested is False
        info = ErrorInfo(category="auth", retryable=False, safe_message="x")
        d2 = RunDetail(
            "r", "a", "error", _NOW, _NOW, 1, 2, 3, 0, 0, None, "m", None, None, 0, None, None,
            "boom", None, tenant_id="t", error_info=info,
        )  # fmt: skip
        assert (d2.tenant_id, d2.error_info) == ("t", info)

    def test_pause_pair_positional_and_default(self) -> None:
        p = PausePair(1, _NOW, None, None, "waiting_client_tool", [], [], None)
        assert p.pause_id is None
        assert PausePair(1, _NOW, None, None, "x", [], [], None, pause_id="p1").pause_id == "p1"
