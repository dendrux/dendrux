"""Runtime integration for reasoning/thinking — usage rollup + public DTOs.

Pure-function tests; the full agent.run/stream + Postgres persistence path is
covered by the integration suite and the e2e example.
"""

from __future__ import annotations

from types import SimpleNamespace

from dendrux.loops.react import _accumulate_usage, _snapshot_usage
from dendrux.store import _llm_to_public, _run_to_detail
from dendrux.types import UsageStats


class TestUsageRollup:
    def test_accumulate_adds_reasoning_tokens(self) -> None:
        total = UsageStats(input_tokens=0, output_tokens=0, total_tokens=0)
        _accumulate_usage(total, UsageStats(output_tokens=10, total_tokens=10, reasoning_tokens=4))
        _accumulate_usage(total, UsageStats(output_tokens=20, total_tokens=20, reasoning_tokens=6))
        assert total.reasoning_tokens == 10

    def test_accumulate_ignores_none_reasoning(self) -> None:
        total = UsageStats(reasoning_tokens=5)
        _accumulate_usage(total, UsageStats(output_tokens=10, total_tokens=10))
        assert total.reasoning_tokens == 5  # unchanged when step has None

    def test_snapshot_carries_reasoning_tokens(self) -> None:
        snap = _snapshot_usage(UsageStats(output_tokens=10, total_tokens=10, reasoning_tokens=7))
        assert snap.reasoning_tokens == 7


def _fake_llm_record(**over: object) -> SimpleNamespace:
    base = dict(
        iteration_index=1,
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        cost_usd=None,
        duration_ms=12,
        provider_request=None,
        provider_response=None,
        created_at=None,
        reasoning_tokens=None,
        semantic_response=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestPublicDTOs:
    def test_llm_call_exposes_reasoning(self) -> None:
        rec = _fake_llm_record(
            reasoning_tokens=42,
            semantic_response={"text": "hi", "reasoning": "let me think"},
        )
        call = _llm_to_public(rec)
        assert call.reasoning_tokens == 42
        assert call.reasoning == "let me think"

    def test_llm_call_reasoning_none_when_absent(self) -> None:
        call = _llm_to_public(_fake_llm_record())
        assert call.reasoning_tokens is None
        assert call.reasoning is None

    def test_llm_call_reasoning_none_when_semantic_has_no_reasoning(self) -> None:
        call = _llm_to_public(_fake_llm_record(semantic_response={"text": "hi"}))
        assert call.reasoning is None

    def test_run_detail_exposes_total_reasoning_tokens(self) -> None:
        rec = SimpleNamespace(
            id="r1",
            agent_name="A",
            status="success",
            created_at=None,
            updated_at=None,
            iteration_count=2,
            total_input_tokens=100,
            total_output_tokens=50,
            total_cache_read_tokens=0,
            total_cache_creation_tokens=0,
            total_reasoning_tokens=12,
            total_cost_usd=None,
            model="claude-sonnet-4-6",
            strategy="react",
            parent_run_id=None,
            delegation_level=0,
            input_data=None,
            answer="ok",
            error=None,
            failure_reason=None,
        )
        detail = _run_to_detail(rec)
        assert detail.total_reasoning_tokens == 12
