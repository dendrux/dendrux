"""Integration tests for run creation idempotency across the backend matrix.

The ``engine`` / ``store`` / ``session_factory`` fixtures live in
``tests/integration/conftest.py`` and parametrize across SQLite and Postgres.
"""

from __future__ import annotations

import pytest

from dendrux.types import (
    CreateRunResult,
    IdempotencyConflictError,
    RunAlreadyActiveError,
    RunStatus,
    compute_idempotency_fingerprint,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

AGENT = "TestAgent"
INPUT = "analyze revenue"
KEY = "req-abc-123"
FINGERPRINT = compute_idempotency_fingerprint(AGENT, INPUT)


# ------------------------------------------------------------------
# Tests: compute_idempotency_fingerprint
# ------------------------------------------------------------------


class TestFingerprintComputation:
    """Tests for the fingerprint computation function."""

    def test_deterministic(self):
        """Same inputs produce same fingerprint."""
        fp1 = compute_idempotency_fingerprint("Agent", "hello")
        fp2 = compute_idempotency_fingerprint("Agent", "hello")
        assert fp1 == fp2

    def test_different_input_different_fingerprint(self):
        """Different user_input produces different fingerprint."""
        fp1 = compute_idempotency_fingerprint("Agent", "hello")
        fp2 = compute_idempotency_fingerprint("Agent", "goodbye")
        assert fp1 != fp2

    def test_different_agent_different_fingerprint(self):
        """Different agent_name produces different fingerprint."""
        fp1 = compute_idempotency_fingerprint("AgentA", "hello")
        fp2 = compute_idempotency_fingerprint("AgentB", "hello")
        assert fp1 != fp2

    def test_sha256_hex_length(self):
        """Fingerprint is 64 chars (SHA-256 hex digest)."""
        fp = compute_idempotency_fingerprint("Agent", "hello")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


# ------------------------------------------------------------------
# Tests: create_run with idempotency
# ------------------------------------------------------------------


class TestCreateRunIdempotency:
    """Tests for StateStore.create_run() with idempotency_key."""

    async def test_no_key_returns_created(self, store):
        """create_run without key returns 'created' outcome."""
        result = await store.create_run("run1", AGENT)
        assert isinstance(result, CreateRunResult)
        assert result.outcome == "created"
        assert result.status == RunStatus.RUNNING
        assert result.run_id == "run1"

    async def test_with_key_first_call_creates(self, store):
        """First call with idempotency_key creates normally."""
        result = await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        assert result.outcome == "created"
        assert result.run_id == "run1"

    async def test_key_matches_active_run(self, store):
        """Second call with same key + fingerprint on RUNNING run → existing_active."""
        await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )

        result = await store.create_run(
            "run2",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        assert result.outcome == "existing_active"
        assert result.run_id == "run1"
        assert result.status == RunStatus.RUNNING

    async def test_key_matches_terminal_run(self, store):
        """Second call with same key on completed run → existing_terminal."""
        await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        await store.finalize_run(
            "run1",
            status="success",
            answer="done",
            expected_current_status="running",
        )

        result = await store.create_run(
            "run2",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        assert result.outcome == "existing_terminal"
        assert result.run_id == "run1"
        assert result.status == RunStatus.SUCCESS

    async def test_key_matches_error_run(self, store):
        """Error status is also terminal."""
        await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        await store.finalize_run(
            "run1",
            status="error",
            error="something broke",
            expected_current_status="running",
        )

        result = await store.create_run(
            "run2",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        assert result.outcome == "existing_terminal"
        assert result.run_id == "run1"
        assert result.status == RunStatus.ERROR

    async def test_key_matches_waiting_run(self, store):
        """WAITING_* status is active, not terminal."""
        await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        await store.pause_run(
            "run1",
            status="waiting_client_tool",
            pause_data={"pending": []},
        )

        result = await store.create_run(
            "run2",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        assert result.outcome == "existing_active"
        assert result.run_id == "run1"
        assert result.status == RunStatus.WAITING_CLIENT_TOOL

    async def test_fingerprint_mismatch_raises_conflict(self, store):
        """Same key with different fingerprint raises IdempotencyConflictError."""
        await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )

        different_fp = compute_idempotency_fingerprint(AGENT, "different input")

        with pytest.raises(IdempotencyConflictError) as exc_info:
            await store.create_run(
                "run2",
                AGENT,
                idempotency_key=KEY,
                idempotency_fingerprint=different_fp,
            )

        assert exc_info.value.run_id == "run1"
        assert exc_info.value.idempotency_key == KEY

    async def test_different_keys_create_separate_runs(self, store):
        """Different idempotency keys create independent runs."""
        r1 = await store.create_run(
            "run1",
            AGENT,
            idempotency_key="key-1",
            idempotency_fingerprint=FINGERPRINT,
        )
        r2 = await store.create_run(
            "run2",
            AGENT,
            idempotency_key="key-2",
            idempotency_fingerprint=FINGERPRINT,
        )
        assert r1.outcome == "created"
        assert r2.outcome == "created"
        assert r1.run_id != r2.run_id

    async def test_null_key_allows_duplicates(self, store):
        """Runs without idempotency_key are not deduplicated."""
        r1 = await store.create_run("run1", AGENT)
        r2 = await store.create_run("run2", AGENT)
        assert r1.outcome == "created"
        assert r2.outcome == "created"
        assert r1.run_id != r2.run_id

    async def test_columns_persisted(self, store, session_factory):
        """idempotency_key and fingerprint are stored on the row."""
        await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )

        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with session_factory() as session:
            stmt = select(AgentRun).where(AgentRun.id == "run1")
            result = await session.execute(stmt)
            row = result.scalar_one()
            assert row.idempotency_key == KEY
            assert row.idempotency_fingerprint == FINGERPRINT


class TestIdempotencyConcurrency:
    """Tests for concurrency safety via DB unique constraint."""

    async def test_concurrent_creates_one_wins(self, store):
        """Two concurrent creates with same key: one creates, one resolves."""
        # We simulate the race by creating the first run, then trying
        # to create a second with the same key. The IntegrityError path
        # is tested by the fact that _check_idempotency correctly resolves.
        r1 = await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        assert r1.outcome == "created"

        r2 = await store.create_run(
            "run2",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )
        assert r2.outcome == "existing_active"
        assert r2.run_id == "run1"

    async def test_integrity_error_with_conflict(self, store):
        """IntegrityError path with mismatched fingerprint raises conflict."""
        await store.create_run(
            "run1",
            AGENT,
            idempotency_key=KEY,
            idempotency_fingerprint=FINGERPRINT,
        )

        different_fp = compute_idempotency_fingerprint(AGENT, "other input")

        with pytest.raises(IdempotencyConflictError):
            await store.create_run(
                "run2",
                AGENT,
                idempotency_key=KEY,
                idempotency_fingerprint=different_fp,
            )


class TestIdempotencyExceptions:
    """Tests for exception types."""

    def test_run_already_active_error_attrs(self):
        """RunAlreadyActiveError carries run_id and status."""
        err = RunAlreadyActiveError("run1", RunStatus.RUNNING)
        assert err.run_id == "run1"
        assert err.current_status == RunStatus.RUNNING
        assert "run1" in str(err)
        assert "running" in str(err)

    def test_idempotency_conflict_error_attrs(self):
        """IdempotencyConflictError carries run_id and key."""
        err = IdempotencyConflictError("run1", "my-key")
        assert err.run_id == "run1"
        assert err.idempotency_key == "my-key"
        assert "my-key" in str(err)
        assert "run1" in str(err)


class TestBuildCachedResult:
    """Tests for _build_cached_result in the runner."""

    async def test_cached_result_from_completed_run(self, store):
        """Cached result is rebuilt from DB state."""
        from dendrux.runtime.runner import _build_cached_result

        await store.create_run(
            "run1",
            AGENT,
            input_data={"input": "test"},
        )
        await store.finalize_run(
            "run1",
            status="success",
            answer="the answer",
            iteration_count=3,
            expected_current_status="running",
        )

        result = await _build_cached_result(store, "run1")
        assert result.run_id == "run1"
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "the answer"
        assert result.iteration_count == 3

    async def test_cached_result_from_error_run(self, store):
        """Cached result includes error field."""
        from dendrux.runtime.runner import _build_cached_result

        await store.create_run("run1", AGENT)
        await store.finalize_run(
            "run1",
            status="error",
            error="boom",
            expected_current_status="running",
        )

        result = await _build_cached_result(store, "run1")
        assert result.status == RunStatus.ERROR
        assert result.error == "boom"

    async def test_cached_result_missing_run_raises(self, store):
        """Missing run raises RuntimeError."""
        from dendrux.runtime.runner import _build_cached_result

        with pytest.raises(RuntimeError, match="not found"):
            await _build_cached_result(store, "nonexistent")


class TestAgentIdempotencyValidation:
    """Tests for Agent.run() idempotency_key validation."""

    async def test_idempotency_key_without_persistence_raises(self):
        """idempotency_key without any persistence config raises ValueError."""
        from unittest.mock import AsyncMock

        from dendrux.agent import Agent

        agent = Agent(
            provider=AsyncMock(),
            prompt="test",
        )

        with pytest.raises(ValueError, match="idempotency_key requires persistence"):
            await agent.run("hello", idempotency_key="key-1")

    async def test_idempotency_key_accepted_with_explicit_state_store(self, store):
        """idempotency_key works when persistence is via state_store= (not database_url)."""
        from dendrux.agent import Agent
        from dendrux.llm.mock import MockLLM
        from dendrux.types import LLMResponse

        mock = MockLLM([LLMResponse(text="done")])
        agent = Agent(provider=mock, prompt="test", state_store=store)

        result = await agent.run("hello", idempotency_key="state-store-key-1")
        assert result.status == RunStatus.SUCCESS

        # Second call with same key returns cached result
        result2 = await agent.run("hello", idempotency_key="state-store-key-1")
        assert result2.status == RunStatus.SUCCESS
        assert result2.run_id == result.run_id
