"""Integration tests for lifecycle write durability across the backend matrix.

Verifies that the 5 wrapped lifecycle methods in SQLAlchemyStateStore
retry on transient OperationalError and propagate logical failures
(IntegrityError, CAS misses) immediately.

The ``engine`` / ``store`` fixtures live in
``tests/integration/conftest.py`` and parametrize across SQLite and
Postgres. Failure injection is done by monkey-patching
session.execute() or session.commit() to raise OperationalError once —
the retry policy is backend-agnostic.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from dendrux.types import RunStatus

# ------------------------------------------------------------------
# Failure injection helpers
#
# Use plain functions (not bound methods) so Python's descriptor
# protocol passes the session instance as the first argument.
# ------------------------------------------------------------------


class _FailOnceCommit:
    """Context manager: patches AsyncSession.commit to fail once with OperationalError."""

    def __init__(self):
        self.fail_count = 0

    def __enter__(self):
        from sqlalchemy.ext.asyncio import AsyncSession

        original = AsyncSession.commit
        ctx = self

        async def _replacement(session_self):
            if ctx.fail_count == 0:
                ctx.fail_count += 1
                raise OperationalError("database is locked", {}, None)
            return await original(session_self)

        self._patcher = patch.object(AsyncSession, "commit", new=_replacement)
        self._patcher.start()
        return self

    def __exit__(self, *args):
        self._patcher.stop()


class _FailOnceExecute:
    """Context manager: patches AsyncSession.execute to fail once with OperationalError."""

    def __init__(self):
        self.fail_count = 0

    def __enter__(self):
        from sqlalchemy.ext.asyncio import AsyncSession

        original = AsyncSession.execute
        ctx = self

        async def _replacement(session_self, *args, **kwargs):
            if ctx.fail_count == 0:
                ctx.fail_count += 1
                raise OperationalError("connection reset", {}, None)
            return await original(session_self, *args, **kwargs)

        self._patcher = patch.object(AsyncSession, "execute", new=_replacement)
        self._patcher.start()
        return self

    def __exit__(self, *args):
        self._patcher.stop()


# ------------------------------------------------------------------
# Tests: create_run
# ------------------------------------------------------------------


class TestCreateRunDurability:
    """create_run retries on transient failures."""

    async def test_create_run_retries_on_commit_failure(self, store):
        with _FailOnceCommit():
            result = await store.create_run("run1", "TestAgent")

        assert result.outcome == "created"
        assert result.status == RunStatus.RUNNING

        run = await store.get_run("run1")
        assert run is not None
        assert run.agent_name == "TestAgent"

    async def test_create_run_retries_on_execute_failure(self, store):
        with _FailOnceExecute():
            result = await store.create_run("run1", "TestAgent")

        assert result.outcome == "created"
        run = await store.get_run("run1")
        assert run is not None

    async def test_create_run_idempotency_conflict_not_retried(self, store):
        """IntegrityError from idempotency unique constraint is resolved, not retried."""
        await store.create_run(
            "run1", "TestAgent", idempotency_key="key1", idempotency_fingerprint="fp1"
        )

        from dendrux.types import IdempotencyConflictError

        with pytest.raises(IdempotencyConflictError):
            await store.create_run(
                "run2", "TestAgent", idempotency_key="key1", idempotency_fingerprint="fp_different"
            )


# ------------------------------------------------------------------
# Tests: finalize_run
# ------------------------------------------------------------------


class TestFinalizeRunDurability:
    """finalize_run retries on transient failures."""

    async def test_finalize_retries_on_commit_failure(self, store):
        await store.create_run("run1", "TestAgent")

        with _FailOnceCommit():
            ok = await store.finalize_run(
                "run1", status="success", expected_current_status="running"
            )

        assert ok is True
        run = await store.get_run("run1")
        assert run.status == "success"

    async def test_finalize_retries_on_execute_failure(self, store):
        await store.create_run("run1", "TestAgent")

        with _FailOnceExecute():
            ok = await store.finalize_run(
                "run1", status="error", error="boom", expected_current_status="running"
            )

        assert ok is True
        run = await store.get_run("run1")
        assert run.status == "error"

    async def test_finalize_cas_miss_not_retried(self, store):
        """CAS miss (wrong expected status) returns False without retry."""
        await store.create_run("run1", "TestAgent")

        ok = await store.finalize_run(
            "run1", status="success", expected_current_status="waiting_client_tool"
        )
        assert ok is False

        run = await store.get_run("run1")
        assert run.status == "running"


# ------------------------------------------------------------------
# Tests: pause_run
# ------------------------------------------------------------------


class TestPauseRunDurability:
    """pause_run retries on transient failures."""

    async def test_pause_retries_on_commit_failure(self, store):
        await store.create_run("run1", "TestAgent")

        with _FailOnceCommit():
            await store.pause_run(
                "run1",
                status="waiting_client_tool",
                pause_data={"tool": "greet"},
            )

        run = await store.get_run("run1")
        assert run.status == "waiting_client_tool"

    async def test_pause_retries_on_execute_failure(self, store):
        await store.create_run("run1", "TestAgent")

        with _FailOnceExecute():
            await store.pause_run(
                "run1",
                status="waiting_human_input",
                pause_data={"prompt": "confirm?"},
            )

        run = await store.get_run("run1")
        assert run.status == "waiting_human_input"

        pause = await store.get_pause_state("run1")
        assert pause["prompt"] == "confirm?"


# ------------------------------------------------------------------
# Tests: claim_paused_run
# ------------------------------------------------------------------


class TestClaimPausedRunDurability:
    """claim_paused_run retries on transient failures."""

    async def test_claim_retries_on_commit_failure(self, store):
        await store.create_run("run1", "TestAgent")
        await store.pause_run("run1", status="waiting_client_tool", pause_data={"tool": "greet"})

        with _FailOnceCommit():
            ok = await store.claim_paused_run("run1", expected_status="waiting_client_tool")

        assert ok is True
        run = await store.get_run("run1")
        assert run.status == "running"

    async def test_claim_cas_miss_not_retried(self, store):
        """CAS miss on claim returns False, not retried into nonsense."""
        await store.create_run("run1", "TestAgent")
        await store.pause_run("run1", status="waiting_client_tool", pause_data={"tool": "greet"})

        ok = await store.claim_paused_run("run1", expected_status="waiting_human_input")
        assert ok is False

        run = await store.get_run("run1")
        assert run.status == "waiting_client_tool"


# ------------------------------------------------------------------
# Tests: submit_and_claim
# ------------------------------------------------------------------


class TestSubmitAndClaimDurability:
    """submit_and_claim retries on transient failures."""

    async def test_submit_and_claim_retries_on_commit_failure(self, store):
        await store.create_run("run1", "TestAgent")
        await store.pause_run("run1", status="waiting_client_tool", pause_data={"pending": True})

        with _FailOnceCommit():
            ok = await store.submit_and_claim(
                "run1",
                expected_status="waiting_client_tool",
                submitted_data={"result": "done"},
            )

        assert ok is True
        run = await store.get_run("run1")
        assert run.status == "running"

    async def test_submit_and_claim_retries_on_execute_failure(self, store):
        await store.create_run("run1", "TestAgent")
        await store.pause_run("run1", status="waiting_client_tool", pause_data={"pending": True})

        with _FailOnceExecute():
            ok = await store.submit_and_claim(
                "run1",
                expected_status="waiting_client_tool",
                submitted_data={"result": "done"},
            )

        assert ok is True

    async def test_submit_and_claim_cas_miss_not_retried(self, store):
        """CAS miss on submit_and_claim returns False, not retried."""
        await store.create_run("run1", "TestAgent")
        await store.pause_run("run1", status="waiting_client_tool", pause_data={"pending": True})

        ok = await store.submit_and_claim(
            "run1",
            expected_status="waiting_human_input",
            submitted_data={"result": "done"},
        )
        assert ok is False


# ------------------------------------------------------------------
# Tests: retry_transient_db helper directly
# ------------------------------------------------------------------


class TestRetryTransientDb:
    """Direct tests for the retry helper."""

    async def test_retries_operational_error(self):
        from dendrux.runtime.durability import retry_transient_db

        call_count = 0

        async def _attempt():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OperationalError("database is locked", {}, None)
            return "ok"

        result = await retry_transient_db(_attempt, label="test", run_id="r1")
        assert result == "ok"
        assert call_count == 3

    async def test_propagates_after_max_retries(self):
        from dendrux.runtime.durability import retry_transient_db

        async def _always_fail():
            raise OperationalError("persistent failure", {}, None)

        with pytest.raises(OperationalError, match="persistent failure"):
            await retry_transient_db(_always_fail, label="test", run_id="r1")

    async def test_does_not_retry_integrity_error(self):
        from dendrux.runtime.durability import retry_transient_db

        call_count = 0

        async def _attempt():
            nonlocal call_count
            call_count += 1
            raise IntegrityError("unique constraint", {}, None)

        with pytest.raises(IntegrityError):
            await retry_transient_db(_attempt, label="test", run_id="r1")

        assert call_count == 1

    async def test_does_not_retry_value_error(self):
        from dendrux.runtime.durability import retry_transient_db

        call_count = 0

        async def _attempt():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await retry_transient_db(_attempt, label="test", run_id="r1")

        assert call_count == 1
