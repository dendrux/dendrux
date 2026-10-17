"""Unit tests for run-control building blocks.

- RunTaskManager registration of batch run() tasks and interrupt bookkeeping
  (shared tasks, double cancel, swallowed interrupts).
- Tenant inheritance for delegated runs.
- Heartbeat targeting and rate limiting.
- PauseState round-trip of the new pause fields; PauseOutcome shape.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from dendrux.runtime.context import (
    DelegationContext,
    aheartbeat,
    heartbeat,
    reset_delegation_context,
    resolve_child_tenant,
    set_delegation_context,
)
from dendrux.runtime.tasks import RunTaskManager
from dendrux.types import PauseOutcome, PauseState, RunStatus, UsageStats


class TestTaskRegistration:
    async def test_cancel_interrupts_registered_task_once(self) -> None:
        mgr = RunTaskManager()
        started = asyncio.Event()
        seen: dict[str, Any] = {}

        async def body() -> None:
            mgr.register_current("r1")
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                seen["take"] = mgr.take_interrupt("r1")
                task = asyncio.current_task()
                assert task is not None
                seen["remaining"] = task.uncancel()
            finally:
                mgr.release_current("r1")

        task = asyncio.create_task(body())
        await started.wait()
        assert mgr.cancel("r1") is True
        assert mgr.cancel("r1") is True  # second request does not cancel twice
        assert task.cancelling() == 1
        await task
        assert seen == {"take": (True, False), "remaining": 0}
        assert mgr.cancel("r1") is False  # released

    async def test_shared_task_reports_other_pending_runs(self) -> None:
        mgr = RunTaskManager()
        started = asyncio.Event()
        seen: dict[str, Any] = {}

        async def body() -> None:
            mgr.register_current("parent")
            mgr.register_current("child")
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                seen["child"] = mgr.take_interrupt("child")
                seen["parent"] = mgr.take_interrupt("parent")
                task = asyncio.current_task()
                assert task is not None
                task.uncancel()
            finally:
                mgr.release_current("child")
                mgr.release_current("parent")

        task = asyncio.create_task(body())
        await started.wait()
        mgr.cancel("child")
        mgr.cancel("parent")
        assert task.cancelling() == 1
        await task
        assert seen == {"child": (True, True), "parent": (True, False)}

    async def test_take_interrupt_for_unflagged_run(self) -> None:
        mgr = RunTaskManager()
        assert mgr.take_interrupt("nope") == (False, False)
        mgr.register_current("r1")
        assert mgr.take_interrupt("r1") == (False, False)
        mgr.release_current("r1")
        mgr.release_current("r1")  # idempotent

    async def test_release_withdraws_unobserved_interrupt(self) -> None:
        """A cancel the run never observed (swallowed by a tool) must not leak."""
        mgr = RunTaskManager()
        started = asyncio.Event()
        resume = asyncio.Event()

        async def body() -> int:
            mgr.register_current("r1")
            started.set()
            with contextlib.suppress(asyncio.CancelledError):  # a misbehaving tool swallows it
                await asyncio.sleep(10)
            await resume.wait()
            mgr.release_current("r1")
            task = asyncio.current_task()
            assert task is not None
            return task.cancelling()

        task = asyncio.create_task(body())
        await started.wait()
        mgr.cancel("r1")
        await asyncio.sleep(0)
        resume.set()
        assert await task == 0

    async def test_done_task_is_not_cancelled(self) -> None:
        mgr = RunTaskManager()

        async def body() -> None:
            mgr.register_current("r1")

        await asyncio.create_task(body())
        assert mgr.cancel("r1") is False


class TestTenantInheritance:
    def _ctx(self, tenant: str | None) -> DelegationContext:
        return DelegationContext(run_id="p", tenant_id=tenant)

    def test_root_keeps_tenant(self) -> None:
        assert resolve_child_tenant(None, "t1") == "t1"
        assert resolve_child_tenant(None, None) is None

    def test_child_inherits(self) -> None:
        assert resolve_child_tenant(self._ctx("t1"), None) == "t1"
        assert resolve_child_tenant(self._ctx(None), None) is None
        assert resolve_child_tenant(self._ctx("t1"), "t1") == "t1"

    def test_explicit_different_tenant_rejected(self) -> None:
        with pytest.raises(ValueError, match="tenant"):
            resolve_child_tenant(self._ctx("t1"), "t2")
        with pytest.raises(ValueError, match="tenant"):
            resolve_child_tenant(self._ctx(None), "t2")


class _Store:
    def __init__(self) -> None:
        self.touched: list[str] = []

    async def touch_progress(self, run_id: str) -> None:
        self.touched.append(run_id)


class _FailingStore:
    async def touch_progress(self, run_id: str) -> None:
        raise RuntimeError("db down")


class TestHeartbeat:
    async def test_rate_limited_per_run(self) -> None:
        store = _Store()
        ctx = DelegationContext(
            run_id="r1", persisted=True, state_store=store, loop=asyncio.get_running_loop()
        )
        token = set_delegation_context(ctx)
        try:
            await aheartbeat()
            await aheartbeat()  # within 1s: coalesced
            heartbeat()
            await asyncio.sleep(0)
        finally:
            reset_delegation_context(token)
        assert store.touched == ["r1"]

    async def test_sync_heartbeat_schedules_on_loop(self) -> None:
        store = _Store()
        ctx = DelegationContext(
            run_id="r1", persisted=True, state_store=store, loop=asyncio.get_running_loop()
        )
        token = set_delegation_context(ctx)
        try:
            heartbeat()
            for _ in range(5):
                await asyncio.sleep(0)
        finally:
            reset_delegation_context(token)
        assert store.touched == ["r1"]

    async def test_noop_for_ephemeral_run_and_failure_is_swallowed(self) -> None:
        store = _Store()
        token = set_delegation_context(DelegationContext(run_id="r1", state_store=store))
        try:
            heartbeat()
            await aheartbeat()
        finally:
            reset_delegation_context(token)
        assert store.touched == []

        token = set_delegation_context(
            DelegationContext(run_id="r2", persisted=True, state_store=_FailingStore())
        )
        try:
            await aheartbeat()  # logged, not raised
        finally:
            reset_delegation_context(token)


class TestPauseTypes:
    def test_pause_state_round_trips_control_fields(self) -> None:
        pause = PauseState(
            agent_name="a",
            pending_tool_calls=[],
            history=[],
            steps=[],
            iteration=2,
            trace_order_offset=0,
            usage=UsageStats(),
            approval_gates={},
            pause_reason="requested",
            pause_message="hold",
            pending_decisions={"c1": {"approved": False, "reason": "no"}},
        )
        data = pause.to_dict()
        restored = PauseState.from_dict(data)
        assert restored.pause_reason == "requested"
        assert restored.pause_message == "hold"
        assert restored.pending_decisions == {"c1": {"approved": False, "reason": "no"}}
        assert restored.approval_gates == {}

    def test_legacy_pause_state_has_no_control_fields(self) -> None:
        pause = PauseState(
            agent_name="a",
            pending_tool_calls=[],
            history=[],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        data = pause.to_dict()
        assert "pause_reason" not in data and "pending_decisions" not in data
        restored = PauseState.from_dict(data)
        assert restored.pause_reason is None and restored.pending_decisions is None

    def test_paused_status_and_outcome(self) -> None:
        assert RunStatus.PAUSED == "paused"
        outcome = PauseOutcome(
            run_id="r", status_before=RunStatus.RUNNING, status_after=RunStatus.RUNNING
        )
        assert outcome.affected == []
