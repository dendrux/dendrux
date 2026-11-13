"""Published managed contracts reject broken stores and skip undeclared support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conformance.test_kit_catches_broken_stores import (
    NONE,
    failed_functions,
    mutant,
    run_slice,
)
from tests.conformance.test_kit_verdicts import run_kit

if TYPE_CHECKING:
    from _pytest.pytester import Pytester

pytest_plugins = ["pytester"]


def _creator_after_lost_ack(**claim_overrides: int) -> str:
    return mutant(
        """
    async def bind_managed_child(self, request):
        from dataclasses import replace
        from dendrux.runtime.state import BindManagedChildResult
        try:
            result = await super().bind_managed_child(request)
        except PersistenceError:
            child = await self.get_run(request.child_run_id)
            if child is None:
                raise
            result = BindManagedChildResult(
                "created", child.id, request.parent_wait.pause_id,
                ClaimResult(child.claim_epoch, child.storage_format,
                            await self.get_loop_state(child.id), None),
            )
        if result.outcome == "created":
            result = replace(result, child_claim=replace(result.child_claim, **OVERRIDES))
        return result
    """.replace("OVERRIDES", repr(claim_overrides))
    )


BROKEN = {
    "lost acknowledgement returns the wrong creator epoch": (
        _creator_after_lost_ack(claim_epoch=99),
        "test_managed_bind_lost_ack_grants_only_creator_authority",
    ),
    "lost acknowledgement returns the wrong creator storage format": (
        _creator_after_lost_ack(storage_format=1),
        "test_managed_bind_lost_ack_grants_only_creator_authority",
    ),
    "bind omits the event": (
        mutant("""
        async def _bind_event(self, session, request):
            pass
        """),
        "test_managed_bind_publishes_wait_intent_child_and_event",
    ),
    "bind commits the parent before the child": (
        mutant("""
        async def _bind_park_parent(self, session, request, parent, wait):
            await super()._bind_park_parent(session, request, parent, wait)
            await session.commit()
        """),
        "test_managed_bind_is_all_or_nothing",
    ),
    "readback gives a second execution token": (
        mutant("""
        async def bind_managed_child(self, request):
            from dataclasses import replace
            result = await super().bind_managed_child(request)
            if result.outcome == "existing":
                child = await self.get_run(result.child_run_id)
                result = replace(result, outcome="created", child_claim=ClaimResult(
                    child.claim_epoch, 2, await self.get_loop_state(child.id), None
                ))
            return result
        """),
        "test_managed_bind_readback_never_grants_another_token",
    ),
    "lost acknowledgement lets a replay claim ownership": (
        mutant("""
        async def bind_managed_child(self, request):
            from dataclasses import replace
            result = await super().bind_managed_child(request)
            if result.outcome == "existing":
                child = await self.get_run(result.child_run_id)
                result = replace(result, outcome="created", child_claim=ClaimResult(
                    child.claim_epoch, 2, await self.get_loop_state(child.id), None
                ))
            return result
        """),
        "test_managed_bind_lost_ack_grants_only_creator_authority",
    ),
    "bind writes an event before refusing": (
        mutant("""
        async def bind_managed_child(self, request):
            from dendrux.errors import RunFencedError, RunCancelledError, RunNotFoundError
            try:
                return await super().bind_managed_child(request)
            except (RunFencedError, RunCancelledError, RunNotFoundError):
                await self.save_run_event(request.parent_run_id, event_type="kit.bad_refusal")
                raise
        """),
        "test_managed_bind_refuses_parent_without_writes",
    ),
    "direct child reclaim mutates before refusing": (
        mutant("""
        async def reclaim_run(self, run_id, **kw):
            from dendrux.errors import ManagedChildControlError
            from dendrux.runtime.context import managed_root_reclaim_scope
            with managed_root_reclaim_scope(run_id):
                await super().reclaim_run(run_id, **kw)
            raise ManagedChildControlError(run_id)
        """),
        "test_managed_child_reclaim_refuses_before_any_mutation",
    ),
    "root driven reclaim is refused too": (
        mutant("""
        async def reclaim_run(self, run_id, **kw):
            from dendrux.errors import ManagedChildControlError
            raise ManagedChildControlError(run_id)
        """),
        "test_managed_child_reclaim_refuses_before_any_mutation",
    ),
}


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_managed_contract_rejects_broken_store(pytester: Pytester, name: str) -> None:
    mutation, function = BROKEN[name]
    broken = run_slice(pytester, mutation, function)
    assert failed_functions(broken) == {function}, broken.stdout.str()[-3000:]
    assert "NOT CERTIFIED" in broken.stdout.str()


def test_managed_contract_slice_passes_on_intact_store(pytester: Pytester) -> None:
    intact = run_slice(pytester, NONE, "managed_")
    intact.assert_outcomes(passed=21)
    assert "managed_subagents" in intact.stdout.str()


def test_managed_lost_ack_accepts_a_correct_creator_claim(pytester: Pytester) -> None:
    result = run_slice(
        pytester,
        _creator_after_lost_ack(),
        "test_managed_bind_lost_ack_grants_only_creator_authority",
    )
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize(
    "prerequisite", ["fenced_writes", "retry_ids", "checkpointing", "atomic_completion", "recovery"]
)
def test_narrowed_prerequisite_skips_managed_and_require_refuses(
    pytester: Pytester, prerequisite: str
) -> None:
    mutation = """
from dataclasses import replace
from dendrux.errors import StoreCapabilityError
from dendrux.runtime._capabilities import require_managed_subagents

def mutate(store):
    store.capabilities = replace(store.capabilities, **{PREREQUISITE: False})
    assert store.capabilities.managed_subagents is True
    with pytest.raises(StoreCapabilityError) as refused:
        require_managed_subagents(store)
    assert refused.value.capability == "managed_subagents"
    assert refused.value.missing == (PREREQUISITE,)
    return store
""".replace("PREREQUISITE", repr(prerequisite))
    result = run_slice(pytester, mutation, "managed_")
    result.assert_outcomes(passed=1, skipped=20)
    assert "managed_subagents" not in result.stdout.str().split("capabilities=")[1].split(")")[0]


def test_undeclared_managed_capability_skips_and_require_refuses(pytester: Pytester) -> None:
    mutation = """
from dataclasses import replace
from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

class OrdinaryStore(SQLiteStateStore):
    capabilities = replace(SQLiteStateStore.capabilities, managed_subagents=False)

    async def bind_managed_child(self, request):
        raise AssertionError("an undeclared bind must never be called")

def mutate(store):
    store.__class__ = OrdinaryStore
    store.capabilities = replace(store.capabilities, managed_subagents=False)
    return store
"""
    result = run_slice(pytester, mutation, "managed_")
    result.assert_outcomes(passed=1, skipped=20)
    assert "managed_subagents" not in result.stdout.str().split("capabilities=")[1].split(")")[0]


def test_managed_fault_cuts_are_reported_uncertified_without_injector(pytester: Pytester) -> None:
    result = run_kit(pytester, injector=None, k="managed_")
    result.assert_outcomes(passed=16, skipped=5)
    assert "crash recovery: not certified" in result.stdout.str()
