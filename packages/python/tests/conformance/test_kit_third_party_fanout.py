"""K11: a minimal third-party store that declares ``managed_fanout`` passes the managed kit.

The store is not a ``SQLStateStore`` subclass. It declares its own capabilities and forwards
the state-store protocol to an inner store by composition (explicit methods, so declaration
checks see them). It implements ``get_managed_children`` itself, on the public protocol only
(per-child ``get_run`` / ``get_loop_state`` / ``get_pause_state``), and forwards the atomic
batch bind. The kit's fan-out and 0.7 managed contracts must pass on it, and fan-out discovery
must come from its declaration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.conformance.test_kit_catches_broken_stores import failed_functions, run_slice

if TYPE_CHECKING:
    from _pytest.pytester import Pytester

pytest_plugins = ["pytester"]

THIRD_PARTY = '''
import functools
import inspect

from dendrux.runtime._capabilities import fanout_support
from dendrux.runtime._managed_fanout_store import ManagedChildRow, ManagedFanoutStore
from dendrux.runtime.state import UNSCOPED, ManagedStateStore, StateStore, StoreCapabilities

_FORWARDED = sorted(
    {
        name
        for protocol in (StateStore, ManagedStateStore, ManagedFanoutStore)
        for name, _ in inspect.getmembers(protocol, inspect.isfunction)
        if not name.startswith("_")
    }
    - {"get_managed_children"}
)


def _forward(name):
    async def method(self, *args, **kwargs):
        return await getattr(self._inner, name)(*args, **kwargs)

    method.__name__ = name
    method.__signature__ = inspect.signature(getattr(StateStore, name, None)
                                             or getattr(ManagedStateStore, name, None)
                                             or getattr(ManagedFanoutStore, name))
    return method


class ThirdPartyStore:
    """Composition, not inheritance: only the declaration says what it supports."""

    def __init__(self, inner):
        self._inner = inner
        self._engine = inner._engine  # for the kit's SQL fault injector and clock only
        self.capabilities = StoreCapabilities(
            fenced_writes=True, retry_ids=True, checkpointing=True, atomic_completion=True,
            recovery=True, managed_subagents=True, managed_fanout=True,
            deployment=inner.capabilities.deployment,
        )

    async def get_managed_children(self, parent_run_id, child_run_ids, *, tenant_id):
        ids = list(child_run_ids)
        if len(ids) > 32 or len(set(ids)) != len(ids):
            raise ValueError("at most 32 distinct ids")
        rows = []
        for run_id in ids:
            run = await self._inner.get_run(run_id)
            if run is None or run.parent_run_id != parent_run_id:
                continue
            if tenant_id is not UNSCOPED and run.tenant_id != tenant_id:
                continue
            state = await self._inner.get_loop_state(run_id) or {}
            pause = await self._inner.get_pause_state(run_id) or {}
            rows.append(ManagedChildRow(
                run_id, run.parent_run_id, run.tenant_id, str(run.status),
                state.get("managed"), state.get("managed_wait"), state.get("loop"),
                pause.get("pause_id"), pause.get("pause_reason"),
            ))
        return rows


for _name in _FORWARDED:
    setattr(ThirdPartyStore, _name, _forward(_name))


def mutate(store):
    wrapped = ThirdPartyStore(store)
    assert fanout_support(wrapped).capable
    return wrapped
'''


def test_third_party_fanout_store_passes_the_fanout_contracts(pytester: Pytester) -> None:
    result = run_slice(pytester, THIRD_PARTY, "test_managed_fanout")
    text = result.stdout.str()
    assert failed_functions(result) == set(), text[-3000:]
    result.assert_outcomes(passed=31)
    assert "ThirdPartyStore" in text and "managed_fanout" in text.split("capabilities=")[1]


def test_third_party_fanout_store_passes_the_managed_contracts(pytester: Pytester) -> None:
    result = run_slice(pytester, THIRD_PARTY, "test_managed_subagents")
    assert failed_functions(result) == set(), result.stdout.str()[-3000:]
    result.assert_outcomes(passed=22)


def test_third_party_fanout_read_is_held_to_the_contract(pytester: Pytester) -> None:
    """Revert check: a third-party read that ignores the tenant scope is caught."""
    broken = THIRD_PARTY.replace(
        "            if tenant_id is not UNSCOPED and run.tenant_id != tenant_id:\n"
        "                continue\n",
        "",
    )
    assert broken != THIRD_PARTY
    result = run_slice(pytester, broken, "test_managed_fanout")
    assert failed_functions(result) == {
        "test_managed_children_projection_is_scoped_ordered_and_bounded"
    }, result.stdout.str()[-3000:]
