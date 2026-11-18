"""K11: a third-party store built by composition runs the managed and fan-out kit contracts.

Claim (narrowed after review): composition compatibility. The store is not a ``SQLStateStore``
subclass; it declares its own capabilities and forwards every protocol method, including the
atomic batch bind and the single-statement status projection, to an inner store through
explicit methods (so declaration checks see them). This proves the kit, the capability
declaration and ``fanout_support`` work through the protocol alone. It is not a second,
independent implementation of the projection: a projection built from several reads could
mix states across a concurrent pause, and the kit cannot reject that deterministically (it
sees only protocol calls, so it cannot run a writer between an implementation's internal
reads; a concurrent probe would rely on timing). The SQL store's single snapshot is evidenced
by its statement budget instead (one SELECT, ``test_bind_and_status_read_statement_counts``).
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
from dendrux.runtime._managed_fanout_store import ManagedFanoutStore
from dendrux.runtime.state import ManagedStateStore, StateStore, StoreCapabilities

_FORWARDED = sorted(
    {
        name
        for protocol in (StateStore, ManagedStateStore, ManagedFanoutStore)
        for name, _ in inspect.getmembers(protocol, inspect.isfunction)
        if not name.startswith("_")
    }
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
    result.assert_outcomes(passed=42)
    assert "ThirdPartyStore" in text and "managed_fanout" in text.split("capabilities=")[1]


def test_third_party_fanout_store_passes_the_managed_contracts(pytester: Pytester) -> None:
    result = run_slice(pytester, THIRD_PARTY, "test_managed_subagents")
    assert failed_functions(result) == set(), result.stdout.str()[-3000:]
    result.assert_outcomes(passed=22)


def test_third_party_forwarding_is_held_to_the_contract(pytester: Pytester) -> None:
    """Revert check: forwarding the status read without the caller's tenant scope is caught."""
    broken = THIRD_PARTY.replace(
        "for _name in _FORWARDED:",
        """async def _unscoped(self, parent_run_id, child_run_ids, *, tenant_id):
    from dendrux.runtime.state import UNSCOPED
    return await self._inner.get_managed_children(parent_run_id, child_run_ids, tenant_id=UNSCOPED)


for _name in _FORWARDED:""",
    ).replace(
        "def mutate(store):",
        "ThirdPartyStore.get_managed_children = _unscoped\n\n\ndef mutate(store):",
    )
    assert broken != THIRD_PARTY
    result = run_slice(pytester, broken, "test_managed_fanout")
    assert failed_functions(result) == {
        "test_managed_children_projection_is_scoped_ordered_and_bounded"
    }, result.stdout.str()[-3000:]
