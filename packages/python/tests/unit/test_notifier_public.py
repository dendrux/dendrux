"""Public notifier types preserve the protocol and shared subscription state."""

import asyncio

import pytest

import dendrux.store as store
from dendrux.db import _notify


@pytest.mark.parametrize("name", ["EventNotifier", "NotifierState", "Subscription"])
def test_notifier_protocol_types_are_public_exports(name):
    assert getattr(store, name, None) is getattr(_notify, name)
    assert name in store.__all__


def test_public_subscription_preserves_shared_health():
    state_type = getattr(store, "NotifierState", None)
    subscription_type = getattr(store, "Subscription", None)
    assert state_type is not None and subscription_type is not None
    state = state_type(healthy=False, generation=0)
    subscription = subscription_type(asyncio.Event(), state)
    state.healthy = True
    state.generation += 1
    assert subscription.state is state
    assert subscription.state.healthy and subscription.state.generation == 1
