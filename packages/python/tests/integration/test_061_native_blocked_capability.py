"""The runtime finds the native blocked-mark path by a private capability, not by store class."""

import pytest

from dendrux.errors import PersistenceError
from dendrux.runtime import _blocked_content
from dendrux.runtime.state import claimed_store


class _CustomStore:
    """Stands in for a developer's store; only the attributes under test matter."""


class _ClaimsByFlag(_CustomStore):
    _native_blocked_mark = True


class _ClaimsByMethod(_CustomStore):
    def _native_blocked_mark(self):
        return object()

    async def _blocked_run_ids(self, run_ids, *, tenant_id):
        return set()


def test_the_native_store_has_the_capability(store):
    assert _blocked_content._native_blocked_store(store) is store
    assert _blocked_content._native_store(store) is store


def test_a_claimed_native_store_resolves_to_the_shared_store(store):
    claimed = claimed_store(store, "01HZZZZZZZZZZZZZZZZZZZZZZZ", 1)
    assert claimed is not store
    assert _blocked_content._native_blocked_store(claimed) is store


@pytest.mark.parametrize("custom", [_CustomStore, _ClaimsByFlag, _ClaimsByMethod])
def test_a_custom_store_is_never_trusted_as_native(custom):
    # Setting the attribute, or a method returning anything but the runtime's private
    # marker, keeps the store on the custom path: mark first, or fail closed.
    assert _blocked_content._native_blocked_store(custom()) is None
    with pytest.raises(PersistenceError):
        _blocked_content._native_store(custom())


def test_a_store_without_the_capability_is_not_native_even_when_mocked():
    from unittest.mock import MagicMock

    assert _blocked_content._native_blocked_store(MagicMock()) is None
