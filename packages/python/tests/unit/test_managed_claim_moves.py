"""T0 pure moves: managed claim and recovery code lives in its own modules.

The original modules keep every name, so callers and patch seams are unchanged.
"""

from __future__ import annotations

import pytest

from dendrux.runtime import _managed_claim, _reclaim, _reclaim_managed, submit

CLAIM_NAMES = [
    "managed_submission_descriptor",
    "refuse_managed_results",
    "_managed_adoption_validator",
]
RECLAIM_NAMES = [
    "_pending_adoption",
    "_finish_saved_adoption",
    "_ManagedPending",
    "_refuse_bound_pending",
    "_managed_handoff",
    "_refuse_recovered_managed_results",
    "_handed_off",
]


@pytest.mark.parametrize("name", CLAIM_NAMES)
def test_submit_reexports_moved_claim_name(name: str) -> None:
    assert getattr(submit, name) is getattr(_managed_claim, name)


@pytest.mark.parametrize("name", RECLAIM_NAMES)
def test_reclaim_imports_moved_name(name: str) -> None:
    assert getattr(_reclaim, name) is getattr(_reclaim_managed, name)


def test_core_flow_stays_in_reclaim() -> None:
    assert _reclaim._check_before_claim.__module__ == _reclaim.__name__
    assert _reclaim.reclaim.__module__ == _reclaim.__name__
