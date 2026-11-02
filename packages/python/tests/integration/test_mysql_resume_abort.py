"""Inherited native release and pause visibility contracts on MySQL."""

# ruff: noqa: F401, F811
import pytest

from tests.integration._mysql import mysql_engine, store
from tests.integration.test_resume_abort_pause_and_cancel import (
    test_aborted_resume_keeps_approval_pause_active as _abort_visibility,
)
from tests.integration.test_resume_abort_pause_and_cancel import (
    test_cancel_during_claimed_setup_wins_release as _cancel_setup,
)
from tests.integration.test_resume_abort_pause_and_cancel import (
    test_concurrent_cancel_release_has_one_terminal_event as _release_race,
)


@pytest.mark.parametrize("view", ["pending", "history", "follow", "custom"])
async def test_mysql_aborted_resume_preserves_pause(store, mysql_engine, monkeypatch, view):
    await _abort_visibility(store, mysql_engine, monkeypatch, view)


@pytest.mark.parametrize("kind", ["input", "approve"])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_mysql_claimed_setup_cancel_wins(
    store, mysql_engine, monkeypatch, kind, storage_format
):
    await _cancel_setup(store, mysql_engine, monkeypatch, kind, storage_format)


async def test_mysql_concurrent_cancel_release(store, mysql_engine):
    await _release_race(store, mysql_engine)
