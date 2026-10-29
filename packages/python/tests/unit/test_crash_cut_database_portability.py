"""Crash-cut tests delegate database safety to the shared test harness."""

from types import SimpleNamespace

import pytest
from sqlalchemy.engine import make_url

from tests.integration import test_060_core_crash_cuts as cuts


async def test_crash_cut_accepts_another_harness_database(monkeypatch, tmp_path):
    class ReachedCutError(Exception):
        pass

    url = make_url("postgresql+asyncpg://localhost/coordinator_test")
    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"), url=url)

    async def reached(url_string, cut, effect_path):
        # No connection is opened: this tests the handoff to the guarded harness.
        assert make_url(url_string).database == "coordinator_test"
        assert cut == "assistant"
        raise ReachedCutError

    monkeypatch.setattr(cuts, "_kill_at_cut", reached)
    with pytest.raises(ReachedCutError):
        await cuts.test_a5_sigkill_committed_boundaries(engine, None, tmp_path, "assistant")
