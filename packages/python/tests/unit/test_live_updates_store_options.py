"""Reject bus settings on every database-options entry path."""

import pytest


@pytest.mark.parametrize("key", ["live_updates", "notify_channel"])
@pytest.mark.parametrize("value", [None, False, True, "private-sentinel"])
@pytest.mark.parametrize("entry", ["split", "agent", "engine", "sweep", "sweep_noop", "reader"])
async def test_database_options_refused(entry, key, value):
    from datetime import timedelta

    from dendrux import Agent, sweep
    from dendrux.db._engine_options import split_database_options
    from dendrux.db.session import get_engine
    from dendrux.store import RunStore

    options = {key: value}
    with pytest.raises(ValueError, match=r"Agent\(state_store=") as raised:
        if entry == "split":
            split_database_options(options)
        elif entry == "agent":
            Agent(prompt="test", database_url="sqlite+aiosqlite://", database_options=options)
        elif entry == "engine":
            await get_engine("sqlite+aiosqlite://", database_options=options)
        elif entry == "sweep_noop":
            await sweep(database_url="sqlite+aiosqlite://", database_options=options)
        elif entry == "sweep":
            await sweep(
                database_url="sqlite+aiosqlite://",
                database_options=options,
                stale_running=timedelta(hours=1),
            )
        else:
            RunStore.from_database_url("sqlite+aiosqlite://", database_options=options)
    assert "private-sentinel" not in str(raised.value)
    assert options == {key: value}
