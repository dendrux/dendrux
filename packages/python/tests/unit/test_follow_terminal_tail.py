"""Terminal observation must allow a legacy writer's trailing commit to drain."""

from dendrux.store import EventFeed, RunStore, _follow
from tests.unit.test_follow_coordinator import BatchedState, Clock, attach, ticks
from tests.unit.test_follow_coordinator import clock as clock
from tests.unit.test_follow_coordinator import fast_budget as fast_budget


async def test_terminal_status_waits_for_late_tail(clock: Clock) -> None:
    state = BatchedState()
    state.create("r")
    reader = RunStore(state)
    try:
        follow = await attach(EventFeed(reader), "r")
        state.statuses["r"] = "error"
        coordinator = reader._follow_coordinator()
        coordinator.request(coordinator.runs["r"], "hint")
        await ticks(40)
        assert follow._follower.end_reason is None
        state.write("r", "run.error")
        clock.advance(_follow.TERMINAL_DRAIN_GRACE_S)
        await ticks(50)
        assert [e.event_type for e in follow._follower.buffer] == ["run.error"]
        assert follow._follower.end_reason == "terminal"
    finally:
        await reader.close()


async def test_terminal_at_start_waits_for_trailing_event(clock: Clock) -> None:
    import asyncio

    state = BatchedState()
    state.create("r", status="error")
    reader = RunStore(state)
    follow = reader.follow("r", tenant_id=None)
    starting = asyncio.create_task(follow.start())
    try:
        await ticks()
        state.write("r", "run.error")
        clock.advance(0.1)
        await ticks(40)
        await starting
        assert not follow.drained_terminal
    finally:
        await follow.aclose()
        await reader.close()
