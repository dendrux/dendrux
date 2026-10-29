"""0.5.0 T1 change round: ``RunStore.follow`` / ``EventFeed`` and ``redact_event``.

The hardened live feed as a store-level primitive: tenant scope required and
fail-closed, one shared poller per run, terminal close with a drain, a
follower cap, max duration, lagging followers ended with a cursor, and
redaction that keeps only the active pause's arguments.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import warnings
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.errors import RunNotFoundError
from dendrux.runtime.state import UNSCOPED, SQLAlchemyStateStore
from dendrux.store import (
    EventFeed,
    FollowCapacityError,
    FollowEnd,
    FollowHeartbeat,
    RunStore,
    StoredEvent,
    redact_event,
)


@pytest.fixture
async def engine(tmp_path: Any) -> Any:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'follow.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def state(engine: Any) -> SQLAlchemyStateStore:
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def store(engine: Any) -> RunStore:
    return RunStore.from_engine(engine)


def _feed(store: RunStore, **kw: Any) -> EventFeed:
    params: dict[str, Any] = {
        "poll_interval_s": 0.01,
        "poll_max_s": 0.02,
        "status_every": 1,
        "heartbeat_s": 30.0,
    }
    params.update(kw)
    return EventFeed(store, **params)


async def _collect(follow: Any) -> list[Any]:
    items: list[Any] = []
    async with follow as f:
        async for item in f:
            items.append(item)
    return items


def _seqs(items: list[Any]) -> list[int]:
    return [i.sequence_index for i in items if isinstance(i, StoredEvent)]


_PAUSED = {
    "status": "waiting_client_tool",
    "pending_tool_calls": [{"id": "c1", "name": "read", "target": "client", "params": {"s": 1}}],
}


class TestRedactEvent:
    def _event(self, event_type: str, data: dict[str, Any]) -> StoredEvent:
        return StoredEvent(
            sequence_index=0,
            iteration_index=0,
            event_type=event_type,
            correlation_id=None,
            data=data,
            created_at=None,
        )

    def test_hides_prompt_and_provider_bodies(self) -> None:
        ev = self._event(
            "run.started",
            {"agent_name": "a", "system_prompt": "S", "provider_request": {}, "pii_mapping": {}},
        )
        assert redact_event(ev).data == {"agent_name": "a"}
        assert redact_event(ev, include_payloads=True) is ev

    def test_past_pause_hides_args_active_pause_keeps_them(self) -> None:
        ev = self._event("run.paused", _PAUSED)
        assert "params" not in redact_event(ev).data["pending_tool_calls"][0]
        kept = redact_event(ev, active_pause=True).data["pending_tool_calls"][0]
        assert kept["params"] == {"s": 1}
        assert ev.data == _PAUSED  # the input is not mutated


class TestApprovalPlaceholders:
    def test_redact_event_shows_approval_args_as_placeholders_in_every_mode(self) -> None:
        mapping = {"<<EMAIL_ADDRESS_1>>": "bob@example.com"}
        ev = StoredEvent(
            sequence_index=3,
            iteration_index=0,
            event_type="run.paused",
            correlation_id=None,
            data={
                "status": "waiting_approval",
                "pending_tool_calls": [
                    {"id": "c", "name": "send", "params": {"to": "bob@example.com"}}
                ],
            },
            created_at=None,
        )
        for payloads in (False, True):
            for status in ("waiting_approval", "waiting_human_input"):
                unchecked = dataclasses.replace(ev, data={**ev.data, "status": status})
                hidden_without_mapping = redact_event(
                    unchecked, include_payloads=payloads, active_pause=True
                )
                call = hidden_without_mapping.data["pending_tool_calls"][0]
                assert call["params"] is None and call["args_hidden"] is True
            hidden = redact_event(
                ev, include_payloads=payloads, active_pause=True, pii_mapping=mapping
            )
            assert hidden.data["pending_tool_calls"][0]["params"] is None
            assert hidden.data["pending_tool_calls"][0]["args_hidden"] is True
            shown = redact_event(
                ev,
                include_payloads=payloads,
                active_pause=True,
                pii_mapping=mapping,
                approval_params={"c": {"to": "<<EMAIL_ADDRESS_1>>"}},
            )
            assert shown.data["pending_tool_calls"][0]["params"] == {"to": "<<EMAIL_ADDRESS_1>>"}
        client = dataclasses.replace(ev, data={**ev.data, "status": "waiting_client_tool"})
        kept = redact_event(client, active_pause=True, pii_mapping=mapping)
        assert kept.data["pending_tool_calls"][0]["params"] == {"to": "bob@example.com"}  # 0.4.1


class TestScope:
    async def test_tenant_is_required(self, store: RunStore) -> None:
        with pytest.raises(TypeError):
            store.follow("00000000000000000000000301")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            await store.get_pending("00000000000000000000000301")  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad", ["", 7, ["A"]])
    async def test_invalid_scope_fails_closed(self, store: RunStore, bad: Any) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            store.follow("00000000000000000000000301", tenant_id=bad)
        with pytest.raises(ValueError, match="tenant_id"):
            await store.get_pending("00000000000000000000000301", tenant_id=bad)

    async def test_foreign_and_missing_runs_raise_not_found(
        self, state: Any, store: RunStore
    ) -> None:
        await state.create_run("ra", "a", tenant_id="A")
        for run_id, tenant in (("ra", "B"), ("ra", None), ("nope", "A")):
            follow = store.follow(run_id, tenant_id=tenant)
            with pytest.raises(RunNotFoundError):
                await follow.start()
            await follow.aclose()
            with pytest.raises(RunNotFoundError):
                await anext(aiter(store.follow(run_id, tenant_id=tenant)))
        feed = store._event_feed
        assert feed is not None and feed.followers == 0  # nothing was held

    async def test_explicit_unscoped_reads_any_tenant(self, state: Any, store: RunStore) -> None:
        await state.create_run("ra", "a", tenant_id="A")
        await state.save_run_event("ra", event_type="x")
        await state.finalize_run("ra", status="success")
        items = await _collect(store.follow("ra", tenant_id=UNSCOPED))
        assert _seqs(items) == [0]


class TestFollowLifecycle:
    async def test_terminal_run_drains_then_ends(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a", tenant_id="T")
        for i in range(3):
            await state.save_run_event("00000000000000000000000301", event_type=f"e{i}")
        await state.finalize_with_event(
            "00000000000000000000000301",
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
        )
        items = await _collect(_feed(store).follow("00000000000000000000000301", tenant_id="T"))
        assert _seqs(items) == [0, 1, 2, 3]
        assert items[-1] == FollowEnd("terminal", 3)

    async def test_drained_terminal_is_reported_at_start(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event("00000000000000000000000301", event_type="run.completed")
        await state.finalize_run("00000000000000000000000301", status="success")
        feed = _feed(store)
        follow = feed.follow("00000000000000000000000301", tenant_id=None, after_sequence_index=0)
        await follow.start()
        assert follow.drained_terminal
        assert [i async for i in follow] == [FollowEnd("terminal", 0)]
        assert feed.followers == 0

    async def test_live_run_ends_when_it_finishes(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")

        async def finish() -> None:
            await asyncio.sleep(0.05)
            await state.save_run_event("00000000000000000000000301", event_type="tick")
            await state.finalize_run(
                "00000000000000000000000301", status="error"
            )  # no terminal event

        writer = asyncio.create_task(finish())
        items = await _collect(_feed(store).follow("00000000000000000000000301", tenant_id=None))
        await writer
        assert _seqs(items) == [0]
        assert items[-1] == FollowEnd("terminal", 0)

    async def test_until_terminal_false_keeps_following(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event("00000000000000000000000301", event_type="x")
        await state.finalize_run("00000000000000000000000301", status="success")
        follow = _feed(store).follow(
            "00000000000000000000000301", tenant_id=None, until_terminal=False
        )
        async with follow as f:
            first = await asyncio.wait_for(anext(f), 5)
            await state.save_run_event("00000000000000000000000301", event_type="late")
            second = await asyncio.wait_for(anext(f), 5)
        assert (first.event_type, second.event_type) == ("x", "late")

    async def test_max_duration_ends_with_cursor(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event("00000000000000000000000301", event_type="tick")
        items = await _collect(
            _feed(store).follow("00000000000000000000000301", tenant_id=None, max_duration_s=0.1)
        )
        assert items[-1] == FollowEnd("max_duration", 0)

    async def test_heartbeat_when_idle(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        async with _feed(store, heartbeat_s=0.02).follow(
            "00000000000000000000000301", tenant_id=None
        ) as f:
            assert isinstance(await asyncio.wait_for(anext(f), 5), FollowHeartbeat)

    async def test_lagging_follower_ends_with_its_cursor(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        async with _feed(store, buffer=2).follow("00000000000000000000000301", tenant_id=None) as f:
            first = asyncio.ensure_future(anext(f))
            await asyncio.sleep(0.03)
            for i in range(6):
                await state.save_run_event("00000000000000000000000301", event_type=f"e{i}")
            await asyncio.wait_for(first, 5)
            rest = [i async for i in f]
        assert isinstance(rest[-1], FollowEnd) and rest[-1].reason == "lagging"
        assert isinstance(rest[-1].cursor, int)


class TestSharing:
    async def test_many_followers_share_one_poller(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        feed = _feed(store)
        polls = {"n": 0}
        original = store.get_events

        async def counting(run_id: str, **kw: Any) -> Any:
            polls["n"] += 1
            return await original(run_id, **kw)

        store.get_events = counting  # type: ignore[method-assign]
        follows = [feed.follow("00000000000000000000000301", tenant_id=None) for _ in range(20)]
        for f in follows:
            await f.start()
        waiting = [asyncio.ensure_future(anext(f)) for f in follows]
        await asyncio.sleep(0.15)
        assert feed.watched_runs == 1 and feed.followers == 20
        assert polls["n"] < 30  # one poller, not twenty
        await state.save_run_event("00000000000000000000000301", event_type="hello")
        items = await asyncio.wait_for(asyncio.gather(*waiting), 5)
        assert all(i.event_type == "hello" for i in items)
        for f in follows:
            await f.aclose()
        assert feed.watched_runs == 0 and feed.followers == 0

    async def test_late_joiner_catches_up(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        feed = _feed(store)
        async with feed.follow("00000000000000000000000301", tenant_id=None) as first:
            pending = asyncio.ensure_future(anext(first))
            await asyncio.sleep(0.03)
            for i in range(5):
                await state.save_run_event("00000000000000000000000301", event_type=f"e{i}")
            await asyncio.wait_for(pending, 5)
            await asyncio.sleep(0.05)
            async with feed.follow("00000000000000000000000301", tenant_id=None) as late:
                got = [(await anext(late)).sequence_index for _ in range(5)]
        assert got == [0, 1, 2, 3, 4]

    async def test_bogus_cursor_does_not_hide_events_from_others(
        self, state: Any, store: RunStore
    ) -> None:
        await state.create_run("00000000000000000000000301", "a")
        feed = _feed(store)
        liar = feed.follow(
            "00000000000000000000000301", tenant_id=None, after_sequence_index=10_000
        )
        honest = feed.follow("00000000000000000000000301", tenant_id=None)
        liar_next = asyncio.ensure_future(anext(liar))
        await asyncio.sleep(0.03)
        honest_next = asyncio.ensure_future(anext(honest))
        await asyncio.sleep(0.03)
        await state.save_run_event("00000000000000000000000301", event_type="real")
        assert (await asyncio.wait_for(honest_next, 5)).event_type == "real"
        liar_next.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await liar_next
        await liar.aclose()
        await honest.aclose()

    async def test_capacity_is_enforced_and_released(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        feed = _feed(store, max_followers=2, max_followers_per_tenant=None)  # global cap only
        a, b = (
            feed.follow("00000000000000000000000301", tenant_id=None),
            feed.follow("00000000000000000000000301", tenant_id=None),
        )
        await a.start()
        await b.start()
        c = feed.follow("00000000000000000000000301", tenant_id=None)
        with pytest.raises(FollowCapacityError) as caught:
            await c.start()
        assert caught.value.limit == 2
        await a.aclose()
        await c.start()
        assert feed.followers == 2
        await b.aclose()
        await c.aclose()
        await c.aclose()  # idempotent
        assert feed.followers == 0


class TestFollowRedaction:
    async def test_active_pause_args_visible_past_hidden(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a", tenant_id="T")
        await state.save_run_event(
            "00000000000000000000000301", event_type="run.started", data={"system_prompt": "S"}
        )
        await state.save_run_event(
            "00000000000000000000000301", event_type="run.paused", data=_PAUSED
        )  # past
        await state.save_run_event("00000000000000000000000301", event_type="run.resumed", data={})
        await state.save_run_event(
            "00000000000000000000000301", event_type="run.paused", data=_PAUSED
        )  # active
        await state.pause_run(
            "00000000000000000000000301", status="waiting_client_tool", pause_data={"x": 1}
        )
        async with _feed(store).follow("00000000000000000000000301", tenant_id="T") as f:
            items = [await asyncio.wait_for(anext(f), 5) for _ in range(4)]
            await state.save_run_event(
                "00000000000000000000000301", event_type="run.resumed", data={}
            )
            await state.save_run_event(
                "00000000000000000000000301", event_type="run.paused", data=_PAUSED
            )  # live
            live = [await asyncio.wait_for(anext(f), 5) for _ in range(2)]
        assert "system_prompt" not in items[0].data
        assert "params" not in items[1].data["pending_tool_calls"][0]
        assert items[3].data["pending_tool_calls"][0]["params"] == {"s": 1}
        assert live[1].data["pending_tool_calls"][0]["params"] == {"s": 1}

    async def test_include_payloads_shows_everything(self, state: Any, store: RunStore) -> None:
        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event(
            "00000000000000000000000301", event_type="run.started", data={"system_prompt": "S"}
        )
        await state.finalize_run("00000000000000000000000301", status="success")
        items = await _collect(
            _feed(store).follow("00000000000000000000000301", tenant_id=None, include_payloads=True)
        )
        assert items[0].data == {"system_prompt": "S"}


class TestStreamEvents:
    async def test_omitted_tenant_warns_and_reads_unscoped(
        self, state: Any, store: RunStore
    ) -> None:
        await state.create_run("ra", "a", tenant_id="A")
        await state.save_run_event("ra", event_type="x")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(2):
                gen = store.stream_events("ra", poll_interval_s=0.01)
                assert (await anext(gen)).event_type == "x"
                await gen.aclose()
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1  # deduplicated per location by `warnings`
        assert deprecations[0].filename == __file__
        assert "follow()" in str(deprecations[0].message)

    async def test_explicit_tenant_is_scoped_and_silent(self, state: Any, store: RunStore) -> None:
        await state.create_run("ra", "a", tenant_id="A")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RunNotFoundError):
                await anext(store.stream_events("ra", tenant_id="B"))


class TestFixRound:
    async def test_overflow_never_accepts_6_10_15_after_a_gap(
        self, state: Any, store: RunStore
    ) -> None:
        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event("00000000000000000000000301", event_type="tick")
        feed = _feed(store, buffer=5)
        async with feed.follow("00000000000000000000000301", tenant_id=None) as follow:
            assert (await anext(follow)).sequence_index == 0
            follower = follow._follower
            assert follower is not None
            for index in range(1, 8):
                follower.push(StoredEvent(index, 0, "tick", None, {}, None))
            delivered = []
            for index in (6, 10, 15):
                delivered.append(await anext(follow))
                follower.push(StoredEvent(index, 0, "tick", None, {}, None))
            delivered.extend([item async for item in follow])
        assert _seqs(delivered) == [1, 2, 3, 4, 5]
        assert delivered[-1] == FollowEnd("lagging", 5)

    @pytest.mark.parametrize("live", [False, True])
    async def test_duration_bounds_slow_delivery(
        self, state: Any, store: RunStore, live: bool
    ) -> None:
        await state.create_run("00000000000000000000000301", "a")
        for _ in range(3):
            await state.save_run_event("00000000000000000000000301", event_type="tick")
        async with _feed(store).follow(
            "00000000000000000000000301", tenant_id=None, max_duration_s=0.04
        ) as follow:
            first = await anext(follow)
            if live:
                # Skip catch-up for this variant and replenish the live queue.
                await anext(follow)
                first = await anext(follow)
                assert follow._follower is not None
                follow._follower.push(StoredEvent(3, 0, "tick", None, {}, None))
            await asyncio.sleep(0.06)
            async with asyncio.timeout(2):
                assert await anext(follow) == FollowEnd("max_duration", first.sequence_index)

    async def test_blocked_read_obeys_duration(
        self, state: Any, store: RunStore, monkeypatch: Any
    ) -> None:
        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event("00000000000000000000000301", event_type="tick")

        async def blocked(*a: Any, **kw: Any) -> Any:
            await asyncio.Event().wait()

        monkeypatch.setattr(store, "get_events", blocked)
        async with _feed(store).follow(
            "00000000000000000000000301", tenant_id=None, max_duration_s=0.03
        ) as follow:
            assert await anext(follow) == FollowEnd("max_duration", None)

    @pytest.mark.parametrize("mode", ["break", "exception", "gc", "cancel", "start_only"])
    async def test_abandoned_followers_release_capacity(
        self, state: Any, store: RunStore, mode: str
    ) -> None:
        import gc

        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event("00000000000000000000000301", event_type="tick")
        feed = _feed(store)
        for _ in range(600 if mode == "break" else 3):
            follow = feed.follow("00000000000000000000000301", tenant_id=None)
            if mode == "start_only":
                await follow.start()
            elif mode == "cancel":
                await anext(follow)
                task = asyncio.create_task(anext(follow))
                await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif mode == "gc":
                await anext(follow)
            else:
                try:
                    async for _item in follow:
                        if mode == "exception":
                            raise ValueError("consumer error")
                        break
                except ValueError:
                    pass
            del follow
            if mode == "gc":
                gc.collect()
            async with asyncio.timeout(2):
                while feed.followers or feed.watched_runs:
                    await asyncio.sleep(0.001)
        assert feed.followers == feed.watched_runs == 0
        await feed.aclose()

    async def test_slow_cancellation_stays_tracked(
        self, state: Any, store: RunStore, monkeypatch: Any
    ) -> None:
        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event("00000000000000000000000301", event_type="tick")
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = store.get_events

        async def slow(*a: Any, **kw: Any) -> Any:
            if asyncio.current_task().get_name().startswith("dendrux-run-feed:"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    raise
            return await original(*a, **kw)

        monkeypatch.setattr(store, "get_events", slow)
        feed = _feed(store, max_followers=1, stop_timeout_s=0.001)
        follow = feed.follow("00000000000000000000000301", tenant_id=None)
        await anext(follow)
        await entered.wait()
        poller = feed._feeds["00000000000000000000000301"].task
        await follow.aclose()
        assert cancelled.is_set() and not poller.done()
        assert feed.followers == 0 and feed.watched_runs == 1
        with pytest.raises(FollowCapacityError):
            await feed.follow("00000000000000000000000301", tenant_id=None).start()
        shutdown = asyncio.create_task(feed.aclose())
        await asyncio.sleep(0)
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert feed.watched_runs == 1
        release.set()
        await feed.aclose()
        assert poller.done() and feed.watched_runs == 0

    @pytest.mark.parametrize(
        "transition", ["resume", "resume_event_only", "terminal", "replacement"]
    )
    async def test_active_pause_is_refreshed_before_delivery(
        self, state: Any, store: RunStore, transition: str
    ) -> None:
        await state.create_run("00000000000000000000000301", "a")
        await state.save_run_event(
            "00000000000000000000000301", event_type="run.paused", data=_PAUSED
        )
        await state.pause_run(
            "00000000000000000000000301", status="waiting_client_tool", pause_data={}
        )
        follow = _feed(store).follow("00000000000000000000000301", tenant_id=None)
        await follow.start()
        if transition == "terminal":
            await state.finalize_run("00000000000000000000000301", status="success")
        elif transition == "resume_event_only":
            await state.save_run_event("00000000000000000000000301", event_type="run.resumed")
        elif transition == "resume":
            await state.claim_paused_run(
                "00000000000000000000000301", expected_status="waiting_client_tool"
            )
            await state.save_run_event("00000000000000000000000301", event_type="run.resumed")
        else:
            await state.save_run_event("00000000000000000000000301", event_type="run.resumed")
            await state.save_run_event(
                "00000000000000000000000301", event_type="run.paused", data=_PAUSED
            )
        async with follow:
            old = await anext(follow)
        assert "params" not in old.data["pending_tool_calls"][0]

    async def test_poller_status_projection_excludes_history(
        self, state: Any, store: RunStore, engine: Any
    ) -> None:
        from sqlalchemy import event

        await state.create_run("00000000000000000000000301", "a")
        await state.pause_run(
            "00000000000000000000000301",
            status="waiting_client_tool",
            pause_data={"history": ["x" * 10000]},
        )
        queries = []

        def capture(_c: Any, _cur: Any, stmt: str, *a: Any) -> None:
            queries.append(stmt)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            async with _feed(store, heartbeat_s=0.04).follow(
                "00000000000000000000000301", tenant_id=None
            ) as follow:
                await anext(follow)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert any("agent_runs.status" in q for q in queries)
        assert not any("pause_data" in q for q in queries)


async def test_closed_feed_cannot_start_a_preacquired_follower(state: Any, store: RunStore) -> None:
    await state.create_run("00000000000000000000000301", "a")
    feed = _feed(store)
    follow = feed.follow("00000000000000000000000301", tenant_id=None)
    await follow.start()
    await feed.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await anext(follow)
    assert feed.followers == feed.watched_runs == 0
