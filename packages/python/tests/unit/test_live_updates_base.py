"""The live-update seam is bounded, lossy and independent of store results."""

import asyncio
import json
import logging
import pickle
from dataclasses import asdict

import pytest

from dendrux import _codes, errors
from dendrux.db import _live_updates as live

NEW_CODES = (
    "live_updates_degraded",
    "live_updates_recovered",
    "live_updates_unavailable",
    "live_updates_missed_hint",
    "live_updates_send_failed",
    "live_updates_send_recovered",
    "live_updates_overflow",
    "live_updates_wrong_loop",
    "live_updates_bad_message",
    "live_updates_close_timeout",
    "live_updates_tls_unknown",
    "live_updates_many_stores",
    "follow_read_failed",
    "follow_coordinator_restart",
)


@pytest.mark.parametrize("name", NEW_CODES)
def test_live_codes_registered(name, caplog):
    assert name in _codes.REGISTRY
    code = _codes.REGISTRY[name]
    with caplog.at_level(logging.INFO):
        _codes.log_code(logging.getLogger(__name__), code, run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV")
    assert caplog.records[-1].code == name


def test_live_code_fields_are_validated(caplog):
    assert "live_updates_send_failed" in _codes.REGISTRY
    with caplog.at_level(logging.INFO):
        _codes.log_code(
            logging.getLogger(__name__),
            _codes.LIVE_UPDATES_SEND_FAILED,
            error_kind="timeout",
            count=10,
        )
        _codes.log_code(
            logging.getLogger(__name__),
            _codes.LIVE_UPDATES_MANY_STORES,
            count=17,
            database_key="a" * 64,
        )
    assert caplog.records[0].error_kind == "timeout"
    assert caplog.records[0].kind == "live_updates"
    assert caplog.records[1].database_key == "a" * 64
    with pytest.raises(ValueError):
        _codes.log_code(
            logging.getLogger(__name__), _codes.LIVE_UPDATES_SEND_FAILED, error_kind="VENDOR_SECRET"
        )


def test_error_move_preserves_public_aliases_and_pickle():
    from dendrux.store import FollowCapacityError
    from dendrux.store._follow import FollowCapacityError as OldCapacityError

    assert getattr(errors, "FollowCapacityError", None) is FollowCapacityError is OldCapacityError
    original = FollowCapacityError(100, per_tenant=True)
    copy = pickle.loads(pickle.dumps(original))
    assert type(copy) is FollowCapacityError
    assert (copy.limit, copy.per_tenant, str(copy)) == (100, True, str(original))


def test_unavailable_error_sanitizes_reason(caplog):
    cls = getattr(errors, "LiveUpdatesUnavailableError", None)
    assert cls is not None
    assert issubclass(cls, RuntimeError)
    with caplog.at_level(logging.WARNING):
        error = cls("VENDOR_SECRET")
    assert error.reason == "other"
    assert "VENDOR_SECRET" not in str(error) + repr(vars(error)) + caplog.text
    assert caplog.records[-1].code == "live_updates_unavailable"
    assert pickle.loads(pickle.dumps(error)).reason == "other"


class Bus(live.LiveUpdates):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self.aborts = 0
        self.action = None

    async def send(self, messages):
        self.calls.append(messages)
        if self.action is not None:
            await self.action()

    async def receive(self):
        yield None
        await asyncio.Future()

    def abort_sender(self):
        self.aborts += 1


INT_LIMITS = ("max_pending", "max_batch", "max_message_bytes", "max_run_id_len", "max_inbound")
TIME_LIMITS = (
    "send_timeout_s",
    "flush_deadline_s",
    "close_timeout_s",
    "backoff_initial_s",
    "backoff_max_s",
    "health_check_s",
)


@pytest.mark.parametrize("name", INT_LIMITS + TIME_LIMITS)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "secret"])
def test_refuse_invalid_limits(name, value):
    with pytest.raises(ValueError):
        Bus(**{name: value})


@pytest.mark.parametrize("name", INT_LIMITS)
def test_refuse_fractional_counts(name):
    with pytest.raises(ValueError):
        Bus(**{name: 1.5})


@pytest.mark.parametrize(
    "kwargs", [dict(max_pending=1), dict(send_timeout_s=2), dict(backoff_initial_s=31)]
)
def test_refuse_inconsistent_limits(kwargs):
    with pytest.raises(ValueError):
        Bus(**kwargs)


def test_limits_ten_times_realistic_and_store_binding():
    bus = Bus(
        max_pending=10_000,
        max_batch=500,
        max_message_bytes=7900,
        max_run_id_len=260,
        max_inbound=1000,
        send_timeout_s=0.5,
        flush_deadline_s=1,
        close_timeout_s=3,
        backoff_initial_s=0.5,
        backoff_max_s=30,
        health_check_s=15,
    )
    assert bus.max_pending == 10_000
    owner = object()
    bus.bind(owner)
    bus.bind(owner)
    with pytest.raises(ValueError):
        bus.bind(object())


@pytest.mark.parametrize("seed", range(20))
def test_wire_round_trip_and_truncation(seed):
    import random

    rng = random.Random(seed)
    alphabet = 'abc\\"\n\x00é界😀'
    ids = ["".join(rng.choices(alphabet, k=rng.randint(1, 26))) for _ in range(500)]
    bus = Bus()
    messages = bus._encode(ids)
    assert all(len(m.encode()) <= 7900 for m in messages)
    assert [item for m in messages for item in bus._decode(m)] == ids
    for message in messages:
        for end in (0, 1, len(message) // 2, len(message) - 1):
            assert bus._decode(message[:end]) is None


def test_ulid_batch_is_two_compact_messages():
    bus = Bus()
    ids = [f"{i:026d}" for i in range(500)]
    messages = bus._encode(ids)
    assert len(messages) == 2
    assert all(m == json.dumps(json.loads(m), separators=(",", ":")) for m in messages)
    assert [x for m in messages for x in bus._decode(m)] == ids


@pytest.mark.parametrize(
    "message",
    [
        "{}",
        "null",
        '"id"',
        "[1]",
        "[null]",
        '[""]',
        '[["x"]]',
        '["ok",false]',
        '["x"] trailing',
        "[" * 2000 + "]" * 2000,
        '["' + "x" * 256 + '"]',
        json.dumps(["x"] * 501),
        " " * 7901,
        '["\ud800"]',
    ],
)
def test_decoder_refuses_whole_bad_message(message, caplog):
    bus = Bus()
    assert bus._decode(message) is None
    assert bus._decode(message) is None
    assert bus.stats().bad_messages == 2
    assert [r.code for r in caplog.records] == ["live_updates_bad_message"]
    assert message not in caplog.text


def test_decoder_checks_utf8_byte_count_and_id_length():
    bus = Bus(max_message_bytes=100, max_run_id_len=96)
    assert bus._decode(json.dumps(["é" * 49], ensure_ascii=False)) is None
    bus = Bus(max_message_bytes=104, max_run_id_len=100)
    assert bus._decode(json.dumps(["é" * 50], ensure_ascii=False)) == ("é" * 50,)
    assert Bus(max_run_id_len=260)._decode(json.dumps(["x" * 260])) == ("x" * 260,)
    assert Bus(max_run_id_len=260)._decode(json.dumps(["x" * 261])) is None


@pytest.mark.asyncio
async def test_pending_overflow_dedup_oldest_and_episode(caplog):
    bus = Bus(max_pending=4, max_batch=2)
    bus.enqueue(["a", "b", "c", "d", "a", "e", "f"])
    assert bus.stats().pending == 4
    assert bus.stats().dropped_overflow == 2
    assert len(caplog.records) == 1
    await bus.flush()
    assert [x for call in bus.calls for m in call for x in json.loads(m)] == ["c", "d", "e", "f"]
    bus.enqueue(["g", "h", "i", "j", "k"])
    assert len(caplog.records) == 2


@pytest.mark.asyncio
async def test_pending_scale_invalid_and_unencodable_ids():
    bus = Bus(max_run_id_len=260)
    bus.enqueue([f"{i:026d}" for i in range(10_000)])
    assert bus.stats().pending == 10_000
    bus.enqueue(["x" * 261, "", None, "\ud800"])
    assert bus.stats().dropped_invalid == 4
    bus.enqueue(["SENTINEL_LAST_RUN"])
    assert bus.stats().dropped_overflow == 1
    tiny = Bus(max_message_bytes=5, max_run_id_len=1)
    tiny.enqueue(["é"])
    assert tiny.stats().pending == 0
    assert tiny.stats().dropped_oversize == 1
    assert "SENTINEL_LAST_RUN" not in repr(asdict(bus.stats()))


@pytest.mark.asyncio
async def test_stranded_age_and_dedup_timestamp(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])
    bus = Bus()
    bus.enqueue(["a"])
    now[0] = 110.0
    bus.enqueue(["a", "b"])
    assert bus.stats().oldest_pending_age_s == 10.0
    await bus.flush()
    assert bus.stats().oldest_pending_age_s == 0
    assert bus.stats().sent == 2


@pytest.mark.asyncio
async def test_wrong_loop_rejected_before_pending_mutation(caplog):
    bus = Bus()
    bus.enqueue(["original"])

    async def other_loop():
        bus.enqueue(["foreign"])
        bus.enqueue(["foreign2"])
        await bus.flush()

    await asyncio.to_thread(asyncio.run, other_loop())
    assert bus.stats().pending == 1
    assert bus.stats().dropped_wrong_loop == 2
    assert [r.code for r in caplog.records] == ["live_updates_wrong_loop"]
    await bus.flush()
    assert bus.calls == [['["original"]']]


@pytest.mark.asyncio
async def test_flush_coalesces_without_background_sender():
    bus = Bus()
    started, finish = asyncio.Event(), asyncio.Event()

    async def blocked():
        started.set()
        await finish.wait()

    bus.action = blocked
    bus.enqueue(["a"])
    first = asyncio.create_task(bus.flush())
    await started.wait()
    bus.enqueue(["b"])
    await bus.flush()
    assert len(bus.calls) == 1
    finish.set()
    await first
    assert len(bus.calls) == 2
    assert bus.stats().sent == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("VENDOR_SECRET"),
        OSError("VENDOR_SECRET"),
        RuntimeError("VENDOR_SECRET"),
        asyncio.CancelledError("VENDOR_SECRET"),
    ],
)
async def test_send_failure_abort_backoff_recovery(error, caplog, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])
    bus = Bus(max_pending=10, max_batch=1, backoff_initial_s=0.5, backoff_max_s=1)

    async def fail():
        raise error

    bus.action = fail
    with caplog.at_level(logging.INFO):
        bus.enqueue(["a", "b"])
        await bus.flush()
        assert bus.aborts == 1
        assert bus.stats().dropped_failed == 1
        assert bus.stats().pending == 1
        await bus.flush()
        assert len(bus.calls) == 1
        now[0] += 0.5
        await bus.flush()
        bus.enqueue(["c"])
        now[0] += 0.9
        await bus.flush()
        assert len(bus.calls) == 2
        now[0] += 0.11
        bus.action = None
        await bus.flush()
    assert [r.code for r in caplog.records] == [
        "live_updates_send_failed",
        "live_updates_send_recovered",
    ]
    assert "VENDOR_SECRET" not in caplog.text + repr(bus.stats())
    assert bus.stats().sent == 1


@pytest.mark.asyncio
async def test_caller_cancellation_aborts_before_ownership_release():
    bus = Bus()
    started = asyncio.Event()
    observed = []

    async def blocked():
        started.set()
        await asyncio.Future()

    def abort():
        observed.append(bus._in_flight)

    bus.action = blocked
    bus.abort_sender = abort
    bus.enqueue(["a"])
    task = asyncio.create_task(bus.flush())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observed == [True]
    assert bus._in_flight is False
    assert bus.stats().dropped_failed == 1
    bus.action = None
    bus.enqueue(["b"])
    await bus.flush()
    assert bus.stats().sent == 1


@pytest.mark.asyncio
async def test_absolute_deadline_includes_encoding(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])
    bus = Bus()
    encode = bus._encode

    def slow_encode(ids):
        now[0] += 2
        return encode(ids)

    bus._encode = slow_encode
    bus.enqueue(["a"])
    await bus.flush()
    assert bus.calls == []
    assert bus.stats().pending == 1
    assert bus.stats().oldest_pending_age_s == 2


@pytest.mark.asyncio
async def test_stalled_connect_uses_send_timeout():
    bus = Bus(send_timeout_s=0.01, flush_deadline_s=0.02)
    cancelled = []

    async def connect():
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    bus.action = connect
    bus.enqueue(["a"])
    await asyncio.wait_for(bus.flush(), 1)
    assert cancelled == [True]
    assert bus.aborts == 1
    assert bus.stats().dropped_failed == 1


@pytest.mark.asyncio
async def test_successful_sends_share_absolute_deadline(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])
    bus = Bus(max_batch=1)

    async def slow():
        now[0] += 0.6

    bus.action = slow
    bus.enqueue(["a", "b", "c"])
    await bus.flush()
    assert len(bus.calls) == 2
    assert bus.stats().pending == 1


@pytest.mark.asyncio
async def test_bad_iterable_and_abort_hook_cannot_fail_save():
    bus = Bus()

    def broken_ids():
        yield "a"
        raise RuntimeError("VENDOR_SECRET")

    bus.enqueue(broken_ids())

    async def fail():
        raise OSError("VENDOR_SECRET")

    def abort():
        raise RuntimeError("VENDOR_SECRET")

    bus.action = fail
    bus.abort_sender = abort
    await bus.flush()
    assert bus._in_flight is False
    assert bus.stats().dropped_failed == 1


def test_subclass_bind_cannot_bypass_store_ownership():
    class Custom(Bus):
        def bind(self, store):
            self.learned = store

    bus = Custom()
    first = object()
    bus.bind(first)
    assert bus.learned is first
    with pytest.raises(ValueError):
        bus.bind(object())
    assert bus.learned is first


def test_empty_pending_age_is_exactly_zero():
    assert Bus().stats().oldest_pending_age_s == 0


class Consumer:
    def __init__(self):
        self.hints = []
        self.gaps = 0

    def on_hints(self, ids):
        self.hints.append(ids)

    def on_gap(self):
        self.gaps += 1


class ListeningBus(Bus):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.incoming = asyncio.Queue()
        self.starts = 0
        self.stops = 0
        self.listen_aborts = 0
        self.sender_closes = 0
        self.ready = True
        self.stop_gate = None
        self.close_gate = None

    async def receive(self):
        self.starts += 1
        try:
            if self.ready:
                yield None
            while True:
                message = await self.incoming.get()
                if isinstance(message, BaseException):
                    raise message
                if message == "END":
                    return
                yield message
        finally:
            if self.stop_gate is not None:
                await self.stop_gate.wait()
            self.stops += 1

    def abort_listener(self):
        self.listen_aborts += 1
        if self.stop_gate is not None:
            self.stop_gate.set()

    async def aclose_sender(self):
        self.sender_closes += 1
        if self.close_gate is not None:
            await self.close_gate.wait()


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def test_public_export_and_abstract_hooks():
    from dendrux import store

    assert getattr(store, "LiveUpdates", None) is live.LiveUpdates
    assert "LiveUpdates" in store.__all__
    with pytest.raises(TypeError):
        live.LiveUpdates()


@pytest.mark.asyncio
async def test_inbound_bound_drops_oldest_and_preserves_gap():
    bus = Bus()
    buffer = bus._new_inbound_buffer()
    for i in range(1000):
        buffer.put(str(i))
    assert buffer.qsize() == 1000
    assert await buffer.get() == "0"
    buffer.put("1000")
    buffer.put("1001")
    assert buffer.qsize() == 1000
    assert await buffer.get() == bus.GAP
    assert await buffer.get() == "2"
    assert bus.stats().dropped_inbound == 1
    for _ in range(999):
        await buffer.get()
    waiter = asyncio.create_task(buffer.get())
    await asyncio.sleep(0)
    assert not waiter.done()
    buffer.put("last")
    assert await waiter == "last"


@pytest.mark.asyncio
async def test_hub_capacity_callbacks_and_isolation(caplog):
    bus = Bus()
    consumer = Consumer()
    tokens = [bus._register(consumer) for _ in range(10_000)]
    with pytest.raises(errors.FollowCapacityError) as exc:
        bus._register(consumer)
    assert exc.value.limit == 10_000
    coordinator = bus._register(Consumer(), stream_events=False)
    bus._unregister(tokens.pop())
    bus._unregister(coordinator)
    token = bus._register(consumer)
    bus._deliver('["a","b"]')
    assert len(consumer.hints) == 10_000
    assert consumer.hints[0] == ("a", "b")
    bus._deliver(bus.GAP)
    assert consumer.gaps == 10_000
    bus._unregister(token)
    bus._unregister(token)
    assert caplog.records[-1].code == "follow_capacity"


@pytest.mark.asyncio
async def test_faulty_consumer_does_not_block_other_consumers():
    bus = Bus()

    class Broken(Consumer):
        def on_hints(self, ids):
            raise RuntimeError("VENDOR_SECRET")

        def on_gap(self):
            raise asyncio.CancelledError("VENDOR_SECRET")

    bus._register(Broken())
    good = Consumer()
    bus._register(good)
    bus._deliver('["a"]')
    bus._deliver("invalid")
    bus._deliver(bus.GAP)
    assert good.hints == [("a",)]
    assert good.gaps == 1
    assert bus.stats().gaps == 1


@pytest.mark.asyncio
async def test_two_leases_health_generation_and_idempotent_release(caplog):
    bus = ListeningBus()
    consumer = Consumer()
    bus._register(consumer)
    assert not bus.healthy
    one = await bus._acquire_listener()
    two = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    assert bus.starts == 1
    assert bus._generation == 1
    await bus.incoming.put('["a"]')
    await until(lambda: consumer.hints)
    await bus._release_listener(one)
    await bus._release_listener(one)
    assert bus.healthy and bus.stops == 0
    await bus._release_listener(two)
    assert not bus.healthy and bus.stops == 1
    assert bus.aborts == 0
    assert bus._generation == consumer.gaps == 2
    assert bus.stats().gaps == 2
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [OSError("VENDOR_SECRET"), asyncio.CancelledError("VENDOR_SECRET"), "END"]
)
async def test_listener_reconnect_closes_old_connection_and_surfaces_gap(failure, caplog):
    bus = ListeningBus(backoff_initial_s=0.001, backoff_max_s=0.002)
    consumer = Consumer()
    bus._register(consumer)
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    await bus.incoming.put(failure)
    await until(lambda: bus.starts == 2 and bus.healthy)
    assert bus.stops == 1
    assert bus._generation == 3
    await bus.incoming.put(bus.GAP)
    await until(lambda: bus._generation == 4)
    assert bus.healthy
    assert consumer.gaps == 4
    assert "VENDOR_SECRET" not in caplog.text + repr(bus.stats())
    await bus._release_listener(lease)
    await bus.close()


@pytest.mark.asyncio
async def test_listener_setup_deadline_cleans_half_open_connection(caplog):
    bus = ListeningBus(send_timeout_s=0.005, backoff_initial_s=0.01)
    bus.ready = False
    lease = await bus._acquire_listener()
    await until(lambda: bus.stops >= 1)
    assert not bus.healthy
    assert any(r.code == "live_updates_degraded" for r in caplog.records)
    await bus._release_listener(lease)
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ['["premature"]', live.LiveUpdates.GAP])
async def test_listener_requires_subscription_ack(first):
    bus = ListeningBus(backoff_initial_s=0.001)
    bus.ready = False
    consumer = Consumer()
    bus._register(consumer)
    await bus.incoming.put(first)
    lease = await bus._acquire_listener()
    await until(lambda: bus.stops >= 1)
    assert not bus.healthy
    assert consumer.hints == []
    await bus._release_listener(lease)
    await bus.close()


@pytest.mark.asyncio
async def test_wrong_loop_lease_raises_before_refcount_change():
    bus = ListeningBus()
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)

    async def other():
        with pytest.raises(RuntimeError, match="event loop"):
            await bus._acquire_listener()
        with pytest.raises(RuntimeError, match="event loop"):
            await bus._release_listener(lease)

    await asyncio.to_thread(asyncio.run, other())
    assert bus.starts == 1
    await bus._release_listener(lease)
    assert bus.stops == 1
    await bus.close()


@pytest.mark.asyncio
async def test_listener_start_stop_serialized():
    bus = ListeningBus()
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    bus.stop_gate = asyncio.Event()
    releasing = asyncio.create_task(bus._release_listener(lease))
    await asyncio.sleep(0)
    acquiring = asyncio.create_task(bus._acquire_listener())
    await asyncio.sleep(0)
    assert bus.starts == 1 and not acquiring.done()
    bus.stop_gate.set()
    await releasing
    next_lease = await acquiring
    await until(lambda: bus.starts == 2)
    await bus._release_listener(next_lease)
    await bus.close()


@pytest.mark.asyncio
async def test_last_lease_forces_stalled_listener_stop(caplog):
    bus = ListeningBus(close_timeout_s=0.01)
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    bus.stop_gate = asyncio.Event()
    await asyncio.wait_for(bus._release_listener(lease), 1)
    assert bus.listen_aborts == 1
    assert bus.aborts == 0
    assert not bus.healthy
    assert any(r.code == "live_updates_close_timeout" for r in caplog.records)
    await until(lambda: bus.stops == 1)
    await bus.close()


@pytest.mark.asyncio
async def test_close_waits_inflight_final_flush_and_closes_once():
    bus = ListeningBus()
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    started, finish = asyncio.Event(), asyncio.Event()

    async def blocked():
        started.set()
        await finish.wait()

    bus.action = blocked
    bus.enqueue(["a"])
    flushing = asyncio.create_task(bus.flush())
    await started.wait()
    bus.enqueue(["b"])
    closing = asyncio.create_task(bus.close())
    await asyncio.sleep(0)
    concurrent = asyncio.create_task(bus.close())
    bus.enqueue(["late"])
    assert bus.stats().dropped_closed == 1
    done, _ = await asyncio.wait({closing}, timeout=0.02)
    assert not done
    assert bus.sender_closes == bus.stops == 0
    finish.set()
    await asyncio.gather(flushing, closing, concurrent)
    assert bus.sender_closes == 1
    assert bus.stats().sent == 2
    assert bus.stops == 1
    await bus._release_listener(lease)
    await bus.close()
    assert bus.sender_closes == 1
    with pytest.raises(errors.LiveUpdatesUnavailableError):
        await bus._acquire_listener()


@pytest.mark.asyncio
async def test_close_flushes_stranded_hints_and_drops_backoff_remainder():
    bus = ListeningBus()
    bus.enqueue(["a"])
    await bus.close()
    assert bus.stats().sent == 1
    bus.enqueue(["b"])
    await bus.flush()
    assert bus.stats().dropped_closed == 1
    assert len(bus.calls) == 1
    failed = ListeningBus(max_batch=1)

    async def fail():
        raise OSError("VENDOR_SECRET")

    failed.action = fail
    failed.enqueue(["a", "b"])
    await failed.flush()
    await failed.close()
    assert len(failed.calls) == 1
    assert failed.stats().pending == 0
    assert failed.stats().dropped_closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["inflight", "listener", "sender"])
async def test_close_total_timeout_aborts_both_resources(phase, caplog):
    bus = ListeningBus(close_timeout_s=0.01)
    await bus._acquire_listener()
    await until(lambda: bus.healthy)
    flush_task = None
    if phase == "inflight":
        gate = asyncio.Event()
        bus.action = gate.wait
        bus.enqueue(["a"])
        flush_task = asyncio.create_task(bus.flush())
        await until(lambda: bus._in_flight)
    elif phase == "listener":
        bus.stop_gate = asyncio.Event()
    else:
        bus.close_gate = asyncio.Event()
    await asyncio.wait_for(bus.close(), 1)
    assert bus.aborts >= 1 and bus.listen_aborts >= 1
    assert bus._state == "closed"
    assert not bus.healthy
    assert any(r.code == "live_updates_close_timeout" for r in caplog.records)
    if flush_task:
        gate.set()
        await flush_task
    await until(lambda: bus.stops == 1)
    await bus.close()


@pytest.mark.asyncio
async def test_cancelled_closer_leaves_closing_and_can_retry():
    bus = ListeningBus()
    await bus._acquire_listener()
    await until(lambda: bus.healthy)
    bus.close_gate = asyncio.Event()
    closing = asyncio.create_task(bus.close())
    await until(lambda: bus.sender_closes == 1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert bus.aborts == 1 and bus.listen_aborts == 1
    assert bus._state == "closing"
    bus.enqueue(["late"])
    assert bus.stats().dropped_closed == 1
    bus.close_gate.set()
    await bus.close()
    assert bus._state == "closed"
    assert bus.sender_closes == 2


@pytest.mark.asyncio
async def test_listener_cancellation_cleanup_error_never_reconnects():
    class CleanupFailure(ListeningBus):
        async def receive(self):
            self.starts += 1
            try:
                yield None
                await asyncio.Future()
            finally:
                self.stops += 1
                raise RuntimeError("VENDOR_SECRET")

    bus = CleanupFailure(close_timeout_s=0.02, backoff_initial_s=0.001)
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    await bus._release_listener(lease)
    try:
        assert bus.starts == 1
        assert bus._listener_task is None
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_acquire_racing_close_cannot_restart_after_waiting_for_stop():
    bus = ListeningBus(close_timeout_s=0.02)
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    bus.stop_gate = asyncio.Event()
    # Simulate a transport whose synchronous abort cannot unblock its cleanup.
    bus.abort_listener = lambda: None
    await bus._release_listener(lease)
    acquiring = asyncio.create_task(bus._acquire_listener())
    await asyncio.sleep(0)
    closing = asyncio.create_task(bus.close())
    await asyncio.sleep(0)
    bus.stop_gate.set()
    try:
        with pytest.raises(errors.LiveUpdatesUnavailableError):
            await acquiring
    finally:
        await closing


@pytest.mark.asyncio
async def test_reconnect_health_falls_before_slow_iterator_cleanup():
    cleanup, fail, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Iterator:
        acknowledged = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.acknowledged:
                self.acknowledged = True
                return None
            await fail.wait()
            raise OSError("VENDOR_SECRET")

        async def aclose(self):
            closing.set()
            await cleanup.wait()

    bus = ListeningBus(backoff_initial_s=0.001)
    bus.receive = lambda: Iterator()
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    fail.set()
    try:
        await until(closing.is_set)
        assert not bus.healthy
    finally:
        cleanup.set()
        await bus._release_listener(lease)
        await bus.close()


def test_capacity_error_unpickles_legacy_helper_path():
    # Protocol 0 fixture emitted before the helper moved; no third-party bytes.
    data = b"cdendrux.store._follow\n_capacity_error\np0\n(I5\nI01\ntp1\nRp2\n."
    restored = pickle.loads(data)
    assert isinstance(restored, errors.FollowCapacityError)
    assert restored.limit == 5 and restored.per_tenant


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_sweep_pages_share_one_announcement_deadline():
    bus = Bus(send_timeout_s=0.2, flush_deadline_s=0.2)
    cancelled = []
    allowance = 0.005

    async def send_page():
        try:
            await asyncio.sleep(0.15)
        except asyncio.CancelledError:
            cancelled.append(True)
            await asyncio.sleep(allowance)
            raise

    bus.action = send_page
    start = live.monotonic()
    deadline = start + bus.flush_deadline_s
    async with asyncio.timeout(1):
        for page in range(10):
            if live.monotonic() >= deadline:
                break
            bus.enqueue([str(page)])
            await bus._flush_before(deadline)
    assert cancelled == [True]
    assert len(bus.calls) == 2
    assert live.monotonic() - start < bus.flush_deadline_s + allowance + 0.03
    bus.enqueue(["stranded"])
    await bus._flush_before(deadline)
    assert len(bus.calls) == 2
    assert bus.stats().pending == 1


@pytest.mark.asyncio
async def test_flush_before_expired_budget_keeps_pending():
    bus = Bus()
    bus.enqueue(["a"])
    await bus._flush_before(live.monotonic() - 1)
    assert bus.calls == []
    assert bus.stats().pending == 1


@pytest.mark.asyncio
async def test_flush_before_keeps_loop_and_closed_guards():
    bus = Bus()
    bus.enqueue(["a"])

    async def foreign():
        await bus._flush_before(live.monotonic() + 10)

    await asyncio.to_thread(asyncio.run, foreign())
    assert bus.calls == []
    assert bus.stats().pending == 1
    await bus.close()
    calls = len(bus.calls)
    await bus._flush_before(live.monotonic() + 10)
    assert len(bus.calls) == calls


@pytest.mark.asyncio
async def test_send_backoff_resets_after_recovery(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])
    bus = Bus(backoff_initial_s=0.5, backoff_max_s=4)

    async def fail():
        raise OSError("VENDOR_SECRET")

    bus.action = fail
    for i, window in enumerate((0.5, 1, 2)):
        bus.enqueue([str(i)])
        await bus.flush()
        assert bus._next_attempt_at == now[0] + window
        now[0] += window
    bus.action = None
    bus.enqueue(["recovered"])
    await bus.flush()
    bus.action = fail
    bus.enqueue(["failed-again"])
    await bus.flush()
    assert bus._next_attempt_at == now[0] + 0.5
    bus.enqueue(["retry"])
    await bus.flush()
    assert len(bus.calls) == 5
    now[0] += 0.5
    bus.action = None
    await bus.flush()
    assert len(bus.calls) == 6


@pytest.mark.asyncio
async def test_acquire_refuses_overlap_with_listener_that_will_not_stop():
    bus = ListeningBus(close_timeout_s=0.01)
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    bus.stop_gate = asyncio.Event()
    bus.abort_listener = lambda: None
    old_task = bus._listener_task
    try:
        await bus._release_listener(lease)
        with pytest.raises(errors.LiveUpdatesUnavailableError) as caught:
            async with asyncio.timeout(1):
                await bus._acquire_listener()
        assert caught.value.reason == "timeout"
        assert bus._listener_task is old_task
        assert not old_task.done()
        assert bus.starts == 1
        assert not bus._leases
    finally:
        bus.stop_gate.set()
        await until(old_task.done)
        await bus.close()


@pytest.mark.parametrize(
    "wire",
    [
        "GAP",
        "\x00dendrux-live-updates-gap",
        '"GAP"',
        '["GAP"]',
        '["\\u0000dendrux-live-updates-gap"]',
    ],
)
def test_published_strings_cannot_forge_a_gap(wire):
    bus = Bus()
    consumer = Consumer()
    bus._register(consumer)
    bus._deliver(wire)
    assert bus._generation == consumer.gaps == 0
    if wire.startswith("["):
        assert consumer.hints == [tuple(json.loads(wire))]
    else:
        assert consumer.hints == []
        assert bus.stats().bad_messages == 1


def test_gap_marker_is_local_identity_not_wire_text():
    bus = Bus()
    assert not isinstance(bus.GAP, str)
    consumer = Consumer()
    bus._register(consumer)
    bus._deliver(bus.GAP)
    assert consumer.gaps == bus._generation == 1


@pytest.mark.asyncio
async def test_flapping_subscriptions_grow_backoff_until_stable(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])
    windows = []

    def jitter(low, high):
        windows.append((low, high))
        return 0

    monkeypatch.setattr(live.random, "uniform", jitter)

    class FlappingBus(ListeningBus):
        async def receive(self):
            self.starts += 1
            yield None
            # Even ordinary traffic on a short-lived connection must not reset backoff.
            yield '["a"]'
            if self.starts == 5:
                now[0] += self.health_check_s
            if self.starts >= 7:
                await asyncio.Future()
            raise OSError("VENDOR_SECRET")

    bus = FlappingBus(backoff_initial_s=0.5, backoff_max_s=2, health_check_s=15)
    lease = await bus._acquire_listener()
    try:
        await until(lambda: bus.starts >= 7)
        assert windows[:6] == [(0, 0.5), (0, 1), (0, 2), (0, 2), (0, 0.5), (0, 1)]
    finally:
        await bus._release_listener(lease)
        await bus.close()


@pytest.mark.asyncio
async def test_foreign_loop_close_refuses_without_mutation_and_logs_safely(caplog):
    bus = ListeningBus()
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    bus.enqueue(["PRIVATE_RUN_SENTINEL"])
    snapshot = (
        bus._state,
        tuple(bus._pending.items()),
        set(bus._leases),
        bus._listener_task,
        bus._loop,
        bus.stats(),
    )

    async def foreign():
        with pytest.raises(RuntimeError, match="event loop"):
            await bus.close()

    try:
        await asyncio.to_thread(asyncio.run, foreign())
        await asyncio.to_thread(asyncio.run, foreign())
        assert (
            bus._state,
            tuple(bus._pending.items()),
            set(bus._leases),
            bus._listener_task,
            bus._loop,
        ) == snapshot[:5]
        assert bus.healthy
        assert bus.aborts == bus.listen_aborts == bus.sender_closes == bus.stops == 0
        assert bus.stats().pending == snapshot[5].pending
        assert [r.code for r in caplog.records] == ["live_updates_wrong_loop"]
        assert "PRIVATE_RUN_SENTINEL" not in caplog.text
    finally:
        await bus._release_listener(lease)
        await bus.close()


@pytest.mark.parametrize("failure", [ValueError("rejected"), asyncio.CancelledError()])
def test_failed_bind_hook_rolls_back_ownership(failure):
    class RejectOnce(Bus):
        def bind(self, store):
            super().bind(store)
            if self.reject:
                raise failure

    bus = RejectOnce()
    bus.reject = True
    with pytest.raises(type(failure)):
        bus.bind(object())
    assert bus._store is None and not bus._bound
    bus.reject = False
    store = object()
    bus.bind(store)
    assert bus._bound and bus._store is store
    bus.reject = True
    with pytest.raises(type(failure)):
        bus.bind(store)
    assert bus._bound and bus._store is store
    with pytest.raises(ValueError):
        bus.bind(object())


def test_enqueue_refuses_bare_string_before_queue_or_loop_mutation():
    bus = Bus()
    with pytest.raises(TypeError):
        bus.enqueue("R1")
    assert bus.stats().pending == 0
    assert bus._loop is None


def test_unavailable_error_pickle_does_not_emit_a_second_log(caplog):
    error = errors.LiveUpdatesUnavailableError("timeout")
    assert len(caplog.records) == 1
    serialized = pickle.dumps(error)
    caplog.clear()
    restored = pickle.loads(serialized)
    assert type(restored) is type(error)
    assert restored.reason == error.reason
    assert restored.args == error.args
    assert caplog.records == []


@pytest.mark.asyncio
async def test_overflow_diagnostic_reports_dropped_count_for_enqueue(caplog):
    bus = Bus(max_pending=2, max_batch=2)
    bus.enqueue(["a", "b", "c", "d", "e"])
    logs = [r for r in caplog.records if r.code == "live_updates_overflow"]
    assert len(logs) == 1 and logs[0].count == 3
    assert bus.stats().dropped_overflow == 3
    bus.enqueue(["f", "g"])
    assert len(caplog.records) == 1
    await bus.flush()
    caplog.clear()

    def interrupted():
        yield from ["h", "i", "j", "k"]
        raise RuntimeError("VENDOR_SECRET")

    bus.enqueue(interrupted())
    assert len(caplog.records) == 1
    assert caplog.records[0].count == 2
    assert "VENDOR_SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_close_timeout_is_logged_once_across_nested_shutdown_steps(monkeypatch, caplog):
    bus = ListeningBus(close_timeout_s=1)
    await bus._acquire_listener()
    await until(lambda: bus.healthy)
    bus.stop_gate = asyncio.Event()
    now = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])

    async def expired_wait(tasks, *, timeout):
        now[0] += timeout + 0.001
        return set(), tasks

    monkeypatch.setattr(live.asyncio, "wait", expired_wait)
    await bus.close()
    await until(lambda: bus.stops == 1)
    assert bus._state == "closed"
    logs = [r for r in caplog.records if r.code == "live_updates_close_timeout"]
    assert len(logs) == 1
    await bus.close()
    assert len([r for r in caplog.records if r.code == "live_updates_close_timeout"]) == 1


@pytest.mark.asyncio
async def test_close_timeout_suppresses_late_listener_stop_diagnostic(monkeypatch, caplog):
    bus = ListeningBus(close_timeout_s=0.01)
    lease = await bus._acquire_listener()
    await until(lambda: bus.healthy)
    bus.stop_gate = asyncio.Event()
    bus.abort_listener = lambda: None
    waiting, finish_wait = asyncio.Event(), asyncio.Event()
    listener = bus._listener_task

    async def delayed_wait(tasks, *, timeout):
        waiting.set()
        await finish_wait.wait()
        return set(), tasks

    monkeypatch.setattr(live.asyncio, "wait", delayed_wait)
    releasing = asyncio.create_task(bus._release_listener(lease))
    try:
        await until(waiting.is_set)
        await bus.close()
        assert bus._state == "closed"
        finish_wait.set()
        await releasing
        assert len([r for r in caplog.records if r.code == "live_updates_close_timeout"]) == 1
    finally:
        finish_wait.set()
        bus.stop_gate.set()
        await releasing
        await until(listener.done)


@pytest.mark.asyncio
async def test_foreign_enqueue_cannot_relog_owner_overflow_after_drain(caplog):
    import threading

    bus = Bus(max_pending=2, max_batch=2)
    bus.enqueue(["seed"])
    started, resume = threading.Event(), threading.Event()

    def foreign_ids():
        yield "foreign1"
        started.set()
        assert resume.wait(timeout=2)
        yield "foreign2"

    async def foreign():
        bus.enqueue(foreign_ids())

    task = asyncio.create_task(asyncio.to_thread(asyncio.run, foreign()))
    try:
        await until(started.is_set)
        bus.enqueue(["a", "b", "c"])
        await bus.flush()
    finally:
        resume.set()
        await task
    assert bus.stats().pending == 0
    assert bus.stats().dropped_wrong_loop == 2
    assert bus.stats().dropped_overflow == 2
    assert not bus._overflow_episode
    assert len([r for r in caplog.records if r.code == "live_updates_overflow"]) == 1
