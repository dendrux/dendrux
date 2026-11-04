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
    bus = Bus(max_message_bytes=100, max_run_id_len=260)
    assert bus._decode(json.dumps(["é" * 49], ensure_ascii=False)) is None
    bus = Bus(max_message_bytes=104, max_run_id_len=260)
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
    bus.enqueue(["last"])
    assert bus.stats().dropped_overflow == 1
    tiny = Bus(max_message_bytes=4)
    tiny.enqueue(["x"])
    assert tiny.stats().pending == 0
    assert tiny.stats().dropped_oversize == 1
    assert "last" not in repr(asdict(bus.stats()))


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
