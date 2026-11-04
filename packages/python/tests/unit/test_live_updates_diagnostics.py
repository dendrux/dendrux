"""Actionable configuration errors and payload-free bus diagnostics."""

import logging
import ssl

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import _codes
from dendrux.db import _live_updates as live
from dendrux.db import _redis_updates as redis
from dendrux.errors import StoreCapabilityError
from dendrux.runtime.submit import _require_approval_token
from dendrux.store import PostgresStateStore
from tests.unit.test_live_updates_base import Bus


def test_received_counts_only_valid_ids_and_exposes_age(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: clock[0])
    bus = Bus()
    assert bus.stats().received == 0
    assert bus.stats().last_received_age_s is None
    assert bus.stats().last_error_type is None
    bus._deliver('["a","b"]')
    bus._deliver("invalid")
    bus._deliver(bus.GAP)
    clock[0] += 4
    assert bus.stats().received == 2
    assert bus.stats().last_received_age_s == 4
    bus._deliver("[]")
    assert bus.stats().last_received_age_s == 4


async def test_send_failure_health_and_exception_type_are_visible(caplog):
    class CustomBusError(Exception):
        pass

    bus = Bus()
    bus._health(True)

    async def fail():
        raise CustomBusError("RF2_SECRET_PAYLOAD")

    bus.action = fail
    caplog.set_level(logging.INFO)
    bus.enqueue(["id"])
    await bus.flush()
    assert not bus.healthy
    assert bus.stats().last_error_type == "CustomBusError"
    records = [r for r in caplog.records if r.code == "live_updates_send_failed"]
    assert len(records) == 1 and records[0].error_type == "CustomBusError"
    assert "RF2_SECRET_PAYLOAD" not in caplog.text + repr(bus.stats())
    bus.action = None
    bus._next_attempt_at = 0
    bus.enqueue(["next"])
    await bus.flush()
    assert bus.healthy
    await bus.close()


def test_listener_failure_includes_exception_class_and_never_text(caplog):
    bus = Bus()
    bus._degraded(KeyError("RF2_SECRET_PAYLOAD"))
    assert bus.stats().last_error_type == "KeyError"
    record = next(r for r in caplog.records if r.code == "live_updates_degraded")
    assert record.error_type == "KeyError"
    assert "RF2_SECRET_PAYLOAD" not in repr(vars(record))


@pytest.mark.parametrize("limits", [{"max_message_bytes": 10}, {"health_check_s": 0.001}])
def test_hostile_limits_refused_at_construction(limits):
    with pytest.raises(ValueError):
        Bus(**limits)


def test_ten_times_realistic_id_and_probe_frequency_fit():
    bus = Bus(max_run_id_len=260, max_message_bytes=264, health_check_s=1.5)
    bus.enqueue(["x" * 260])
    assert bus.stats().pending == 1


def test_oversize_drop_logs_once_without_payload(caplog):
    bus = Bus(max_run_id_len=10, max_message_bytes=14)
    bus.enqueue(["é" * 10, "é" * 10])
    assert bus.stats().dropped_oversize == 2
    records = [r for r in caplog.records if r.code == "live_updates_oversize"]
    assert len(records) == 1
    assert "é" not in caplog.text


def test_missing_super_init_has_actionable_error():
    class Broken(Bus):
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match=r"super\(\).__init__"):
        Broken().bind(object())


def test_postgres_option_error_names_native_choices():
    engine = create_async_engine("postgresql+asyncpg://u@localhost/db")
    with pytest.raises(ValueError) as caught:
        PostgresStateStore(engine, live_updates="postgres")
    assert "True" in str(caught.value) and "PostgresUpdates()" in str(caught.value)


def test_capability_error_gives_signature_and_narrowing_call():
    error = StoreCapabilityError("CustomStore", "fenced_writes", "save_trace", ("claim_epoch",))
    assert "signature" in str(error)
    assert "StoreCapabilities(" in str(error)
    assert "capabilities =" in str(error)


def test_pause_token_error_tells_client_where_to_get_token():
    with pytest.raises(ValueError) as caught:
        _require_approval_token(True, None)
    assert caught.value.code == "pause_id_required"
    assert caught.value.field == "expected_pause_id"
    assert "get_pending" in str(caught.value) and "pause_id" in str(caught.value)


@pytest.mark.parametrize("options", ["x", ["x"], 1])
def test_redis_options_require_mapping(options):
    with pytest.raises((TypeError, ValueError), match="client_options must be a mapping"):
        redis.RedisUpdates("redis://localhost/5", client_options=options)


def test_redis_bad_port_and_bad_option_have_distinct_guidance():
    with pytest.raises(ValueError) as port:
        redis.RedisUpdates("redis://localhost:RF2_SECRET_PAYLOAD/5")
    with pytest.raises(ValueError) as option:
        redis.RedisUpdates("redis://localhost/5", client_options={"RF2_SECRET_PAYLOAD": "x"})
    assert "port" in str(port.value)
    assert "option" in str(option.value)
    assert str(port.value) != str(option.value)
    assert "RF2_SECRET_PAYLOAD" not in str(port.value) + str(option.value)
    assert port.value.__context__ is None and option.value.__context__ is None


def test_tls_error_kind_is_based_on_exception_type():
    assert live._error_kind(ssl.SSLError("RF2_SECRET_PAYLOAD")) == "tls"
    updates = redis.RedisUpdates("rediss://localhost/5")
    wrapped = OSError("RF2_SECRET_PAYLOAD")
    wrapped.__cause__ = ssl.SSLError("RF2_SECRET_PAYLOAD")
    plain = updates._plain_error(wrapped)
    assert live._error_kind(plain) == "tls"
    assert "RF2_SECRET_PAYLOAD" not in str(plain)


def test_custom_exception_class_is_valid_for_failure_log(caplog):
    _codes.log_code(
        logging.getLogger(__name__),
        _codes.LIVE_UPDATES_SEND_FAILED,
        error_type="CustomBusError",
        error_kind="other",
        count=1,
    )
    assert caplog.records[-1].error_type == "CustomBusError"


async def test_close_hook_failure_is_safe_and_diagnosable(caplog):
    class CloseFailure(Bus):
        async def aclose_sender(self):
            raise LookupError("RF2_SECRET_PAYLOAD")

    bus = CloseFailure()
    await bus.close()
    assert bus._state == "closed" and bus.aborts == 1
    assert bus.stats().last_error_type == "LookupError"
    record = next(r for r in caplog.records if r.code == "live_updates_unavailable")
    assert record.error_type == "LookupError"
    assert "RF2_SECRET_PAYLOAD" not in repr(vars(record))


async def test_small_wire_limit_also_counts_oversize_below_channel_parser_limit():
    updates = redis.RedisUpdates("redis://localhost/5", max_message_bytes=20, max_run_id_len=16)
    subscription = redis._Subscription(updates)
    assert subscription._message(b"x" * 21) is None
    assert updates.stats().dropped_oversize == 1
    await updates.close()
