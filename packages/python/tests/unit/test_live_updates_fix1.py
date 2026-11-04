"""Review regressions for sender health, hub registration, and diagnostics."""

import inspect
import pickle

import pytest

from dendrux.db import _live_updates as live
from dendrux.db._postgres_updates import PostgresUpdates
from dendrux.errors import StoreCapabilityError
from dendrux.runtime.submit import _pause_id_required
from tests.unit.test_live_updates_base import Bus


async def test_idle_sender_recovers_health_at_backoff_without_a_write(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(live, "monotonic", lambda: clock[0])
    bus = Bus(backoff_initial_s=0.5)
    bus._health(True)
    generation = bus.generation
    bus._send_failure(TimeoutError("SECRET"), 1)
    assert not bus.healthy
    clock[0] += 0.5
    assert bus.healthy
    assert bus.stats().last_error_type == "TimeoutError"
    assert bus.generation == generation
    bus._send_failure(TimeoutError("SECRET"), 1)
    assert not bus.healthy
    assert bus.generation == generation
    clock[0] += 1
    assert bus.healthy
    await bus.close()


def test_postgres_members_never_expose_plaintext_dsn():
    bus = PostgresUpdates(url="postgresql://u:SECRET@SECRET-HOST/db")
    assert "SECRET" not in repr(inspect.getmembers(bus))


def test_health_floor_is_one_tenth_second():
    with pytest.raises(ValueError, match="health_check_s"):
        Bus(health_check_s=0.099)
    assert Bus(health_check_s=0.1)
    assert Bus(health_check_s=1.5)  # ten times the default probe frequency


def test_pause_guidance_survives_pickle():
    original = _pause_id_required()
    copied = pickle.loads(pickle.dumps(original))
    assert str(copied) == str(original)
    assert copied.code == "pause_id_required"


@pytest.mark.parametrize(
    "capability",
    [
        "version",
        "deployment",
        "fenced_writes",
        "retry_ids",
        "checkpointing",
        "atomic_completion",
        "recovery",
    ],
)
def test_capability_advice_is_specific_and_safe(capability):
    text = str(StoreCapabilityError("Store", capability, "method", ("keyword",)))
    assert "deployment=self.capabilities.deployment" not in text
    if capability == "version":
        assert "version=1" in text
    elif capability == "deployment":
        assert "single_process" in text and "multi_server" in text
    else:
        assert f"{capability}=False" in text
    if capability == "fenced_writes":
        assert "multi_server" in text and "removes fencing" in text


async def test_public_consumers_get_full_batch_without_reading_run_id():
    bus = Bus()
    batches = []

    class Consumer:
        @property
        def run_id(self):
            raise AssertionError("public attribute must not be inspected")

        def on_hints(self, ids):
            batches.append(ids)

        def on_gap(self):
            pass

    bus.register(Consumer())
    bus._deliver('["other","mine","mine"]')
    assert batches == [("other", "mine", "mine")]
    assert len(bus._consumers) == bus._stream_consumers == 1
    await bus.close()


async def test_private_marker_failure_leaves_registration_untouched():
    bus = Bus()

    class Consumer:
        @property
        def _live_updates_run_waker(self):
            raise ValueError("marker failure")

        def on_hints(self, ids):
            pass

        def on_gap(self):
            pass

    with pytest.raises(ValueError):
        bus.register(Consumer())
    assert not bus._consumers and not bus._general_consumers and not bus._waker_runs
    assert bus._stream_consumers == 0
    await bus.close()


def test_raw_question_mark_password_has_encoding_guidance():
    with pytest.raises(ValueError) as caught:
        PostgresUpdates(url="postgresql://u:a?SECRET=value@host/db")
    assert "percent-encode" in str(caught.value)
    assert "SECRET" not in str(caught.value)


def test_oversize_advice_includes_foreign_publishers():
    from dendrux import _codes

    assert "channel isolation" in _codes.LIVE_UPDATES_OVERSIZE.do


async def test_waker_index_does_not_grow_with_completed_runs():
    bus = Bus()

    class Waker:
        _live_updates_run_waker = True

        def on_hints(self, ids):
            pass

        def on_gap(self):
            pass

    for index in range(100_000):
        waker = Waker()
        waker.run_id = str(index)
        handle = bus.register(waker)
        assert len(bus._wakers_by_run) == 1
        bus.unregister(handle)
        assert not bus._wakers_by_run
    assert not bus._waker_runs and not bus._consumers
    await bus.close()
