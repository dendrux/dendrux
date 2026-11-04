"""Fix round 1 for RedisUpdates: URL options, close ownership, bounded input, one detection budget.

Tests named ``real_*`` use the installed redis-py (no server, no network): its URL parser, pool
and protocol parser are the things under test. The rest use the scripted stand-in.
"""

from __future__ import annotations

import asyncio
import gc
import weakref
from time import monotonic
from typing import Any

import pytest

from dendrux.db import _redis_updates as ru
from tests.unit.test_redis_updates_fakes import FakeWorld

HOST = "redis.sentinel-host.example"
FAST: dict[str, Any] = dict(
    send_timeout_s=0.05,
    flush_deadline_s=0.2,
    close_timeout_s=1.0,
    backoff_initial_s=0.01,
    backoff_max_s=0.02,
    health_check_s=0.05,
)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    fake = FakeWorld()
    monkeypatch.setattr(ru, "_load_api", fake.api)
    return fake


def make(url: str = f"redis://{HOST}:6380/0", **overrides: Any) -> ru.RedisUpdates:
    return ru.RedisUpdates(url, **{**FAST, **overrides})


# -- R1: URL query options may not change the fixed transport settings -------------------------

PROTECTED_QUERIES = [
    "socket_timeout=900",
    "socket_connect_timeout=900",
    "max_connections=100",
    "health_check_interval=1",
    "retry_on_timeout=true",
    "retry_on_error=1",
    "retry=1",
    "protocol=3",
    "parser_class=x",
    "connection_class=x",
    "single_connection_client=true",
    "socket%5Ftimeout=900",  # percent-encoded key: redis-py decodes it, so we must too
    "db=1&max_connections=5",
]


@pytest.mark.parametrize("query", PROTECTED_QUERIES)
def test_real_a_url_query_may_not_change_a_protected_setting(query: str) -> None:
    url = f"redis://:s3cret-sentinel@{HOST}:6380/0?{query}"
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates(url, **FAST)
    assert "sentinel" not in str(caught.value) and "6380" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_real_harmless_url_query_options_still_work_and_the_fixed_ones_win() -> None:
    updates = ru.RedisUpdates(f"redis://{HOST}/0?client_name=dx", **FAST)
    pool = updates._new_pool(0.2)
    connection = pool.make_connection()
    assert pool.max_connections == 1
    assert (connection.socket_timeout, connection.socket_connect_timeout) == (0.2, 0.05)
    assert connection.health_check_interval == 0


# -- R2: closing the sender never loses the handle that can abort its socket -------------------


async def eventually(predicate: Any) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


def stall_cleanup(world: FakeWorld) -> tuple[asyncio.Event, asyncio.Event]:
    """Make the sender's client and pool cleanup hang; returns (entered, never_released)."""
    entered, gate = asyncio.Event(), asyncio.Event()

    async def hang(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await gate.wait()

    world.pools[0].aclose = hang  # type: ignore[method-assign]
    world.clients[0].aclose = hang  # type: ignore[method-assign]
    return entered, gate


async def test_a_stalled_close_deadline_still_aborts_the_sender_socket(world: FakeWorld) -> None:
    updates = make(close_timeout_s=0.2)
    await updates.send(['["a"]'])
    stall_cleanup(world)
    async with asyncio.timeout(5):
        await updates.close()  # the total close deadline passes while cleanup hangs
    assert world.connections[0].aborted


async def test_a_cancelled_close_aborts_the_sender_socket_and_a_retry_finishes(
    world: FakeWorld,
) -> None:
    updates = make(close_timeout_s=30.0)
    await updates.send(['["a"]'])
    entered, _ = stall_cleanup(world)
    task = asyncio.ensure_future(updates.close())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world.connections[0].aborted  # before any retry
    async with asyncio.timeout(5):
        await updates.close()  # a later close finishes the job
    assert updates._state == "closed"


async def test_a_cancelled_aclose_sender_aborts_the_socket_itself(world: FakeWorld) -> None:
    updates = make()
    await updates.send(['["a"]'])
    entered, _ = stall_cleanup(world)
    task = asyncio.ensure_future(updates.aclose_sender())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world.connections[0].aborted


async def test_real_pool_cleanup_that_never_returns_cannot_strand_a_socket() -> None:
    from types import SimpleNamespace

    updates = ru.RedisUpdates(
        f"redis://{HOST}/0", send_timeout_s=0.01, flush_deadline_s=0.05, close_timeout_s=0.2
    )
    pool = updates._sender().connection_pool  # redis-py's own pool, no connection made
    aborted: list[bool] = []
    transport = SimpleNamespace(abort=lambda: aborted.append(True))
    pool._available_connections.append(
        SimpleNamespace(_writer=SimpleNamespace(transport=transport))
    )

    async def never() -> None:
        await asyncio.Event().wait()

    pool.aclose = never  # redis-py 8 awaits maintenance work here before it disconnects
    async with asyncio.timeout(5):
        await updates.close()
    assert aborted == [True]
    pool._available_connections.clear()


# -- R3: an incoming message is bounded before it is read into memory --------------------------


def listen_parser(updates: ru.RedisUpdates) -> tuple[Any, Any]:
    """The protocol parser redis-py would use on the listen connection, over an in-memory stream."""
    subscription = ru._Subscription(updates)
    connection = subscription.pool.make_connection()
    connection._reader = asyncio.StreamReader()
    connection._parser.on_connect(connection)
    return connection._parser, connection._reader


def message(size: int) -> bytes:
    return (
        b"*3\r\n$7\r\nmessage\r\n$1\r\nc\r\n$"
        + str(size).encode()
        + b"\r\n"
        + b"x" * size
        + b"\r\n"
    )


async def test_real_a_message_at_the_bound_is_read_and_one_byte_more_is_discarded() -> None:
    updates = ru.RedisUpdates(f"redis://{HOST}/0", **FAST)  # max_message_bytes 7,900
    parser, reader = listen_parser(updates)
    reader.feed_data(message(7_900))
    assert len((await parser.read_response())[2]) == 7_900
    reader.feed_data(message(7_901))
    assert (await parser.read_response())[2] is ru._DISCARDED
    reader.feed_data(message(1_000))
    assert len((await parser.read_response())[2]) == 1_000


async def test_real_a_realistic_message_ten_times_over_is_read() -> None:
    parser, reader = listen_parser(ru.RedisUpdates(f"redis://{HOST}/0", **FAST))
    reader.feed_data(message(1_000))  # a normal hint is about 100 bytes
    assert len((await parser.read_response())[2]) == 1_000


async def test_real_a_huge_declared_length_cannot_wait_beyond_the_read_deadline() -> None:
    parser, reader = listen_parser(ru.RedisUpdates(f"redis://{HOST}/0", **FAST))
    reader.feed_data(b"*3\r\n$7\r\nmessage\r\n$1\r\nc\r\n$1000000000\r\n")  # and no body at all
    with pytest.raises(ConnectionError):
        async with asyncio.timeout(0.02):
            await parser.read_response()
    assert sum(map(len, parser._chunks)) + len(parser._buffer) < 1_000


async def test_real_the_bound_follows_max_message_bytes() -> None:
    parser, reader = listen_parser(ru.RedisUpdates(f"redis://{HOST}/0", max_message_bytes=3_000))
    reader.feed_data(message(3_000))
    assert len((await parser.read_response())[2]) == 3_000
    reader.feed_data(message(3_001))
    assert (await parser.read_response())[2] is ru._DISCARDED


def test_real_a_redis_client_without_a_boundable_parser_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "redis._parsers", None)
    with pytest.raises(ImportError, match=r"dendrux\[redis\]"):
        ru._load_api()


# -- R4: one detection budget, no reconnect inside redis-py ------------------------------------


async def test_neither_pool_lets_redis_py_retry_or_reconnect_by_itself(world: FakeWorld) -> None:
    updates = make()
    await updates.send(['["a"]'])
    iterator = updates.receive()
    await anext(iterator)
    sender_pool, listener_pool = world.pools
    for pool in (sender_pool, listener_pool):
        retry = pool.kwargs["retry"]
        assert retry.retries == 0
        assert retry.supported_errors == ()  # nothing is retried, so nothing is reconnected
    await iterator.aclose()


async def test_the_pong_wait_is_clamped_to_the_health_check_interval(world: FakeWorld) -> None:
    world.pong_mode = "silent"
    updates = make(health_check_s=0.03, send_timeout_s=0.5, flush_deadline_s=1.0)
    iterator = updates.receive()
    await anext(iterator)
    waiting = asyncio.ensure_future(anext(iterator))
    await eventually(lambda: bool(world.pubsubs[0].pings))
    subscription = updates._wire.subscription
    assert subscription is not None
    # 4 x send_timeout_s would be 2 s; detection is idle (30 ms) + at most 30 ms of probing
    assert subscription._pong_by - monotonic() <= updates.health_check_s
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(waiting, 5)
    assert world.pubsubs[0].closed


async def test_a_read_that_ignores_its_timeout_is_cut_off_by_the_enclosing_deadline(
    world: FakeWorld,
) -> None:
    updates = make(health_check_s=0.03)
    iterator = updates.receive()
    await anext(iterator)
    world.hang_reads = True  # e.g. redis-py busy reconnecting inside its error handler
    with pytest.raises(TimeoutError) as caught:
        await asyncio.wait_for(anext(iterator), 5)
    assert "Redis live updates" in str(caught.value)  # ours, not the test's tripwire
    assert world.pubsubs[0].closed and world.pools[0].closed


async def test_a_ping_that_cannot_be_written_is_cut_off_by_the_probe_budget(
    world: FakeWorld,
) -> None:
    world.hang_ping = True
    updates = make(health_check_s=0.03)
    iterator = updates.receive()
    await anext(iterator)
    with pytest.raises(TimeoutError) as caught:
        await asyncio.wait_for(anext(iterator), 5)
    assert "Redis live updates" in str(caught.value)


# -- R5: setup errors and internal objects carry no caller text ---------------------------------


async def test_a_setup_failure_inside_receive_is_sanitized_like_any_other(world: FakeWorld) -> None:
    updates = make()
    world.pool_error = RuntimeError("s3cret-sentinel driver text")  # after construction passed
    with pytest.raises(RuntimeError) as caught:
        await anext(updates.receive())
    assert "sentinel" not in str(caught.value) and "sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.parametrize("bad", ["latin-1", "utf-16", "ascii-nope", "s3cret-sentinel", 8])
def test_real_an_encoding_that_is_not_utf8_is_refused_without_echoing_it(bad: Any) -> None:
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates(f"redis://{HOST}/0", client_options={"encoding": bad}, **FAST)
    assert "sentinel" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.parametrize("name", ["utf-8", "UTF-8", "utf8", "U8"])
def test_real_the_spellings_of_utf8_are_accepted(name: str) -> None:
    assert ru.RedisUpdates(f"redis://{HOST}/0", client_options={"encoding": name}, **FAST)


def test_real_internal_clients_do_not_show_host_or_port_after_lazy_init() -> None:
    updates = ru.RedisUpdates(f"redis://:s3cret-sentinel@{HOST}:6380/0", **FAST)
    updates._sender()  # lazily builds redis-py's pool and client
    ru._Subscription(updates)  # and a listen pool and pub/sub object
    for text in (repr(vars(updates)), repr(updates.__dict__), repr(updates)):
        for sentinel in ("sentinel", HOST, "6380"):
            assert sentinel not in text


# -- V2: a malformed URL never reaches an error, a cause or a context ---------------------------

MALFORMED = [
    "redis://user:s3cret-sentinel@／sentinel-host:6380/0",  # fullwidth solidus in the host
    "redis://user:s3cret-sentinel@sentinel-host℀:6380/0",  # NFKC turns it into 'a/c'
    "redis://user:s3cret-sentinel@[sentinel-host:6380/0",  # unbalanced IPv6 bracket
    "rediss://:s3cret-sentinel@[::1/0?db=1",
]


@pytest.mark.parametrize("url", MALFORMED)
def test_real_a_malformed_url_is_refused_with_fixed_text(url: str) -> None:
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates(url, **FAST)
    error = caught.value
    for text in (str(error), repr(error)):
        assert "sentinel" not in text and "6380" not in text
    assert error.__cause__ is None and error.__context__ is None


# -- V3: the effective encoding is UTF-8 whichever way it is given -----------------------------


@pytest.mark.parametrize(
    "query",
    [
        "encoding=latin-1",
        "encoding=utf-16",
        "enc%6Fding=latin-1",  # percent-encoded key
        "encoding=lat%69n-1",  # percent-encoded value
        "encoding=utf-8&encoding=latin-1",  # a duplicate that redis-py may pick instead
        "encoding=latin-1&encoding=utf-8",
        "encoding=",
    ],
)
def test_real_a_url_encoding_that_is_not_utf8_is_refused(query: str) -> None:
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates(f"redis://:s3cret-sentinel@{HOST}/0?{query}", **FAST)
    assert "sentinel" not in str(caught.value) and "latin" not in str(caught.value)


@pytest.mark.parametrize("query", ["encoding=utf-8", "encoding=UTF8", "enc%6Fding=u%74f-8"])
def test_real_a_utf8_url_encoding_is_accepted(query: str) -> None:
    assert ru.RedisUpdates(f"redis://{HOST}/0?{query}", **FAST)


# -- V4: the whole probe hangs off one absolute deadline (idle deadline + probe budget) ---------


async def test_a_late_idle_read_is_charged_against_the_probe_budget(world: FakeWorld) -> None:
    world.pong_mode = "silent"
    updates = make(health_check_s=0.1, send_timeout_s=0.5, flush_deadline_s=1.0)
    iterator = updates.receive()
    await anext(iterator)
    subscription = updates._wire.subscription
    assert subscription is not None
    idle_until = subscription._idle_until
    pubsub = world.pubsubs[0]
    original = pubsub.get_message
    late = True

    async def get_message(**kwargs: Any) -> Any:
        nonlocal late
        if late:  # the idle read comes back after its own timeout, but inside the grace
            late = False
            await asyncio.sleep(0.15)
            return None
        return await original(**kwargs)

    pubsub.get_message = get_message  # type: ignore[method-assign]
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(anext(iterator), 5)
    assert pubsub.pings, "the probe still ran"
    # the pong deadline did not get a fresh full budget after the late return
    assert subscription._pong_by <= idle_until + updates.health_check_s + 1e-6


async def test_an_idle_read_that_returns_after_the_whole_budget_never_pings(
    world: FakeWorld,
) -> None:
    updates = make(health_check_s=0.05, send_timeout_s=0.5, flush_deadline_s=1.0)
    iterator = updates.receive()
    await anext(iterator)
    pubsub = world.pubsubs[0]

    async def get_message(**kwargs: Any) -> Any:
        await asyncio.sleep(0.3)  # longer than idle (50 ms) plus probe (50 ms)
        return None

    pubsub.get_message = get_message  # type: ignore[method-assign]
    with pytest.raises(TimeoutError) as caught:
        await asyncio.wait_for(anext(iterator), 5)
    assert "Redis live updates" in str(caught.value)
    assert pubsub.pings == []


# -- V1: the bound is proven on this redis-py before the transport is accepted -----------------


def absent_hook(base: type) -> type:
    """redis-py after a refactor that removed ``_read`` (or made it something else)."""
    return type("NoReadParser", (base,), {"_read": None})


def bypassed_hook(base: type) -> type:
    """redis-py after a refactor: bulk bodies are read by a renamed method, ``_read`` is unused."""
    import inspect
    import sys
    import textwrap

    source = textwrap.dedent(inspect.getsource(base._read_response)).replace(  # type: ignore[attr-defined]
        "self._read(", "self._read_bulk("
    )
    namespace = dict(vars(sys.modules[base.__module__]))
    exec(source, namespace)  # noqa: S102 - rebuilds the very method, with one name changed
    return type(
        "BypassedParser",
        (base,),
        {"_read_bulk": base._read, "_read_response": namespace["_read_response"]},  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("variant", [absent_hook, bypassed_hook], ids=["absent", "bypassed"])
def test_real_a_redis_client_whose_bulk_hook_changed_is_refused_not_used_unbounded(
    monkeypatch: pytest.MonkeyPatch, variant: Any
) -> None:
    import redis
    import redis._parsers as parsers

    monkeypatch.setattr(parsers, "_AsyncRESP2Parser", variant(parsers._AsyncRESP2Parser))
    with pytest.raises(ImportError) as caught:
        ru.RedisUpdates(f"redis://:s3cret-sentinel@{HOST}/0", **FAST)
    text = str(caught.value)
    assert redis.__version__ in text and "dendrux[redis]" in text  # names the client and the fix
    assert "sentinel" not in text


def test_real_the_installed_redis_client_passes_its_own_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ru.RedisUpdates(f"redis://{HOST}/0", **FAST)  # and again, from the cache
    assert ru.RedisUpdates(f"redis://{HOST}/0", max_message_bytes=3_000, **FAST)


async def test_real_the_check_also_runs_inside_a_running_event_loop() -> None:
    import redis._parsers as parsers

    # an application builds RedisUpdates inside its own loop; the check must not need a free one
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(parsers, "_AsyncRESP2Parser", absent_hook(parsers._AsyncRESP2Parser))
        with pytest.raises(ImportError):
            ru.RedisUpdates(f"redis://{HOST}/0", **FAST)


async def test_real_parser_cache_releases_closed_evicted_classes() -> None:
    capacity = getattr(ru, "_PARSER_CACHE_CAPACITY", 32)
    refs = []
    for limit in range(20_000, 20_000 + capacity * 4):
        updates = ru.RedisUpdates(f"redis://{HOST}/0", max_message_bytes=limit)
        refs.append(weakref.ref(updates._listen_parser))
        await updates.close()
        del updates
    gc.collect()
    assert sum(ref() is not None for ref in refs) <= capacity
    assert all(ref() is None for ref in refs[:-capacity])


async def test_real_parser_eviction_preserves_active_bounds_and_rechecks_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = []
    original = ru._problem_with

    def check(connection: type, parser: type, limit: int) -> str | None:
        checked.append(limit)
        return original(connection, parser, limit)

    monkeypatch.setattr(ru, "_problem_with", check)
    api = ru._load_api()
    capacity = getattr(ru, "_PARSER_CACHE_CAPACITY", 32)
    active = ru.RedisUpdates(f"redis://{HOST}/0", max_message_bytes=9_000)
    hot = api.bounded_parser(9_001)
    for limit in range(30_000, 30_000 + capacity * 4):
        api.bounded_parser(limit)
        assert api.bounded_parser(9_001) is hot  # hits refresh recency
    replacement = api.bounded_parser(9_000)
    assert checked.count(9_000) == 2  # the evicted entry must be checked again
    assert checked.count(9_001) == 1
    assert replacement is not active._listen_parser
    parser, reader = listen_parser(active)
    reader.feed_data(message(9_000))
    assert len((await parser.read_response())[2]) == 9_000
    reader.feed_data(message(9_001))
    assert (await asyncio.wait_for(parser.read_response(), 1))[2] is ru._DISCARDED
    reader.feed_data(message(1_000))
    assert len((await parser.read_response())[2]) == 1_000
    await active.close()


def assert_encoding_refused(path: str, value: str) -> None:
    url = f"redis://{HOST}/0"
    options = (
        {"url": url + "?encoding=" + value}
        if path == "url"
        else {"url": url, "client_options": {"encoding": value}}
    )
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates(**options)
    error = caught.value
    expected = f"{path} encoding must be utf-8, the encoding of the wire format"
    assert str(error) == expected
    assert repr(error) == repr(ValueError(expected))
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("path", ["url", "client_options"])
def test_real_surrogate_encoding_is_refused_without_disclosure(path: str) -> None:
    assert_encoding_refused(path, chr(0xD800) + "secret-sentinel")


@pytest.mark.parametrize("path", ["url", "client_options"])
@pytest.mark.parametrize("failure", [LookupError, UnicodeError, ValueError, TypeError])
def test_codec_validation_errors_are_refused_without_disclosure(
    monkeypatch: pytest.MonkeyPatch, path: str, failure: type[Exception]
) -> None:
    def lookup(name: str) -> Any:
        raise failure("secret-sentinel")

    monkeypatch.setattr(ru.codecs, "lookup", lookup)
    assert_encoding_refused(path, "secret-sentinel")
