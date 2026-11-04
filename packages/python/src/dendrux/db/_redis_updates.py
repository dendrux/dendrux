"""Redis Pub/Sub transport for live updates.

``RedisUpdates`` implements the two transport hooks of ``LiveUpdates`` on ``redis.asyncio``:

* ``send`` publishes the base's encoded messages in one pipelined round trip on a connection
  that only the sender owns (a one-slot pool, no retries, no idle second client);
* ``receive`` subscribes on a dedicated connection, reports healthy only after the server
  acknowledged the subscription, and proves liveness with a token-matched ``PING`` on that same
  connection. It never reconnects by itself: redis-py would resubscribe silently and hide a gap.

The module imports without the ``redis`` package; constructing ``RedisUpdates`` is what needs
``pip install dendrux[redis]``. Nothing here logs. Errors that leave the transport are plain
``TimeoutError`` / ``ConnectionError`` / ``RuntimeError`` objects with a fixed text and no
chained original, because redis-py puts host and port into its messages.
"""

from __future__ import annotations

import asyncio
import codecs
import secrets
import ssl
import threading
from collections.abc import Mapping
from contextlib import suppress
from functools import lru_cache
from time import monotonic
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlsplit

from dendrux.db._live_updates import LiveUpdates

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from dendrux.db._live_updates import _Gap

INSTALL_HINT = "RedisUpdates needs the Redis client: pip install dendrux[redis]"
MAX_CHANNEL_LENGTH = 512
_SCHEMES = ("redis://", "rediss://", "unix://")
# Options that would defeat a guarantee of this transport, or move the destination away from url.
_FORBIDDEN_OPTIONS = frozenset(
    {
        "socket_timeout",
        "socket_connect_timeout",
        "retry",
        "retry_on_timeout",
        "retry_on_error",
        "max_connections",
        "connection_pool",
        "connection_class",
        "single_connection_client",
        "health_check_interval",
        "host",
        "port",
        "unix_socket_path",
        "protocol",
        "parser_class",
        "decode_responses",
    }
)
_POOL_ATTRIBUTES = ("_in_use_connections", "_available_connections")


def _load_api() -> Any:
    """The parts of ``redis.asyncio`` this module uses; the one place that imports redis.

    Importing the parser classes proves nothing about how they read a message, so the bound on
    incoming messages is not trusted here: ``bounded_parser(limit)`` runs the parser on in-memory
    streams and only returns it if it reads a normal message, discards an oversized one,
    and preserves the next reply.
    """
    try:
        from redis import exceptions
        from redis._parsers import _AsyncRESP2Parser
        from redis.asyncio import ConnectionPool, Redis
        from redis.asyncio.connection import Connection
        from redis.asyncio.retry import Retry
        from redis.backoff import NoBackoff
    except ImportError:
        raise ImportError(INSTALL_HINT) from None
    return SimpleNamespace(
        bounded_parser=lambda limit: _verified_parser(Connection, _AsyncRESP2Parser, limit),
        ConnectionPool=ConnectionPool,
        Redis=Redis,
        Retry=Retry,
        NoBackoff=NoBackoff,
        ConnectionError=exceptions.ConnectionError,
        TimeoutError=exceptions.TimeoutError,
        RedisError=exceptions.RedisError,
    )


_DISCARDED = b"\x00dendrux-discarded-bulk"
_DISCARD_CHUNK_BYTES = 65_536


async def _discard_bulk(parser: Any, length: int) -> bytes:
    remaining = length + 2
    buffered = min(remaining, max(0, len(parser._buffer) - parser._pos))
    remaining -= buffered
    try:
        while remaining:
            size = min(remaining, _DISCARD_CHUNK_BYTES)
            await parser._stream.readexactly(size)
            remaining -= size
    except (asyncio.CancelledError, asyncio.IncompleteReadError):
        # A partial discard cannot be replayed from redis-py's retry buffer.
        # Force teardown even when get_message would normally swallow a poll timeout.
        raise ConnectionError("Redis live updates discard interrupted") from None
    parser._pos += length + 2
    return _DISCARDED


def _bounded_parser(base: type, limit: int) -> type:
    """Drain oversized bulks without appending their bodies to redis-py's retry buffer."""

    class BoundedParser(base):  # type: ignore[misc]
        async def _read(self, length: int) -> bytes:
            if length > limit:
                return await _discard_bulk(self, length)
            return bytes(await super()._read(length))

    return BoundedParser


def _reply(body: bytes) -> bytes:
    """A pub/sub message reply as RESP2 puts it on the wire."""
    return (
        b"*3\r\n$7\r\nmessage\r\n$1\r\nc\r\n$" + str(len(body)).encode() + b"\r\n" + body + b"\r\n"
    )


async def _check_parser(connection_class: type, parser_class: type, limit: int) -> str | None:
    """Why ``parser_class`` cannot be trusted to bound a message, or ``None`` if it can."""
    connection = connection_class(parser_class=parser_class, protocol=2)
    reader = asyncio.StreamReader()
    connection._reader = reader
    parser = connection._parser
    parser.on_connect(connection)
    reader.feed_data(_reply(b"hello"))
    if (await asyncio.wait_for(parser.read_response(), 2))[2] != b"hello":
        return "it does not read an ordinary message"
    reader.feed_data(_reply(b"x" * (limit + 1)) + _reply(b"after"))
    if (await asyncio.wait_for(parser.read_response(), 2))[2] is not _DISCARDED:
        return "it does not discard an oversized message"
    if (await asyncio.wait_for(parser.read_response(), 2))[2] != b"after":
        return "it does not preserve the following message"
    return None


def _problem_with(connection_class: type, parser_class: type, limit: int) -> str | None:
    """Run the check on a short-lived event loop in its own thread.

    A thread, because RedisUpdates is built inside a running event loop as often as outside one.
    """
    outcome: list[str | None] = []

    def run() -> None:
        try:
            outcome.append(asyncio.run(_check_parser(connection_class, parser_class, limit)))
        except BaseException:
            outcome.append("it could not be checked")

    worker = threading.Thread(target=run, name="dendrux-redis-parser-check", daemon=True)
    worker.start()
    worker.join(10)
    return outcome[0] if outcome else "the check did not finish"


# Retain at most 32 checked configurations; active instances own their parser after eviction.
_PARSER_CACHE_CAPACITY = 32


@lru_cache(maxsize=_PARSER_CACHE_CAPACITY)
def _checked_parser(connection_class: type, base: type, limit: int) -> tuple[type, str | None]:
    parser = _bounded_parser(base, limit)
    return parser, _problem_with(connection_class, parser, limit)


def _verified_parser(connection_class: type, base: type, limit: int) -> type:
    """The bounded parser for ``base``, or ``ImportError`` if redis-py cannot be made to bound."""
    parser, problem = _checked_parser(connection_class, base, limit)
    if problem is not None:
        import redis

        raise ImportError(
            f"RedisUpdates cannot bound incoming messages with redis-py {redis.__version__} "
            f"({problem}); use a tested release (pip install 'dendrux[redis]': 5.0.1, 5.3.1, "
            "6.4.0, 7.4.0 and 8.1.0 are exercised)"
        )
    return parser


class _Wire:
    """The transport handles of one RedisUpdates.

    Its repr shows nothing: redis-py's pool, client and connection reprs contain the host,
    port and user, and ``vars(updates)`` must stay free of them.
    """

    __slots__ = ("client", "pipeline", "pool", "subscription")

    def __init__(self) -> None:
        self.pool: Any = None
        self.client: Any = None
        self.pipeline: Any = None
        self.subscription: _Subscription | None = None

    def __repr__(self) -> str:
        return "<wire>"


class _Redacted:
    """Holds a value whose text must never reach a repr, a log line or a traceback."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def reveal(self) -> Any:
        return self._value

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("a RedisUpdates connection setting cannot be pickled")


def _query_of(url: str) -> list[tuple[str, str]] | None:
    """The URL's decoded query pairs, or ``None`` if the URL cannot be parsed.

    The parser's own error text repeats the URL (credentials included), so it is dropped here.
    """
    try:
        return parse_qsl(urlsplit(url).query, keep_blank_values=True)
    except Exception:
        return None


def _checked_query(query: list[tuple[str, str]]) -> None:
    # redis-py lets a query parameter override a keyword argument, so the fixed settings must be
    # refused here too; parse_qsl decodes keys exactly as redis-py's URL parser does
    refused = sorted({key for key, _ in query if key in _FORBIDDEN_OPTIONS})
    if refused:
        raise ValueError(
            "url may not set " + ", ".join(refused) + " in its query: RedisUpdates fixes the "
            "timeouts, retries, protocol and single connection"
        )
    # every spelling counts: redis-py takes one of several duplicates and lets it win
    if not all(_is_utf8(value) for key, value in query if key == "encoding"):
        raise ValueError("url encoding must be utf-8, the encoding of the wire format")


def _checked_url(url: object) -> str:
    if not isinstance(url, str) or not url.startswith(_SCHEMES):
        raise ValueError("url must be a string starting with redis://, rediss:// or unix://")
    query = _query_of(url)  # outside any handler: nothing of the original error is retained
    if query is None:
        raise ValueError("url is not a valid Redis URL")
    _checked_query(query)
    _check_port(url)
    return url


def _checked_channel(channel: object) -> str:
    if (
        not isinstance(channel, str)
        or not 0 < len(channel) <= MAX_CHANNEL_LENGTH
        or not channel.isprintable()
        or " " in channel
    ):
        raise ValueError(
            f"channel must be 1 to {MAX_CHANNEL_LENGTH} printable characters without spaces"
        )
    return channel


def _is_utf8(name: object) -> bool:
    try:
        return isinstance(name, str) and codecs.lookup(name).name == "utf-8"
    except (LookupError, UnicodeError, ValueError, TypeError):
        return False


def _option_mapping(options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("client_options must be a mapping")
    return dict(options or {})


def _check_port(url: str) -> None:
    valid = False
    try:
        port = urlsplit(url).port
        valid = port is None or 0 < port <= 65_535
    except ValueError:
        pass
    if not valid:
        raise ValueError("Redis URL port must be a number from 1 to 65535")


def _tls_failure(error: BaseException) -> bool:
    current: BaseException | None = error
    for _ in range(8):
        if current is None:
            return False
        if isinstance(current, ssl.SSLError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _redis_error(api: Any, error: BaseException) -> Exception:
    if _tls_failure(error):
        return ssl.SSLError("Redis live updates TLS handshake failed")
    if isinstance(error, (TimeoutError, api.TimeoutError)):
        return TimeoutError("Redis live updates timed out")
    if isinstance(error, (OSError, api.ConnectionError)):
        return ConnectionError("Redis live updates connection failed")
    return RuntimeError("Redis live updates failed")


def _checked_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    chosen = _option_mapping(options)
    if "encoding" in chosen and not _is_utf8(chosen["encoding"]):
        raise ValueError("client_options encoding must be utf-8, the encoding of the wire format")
    refused = sorted(key for key in chosen if key in _FORBIDDEN_OPTIONS)
    if refused:
        raise ValueError(
            "client_options may not set " + ", ".join(refused) + ": RedisUpdates fixes the "
            "timeouts, retries and single connection, and url gives the destination"
        )
    return chosen


def _abort_transports(pool: Any, extra: Iterable[Any] = ()) -> None:
    """Synchronously drop the sockets of every connection a pool holds, plus ``extra``."""
    connections = [conn for conn in extra if conn is not None]
    if pool is not None:
        for name in _POOL_ATTRIBUTES:
            connections.extend(getattr(pool, name, None) or ())
    for connection in connections:
        with suppress(Exception):
            connection._writer.transport.abort()


def _text(data: object) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data if isinstance(data, str) else ""


class _Subscription:
    """One listen connection: its pool, its Pub/Sub object and its liveness state."""

    def __init__(self, owner: RedisUpdates) -> None:
        self.owner = owner
        self.pool = owner._new_pool(owner.send_timeout_s * 4, listening=True)
        self.pubsub = owner._api.Redis(connection_pool=self.pool).pubsub()
        #: Probe budget: writing the PING and waiting for its pong. Clamped to the health-check
        #: interval so detection (idle wait + probe) never exceeds twice that interval.
        self.probe_s = min(owner.send_timeout_s * 4, owner.health_check_s)
        self._token: str | None = None
        self._idle_until = self._detect_by = self._pong_by = 0.0
        self._alive()
        owner._wire.subscription = self

    def _alive(self) -> None:
        """Proof of life: restart the idle wait. One absolute deadline then covers everything
        that follows (idle wait, PING write, pong wait, and whatever redis-py does around them).
        While a PING is outstanding only its pong counts: other traffic does not move it."""
        if self._token is not None:
            return
        self._idle_until = monotonic() + self.owner.health_check_s
        self._detect_by = self._idle_until + self.probe_s

    def abort(self) -> None:
        _abort_transports(self.pool, [getattr(self.pubsub, "connection", None)])

    async def open(self) -> None:
        owner = self.owner
        async with asyncio.timeout(owner.send_timeout_s * 4) as scope:
            await self.pubsub.subscribe(owner._channel)
            while True:
                remaining = max(0.0, (scope.when() or 0.0) - asyncio.get_running_loop().time())
                reply = await self.pubsub.get_message(
                    ignore_subscribe_messages=False, timeout=remaining
                )
                if self._acknowledges(reply):
                    break
        self._alive()

    def _acknowledges(self, reply: Any) -> bool:
        return (
            isinstance(reply, dict)
            and reply.get("type") == "subscribe"
            and _text(reply.get("channel")) == self.owner._channel
        )

    async def step(self) -> str | _Gap | None:
        """Read at most one reply; the next message or gap to yield, or ``None`` to go on.

        The read sits inside the absolute detection deadline, so nothing redis-py does around it
        (a reconnect, a handshake) can outlast the budget, and a late return is charged to it.
        """
        deadline = self._pong_by if self._token is not None else self._idle_until
        wait = deadline - monotonic()
        if wait <= 0:
            await self._on_silence()
            return None
        async with asyncio.timeout(self._detect_by - monotonic()):
            reply = await self.pubsub.get_message(ignore_subscribe_messages=False, timeout=wait)
        return self._interpret(reply) if isinstance(reply, dict) else None

    def _interpret(self, reply: dict[str, Any]) -> str | _Gap | None:
        kind = reply.get("type")
        if kind == "message":
            return self._message(reply.get("data"))
        if kind == "pong" and self._token is not None and _text(reply.get("data")) == self._token:
            self._token = None
            self._alive()
        elif self._acknowledges(reply):
            # a second subscribe acknowledgement means the client resubscribed behind our back
            return LiveUpdates.GAP
        return None

    def _message(self, data: object) -> str | None:
        self._alive()
        if data is _DISCARDED:
            self.owner._oversize_dropped()
            return None
        text = _text(data)
        if len(text.encode("utf-8")) > self.owner.max_message_bytes:
            self.owner._oversize_dropped()
            return None
        return text

    async def _on_silence(self) -> None:
        remaining = self._detect_by - monotonic()
        if self._token is not None or remaining <= 0:  # no pong in time, or no budget left to ask
            raise TimeoutError
        self._token = secrets.token_hex(8)
        self._pong_by = self._detect_by
        async with asyncio.timeout(remaining):
            await self.pubsub.ping(self._token)

    async def close(self) -> None:
        self.abort()  # first, synchronously: nothing below may leave the socket open
        if self.owner._wire.subscription is self:
            self.owner._wire.subscription = None
        with suppress(Exception):
            await self.pubsub.aclose()
        with suppress(Exception):
            await self.pool.aclose()


class RedisUpdates(LiveUpdates):
    """Live updates over Redis Pub/Sub (``pip install dendrux[redis]``).

    Pub/Sub ignores the Redis database number, so channel names must be unique per application
    and environment; the default is a starting point, not a namespace. Every subscriber receives
    every message on the channel: Redis moves the fan-out off the database, it does not remove it.
    """

    def __init__(
        self,
        url: str,
        *,
        channel: str = "dendrux:events",
        client_options: Mapping[str, Any] | None = None,
        **limits: Any,
    ) -> None:
        self._api = _load_api()
        super().__init__(**limits)
        self._channel = _checked_channel(channel)
        self._url = _Redacted(_checked_url(url))
        self._options = _Redacted(_checked_options(client_options))
        self._wire = _Wire()
        # proven on the installed redis-py, here, before anything connects (ImportError if not)
        self._listen_parser = self._api.bounded_parser(
            max(self.max_message_bytes, 4 * MAX_CHANNEL_LENGTH)
        )
        self._check_options()

    def __repr__(self) -> str:
        return f"RedisUpdates(channel={self._channel!r})"

    def _new_pool(self, socket_timeout_s: float, *, listening: bool = False) -> Any:
        api = self._api
        options = {
            **self._options.reveal(),
            "max_connections": 1,
            "socket_timeout": socket_timeout_s,
            "socket_connect_timeout": self.send_timeout_s,
            "retry": api.Retry(api.NoBackoff(), 0, supported_errors=()),
        }
        if listening:
            # RESP2 with a parser that drains oversized messages in bounded chunks
            options["protocol"] = 2
            options["parser_class"] = self._listen_parser
        return api.ConnectionPool.from_url(self._url.reveal(), **options)

    def _check_options(self) -> None:
        """Refuse a bad url or option now, without echoing either."""
        valid = True
        try:
            self._new_pool(self.send_timeout_s).make_connection()
            self._new_pool(self.send_timeout_s, listening=True).make_connection()
        except Exception:
            valid = False
        if not valid:
            raise ValueError(
                "A Redis URL option or client_options setting is unsupported; "
                "check the option names and types"
            )

    def _plain_error(self, error: BaseException) -> Exception:
        return _redis_error(self._api, error)

    # -- send ------------------------------------------------------------------------------

    def _sender(self) -> Any:
        if self._wire.client is None:
            self._wire.pool = self._new_pool(self.send_timeout_s)
            self._wire.client = self._api.Redis(connection_pool=self._wire.pool)
        return self._wire.client

    async def send(self, messages: Sequence[str]) -> None:
        pipeline = self._sender().pipeline(transaction=False)
        for message in messages:
            pipeline.publish(self._channel, message)
        self._wire.pipeline = pipeline
        failure: Exception | None = None
        try:
            await pipeline.execute()
        except Exception as error:
            failure = self._plain_error(error)
        finally:
            self._wire.pipeline = None
        if failure is not None:
            raise failure

    def abort_sender(self) -> None:
        wire = self._wire
        pipeline, pool = wire.pipeline, wire.pool
        wire.pipeline = wire.pool = wire.client = None  # the next send starts from scratch
        _abort_transports(pool, [getattr(pipeline, "connection", None)])

    async def aclose_sender(self) -> None:
        # The handles stay until cleanup is over: if cleanup stalls, is cancelled or times out,
        # the ``finally`` still finds the pool and drops its sockets synchronously.
        client, pool = self._wire.client, self._wire.pool
        try:
            if client is not None:
                with suppress(Exception):
                    await client.aclose()
            if pool is not None:
                with suppress(Exception):
                    await pool.aclose()
        finally:
            self.abort_sender()

    # -- receive ---------------------------------------------------------------------------

    async def receive(self) -> AsyncIterator[str | None | _Gap]:
        subscription: _Subscription | None = None
        failure: Exception | None = None
        try:
            subscription = _Subscription(self)
            await subscription.open()
            yield None
            while True:
                item = await subscription.step()
                if item is not None:
                    yield item
        except Exception as error:
            failure = self._plain_error(error)
        finally:
            if subscription is not None:
                await subscription.close()
        if failure is not None:
            raise failure

    def abort_listener(self) -> None:
        if self._wire.subscription is not None:
            self._wire.subscription.abort()
