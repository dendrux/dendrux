"""PostgreSQL LISTEN/NOTIFY live updates.

Two dedicated asyncpg connections, never from the store's pool: a sender opened
lazily inside ``send`` and a listener owned by ``receive``. Notifications carry
run ids only; readers always re-read the database. No DSN, host, password or
driver text ever reaches a log line or an error.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote, urlencode

from dendrux import _codes
from dendrux.db._live_updates import LiveUpdates
from dendrux.db._notify import validate_channel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from sqlalchemy.engine import URL

    from dendrux.db._live_updates import _Gap

logger = logging.getLogger(__name__)

#: Postgres rejects NOTIFY payloads of 8,000 bytes or more.
MAX_NOTIFY_BYTES = 7_999
APPLICATION_NAME = "dendrux-live-updates"
_NOTIFY_SQL = "SELECT pg_notify($1, m) FROM unnest($2::text[]) AS m"

# Keys asyncpg's DSN parser reads itself (checked against the installed version
# at import, see ``_dsn_keys``), plus the two startup settings the spec keeps.
_FALLBACK_DSN_KEYS = frozenset(
    {
        "host",
        "port",
        "user",
        "password",
        "database",
        "dbname",
        "passfile",
        "service",
        "sslmode",
        "sslcert",
        "sslkey",
        "sslrootcert",
        "sslcrl",
        "sslpassword",
        "sslnegotiation",
        "ssl_min_protocol_version",
        "ssl_max_protocol_version",
        "target_session_attrs",
        "krbsrvname",
        "gsslib",
    }
)
_STARTUP_SETTINGS = frozenset({"application_name", "options"})
_FORCED_OPTIONS = ("statement_cache_size", "timeout")
_DESTINATION_KEYS = frozenset({"host", "port", "user", "password", "database", "dbname"})


def _dsn_keys() -> frozenset[str]:
    """Query keys the installed asyncpg DSN parser understands (fallback: a fixed list)."""
    try:
        import inspect

        from asyncpg import connect_utils  # type: ignore[import-untyped]

        source = inspect.getsource(connect_utils._parse_connect_dsn_and_args)
    except Exception:  # pragma: no cover - asyncpg missing or source unavailable
        return _FALLBACK_DSN_KEYS
    found = frozenset(re.findall(r"if '(\w+)' in query", source))
    return found or _FALLBACK_DSN_KEYS


def _parse(url: str | URL) -> URL:
    from sqlalchemy.engine import make_url

    parsed: URL | None
    try:
        parsed = make_url(url)
    except Exception:
        parsed = None  # raised below, outside the handler, so no chain keeps the URL
    if parsed is None:
        raise ValueError("PostgresUpdates could not parse the database URL.")
    if parsed.get_backend_name() not in ("postgresql", "postgres"):
        raise ValueError("PostgresUpdates needs a PostgreSQL URL.")
    return parsed


def _refuse(key: str) -> ValueError:
    return ValueError(f"PostgresUpdates refuses an ambiguous {key!r} in the URL; pass url=.")


def _single(key: str, value: object) -> str:
    if isinstance(value, tuple):
        raise _refuse(key)
    return str(value)


def _one(url_value: object, query: Mapping[str, Any], *keys: str) -> str | None:
    """One value for a destination field given in the URL part or the query, never both."""
    found = [str(url_value)] if url_value is not None else []
    found += [_single(key, query[key]) for key in keys if key in query]
    if len(found) > 1:
        raise _refuse(keys[0])
    return found[0] if found else None


def _query_hosts(parsed: URL) -> list[tuple[str, str]]:
    """Query ``host`` (and ``port``) kept as asyncpg reads them; refused next to an authority."""
    query = parsed.query
    if "host" not in query:
        return []
    if parsed.host is not None or parsed.port is not None:
        raise _refuse("host")
    value = query["host"]
    hosts = ",".join(value) if isinstance(value, tuple) else str(value)
    pairs = [("host", hosts)]
    if "port" in query:
        if any(":" in spec and not spec.startswith(("/", "[")) for spec in hosts.split(",")):
            raise _refuse("port")
        pairs.append(("port", _single("port", query["port"])))
    return pairs


def _options(parsed: URL, allowed: frozenset[str]) -> list[tuple[str, str]]:
    """Non-destination query keys, each one asyncpg reads; any other key is refused by name."""
    pairs: list[tuple[str, str]] = []
    for key, value in parsed.query.items():
        if key not in allowed:
            raise ValueError(
                f"PostgresUpdates cannot use the URL option {key!r}; pass url= explicitly."
            )
        if key not in _DESTINATION_KEYS:
            pairs.append((key, _single(key, value)))
    return pairs


def _authority(user: str | None, password: str | None, host: str, port: str | None) -> str:
    userinfo = ""
    if user is not None:
        userinfo = quote(user, safe="")
        if password is not None:
            userinfo += ":" + quote(password, safe="")
        userinfo += "@"
    if ":" in host:
        host = f"[{host}]"
    elif host:
        host = quote(host, safe="")
    return userinfo + host + (f":{port}" if port is not None else "")


def asyncpg_dsn(url: str | URL) -> str:
    """Convert a SQLAlchemy PostgreSQL URL to an asyncpg DSN with the same destination.

    Destination fields (user, password, host, port, database) given either in
    the URL parts or as query keys are normalized so asyncpg resolves them
    exactly as SQLAlchemy would; a field given twice is refused. Credentials,
    IPv6 hosts and the database are escaped. Every other query key asyncpg's
    DSN parser accepts is kept; any other key raises ``ValueError`` naming only
    the key, so nothing is silently dropped.
    """
    parsed = _parse(url)
    query = parsed.query
    hosts = _query_hosts(parsed)
    user = _one(parsed.username, query, "user")
    password = _one(parsed.password, query, "password")
    database = _one(parsed.database, query, "database", "dbname")
    port = None if hosts else _one(parsed.port, query, "port")
    pairs = _options(parsed, _dsn_keys() | _STARTUP_SETTINGS)
    if password is not None and user is None:
        pairs.append(("password", password))
    authority = _authority(user, password if user is not None else None, parsed.host or "", port)
    path = "/" + quote(database, safe="") if database is not None else ""
    query_text = "?" + urlencode([*hosts, *pairs]) if hosts or pairs else ""
    return f"postgresql://{authority}{path}{query_text}"


def _check_connect_args(connect_args: Mapping[str, Any] | None) -> dict[str, Any]:
    if connect_args is None:
        return {}
    if not isinstance(connect_args, Mapping):
        raise ValueError("connect_args must be a mapping.")
    for key in _FORCED_OPTIONS:
        if key in connect_args:
            raise ValueError(f"connect_args may not set {key!r}; PostgresUpdates controls it.")
    return dict(connect_args)


def connect_options(
    connect_args: Mapping[str, Any], *, timeout: float, url_settings: frozenset[str]
) -> dict[str, Any]:
    """Caller options first, then the forced ones; server settings are merged."""
    options = dict(connect_args)
    options["statement_cache_size"] = 0
    options["timeout"] = timeout
    settings = dict(options.get("server_settings") or {})
    if "application_name" not in settings and "application_name" not in url_settings:
        settings["application_name"] = APPLICATION_NAME
    if settings:
        options["server_settings"] = settings
    return options


def _hosts(dsn: str) -> list[str]:
    from sqlalchemy.engine import make_url

    parsed = make_url(dsn)
    query_hosts = parsed.query.get("host")
    if query_hosts is None:
        return [parsed.host or ""]
    joined = query_hosts if isinstance(query_hosts, str) else ",".join(query_hosts)
    return [_host_of(spec) for spec in joined.split(",")]


def _host_of(spec: str) -> str:
    if spec.startswith("/"):
        return spec
    if spec.startswith("["):
        return spec[1 : spec.find("]")]
    return spec.split(":", 1)[0]


def _loopback(host: str) -> bool:
    if not host or host.startswith("/") or host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class PostgresUpdates(LiveUpdates):
    """Live updates over PostgreSQL ``LISTEN``/``NOTIFY``.

    ``url`` sends (and listens, unless ``listen_url`` is given); it defaults to
    the bound store's engine URL. Behind PgBouncer transaction pooling, pass a
    direct or session-pooled ``listen_url``: LISTEN needs a session, NOTIFY does
    not. A read replica never receives notifications. ``connect_args`` go to
    ``asyncpg.connect`` (an SSL context, for example); the engine's own
    ``connect_args`` cannot be recovered, so pass TLS settings here or in the URL.
    """

    def __init__(
        self,
        *,
        url: str | URL | None = None,
        listen_url: str | URL | None = None,
        connect_args: Mapping[str, Any] | None = None,
        channel: str = "dendrux_events",
        **limits: Any,
    ) -> None:
        super().__init__(**limits)
        if self.max_message_bytes > MAX_NOTIFY_BYTES:
            raise ValueError(f"max_message_bytes must be at most {MAX_NOTIFY_BYTES} on Postgres.")
        self.channel = validate_channel(channel)
        self._connect_args = _check_connect_args(connect_args)
        self._send_dsn: str | None = asyncpg_dsn(url) if url is not None else None
        explicit_listen = asyncpg_dsn(listen_url) if listen_url is not None else None
        self._explicit_listen = explicit_listen
        self._listen_dsn: str | None = explicit_listen or self._send_dsn
        self._tls_checked = False
        self._sender: Any = None
        # Silence (health_check_s) plus this probe stays within 2 x health_check_s.
        self._probe_timeout_s = min(self.send_timeout_s * 4, self.health_check_s)
        self._listener: Any = None

    def bind(self, store: object) -> None:
        """Default ``url`` to the store's engine URL; refuse a store without one."""
        if self._send_dsn is not None:
            return
        engine = getattr(store, "_engine", None)
        engine_url = getattr(engine, "url", None)
        if engine_url is None:
            raise ValueError("PostgresUpdates needs url= for a store without a SQLAlchemy engine.")
        self._send_dsn = asyncpg_dsn(engine_url)
        self._listen_dsn = self._explicit_listen or self._send_dsn

    def _check_tls(self) -> None:
        """Warn once when a remote destination has no explicit TLS setting."""
        if self._tls_checked:
            return
        self._tls_checked = True
        if "ssl" in self._connect_args:
            return
        for dsn in {self._send_dsn, self._listen_dsn} - {None}:
            assert dsn is not None
            if "sslmode=" in dsn.partition("?")[2]:
                continue
            if not all(_loopback(host) for host in _hosts(dsn)):
                _codes.log_code(logger, _codes.LIVE_UPDATES_TLS_UNKNOWN)
                return

    async def _connect(self, dsn: str | None, timeout: float) -> Any:
        import asyncpg

        if dsn is None:
            raise ValueError("PostgresUpdates is not bound to a store and has no url=.")
        self._check_tls()
        url_settings = frozenset(k for k in _STARTUP_SETTINGS if f"{k}=" in dsn.partition("?")[2])
        options = connect_options(self._connect_args, timeout=timeout, url_settings=url_settings)
        return await asyncpg.connect(dsn, **options)

    async def send(self, messages: Sequence[str]) -> None:
        """One ``pg_notify`` statement per call on the dedicated sender connection."""
        connection = self._sender
        if connection is None or connection.is_closed():
            connection = await self._connect(self._send_dsn, self.send_timeout_s)
            self._sender = connection
        await connection.execute(_NOTIFY_SQL, self.channel, list(messages))

    def abort_sender(self) -> None:
        """Terminate the sender connection now; the next ``send`` reconnects."""
        connection, self._sender = self._sender, None
        if connection is not None:
            connection.terminate()

    async def aclose_sender(self) -> None:
        """Close the sender connection, terminating it if a clean close fails."""
        connection, self._sender = self._sender, None
        if connection is None or connection.is_closed():
            return
        try:
            await connection.close(timeout=self.send_timeout_s)
        except BaseException:
            connection.terminate()
            raise

    def abort_listener(self) -> None:
        """Terminate the listen connection now; ``receive`` then ends."""
        connection = self._listener
        if connection is not None:
            connection.terminate()

    async def receive(self) -> AsyncIterator[str | None | _Gap]:
        """LISTEN on a dedicated connection; ``None`` once LISTEN has executed.

        Liveness: after ``health_check_s`` without a message, ``SELECT 1`` runs
        on this same connection with ``min(4 x send_timeout_s, health_check_s)``
        to answer, so a dead connection is found within ``2 x health_check_s``;
        a failure or a lost connection ends the stream.
        The connection is always terminated when the stream ends.
        """
        buffer = self._new_inbound_buffer()
        lost = asyncio.Event()
        connection = None
        try:
            connection = await self._connect(self._listen_dsn, self.send_timeout_s * 4)
            self._listener = connection
            connection.add_termination_listener(lambda _connection: lost.set())
            await connection.add_listener(
                self.channel, lambda _c, _pid, _channel, payload: buffer.put(payload)
            )
            yield None
            while True:
                yield await self._next_message(connection, buffer, lost)
        finally:
            if self._listener is connection:
                self._listener = None
            if connection is not None:
                connection.terminate()

    async def _next_message(self, connection: Any, buffer: Any, lost: asyncio.Event) -> str | _Gap:
        while True:
            if lost.is_set():
                raise ConnectionError("live updates listener lost")
            message = await self._wait_message(buffer, lost)
            if message is not None:
                return cast("str | _Gap", message)
            if not lost.is_set():
                async with asyncio.timeout(self._probe_timeout_s):
                    await connection.execute("SELECT 1")

    async def _wait_message(self, buffer: Any, lost: asyncio.Event) -> object | None:
        """The next buffered item, or ``None`` after ``health_check_s`` of silence or loss."""
        getter = asyncio.ensure_future(buffer.get())
        waiter = asyncio.ensure_future(lost.wait())
        try:
            await asyncio.wait(
                {getter, waiter}, timeout=self.health_check_s, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            waiter.cancel()
            if not getter.done():
                getter.cancel()
        return getter.result() if getter.done() and not getter.cancelled() else None
