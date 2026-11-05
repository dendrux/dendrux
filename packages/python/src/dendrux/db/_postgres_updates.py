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
import ssl
from collections.abc import Iterable, Mapping
from contextlib import aclosing, suppress
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qsl, quote, urlencode

from dendrux import _codes
from dendrux.db._live_updates import LiveUpdates, _error_type, _shutdown_socket
from dendrux.db._notify import validate_channel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Sequence

    from sqlalchemy.engine import URL

    from dendrux.db._live_updates import _Gap

logger = logging.getLogger(__name__)

#: Postgres rejects NOTIFY payloads of 8,000 bytes or more.
MAX_NOTIFY_BYTES = 7_999
APPLICATION_NAME = "dendrux-live-updates"
_NOTIFY_SQL = "SELECT pg_notify($1, m) FROM unnest($2::text[]) AS m"


_TRANSPORT_FAILURES = (
    (TimeoutError, TimeoutError, ("Postgres live updates timed out",)),
    (ssl.SSLError, ssl.SSLError, (ssl.SSL_ERROR_SSL, "Postgres live updates TLS failed")),
    (OSError, ConnectionError, ("Postgres live updates connection failed",)),
    (
        ImportError,
        ImportError,
        ("Postgres live updates requires asyncpg; install dendrux[postgres].",),
    ),
    (
        ValueError,
        ValueError,
        ("Postgres live updates configuration failed; check url= and connect_args.",),
    ),
    (Exception, RuntimeError, ("Postgres live updates failed",)),
)


def _transport_error(error: Exception) -> Exception:
    for source, base, arguments in _TRANSPORT_FAILURES:
        if isinstance(error, source):
            failure: Any = base(*arguments)
            # Keep only a validated name; the builtin preserves kind and pickling.
            failure.safe_type = _error_type(error)
            return cast("Exception", failure)
    raise AssertionError("unreachable transport error type")


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
_KNOWN_REJECTED_KEYS = frozenset(
    {"ssl", "prepared_statement_cache_size", "pool_size", "statement_cache_size"}
)
_STARTUP_SETTINGS = frozenset({"application_name", "options"})
_FORCED_OPTIONS = ("statement_cache_size", "timeout")
_DESTINATION_KEYS = frozenset({"host", "port", "user", "password", "database", "dbname"})


@lru_cache(maxsize=1)
def _dsn_keys() -> frozenset[str]:
    """Query keys the installed asyncpg DSN parser understands (fallback: a fixed list)."""
    try:
        import inspect

        from asyncpg import connect_utils  # type: ignore[import-untyped]

        source = inspect.getsource(connect_utils._parse_connect_dsn_and_args)
    except Exception:
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


def _port(value: object) -> str:
    """A port as canonical decimal text, 1..65535; anything else is refused unechoed."""
    text = _single("port", value)
    if not (text.isascii() and text.isdigit()) or not 0 < int(text) <= 65_535:
        raise ValueError("PostgresUpdates needs the URL port as a number from 1 to 65535.")
    return str(int(text))


def _authority_port(parsed: URL) -> str | None:
    """The port beside a URL host, from the URL part or the query (never both)."""
    port = _one(parsed.port, parsed.query, "port")
    return _port(port) if port is not None else None


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
        pairs.append(("port", _port(query["port"])))
    return pairs


def _check_query_keys(keys: Iterable[str]) -> None:
    allowed = _dsn_keys() | _STARTUP_SETTINGS
    for key in keys:
        if key not in allowed:
            label = (
                f"the URL option {key!r}"
                if key in _KNOWN_REJECTED_KEYS
                else "an unsupported option"
            )
            raise ValueError(
                f"PostgresUpdates cannot use {label}; pass PostgresUpdates(url=...) "
                "with an asyncpg URL (use sslmode=require in url= for TLS)."
            )


def _options(parsed: URL) -> list[tuple[str, str]]:
    _check_query_keys(parsed.query)
    return [
        (key, _single(key, value))
        for key, value in parsed.query.items()
        if key not in _DESTINATION_KEYS
    ]


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


def _check_password_encoding(url: str | URL) -> None:
    if not isinstance(url, str):
        return
    userinfo = re.match(r"^[^:]+://[^:/]*(?::([^@]*))?@", url)
    if userinfo is not None and "?" in (userinfo[1] or ""):
        raise ValueError("PostgresUpdates url=: percent-encode reserved password characters.")


def _checked_parse(url: str | URL) -> URL:
    parsed = _parse(url)
    _check_password_encoding(url)
    if isinstance(url, str):
        _check_query_keys(
            [key for key, _ in parse_qsl(url.partition("?")[2], keep_blank_values=True)]
        )
    return parsed


def asyncpg_dsn(url: str | URL) -> str:
    """Convert a SQLAlchemy PostgreSQL URL to an asyncpg DSN with the same destination.

    Destination fields (user, password, host, port, database) given either in
    the URL parts or as query keys are normalized so asyncpg resolves them
    exactly as SQLAlchemy would; a field given twice is refused. Credentials,
    IPv6 hosts and the database are escaped. Every other query key asyncpg's
    DSN parser accepts is kept; any other key raises ``ValueError`` naming only
    the key, so nothing is silently dropped.
    """
    parsed = _checked_parse(url)
    query = parsed.query
    hosts = _query_hosts(parsed)
    user = _one(parsed.username, query, "user")
    password = _one(parsed.password, query, "password")
    database = _one(parsed.database, query, "database", "dbname")
    port = None if hosts else _authority_port(parsed)
    pairs = _options(parsed)
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
    if host.startswith("/") or host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


_HARMLESS_ENGINE_SETTINGS = frozenset(
    {
        "statement_cache_size",
        "prepared_statement_cache_size",
        "command_timeout",
        "timeout",
        "server_settings",
        "prepared_statement_name_func",
        "max_cached_statement_lifetime",
        "max_cacheable_statement_size",
        "record_class",
        "connect_timeout",
        "prepare_threshold",
        "autocommit",
        "application_name",
        "keepalives_idle",
    }
)
_ENGINE_TLS_SETTINGS = frozenset(
    {
        "ssl",
        "direct_tls",
        "sslmode",
        "sslcert",
        "sslkey",
        "sslrootcert",
        "sslcrl",
        "sslpassword",
    }
)
_BIND_HINT = (
    "PostgresUpdates cannot safely inherit the engine connection setup. "
    "Pass PostgresUpdates(url=...) with the intended destination; "
    "transport connect_args with non-None ssl can override TLS settings only."
)


def _standard_creator(creator: Any) -> bool:
    import inspect

    from sqlalchemy import create_engine

    code = getattr(creator, "__code__", None)
    factory = inspect.unwrap(create_engine)
    return code is not None and any(code is candidate for candidate in factory.__code__.co_consts)


def _engine_parameters(engine: Any) -> Mapping[str, Any] | None:
    creator = engine.pool._creator
    if not _standard_creator(creator):
        return None
    values = dict(
        zip(
            creator.__code__.co_freevars,
            (cell.cell_contents for cell in creator.__closure__),
            strict=True,
        )
    )
    expected_args = engine.dialect.create_connect_args(engine.url)[0]
    actual_args = values.get("cargs_tup", values.get("cargs"))
    if values.get("dialect") is not engine.dialect or actual_args != expected_args:
        return None
    params = values.get("cparams")
    return params if isinstance(params, Mapping) else None


def _standard_dialect(dialect: Any) -> bool:
    from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
    from sqlalchemy.dialects.postgresql.psycopg import PGDialectAsync_psycopg
    from sqlalchemy.engine.default import DefaultDialect

    dialect_type = type(dialect)
    if dialect_type not in (PGDialect_asyncpg, PGDialectAsync_psycopg):
        return False
    methods = (
        dialect.connect.__func__ is DefaultDialect.connect,
        dialect.create_connect_args.__func__ is dialect_type.create_connect_args,
    )
    dbapi_factory: Any = dialect_type
    reference = dbapi_factory.import_dbapi()
    dbapi = dialect.dbapi
    return (
        all(methods)
        and not dialect.dispatch.do_connect
        and type(dbapi) is type(reference)
        and vars(reference).items() <= vars(dbapi).items()
        and dbapi.connect.__func__ is reference.connect.__func__
    )


def _inherits_engine_settings(engine: Any, transport_ssl: bool) -> bool:
    if not _standard_dialect(engine.dialect):
        return False
    params = _engine_parameters(engine)
    if params is None:
        return False
    expected = engine.dialect.create_connect_args(engine.url)[1]
    missing = object()
    changed = {
        key
        for key in params.keys() | expected.keys()
        if params.get(key, missing) != expected.get(key, missing)
    }
    allowed = _HARMLESS_ENGINE_SETTINGS | (_ENGINE_TLS_SETTINGS if transport_ssl else frozenset())
    return changed <= allowed


def _check_engine_settings(engine: Any, *, transport_ssl: bool = False) -> None:
    safe = False
    with suppress(Exception):
        safe = _inherits_engine_settings(engine, transport_ssl)
    if not safe:
        raise ValueError(_BIND_HINT)


def _engine_dsn(engine: Any) -> str:
    url = getattr(engine, "url", None)
    if url is None:
        raise ValueError(
            "PostgresUpdates needs PostgresUpdates(url=...) for a store without an engine."
        )
    parsed = _parse(url)
    host = parsed.host or parsed.query.get("host")
    database = _one(parsed.database, parsed.query, "database", "dbname")
    if not host or not database:
        raise ValueError(
            "PostgresUpdates needs an explicit engine host and database; "
            "pass PostgresUpdates(url=...)."
        )
    return asyncpg_dsn(parsed)


class _ConnectionSettings:
    __slots__ = ("connect_args", "send_dsn", "listen_dsn", "explicit_listen")

    def __init__(
        self,
        url: str | URL | None,
        listen_url: str | URL | None,
        connect_args: Mapping[str, Any] | None,
    ) -> None:
        self.connect_args = _check_connect_args(connect_args)
        self.send_dsn = asyncpg_dsn(url) if url is not None else None
        self.explicit_listen = asyncpg_dsn(listen_url) if listen_url is not None else None
        self.listen_dsn = self.explicit_listen or self.send_dsn

    def __repr__(self) -> str:
        return "<redacted>"

    def __reduce__(self) -> Any:
        raise TypeError("PostgresUpdates connection settings cannot be pickled")


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
        self._settings = _ConnectionSettings(url, listen_url, connect_args)
        self._tls_checked = False
        self._sender: Any = None
        # Silence (health_check_s) plus this probe stays within 2 x health_check_s.
        self._probe_timeout_s = min(self.send_timeout_s * 4, self.health_check_s)
        self._listener: Any = None

    def bind(self, store: object) -> None:
        """Default ``url`` to the store's engine URL; refuse a store without one."""
        if self._settings.send_dsn is not None:
            return
        engine = getattr(store, "_engine", None)
        dsn = _engine_dsn(engine)
        _check_engine_settings(
            engine, transport_ssl=self._settings.connect_args.get("ssl") is not None
        )
        self._settings.send_dsn = dsn
        self._settings.listen_dsn = self._settings.explicit_listen or self._settings.send_dsn

    def _check_tls(self) -> None:
        """Warn once when a remote destination has no explicit TLS setting."""
        if self._tls_checked:
            return
        self._tls_checked = True
        if self._settings.connect_args.get("ssl") is not None:
            return
        for dsn in {self._settings.send_dsn, self._settings.listen_dsn} - {None}:
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
        options = connect_options(
            self._settings.connect_args, timeout=timeout, url_settings=url_settings
        )
        return await asyncpg.connect(dsn, **options)

    async def send(self, messages: Sequence[str]) -> None:
        """One ``pg_notify`` statement per call on the dedicated sender connection."""
        failure = None
        try:
            await self._send_messages(messages)
        except Exception as error:
            failure = _transport_error(error)
        if failure is not None:
            raise failure from None

    async def _send_messages(self, messages: Sequence[str]) -> None:
        try:
            connection = self._sender
            if connection is None or connection.is_closed():
                connection = await self._connect(self._settings.send_dsn, self.send_timeout_s)
                self._sender = connection
            await connection.execute(_NOTIFY_SQL, self.channel, list(messages))
        except BaseException:
            self.abort_sender()
            raise

    def abort_sender(self) -> None:
        """Terminate the sender connection now; the next ``send`` reconnects."""
        connection, self._sender = self._sender, None
        self._terminate_connection(connection)

    async def aclose_sender(self) -> None:
        """Close the sender connection, terminating it if a clean close fails."""
        failure = None
        try:
            await self._close_sender_connection()
        except Exception as error:
            failure = _transport_error(error)
        if failure is not None:
            raise failure from None

    async def _close_sender_connection(self) -> None:
        connection, self._sender = self._sender, None
        if connection is None:
            return
        try:
            if not connection.is_closed():
                await connection.close(timeout=self.send_timeout_s)
        except BaseException:
            self._terminate_connection(connection)
            raise

    def abort_listener(self) -> None:
        """Terminate the listen connection now; ``receive`` then ends."""
        connection, self._listener = self._listener, None
        self._terminate_connection(connection)

    def _terminate_connection(self, connection: Any) -> None:
        if connection is None:
            return
        if self._bound_loop_closed():
            _shutdown_socket(getattr(connection, "_transport", None))
        with suppress(Exception):
            connection.terminate()

    async def receive(self) -> AsyncIterator[str | None | _Gap]:
        """LISTEN on a dedicated connection; ``None`` once LISTEN has executed.

        Liveness: after ``health_check_s`` without a message, ``SELECT 1`` runs
        on this same connection with ``min(4 x send_timeout_s, health_check_s)``
        to answer, so a dead connection is found within ``2 x health_check_s``;
        a failure or a lost connection ends the stream.
        The connection is always terminated when the stream ends.
        """
        failure = None
        try:
            async with aclosing(self._listen_messages()) as stream:
                async for message in stream:
                    yield message
        except Exception as error:
            failure = _transport_error(error)
        if failure is not None:
            raise failure from None

    async def _listen_messages(self) -> AsyncGenerator[str | None | _Gap, None]:
        buffer = self._new_inbound_buffer()
        lost = asyncio.Event()
        connection = None
        try:
            connection = await self._connect(self._settings.listen_dsn, self.send_timeout_s * 4)
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
                self._terminate_connection(connection)

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
