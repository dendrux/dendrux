"""Declarative, transport-independent MCP source configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, unquote_plus, urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MCPFailureMode = Literal["strict", "best_effort"]
MCPPhysicalIdentity = (
    tuple[Literal["http"], str] | tuple[Literal["stdio"], tuple[str, ...], Path | None]
)

_SOURCE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

_REDACTED = "[redacted]"


def safe_endpoint(url: str) -> str:
    """Return an endpoint stripped of its query string.

    Presigned URLs carry their credential in the query, so only the scheme,
    host, and path are safe to show. Userinfo cannot appear here: MCPSource
    rejects it at construction.
    """
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _auth_secret_values(auth: Any) -> list[str] | None:
    """Exact secret strings carried by a known auth shape, or None when opaque.

    httpx accepts basic-auth tuples and Auth objects. Tuple elements are
    exact strings the runtime can redact; an Auth object's internals are
    unknowable, so callers must treat any text it may have touched as
    unprovable and suppress rather than redact.
    """
    if auth is None:
        return []
    if isinstance(auth, str):
        return [auth] if auth else []
    if isinstance(auth, (tuple, list)) and all(isinstance(item, str) for item in auth):
        return [item for item in auth if item]
    return None


def source_has_opaque_credentials(source: MCPSource) -> bool:
    """Whether error text from this source could carry unredactable secrets."""
    return _auth_secret_values(source.auth) is None


def safe_source_exception_detail(source: MCPSource, exc: BaseException) -> str:
    """Return credential-safe diagnostic text for a source exception.

    Known credential shapes are scrubbed by exact value. When authentication
    is opaque, no exception text that may have touched it can be proven safe,
    so only an explicit suppression marker is returned.
    """
    if source_has_opaque_credentials(source):
        return "[detail suppressed: opaque auth]"
    return redact_source_text(source, str(exc)) or type(exc).__name__


def redact_source_text(source: MCPSource, text: str) -> str:
    """Strip a source's configured credentials out of third-party text.

    Transport libraries routinely echo the request URL, argv, or header values
    into their exception messages, and that text reaches persisted governance
    events, tool results, and model context. Exact substring replacement is
    used rather than pattern matching because the runtime knows precisely
    which values are secret, so it neither misses one nor guesses.
    """
    if not text:
        return text

    replacements: list[tuple[str, str]] = []
    if source.url:
        parsed_url = urlsplit(source.url)
        endpoint = safe_endpoint(source.url)
        if endpoint != source.url:
            replacements.append((source.url, endpoint))
        query = parsed_url.query
        if query:
            replacements.append((query, _REDACTED))
            # Libraries may normalize percent-encoding or report only a query
            # value rather than the configured URL. Cover both raw and decoded
            # forms independently.
            for component in query.split("&"):
                _, separator, raw_value = component.partition("=")
                if separator and raw_value:
                    replacements.append((raw_value, _REDACTED))
                    replacements.append((unquote_plus(raw_value), _REDACTED))
            replacements.extend(
                (value, _REDACTED) for _, value in parse_qsl(query, keep_blank_values=True) if value
            )
    replacements.extend((value, _REDACTED) for value in source.headers.values() if value)
    for name, value in source.headers.items():
        if name.lower() in {"authorization", "proxy-authorization"}:
            _, separator, credential = value.partition(" ")
            if separator and credential:
                replacements.append((credential, _REDACTED))
    replacements.extend((value, _REDACTED) for value in source.env.values() if value)
    replacements.extend((value, _REDACTED) for value in _auth_secret_values(source.auth) or ())

    # Longest needle first: a full URL must be rewritten before its own query
    # string, or the shorter match would corrupt the replacement.
    for needle, replacement in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        text = text.replace(needle, replacement)
    if source.command:
        for argument in sorted(set(source.command[1:]), key=len, reverse=True):
            if argument:
                text = text.replace(argument, _REDACTED)
    return text


def redact_source_value(source: MCPSource, value: Any) -> Any:
    """Recursively redact configured credentials while preserving value shape."""
    if isinstance(value, str):
        return redact_source_text(source, value)
    if isinstance(value, list):
        return [redact_source_value(source, item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_source_value(source, item) for item in value)
    if isinstance(value, dict):
        return {
            redact_source_text(source, key) if isinstance(key, str) else key: redact_source_value(
                source,
                item,
            )
            for key, item in value.items()
        }
    return value


def _copy_string_mapping(value: Mapping[str, str] | None, field_name: str) -> Mapping[str, str]:
    copied = dict(value or {})
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in copied.items()):
        raise ValueError(f"MCPSource {field_name} must contain only string keys and values.")
    has_newline = any(
        "\r" in key or "\n" in key or "\r" in item or "\n" in item for key, item in copied.items()
    )
    if has_newline:
        raise ValueError(f"MCPSource {field_name} cannot contain newline characters.")
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class MCPSource:
    """Configuration for one remote or subprocess MCP server.

    Prefer :meth:`http` and :meth:`stdio` for construction. The object is
    immutable configuration; use :attr:`physical_identity` when comparing
    connection targets. Authentication objects are passed through to
    ``httpx2.AsyncClient`` by the official MCP SDK adapter.
    """

    name: str
    # Endpoints can contain signed query parameters, and command arguments can
    # contain subprocess credentials. They remain available as configuration
    # but must never appear through the default dataclass representation.
    url: str | None = field(default=None, repr=False)
    command: tuple[str, ...] | None = field(default=None, repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    auth: Any = field(default=None, repr=False, compare=False)
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    cwd: Path | None = None
    connect_timeout: float = 30.0
    call_timeout: float = 120.0
    max_result_bytes: int = 1_000_000
    failure_mode: MCPFailureMode = "strict"

    def __post_init__(self) -> None:
        if not self.name or not _SOURCE_NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"MCPSource name '{self.name}' is not a valid identifier. "
                "Must match [a-zA-Z0-9_-] and be non-empty."
            )
        if "__" in self.name:
            raise ValueError(
                f"MCPSource name '{self.name}' cannot contain '__'. "
                "Double underscore is reserved as the namespace separator."
            )
        if (self.url is None) == (self.command is None):
            raise ValueError(
                "MCPSource requires exactly one transport: "
                "url='...' for HTTP or command=[...] for stdio."
            )
        if self.url is not None and not self.url.strip():
            raise ValueError("MCPSource url must be non-empty.")
        if self.url is not None:
            parsed_url = urlsplit(self.url)
            if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname is None:
                raise ValueError("MCPSource url must be an absolute http:// or https:// URL.")
            if parsed_url.username is not None or parsed_url.password is not None:
                raise ValueError(
                    "MCPSource url must not contain credentials; use headers or auth instead."
                )
        if self.command is not None:
            if not self.command:
                raise ValueError("MCPSource command must be a non-empty sequence of strings.")
            if not all(isinstance(arg, str) for arg in self.command):
                raise ValueError("MCPSource command must contain only strings.")
            if not self.command[0]:
                raise ValueError("MCPSource command executable must be non-empty.")
        if self.connect_timeout <= 0:
            raise ValueError("MCPSource connect_timeout must be greater than zero.")
        if self.call_timeout <= 0:
            raise ValueError("MCPSource call_timeout must be greater than zero.")
        if self.max_result_bytes <= 0:
            raise ValueError("MCPSource max_result_bytes must be greater than zero.")
        if self.failure_mode not in ("strict", "best_effort"):
            raise ValueError("MCPSource failure_mode must be 'strict' or 'best_effort'.")
        if self.command is not None and (self.headers or self.auth is not None):
            raise ValueError("MCPSource headers and auth are only valid for HTTP sources.")
        if self.url is not None and (self.env or self.cwd is not None):
            raise ValueError("MCPSource env and cwd are only valid for stdio sources.")

        object.__setattr__(self, "headers", _copy_string_mapping(self.headers, "headers"))
        object.__setattr__(self, "env", _copy_string_mapping(self.env, "env"))

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        auth: Any = None,
        connect_timeout: float = 30.0,
        call_timeout: float = 120.0,
        max_result_bytes: int = 1_000_000,
        failure_mode: MCPFailureMode = "strict",
    ) -> MCPSource:
        """Configure a production Streamable HTTP MCP source."""
        return cls(
            name=name,
            url=url,
            headers=dict(headers or {}),
            auth=auth,
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
            max_result_bytes=max_result_bytes,
            failure_mode=failure_mode,
        )

    @classmethod
    def stdio(
        cls,
        name: str,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        connect_timeout: float = 30.0,
        call_timeout: float = 120.0,
        max_result_bytes: int = 1_000_000,
        failure_mode: MCPFailureMode = "strict",
    ) -> MCPSource:
        """Configure a trusted local subprocess MCP source."""
        if isinstance(command, str):
            raise ValueError("MCPSource command must be a non-empty sequence of strings.")
        return cls(
            name=name,
            command=tuple(command),
            env=dict(env or {}),
            cwd=Path(cwd) if cwd is not None else None,
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
            max_result_bytes=max_result_bytes,
            failure_mode=failure_mode,
        )

    @property
    def transport(self) -> Literal["http", "stdio"]:
        return "http" if self.url is not None else "stdio"

    @property
    def physical_identity(self) -> MCPPhysicalIdentity:
        """Return the hashable transport target, excluding policy and credentials."""
        if self.url is not None:
            return "http", self.url
        assert self.command is not None
        return "stdio", self.command, self.cwd
