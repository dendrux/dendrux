"""Declarative, transport-independent MCP source configuration."""

from __future__ import annotations

import math
import re

# Public class and factory annotations are resolved at runtime by
# documentation and dependency-injection tooling.
from collections.abc import Mapping, Sequence  # noqa: TC003
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote_plus, urlsplit, urlunsplit

from dendrux.mcp._destination import MCPDestinationPolicy  # noqa: TC001

MCPFailureMode = Literal["strict", "best_effort"]
MCPPhysicalIdentity = (
    tuple[Literal["http"], str] | tuple[Literal["stdio"], tuple[str, ...], Path | None]
)

DEFAULT_MAX_DISCOVERY_BYTES = 16 * 1024 * 1024

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


def exception_text(exc: BaseException) -> str:
    """Render an exception, flattening groups to their leaf messages.

    ``str()`` of an exception group is only "unhandled errors in a TaskGroup",
    which hides the transport failure the SDK wrapped inside it.
    """
    if not isinstance(exc, BaseExceptionGroup):
        return str(exc)
    parts: list[str] = []
    for inner in exc.exceptions:
        text = exception_text(inner)
        if isinstance(inner, BaseExceptionGroup):
            parts.append(text)
        else:
            parts.append(f"{type(inner).__name__}: {text}" if text else type(inner).__name__)
    return "; ".join(parts) or str(exc)


def exception_class_name(exc: BaseException) -> str:
    """Name the failure class, looking through single-leaf exception groups."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return type(exc).__name__


def safe_source_exception_detail(source: MCPSource, exc: BaseException) -> str:
    """Return legacy credential-scrubbed text for a source exception.

    This helper does not populate ``MCPConnectionError.transport_detail``;
    that field is a code-and-count summary or ``None`` and never exception text.

    Known credential shapes are scrubbed by exact value. When authentication
    is opaque, no exception text that may have touched it can be proven safe,
    so only an explicit suppression marker is returned.
    """
    if source_has_opaque_credentials(source):
        return "[detail suppressed: opaque auth]"
    return redact_transport_text(source, exception_text(exc)) or exception_class_name(exc)


def redact_transport_text(source: MCPSource, text: str) -> str:
    """Remove URLs as well as configured secrets from transport diagnostics."""
    return re.sub(r"(?i)https?://[^\s<>\"']+", "[redacted URL]", redact_source_text(source, text))


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
    """Recursively redact configured credentials while preserving value shape.

    A ``ToolOutput`` keeps its images (bytes are not text); its ``result`` is
    scrubbed, and a declared image media type that carried a credential is
    dropped (ingest sniffs the real type anyway).
    """
    from dendrux.media._image import Image, ToolOutput

    if isinstance(value, ToolOutput):
        kept: list[Any] = []
        for image in value.images:
            if (
                isinstance(image, Image)
                and image.media_type is not None
                and redact_source_text(source, image.media_type) != image.media_type
            ):
                image = replace(image, media_type=None)
            kept.append(image)
        return ToolOutput(redact_source_value(source, value.result), images=kept)
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

    Discovery is bounded by ``max_discovery_pages``, ``max_discovery_tools`` and
    ``discovery_timeout`` for every source, and by ``max_discovery_bytes``
    (the ``tools/list`` response bodies, capped before they are decoded) for
    HTTP sources only. A stdio server is a local process the developer
    launches, inside the developer's trust boundary; an HTTP server is the
    untrusted case. Setting ``max_discovery_bytes`` on a stdio source raises
    ``ValueError``.

    Replay-safety hints are ignored on every transport by default. Set
    ``trust_replay_hints=True`` only when the source is trusted to label effects
    correctly; then literal read-only or idempotent hints permit automatic replay.
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
    reauthenticate_on_401: bool = False
    follow_redirects: bool = True
    max_redirects: int = 20
    destination_policy: MCPDestinationPolicy | None = field(default=None, repr=False)
    max_discovery_pages: int = 100
    discovery_timeout: float = 60.0
    max_discovery_tools: int = 2000
    max_discovery_bytes: int = DEFAULT_MAX_DISCOVERY_BYTES
    trust_replay_hints: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.trust_replay_hints, bool):
            raise ValueError("MCPSource trust_replay_hints must be a bool.")
        self._validate_name()
        self._validate_transport()
        self._validate_http_settings()
        self._validate_limits()
        self._validate_transport_options()
        object.__setattr__(self, "headers", _copy_string_mapping(self.headers, "headers"))
        object.__setattr__(self, "env", _copy_string_mapping(self.env, "env"))

    def _validate_name(self) -> None:
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

    def _validate_transport(self) -> None:
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

    def _validate_http_settings(self) -> None:
        if not isinstance(self.reauthenticate_on_401, bool):
            raise ValueError("MCPSource reauthenticate_on_401 must be a bool.")
        if not isinstance(self.follow_redirects, bool):
            raise ValueError("MCPSource follow_redirects must be a bool.")
        if type(self.max_redirects) is not int or self.max_redirects < 0:
            raise ValueError("MCPSource max_redirects must be a non-negative integer.")
        if self.destination_policy is not None and not callable(self.destination_policy):
            raise ValueError("MCPSource destination_policy must be callable.")
        if self.command is not None and (
            self.reauthenticate_on_401
            or not self.follow_redirects
            or self.max_redirects != 20
            or self.destination_policy is not None
        ):
            raise ValueError("MCPSource HTTP security settings are only valid for HTTP sources.")

    def _validate_limits(self) -> None:
        if self.connect_timeout <= 0:
            raise ValueError("MCPSource connect_timeout must be greater than zero.")
        if self.call_timeout <= 0:
            raise ValueError("MCPSource call_timeout must be greater than zero.")
        if self.max_result_bytes <= 0:
            raise ValueError("MCPSource max_result_bytes must be greater than zero.")
        self._validate_discovery_bounds()
        if self.failure_mode not in ("strict", "best_effort"):
            raise ValueError("MCPSource failure_mode must be 'strict' or 'best_effort'.")

    def _validate_transport_options(self) -> None:
        if self.command is not None and (self.headers or self.auth is not None):
            raise ValueError("MCPSource headers and auth are only valid for HTTP sources.")
        if self.url is not None and (self.env or self.cwd is not None):
            raise ValueError("MCPSource env and cwd are only valid for stdio sources.")

    def _validate_discovery_bounds(self) -> None:
        if self.command is not None and self.max_discovery_bytes != DEFAULT_MAX_DISCOVERY_BYTES:
            raise ValueError(
                "MCPSource max_discovery_bytes is only valid for HTTP sources: a stdio "
                "server is a local process you launch, so its discovery is bounded by "
                "max_discovery_tools, max_discovery_pages and discovery_timeout instead."
            )
        if type(self.max_discovery_pages) is not int or self.max_discovery_pages < 1:
            raise ValueError("MCPSource max_discovery_pages must be a positive integer.")
        for name in ("max_discovery_tools", "max_discovery_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"MCPSource {name} must be a positive integer.")
        timeout = self.discovery_timeout
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("MCPSource discovery_timeout must be a positive, finite number.")

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
        reauthenticate_on_401: bool = False,
        follow_redirects: bool = True,
        max_redirects: int = 20,
        destination_policy: MCPDestinationPolicy | None = None,
        max_discovery_pages: int = 100,
        discovery_timeout: float = 60.0,
        max_discovery_tools: int = 2000,
        max_discovery_bytes: int = DEFAULT_MAX_DISCOVERY_BYTES,
        trust_replay_hints: bool = False,
    ) -> MCPSource:
        """Configure a production Streamable HTTP MCP source."""
        return cls(
            name=name,
            trust_replay_hints=trust_replay_hints,
            url=url,
            headers=dict(headers or {}),
            auth=auth,
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
            max_result_bytes=max_result_bytes,
            failure_mode=failure_mode,
            reauthenticate_on_401=reauthenticate_on_401,
            follow_redirects=follow_redirects,
            max_redirects=max_redirects,
            destination_policy=destination_policy,
            max_discovery_pages=max_discovery_pages,
            discovery_timeout=discovery_timeout,
            max_discovery_tools=max_discovery_tools,
            max_discovery_bytes=max_discovery_bytes,
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
        max_discovery_pages: int = 100,
        discovery_timeout: float = 60.0,
        max_discovery_tools: int = 2000,
        trust_replay_hints: bool = False,
    ) -> MCPSource:
        """Configure a trusted local subprocess MCP source."""
        if isinstance(command, str):
            raise ValueError("MCPSource command must be a non-empty sequence of strings.")
        return cls(
            name=name,
            trust_replay_hints=trust_replay_hints,
            command=tuple(command),
            env=dict(env or {}),
            cwd=Path(cwd) if cwd is not None else None,
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
            max_result_bytes=max_result_bytes,
            failure_mode=failure_mode,
            max_discovery_pages=max_discovery_pages,
            discovery_timeout=discovery_timeout,
            max_discovery_tools=max_discovery_tools,
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
