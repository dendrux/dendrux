"""Declarative, transport-independent MCP source configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MCPFailureMode = Literal["strict", "best_effort"]

_SOURCE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _copy_string_mapping(
    value: Mapping[str, str] | None, field_name: str
) -> Mapping[str, str]:
    copied = dict(value or {})
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in copied.items()):
        raise ValueError(f"MCPSource {field_name} must contain only string keys and values.")
    has_newline = any(
        "\r" in key or "\n" in key or "\r" in item or "\n" in item
        for key, item in copied.items()
    )
    if has_newline:
        raise ValueError(f"MCPSource {field_name} cannot contain newline characters.")
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class MCPSource:
    """Configuration for one remote or subprocess MCP server.

    Prefer :meth:`http` and :meth:`stdio` for construction. This object is
    immutable so it can later be used safely as part of a connection-pool key.
    Authentication objects are passed through to ``httpx2.AsyncClient`` by the
    official MCP SDK adapter.
    """

    name: str
    url: str | None = None
    command: tuple[str, ...] | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    auth: Any = field(default=None, repr=False, compare=False)
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    cwd: Path | None = None
    connect_timeout: float = 30.0
    call_timeout: float = 120.0
    max_result_bytes: int = 1_000_000
    allowed_tools: frozenset[str] | None = None
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
        if self.allowed_tools is not None:
            allowed = frozenset(self.allowed_tools)
            if not all(isinstance(name, str) and name for name in allowed):
                raise ValueError("MCPSource allowed_tools must contain non-empty strings.")
            object.__setattr__(self, "allowed_tools", allowed)

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
        allowed_tools: Sequence[str] | None = None,
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
            allowed_tools=frozenset(allowed_tools) if allowed_tools is not None else None,
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
        allowed_tools: Sequence[str] | None = None,
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
            allowed_tools=frozenset(allowed_tools) if allowed_tools is not None else None,
            failure_mode=failure_mode,
        )

    @property
    def transport(self) -> Literal["http", "stdio"]:
        return "http" if self.url is not None else "stdio"
