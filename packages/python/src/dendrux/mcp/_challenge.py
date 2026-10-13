"""Strict ``WWW-Authenticate`` parsing into a value-safe ``MCPAuthChallenge``.

The header is server-authored. A hostile or buggy server can echo a token
into any parameter, so only the ``Bearer`` challenge is read and every field
is validated against a narrow grammar before it is kept. Free text such as
``error_description`` and ``realm`` is never stored.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from dendrux.mcp._errors import MCPAuthChallenge

if TYPE_CHECKING:
    from collections.abc import Iterable

# RFC 9110 token: auth-scheme and auth-param names.
_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_SCHEME = re.compile(r"[A-Za-z0-9._~+/-]{1,32}")
_ERROR = re.compile(r"[A-Za-z0-9_]{1,64}")
# RFC 6749 section 3.3 scope-token: %x21 / %x23-5B / %x5D-7E.
_SCOPE_TOKEN = re.compile(r"[\x21\x23-\x5b\x5d-\x7e]{1,128}")
_MAX_SCOPES = 32
_MAX_URL_LENGTH = 2048
# Bounds parsing work on an oversized header; longer values are ignored.
_MAX_HEADER_LENGTH = 16384

_Challenge = tuple[str, dict[str, str]]


def _quoted(value: str, start: int) -> tuple[str | None, int]:
    """Read a quoted-string at ``start``; None when it is unterminated."""
    out: list[str] = []
    index = start + 1
    end = len(value)
    while index < end:
        char = value[index]
        if char == "\\":
            if index + 1 >= end:
                return None, end
            out.append(value[index + 1])
            index += 2
            continue
        if char == '"':
            return "".join(out), index + 1
        out.append(char)
        index += 1
    return None, end


def _skip_element(value: str, index: int) -> int:
    """Advance to the next top-level comma, stepping over quoted strings."""
    end = len(value)
    while index < end and value[index] != ",":
        if value[index] == '"':
            _, index = _quoted(value, index)
            continue
        index += 1
    return index


def _skip_space(value: str, index: int) -> int:
    while index < len(value) and value[index] in " \t":
        index += 1
    return index


def _parse_challenges(value: str) -> list[_Challenge]:
    """Split one header value into ``(scheme, params)`` challenges.

    Parameter names are lower-cased; the first occurrence of a name wins.
    token68 credentials and anything unparseable are skipped.
    """
    challenges: list[_Challenge] = []
    current: _Challenge | None = None
    index = 0
    end = len(value)
    while index < end:
        if value[index] in " \t,":
            index += 1
            continue
        match = _TOKEN.match(value, index)
        if match is None:
            index = _skip_element(value, index)
            continue
        name = match.group()
        after = _skip_space(value, match.end())
        if after < end and value[after] == "=":
            after = _skip_space(value, after + 1)
            param: str | None = None
            if after < end and value[after] == '"':
                param, after = _quoted(value, after)
            else:
                token = _TOKEN.match(value, after)
                if token is not None:
                    param, after = token.group(), token.end()
            if param is None:
                index = _skip_element(value, after)
                continue
            if current is not None:
                current[1].setdefault(name.lower(), param)
            index = after
            continue
        current = (name, {})
        challenges.append(current)
        index = match.end()
    return challenges


def _safe_url(value: str | None) -> str | None:
    if value is None or len(value) > _MAX_URL_LENGTH:
        return None
    if any(ord(char) <= 0x20 or ord(char) >= 0x7F for char in value):
        return None
    if "#" in value or "\\" in value:
        return None
    try:
        parts = urlsplit(value)
        host = parts.hostname
        _ = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not host or "@" in parts.netloc:
        return None
    return value


def _safe_scopes(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [item for item in value.split(" ") if _SCOPE_TOKEN.fullmatch(item)]
    return tuple(items[:_MAX_SCOPES])


def _matches(pattern: re.Pattern[str], value: str | None) -> str | None:
    return value if value is not None and pattern.fullmatch(value) else None


def challenge_from_headers(values: Iterable[str]) -> MCPAuthChallenge:
    """Build a validated challenge from every ``WWW-Authenticate`` value.

    The first ``Bearer`` challenge (case-insensitive) across all values is
    used; other challenges are ignored. Without one, only the first
    challenge's scheme is kept. Invalid fields become empty, never errors.
    """
    challenges: list[_Challenge] = []
    for value in values:
        if len(value) <= _MAX_HEADER_LENGTH:
            challenges.extend(_parse_challenges(value))
    bearer = next((c for c in challenges if c[0].lower() == "bearer"), None)
    if bearer is None:
        scheme = challenges[0][0] if challenges else None
        return MCPAuthChallenge(scheme=_matches(_SCHEME, scheme))
    scheme, params = bearer
    return MCPAuthChallenge(
        scheme=_matches(_SCHEME, scheme),
        error=_matches(_ERROR, params.get("error")),
        scope=_safe_scopes(params.get("scope")),
        resource_metadata_url=_safe_url(params.get("resource_metadata")),
    )


def parse_auth_challenge(response: Any) -> MCPAuthChallenge | None:
    """Parse a 401/403 response's ``WWW-Authenticate`` headers.

    Returns None when the response carries no such header. Parsing never
    raises: the challenge is diagnostic and must never turn a rejection into
    a different failure.
    """
    try:
        headers = response.headers
        get_list = getattr(headers, "get_list", None)
        raw = (
            get_list("www-authenticate")
            if callable(get_list)
            else [headers.get("www-authenticate")]
        )
        values = [value for value in raw if isinstance(value, str) and value]
    except Exception:
        return None
    if not values:
        return None
    try:
        return challenge_from_headers(values)
    except Exception:
        return MCPAuthChallenge()
