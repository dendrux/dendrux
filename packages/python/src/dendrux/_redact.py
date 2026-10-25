"""Masking database URLs for logs, errors and CLI output (internal, shared).

One implementation for every place that prints or logs a database URL (the
engine/session layer, ``dendrux db``, the CLI), so the edge cases
are handled once: passwords in the userinfo or the query string, and
unparseable input.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sqlalchemy.engine import make_url

if TYPE_CHECKING:
    from types import TracebackType

    from sqlalchemy.engine import URL

__all__ = ["mask_database_url", "mask_database_urls_in", "safe_exception_info"]

UNPARSEABLE_URL = "<unparseable database url>"

# The userinfo password renders as ``***``; ``*`` would be percent-encoded in a query.
_SECRET_QUERY_VALUE = "redacted"
_SECRET_QUERY_KEYS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "sslpassword",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "api_key",
        "apikey",
    }
)
# ``scheme://...`` up to whitespace or a quote/bracket that commonly ends it.
_URL_IN_TEXT = re.compile(r"\b[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>]+")


def mask_database_url(url: str | URL) -> str:
    """``url`` with its password shown as ``***`` and secret query values as ``redacted``.

    Never raises: input that does not parse as a URL is returned as
    ``"<unparseable database url>"``, never echoed.
    """
    try:
        parsed = url if not isinstance(url, str) else make_url(url)
        query = {
            key: (_SECRET_QUERY_VALUE if key.lower() in _SECRET_QUERY_KEYS else value)
            for key, value in parsed.query.items()
        }
        return str(parsed.set(query=query).render_as_string(hide_password=True))
    except Exception:
        return UNPARSEABLE_URL


def mask_database_urls_in(text: str) -> str:
    """``text`` with every ``scheme://...`` substring masked by :func:`mask_database_url`.

    For messages that may embed a DSN, such as a driver error or an
    exception ``repr``.
    """
    return _URL_IN_TEXT.sub(lambda m: mask_database_url(m.group(0)), text)


def safe_exception_info(
    error: BaseException | None = None,
) -> bool | tuple[type[BaseException], BaseException, TracebackType | None]:
    """Keep ordinary stacks and secondary failures without provider chains.

    The original exception and its chain remain untouched for callers. Custom
    provider errors must use ProviderError to identify vendor-bearing chains.
    """
    import copy
    import sys

    from dendrux.errors import ProviderError

    error = error if error is not None else sys.exception()
    current = error
    seen: set[int] = set()
    for _ in range(32):
        if current is None:
            return True
        if isinstance(current, ProviderError) or id(current) in seen:
            break
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    if error is None or isinstance(error, ProviderError):
        return False
    try:
        own = copy.copy(error)
        if not isinstance(own, BaseException) or own is error:
            raise TypeError("exception copy retained its original identity")
    except Exception:
        own = RuntimeError(f"{type(error).__name__}: {error}")
    BaseException.__setattr__(own, "__cause__", None)
    BaseException.__setattr__(own, "__context__", None)
    BaseException.__setattr__(own, "__suppress_context__", True)
    BaseException.__setattr__(own, "__traceback__", error.__traceback__)
    return type(own), own, own.__traceback__
