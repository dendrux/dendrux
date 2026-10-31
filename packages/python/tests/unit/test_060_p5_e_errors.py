"""Connection-reset classification uses the platform's ECONNRESET, never a literal."""

from __future__ import annotations

import errno
from typing import TYPE_CHECKING

from dendrux.runtime.stores._errors import classify_error

if TYPE_CHECKING:
    import pytest

# Values that are no real errno on any supported platform.
_FAKE_RESET = 31001
_FAKE_OTHER = 31002


def _wrapped(os_code: int) -> Exception:
    original = Exception(9999, "driver text")
    original.__cause__ = OSError(os_code)
    return original


def test_connection_reset_follows_the_platform_errno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(errno, "ECONNRESET", _FAKE_RESET)
    assert classify_error(_wrapped(_FAKE_RESET)).kind == "connection_lost"


def test_a_literal_linux_number_is_not_a_reset_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(errno, "ECONNRESET", _FAKE_OTHER)
    assert classify_error(_wrapped(104)).kind == "other"
