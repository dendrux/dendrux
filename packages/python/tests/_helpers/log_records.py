"""Text of a log record for leak scans, without the digits the logging module adds itself."""

from __future__ import annotations

import logging
import re

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_STANDARD = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def record_text(record: logging.LogRecord) -> str:
    """Return everything a record can carry as text: message, string fields and every extra.

    The standard numeric fields (``created``, ``msecs``, ``relativeCreated``, ``thread``,
    ``process``, ``lineno``, ``levelno``) are clocks and counters set by the logging module, never
    caller data, so a short digit sentinel such as a port can match them by chance. Every other
    field is kept, and an extra is kept whatever its type, so a port passed as an int still counts.
    """
    kept = {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD or not isinstance(value, int | float)
    }
    return record.getMessage() + " " + repr(kept)


def without_addresses(text: str) -> str:
    """Drop the memory addresses of default reprs (``<Event at 0x107aa6380>``).

    An address is random hex, so a short digit sentinel can match it by chance. It carries no
    caller data, so removing it hides no leak.
    """
    return _ADDRESS.sub("0x", text)
