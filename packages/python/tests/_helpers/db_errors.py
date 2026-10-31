"""Structured driver faults for transaction retry tests."""

import sqlite3


def locked_error() -> sqlite3.OperationalError:
    """A SQLite BUSY error with the metadata a real driver supplies."""
    error = sqlite3.OperationalError("database is locked")
    error.sqlite_errorcode = sqlite3.SQLITE_BUSY
    return error


class LostConnectionError(Exception):
    """A lost commit acknowledgement, never a retryable lock failure."""

    sqlstate = "08006"
