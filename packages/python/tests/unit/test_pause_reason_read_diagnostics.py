"""Pause reason lookup remains best effort without reflecting custom-store SQL."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import StatementError

from dendrux.runtime.runner import _pause_request_message
from dendrux.runtime.state import PAUSE_REQUEST_META_KEY


async def test_custom_store_pause_reason_fault_is_safe_and_does_not_block_pause(caplog):
    secret = "PRIVATE_PAUSE_READ_BIND"
    store = SimpleNamespace(
        get_run=AsyncMock(
            side_effect=StatementError(
                secret, "SELECT " + secret, {"secret": secret}, RuntimeError(secret)
            )
        )
    )
    assert await _pause_request_message(store, "r") is None
    assert "Failed to read pause request" in caplog.text
    assert secret not in caplog.text and "SELECT" not in caplog.text


@pytest.mark.parametrize(
    "record,expected",
    [
        (None, None),
        (SimpleNamespace(meta=None), None),
        (SimpleNamespace(meta={PAUSE_REQUEST_META_KEY: "malformed"}), None),
        (SimpleNamespace(meta={PAUSE_REQUEST_META_KEY: {"reason": 17}}), None),
        (
            SimpleNamespace(meta={PAUSE_REQUEST_META_KEY: {"reason": "owner request"}}),
            "owner request",
        ),
    ],
)
async def test_pause_reason_reader_handles_missing_legacy_shapes(record, expected):
    store = SimpleNamespace(get_run=AsyncMock(return_value=record))
    assert await _pause_request_message(store, "r") == expected
