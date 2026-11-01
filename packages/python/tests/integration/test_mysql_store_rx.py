"""Run shared RX transaction regressions on MySQL and MariaDB."""

# ruff: noqa: F401, F811
import pytest

from tests.integration._mysql import mysql_engine, store
from tests.integration.test_sql_store_rx import (
    test_commit_identity_resolves_or_retries,
    test_flush_failure_retries_whole_transaction,
    test_inflight_commit_readback_waits_for_writer,
    test_inflight_create_reproves_identity,
    test_keyed_create_normalizes_legacy_none_fingerprint,
    test_keyed_create_normalizes_none_fingerprint,
    test_locked_validation_callback_preserves_attribute_error,
    test_precommit_heartbeat_retries,
    test_readback_lock_timeout_is_unknown,
)


@pytest.fixture
def backend_engine(mysql_engine):
    return mysql_engine
