"""Tests for the integration conftest's Django-style test DB safeguard.

The conftest rewrites DENDRUX_TEST_PG_URL so the database name always
ends in ``_test``, preventing accidental destruction of dev/prod data.
These tests cover the URL-rewrite helper in isolation; the auto-create
side (``_ensure_test_db_exists``) is exercised by the actual PG matrix.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import _rewrite_to_test_db


class TestRewriteToTestDb:
    def test_appends_test_suffix_when_missing(self):
        url, db_name, rewritten = _rewrite_to_test_db(
            "postgresql+asyncpg://user:pw@host:5432/dendrux"
        )
        assert db_name == "dendrux_test"
        assert url == "postgresql+asyncpg://user:pw@host:5432/dendrux_test"
        assert rewritten is True

    def test_passes_through_when_already_test(self):
        url, db_name, rewritten = _rewrite_to_test_db(
            "postgresql+asyncpg://user:pw@host:5432/dendrux_test"
        )
        assert db_name == "dendrux_test"
        assert url == "postgresql+asyncpg://user:pw@host:5432/dendrux_test"
        assert rewritten is False

    def test_preserves_query_string_and_credentials(self):
        original = "postgresql+asyncpg://u:p@h:5432/myapp?sslmode=require"
        url, db_name, rewritten = _rewrite_to_test_db(original)
        assert db_name == "myapp_test"
        assert "sslmode=require" in url
        assert "u:p@h:5432" in url
        assert rewritten is True

    def test_rejects_url_with_no_db_name(self):
        with pytest.raises(pytest.UsageError, match="no database name"):
            _rewrite_to_test_db("postgresql+asyncpg://user:pw@host:5432/")

    def test_rejects_db_name_with_unsafe_characters(self):
        # Hostile name that survives appending _test would still trip the
        # alphanumeric-only check — DDL identifiers can't be bind-parameterized.
        with pytest.raises(pytest.UsageError, match="outside"):
            _rewrite_to_test_db('postgresql+asyncpg://u:p@h:5432/foo";DROP TABLE users;--')

    def test_arbitrary_app_name_gets_test_suffix(self):
        _, db_name, rewritten = _rewrite_to_test_db("postgresql+asyncpg://u:p@h:5432/excel_agent")
        assert db_name == "excel_agent_test"
        assert rewritten is True
