"""Activation checks history ownership before examining the schema."""

from unittest.mock import MagicMock

import pytest

from dendrux.db import _activate, _version_table
from tests.migrations.test_060_activate import HEAD, PRIOR


@pytest.mark.parametrize("revision,adoption", [(HEAD, True), (PRIOR, False), (None, True)])
def test_activation_refuses_noncurrent_history(monkeypatch, revision, adoption):
    conn = MagicMock()
    monkeypatch.setattr(
        _version_table,
        "read_history",
        lambda c: _version_table.MigrationHistory(revision, adoption),
    )
    schema = MagicMock()
    monkeypatch.setattr(_activate, "schema_problems", schema)
    with pytest.raises(RuntimeError, match="Run `dendrux db migrate` first"):
        _activate._require_current(conn)
    schema.assert_not_called()
    conn.commit.assert_not_called()


@pytest.mark.parametrize("revision", [HEAD, None])
@pytest.mark.parametrize("problems", [[], ["missing index", "missing column"]])
def test_activation_checks_schema_after_current_or_unstamped_history(
    monkeypatch, revision, problems
):
    conn = MagicMock()
    monkeypatch.setattr(
        _version_table, "read_history", lambda c: _version_table.MigrationHistory(revision)
    )
    schema = MagicMock(return_value=problems)
    monkeypatch.setattr(_activate, "schema_problems", schema)
    if problems:
        with pytest.raises(RuntimeError) as caught:
            _activate._require_current(conn)
        assert str(caught.value) == "Database check is not current: missing index, missing column"
    else:
        assert _activate._require_current(conn) is None
    schema.assert_called_once_with(conn)
    conn.commit.assert_not_called()
