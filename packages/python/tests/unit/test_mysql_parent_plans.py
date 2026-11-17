"""Parent-filtered store reads keep MySQL's access inside the parent index."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.dialects.mysql.mariadb import MariaDBDialect

from dendrux.db.models import AgentRun
from dendrux.runtime.state import UNSCOPED
from dendrux.runtime.stores.sql.mysql import MySQLStateStore
from dendrux.runtime.stores.sql.postgres import PostgresStateStore
from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

ROOT = "0" * 26
OPERATIONS = (
    "delegation",
    "descendants_bounded",
    "descendants_unbounded",
    "children",
    "cancel_children",
    "list_children",
    "count_children",
    "list_unfiltered",
)


async def _statements(store, operation, after):
    empty = MagicMock()
    empty.all.return_value = []
    empty.scalar_one.return_value = 0
    empty.scalars.return_value = MagicMock(all=lambda: [])
    empty.scalars.return_value.__iter__.return_value = iter([])
    session = MagicMock()
    session.execute = AsyncMock(return_value=empty)
    session.scalars = AsyncMock(return_value=empty)
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = session
    store._session_factory = factory
    if operation == "delegation":
        root_result = MagicMock()
        root_result.scalar_one_or_none.return_value = AgentRun(
            id=ROOT, agent_name="root", status="running", delegation_level=0
        )
        session.execute.side_effect = [root_result, empty]
        await store.get_delegation_info(ROOT, max_rows=20)
    elif operation.startswith("descendants_"):
        await store._descendant_batch(
            session,
            ROOT,
            [ROOT, "parent2"],
            tenant_id="t1",
            light=True,
            max_nodes=20 if operation == "descendants_bounded" else None,
            loaded=2,
        )
    elif operation in {"children", "cancel_children"}:
        method = (
            store.get_children_page
            if operation == "children"
            else store.get_managed_cancel_children_page
        )
        await method([ROOT, "parent2"], after=after, limit=20, tenant_id="t1")
    elif operation == "count_children":
        await store.count_runs(parent_run_id=ROOT, tenant_id=UNSCOPED)
    else:
        await store.list_runs(
            parent_run_id=ROOT if operation == "list_children" else None,
            tenant_id=UNSCOPED,
            limit=20,
        )
    calls = session.execute.call_args_list + session.scalars.call_args_list
    return [call.args[0] for call in calls]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("after", [None, "cursor"])
@pytest.mark.parametrize(
    "store_type,dialect,hinted",
    [
        (MySQLStateStore, mysql.dialect(), True),
        (MySQLStateStore, MariaDBDialect(), False),
        (PostgresStateStore, postgresql.dialect(), False),
        (SQLiteStateStore, sqlite.dialect(), False),
    ],
    ids=["mysql", "mariadb", "postgres", "sqlite"],
)
async def test_parent_filtered_reads_preserve_sql_except_mysql_access_hint(
    operation, after, store_type, dialect, hinted
):
    statements = await _statements(object.__new__(store_type), operation, after)
    baseline_store = object.__new__(store_type)
    baseline_store._parent_run_query = lambda statement: statement
    baseline = await _statements(baseline_store, operation, after)
    assert len(statements) == len(baseline)
    hint = " FORCE INDEX (ix_agent_runs_parent_run_id)"
    for statement, original in zip(statements, baseline, strict=True):
        compiled = statement.compile(dialect=dialect)
        expected = original.compile(dialect=dialect)
        is_parent_read = "parent_run_id IN" in str(expected) or operation in {
            "list_children",
            "count_children",
        }
        assert (hint in str(compiled)) is (hinted and is_parent_read)
        assert str(compiled).replace(hint, "") == str(expected)
        assert compiled.params == expected.params
