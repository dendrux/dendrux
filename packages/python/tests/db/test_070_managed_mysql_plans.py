"""F7: the 0.7 managed-subagent reads on MySQL 8.4 and MariaDB 10.11.

Root ids and tenants compare exactly under the NO PAD binary column collation: case and a
trailing space make different ids, so a live look-alike root never hides a stopped one and
a look-alike tenant never reaches another tenant's root.

Each hot 0.7 read is captured from the real store call on a populated schema (several
thousand runs in managed trees across six tenant scopes, legacy runs, interactions, notes,
intents and events, then ``ANALYZE TABLE``) and its ``EXPLAIN`` must use the intended
index with no full table or full index scan: the root-stop probe, the combined managed
control read, the tree reads (children, cancel children, descendant page, subtree, subtree
usage, delegation ancestry and descendants) and managed bind/admit (fresh and replay).
``-s`` prints every plan. The MySQL-family waiting-sweep keyset page is certified by
``tests/integration/test_managed_sweep.py::test_mysql_waiting_pages_bound_actual_reads``.

Runs only when ``DENDRUX_TEST_MYSQL_URL`` / ``DENDRUX_TEST_MARIADB_URL`` are set. The CI
mysql-family job runs ``tests/db`` and fails when these tests skip there.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import event, insert, text
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._dialects._mysql_types import nopad_collation
from dendrux.db._engine_options import engine_options
from dendrux.db._settings import seed_settings
from dendrux.db.models import AgentRun, Base, LLMInteraction, RunEvent, RunInbox, ToolCallRecord
from dendrux.runtime.state import MANAGED_HEADER_KEY, ManagedHeader
from dendrux.runtime.stores._factory import store_for_engine
from tests.integration.test_managed_bind import bind_request

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

TENANTS: tuple[str | None, ...] = ("t0", "t1", "t2", "t3", "t4", None)
TREES = 20  # per tenant scope
FANOUT = 3
LEVELS = 3  # below the root: 1 + 3 + 9 + 27 = 40 managed runs per tree
LEGACY_RUNS = 1500
TARGET = ("t1", 10)  # mid-way through the id space, so keyset ranges stay wide
STOPPED = ("t1", 3)  # its root is cancel-requested
CHUNK = 500
BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)

# Base tables whose access is gated; a table in neither set is reported as a problem.
GATED = {"agent_runs", "managed_root", "tool_calls", "run_inbox", "llm_interactions", "run_events"}
# Derived tables, CTE work tables and the handful of settings rows are bounded or tiny.
UNGATED = {"descendants", "cancel_children", "dendrux_settings"}
SCANS = {"ALL", "index"}
_NO_ACCESS = re.compile(r"Impossible WHERE|no matching row in const table", re.IGNORECASE)
_LOCK = re.compile(r"\s+(FOR UPDATE|FOR SHARE|LOCK IN SHARE MODE)(\s+(NOWAIT|SKIP LOCKED))?\s*$")

PK = "PRIMARY"
PARENT = "ix_agent_runs_parent_run_id"
INBOX = "ix_run_inbox_run_pending"
USAGE = "ix_llm_interactions_agent_run_id"
INTENT = "uq_tool_calls_run_call"

# The (table alias, index) pairs each call's plans must show, beyond "no full scan".
EXPECTED: dict[str, set[tuple[str, str]]] = {
    "root_stop_live": {("agent_runs", PK), ("managed_root", PK)},
    "root_stop_stopped": {("agent_runs", PK)},
    "control_live": {("agent_runs", PK), ("managed_root", PK), ("run_inbox", INBOX)},
    "control_stopped": {("agent_runs", PK)},
    "children_page": {("agent_runs", PARENT)},
    "children_page_after": {("agent_runs", PARENT)},
    "list_children": {("agent_runs", PARENT)},
    "cancel_children_page": {("agent_runs", PARENT), ("agent_runs", PK)},
    "cancel_children_page_after": {("agent_runs", PARENT), ("agent_runs", PK)},
    "descendant_page": {("agent_runs", PK), ("agent_runs", PARENT)},
    "descendant_page_after": {("agent_runs", PK), ("agent_runs", PARENT)},
    "subtree": {("agent_runs", PK), ("agent_runs", PARENT), ("llm_interactions", USAGE)},
    "subtree_bounded": {("agent_runs", PK), ("agent_runs", PARENT), ("llm_interactions", USAGE)},
    "subtree_usage": {("agent_runs", PARENT), ("llm_interactions", USAGE)},
    "run_usage": {("llm_interactions", USAGE)},
    "delegation_ancestry": {("agent_runs", PK), ("agent_runs", PARENT)},
    "delegation_descendants": {("agent_runs", PK), ("agent_runs", PARENT)},
    "bind": {
        ("agent_runs", PK),
        ("managed_root", PK),
        ("agent_runs", PARENT),
        ("tool_calls", INTENT),
    },
    "bind_replay": {("agent_runs", PK), ("managed_root", PK), ("tool_calls", INTENT)},
}


def _id(seq: int) -> str:
    """A deterministic ULID-shaped id; ids grow with creation order."""
    return f"{seq:026d}"


@dataclass
class Tree:
    root: str
    tenant: str | None
    levels: list[list[str]] = field(default_factory=list)  # levels[0] = [root]


@dataclass
class Seeded:
    target: Tree
    stopped: Tree
    bind_parent: str
    bind_path: tuple[str, ...]
    call_id: str
    live_leaf: str
    stopped_leaf: str
    runs: int = 0


@dataclass
class _Rows:
    runs: list[dict[str, Any]] = field(default_factory=list)
    llm: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    intents: list[dict[str, Any]] = field(default_factory=list)
    seq: int = 0

    def next_id(self) -> str:
        self.seq += 1
        return _id(self.seq)

    def run(self, **values: Any) -> str:
        run_id = self.next_id()
        row = {  # every row carries every key: one executemany per batch
            "id": run_id,
            "agent_name": "worker",
            "tenant_id": None,
            "status": "success",
            "parent_run_id": None,
            "delegation_level": 0,
            "storage_format": 2,
            "claim_epoch": 0,
            "cancel_requested": False,
            "loop_state": None,
            "created_at": BASE_TIME + timedelta(seconds=self.seq),
            "total_input_tokens": 3,
            "total_output_tokens": 2,
            **values,
        }
        self.runs.append(row)
        for i in range(2):
            self.llm.append(
                {
                    "id": self.next_id(),
                    "agent_run_id": run_id,
                    "iteration_index": i,
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "cost_usd": 0.01,
                }
            )
        self.events.append(
            {
                "id": self.next_id(),
                "agent_run_id": run_id,
                "event_type": "run.started",
                "sequence_index": 0,
            }
        )
        if self.seq % 4 == 0:
            self.notes.append(
                {
                    "id": self.next_id(),
                    "run_id": run_id,
                    "text": "note",
                    "delivered_at": BASE_TIME,
                }
            )
        return run_id


def _state(header: ManagedHeader, **extra: Any) -> dict[str, Any]:
    return {
        "v": 1,
        "loop": "react",
        "phase": "dispatch",
        MANAGED_HEADER_KEY: header.to_dict(),
        **extra,
    }


def _tree(rows: _Rows, tenant: str | None, *, stopped: bool) -> Tree:
    root = _id(rows.seq + 1)  # the id rows.run assigns next
    rows.run(
        tenant_id=tenant,
        status="running",
        cancel_requested=stopped,
        loop_state=_state(ManagedHeader(root_run_id=root, path=())),
    )
    tree = Tree(root=root, tenant=tenant, levels=[[root]])
    paths = {root: ()}
    for depth in range(1, LEVELS + 1):
        level = []
        for parent in tree.levels[-1]:
            for k in range(FANOUT):
                call = rows.next_id()
                path = (*paths[parent], f"n{k}")
                child = rows.run(
                    tenant_id=tenant,
                    parent_run_id=parent,
                    delegation_level=depth,
                    status="success" if depth == LEVELS else "paused",
                    loop_state=_state(ManagedHeader(root, path, parent, call)),
                )
                rows.intents.append(
                    {
                        "id": rows.next_id(),
                        "agent_run_id": parent,
                        "tool_call_id": call,
                        "tool_name": "worker",
                        "target": "agent",
                        "outcome": "completed",
                        "meta": {"managed": {"v": 1, "child_run_id": child}},
                    }
                )
                paths[child] = path
                level.append(child)
        tree.levels.append(level)
    # Legacy links into the tree: counted as candidates, never as managed members.
    for parent in tree.levels[1]:
        rows.run(tenant_id=tenant, parent_run_id=parent, delegation_level=2, storage_format=1)
    return tree


def _seed_rows() -> tuple[_Rows, Seeded]:
    rows = _Rows()
    trees: dict[tuple[str | None, int], Tree] = {}
    for tenant in TENANTS:
        for index in range(TREES):
            trees[(tenant, index)] = _tree(rows, tenant, stopped=(tenant, index) == STOPPED)
    for i in range(LEGACY_RUNS):
        rows.run(tenant_id=TENANTS[i % len(TENANTS)], storage_format=1)
    target, stopped = trees[TARGET], trees[STOPPED]
    by_id = {row["id"]: row for row in rows.runs}
    # The bind parent: a running depth-2 node with one pending managed call.
    bind_parent = target.levels[2][4]
    call_id = rows.next_id()
    parent_row = by_id[bind_parent]
    header = ManagedHeader.from_dict(
        parent_row["loop_state"][MANAGED_HEADER_KEY], run_id=bind_parent
    )
    parent_row["status"] = "running"
    parent_row["loop_state"] = _state(
        header,
        submission=None,
        pending=[
            {
                "call_id": call_id,
                "tool": "reviewer",
                "phase": "dispatchable",
                "order_index": 3,
                "target": "agent",
            }
        ],
    )
    live_leaf = target.levels[LEVELS][7]
    by_id[live_leaf]["status"] = "running"
    rows.notes.append(
        {"id": rows.next_id(), "run_id": live_leaf, "text": "pending", "delivered_at": None}
    )
    seeded = Seeded(
        target=target,
        stopped=stopped,
        bind_parent=bind_parent,
        bind_path=(*header.path, "reviewer"),
        call_id=call_id,
        live_leaf=live_leaf,
        stopped_leaf=stopped.levels[LEVELS][5],
        runs=len(rows.runs),
    )
    return rows, seeded


async def seed(engine: Any) -> Seeded:
    """Insert the population (parents before children) and refresh index statistics."""
    rows, seeded = _seed_rows()
    by_depth = sorted(rows.runs, key=lambda row: row["delegation_level"])
    async with engine.begin() as conn:
        for model, batch in (
            (AgentRun, by_depth),
            (LLMInteraction, rows.llm),
            (RunEvent, rows.events),
            (RunInbox, rows.notes),
            (ToolCallRecord, rows.intents),
        ):
            for start in range(0, len(batch), CHUNK):
                await conn.execute(insert(model), batch[start : start + CHUNK])
    if engine.dialect.name in {"mysql", "mariadb"}:
        async with engine.connect() as conn:
            analyzed = text(
                "ANALYZE TABLE agent_runs, llm_interactions, run_events, run_inbox, tool_calls"
            )
            (await conn.execute(analyzed)).all()
    return seeded


@contextmanager
def captured(engine: Any) -> Iterator[list[tuple[str, Any]]]:
    """Statements (with driver parameters) a block sends that EXPLAIN can describe."""
    statements: list[tuple[str, Any]] = []

    def capture(
        conn: Any, cursor: Any, statement: str, params: Any, context: Any, many: Any
    ) -> None:
        # Session-variable probes (connection checks) are not data reads.
        if "@@" in statement:
            return
        if statement.lstrip().upper().startswith(("SELECT", "WITH", "UPDATE")):
            statements.append((statement, params))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        yield statements
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


async def exercise(store: Any, fx: Seeded) -> dict[str, list[tuple[str, Any]]]:
    """Run each hot 0.7 read through the public store call; return its statements by label."""
    engine = store._engine
    target, tenant = fx.target, fx.target.tenant
    level1 = target.levels[1]
    out: dict[str, list[tuple[str, Any]]] = {}
    await store.get_run(fx.target.root)  # connection checks run before any capture

    with captured(engine) as out["root_stop_live"]:
        assert await store.is_managed_root_stopped(fx.live_leaf) is False
    with captured(engine) as out["root_stop_stopped"]:
        assert await store.is_managed_root_stopped(fx.stopped_leaf) is True
    with captured(engine) as out["control_live"]:
        control = await store.get_managed_run_control(fx.live_leaf)
    assert (control.cancel_requested, control.has_pending_notes) == (False, True)
    with captured(engine) as out["control_stopped"]:
        assert (await store.get_managed_run_control(fx.stopped_leaf)).cancel_requested is True

    with captured(engine) as out["children_page"]:
        first = await store.get_children_page([target.root], after=None, limit=2, tenant_id=tenant)
    assert first == sorted(level1)[:2]
    with captured(engine) as out["children_page_after"]:
        rest = await store.get_children_page(
            [target.root], after=first[-1], limit=2, tenant_id=tenant
        )
    assert rest == sorted(level1)[2:]
    with captured(engine) as out["list_children"]:
        listed = await store.list_runs(parent_run_id=target.root, limit=2, tenant_id=tenant)
    assert [row.id for row in listed] == list(reversed(level1))[:2]
    with captured(engine) as out["cancel_children_page"]:
        page = await store.get_managed_cancel_children_page(
            level1, after=None, limit=500, tenant_id=tenant
        )
    level2_and_legacy = [row[0] for row in page]
    assert set(target.levels[2]) < set(level2_and_legacy)
    assert len(level2_and_legacy) == len(target.levels[2]) + len(level1)
    with captured(engine) as out["cancel_children_page_after"]:
        page = await store.get_managed_cancel_children_page(
            level1, after=level2_and_legacy[5], limit=500, tenant_id=tenant
        )
    assert [row[0] for row in page] == level2_and_legacy[6:]

    members = sum(len(level) for level in target.levels) + len(level1)
    with captured(engine) as out["descendant_page"]:
        descendants = await store.get_descendant_page(target.root, limit=20, tenant_id=tenant)
    assert len(descendants) == 20
    with captured(engine) as out["descendant_page_after"]:
        more = await store.get_descendant_page(
            target.root, after=descendants[-1], limit=500, tenant_id=tenant
        )
    assert len(descendants) + len(more) == members - 1
    with captured(engine) as out["subtree"]:
        subtree = await store.get_subtree(target.root, tenant_id=tenant)
    assert subtree is not None and len(subtree.nodes) == members
    with captured(engine) as out["subtree_bounded"]:
        bounded = await store.get_subtree(target.root, tenant_id=tenant, max_nodes=members + 1)
    assert bounded is not None and len(bounded.nodes) == members
    with captured(engine) as out["subtree_usage"]:
        usage = await store.subtree_usage(target.root)
    assert usage.llm_calls == 2 * members
    with captured(engine) as out["run_usage"]:
        assert (await store.run_usage(fx.live_leaf)).llm_calls == 2
    with captured(engine) as out["delegation_ancestry"]:
        info = await store.get_delegation_info(target.levels[2][0])
    assert info is not None and len(info.ancestry) == 2 and info.ancestry_complete
    assert len(info.children) == FANOUT
    with captured(engine) as out["delegation_descendants"]:
        info = await store.get_delegation_info(target.root, max_rows=20)
    assert info is not None and info.truncated

    request = bind_request(
        fx.bind_parent,
        fx.call_id,
        root=target.root,
        tenant=tenant,
        path=fx.bind_path,
    )
    with captured(engine) as out["bind"]:
        assert (await store.bind_managed_child(request)).outcome == "created"
    with captured(engine) as out["bind_replay"]:
        assert (await store.bind_managed_child(request)).outcome == "existing"
    return out


def _base_table(table: str) -> str:
    return re.sub(r"_\d+$", "", table)


def gated_access(plan: list[dict[str, Any]]) -> tuple[set[tuple[str, str]], list[str]]:
    """The (table, index) pairs a plan uses, and its problems (full scans, unknown tables).

    Rows without a table (no table access), derived tables (``<derived2>``,
    ``<union2,3>``) and CTE work tables are bounded intermediates. A table outside
    the gated and ungated sets is a problem, so a new table in a hot read is noticed.
    """
    used: set[tuple[str, str]] = set()
    problems: list[str] = []
    for row in plan:
        table = row.get("table")
        if table is None or table.startswith("<") or table in UNGATED:
            continue
        if row.get("type") is None and _NO_ACCESS.search(str(row.get("Extra") or "")):
            continue  # decided from const tables: nothing is read
        base = _base_table(table)
        if base not in GATED:
            problems.append(f"unexpected table {table}")
        elif row.get("type") in SCANS or not row.get("key"):
            problems.append(f"full scan of {table}: type={row.get('type')} key={row.get('key')}")
        elif base == "managed_root" and int(row.get("rows") or 0) > 1:
            problems.append(f"root probe estimates {row.get('rows')} rows")  # at most one root
        if row.get("key"):
            used.update((base, key) for key in str(row["key"]).split(","))
    return used, problems


async def explain(conn: Any, statement: str, params: Any) -> list[dict[str, Any]]:
    """Traditional EXPLAIN rows (the plan of a locking read is its plain read's plan)."""
    result = await conn.exec_driver_sql("EXPLAIN " + _LOCK.sub("", statement), params)
    keys = ("id", "select_type", "table", "type", "key", "rows", "Extra")
    return [{key: row.get(key) for key in keys} for row in result.mappings().all()]


@pytest.fixture
async def mysql_store(mysql_url: str) -> AsyncIterator[Any]:
    engine = create_async_engine(mysql_url, **engine_options(mysql_url))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lambda sync: seed_settings(sync, active=True))
    store = store_for_engine(engine)
    try:
        yield store
    finally:
        await store.close()
        await engine.dispose()


async def test_managed_reads_use_their_indexes(
    mysql_store: Any, mysql_backend: str, record_property: Any
) -> None:
    store = mysql_store
    fx = await seed(store._engine)
    assert fx.runs > 6000
    calls = await exercise(store, fx)
    assert set(calls) == set(EXPECTED)
    report: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    async with store._engine.connect() as conn:
        for label, statements in calls.items():
            plans: list[dict[str, Any]] = []
            used: set[tuple[str, str]] = set()
            problems: list[str] = [] if statements else ["no statement captured"]
            for sql, params in statements:
                try:
                    plan = await explain(conn, sql, params)
                except Exception as exc:  # noqa: BLE001 - report every label in one CI run
                    problems.append(f"EXPLAIN failed: {type(exc).__name__}: {str(exc)[:300]}")
                    plans.append({"sql": sql, "error": True})
                    continue
                plans.append({"sql": sql, "plan": plan})
                access, found = gated_access(plan)
                used |= access
                problems += found
            missing = sorted(EXPECTED[label] - used)
            if missing:
                problems.append(f"intended index not used: {missing}")
            report[label] = plans
            print(f"[{mysql_backend}] {label}: {json.dumps(plans, default=str)}")
            if problems:
                failures[label] = {"problems": problems, "plans": plans}
    record_property("managed_plans", json.dumps(report, default=str))
    # Every label is checked before failing, so one CI run reports every plan problem.
    assert not failures, json.dumps(failures, default=str, indent=1)


# Root ids and tenants that differ only by case or a trailing space (ids fit VARCHAR(26)).
STOPPED_ROOT = "0F7ROOTSTOPPED00000000001"
LIVE_ROOT = "0F7ROOTLIVE00000000000001"
TENANT = "t1"


def _child(run_id: str, root: str, tenant: str) -> dict[str, Any]:
    header = {
        "v": 1,
        "root_run_id": root,
        "path": ["c"],
        "invocation": {"parent_run_id": LIVE_ROOT, "tool_call_id": "call"},
    }
    return {
        "id": run_id,
        "agent_name": "child",
        "tenant_id": tenant,
        "status": "running",
        "storage_format": 2,
        "loop_state": {"v": 1, "loop": "react", MANAGED_HEADER_KEY: header},
    }


# child id -> (header root, child tenant, expected stopped)
COLLATION_CASES: dict[str, tuple[str, str, bool]] = {
    "0F7CHILDEXACTSTOPPED": (STOPPED_ROOT, TENANT, True),  # live look-alikes must not match
    "0F7CHILDLOWERDECOY": (STOPPED_ROOT.lower(), TENANT, False),  # the exact live row matches
    "0F7CHILDEXACTLIVE": (LIVE_ROOT, TENANT, False),
    "0F7CHILDLOWERLIVE": (LIVE_ROOT.lower(), TENANT, True),
    "0F7CHILDSPACELIVE": (LIVE_ROOT + " ", TENANT, True),
    "0F7CHILDTENANTCASE": (LIVE_ROOT, TENANT.upper(), True),
    "0F7CHILDTENANTSPACE": (LIVE_ROOT, TENANT + " ", True),
}


async def test_root_ids_and_tenants_compare_exactly(mysql_store: Any) -> None:
    store = mysql_store
    engine = store._engine
    expected = nopad_collation(engine.dialect)
    async with engine.connect() as conn:
        collations = dict(
            (
                await conn.execute(
                    text(
                        "SELECT COLUMN_NAME, COLLATION_NAME FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'agent_runs' "
                        "AND COLUMN_NAME IN ('id', 'tenant_id', 'parent_run_id')"
                    )
                )
            ).all()
        )
    assert collations == dict.fromkeys(("id", "tenant_id", "parent_run_id"), expected)
    roots = [
        {"id": STOPPED_ROOT, "cancel_requested": True},
        {"id": STOPPED_ROOT.lower(), "cancel_requested": False},
        {"id": STOPPED_ROOT + " ", "cancel_requested": False},
        {"id": LIVE_ROOT, "cancel_requested": False},
    ]
    async with engine.begin() as conn:
        await conn.execute(
            insert(AgentRun),
            [
                {"agent_name": "root", "tenant_id": TENANT, "status": "running", **root}
                for root in roots
            ],
        )
        await conn.execute(
            insert(AgentRun),
            [_child(run_id, *case[:2]) for run_id, case in COLLATION_CASES.items()],
        )
    async with engine.connect() as conn:
        found = (
            await conn.execute(
                text("SELECT COUNT(*) FROM agent_runs WHERE id = :id"), {"id": STOPPED_ROOT}
            )
        ).scalar_one()
    assert found == 1
    outcomes = {}
    for run_id in COLLATION_CASES:
        control = await store.get_managed_run_control(run_id)
        outcomes[run_id] = (await store.is_managed_root_stopped(run_id), control.cancel_requested)
    assert outcomes == {run_id: (case[2], case[2]) for run_id, case in COLLATION_CASES.items()}
