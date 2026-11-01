"""Native MySQL/MariaDB executions of the frozen SQL store contract."""

# ruff: noqa: F401, F811
import pytest
from sqlalchemy import select

from dendrux.db.models import AgentRun
from tests.integration._mysql import mysql_engine, store  # noqa: F401
from tests.integration.test_sql_store_generic import (
    test_append_lost_ack_reads_its_retry_identity,
    test_consumed_evidence_proves_ack_after_later_progress,
    test_create_and_pause_finalize_lost_ack_are_proven,
    test_first_connection_facts_resolve_once,
    test_json_contains_distinguishes_types,
    test_json_hook_truth_table,
    test_json_null_is_explicit,
    test_lost_claim_ack_never_proves_ownership,
    test_uncommitted_transition_retries_without_returning_a_cas_miss,
    test_update_many_exact_bounded_set,
    test_update_one_reads_epoch_by_immutable_key,
    test_update_validation_and_composite_key_bounds,
)
from tests.integration.test_sql_store_generic import (
    test_partial_sweep_records_each_committed_run as _partial_sweep_contract,
)


@pytest.fixture(autouse=True)
def native_commit_error(monkeypatch):
    from tests._helpers import db_errors

    class MySQLLostConnectionError(Exception):
        def __init__(self):
            super().__init__(2013, "PRIVATE-DRIVER-SENTINEL")

    monkeypatch.setattr(db_errors, "LostConnectionError", MySQLLostConnectionError)


# These imported contracts are deliberately collected against local native fixtures.
__all__ = [name for name in globals() if name.startswith("test_")]


@pytest.mark.parametrize("tenant", [None, "t", "t ", "T", "é", "e\u0301"])
async def test_tenant_scope_store_roundtrip(store, tenant):
    from dendrux.types import generate_ulid

    run_id = generate_ulid()
    created = await store.create_run(
        run_id, "agent", tenant_id=tenant, idempotency_key="key", idempotency_fingerprint="digest"
    )
    again = await store.create_run(
        generate_ulid(),
        "agent",
        tenant_id=tenant,
        idempotency_key="key",
        idempotency_fingerprint="digest",
    )
    assert created.run_id == again.run_id == run_id
    async with store._session_factory() as session:
        assert await session.scalar(select(AgentRun.id).where(AgentRun.id == run_id)) == run_id


@pytest.mark.parametrize("value", ["t ", "T", "é", "e\u0301"])
async def test_json_identifier_no_pad(store, value):
    await store.create_run("json-id", "agent", meta={"id": "t"})
    async with store._session_factory() as session:
        result = await session.scalar(
            select(store.json_get(AgentRun.meta, ("id",), "str") == value).where(
                AgentRun.id == "json-id"
            )
        )
    assert result is False


async def test_partial_sweep_records_each_committed_run(store, mysql_engine, monkeypatch):
    await _partial_sweep_contract(store, mysql_engine, monkeypatch)
