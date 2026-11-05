"""Readback must retain fence evidence and the proven-absent retry signal."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Column, Integer, MetaData, Table

from dendrux._meta_keys import COMMIT_PROOFS_SESSION_KEY
from dendrux.db.models import AgentRun
from dendrux.errors import StoreOutcomeUnknown
from dendrux.runtime.durability import _CommitNotAppliedError
from dendrux.runtime.stores.sql._commit import _CommitReadback
from dendrux.types import generate_ulid


@pytest.mark.parametrize("mode", ["fenced", "missing-fence", "proven-absent"])
async def test_readback_requires_parent_fence_and_preserves_absent_signal(mode):
    run_id = generate_ulid()
    table = AgentRun.__table__
    if mode == "missing-fence":
        table = Table("other_proof", MetaData(), Column("id", Integer))
    predicate = table.c.id == run_id
    session = SimpleNamespace(
        info={COMMIT_PROOFS_SESSION_KEY: [(table, predicate, {"status": "success"})]}
    )
    store = SimpleNamespace(_prepare_commit_readback=AsyncMock())
    proof = _CommitReadback(
        store, session, (store, "finalize_run", {"run_id": run_id, "claim_epoch": 7})
    )
    proof._lock_touched = AsyncMock()
    proof._check_proof = AsyncMock()
    absent = _CommitNotAppliedError()
    if mode == "proven-absent":
        proof._check_proof.side_effect = absent
        with pytest.raises(_CommitNotAppliedError) as caught:
            await proof._prove()
        assert caught.value is absent
    elif mode == "missing-fence":
        with pytest.raises(StoreOutcomeUnknown):
            await proof._prove()
    else:
        await proof._prove()
        assert proof._check_proof.await_args.args[2] == {"status": "success", "claim_epoch": 7}
