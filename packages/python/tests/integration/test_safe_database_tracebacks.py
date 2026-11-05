"""Normalized SQL-store errors must not retain raw driver errors in internal frame locals."""

import errno
import traceback
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.errors import StoreOutcomeUnknown
from dendrux.runtime.stores.sql._commit import _CommitReadback
from dendrux.types import generate_ulid

SENTINEL = "PRIVATE_FRAME_LOCALS_SENTINEL"


def internal_frame_leaks(error):
    hits = []
    for frame, _lineno in traceback.walk_tb(error.__traceback__):
        if frame.f_globals.get("__name__", "").startswith("dendrux."):
            hits.extend(
                (frame.f_code.co_name, name)
                for name, value in frame.f_locals.items()
                if SENTINEL in repr(value)
            )
    return hits


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("readback_failure", [False, True])
async def test_normalized_unknown_discards_raw_driver_frames(
    store, monkeypatch, caplog, wrapped, readback_failure
):
    run_id = generate_ulid()
    await store.create_run(run_id, "traceback")
    commit = AsyncSession.commit
    commits = 0

    async def lose_ack(session):
        nonlocal commits
        commits += 1
        await commit(session)
        fault = ConnectionResetError(errno.ECONNRESET, SENTINEL)
        if wrapped:
            raise OperationalError("COMMIT", None, fault)
        raise fault

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lose_ack)
        if readback_failure:
            patch.setattr(
                _CommitReadback, "_check_proof", AsyncMock(side_effect=OSError(errno.EIO, SENTINEL))
            )
        with pytest.raises(StoreOutcomeUnknown) as caught:
            if readback_failure:
                await store.save_run_event(run_id, event_type="tick")
            else:
                await store.update_run_meta(run_id, {"landed": True})
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        assert SENTINEL not in "".join(traceback.format_exception(caught.value))
        assert internal_frame_leaks(caught.value) == []
        assert SENTINEL not in repr([vars(record) for record in caplog.records])
    assert commits == 1
