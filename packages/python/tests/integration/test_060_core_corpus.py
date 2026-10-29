"""A7: format-2 continuation preserves every recorded provider conversation."""

import pytest
from sqlalchemy import update

from dendrux.db.models import DendruxSetting
from tests.conformance import harness as h
from tests.conformance.scenarios import CELLS


@pytest.mark.parametrize("cell", CELLS, ids=[c.id for c in CELLS])
async def test_a7_format2_provider_corpus(cell, store, engine, monkeypatch):
    async with engine.begin() as conn:
        await conn.execute(
            update(DendruxSetting).where(DendruxSetting.key == "storage_format").values(value="2")
        )
    from dendrux.runtime.replay import rebuild_history

    def provider_view(message):
        # The only projections allowed by contracts §12.28.
        view = h.message_view(message)
        for tc, value in zip(message.tool_calls or [], view["tool_calls"] or [], strict=True):
            raw = value.pop("raw_arguments")
            value["raw_length"] = len(raw) if raw is not None else getattr(tc, "raw_length", None)
        for value in view["parts"] or []:
            if value["type"] == "image":
                ref = value["ref"]
                value.clear()
                value.update(ref=ref, media_type=ref["media_type"], byte_length=ref["bytes"])
        return view

    original = h.RecordingProvider.complete
    comparisons = []

    async def checked(provider, messages, tools=None, **kwargs):
        run_id = kwargs.get("run_id")
        record = await store.get_run(run_id) if run_id else None
        if record is not None and record.storage_format == 2:
            rebuilt = await rebuild_history(store, run_id, reasoning_for=provider.model)
            comparisons.append(
                h.diff(
                    [provider_view(m) for m in messages],
                    [provider_view(m) for m in rebuilt.messages],
                )
            )
        return await original(provider, messages, tools, **kwargs)

    monkeypatch.setattr(h.RecordingProvider, "complete", checked)
    recording = await h.run_cell(cell, engine, store)
    assert all(d is None for d in comparisons), comparisons
    # §12.28 uses the live format-2 provider boundary as the oracle. The
    # unchanged format-1 corpus/goldens remain checked by conformance.
    assert len(comparisons) == sum(call["run_id"] is not None for call in recording["calls"])
    assert len(recording["calls"]) == len(h.read_json(h.corpus_path(cell.id))["calls"])
