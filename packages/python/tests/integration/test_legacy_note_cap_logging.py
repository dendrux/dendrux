"""Custom stores without a note-count bound make the limitation observable."""

from dendrux import Agent


async def test_legacy_note_store_logs_missing_cap(store, caplog):
    await store.create_run("00000000000000000000000303", "a")

    class LegacyNotes:
        async def add_note(self, run_id, note_id, text, *, metadata=None, tenant_id=None):
            return True

        def __getattr__(self, name):
            return getattr(store, name)

    agent = Agent(prompt="p", state_store=LegacyNotes(), strict_tenancy=False)
    try:
        assert await agent.submit_note("00000000000000000000000303", "hello")
        assert any(
            getattr(record, "run_id", None) == "00000000000000000000000303"
            and getattr(record, "code", None) == "note_cap_unenforced"
            for record in caplog.records
        )
    finally:
        await agent.close()
