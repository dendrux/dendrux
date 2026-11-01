"""The backend fixture restores active settings after cleaning model tables."""

from sqlalchemy import text


async def test_clean_integration_schema_is_active(engine):
    async with engine.connect() as conn:
        settings = dict(
            (
                await conn.execute(
                    text(
                        "SELECT key, value FROM dendrux_settings "
                        "WHERE key IN ('storage_format', 'activate_step')"
                    )
                )
            ).all()
        )
    assert settings == {"storage_format": "2", "activate_step": "done"}
