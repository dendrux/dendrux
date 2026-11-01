"""Schema checks must include behavioral foreign keys and text collations."""

import pytest
from sqlalchemy import text

from dendrux.db._schema_check import ensure_schema_current
from dendrux.errors import SchemaOutOfDateError
from tests.db.test_060_p5_mysql_check import _create_all, _problems
from tests.db.test_060_p5_mysql_ddl import COLLATIONS, _column_sql


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_longtext_uses_binary_nopad_collation(server):
    spec = _column_sql(server, "tool_calls")["error_message"]
    assert f"CHARACTER SET utf8mb4 COLLATE {COLLATIONS[server]}" in spec


async def test_additional_foreign_key_is_reported(mysql_url):
    engine = await _create_all(mysql_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE app_parent (id VARCHAR(255) CHARACTER SET utf8mb4 "
                    "COLLATE "
                    + ("utf8mb4_nopad_bin" if engine.dialect.is_mariadb else "utf8mb4_0900_bin")
                    + " PRIMARY KEY) ENGINE=InnoDB"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE agent_runs ADD CONSTRAINT fk_unexpected "
                    "FOREIGN KEY (agent_name) REFERENCES app_parent(id) ON DELETE CASCADE"
                )
            )
        assert any("unexpected foreign key" in problem for problem in await _problems(engine))
        with pytest.raises(SchemaOutOfDateError, match="unexpected foreign key"):
            await ensure_schema_current(engine)
    finally:
        await engine.dispose()


async def test_old_text_collation_is_reported(mysql_url):
    engine = await _create_all(mysql_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "ALTER TABLE tool_calls MODIFY error_message LONGTEXT "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"
                )
            )
        assert any("error_message (collation" in problem for problem in await _problems(engine))
        with pytest.raises(SchemaOutOfDateError, match="collation"):
            await ensure_schema_current(engine)
    finally:
        await engine.dispose()
