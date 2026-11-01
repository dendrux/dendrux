"""The MySQL work must not change the Postgres or SQLite schema (frozen at b2938693).

``create_all`` is run against a mock engine per dialect and every emitted DDL statement is
compared, whitespace-normalised and order-independent (index creation order follows set
iteration), with the golden files captured from the models at b2938693.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from sqlalchemy import create_mock_engine, event

from dendrux.db import models
from dendrux.db.models import Base

GOLDEN = Path(__file__).parent / "golden"


def _normalise(statement: str) -> str:
    return re.sub(r"\s+", " ", statement).strip()


def create_all_ddl(url: str) -> list[str]:
    """Every DDL statement ``create_all`` emits for the dialect of ``url``."""
    statements: list[str] = []

    def executor(sql, *multiparams, **params):  # noqa: ANN001, ANN002, ANN003
        statements.append(_normalise(str(sql.compile(dialect=engine.dialect))))

    engine = create_mock_engine(url, executor)
    engine.info = {}  # the models' before_create hook records fresh-schema mode here
    # The settings seed is DML on a live connection, not DDL; it is covered elsewhere.
    event.remove(Base.metadata, "after_create", models._seed_storage_settings)
    try:
        Base.metadata.create_all(engine, checkfirst=False)
    finally:
        event.listen(Base.metadata, "after_create", models._seed_storage_settings)
    return statements


def golden(dialect: str) -> list[str]:
    raw = (GOLDEN / f"create_all_{dialect}_b2938693.sql").read_text()
    return [_normalise(part) for part in raw.split(";") if part.strip()]


@pytest.mark.parametrize(
    ("dialect", "url"),
    [("postgresql", "postgresql+asyncpg://"), ("sqlite", "sqlite+aiosqlite://")],
)
def test_create_all_ddl_is_unchanged_since_b2938693(dialect: str, url: str) -> None:
    actual = create_all_ddl(url)
    expected = golden(dialect)
    assert len(actual) == len(expected) == 32
    assert sorted(actual) == sorted(expected)
