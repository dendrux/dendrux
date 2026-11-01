"""The MySQL load reset connects to the server first, before its database exists."""

from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.ext import asyncio as sa_asyncio

from tools import load


class _ReachedError(Exception):
    pass


async def test_the_admin_connection_names_no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def fake(url: object, **_: object) -> object:
        seen.append(url)
        raise _ReachedError

    monkeypatch.setattr(sa_asyncio, "create_async_engine", fake)
    url = "mysql+asyncmy://root:dendrux_test@127.0.0.1:53306/dendrux_load_mysql_test"
    assert make_url(url).database == "dendrux_load_mysql_test"
    with pytest.raises(_ReachedError):
        await load.reset_mysql_db(url)
    [admin] = seen
    assert admin.database is None
    assert admin.host == "127.0.0.1"
