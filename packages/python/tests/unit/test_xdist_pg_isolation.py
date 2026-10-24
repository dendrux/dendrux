"""Each xdist worker gets its own Postgres database so parallel runs cannot collide."""

from __future__ import annotations

import pytest

from tests._helpers.xdist_pg import worker_scoped_pg_url

BASE = "postgresql+asyncpg://u:p@h:5432/{}"


class TestWorkerScopedPgUrl:
    def test_serial_run_keeps_the_url(self) -> None:
        assert worker_scoped_pg_url(BASE.format("dendrux_test"), None) == BASE.format(
            "dendrux_test"
        )

    @pytest.mark.parametrize("worker", ["gw0", "gw7"])
    def test_worker_name_goes_before_the_test_suffix(self, worker: str) -> None:
        got = worker_scoped_pg_url(BASE.format("dendrux_test"), worker)
        assert got == BASE.format(f"dendrux_{worker}_test")

    def test_url_without_suffix_still_ends_in_test(self) -> None:
        got = worker_scoped_pg_url(BASE.format("dendrux"), "gw1")
        assert got == BASE.format("dendrux_gw1_test")

    def test_query_string_survives(self) -> None:
        got = worker_scoped_pg_url(BASE.format("x_test") + "?sslmode=require", "gw2")
        assert got == BASE.format("x_gw2_test") + "?sslmode=require"

    def test_distinct_workers_get_distinct_databases(self) -> None:
        urls = {worker_scoped_pg_url(BASE.format("d_test"), f"gw{i}") for i in range(4)}
        assert len(urls) == 4
