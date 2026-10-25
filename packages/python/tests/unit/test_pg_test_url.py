"""The Postgres test URL is validated with SQLAlchemy semantics before any engine exists."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.engine import make_url

from tests._helpers import xdist_pg
from tests._helpers.pg_url import UnsafeTestDatabaseUrl, maintenance_url, validate_test_pg_url
from tests.integration import _migration_schema
from tests.integration import conftest as integration_conftest

BASE = "postgresql+asyncpg://u:p@h:5432/{}"
OVERRIDE = BASE.format("safe_test") + "?database=production"


def _no_engine(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("an engine was created for an unsafe URL")


class TestValidator:
    def test_database_query_override_is_rejected(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl, match="database"):
            validate_test_pg_url(OVERRIDE)

    @pytest.mark.parametrize("key", ["database", "dbname", "DATABASE"])
    def test_every_override_spelling_is_rejected(self, key: str) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl):
            validate_test_pg_url(BASE.format("safe_test") + f"?{key}=production")

    @pytest.mark.parametrize(
        "query",
        [
            "host=another.invalid",
            "host=another.invalid:6543",
            "host=%2Fprivate%2Ftmp",
            "port=6543",
            "hostaddr=10.0.0.9",
            "unix_sock=%2Ftmp%2F.s.PGSQL.5432",
            "service=prod",
            "HOST=another.invalid",
        ],
    )
    def test_routing_overrides_are_rejected(self, query: str) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl, match="override"):
            validate_test_pg_url(BASE.format("safe_test") + f"?{query}")

    def test_routing_overrides_are_rejected_for_libpq_dialects_too(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl):
            validate_test_pg_url("postgresql+psycopg2://u:p@h/safe_test?host=another.invalid")

    def test_worker_scoping_rejects_routing_overrides(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl):
            xdist_pg.worker_scoped_pg_url(BASE.format("safe_test") + "?host=another.invalid", "gw0")

    def test_effective_database_is_what_the_dialect_would_connect_to(self) -> None:
        url = validate_test_pg_url(BASE.format("safe_test") + "?sslmode=require")
        assert url.database == "safe_test"
        assert url.query == {"sslmode": "require"}

    def test_suffix_is_appended_and_flagged(self) -> None:
        assert validate_test_pg_url(BASE.format("app")).database == "app_test"

    def test_exact_name_is_required_without_the_suffix_rewrite(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl, match="_test"):
            validate_test_pg_url(BASE.format("app"), append_suffix=False)

    def test_non_postgres_url_is_rejected(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl, match="postgres"):
            validate_test_pg_url("sqlite+aiosqlite:///x_test")

    def test_unparseable_url_is_rejected_without_echoing_it(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl) as exc:
            validate_test_pg_url("not a url with secret-pw")
        assert "secret-pw" not in str(exc.value)

    def test_identifier_longer_than_63_is_rejected(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl, match="63"):
            validate_test_pg_url(BASE.format("a" * 59 + "_test"))
        with pytest.raises(UnsafeTestDatabaseUrl, match="63"):
            validate_test_pg_url(BASE.format("a" * 60))  # the appended suffix overflows
        assert validate_test_pg_url(BASE.format("a" * 58 + "_test")).database is not None

    def test_maintenance_url_drops_the_database_and_keeps_options(self) -> None:
        url = validate_test_pg_url(BASE.format("safe_test") + "?sslmode=require")
        admin = maintenance_url(url)
        assert admin.database == "postgres"
        assert admin.query == {"sslmode": "require"}
        assert admin.password == "p"
        assert make_url(admin.render_as_string(hide_password=False)) == admin


class TestEveryEntryPointUsesIt:
    def test_integration_rewrite_rejects_the_override(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl):
            integration_conftest._rewrite_to_test_db(OVERRIDE)

    def test_integration_rewrite_rejects_a_too_long_name(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl):
            integration_conftest._rewrite_to_test_db(BASE.format("a" * 70))

    def test_worker_scoping_rejects_the_override(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl):
            xdist_pg.worker_scoped_pg_url(OVERRIDE, "gw0")
        with pytest.raises(UnsafeTestDatabaseUrl):
            xdist_pg.worker_scoped_pg_url(OVERRIDE, None)

    def test_worker_scoping_rejects_a_name_that_no_longer_fits(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl, match="63"):
            xdist_pg.worker_scoped_pg_url(BASE.format("a" * 55 + "_test"), "gw10")

    def test_worker_id_must_be_a_plain_token(self) -> None:
        with pytest.raises(UnsafeTestDatabaseUrl):
            xdist_pg.worker_scoped_pg_url(BASE.format("x_test"), 'gw0";DROP')

    async def test_migration_schema_validates_before_creating_an_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_migration_schema, "create_async_engine", _no_engine)
        with pytest.raises(UnsafeTestDatabaseUrl):
            async with _migration_schema.migration_schema(OVERRIDE, "s"):
                pass

    async def test_database_bootstrap_validates_before_creating_an_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(integration_conftest, "create_async_engine", _no_engine)
        with pytest.raises(UnsafeTestDatabaseUrl):
            await integration_conftest._ensure_test_db_exists(OVERRIDE, "safe_test")
