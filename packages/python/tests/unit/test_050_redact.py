"""0.5.0: the shared database-URL masking helper (``dendrux._redact``)."""

from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url

from dendrux._redact import UNPARSEABLE_URL, mask_database_url, mask_database_urls_in


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgresql+asyncpg://postgres:root@localhost:5432/dendrux",
            "postgresql+asyncpg://postgres:***@localhost:5432/dendrux",
        ),
        ("postgresql://u:p%40ss@h/db", "postgresql://u:***@h/db"),
        ("postgresql://u@h/db", "postgresql://u@h/db"),
        ("sqlite+aiosqlite:///dendrux.db", "sqlite+aiosqlite:///dendrux.db"),
        ("sqlite+aiosqlite://", "sqlite+aiosqlite://"),
    ],
)
def test_masks_userinfo_password(url: str, expected: str) -> None:
    assert mask_database_url(url) == expected


def test_accepts_url_objects() -> None:
    assert mask_database_url(make_url("postgresql://a:b@h/d")) == "postgresql://a:***@h/d"


@pytest.mark.parametrize("key", ["password", "PASSWORD", "sslpassword", "token", "api_key"])
def test_masks_secret_query_params(key: str) -> None:
    masked = mask_database_url(f"postgresql://u@h/db?{key}=s3cret&sslmode=require")
    assert "s3cret" not in masked
    assert "sslmode=require" in masked
    assert f"{key}=redacted" in masked


@pytest.mark.parametrize("bad", ["not a url", "", "://nohost", "postgresql://u:p@h:notaport/db"])
def test_unparseable_is_never_echoed(bad: str) -> None:
    assert mask_database_url(bad) == UNPARSEABLE_URL


def test_masks_urls_inside_text() -> None:
    text = (
        "Migration failed: could not connect to 'postgresql://u:hunter2@db:5432/x' "
        "and sqlite:///ok.db"
    )
    masked = mask_database_urls_in(text)
    assert "hunter2" not in masked
    assert "postgresql://u:***@db:5432/x" in masked
    assert "sqlite:///ok.db" in masked
    assert masked.startswith("Migration failed: could not connect to '")


def test_text_without_urls_unchanged() -> None:
    assert mask_database_urls_in("plain error, no dsn") == "plain error, no dsn"
