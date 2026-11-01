"""Public wiring of the stores shipped in 0.6.0."""

from __future__ import annotations


def test_mysql_state_store_is_exported_from_the_public_store_module() -> None:
    import dendrux.store as public
    from dendrux.runtime.stores.sql.mysql import MySQLStateStore

    assert public.MySQLStateStore is MySQLStateStore
    assert "MySQLStateStore" in public.__all__
