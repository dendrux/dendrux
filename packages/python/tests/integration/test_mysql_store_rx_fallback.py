"""The optional SQL JSON patch path also works on native MySQL/MariaDB."""

# ruff: noqa: F401
from tests.integration._mysql import mysql_engine, store
from tests.integration.test_sql_store_rx_fallback import (
    test_none_patch_hook_preserves_rewrite_and_finalizes,
)
