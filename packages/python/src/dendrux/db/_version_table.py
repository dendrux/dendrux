"""Ownership of Dendrux's migration history, independent of application Alembic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from sqlalchemy import LargeBinary, case, cast, column, func, inspect, select, table, text

from dendrux import _codes
from dendrux.db._dialects import is_mysql_family
from dendrux.db.models import Base

if TYPE_CHECKING:
    from collections.abc import Collection

    from sqlalchemy import Connection

VERSION_TABLE = "dendrux_alembic_version"
LEGACY_VERSION_TABLE = "alembic_version"
_GUIDANCE = (
    "migration_version_table_conflict: Cannot establish Dendrux migration history. "
    "Back up the database and inspect both version tables against the installed "
    "Dendrux and application migration graphs. Preserve application rows; repair "
    "the stamps manually if needed. Run `dendrux db migrate` to adopt verified legacy history."
)
_REASONS = {
    "unknown_revision": (
        "The dedicated history contains a revision this Dendrux version does not know. "
        "Upgrade Dendrux to the version that wrote this database, or restore a compatible backup. "
        "Do not replace its stamps with older revisions."
    ),
    "unstamped_schema": (
        "Dendrux tables are in an unrecognised state with no migration stamp. "
        "Verify their schema and provenance before an explicit `alembic stamp <revision>`, "
        "or restore a known backup or use a separate database."
    ),
    "bootstrap_version": (
        "An unfinished MySQL/MariaDB bootstrap belongs to another Dendrux version. "
        "Finish it with that version, or back up and remove only the partial Dendrux schema "
        "before retrying. Preserve application tables."
    ),
    "mysql_baseline": (
        "This revision is not in Dendrux's supported MySQL/MariaDB history. "
        "Pre-baseline migrations cannot run on these servers. Restore a supported backup "
        "or use a fresh database; do not relabel the old schema with a newer stamp."
    ),
}
logger = logging.getLogger(__name__)


class MigrationVersionTableConflictError(RuntimeError):
    """The database has ambiguous or incomplete Dendrux migration ownership."""

    def __init__(
        self,
        reason: Literal[
            "unknown_revision", "unstamped_schema", "bootstrap_version", "mysql_baseline"
        ]
        | None = None,
    ) -> None:
        guidance = (
            _GUIDANCE if reason is None else "migration_version_table_conflict: " + _REASONS[reason]
        )
        super().__init__(guidance)
        _codes.log_code(logger, _codes.MIGRATION_VERSION_TABLE_CONFLICT)


@dataclass(frozen=True)
class _Stamp:
    rows: int
    known: int
    revision: str | None = None

    @property
    def valid(self) -> bool:
        return self.rows == self.known == 1


def _read_stamp(conn: Connection, name: str, known: Collection[str]) -> _Stamp:
    version = table(name, column("version_num"))
    if is_mysql_family(conn.dialect):
        match = cast(version.c.version_num, LargeBinary).in_([value.encode() for value in known])
    else:
        collation = "BINARY" if conn.dialect.name == "sqlite" else "C"
        match = version.c.version_num.collate(collation).in_(known)
    # Aggregate every row on the server, including duplicates, with bounded client memory.
    row = conn.execute(
        select(
            func.count(),
            func.count(case((match, 1))),
            func.min(case((match, version.c.version_num))),
        ).select_from(version)
    ).one()
    return _Stamp(int(row[0]), int(row[1]), str(row[2]) if row[2] is not None else None)


def _classify(
    dedicated: _Stamp | None,
    legacy: _Stamp | None,
    *,
    has_schema: bool,
    has_agent_runs: bool,
    resumable: bool = False,
) -> Literal["dedicated", "rename", "bootstrap"]:
    legacy_claim = legacy is not None and legacy.known > 0
    if dedicated is not None and dedicated.rows == 0 and not legacy_claim:
        dedicated = None
    if dedicated is not None:
        if dedicated.valid and has_agent_runs and not legacy_claim:
            return "dedicated"
        if dedicated.known != dedicated.rows:
            raise MigrationVersionTableConflictError("unknown_revision")
        raise MigrationVersionTableConflictError
    if legacy_claim:
        if legacy is not None and legacy.valid and has_agent_runs:
            return "rename"
        raise MigrationVersionTableConflictError
    if not has_schema or resumable:
        return "bootstrap"
    raise MigrationVersionTableConflictError("unstamped_schema")


def _inspect_history(
    conn: Connection, known: Collection[str]
) -> tuple[set[str], _Stamp | None, _Stamp | None]:
    inspector = inspect(conn)
    candidates = {*Base.metadata.tables, VERSION_TABLE, LEGACY_VERSION_TABLE}
    names = {name for name in candidates if inspector.has_table(name)}
    dedicated = _read_stamp(conn, VERSION_TABLE, known) if VERSION_TABLE in names else None
    legacy = (
        _read_stamp(conn, LEGACY_VERSION_TABLE, known) if LEGACY_VERSION_TABLE in names else None
    )
    return names, dedicated, legacy


@dataclass(frozen=True)
class MigrationHistory:
    """A known revision and whether its verified legacy table still needs adoption."""

    revision: str | None
    needs_adoption: bool = False


def read_revision(conn: Connection) -> str | None:
    """Read verified history without renaming or stamping it."""
    return read_history(conn).revision


def read_history(conn: Connection) -> MigrationHistory:
    """Read ownership and adoption state without changing either history table."""
    from alembic.script import ScriptDirectory

    script = ScriptDirectory(str(Path(__file__).with_name("migrations")))
    known = {revision.revision for revision in script.walk_revisions()}
    names, dedicated, legacy = _inspect_history(conn, known)
    legacy_claim = legacy is not None and legacy.known > 0
    resumable = False
    if is_mysql_family(conn.dialect):
        from dendrux.db._dialects import mysql

        head = script.get_current_head()
        resumable = head is not None and mysql._resumable_bootstrap(conn, names, head)
    if not legacy_claim and (dedicated is None or dedicated.rows == 0):
        return MigrationHistory(None)
    action = _classify(
        dedicated,
        legacy,
        has_schema=bool(names.intersection(Base.metadata.tables)),
        has_agent_runs="agent_runs" in names,
        resumable=resumable,
    )
    if action == "rename":
        assert legacy is not None
        return MigrationHistory(legacy.revision, needs_adoption=True)
    return MigrationHistory(dedicated.revision if dedicated is not None else None)


def prepare_version_table(
    conn: Connection,
    known: Collection[str],
    *,
    resumable: bool = False,
    explicit_stamp: bool = False,
) -> str:
    """Classify and, if verified, rename history while the caller holds maintenance lock.

    ``explicit_stamp`` permits operator repair of an unstamped schema, never legacy adoption.
    ``resumable`` is only for a recognized MySQL-family bootstrap marker. Renaming
    commits before Alembic starts so a later migration failure can safely be retried.
    """
    names, dedicated, legacy = _inspect_history(conn, known)
    if explicit_stamp and legacy is not None and legacy.known:
        raise MigrationVersionTableConflictError
    action = _classify(
        dedicated,
        legacy,
        has_schema=bool(names.intersection(Base.metadata.tables)) and not explicit_stamp,
        has_agent_runs="agent_runs" in names,
        resumable=resumable,
    )
    if action == "rename":
        _rename(conn)
    conn.commit()
    return action


def _rename(conn: Connection) -> None:
    from dendrux.db._migration_ddl import bounded_step

    if is_mysql_family(conn.dialect):
        statement = f"RENAME TABLE {LEGACY_VERSION_TABLE} TO {VERSION_TABLE}"
    else:
        statement = f"ALTER TABLE {LEGACY_VERSION_TABLE} RENAME TO {VERSION_TABLE}"
    # A rename changes the stamp's location, so use the existing bounded stamp step.
    bounded_step(conn, statement, revision="", step="stamp", transactional=True)
    if conn.dialect.name == "postgresql":
        _rename_postgres_primary_key(conn)


def _rename_postgres_primary_key(conn: Connection) -> None:
    from dendrux.db._migration_ddl import bounded_step

    found = conn.scalar(
        text(
            "SELECT 1 FROM pg_constraint WHERE conrelid = to_regclass(:table) "
            "AND contype = 'p' AND conname = :constraint"
        ),
        {"table": VERSION_TABLE, "constraint": "alembic_version_pkc"},
    )
    if found:
        bounded_step(
            conn,
            f"ALTER TABLE {VERSION_TABLE} RENAME CONSTRAINT alembic_version_pkc "
            "TO dendrux_alembic_version_pkc",
            revision="",
            step="stamp",
            transactional=True,
        )
