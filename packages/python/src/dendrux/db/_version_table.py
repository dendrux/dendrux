"""Ownership of Dendrux's migration history, independent of application Alembic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from sqlalchemy import LargeBinary, case, cast, column, func, inspect, select, table
from sqlalchemy.exc import DBAPIError

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
    "the stamps manually before retrying `dendrux db migrate`."
)
logger = logging.getLogger(__name__)


class MigrationVersionTableConflictError(RuntimeError):
    """The database has ambiguous or incomplete Dendrux migration ownership."""

    def __init__(self) -> None:
        super().__init__(_GUIDANCE)
        _codes.log_code(logger, _codes.MIGRATION_VERSION_TABLE_CONFLICT)


@dataclass(frozen=True)
class _Stamp:
    rows: int
    known: int

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
        ).select_from(version)
    ).one()
    return _Stamp(int(row[0]), int(row[1]))


def _classify(
    dedicated: _Stamp | None,
    legacy: _Stamp | None,
    *,
    has_schema: bool,
    has_agent_runs: bool,
    resumable: bool = False,
) -> Literal["dedicated", "rename", "bootstrap"]:
    legacy_claim = legacy is not None and legacy.known > 0
    if dedicated is not None:
        if dedicated.valid and has_agent_runs and not legacy_claim:
            return "dedicated"
        if resumable and dedicated.rows == 0 and not legacy_claim:
            return "bootstrap"
        raise MigrationVersionTableConflictError
    if legacy_claim:
        if legacy is not None and legacy.valid and has_agent_runs:
            return "rename"
        raise MigrationVersionTableConflictError
    if not has_schema or resumable:
        return "bootstrap"
    raise MigrationVersionTableConflictError


def prepare_version_table(
    conn: Connection, known: Collection[str], *, resumable: bool = False
) -> str:
    """Classify and, if verified, rename history while the caller holds maintenance lock.

    ``resumable`` is only for a recognized MySQL-family bootstrap marker. Renaming
    commits before Alembic starts so a later migration failure can safely be retried.
    """
    inspector = inspect(conn)
    candidates = {*Base.metadata.tables, VERSION_TABLE, LEGACY_VERSION_TABLE}
    names = {name for name in candidates if inspector.has_table(name)}
    try:
        dedicated = _read_stamp(conn, VERSION_TABLE, known) if VERSION_TABLE in names else None
        legacy = (
            _read_stamp(conn, LEGACY_VERSION_TABLE, known)
            if LEGACY_VERSION_TABLE in names
            else None
        )
    except DBAPIError:
        raise MigrationVersionTableConflictError from None
    action = _classify(
        dedicated,
        legacy,
        has_schema=bool(names.intersection(Base.metadata.tables)),
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
