"""Image blobs: the dendrux_blobs and dendrux_blob_refs tables.

``dendrux_blobs`` holds one row per ``(tenant_scope, sha256)``: metadata, the
lifecycle ``state`` and ``generation``, and the bytes when stored in the
database. ``dendrux_blob_refs`` records which run record references which
blob (GC reachability); edges go with their run (``ON DELETE CASCADE``) and
keep their blob (``ON DELETE RESTRICT``). Purely additive: no existing table
or row changes.

Revision ID: c6d7e8f9a0b1
Revises: b5c6d7e8f9a0
Create Date: 2026-10-17
"""

from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

revision: str = "c6d7e8f9a0b1"
down_revision: Union[str, None] = "b5c6d7e8f9a0"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


def upgrade() -> None:
    op.create_table(
        "dendrux_blobs",
        sa.Column("tenant_scope", sa.String(260), nullable=False),
        sa.Column("sha256", sa.CHAR(64), nullable=False),
        sa.Column("generation", sa.String(26), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("media_type", sa.String(32), nullable=False),
        sa.Column("byte_length", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("storage", sa.String(16), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=True),
        sa.Column("ingest_transform", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state_changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("purged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("purge_reason", sa.String(200), nullable=True),
        sa.PrimaryKeyConstraint("tenant_scope", "sha256"),
    )
    op.create_index("ix_dendrux_blobs_gc", "dendrux_blobs", ["state", "last_used_at"])
    op.create_index("ix_dendrux_blobs_work", "dendrux_blobs", ["state", "state_changed_at"])

    op.create_table(
        "dendrux_blob_refs",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("tenant_scope", sa.String(260), nullable=False),
        sa.Column("sha256", sa.CHAR(64), nullable=False),
        sa.Column(
            "run_id",
            sa.String(26),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("owner_kind", sa.String(32), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_scope", "sha256"],
            ["dendrux_blobs.tenant_scope", "dendrux_blobs.sha256"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "run_id", "owner_kind", "owner_id", "ordinal", name="uq_dendrux_blob_refs_owner"
        ),
    )
    op.create_index(
        "ix_dendrux_blob_refs_blob", "dendrux_blob_refs", ["tenant_scope", "sha256"]
    )
    op.create_index("ix_dendrux_blob_refs_run", "dendrux_blob_refs", ["run_id", "tenant_scope"])


def downgrade() -> None:
    op.drop_index("ix_dendrux_blob_refs_run", table_name="dendrux_blob_refs")
    op.drop_index("ix_dendrux_blob_refs_blob", table_name="dendrux_blob_refs")
    op.drop_table("dendrux_blob_refs")
    op.drop_index("ix_dendrux_blobs_work", table_name="dendrux_blobs")
    op.drop_index("ix_dendrux_blobs_gc", table_name="dendrux_blobs")
    op.drop_table("dendrux_blobs")
