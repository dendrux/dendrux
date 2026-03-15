"""sprint 3: add pause_data column to agent_runs

Revision ID: a1b2c3d4e5f6
Revises: 2ce3bb57b0c5
Create Date: 2026-03-15 18:00:00.000000+00:00
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "2ce3bb57b0c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("pause_data", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "pause_data")
