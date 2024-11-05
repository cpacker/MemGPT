"""Reset id field to old ids

Revision ID: 91b25c1cb0ae
Revises: eff245f340f9
Create Date: 2024-11-05 11:07:49.862962

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "91b25c1cb0ae"
down_revision: Union[str, None] = "eff245f340f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop foreign keys that depend on `_id` in `organization`
    op.drop_constraint("tool__organization_id_fkey", "tool", type_="foreignkey")
    op.drop_constraint("user__organization_id_fkey", "user", type_="foreignkey")

    # Add new `id` column with unique constraint and drop `_id` column in `organization`
    op.add_column("organization", sa.Column("id", sa.String(), nullable=False))
    op.create_unique_constraint("uq_organization_id", "organization", ["id"])
    op.drop_column("organization", "_id")

    # Modify `tool` table to use new `organization_id` with updated foreign key
    op.add_column("tool", sa.Column("id", sa.String(), nullable=False))
    op.add_column("tool", sa.Column("organization_id", sa.String(), nullable=False))
    op.drop_column("tool", "_id")
    op.drop_column("tool", "_organization_id")
    op.create_foreign_key(None, "tool", "organization", ["organization_id"], ["id"])

    # Modify `user` table to use new `organization_id` with updated foreign key
    op.add_column("user", sa.Column("id", sa.String(), nullable=False))
    op.add_column("user", sa.Column("organization_id", sa.String(), nullable=False))
    op.drop_column("user", "_id")
    op.drop_column("user", "_organization_id")
    op.create_foreign_key(None, "user", "organization", ["organization_id"], ["id"])


def downgrade() -> None:
    # Drop foreign keys that depend on `organization_id` in `organization`
    op.drop_constraint(None, "tool", type_="foreignkey")
    op.drop_constraint(None, "user", type_="foreignkey")

    # Revert changes in `user` table
    op.add_column("user", sa.Column("_organization_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.add_column("user", sa.Column("_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.drop_column("user", "organization_id")
    op.drop_column("user", "id")
    op.create_foreign_key("user__organization_id_fkey", "user", "organization", ["_organization_id"], ["_id"])

    # Revert changes in `tool` table
    op.add_column("tool", sa.Column("_organization_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.add_column("tool", sa.Column("_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.drop_column("tool", "organization_id")
    op.drop_column("tool", "id")
    op.create_foreign_key("tool__organization_id_fkey", "tool", "organization", ["_organization_id"], ["_id"])

    # Revert changes in `organization` table
    op.add_column("organization", sa.Column("_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.drop_constraint("uq_organization_id", "organization", type_="unique")
    op.drop_column("organization", "id")
