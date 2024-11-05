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

    # Add new `id` column to `organization`, `tool`, and `user` tables
    op.add_column("organization", sa.Column("id", sa.String(), nullable=False))
    op.add_column("tool", sa.Column("id", sa.String(), nullable=False))
    op.add_column("tool", sa.Column("organization_id", sa.String(), nullable=False))
    op.add_column("user", sa.Column("id", sa.String(), nullable=False))
    op.add_column("user", sa.Column("organization_id", sa.String(), nullable=False))

    # Drop old `_id` and `_organization_id` columns
    op.drop_column("organization", "_id")
    op.drop_column("tool", "_id")
    op.drop_column("tool", "_organization_id")
    op.drop_column("user", "_id")
    op.drop_column("user", "_organization_id")

    # Set `id` as primary key for each table
    op.create_primary_key("pk_organization_id", "organization", ["id"])
    op.create_primary_key("pk_tool_id", "tool", ["id"])
    op.create_primary_key("pk_user_id", "user", ["id"])

    # Re-create foreign key constraints based on the new `id` columns
    op.create_foreign_key(None, "tool", "organization", ["organization_id"], ["id"])
    op.create_unique_constraint("uix_name_organization", "tool", ["name", "organization_id"])
    op.create_foreign_key(None, "user", "organization", ["organization_id"], ["id"])


def downgrade() -> None:
    # Drop foreign keys that depend on `organization_id` in `organization`
    op.drop_constraint(None, "tool", type_="foreignkey")
    op.drop_constraint(None, "user", type_="foreignkey")

    # Re-add old `_id` and `_organization_id` columns
    op.add_column("organization", sa.Column("_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.add_column("tool", sa.Column("_organization_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.add_column("tool", sa.Column("_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.add_column("user", sa.Column("_organization_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.add_column("user", sa.Column("_id", sa.VARCHAR(), autoincrement=False, nullable=False))

    # Revert primary keys on `organization`, `tool`, and `user` tables
    op.drop_constraint("pk_organization_id", "organization", type_="primary")
    op.drop_constraint("pk_tool_id", "tool", type_="primary")
    op.drop_constraint("pk_user_id", "user", type_="primary")

    # Drop new `id` columns from each table
    op.drop_column("organization", "id")
    op.drop_column("tool", "id")
    op.drop_column("tool", "organization_id")
    op.drop_column("user", "id")
    op.drop_column("user", "organization_id")

    # Re-create foreign keys and unique constraint based on the old columns
    op.create_foreign_key("tool__organization_id_fkey", "tool", "organization", ["_organization_id"], ["_id"])
    op.create_unique_constraint("uix_name_organization", "tool", ["name", "_organization_id"])
    op.create_foreign_key("user__organization_id_fkey", "user", "organization", ["_organization_id"], ["_id"])
