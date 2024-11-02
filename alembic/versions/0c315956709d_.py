from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# Revision identifiers, used by Alembic.
revision: str = "0c315956709d"
down_revision: Union[str, None] = "9a505cc7eca9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Step 1: Rename tables first
    op.rename_table("organizations", "organization")
    op.rename_table("tools", "tool")
    op.rename_table("users", "user")

    # Step 2: Rename `id` to `_id` in each table, keeping it as the primary key
    op.alter_column("organization", "id", new_column_name="_id", existing_type=sa.String, nullable=False)
    op.alter_column("tool", "id", new_column_name="_id", existing_type=sa.String, nullable=False)
    op.alter_column("user", "id", new_column_name="_id", existing_type=sa.String, nullable=False)

    # Step 3: Add `_organization_id` to `tool` table
    # This is required for the unique constraint below
    op.add_column("tool", sa.Column("_organization_id", sa.String, nullable=True))

    # Step 4: Modify nullable constraints on `tool` columns
    op.alter_column("tool", "tags", existing_type=sa.JSON, nullable=True)
    op.alter_column("tool", "source_type", existing_type=sa.String, nullable=True)
    op.alter_column("tool", "json_schema", existing_type=sa.JSON, nullable=True)

    # Step 5: Add unique constraint on `name` and `_organization_id` in `tool` table
    op.create_unique_constraint("uq_tool_name_organization", "tool", ["name", "_organization_id"])


def downgrade() -> None:
    # Reverse unique constraint first
    op.drop_constraint("uq_tool_name_organization", "tool", type_="unique")

    # Reverse nullable constraints on `tool` columns
    op.alter_column("tool", "tags", existing_type=sa.JSON, nullable=False)
    op.alter_column("tool", "source_type", existing_type=sa.String, nullable=False)
    op.alter_column("tool", "json_schema", existing_type=sa.JSON, nullable=False)

    # Remove `_organization_id` column from `tool` table
    op.drop_column("tool", "_organization_id")

    # Reverse the column renaming from `_id` back to `id`
    op.alter_column("organization", "_id", new_column_name="id", existing_type=sa.String, nullable=False)
    op.alter_column("tool", "_id", new_column_name="id", existing_type=sa.String, nullable=False)
    op.alter_column("user", "_id", new_column_name="id", existing_type=sa.String, nullable=False)

    # Reverse table renaming last
    op.rename_table("organization", "organizations")
    op.rename_table("tool", "tools")
    op.rename_table("user", "users")
