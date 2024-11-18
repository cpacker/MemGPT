"""Move files to orm

Revision ID: c85a3d07c028
Revises: cda66b6cb0d6
Create Date: 2024-11-12 13:58:57.221081

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c85a3d07c028"
down_revision: Union[str, None] = "cda66b6cb0d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("files", sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True))
    op.add_column("files", sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False))
    op.add_column("files", sa.Column("_created_by_id", sa.String(), nullable=True))
    op.add_column("files", sa.Column("_last_updated_by_id", sa.String(), nullable=True))
    op.add_column("files", sa.Column("organization_id", sa.String(), nullable=True))
    # Populate `organization_id` based on `user_id`
    # Use a raw SQL query to update the organization_id
    op.execute(
        """
        UPDATE files
        SET organization_id = users.organization_id
        FROM users
        WHERE files.user_id = users.id
    """
    )
    op.alter_column("files", "organization_id", nullable=False)
    op.create_foreign_key(None, "files", "organizations", ["organization_id"], ["id"])
    op.create_foreign_key(None, "files", "sources", ["source_id"], ["id"])
    op.drop_column("files", "user_id")
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("files", sa.Column("user_id", sa.VARCHAR(), autoincrement=False, nullable=False))
    op.drop_constraint(None, "files", type_="foreignkey")
    op.drop_constraint(None, "files", type_="foreignkey")
    op.drop_column("files", "organization_id")
    op.drop_column("files", "_last_updated_by_id")
    op.drop_column("files", "_created_by_id")
    op.drop_column("files", "is_deleted")
    op.drop_column("files", "updated_at")
    # ### end Alembic commands ###
