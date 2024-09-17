"""add unique constraints per org

Revision ID: 36988c9642bd
Revises: 4d715e99d90f
Create Date: 2024-09-10 19:04:27.019205

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "36988c9642bd"
down_revision: Union[str, None] = "4d715e99d90f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_unique_constraint("unique_agent_name_per_organization", "agent", ["_organization_id", "name"])
    op.create_unique_constraint("unique_block_name_per_organization", "block", ["_organization_id", "name"])
    op.create_unique_constraint("unique_source_name_per_organization", "source", ["_organization_id", "name"])
    op.create_unique_constraint("unique_tool_name_per_organization", "tool", ["_organization_id", "name"])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint("unique_tool_name_per_organization", "tool", type_="unique")
    op.drop_constraint("unique_source_name_per_organization", "source", type_="unique")
    op.drop_constraint("unique_block_name_per_organization", "block", type_="unique")
    op.drop_constraint("unique_agent_name_per_organization", "agent", type_="unique")
    # ### end Alembic commands ###
