"""Tweak created_at field for messages

Revision ID: e1a625072dbf
Revises: 95badb46fdf9
Create Date: 2024-12-07 14:28:27.643583

"""

from typing import Sequence, Union

from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1a625072dbf"
down_revision: Union[str, None] = "95badb46fdf9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column("messages", "created_at", existing_type=postgresql.TIMESTAMP(timezone=True), nullable=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column("messages", "created_at", existing_type=postgresql.TIMESTAMP(timezone=True), nullable=False)
    # ### end Alembic commands ###
