"""merge c2304abb2dd2 and e1a625072dbf

Revision ID: 2c3a0f5ca145
Revises: c2304abb2dd2, e1a625072dbf
Create Date: 2024-12-10 06:18:06.944572

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2c3a0f5ca145'
down_revision: Union[str, None] = ('c2304abb2dd2', 'e1a625072dbf')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
