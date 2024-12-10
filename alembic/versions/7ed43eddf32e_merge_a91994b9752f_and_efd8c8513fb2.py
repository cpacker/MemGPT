"""merge a91994b9752f and efd8c8513fb2

Revision ID: 7ed43eddf32e
Revises: a91994b9752f, efd8c8513fb2
Create Date: 2024-12-10 07:08:24.941213

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7ed43eddf32e'
down_revision: Union[str, None] = ('a91994b9752f', 'efd8c8513fb2')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
