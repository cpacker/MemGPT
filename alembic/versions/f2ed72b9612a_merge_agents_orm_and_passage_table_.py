"""merge agents ORM and passage table separation

Revision ID: f2ed72b9612a
Revises: 458325ef2daf, d05669b60ebe
Create Date: 2024-12-13 15:01:42.832476

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2ed72b9612a'
down_revision: Union[str, None] = ('458325ef2daf', 'd05669b60ebe')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
