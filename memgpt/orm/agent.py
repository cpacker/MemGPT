from typing import Optional
from sqlalchemy import String
from sqlachemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import OrganizationMixin

class Agent(SqlalchemyBase):
    __tablename__ = 'agent'

    name:Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="a name to identify the token")

    # is this correct? do agents belong to a single user?
    user: Mapped["User"] = relationship("User", back_populates="agents")