from uuid import UUID

from sqlalchemy import UUID as SQLUUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from memgpt.orm.base import Base


class UsersAgents(Base):
    __tablename__ = "users_agents"

    _agent_id: Mapped[UUID] = mapped_column(SQLUUID, ForeignKey("agent._id"), primary_key=True)
    _user_id: Mapped[UUID] = mapped_column(SQLUUID, ForeignKey("user._id"), primary_key=True)
