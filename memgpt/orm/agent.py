from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import OrganizationMixin
from memgpt.orm.users_agents import UsersAgents
if TYPE_CHECKING:
    from memgpt.orm.organization import Organization
    from memgpt.orm.user import User

class Agent(SqlalchemyBase, OrganizationMixin):
    __tablename__ = 'agent'

    name:Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="a human-readable identifier for an agent, non-unique.")
    persona: Mapped[str] = mapped_column(doc="the persona text for the agent, current state.")
    # todo: this doesn't allign with 1:M agents to users!
    human: Mapped[str] = mapped_column(doc="the human text for the agent and the current user, current state.")
    preset: Mapped[str] = mapped_column(doc="the preset text for the agent, current state.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="agents")
    users: Mapped[List["User"]] = relationship("User",
                                               back_populates="agents",
                                               secondary="users_agents",
                                               doc="the users associated with this agent.")
