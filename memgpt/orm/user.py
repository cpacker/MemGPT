from typing import Optional, TYPE_CHECKING, List
from pydantic import EmailStr
from sqlalchemy import String
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Mapped, relationship, mapped_column

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.organization import Organization
from memgpt.orm.mixins import OrganizationMixin
from memgpt.data_types import User as PydanticUser

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from memgpt.orm.agent import Agent
    from memgpt.orm.token import Token
    from memgpt.orm.job import Job

class User(SqlalchemyBase, OrganizationMixin):
    """User ORM class"""
    __tablename__ = "user"
    __pydantic_model__ = PydanticUser

    name:Mapped[Optional[str]] = mapped_column(nullable=True, doc="The display name of the user.")
    email:Mapped[Optional[EmailStr]] = mapped_column(String,
                                                     nullable=True,
                                                     doc="The email address of the user. Uninforced at this time.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")
    agents: Mapped[List["Agent"]] = relationship("Agent",
                                                 secondary="users_agents",
                                                 back_populates="users",
                                                 doc="the agents associated with this user.")
    tokens: Mapped[List["Token"]] = relationship("Token", back_populates="user", doc="the tokens associated with this user.")
    jobs: Mapped[List["Job"]] = relationship("Job", back_populates="user", doc="the jobs associated with this user.")


    @classmethod
    def default(cls, db_session:"Session") -> "User":
        """Get the default user, or create it if it doesn't exist.
        Note: this is only for local client use.
        """
        try:
            with db_session.begin() as session:
                return session.query(cls).one()
        except NoResultFound:
            return cls(name="Default User",
                       organization=Organization.default(db_session)).create(db_session)