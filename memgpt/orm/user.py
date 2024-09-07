from typing import TYPE_CHECKING, List, Optional

from pydantic import EmailStr
from sqlalchemy import String
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.mixins import OrganizationMixin
from memgpt.orm.organization import Organization
from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.schemas.user import User as PydanticUser

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from memgpt.orm.agent import Agent
    from memgpt.orm.job import Job
    from memgpt.orm.token import Token


class User(SqlalchemyBase, OrganizationMixin):
    """User ORM class"""

    __tablename__ = "user"
    __pydantic_model__ = PydanticUser

    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The display name of the user.")
    email: Mapped[Optional[EmailStr]] = mapped_column(String, nullable=True, doc="The email address of the user. Uninforced at this time.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")
    agents: Mapped[List["Agent"]] = relationship(
        "Agent", secondary="users_agents", back_populates="users", doc="the agents associated with this user."
    )
    tokens: Mapped[List["Token"]] = relationship("Token", back_populates="user", doc="the tokens associated with this user.")
    jobs: Mapped[List["Job"]] = relationship("Job", back_populates="user", doc="the jobs associated with this user.")

    @classmethod
    def default(cls, db_session: "Session") -> "User":
        """Get the default user, or create it if it doesn't exist.
        Note: this is only for local client use.
        """
        default_user_label = "Default User"
        try:
            return db_session.query(cls).filter(cls.name == default_user_label).one()
        except NoResultFound:
            org = Organization.default(db_session)
            return cls(name=default_user_label, organization=org).create(db_session)
