from typing import TYPE_CHECKING, List, Optional

from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.document import Document
from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.schemas.organization import Organization as PydanticOrganization

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from memgpt.orm.agent import Agent
    from memgpt.orm.source import Source
    from memgpt.orm.tool import Tool
    from memgpt.orm.user import User


class Organization(SqlalchemyBase):
    """The highest level of the object tree. All Entities belong to one and only one Organization."""

    __tablename__ = "organization"
    __pydantic_model__ = PydanticOrganization

    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The display name of the organization.")

    # relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    agents: Mapped[List["Agent"]] = relationship("Agent", back_populates="organization", cascade="all, delete-orphan")
    sources: Mapped[List["Source"]] = relationship("Source", back_populates="organization", cascade="all, delete-orphan")
    tools: Mapped[List["Tool"]] = relationship("Tool", back_populates="organization", cascade="all, delete-orphan")
    documents: Mapped[List["Document"]] = relationship("Document", back_populates="organization", cascade="all, delete-orphan")

    @classmethod
    def default(cls, db_session: "Session") -> "Organization":
        """Get the default org, or create it if it doesn't exist."""
        try:
            return db_session.query(cls).filter(cls.name == "Default Organization").one()
        except NoResultFound:
            return cls(name="Default Organization").create(db_session)
