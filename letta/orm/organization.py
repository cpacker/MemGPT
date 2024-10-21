from typing import TYPE_CHECKING

from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.organization import Organization as PydanticOrganization

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class Organization(SqlalchemyBase):
    """The highest level of the object tree. All Entities belong to one and only one Organization."""

    __tablename__ = "organizations"
    __pydantic_model__ = PydanticOrganization

    name: Mapped[str] = mapped_column(doc="The display name of the organization.")

    # TODO: Map these relationships later when we actually make these models
    # below is just a suggestion
    # users: Mapped[List["User"]] = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    # agents: Mapped[List["Agent"]] = relationship("Agent", back_populates="organization", cascade="all, delete-orphan")
    # sources: Mapped[List["Source"]] = relationship("Source", back_populates="organization", cascade="all, delete-orphan")
    # tools: Mapped[List["Tool"]] = relationship("Tool", back_populates="organization", cascade="all, delete-orphan")
    # documents: Mapped[List["Document"]] = relationship("Document", back_populates="organization", cascade="all, delete-orphan")

    @classmethod
    def default(cls, db_session: "Session") -> "Organization":
        """Get the default org, or create it if it doesn't exist."""
        try:
            return db_session.query(cls).filter(cls.name == "Default Organization").one()
        except NoResultFound:
            return cls(name="Default Organization").create(db_session)
