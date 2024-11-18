from typing import TYPE_CHECKING, List

from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.file import FileMetadata
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.organization import Organization as PydanticOrganization

if TYPE_CHECKING:

    from letta.orm.tool import Tool
    from letta.orm.user import User


class Organization(SqlalchemyBase):
    """The highest level of the object tree. All Entities belong to one and only one Organization."""

    __tablename__ = "organizations"
    __pydantic_model__ = PydanticOrganization

    name: Mapped[str] = mapped_column(doc="The display name of the organization.")

    # relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    tools: Mapped[List["Tool"]] = relationship("Tool", back_populates="organization", cascade="all, delete-orphan")
    sources: Mapped[List["Source"]] = relationship("Source", back_populates="organization", cascade="all, delete-orphan")
    agents_tags: Mapped[List["AgentsTags"]] = relationship("AgentsTags", back_populates="organization", cascade="all, delete-orphan")
    files: Mapped[List["FileMetadata"]] = relationship("FileMetadata", back_populates="organization", cascade="all, delete-orphan")
    # TODO: Map these relationships later when we actually make these models
    # below is just a suggestion
    # agents: Mapped[List["Agent"]] = relationship("Agent", back_populates="organization", cascade="all, delete-orphan")
    # tools: Mapped[List["Tool"]] = relationship("Tool", back_populates="organization", cascade="all, delete-orphan")
    # documents: Mapped[List["Document"]] = relationship("Document", back_populates="organization", cascade="all, delete-orphan")
