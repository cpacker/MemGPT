from typing import TYPE_CHECKING, List, Union

from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.organization import Organization as PydanticOrganization

if TYPE_CHECKING:

    from letta.orm.agent import Agent
    from letta.orm.file import FileMetadata
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
    blocks: Mapped[List["Block"]] = relationship("Block", back_populates="organization", cascade="all, delete-orphan")
    sources: Mapped[List["Source"]] = relationship("Source", back_populates="organization", cascade="all, delete-orphan")
    files: Mapped[List["FileMetadata"]] = relationship("FileMetadata", back_populates="organization", cascade="all, delete-orphan")
    sandbox_configs: Mapped[List["SandboxConfig"]] = relationship(
        "SandboxConfig", back_populates="organization", cascade="all, delete-orphan"
    )
    sandbox_environment_variables: Mapped[List["SandboxEnvironmentVariable"]] = relationship(
        "SandboxEnvironmentVariable", back_populates="organization", cascade="all, delete-orphan"
    )

    # relationships
    agents: Mapped[List["Agent"]] = relationship("Agent", back_populates="organization", cascade="all, delete-orphan")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="organization", cascade="all, delete-orphan")
    source_passages: Mapped[List["SourcePassage"]] = relationship(
        "SourcePassage", 
        back_populates="organization", 
        cascade="all, delete-orphan"
    )
    agent_passages: Mapped[List["AgentPassage"]] = relationship(
        "AgentPassage", 
        back_populates="organization", 
        cascade="all, delete-orphan"
    )

    @property
    def passages(self) -> List[Union["SourcePassage", "AgentPassage"]]:
        """Convenience property to get all passages"""
        return self.source_passages + self.agent_passages


