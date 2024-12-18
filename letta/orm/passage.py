from typing import TYPE_CHECKING

from sqlalchemy import JSON, Column, Index
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from letta.config import LettaConfig
from letta.constants import MAX_EMBEDDING_DIM
from letta.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from letta.orm.mixins import AgentMixin, FileMixin, OrganizationMixin, SourceMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.passage import Passage as PydanticPassage
from letta.settings import settings

config = LettaConfig()

if TYPE_CHECKING:
    from letta.orm.agent import Agent
    from letta.orm.organization import Organization


class BasePassage(SqlalchemyBase, OrganizationMixin):
    """Base class for all passage types with common fields"""

    __abstract__ = True
    __pydantic_model__ = PydanticPassage

    id: Mapped[str] = mapped_column(primary_key=True, doc="Unique passage identifier")
    text: Mapped[str] = mapped_column(doc="Passage text content")
    embedding_config: Mapped[dict] = mapped_column(EmbeddingConfigColumn, doc="Embedding configuration")
    metadata_: Mapped[dict] = mapped_column(JSON, doc="Additional metadata")

    # Vector embedding field based on database type
    if settings.letta_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
    else:
        embedding = Column(CommonVector)

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """Relationship to organization"""
        return relationship("Organization", back_populates="passages", lazy="selectin")

    @declared_attr
    def __table_args__(cls):
        if settings.letta_pg_uri_no_default:
            return (Index(f"{cls.__tablename__}_org_idx", "organization_id"), {"extend_existing": True})
        return ({"extend_existing": True},)


class SourcePassage(BasePassage, FileMixin, SourceMixin):
    """Passages derived from external files/sources"""

    __tablename__ = "source_passages"

    @declared_attr
    def file(cls) -> Mapped["FileMetadata"]:
        """Relationship to file"""
        return relationship("FileMetadata", back_populates="source_passages", lazy="selectin")

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        return relationship("Organization", back_populates="source_passages", lazy="selectin")

    @declared_attr
    def source(cls) -> Mapped["Source"]:
        """Relationship to source"""
        return relationship("Source", back_populates="passages", lazy="selectin", passive_deletes=True)


class AgentPassage(BasePassage, AgentMixin):
    """Passages created by agents as archival memories"""

    __tablename__ = "agent_passages"

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        return relationship("Organization", back_populates="agent_passages", lazy="selectin")

    @declared_attr
    def agent(cls) -> Mapped["Agent"]:
        """Relationship to agent"""
        return relationship("Agent", back_populates="agent_passages", lazy="selectin", passive_deletes=True)
