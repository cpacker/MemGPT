from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import Column, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship, declared_attr
from sqlalchemy.types import TypeDecorator, BINARY

import numpy as np
import base64

from letta.orm.source import EmbeddingConfigColumn
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.orm.mixins import AgentMixin, FileMixin, OrganizationMixin, SourceMixin
from letta.schemas.passage import Passage as PydanticPassage


from letta.config import LettaConfig
from letta.constants import MAX_EMBEDDING_DIM
from letta.orm.custom_columns import CommonVector
from letta.orm.mixins import FileMixin, OrganizationMixin
from letta.orm.source import 

from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.passage import Passage as PydanticPassage
from letta.settings import settings

config = LettaConfig()

if TYPE_CHECKING:
    from letta.orm.organization import Organization
    from letta.orm.agent import Agent


class CommonVector(TypeDecorator):
    """Common type for representing vectors in SQLite"""

    impl = BINARY
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(BINARY())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, list):
            value = np.array(value, dtype=np.float32)
        return base64.b64encode(value.tobytes())

    def process_result_value(self, value, dialect):
        if not value:
            return value
        if dialect.name == "sqlite":
            value = base64.b64decode(value)
        return np.frombuffer(value, dtype=np.float32)

class BasePassage(SqlalchemyBase, OrganizationMixin):
    """Base class for all passage types with common fields"""
    __abstract__ = True
    __pydantic_model__ = PydanticPassage

    id: Mapped[str] = mapped_column(primary_key=True, doc="Unique passage identifier")
    text: Mapped[str] = mapped_column(doc="Passage text content")
    embedding_config: Mapped[dict] = mapped_column(EmbeddingConfigColumn, doc="Embedding configuration")
    metadata_: Mapped[dict] = mapped_column(JSON, doc="Additional metadata")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

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
            return (
                Index(f'{cls.__tablename__}_org_idx', 'organization_id'),
                {"extend_existing": True}
            )
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
