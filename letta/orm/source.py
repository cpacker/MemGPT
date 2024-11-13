from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, TypeDecorator
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.source import Source as PydanticSource

if TYPE_CHECKING:
    from letta.orm.organization import Organization


class EmbeddingConfigColumn(TypeDecorator):
    """Custom type for storing EmbeddingConfig as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            # return vars(value)
            if isinstance(value, EmbeddingConfig):
                return value.model_dump()
        return value

    def process_result_value(self, value, dialect):
        if value:
            return EmbeddingConfig(**value)
        return value


class Source(SqlalchemyBase, OrganizationMixin):
    """A source represents an embedded text passage"""

    __tablename__ = "sources"
    __pydantic_model__ = PydanticSource

    name: Mapped[str] = mapped_column(doc="the name of the source, must be unique within the org", nullable=False)
    description: Mapped[str] = mapped_column(nullable=True, doc="a human-readable description of the source")
    embedding_config: Mapped[EmbeddingConfig] = mapped_column(EmbeddingConfigColumn, doc="Configuration settings for embedding.")
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, doc="metadata for the source.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="sources")
    files: Mapped[List["Source"]] = relationship("FileMetadata", back_populates="source", cascade="all, delete-orphan")
    # agents: Mapped[List["Agent"]] = relationship("Agent", secondary="sources_agents", back_populates="sources")
