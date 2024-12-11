import base64
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
from sqlalchemy import JSON, Column, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import BINARY, TypeDecorator

from letta.config import LettaConfig
from letta.constants import MAX_EMBEDDING_DIM
from letta.orm.mixins import FileMixin, OrganizationMixin
from letta.orm.source import EmbeddingConfigColumn
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.passage import Passage as PydanticPassage
from letta.settings import settings

config = LettaConfig()

if TYPE_CHECKING:
    from letta.orm.organization import Organization


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


# TODO: After migration to Passage, will need to manually delete passages where files
#       are deleted on web
class Passage(SqlalchemyBase, OrganizationMixin, FileMixin):
    """Defines data model for storing Passages"""

    __tablename__ = "passages"
    __table_args__ = {"extend_existing": True}
    __pydantic_model__ = PydanticPassage

    id: Mapped[str] = mapped_column(primary_key=True, doc="Unique passage identifier")
    text: Mapped[str] = mapped_column(doc="Passage text content")
    source_id: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Source identifier")
    embedding_config: Mapped[dict] = mapped_column(EmbeddingConfigColumn, doc="Embedding configuration")
    metadata_: Mapped[dict] = mapped_column(JSON, doc="Additional metadata")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    if settings.letta_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
    else:
        embedding = Column(CommonVector)

    # Foreign keys
    agent_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("agents.id"), nullable=True)

    # Relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="passages", lazy="selectin")
    file: Mapped["FileMetadata"] = relationship("FileMetadata", back_populates="passages", lazy="selectin")
