from typing import TYPE_CHECKING, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.constants import MAX_EMBEDDING_DIM
from memgpt.orm.mixins import AgentMixin, DocumentMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.passage import Passage as PydanticPassage

if TYPE_CHECKING:
    from memgpt.orm.agent import Agent
    from memgpt.orm.document import Document


class Passage(AgentMixin, DocumentMixin, SqlalchemyBase):
    """A segment of text from a document."""

    __tablename__ = "passage"
    __pydantic_model__ = PydanticPassage

    text: Mapped[str] = mapped_column(doc="The text of the passage.")
    # TODO: embedding should support PGVector type and switch based on dialect
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(MAX_EMBEDDING_DIM), doc="The embedding of the passage.", nullable=True)
    embedding_config: Mapped[Optional["EmbeddingConfig"]] = mapped_column(
        JSON, doc="The embedding configuration used by the passage.", nullable=True
    )
    data_source: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Human readable description of where the passage came from.")
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, default=lambda: {}, doc="additional information about the passage.")
    user_id: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The user ID of the user who created this passage")

    # relationships
    document: Mapped[Optional["Document"]] = relationship("Document", back_populates="passages")
    agent: Mapped["Agent"] = relationship("Agent", back_populates="passages")
