from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.models.pydantic_models import EmbeddingConfigModel
from memgpt.orm.mixins import DocumentMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase

if TYPE_CHECKING:
    from memgpt.orm.document import Document


class Passage(DocumentMixin, SqlalchemyBase):
    """A segment of text from a document."""

    __tablename__ = "passage"

    text: Mapped[str] = mapped_column(doc="The text of the passage.")
    embedding: Mapped[Optional[List[float]]] = mapped_column(JSON, doc="The embedding of the passage.", nullable=True)
    embedding_config: Mapped[Optional["EmbeddingConfigModel"]] = mapped_column(
        JSON, doc="The embedding configuration used by the passage.", nullable=True
    )
    data_source: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Human readable description of where the passage came from.")
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, default=lambda: {}, doc="additional information about the passage.")

    # relationships
    document: Mapped["Document"] = relationship("Document", back_populates="passages")

    # TODO: the embedding needs to be padded to the same dimentions to enable them to be stored in the same table
