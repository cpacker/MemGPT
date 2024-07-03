from typing import Optional, TYPE_CHECKING, List
from sqlalchemy import JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import OrganizationMixin

if TYPE_CHECKING:
    from memgpt.orm.organization import Organization
    from memgpt.orm.passage import Passage


class Document(OrganizationMixin, SqlalchemyBase):
    """Represents a file or distinct, complete body of information.
    """
    __tablename__ = "document"

    text: Mapped[str] = mapped_column(doc="The full text for the document.")
    data_source: Optional[str] = mapped_column(nullable=True, doc="Human readable description of where the passage came from.")
    metadata_: Optional[dict] = mapped_column(JSON, default_factory=lambda: {}, doc="additional information about the passage.")


    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="documents")
    passages: Mapped[List["Passage"]] = relationship("Passage", back_populates="document")