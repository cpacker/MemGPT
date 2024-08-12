from typing import List, TYPE_CHECKING
from sqlalchemy.orm import relationship, Mapped, mapped_column


from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import OrganizationMixin

if TYPE_CHECKING:
    from memgpt.orm.agent import Agent
    from memgpt.orm.organization import Organization
    from memgpt.orm.preset import Preset

class Source(OrganizationMixin,SqlalchemyBase):
    """A source represents an embedded text passage"""
    __tablename__ = 'source'

    name: Mapped[str] = mapped_column(doc="the name of the source, must be unique within the org", nullable=False)
    # TODO: feels like embeddings should be a first class object
    embedding_dim:Mapped[int] = mapped_column(doc="the max number of dimensions for embedding vectors", nullable=False)
    embedding_model:Mapped[str] = mapped_column(doc="the name of the embedding model used to generate the embedding", nullable=False)
    description:Mapped[str] = mapped_column(nullable=True, doc="a human-readable description of the source")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="sources")
    agents: Mapped[List["Agent"]] = relationship("Agent", secondary="sources_agents", back_populates="sources")