from sqlalchemy import ForeignKey, UUID as SQLUUID
from uuid import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column

from memgpt.orm.base import Base


class SourcesAgents(Base):
    """Agents can have zero to many sources"""
    __tablename__ = 'sources_agents'

    _agent_id:Mapped[UUID] = mapped_column(SQLUUID, ForeignKey('agent._id'), primary_key=True)
    _source_id:Mapped[UUID] = mapped_column(SQLUUID, ForeignKey('source._id'), primary_key=True)