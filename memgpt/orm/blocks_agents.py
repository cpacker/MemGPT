from sqlalchemy import ForeignKey, UUID as SQLUUID
from uuid import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column, UniqueConstraint

from memgpt.orm.base import Base


class BlocksAgents(Base):
    """Agents must have one or many blocks to make up their core memory"""
    __tablename__ = 'blocks_agents'
    __table_args__ = (
        UniqueConstraint(
            "_agent_id",
            "_block_label",
            name="unique_label_per_agent",
        ),
    )
    # unique agent + block label



    _agent_id:Mapped[UUID] = mapped_column(SQLUUID, ForeignKey('agent._id'), primary_key=True)
    _block_id:Mapped[UUID] = mapped_column(SQLUUID, ForeignKey('block._id'), primary_key=True)
    _block_label:Mapped[str] = mapped_column(str, ForeignKey('block.label'), primary_key=True)