from uuid import UUID

from sqlalchemy import UUID as SQLUUID
from sqlalchemy import ForeignKey, ForeignKeyConstraint, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from memgpt.orm.base import Base


class BlocksAgents(Base):
    """Agents must have one or many blocks to make up their core memory"""

    __tablename__ = "blocks_agents"
    __table_args__ = (
        UniqueConstraint(
            "_agent_id",
            "_block_label",
            name="unique_label_per_agent",
        ),
        ForeignKeyConstraint(["_block_id", "_block_label"], ["block._id", "block.label"]),
        {},
    )

    # unique agent + block label
    _agent_id: Mapped[UUID] = mapped_column(SQLUUID, ForeignKey("agent._id"), primary_key=True)
    _block_id: Mapped[UUID] = mapped_column(SQLUUID, primary_key=True)
    _block_label: Mapped[str] = mapped_column(String, primary_key=True)
