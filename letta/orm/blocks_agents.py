from sqlalchemy import ForeignKey, ForeignKeyConstraint, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.sqlalchemy_base import SqlalchemyBase


class BlocksAgents(SqlalchemyBase):
    """Agents must have one or many blocks to make up their core memory"""

    __tablename__ = "blocks_agents"
    __table_args__ = (
        UniqueConstraint(
            "agent_id",
            "block_label",
            name="unique_label_per_agent",
        ),
        ForeignKeyConstraint(("block_id", "block_label"), ("block.id", "block.label")),
        {},
    )

    # unique agent + block label
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agent.id"), primary_key=True)
    block_id: Mapped[str] = mapped_column(String, primary_key=True)
    block_label: Mapped[str] = mapped_column(String, primary_key=True)
