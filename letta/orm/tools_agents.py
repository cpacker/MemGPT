from sqlalchemy import ForeignKey, ForeignKeyConstraint, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.tools_agents import ToolsAgents as PydanticToolsAgents


class ToolsAgents(SqlalchemyBase):
    """Agents can have one or many tools associated with them."""

    __tablename__ = "tools_agents"
    __pydantic_model__ = PydanticToolsAgents
    __table_args__ = (
        UniqueConstraint(
            "agent_id",
            "tool_name",
            name="unique_tool_per_agent",
        ),
        ForeignKeyConstraint(
            ["tool_id"],
            ["tools.id"],
            name="fk_tool_id",
        ),
    )

    # Each agent must have unique tool names
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"), primary_key=True)
    tool_id: Mapped[str] = mapped_column(String, primary_key=True)
    tool_name: Mapped[str] = mapped_column(String, primary_key=True)

    # relationships
    tool: Mapped["Tool"] = relationship("Tool", back_populates="tools_agents")    # agent: Mapped["Agent"] = relationship("Agent", back_populates="tools_agents")
