from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, String, UniqueConstraint, event
from sqlalchemy.orm import Mapped, mapped_column, relationship

# TODO everything in functions should live in this model
from letta.orm.enums import ToolSourceType
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.tool import Tool as PydanticTool

if TYPE_CHECKING:
    from letta.orm.organization import Organization
    from letta.orm.tools_agents import ToolsAgents


class Tool(SqlalchemyBase, OrganizationMixin):
    """Represents an available tool that the LLM can invoke.

    NOTE: polymorphic inheritance makes more sense here as a TODO. We want a superset of tools
    that are always available, and a subset scoped to the organization. Alternatively, we could use the apply_access_predicate to build
    more granular permissions.
    """

    __tablename__ = "tools"
    __pydantic_model__ = PydanticTool

    # Add unique constraint on (name, _organization_id)
    # An organization should not have multiple tools with the same name
    __table_args__ = (UniqueConstraint("name", "organization_id", name="uix_name_organization"),)

    name: Mapped[str] = mapped_column(doc="The display name of the tool.")
    description: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The description of the tool.")
    tags: Mapped[List] = mapped_column(JSON, doc="Metadata tags used to filter tools.")
    source_type: Mapped[ToolSourceType] = mapped_column(String, doc="The type of the source code.", default=ToolSourceType.json)
    source_code: Mapped[Optional[str]] = mapped_column(String, doc="The source code of the function.")
    json_schema: Mapped[dict] = mapped_column(JSON, default=lambda: {}, doc="The OAI compatable JSON schema of the function.")
    module: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="the module path from which this tool was derived in the codebase."
    )

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="tools", lazy="selectin")
    tools_agents: Mapped[List["ToolsAgents"]] = relationship("ToolsAgents", back_populates="tool", cascade="all, delete-orphan")


# Add event listener to update tool_name in ToolsAgents when Tool name changes
@event.listens_for(Tool, 'before_update')
def update_tool_name_in_tools_agents(mapper, connection, target):
    """Update tool_name in ToolsAgents when Tool name changes."""
    state = target._sa_instance_state
    history = state.get_history('name', passive=True)
    if not history.has_changes():
        return
    
    # Get the new name and update all associated ToolsAgents records
    new_name = target.name
    from letta.orm.tools_agents import ToolsAgents
    connection.execute(
        ToolsAgents.__table__.update().where(
            ToolsAgents.tool_id == target.id
        ).values(tool_name=new_name)
    )
