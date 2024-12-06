import warnings
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from letta.orm.errors import NoResultFound
from letta.orm.organization import Organization
from letta.orm.tool import Tool
from letta.orm.tools_agents import ToolsAgents as ToolsAgentsModel
from letta.schemas.tools_agents import ToolsAgents as PydanticToolsAgents

class ToolsAgentsManager:
    """Manages the relationship between tools and agents."""

    def __init__(self):
        from letta.server.server import db_context
        self.session_maker = db_context

    def add_tool_to_agent(self, agent_id: str, tool_id: str, tool_name: str) -> PydanticToolsAgents:
        """Add a tool to an agent.
        
        When a tool is added to an agent, it will be added to all agents in the same organization.
        """
        with self.session_maker() as session:
            try:
                # Check if the tool-agent combination already exists for this agent
                tools_agents_record = ToolsAgentsModel.read(db_session=session, agent_id=agent_id, tool_name=tool_name)
                warnings.warn(f"Tool name '{tool_name}' already exists for agent '{agent_id}'.")
            except NoResultFound:
                tools_agents_record = PydanticToolsAgents(agent_id=agent_id, tool_id=tool_id, tool_name=tool_name)
                tools_agents_record = ToolsAgentsModel(**tools_agents_record.model_dump(exclude_none=True))
                tools_agents_record.create(session)

            return tools_agents_record.to_pydantic()

    def remove_tool_with_name_from_agent(self, agent_id: str, tool_name: str) -> None:
        """Remove a tool from an agent by its name.
        
        When a tool is removed from an agent, it will be removed from all agents in the same organization.
        """
        with self.session_maker() as session:
            try:
                # Find and delete the tool-agent association for the agent
                tools_agents_record = ToolsAgentsModel.read(db_session=session, agent_id=agent_id, tool_name=tool_name)
                tools_agents_record.hard_delete(session)
                return tools_agents_record.to_pydantic()
            except NoResultFound:
                raise ValueError(f"Tool name '{tool_name}' not found for agent '{agent_id}'.")

    def remove_tool_with_id_from_agent(self, agent_id: str, tool_id: str) -> PydanticToolsAgents:
        """Remove a tool with an ID from an agent."""
        with self.session_maker() as session:
            try:
                tools_agents_record = ToolsAgentsModel.read(db_session=session, agent_id=agent_id, tool_id=tool_id)
                tools_agents_record.hard_delete(session)
                return tools_agents_record.to_pydantic()
            except NoResultFound:
                raise ValueError(f"Tool ID '{tool_id}' not found for agent '{agent_id}'.")

    def list_tool_ids_for_agent(self, agent_id: str) -> List[str]:
        """List all tool IDs associated with a specific agent."""
        with self.session_maker() as session:
            tools_agents_record = ToolsAgentsModel.list(db_session=session, agent_id=agent_id)
            return [record.tool_id for record in tools_agents_record]

    def list_tool_names_for_agent(self, agent_id: str) -> List[str]:
        """List all tool names associated with a specific agent."""
        with self.session_maker() as session:
            tools_agents_record = ToolsAgentsModel.list(db_session=session, agent_id=agent_id)
            return [record.tool_name for record in tools_agents_record]

    def list_agent_ids_with_tool(self, tool_id: str) -> List[str]:
        """List all agents associated with a specific tool."""
        with self.session_maker() as session:
            tools_agents_record = ToolsAgentsModel.list(db_session=session, tool_id=tool_id)
            return [record.agent_id for record in tools_agents_record]

    def get_tool_id_for_name(self, agent_id: str, tool_name: str) -> str:
        """Get the tool ID for a specific tool name for an agent."""
        with self.session_maker() as session:
            try:
                tools_agents_record = ToolsAgentsModel.read(db_session=session, agent_id=agent_id, tool_name=tool_name)
                return tools_agents_record.tool_id
            except NoResultFound:
                raise ValueError(f"Tool name '{tool_name}' not found for agent '{agent_id}'.")

    def remove_all_agent_tools(self, agent_id: str) -> None:
        """Remove all tools associated with an agent."""
        with self.session_maker() as session:
            tools_agents_records = ToolsAgentsModel.list(db_session=session, agent_id=agent_id)
            for record in tools_agents_records:
                record.hard_delete(session)