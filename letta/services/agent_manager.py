from typing import Dict, List, Optional, Tuple

from letta.orm import Agent as AgentModel
from letta.orm import Block as BlockModel
from letta.orm import Source as SourceModel
from letta.orm import Tool as ToolModel
from letta.schemas.agent import AgentState as PydanticAgentState
from letta.schemas.agent import AgentType
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.source import Source as PydanticSource
from letta.schemas.tool import Tool as PydanticTool
from letta.schemas.tool_rule import ToolRule as PydanticToolRule
from letta.schemas.user import User as PydanticUser
from letta.utils import enforce_types


# Static methods
def _set_tools(session, agent: AgentModel, tools: List[PydanticTool]):
    """Set tools for an agent."""
    tool_models = []
    for tool in tools:
        tool_model = session.query(ToolModel).filter_by(id=tool.id).one_or_none()
        if not tool_model:
            raise ValueError(f"Tool with id {tool.id} not found.")
        tool_models.append(tool_model)

    agent.tools = tool_models  # Replace the tools relationship


def _set_sources(session, agent: AgentModel, sources: List[PydanticSource]):
    """Set sources for an agent."""
    source_models = []
    for source in sources:
        source_model = session.query(SourceModel).filter_by(id=source.id).one_or_none()
        if not source_model:
            raise ValueError(f"Source with id {source.id} not found.")
        source_models.append(source_model)

    agent.sources = source_models  # Replace the sources relationship


def _set_blocks(session, agent: AgentModel, blocks: List[PydanticBlock]):
    """Set memory blocks for an agent."""
    block_models = []
    for block in blocks:
        block_model = session.query(BlockModel).filter_by(id=block.id).one_or_none()
        if not block_model:
            raise ValueError(f"Block with id {block.id} not found.")
        block_models.append(block_model)

    agent.memory = block_models  # Replace the memory blocks relationship


# Agent Manager Class
class AgentManager:
    """Manager class to handle business logic related to Agents."""

    def __init__(self):
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_agent(
        self,
        name: str,
        system: str,
        agent_type: AgentType,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        blocks: List[PydanticBlock],
        tools: List[PydanticTool],
        sources: List[PydanticSource],
        tags: List[str],
        description: Optional[str] = None,
        metadata_: Optional[Dict] = None,
        tool_rules: Optional[List[PydanticToolRule]] = None,
        actor: PydanticUser = None,
    ) -> PydanticAgentState:
        """Create a new agent."""
        with self.session_maker() as session:
            # Prepare the agent data
            data = {
                "name": name,
                "system": system,
                "agent_type": agent_type,
                "llm_config": llm_config,
                "embedding_config": embedding_config,
                "organization_id": actor.organization_id,
                "description": description,
                "metadata_": metadata_,
                "tool_rules": tool_rules,
            }

            # Create the new agent using SqlalchemyBase.create
            new_agent = AgentModel(**data)
            _set_tools(session, new_agent, tools)
            _set_sources(session, new_agent, sources)
            _set_blocks(session, new_agent, blocks)
            new_agent.create(session, actor=actor)

            # Convert to PydanticAgentState and return
            return new_agent.to_pydantic()

    # @enforce_types
    # def update_agent(self, agent_update: UpdateAgentState, actor: PydanticUser) -> AgentState:
    #     """Update an existing agent."""
    #     with self.session_maker() as session:
    #         # Retrieve the existing agent
    #         existing_agent = Agent.read(db_session=session, identifier=agent_update.id, actor=actor)
    #
    #         # Update only the fields provided in the update
    #         update_data = agent_update.model_dump(exclude_unset=True, exclude_none=True)
    #         for key, value in update_data.items():
    #             setattr(existing_agent, key, value)
    #
    #         # Commit the updated agent
    #         existing_agent.update(db_session=session, actor=actor)
    #         return existing_agent.to_pydantic()
    #
    @enforce_types
    def list_agents(
        self,
        actor: PydanticUser,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> Tuple[Optional[str], List[PydanticAgentState]]:
        """List agents with pagination."""
        with self.session_maker() as session:
            results = AgentModel.list(db_session=session, cursor=cursor, limit=limit, organization_id=actor.organization_id)
            return [agent.to_pydantic() for agent in results]

    @enforce_types
    def get_agent_by_id(self, agent_id: str, actor: PydanticUser) -> PydanticAgentState:
        """Fetch an agent by its ID."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def delete_agent(self, agent_id: str, actor: PydanticUser) -> None:
        """
        Deletes an agent and its associated relationships.
        Ensures proper permission checks and cascades where applicable.

        Args:
            agent_id: ID of the agent to be deleted.
            actor: User performing the action.
        """
        with self.session_maker() as session:
            # Retrieve the agent
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Delete the agent (hard delete ensures relationships are handled)
            agent.hard_delete(session)

            # Commit the session to apply changes
            session.commit()
