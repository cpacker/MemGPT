from typing import Dict, List, Optional

from letta.constants import BASE_MEMORY_TOOLS, BASE_TOOLS
from letta.orm import Agent as AgentModel
from letta.orm import Block as BlockModel
from letta.orm import Source as SourceModel
from letta.orm import Tool as ToolModel
from letta.orm.errors import NoResultFound
from letta.schemas.agent import AgentState as PydanticAgentState
from letta.schemas.agent import AgentType, CreateAgent, UpdateAgent
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.source import Source as PydanticSource
from letta.schemas.tool_rule import ToolRule as PydanticToolRule
from letta.schemas.user import User as PydanticUser
from letta.services.block_manager import BlockManager
from letta.services.helpers.agent_manager_helper import (
    _process_relationship,
    _process_tags,
    derive_system_message,
)
from letta.services.passage_manager import PassageManager
from letta.services.source_manager import SourceManager
from letta.services.tool_manager import ToolManager
from letta.utils import enforce_types


# Agent Manager Class
class AgentManager:
    """Manager class to handle business logic related to Agents."""

    def __init__(self):
        from letta.server.server import db_context

        self.session_maker = db_context
        self.block_manager = BlockManager()
        self.tool_manager = ToolManager()
        self.source_manager = SourceManager()

    # ======================================================================================================================
    # Basic CRUD operations
    # ======================================================================================================================
    @enforce_types
    def create_agent(
        self,
        agent_create: CreateAgent,
        actor: PydanticUser,
    ) -> PydanticAgentState:
        system = derive_system_message(agent_type=agent_create.agent_type, system=agent_create.system)

        # create blocks (note: cannot be linked into the agent_id is created)
        block_ids = list(agent_create.block_ids or [])  # Create a local copy to avoid modifying the original
        for create_block in agent_create.memory_blocks:
            block = self.block_manager.create_or_update_block(PydanticBlock(**create_block.model_dump()), actor=actor)
            block_ids.append(block.id)

        # TODO: Remove this block once we deprecate the legacy `tools` field
        # create passed in `tools`
        tool_names = []
        if agent_create.include_base_tools:
            tool_names.extend(BASE_TOOLS + BASE_MEMORY_TOOLS)
        if agent_create.tools:
            tool_names.extend(agent_create.tools)

        tool_ids = agent_create.tool_ids or []
        for tool_name in tool_names:
            tool = self.tool_manager.get_tool_by_name(tool_name=tool_name, actor=actor)
            if tool:
                tool_ids.append(tool.id)
        # Remove duplicates
        tool_ids = list(set(tool_ids))

        return self._create_agent(
            name=agent_create.name,
            system=system,
            agent_type=agent_create.agent_type,
            llm_config=agent_create.llm_config,
            embedding_config=agent_create.embedding_config,
            block_ids=block_ids,
            tool_ids=tool_ids,
            source_ids=agent_create.source_ids or [],
            tags=agent_create.tags or [],
            description=agent_create.description,
            metadata_=agent_create.metadata_,
            tool_rules=agent_create.tool_rules,
            actor=actor,
        )

    @enforce_types
    def _create_agent(
        self,
        actor: PydanticUser,
        name: str,
        system: str,
        agent_type: AgentType,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        block_ids: List[str],
        tool_ids: List[str],
        source_ids: List[str],
        tags: List[str],
        description: Optional[str] = None,
        metadata_: Optional[Dict] = None,
        tool_rules: Optional[List[PydanticToolRule]] = None,
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
            _process_relationship(session, new_agent, "tools", ToolModel, tool_ids, replace=True)
            _process_relationship(session, new_agent, "sources", SourceModel, source_ids, replace=True)
            _process_relationship(session, new_agent, "core_memory", BlockModel, block_ids, replace=True)
            _process_tags(new_agent, tags, replace=True)
            new_agent.create(session, actor=actor)

            # Convert to PydanticAgentState and return
            return new_agent.to_pydantic()

    @enforce_types
    def update_agent(self, agent_id: str, agent_update: UpdateAgent, actor: PydanticUser) -> PydanticAgentState:
        """
        Update an existing agent.

        Args:
            agent_id: The ID of the agent to update.
            agent_update: UpdateAgent object containing the updated fields.
            actor: User performing the action.

        Returns:
            PydanticAgentState: The updated agent as a Pydantic model.
        """
        with self.session_maker() as session:
            # Retrieve the existing agent
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Update scalar fields directly
            scalar_fields = {"name", "system", "llm_config", "embedding_config", "message_ids", "tool_rules", "description", "metadata_"}
            for field in scalar_fields:
                value = getattr(agent_update, field, None)
                if value is not None:
                    setattr(agent, field, value)

            # Update relationships using _process_relationship and _process_tags
            if agent_update.tool_ids is not None:
                _process_relationship(session, agent, "tools", ToolModel, agent_update.tool_ids, replace=True)
            if agent_update.source_ids is not None:
                _process_relationship(session, agent, "sources", SourceModel, agent_update.source_ids, replace=True)
            if agent_update.block_ids is not None:
                _process_relationship(session, agent, "core_memory", BlockModel, agent_update.block_ids, replace=True)
            if agent_update.tags is not None:
                _process_tags(agent, agent_update.tags, replace=True)

            # Commit and refresh the agent
            agent.update(session, actor=actor)

            # Convert to PydanticAgentState and return
            return agent.to_pydantic()

    @enforce_types
    def list_agents(
        self,
        actor: PydanticUser,
        tags: Optional[List[str]] = None,
        match_all_tags: bool = False,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        **kwargs,
    ) -> List[PydanticAgentState]:
        """
        List agents that have the specified tags.
        """
        with self.session_maker() as session:
            agents = AgentModel.list(
                db_session=session,
                tags=tags,
                match_all_tags=match_all_tags,
                cursor=cursor,
                limit=limit,
                organization_id=actor.organization_id if actor else None,
                **kwargs,
            )

            return [agent.to_pydantic() for agent in agents]

    @enforce_types
    def get_agent_by_id(self, agent_id: str, actor: PydanticUser) -> PydanticAgentState:
        """Fetch an agent by its ID."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def get_agent_by_name(self, agent_name: str, actor: PydanticUser) -> PydanticAgentState:
        """Fetch an agent by its ID."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, name=agent_name, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def delete_agent(self, agent_id: str, actor: PydanticUser) -> PydanticAgentState:
        """
        Deletes an agent and its associated relationships.
        Ensures proper permission checks and cascades where applicable.

        Args:
            agent_id: ID of the agent to be deleted.
            actor: User performing the action.

        Returns:
            PydanticAgentState: The deleted agent state
        """
        with self.session_maker() as session:
            # Retrieve the agent
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # TODO: @mindy delete this piece when we have a proper passages/sources implementation
            # TODO: This is done very hacky on purpose
            # TODO: 1000 limit is also wack
            passage_manager = PassageManager()
            passage_manager.delete_passages(actor=actor, agent_id=agent_id, limit=1000)

            agent_state = agent.to_pydantic()
            agent.hard_delete(session)
            return agent_state

    # ======================================================================================================================
    # Source Management
    # ======================================================================================================================
    @enforce_types
    def attach_source(self, agent_id: str, source_id: str, actor: PydanticUser) -> None:
        """
        Attaches a source to an agent.

        Args:
            agent_id: ID of the agent to attach the source to
            source_id: ID of the source to attach
            actor: User performing the action

        Raises:
            ValueError: If either agent or source doesn't exist
            IntegrityError: If the source is already attached to the agent
        """
        with self.session_maker() as session:
            # Verify both agent and source exist and user has permission to access them
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # The _process_relationship helper already handles duplicate checking via unique constraint
            _process_relationship(
                session=session,
                agent=agent,
                relationship_name="sources",
                model_class=SourceModel,
                item_ids=[source_id],
                allow_partial=False,
                replace=False,  # Extend existing sources rather than replace
            )

            # Commit the changes
            agent.update(session, actor=actor)

    @enforce_types
    def list_attached_sources(self, agent_id: str, actor: PydanticUser) -> List[PydanticSource]:
        """
        Lists all sources attached to an agent.

        Args:
            agent_id: ID of the agent to list sources for
            actor: User performing the action

        Returns:
            List[str]: List of source IDs attached to the agent
        """
        with self.session_maker() as session:
            # Verify agent exists and user has permission to access it
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Use the lazy-loaded relationship to get sources
            return [source.to_pydantic() for source in agent.sources]

    @enforce_types
    def detach_source(self, agent_id: str, source_id: str, actor: PydanticUser) -> None:
        """
        Detaches a source from an agent.

        Args:
            agent_id: ID of the agent to detach the source from
            source_id: ID of the source to detach
            actor: User performing the action
        """
        with self.session_maker() as session:
            # Verify agent exists and user has permission to access it
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Remove the source from the relationship
            agent.sources = [s for s in agent.sources if s.id != source_id]

            # Commit the changes
            agent.update(session, actor=actor)

    # ======================================================================================================================
    # Block management
    # ======================================================================================================================
    @enforce_types
    def get_block_with_label(
        self,
        agent_id: str,
        block_label: str,
        actor: PydanticUser,
    ) -> PydanticBlock:
        """Gets a block attached to an agent by its label."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            for block in agent.core_memory:
                if block.label == block_label:
                    return block.to_pydantic()
            raise NoResultFound(f"No block with label '{block_label}' found for agent '{agent_id}'")

    @enforce_types
    def update_block_with_label(
        self,
        agent_id: str,
        block_label: str,
        new_block_id: str,
        actor: PydanticUser,
    ) -> PydanticAgentState:
        """Updates which block is assigned to a specific label for an agent."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            new_block = BlockModel.read(db_session=session, identifier=new_block_id, actor=actor)

            if new_block.label != block_label:
                raise ValueError(f"New block label '{new_block.label}' doesn't match required label '{block_label}'")

            # Remove old block with this label if it exists
            agent.core_memory = [b for b in agent.core_memory if b.label != block_label]

            # Add new block
            agent.core_memory.append(new_block)
            agent.update(session, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def attach_block(self, agent_id: str, block_id: str, actor: PydanticUser) -> PydanticAgentState:
        """Attaches a block to an agent."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            block = BlockModel.read(db_session=session, identifier=block_id, actor=actor)

            agent.core_memory.append(block)
            agent.update(session, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def detach_block(
        self,
        agent_id: str,
        block_id: str,
        actor: PydanticUser,
    ) -> PydanticAgentState:
        """Detaches a block from an agent."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            original_length = len(agent.core_memory)

            agent.core_memory = [b for b in agent.core_memory if b.id != block_id]

            if len(agent.core_memory) == original_length:
                raise NoResultFound(f"No block with id '{block_id}' found for agent '{agent_id}' with actor id: '{actor.id}'")

            agent.update(session, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def detach_block_with_label(
        self,
        agent_id: str,
        block_label: str,
        actor: PydanticUser,
    ) -> PydanticAgentState:
        """Detaches a block with the specified label from an agent."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            original_length = len(agent.core_memory)

            agent.core_memory = [b for b in agent.core_memory if b.label != block_label]

            if len(agent.core_memory) == original_length:
                raise NoResultFound(f"No block with label '{block_label}' found for agent '{agent_id}' with actor id: '{actor.id}'")

            agent.update(session, actor=actor)
            return agent.to_pydantic()
