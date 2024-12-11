from typing import Dict, List, Optional, Tuple

from letta.orm import Agent as AgentModel
from letta.orm import AgentsTags
from letta.orm import Block as BlockModel
from letta.orm import Source as SourceModel
from letta.orm import Tool as ToolModel
from letta.prompts import gpt_system
from letta.schemas.agent import AgentState as PydanticAgentState
from letta.schemas.agent import AgentType, CreateAgent, UpdateAgent
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.tool_rule import ToolRule as PydanticToolRule
from letta.schemas.user import User as PydanticUser
from letta.services.block_manager import BlockManager
from letta.services.source_manager import SourceManager
from letta.services.tool_manager import ToolManager
from letta.utils import enforce_types


# Static methods
def _process_relationship(
    session, agent: AgentModel, relationship_name: str, model_class, item_ids: List[str], allow_partial=False, replace=True
):
    """
    Generalized function to handle relationships like tools, sources, and blocks using item IDs.

    Args:
        session: The database session.
        agent: The AgentModel instance.
        relationship_name: The name of the relationship attribute (e.g., 'tools', 'sources').
        model_class: The ORM class corresponding to the related items.
        item_ids: List of IDs to set or update.
        allow_partial: If True, allows missing items without raising errors.
        replace: If True, replaces the entire relationship; otherwise, extends it.

    Raises:
        ValueError: If `allow_partial` is False and some IDs are missing.
    """
    current_relationship = getattr(agent, relationship_name, [])
    if not item_ids:
        if replace:
            setattr(agent, relationship_name, [])
        return

    # Retrieve models for the provided IDs
    found_items = session.query(model_class).filter(model_class.id.in_(item_ids)).all()

    # Validate all items are found if allow_partial is False
    if not allow_partial and len(found_items) != len(item_ids):
        missing = set(item_ids) - {item.id for item in found_items}
        raise ValueError(f"Items not found in {relationship_name}: {missing}")

    if replace:
        # Replace the relationship
        setattr(agent, relationship_name, found_items)
    else:
        # Extend the relationship (only add new items)
        current_ids = {item.id for item in current_relationship}
        new_items = [item for item in found_items if item.id not in current_ids]
        current_relationship.extend(new_items)


def _process_tags(agent: AgentModel, tags: List[str], replace=True):
    """
    Handles tags for an agent.

    Args:
        agent: The AgentModel instance.
        tags: List of tags to set or update.
        replace: If True, replaces all tags; otherwise, extends them.
    """
    if not tags:
        if replace:
            agent.tags = []
        return

    # Ensure tags are unique and prepare for replacement/extension
    new_tags = {AgentsTags(agent_id=agent.id, tag=tag) for tag in set(tags)}
    if replace:
        agent.tags = list(new_tags)
    else:
        existing_tags = {t.tag for t in agent.tags}
        agent.tags.extend([tag for tag in new_tags if tag.tag not in existing_tags])


def derive_system_message(agent_type: AgentType, system: Optional[str] = None):
    if system is None:
        # TODO: don't hardcode
        if agent_type == AgentType.memgpt_agent:
            system = gpt_system.get_system_text("memgpt_chat")
        elif agent_type == AgentType.o1_agent:
            system = gpt_system.get_system_text("memgpt_modified_o1")
        elif agent_type == AgentType.offline_memory_agent:
            system = gpt_system.get_system_text("memgpt_offline_memory")
        elif agent_type == AgentType.chat_only_agent:
            system = gpt_system.get_system_text("memgpt_convo_only")
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

    return system


# Agent Manager Class
# TODO: Make the actor REQUIRED!
class AgentManager:
    """Manager class to handle business logic related to Agents."""

    def __init__(self):
        from letta.server.server import db_context

        self.session_maker = db_context
        self.block_manager = BlockManager()
        self.tool_manager = ToolManager()
        self.source_manager = SourceManager()

    @enforce_types
    def create_agent(
        self,
        agent_create: CreateAgent,
        actor: Optional[PydanticUser] = None,
    ) -> PydanticAgentState:
        system = derive_system_message(agent_type=agent_create.agent_type, system=agent_create.system)

        # TODO:
        # create blocks (note: cannot be linked into the agent_id is created)
        block_ids = list(agent_create.block_ids or [])  # Create a local copy to avoid modifying the original
        for create_block in agent_create.memory_blocks:
            block = self.block_manager.create_or_update_block(PydanticBlock(**create_block.model_dump()), actor=actor)
            block_ids.append(block.id)

        return self._create_agent(
            name=agent_create.name,
            system=system,
            agent_type=agent_create.agent_type,
            llm_config=agent_create.llm_config,
            embedding_config=agent_create.embedding_config,
            block_ids=block_ids,
            tool_ids=agent_create.tool_ids or [],
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
        actor: Optional[PydanticUser] = None,
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
    def update_agent(self, agent_id: str, agent_update: UpdateAgent, actor: Optional[PydanticUser] = None) -> PydanticAgentState:
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
        actor: Optional[PydanticUser] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> Tuple[Optional[str], List[PydanticAgentState]]:
        """List agents with pagination."""
        with self.session_maker() as session:
            results = AgentModel.list(db_session=session, cursor=cursor, limit=limit, organization_id=actor.organization_id)
            return [agent.to_pydantic() for agent in results]

    @enforce_types
    def get_agent_by_id(self, agent_id: str, actor: Optional[PydanticUser] = None) -> PydanticAgentState:
        """Fetch an agent by its ID."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def get_agent_by_name(self, agent_name: str, actor: Optional[PydanticUser] = None) -> PydanticAgentState:
        """Fetch an agent by its ID."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, name=agent_name, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def delete_agent(self, agent_id: str, actor: Optional[PydanticUser] = None) -> None:
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
