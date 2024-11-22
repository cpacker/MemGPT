import warnings
from typing import List

from letta.orm.blocks_agents import BlocksAgents as BlocksAgentsModel
from letta.orm.errors import NoResultFound
from letta.schemas.blocks_agents import BlocksAgents as PydanticBlocksAgents
from letta.utils import enforce_types


# TODO: DELETE THIS ASAP
# TODO: So we have a patch where we manually specify CRUD operations
# TODO: This is because Agent is NOT migrated to the ORM yet
# TODO: Once we migrate Agent to the ORM, we should deprecate any agents relationship table managers
class BlocksAgentsManager:
    """Manager class to handle business logic related to Blocks and Agents."""

    def __init__(self):
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def add_block_to_agent(self, agent_id: str, block_id: str, block_label: str) -> PydanticBlocksAgents:
        """Add a block to an agent. If the label already exists on that agent, this will error."""
        with self.session_maker() as session:
            try:
                # Check if the block-label combination already exists for this agent
                blocks_agents_record = BlocksAgentsModel.read(db_session=session, agent_id=agent_id, block_label=block_label)
                warnings.warn(f"Block label '{block_label}' already exists for agent '{agent_id}'.")
            except NoResultFound:
                blocks_agents_record = PydanticBlocksAgents(agent_id=agent_id, block_id=block_id, block_label=block_label)
                blocks_agents_record = BlocksAgentsModel(**blocks_agents_record.model_dump(exclude_none=True))
                blocks_agents_record.create(session)

            return blocks_agents_record.to_pydantic()

    @enforce_types
    def remove_block_with_label_from_agent(self, agent_id: str, block_label: str) -> PydanticBlocksAgents:
        """Remove a block with a label from an agent."""
        with self.session_maker() as session:
            try:
                # Find and delete the block-label association for the agent
                blocks_agents_record = BlocksAgentsModel.read(db_session=session, agent_id=agent_id, block_label=block_label)
                blocks_agents_record.hard_delete(session)
                return blocks_agents_record.to_pydantic()
            except NoResultFound:
                raise ValueError(f"Block label '{block_label}' not found for agent '{agent_id}'.")

    @enforce_types
    def remove_block_with_id_from_agent(self, agent_id: str, block_id: str) -> PydanticBlocksAgents:
        """Remove a block with a label from an agent."""
        with self.session_maker() as session:
            try:
                # Find and delete the block-label association for the agent
                blocks_agents_record = BlocksAgentsModel.read(db_session=session, agent_id=agent_id, block_id=block_id)
                blocks_agents_record.hard_delete(session)
                return blocks_agents_record.to_pydantic()
            except NoResultFound:
                raise ValueError(f"Block id '{block_id}' not found for agent '{agent_id}'.")

    @enforce_types
    def update_block_id_for_agent(self, agent_id: str, block_label: str, new_block_id: str) -> PydanticBlocksAgents:
        """Update the block ID for a specific block label for an agent."""
        with self.session_maker() as session:
            try:
                blocks_agents_record = BlocksAgentsModel.read(db_session=session, agent_id=agent_id, block_label=block_label)
                blocks_agents_record.block_id = new_block_id
                return blocks_agents_record.to_pydantic()
            except NoResultFound:
                raise ValueError(f"Block label '{block_label}' not found for agent '{agent_id}'.")

    @enforce_types
    def list_block_ids_for_agent(self, agent_id: str) -> List[str]:
        """List all blocks associated with a specific agent."""
        with self.session_maker() as session:
            blocks_agents_record = BlocksAgentsModel.list(db_session=session, agent_id=agent_id)
            return [record.block_id for record in blocks_agents_record]

    @enforce_types
    def list_agent_ids_with_block(self, block_id: str) -> List[str]:
        """List all agents associated with a specific block."""
        with self.session_maker() as session:
            blocks_agents_record = BlocksAgentsModel.list(db_session=session, block_id=block_id)
            return [record.agent_id for record in blocks_agents_record]
