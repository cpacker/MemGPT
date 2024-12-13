from typing import List, Optional

from letta.orm.agent import Agent as AgentModel
from letta.orm.agents_tags import AgentsTags
from letta.orm.errors import NoResultFound
from letta.prompts import gpt_system
from letta.schemas.agent import AgentType


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
        raise NoResultFound(f"Items not found in {relationship_name}: {missing}")

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
