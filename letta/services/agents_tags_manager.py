from typing import List

from letta.orm.agents_tags import AgentsTags as TagsAgentsModel
from letta.orm.errors import NoResultFound
from letta.orm.organization import Organization as OrganizationModel
from letta.schemas.agents_tags import AgentsTags as PydanticAgentsTags
from letta.schemas.user import User as PydanticUser
from letta.utils import enforce_types


class AgentsTagsManager:
    """Manager class to handle business logic related to Tags."""

    def __init__(self):
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def add_tag_to_agent(self, agent_id: str, tag: str, actor: PydanticUser) -> PydanticAgentsTags:
        """Add a tag to an agent."""
        with self.session_maker() as session:
            # Check if the tag already exists for this agent
            existing_tag = TagsAgentsModel.list(
                db_session=session,
                agent_id=agent_id,
                tag=tag,
                _organization_id=OrganizationModel.get_uid_from_identifier(actor.organization_id),
            )
            if existing_tag:
                return existing_tag[0].to_pydantic()  # Return existing tag if already present
            else:
                # Create new tag for the agent
                new_tag = TagsAgentsModel(agent_id=agent_id, tag=tag, organization_id=actor.organization_id)
                new_tag.create(session)
                return new_tag.to_pydantic()

    @enforce_types
    def delete_tag_from_agent(self, agent_id: str, tag: str, actor: PydanticUser):
        """Delete a tag from an agent."""
        with self.session_maker() as session:
            try:
                # Retrieve and delete the tag association
                tag_association = TagsAgentsModel.read(db_session=session, agent_id=agent_id, tag=tag, actor=actor)
                tag_association.delete(session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Tag '{tag}' not found for agent '{agent_id}'.")

    @enforce_types
    def get_agents_by_tag(self, tag: str, actor: PydanticUser) -> List[str]:
        """Retrieve all agent IDs associated with a specific tag."""
        with self.session_maker() as session:
            # Query for all agents with the given tag
            agents_with_tag = TagsAgentsModel.list(
                db_session=session, tag=tag, _organization_id=OrganizationModel.get_uid_from_identifier(actor.organization_id)
            )
            return [record.agent_id for record in agents_with_tag]

    @enforce_types
    def get_tags_for_agent(self, agent_id: str, actor: PydanticUser) -> List[str]:
        """Retrieve all tags associated with a specific agent."""
        with self.session_maker() as session:
            # Query for all tags associated with the given agent
            tags_for_agent = TagsAgentsModel.list(
                db_session=session, agent_id=agent_id, _organization_id=OrganizationModel.get_uid_from_identifier(actor.organization_id)
            )
            return [record.tag for record in tags_for_agent]
