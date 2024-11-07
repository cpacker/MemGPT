import pytest
from sqlalchemy import delete

import letta.utils as utils
from letta.functions.functions import derive_openai_json_schema, parse_source_code
from letta.orm import Organization, Source, Tool, User
from letta.schemas.agent import CreateAgent
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.schemas.organization import Organization as PydanticOrganization
from letta.schemas.source import Source as PydanticSource
from letta.schemas.source import SourceUpdate
from letta.schemas.tool import Tool as PydanticTool
from letta.schemas.tool import ToolUpdate
from letta.services.organization_manager import OrganizationManager

utils.DEBUG = True
from letta.config import LettaConfig
from letta.schemas.user import User as PydanticUser
from letta.schemas.user import UserUpdate
from letta.server.server import SyncServer

DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    embedding_endpoint_type="hugging-face",
    embedding_endpoint="https://embeddings.memgpt.ai",
    embedding_model="letta-free",
    embedding_dim=1024,
    embedding_chunk_size=300,
    azure_endpoint=None,
    azure_version=None,
    azure_deployment=None,
)


@pytest.fixture(autouse=True)
def clear_tables(server: SyncServer):
    """Fixture to clear the organization table before each test."""
    with server.organization_manager.session_maker() as session:
        session.execute(delete(Source))
        session.execute(delete(Tool))  # Clear all records from the Tool table
        session.execute(delete(User))  # Clear all records from the user table
        session.execute(delete(Organization))  # Clear all records from the organization table
        session.commit()  # Commit the deletion


@pytest.fixture
def default_organization(server: SyncServer):
    """Fixture to create and return the default organization."""
    org = server.organization_manager.create_default_organization()
    yield org


@pytest.fixture
def default_user(server: SyncServer, default_organization):
    """Fixture to create and return the default user within the default organization."""
    user = server.user_manager.create_default_user(org_id=default_organization.id)
    yield user


@pytest.fixture
def other_user(server: SyncServer, default_organization):
    """Fixture to create and return the default user within the default organization."""
    user = server.user_manager.create_user(PydanticUser(name="other", organization_id=default_organization.id))
    yield user


@pytest.fixture
def sarah_agent(server: SyncServer, default_user, default_organization):
    """Fixture to create and return a sample agent within the default organization."""
    agent_state = server.create_agent(
        request=CreateAgent(
            name="sarah_agent",
            memory=ChatMemory(
                human="Charles",
                persona="I am a helpful assistant",
            ),
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        ),
        actor=default_user,
    )
    yield agent_state

    server.delete_agent(user_id=default_user.id, agent_id=agent_state.id)


@pytest.fixture
def charles_agent(server: SyncServer, default_user, default_organization):
    """Fixture to create and return a sample agent within the default organization."""
    agent_state = server.create_agent(
        request=CreateAgent(
            name="charles_agent",
            memory=ChatMemory(
                human="Sarah",
                persona="I am a helpful assistant",
            ),
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        ),
        actor=default_user,
    )
    yield agent_state

    server.delete_agent(user_id=default_user.id, agent_id=agent_state.id)


@pytest.fixture
def tool_fixture(server: SyncServer, default_user, default_organization):
    """Fixture to create a tool with default settings and clean up after the test."""

    def print_tool(message: str):
        """
        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.
        """
        print(message)
        return message

    # Set up tool details
    source_code = parse_source_code(print_tool)
    source_type = "python"
    description = "test_description"
    tags = ["test"]

    tool = PydanticTool(description=description, tags=tags, source_code=source_code, source_type=source_type)
    derived_json_schema = derive_openai_json_schema(source_code=tool.source_code, name=tool.name)

    derived_name = derived_json_schema["name"]
    tool.json_schema = derived_json_schema
    tool.name = derived_name

    tool = server.tool_manager.create_tool(tool, actor=default_user)

    # Yield the created tool, organization, and user for use in tests
    yield {"tool": tool}


@pytest.fixture(scope="module")
def server():
    config = LettaConfig.load()

    config.save()

    server = SyncServer(init_with_default_org_and_user=False)
    return server


# ======================================================================================================================
# Organization Manager Tests
# ======================================================================================================================
def test_list_organizations(server: SyncServer):
    # Create a new org and confirm that it is created correctly
    org_name = "test"
    org = server.organization_manager.create_organization(pydantic_org=PydanticOrganization(name=org_name))

    orgs = server.organization_manager.list_organizations()
    assert len(orgs) == 1
    assert orgs[0].name == org_name

    # Delete it after
    server.organization_manager.delete_organization_by_id(org.id)
    assert len(server.organization_manager.list_organizations()) == 0


def test_create_default_organization(server: SyncServer):
    server.organization_manager.create_default_organization()
    retrieved = server.organization_manager.get_default_organization()
    assert retrieved.name == server.organization_manager.DEFAULT_ORG_NAME


def test_update_organization_name(server: SyncServer):
    org_name_a = "a"
    org_name_b = "b"
    org = server.organization_manager.create_organization(pydantic_org=PydanticOrganization(name=org_name_a))
    assert org.name == org_name_a
    org = server.organization_manager.update_organization_name_using_id(org_id=org.id, name=org_name_b)
    assert org.name == org_name_b


def test_list_organizations_pagination(server: SyncServer):
    server.organization_manager.create_organization(pydantic_org=PydanticOrganization(name="a"))
    server.organization_manager.create_organization(pydantic_org=PydanticOrganization(name="b"))

    orgs_x = server.organization_manager.list_organizations(limit=1)
    assert len(orgs_x) == 1

    orgs_y = server.organization_manager.list_organizations(cursor=orgs_x[0].id, limit=1)
    assert len(orgs_y) == 1
    assert orgs_y[0].name != orgs_x[0].name

    orgs = server.organization_manager.list_organizations(cursor=orgs_y[0].id, limit=1)
    assert len(orgs) == 0


# ======================================================================================================================
# User Manager Tests
# ======================================================================================================================
def test_list_users(server: SyncServer):
    # Create default organization
    org = server.organization_manager.create_default_organization()

    user_name = "user"
    user = server.user_manager.create_user(PydanticUser(name=user_name, organization_id=org.id))

    users = server.user_manager.list_users()
    assert len(users) == 1
    assert users[0].name == user_name

    # Delete it after
    server.user_manager.delete_user_by_id(user.id)
    assert len(server.user_manager.list_users()) == 0


def test_create_default_user(server: SyncServer):
    org = server.organization_manager.create_default_organization()
    server.user_manager.create_default_user(org_id=org.id)
    retrieved = server.user_manager.get_default_user()
    assert retrieved.name == server.user_manager.DEFAULT_USER_NAME


def test_update_user(server: SyncServer):
    # Create default organization
    default_org = server.organization_manager.create_default_organization()
    test_org = server.organization_manager.create_organization(PydanticOrganization(name="test_org"))

    user_name_a = "a"
    user_name_b = "b"

    # Assert it's been created
    user = server.user_manager.create_user(PydanticUser(name=user_name_a, organization_id=default_org.id))
    assert user.name == user_name_a

    # Adjust name
    user = server.user_manager.update_user(UserUpdate(id=user.id, name=user_name_b))
    assert user.name == user_name_b
    assert user.organization_id == OrganizationManager.DEFAULT_ORG_ID

    # Adjust org id
    user = server.user_manager.update_user(UserUpdate(id=user.id, organization_id=test_org.id))
    assert user.name == user_name_b
    assert user.organization_id == test_org.id


# ======================================================================================================================
# Tool Manager Tests
# ======================================================================================================================
def test_create_tool(server: SyncServer, tool_fixture, default_user, default_organization):
    tool = tool_fixture["tool"]

    # Assertions to ensure the created tool matches the expected values
    assert tool.created_by_id == default_user.id
    assert tool.organization_id == default_organization.id


def test_get_tool_by_id(server: SyncServer, tool_fixture, default_user):
    tool = tool_fixture["tool"]

    # Fetch the tool by ID using the manager method
    fetched_tool = server.tool_manager.get_tool_by_id(tool.id, actor=default_user)

    # Assertions to check if the fetched tool matches the created tool
    assert fetched_tool.id == tool.id
    assert fetched_tool.name == tool.name
    assert fetched_tool.description == tool.description
    assert fetched_tool.tags == tool.tags
    assert fetched_tool.source_code == tool.source_code
    assert fetched_tool.source_type == tool.source_type


def test_get_tool_with_actor(server: SyncServer, tool_fixture, default_user):
    tool = tool_fixture["tool"]

    # Fetch the tool by name and organization ID
    fetched_tool = server.tool_manager.get_tool_by_name(tool.name, actor=default_user)

    # Assertions to check if the fetched tool matches the created tool
    assert fetched_tool.id == tool.id
    assert fetched_tool.name == tool.name
    assert fetched_tool.created_by_id == default_user.id
    assert fetched_tool.description == tool.description
    assert fetched_tool.tags == tool.tags
    assert fetched_tool.source_code == tool.source_code
    assert fetched_tool.source_type == tool.source_type


def test_list_tools(server: SyncServer, tool_fixture, default_user):
    tool = tool_fixture["tool"]

    # List tools (should include the one created by the fixture)
    tools = server.tool_manager.list_tools(actor=default_user)

    # Assertions to check that the created tool is listed
    assert len(tools) == 1
    assert any(t.id == tool.id for t in tools)


def test_update_tool_by_id(server: SyncServer, tool_fixture, default_user):
    tool = tool_fixture["tool"]
    updated_description = "updated_description"

    # Create a ToolUpdate object to modify the tool's description
    tool_update = ToolUpdate(description=updated_description)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(tool.id, tool_update, actor=default_user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(tool.id, actor=default_user)

    # Assertions to check if the update was successful
    assert updated_tool.description == updated_description


def test_update_tool_source_code_refreshes_schema_and_name(server: SyncServer, tool_fixture, default_user):
    def counter_tool(counter: int):
        """
        Args:
            counter (int): The counter to count to.

        Returns:
            bool: If it successfully counted to the counter.
        """
        for c in range(counter):
            print(c)

        return True

    # Test begins
    tool = tool_fixture["tool"]
    og_json_schema = tool.json_schema

    source_code = parse_source_code(counter_tool)

    # Create a ToolUpdate object to modify the tool's source_code
    tool_update = ToolUpdate(source_code=source_code)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(tool.id, tool_update, actor=default_user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(tool.id, actor=default_user)

    # Assertions to check if the update was successful, and json_schema is updated as well
    assert updated_tool.source_code == source_code
    assert updated_tool.json_schema != og_json_schema

    new_schema = derive_openai_json_schema(source_code=updated_tool.source_code)
    assert updated_tool.json_schema == new_schema


def test_update_tool_source_code_refreshes_schema_only(server: SyncServer, tool_fixture, default_user):
    def counter_tool(counter: int):
        """
        Args:
            counter (int): The counter to count to.

        Returns:
            bool: If it successfully counted to the counter.
        """
        for c in range(counter):
            print(c)

        return True

    # Test begins
    tool = tool_fixture["tool"]
    og_json_schema = tool.json_schema

    source_code = parse_source_code(counter_tool)
    name = "counter_tool"

    # Create a ToolUpdate object to modify the tool's source_code
    tool_update = ToolUpdate(name=name, source_code=source_code)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(tool.id, tool_update, actor=default_user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(tool.id, actor=default_user)

    # Assertions to check if the update was successful, and json_schema is updated as well
    assert updated_tool.source_code == source_code
    assert updated_tool.json_schema != og_json_schema

    new_schema = derive_openai_json_schema(source_code=updated_tool.source_code, name=updated_tool.name)
    assert updated_tool.json_schema == new_schema
    assert updated_tool.name == name


def test_update_tool_multi_user(server: SyncServer, tool_fixture, default_user, other_user):
    tool = tool_fixture["tool"]
    updated_description = "updated_description"

    # Create a ToolUpdate object to modify the tool's description
    tool_update = ToolUpdate(description=updated_description)

    # Update the tool using the manager method, but WITH THE OTHER USER'S ID!
    server.tool_manager.update_tool_by_id(tool.id, tool_update, actor=other_user)

    # Check that the created_by and last_updated_by fields are correct
    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(tool.id, actor=default_user)

    assert updated_tool.last_updated_by_id == other_user.id
    assert updated_tool.created_by_id == default_user.id


def test_delete_tool_by_id(server: SyncServer, tool_fixture, default_user):
    tool = tool_fixture["tool"]

    # Delete the tool using the manager method
    server.tool_manager.delete_tool_by_id(tool.id, actor=default_user)

    tools = server.tool_manager.list_tools(actor=default_user)
    assert len(tools) == 0


# ======================================================================================================================
# Source Manager Tests
# ======================================================================================================================


def test_create_source(server: SyncServer, default_user):
    """Test creating a new source."""
    source_pydantic = PydanticSource(
        name="Test Source",
        description="This is a test source.",
        metadata_={"type": "test"},
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
    )
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)

    # Assertions to check the created source
    assert source.name == source_pydantic.name
    assert source.description == source_pydantic.description
    assert source.metadata_ == source_pydantic.metadata_
    assert source.organization_id == default_user.organization_id


def test_update_source(server: SyncServer, default_user):
    """Test updating an existing source."""
    source_pydantic = PydanticSource(name="Original Source", description="Original description", embedding_config=DEFAULT_EMBEDDING_CONFIG)
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)

    # Update the source
    update_data = SourceUpdate(name="Updated Source", description="Updated description", metadata_={"type": "updated"})
    updated_source = server.source_manager.update_source(source_id=source.id, source_update=update_data, actor=default_user)

    # Assertions to verify update
    assert updated_source.name == update_data.name
    assert updated_source.description == update_data.description
    assert updated_source.metadata_ == update_data.metadata_


def test_delete_source(server: SyncServer, default_user):
    """Test deleting a source."""
    source_pydantic = PydanticSource(
        name="To Delete", description="This source will be deleted.", embedding_config=DEFAULT_EMBEDDING_CONFIG
    )
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)

    # Delete the source
    deleted_source = server.source_manager.delete_source(source_id=source.id, actor=default_user)

    # Assertions to verify deletion
    assert deleted_source.id == source.id

    # Verify that the source no longer appears in list_sources
    sources = server.source_manager.list_sources(actor=default_user)
    assert len(sources) == 0


def test_list_sources(server: SyncServer, default_user):
    """Test listing sources with pagination."""
    # Create multiple sources
    server.source_manager.create_source(PydanticSource(name="Source 1", embedding_config=DEFAULT_EMBEDDING_CONFIG), actor=default_user)
    server.source_manager.create_source(PydanticSource(name="Source 2", embedding_config=DEFAULT_EMBEDDING_CONFIG), actor=default_user)

    # List sources without pagination
    sources = server.source_manager.list_sources(actor=default_user)
    assert len(sources) == 2

    # List sources with pagination
    paginated_sources = server.source_manager.list_sources(actor=default_user, limit=1)
    assert len(paginated_sources) == 1

    # Ensure cursor-based pagination works
    next_page = server.source_manager.list_sources(actor=default_user, cursor=paginated_sources[-1].id, limit=1)
    assert len(next_page) == 1
    assert next_page[0].name != paginated_sources[0].name


def test_get_source_by_id(server: SyncServer, default_user):
    """Test retrieving a source by ID."""
    source_pydantic = PydanticSource(
        name="Retrieve by ID", description="Test source for ID retrieval", embedding_config=DEFAULT_EMBEDDING_CONFIG
    )
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)

    # Retrieve the source by ID
    retrieved_source = server.source_manager.get_source_by_id(source_id=source.id, actor=default_user)

    # Assertions to verify the retrieved source matches the created one
    assert retrieved_source.id == source.id
    assert retrieved_source.name == source.name
    assert retrieved_source.description == source.description


def test_get_source_by_name(server: SyncServer, default_user):
    """Test retrieving a source by name."""
    source_pydantic = PydanticSource(
        name="Unique Source", description="Test source for name retrieval", embedding_config=DEFAULT_EMBEDDING_CONFIG
    )
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)

    # Retrieve the source by name
    retrieved_source = server.source_manager.get_source_by_name(source_name=source.name, actor=default_user)

    # Assertions to verify the retrieved source matches the created one
    assert retrieved_source.name == source.name
    assert retrieved_source.description == source.description


def test_update_source_no_changes(server: SyncServer, default_user):
    """Test update_source with no actual changes to verify logging and response."""
    source_pydantic = PydanticSource(name="No Change Source", description="No changes", embedding_config=DEFAULT_EMBEDDING_CONFIG)
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)

    # Attempt to update the source with identical data
    update_data = SourceUpdate(name="No Change Source", description="No changes")
    updated_source = server.source_manager.update_source(source_id=source.id, source_update=update_data, actor=default_user)

    # Assertions to ensure the update returned the source but made no modifications
    assert updated_source.id == source.id
    assert updated_source.name == source.name
    assert updated_source.description == source.description


# ======================================================================================================================
# AgentsTagsManager Tests
# ======================================================================================================================


def test_add_tag_to_agent(server: SyncServer, sarah_agent, default_user):
    # Add a tag to the agent
    tag_name = "test_tag"
    tag_association = server.agents_tags_manager.add_tag_to_agent(agent_id=sarah_agent.id, tag=tag_name, actor=default_user)

    # Assert that the tag association was created correctly
    assert tag_association.agent_id == sarah_agent.id
    assert tag_association.tag == tag_name


def test_add_duplicate_tag_to_agent(server: SyncServer, sarah_agent, default_user):
    # Add the same tag twice to the agent
    tag_name = "test_tag"
    first_tag = server.agents_tags_manager.add_tag_to_agent(agent_id=sarah_agent.id, tag=tag_name, actor=default_user)
    duplicate_tag = server.agents_tags_manager.add_tag_to_agent(agent_id=sarah_agent.id, tag=tag_name, actor=default_user)

    # Assert that the second addition returns the existing tag without creating a duplicate
    assert first_tag.agent_id == duplicate_tag.agent_id
    assert first_tag.tag == duplicate_tag.tag

    # Get all the tags belonging to the agent
    tags = server.agents_tags_manager.get_tags_for_agent(agent_id=sarah_agent.id, actor=default_user)
    assert len(tags) == 1
    assert tags[0] == first_tag.tag


def test_delete_tag_from_agent(server: SyncServer, sarah_agent, default_user):
    # Add a tag, then delete it
    tag_name = "test_tag"
    server.agents_tags_manager.add_tag_to_agent(agent_id=sarah_agent.id, tag=tag_name, actor=default_user)
    server.agents_tags_manager.delete_tag_from_agent(agent_id=sarah_agent.id, tag=tag_name, actor=default_user)

    # Assert the tag was deleted
    agent_tags = server.agents_tags_manager.get_agents_by_tag(tag=tag_name, actor=default_user)
    assert sarah_agent.id not in agent_tags


def test_delete_nonexistent_tag_from_agent(server: SyncServer, sarah_agent, default_user):
    # Attempt to delete a tag that doesn't exist
    tag_name = "nonexistent_tag"
    with pytest.raises(ValueError, match=f"Tag '{tag_name}' not found for agent '{sarah_agent.id}'"):
        server.agents_tags_manager.delete_tag_from_agent(agent_id=sarah_agent.id, tag=tag_name, actor=default_user)


def test_delete_tag_from_nonexistent_agent(server: SyncServer, default_user):
    # Attempt to delete a tag that doesn't exist
    tag_name = "nonexistent_tag"
    agent_id = "abc"
    with pytest.raises(ValueError, match=f"Tag '{tag_name}' not found for agent '{agent_id}'"):
        server.agents_tags_manager.delete_tag_from_agent(agent_id=agent_id, tag=tag_name, actor=default_user)


def test_get_agents_by_tag(server: SyncServer, sarah_agent, charles_agent, default_user, default_organization):
    # Add a shared tag to multiple agents
    tag_name = "shared_tag"

    # Add the same tag to both agents
    server.agents_tags_manager.add_tag_to_agent(agent_id=sarah_agent.id, tag=tag_name, actor=default_user)
    server.agents_tags_manager.add_tag_to_agent(agent_id=charles_agent.id, tag=tag_name, actor=default_user)

    # Retrieve agents by tag
    agent_ids = server.agents_tags_manager.get_agents_by_tag(tag=tag_name, actor=default_user)

    # Assert that both agents are returned for the tag
    assert sarah_agent.id in agent_ids
    assert charles_agent.id in agent_ids
    assert len(agent_ids) == 2

    # Delete tags from only sarah agent
    server.agents_tags_manager.delete_all_tags_from_agent(agent_id=sarah_agent.id, actor=default_user)
    agent_ids = server.agents_tags_manager.get_agents_by_tag(tag=tag_name, actor=default_user)
    # Assert that both agents are returned for the tag
    assert sarah_agent.id not in agent_ids
    assert charles_agent.id in agent_ids
    assert len(agent_ids) == 1
