import os
import time
from datetime import datetime, timedelta

from httpx._transports import default
from numpy import source
import pytest
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError

from letta.config import LettaConfig
from letta.embeddings import embedding_model
from letta.functions.functions import derive_openai_json_schema, parse_source_code
from letta.orm import (
    Agent,
    Block,
    BlocksAgents,
    FileMetadata,
    Job,
    Message,
    Organization,
    AgentPassage,
    SourcePassage,
    SandboxConfig,
    SandboxEnvironmentVariable,
    Source,
    SourcesAgents,
    Tool,
    ToolsAgents,
    User,
)
from letta.orm.agents_tags import AgentsTags
from letta.orm.errors import NoResultFound, UniqueConstraintViolationError
from letta.schemas.agent import CreateAgent, UpdateAgent
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.block import BlockUpdate, CreateBlock
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import JobStatus, MessageRole
from letta.schemas.file import FileMetadata as PydanticFileMetadata
from letta.schemas.job import Job as PydanticJob
from letta.schemas.job import JobUpdate
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.message import MessageCreate, MessageUpdate
from letta.schemas.organization import Organization as PydanticOrganization
from letta.schemas.passage import Passage as PydanticPassage
from letta.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfigCreate,
    SandboxConfigUpdate,
    SandboxEnvironmentVariableCreate,
    SandboxEnvironmentVariableUpdate,
    SandboxType,
)
from letta.schemas.source import Source as PydanticSource
from letta.schemas.source import SourceUpdate
from letta.schemas.tool import Tool as PydanticTool
from letta.schemas.tool import ToolUpdate
from letta.schemas.tool_rule import InitToolRule
from letta.schemas.user import User as PydanticUser
from letta.schemas.user import UserUpdate
from letta.server.server import SyncServer
from letta.services.block_manager import BlockManager
from letta.services.organization_manager import OrganizationManager
from letta.settings import tool_settings
from tests.helpers.utils import comprehensive_agent_checks

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
CREATE_DELAY_SQLITE = 1
USING_SQLITE = not bool(os.getenv("LETTA_PG_URI"))


@pytest.fixture(autouse=True)
def clear_tables(server: SyncServer):
    """Fixture to clear the organization table before each test."""
    with server.organization_manager.session_maker() as session:
        session.execute(delete(Message))
        session.execute(delete(AgentPassage))
        session.execute(delete(SourcePassage))
        session.execute(delete(Job))
        session.execute(delete(ToolsAgents))  # Clear ToolsAgents first
        session.execute(delete(BlocksAgents))
        session.execute(delete(SourcesAgents))
        session.execute(delete(AgentsTags))
        session.execute(delete(SandboxEnvironmentVariable))
        session.execute(delete(SandboxConfig))
        session.execute(delete(Block))
        session.execute(delete(FileMetadata))
        session.execute(delete(Source))
        session.execute(delete(Tool))  # Clear all records from the Tool table
        session.execute(delete(Agent))
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
def default_source(server: SyncServer, default_user):
    source_pydantic = PydanticSource(
        name="Test Source",
        description="This is a test source.",
        metadata_={"type": "test"},
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
    )
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)
    yield source


@pytest.fixture
def other_source(server: SyncServer, default_user):
    source_pydantic = PydanticSource(
        name="Another Test Source",
        description="This is yet another test source.",
        metadata_={"type": "another_test"},
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
    )
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)
    yield source


@pytest.fixture
def default_file(server: SyncServer, default_source, default_user, default_organization):
    file = server.source_manager.create_file(
        PydanticFileMetadata(file_name="test_file", organization_id=default_organization.id, source_id=default_source.id),
        actor=default_user,
    )
    yield file


@pytest.fixture
def print_tool(server: SyncServer, default_user, default_organization):
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

    # Yield the created tool
    yield tool


@pytest.fixture
def agent_passage_fixture(server: SyncServer, default_user, sarah_agent):
    """Fixture to create an agent passage."""
    passage = server.passage_manager.create_passage(
        PydanticPassage(
            text="Hello, I am an agent passage",
            agent_id=sarah_agent.id,
            organization_id=default_user.organization_id,
            embedding=[0.1],
            embedding_config=DEFAULT_EMBEDDING_CONFIG,
            metadata_={"type": "test"}
        ),
        actor=default_user
    )
    yield passage


@pytest.fixture
def source_passage_fixture(server: SyncServer, default_user, default_file, default_source):
    """Fixture to create a source passage."""
    passage = server.passage_manager.create_passage(
        PydanticPassage(
            text="Hello, I am a source passage",
            source_id=default_source.id,
            file_id=default_file.id,
            organization_id=default_user.organization_id,
            embedding=[0.1],
            embedding_config=DEFAULT_EMBEDDING_CONFIG,
            metadata_={"type": "test"}
        ),
        actor=default_user
    )
    yield passage


@pytest.fixture
def create_test_passages(server: SyncServer, default_file, default_user, sarah_agent, default_source):
    """Helper function to create test passages for all tests."""
    # Create agent passages
    passages = []
    for i in range(5):
        passage = server.passage_manager.create_passage(
            PydanticPassage(
                text=f"Agent passage {i}",
                agent_id=sarah_agent.id,
                organization_id=default_user.organization_id,
                embedding=[0.1],
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
                metadata_={"type": "test"}
            ),
            actor=default_user
        )
        passages.append(passage)
        if USING_SQLITE:
            time.sleep(CREATE_DELAY_SQLITE)

    # Create source passages
    for i in range(5):
        passage = server.passage_manager.create_passage(
            PydanticPassage(
                text=f"Source passage {i}",
                source_id=default_source.id,
                file_id=default_file.id,
                organization_id=default_user.organization_id,
                embedding=[0.1],
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
                metadata_={"type": "test"}
            ),
            actor=default_user
        )
        passages.append(passage)
        if USING_SQLITE:
            time.sleep(CREATE_DELAY_SQLITE)

    return passages


@pytest.fixture
def hello_world_message_fixture(server: SyncServer, default_user, sarah_agent):
    """Fixture to create a tool with default settings and clean up after the test."""
    # Set up message
    message = PydanticMessage(
        organization_id=default_user.organization_id,
        agent_id=sarah_agent.id,
        role="user",
        text="Hello, world!",
    )

    msg = server.message_manager.create_message(message, actor=default_user)
    yield msg


@pytest.fixture
def sandbox_config_fixture(server: SyncServer, default_user):
    sandbox_config_create = SandboxConfigCreate(
        config=E2BSandboxConfig(),
    )
    created_config = server.sandbox_config_manager.create_or_update_sandbox_config(sandbox_config_create, actor=default_user)
    yield created_config


@pytest.fixture
def sandbox_env_var_fixture(server: SyncServer, sandbox_config_fixture, default_user):
    env_var_create = SandboxEnvironmentVariableCreate(
        key="SAMPLE_VAR",
        value="sample_value",
        description="A sample environment variable for testing.",
    )
    created_env_var = server.sandbox_config_manager.create_sandbox_env_var(
        env_var_create, sandbox_config_id=sandbox_config_fixture.id, actor=default_user
    )
    yield created_env_var


@pytest.fixture
def default_block(server: SyncServer, default_user):
    """Fixture to create and return a default block."""
    block_manager = BlockManager()
    block_data = PydanticBlock(
        label="default_label",
        value="Default Block Content",
        description="A default test block",
        limit=1000,
        metadata_={"type": "test"},
    )
    block = block_manager.create_or_update_block(block_data, actor=default_user)
    yield block


@pytest.fixture
def other_block(server: SyncServer, default_user):
    """Fixture to create and return another block."""
    block_manager = BlockManager()
    block_data = PydanticBlock(
        label="other_label",
        value="Other Block Content",
        description="Another test block",
        limit=500,
        metadata_={"type": "test"},
    )
    block = block_manager.create_or_update_block(block_data, actor=default_user)
    yield block


@pytest.fixture
def other_tool(server: SyncServer, default_user, default_organization):
    def print_other_tool(message: str):
        """
        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.
        """
        print(message)
        return message

    # Set up tool details
    source_code = parse_source_code(print_other_tool)
    source_type = "python"
    description = "other_tool_description"
    tags = ["test"]

    tool = PydanticTool(description=description, tags=tags, source_code=source_code, source_type=source_type)
    derived_json_schema = derive_openai_json_schema(source_code=tool.source_code, name=tool.name)

    derived_name = derived_json_schema["name"]
    tool.json_schema = derived_json_schema
    tool.name = derived_name

    tool = server.tool_manager.create_tool(tool, actor=default_user)

    # Yield the created tool
    yield tool


@pytest.fixture
def sarah_agent(server: SyncServer, default_user, default_organization):
    """Fixture to create and return a sample agent within the default organization."""
    agent_state = server.create_agent(
        request=CreateAgent(
            name="sarah_agent",
            memory_blocks=[],
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        ),
        actor=default_user,
    )
    yield agent_state


@pytest.fixture
def charles_agent(server: SyncServer, default_user, default_organization):
    """Fixture to create and return a sample agent within the default organization."""
    agent_state = server.create_agent(
        request=CreateAgent(
            name="charles_agent",
            memory_blocks=[CreateBlock(label="human", value="Charles"), CreateBlock(label="persona", value="I am a helpful assistant")],
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        ),
        actor=default_user,
    )
    yield agent_state


@pytest.fixture
def comprehensive_test_agent_fixture(server: SyncServer, default_user, print_tool, default_source, default_block):
    memory_blocks = [CreateBlock(label="human", value="BananaBoy"), CreateBlock(label="persona", value="I am a helpful assistant")]
    create_agent_request = CreateAgent(
        system="test system",
        memory_blocks=memory_blocks,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        block_ids=[default_block.id],
        tool_ids=[print_tool.id],
        source_ids=[default_source.id],
        tags=["a", "b"],
        description="test_description",
        metadata_={"test_key": "test_value"},
        tool_rules=[InitToolRule(tool_name=print_tool.name)],
        initial_message_sequence=[MessageCreate(role=MessageRole.user, text="hello world")],
    )
    created_agent = server.agent_manager.create_agent(
        create_agent_request,
        actor=default_user,
    )

    yield created_agent, create_agent_request


@pytest.fixture(scope="module")
def server():
    config = LettaConfig.load()

    config.save()

    server = SyncServer(init_with_default_org_and_user=False)
    return server


@pytest.fixture
def agent_passages_setup(server, default_source, default_user, sarah_agent):
    """Setup fixture for agent passages tests"""
    agent_id = sarah_agent.id
    actor = default_user

    server.agent_manager.attach_source(agent_id=agent_id, source_id=default_source.id, actor=actor)

    # Create some source passages
    source_passages = []
    for i in range(3):
        passage = server.passage_manager.create_passage(
            PydanticPassage(
                organization_id=actor.organization_id,
                source_id=default_source.id,
                text=f"Source passage {i}",
                embedding=[0.1],  # Default OpenAI embedding size
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
            ),
            actor=actor
        )
        source_passages.append(passage)

    # Create some agent passages
    agent_passages = []
    for i in range(2):
        passage = server.passage_manager.create_passage(
            PydanticPassage(
                organization_id=actor.organization_id,
                agent_id=agent_id,
                text=f"Agent passage {i}",
                embedding=[0.1],  # Default OpenAI embedding size
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
            ),
            actor=actor
        )
        agent_passages.append(passage)

    yield agent_passages, source_passages

    # Cleanup
    server.source_manager.delete_source(default_source.id, actor=actor)

# ======================================================================================================================
# AgentManager Tests - Basic
# ======================================================================================================================
def test_create_get_list_agent(server: SyncServer, comprehensive_test_agent_fixture, default_user):
    # Test agent creation
    created_agent, create_agent_request = comprehensive_test_agent_fixture
    comprehensive_agent_checks(created_agent, create_agent_request)

    # Test get agent
    get_agent = server.agent_manager.get_agent_by_id(agent_id=created_agent.id, actor=default_user)
    comprehensive_agent_checks(get_agent, create_agent_request)

    # Test get agent name
    get_agent_name = server.agent_manager.get_agent_by_name(agent_name=created_agent.name, actor=default_user)
    comprehensive_agent_checks(get_agent_name, create_agent_request)

    # Test list agent
    list_agents = server.agent_manager.list_agents(actor=default_user)
    assert len(list_agents) == 1
    comprehensive_agent_checks(list_agents[0], create_agent_request)

    # Test deleting the agent
    server.agent_manager.delete_agent(get_agent.id, default_user)
    list_agents = server.agent_manager.list_agents(actor=default_user)
    assert len(list_agents) == 0


def test_update_agent(server: SyncServer, comprehensive_test_agent_fixture, other_tool, other_source, other_block, default_user):
    agent, _ = comprehensive_test_agent_fixture
    update_agent_request = UpdateAgent(
        name="train_agent",
        description="train description",
        tool_ids=[other_tool.id],
        source_ids=[other_source.id],
        block_ids=[other_block.id],
        tool_rules=[InitToolRule(tool_name=other_tool.name)],
        tags=["c", "d"],
        system="train system",
        llm_config=LLMConfig.default_config("gpt-4o-mini"),
        embedding_config=EmbeddingConfig.default_config(model_name="letta"),
        message_ids=["10", "20"],
        metadata_={"train_key": "train_value"},
    )

    updated_agent = server.agent_manager.update_agent(agent.id, update_agent_request, actor=default_user)
    comprehensive_agent_checks(updated_agent, update_agent_request)
    assert updated_agent.message_ids == update_agent_request.message_ids


# ======================================================================================================================
# AgentManager Tests - Sources Relationship
# ======================================================================================================================


def test_attach_source(server: SyncServer, sarah_agent, default_source, default_user):
    """Test attaching a source to an agent."""
    # Attach the source
    server.agent_manager.attach_source(agent_id=sarah_agent.id, source_id=default_source.id, actor=default_user)

    # Verify attachment through get_agent_by_id
    agent = server.agent_manager.get_agent_by_id(sarah_agent.id, actor=default_user)
    assert default_source.id in [s.id for s in agent.sources]

    # Verify that attaching the same source again doesn't cause issues
    server.agent_manager.attach_source(agent_id=sarah_agent.id, source_id=default_source.id, actor=default_user)
    agent = server.agent_manager.get_agent_by_id(sarah_agent.id, actor=default_user)
    assert len([s for s in agent.sources if s.id == default_source.id]) == 1


def test_list_attached_source_ids(server: SyncServer, sarah_agent, default_source, other_source, default_user):
    """Test listing source IDs attached to an agent."""
    # Initially should have no sources
    sources = server.agent_manager.list_attached_sources(sarah_agent.id, actor=default_user)
    assert len(sources) == 0

    # Attach sources
    server.agent_manager.attach_source(sarah_agent.id, default_source.id, actor=default_user)
    server.agent_manager.attach_source(sarah_agent.id, other_source.id, actor=default_user)

    # List sources and verify
    sources = server.agent_manager.list_attached_sources(sarah_agent.id, actor=default_user)
    assert len(sources) == 2
    source_ids = [s.id for s in sources]
    assert default_source.id in source_ids
    assert other_source.id in source_ids


def test_detach_source(server: SyncServer, sarah_agent, default_source, default_user):
    """Test detaching a source from an agent."""
    # Attach source
    server.agent_manager.attach_source(sarah_agent.id, default_source.id, actor=default_user)

    # Verify it's attached
    agent = server.agent_manager.get_agent_by_id(sarah_agent.id, actor=default_user)
    assert default_source.id in [s.id for s in agent.sources]

    # Detach source
    server.agent_manager.detach_source(sarah_agent.id, default_source.id, actor=default_user)

    # Verify it's detached
    agent = server.agent_manager.get_agent_by_id(sarah_agent.id, actor=default_user)
    assert default_source.id not in [s.id for s in agent.sources]

    # Verify that detaching an already detached source doesn't cause issues
    server.agent_manager.detach_source(sarah_agent.id, default_source.id, actor=default_user)


def test_attach_source_nonexistent_agent(server: SyncServer, default_source, default_user):
    """Test attaching a source to a nonexistent agent."""
    with pytest.raises(NoResultFound):
        server.agent_manager.attach_source(agent_id="nonexistent-agent-id", source_id=default_source.id, actor=default_user)


def test_attach_source_nonexistent_source(server: SyncServer, sarah_agent, default_user):
    """Test attaching a nonexistent source to an agent."""
    with pytest.raises(NoResultFound):
        server.agent_manager.attach_source(agent_id=sarah_agent.id, source_id="nonexistent-source-id", actor=default_user)


def test_detach_source_nonexistent_agent(server: SyncServer, default_source, default_user):
    """Test detaching a source from a nonexistent agent."""
    with pytest.raises(NoResultFound):
        server.agent_manager.detach_source(agent_id="nonexistent-agent-id", source_id=default_source.id, actor=default_user)


def test_list_attached_source_ids_nonexistent_agent(server: SyncServer, default_user):
    """Test listing sources for a nonexistent agent."""
    with pytest.raises(NoResultFound):
        server.agent_manager.list_attached_sources(agent_id="nonexistent-agent-id", actor=default_user)


def test_list_attached_agents(server: SyncServer, sarah_agent, charles_agent, default_source, default_user):
    """Test listing agents that have a particular source attached."""
    # Initially should have no attached agents
    attached_agents = server.source_manager.list_attached_agents(source_id=default_source.id, actor=default_user)
    assert len(attached_agents) == 0

    # Attach source to first agent
    server.agent_manager.attach_source(agent_id=sarah_agent.id, source_id=default_source.id, actor=default_user)

    # Verify one agent is now attached
    attached_agents = server.source_manager.list_attached_agents(source_id=default_source.id, actor=default_user)
    assert len(attached_agents) == 1
    assert sarah_agent.id in [a.id for a in attached_agents]

    # Attach source to second agent
    server.agent_manager.attach_source(agent_id=charles_agent.id, source_id=default_source.id, actor=default_user)

    # Verify both agents are now attached
    attached_agents = server.source_manager.list_attached_agents(source_id=default_source.id, actor=default_user)
    assert len(attached_agents) == 2
    attached_agent_ids = [a.id for a in attached_agents]
    assert sarah_agent.id in attached_agent_ids
    assert charles_agent.id in attached_agent_ids

    # Detach source from first agent
    server.agent_manager.detach_source(agent_id=sarah_agent.id, source_id=default_source.id, actor=default_user)

    # Verify only second agent remains attached
    attached_agents = server.source_manager.list_attached_agents(source_id=default_source.id, actor=default_user)
    assert len(attached_agents) == 1
    assert charles_agent.id in [a.id for a in attached_agents]


def test_list_attached_agents_nonexistent_source(server: SyncServer, default_user):
    """Test listing agents for a nonexistent source."""
    with pytest.raises(NoResultFound):
        server.source_manager.list_attached_agents(source_id="nonexistent-source-id", actor=default_user)


# ======================================================================================================================
# AgentManager Tests - Tags Relationship
# ======================================================================================================================


def test_list_agents_by_tags_match_all(server: SyncServer, sarah_agent, charles_agent, default_user):
    """Test listing agents that have ALL specified tags."""
    # Create agents with multiple tags
    server.agent_manager.update_agent(sarah_agent.id, UpdateAgent(tags=["test", "production", "gpt4"]), actor=default_user)
    server.agent_manager.update_agent(charles_agent.id, UpdateAgent(tags=["test", "development", "gpt4"]), actor=default_user)

    # Search for agents with all specified tags
    agents = server.agent_manager.list_agents(tags=["test", "gpt4"], match_all_tags=True, actor=default_user)
    assert len(agents) == 2
    agent_ids = [a.id for a in agents]
    assert sarah_agent.id in agent_ids
    assert charles_agent.id in agent_ids

    # Search for tags that only sarah_agent has
    agents = server.agent_manager.list_agents(tags=["test", "production"], match_all_tags=True, actor=default_user)
    assert len(agents) == 1
    assert agents[0].id == sarah_agent.id


def test_list_agents_by_tags_match_any(server: SyncServer, sarah_agent, charles_agent, default_user):
    """Test listing agents that have ANY of the specified tags."""
    # Create agents with different tags
    server.agent_manager.update_agent(sarah_agent.id, UpdateAgent(tags=["production", "gpt4"]), actor=default_user)
    server.agent_manager.update_agent(charles_agent.id, UpdateAgent(tags=["development", "gpt3"]), actor=default_user)

    # Search for agents with any of the specified tags
    agents = server.agent_manager.list_agents(tags=["production", "development"], match_all_tags=False, actor=default_user)
    assert len(agents) == 2
    agent_ids = [a.id for a in agents]
    assert sarah_agent.id in agent_ids
    assert charles_agent.id in agent_ids

    # Search for tags where only sarah_agent matches
    agents = server.agent_manager.list_agents(tags=["production", "nonexistent"], match_all_tags=False, actor=default_user)
    assert len(agents) == 1
    assert agents[0].id == sarah_agent.id


def test_list_agents_by_tags_no_matches(server: SyncServer, sarah_agent, charles_agent, default_user):
    """Test listing agents when no tags match."""
    # Create agents with tags
    server.agent_manager.update_agent(sarah_agent.id, UpdateAgent(tags=["production", "gpt4"]), actor=default_user)
    server.agent_manager.update_agent(charles_agent.id, UpdateAgent(tags=["development", "gpt3"]), actor=default_user)

    # Search for nonexistent tags
    agents = server.agent_manager.list_agents(tags=["nonexistent1", "nonexistent2"], match_all_tags=True, actor=default_user)
    assert len(agents) == 0

    agents = server.agent_manager.list_agents(tags=["nonexistent1", "nonexistent2"], match_all_tags=False, actor=default_user)
    assert len(agents) == 0


def test_list_agents_by_tags_with_other_filters(server: SyncServer, sarah_agent, charles_agent, default_user):
    """Test combining tag search with other filters."""
    # Create agents with specific names and tags
    server.agent_manager.update_agent(sarah_agent.id, UpdateAgent(name="production_agent", tags=["production", "gpt4"]), actor=default_user)
    server.agent_manager.update_agent(charles_agent.id, UpdateAgent(name="test_agent", tags=["production", "gpt3"]), actor=default_user)

    # List agents with specific tag and name pattern
    agents = server.agent_manager.list_agents(actor=default_user, tags=["production"], match_all_tags=True, name="production_agent")
    assert len(agents) == 1
    assert agents[0].id == sarah_agent.id


def test_list_agents_by_tags_pagination(server: SyncServer, default_user, default_organization):
    """Test pagination when listing agents by tags."""
    # Create first agent
    agent1 = server.create_agent(
        request=CreateAgent(
            name="agent1",
            tags=["pagination_test", "tag1"],
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
            memory_blocks=[],
        ),
        actor=default_user,
    )

    if USING_SQLITE:
        time.sleep(CREATE_DELAY_SQLITE)  # Ensure distinct created_at timestamps

    # Create second agent
    agent2 = server.create_agent(
        request=CreateAgent(
            name="agent2",
            tags=["pagination_test", "tag2"],
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
            memory_blocks=[],
        ),
        actor=default_user,
    )

    # Get first page
    first_page = server.agent_manager.list_agents(tags=["pagination_test"], match_all_tags=True, actor=default_user, limit=1)
    assert len(first_page) == 1
    first_agent_id = first_page[0].id

    # Get second page using cursor
    second_page = server.agent_manager.list_agents(
        tags=["pagination_test"], match_all_tags=True, actor=default_user, cursor=first_agent_id, limit=1
    )
    assert len(second_page) == 1
    assert second_page[0].id != first_agent_id

    # Verify we got both agents with no duplicates
    all_ids = {first_page[0].id, second_page[0].id}
    assert len(all_ids) == 2
    assert agent1.id in all_ids
    assert agent2.id in all_ids


# ======================================================================================================================
# AgentManager Tests - Blocks Relationship
# ======================================================================================================================


def test_attach_block(server: SyncServer, sarah_agent, default_block, default_user):
    """Test attaching a block to an agent."""
    # Attach block
    server.agent_manager.attach_block(agent_id=sarah_agent.id, block_id=default_block.id, actor=default_user)

    # Verify attachment
    agent = server.agent_manager.get_agent_by_id(sarah_agent.id, actor=default_user)
    assert len(agent.memory.blocks) == 1
    assert agent.memory.blocks[0].id == default_block.id
    assert agent.memory.blocks[0].label == default_block.label


def test_attach_block_duplicate_label(server: SyncServer, sarah_agent, default_block, other_block, default_user):
    """Test attempting to attach a block with a duplicate label."""
    # Set up both blocks with same label
    server.block_manager.update_block(default_block.id, BlockUpdate(label="same_label"), actor=default_user)
    server.block_manager.update_block(other_block.id, BlockUpdate(label="same_label"), actor=default_user)

    # Attach first block
    server.agent_manager.attach_block(agent_id=sarah_agent.id, block_id=default_block.id, actor=default_user)

    # Attempt to attach second block with same label
    with pytest.raises(IntegrityError):
        server.agent_manager.attach_block(agent_id=sarah_agent.id, block_id=other_block.id, actor=default_user)


def test_detach_block(server: SyncServer, sarah_agent, default_block, default_user):
    """Test detaching a block by ID."""
    # Set up: attach block
    server.agent_manager.attach_block(agent_id=sarah_agent.id, block_id=default_block.id, actor=default_user)

    # Detach block
    server.agent_manager.detach_block(agent_id=sarah_agent.id, block_id=default_block.id, actor=default_user)

    # Verify detachment
    agent = server.agent_manager.get_agent_by_id(sarah_agent.id, actor=default_user)
    assert len(agent.memory.blocks) == 0


def test_detach_nonexistent_block(server: SyncServer, sarah_agent, default_user):
    """Test detaching a block that isn't attached."""
    with pytest.raises(NoResultFound):
        server.agent_manager.detach_block(agent_id=sarah_agent.id, block_id="nonexistent-block-id", actor=default_user)


def test_update_block_label(server: SyncServer, sarah_agent, default_block, default_user):
    """Test updating a block's label updates the relationship."""
    # Attach block
    server.agent_manager.attach_block(agent_id=sarah_agent.id, block_id=default_block.id, actor=default_user)

    # Update block label
    new_label = "new_label"
    server.block_manager.update_block(default_block.id, BlockUpdate(label=new_label), actor=default_user)

    # Verify relationship is updated
    agent = server.agent_manager.get_agent_by_id(sarah_agent.id, actor=default_user)
    block = agent.memory.blocks[0]
    assert block.id == default_block.id
    assert block.label == new_label


def test_update_block_label_multiple_agents(server: SyncServer, sarah_agent, charles_agent, default_block, default_user):
    """Test updating a block's label updates relationships for all agents."""
    # Attach block to both agents
    server.agent_manager.attach_block(agent_id=sarah_agent.id, block_id=default_block.id, actor=default_user)
    server.agent_manager.attach_block(agent_id=charles_agent.id, block_id=default_block.id, actor=default_user)

    # Update block label
    new_label = "new_label"
    server.block_manager.update_block(default_block.id, BlockUpdate(label=new_label), actor=default_user)

    # Verify both relationships are updated
    for agent_id in [sarah_agent.id, charles_agent.id]:
        agent = server.agent_manager.get_agent_by_id(agent_id, actor=default_user)
        # Find our specific block by ID
        block = next(b for b in agent.memory.blocks if b.id == default_block.id)
        assert block.label == new_label


def test_get_block_with_label(server: SyncServer, sarah_agent, default_block, default_user):
    """Test retrieving a block by its label."""
    # Attach block
    server.agent_manager.attach_block(agent_id=sarah_agent.id, block_id=default_block.id, actor=default_user)

    # Get block by label
    block = server.agent_manager.get_block_with_label(agent_id=sarah_agent.id, block_label=default_block.label, actor=default_user)

    assert block.id == default_block.id
    assert block.label == default_block.label


# ======================================================================================================================
# Agent Manager - Passages Tests
# ======================================================================================================================

def test_agent_list_passages_basic(server, default_user, sarah_agent, agent_passages_setup):
    """Test basic listing functionality of agent passages"""
    
    all_passages = server.agent_manager.list_passages(actor=default_user, agent_id=sarah_agent.id)
    assert len(all_passages) == 5  # 3 source + 2 agent passages


def test_agent_list_passages_ordering(server, default_user, sarah_agent, agent_passages_setup):
    """Test ordering of agent passages""" 

    # Test ascending order
    asc_passages = server.agent_manager.list_passages(actor=default_user, agent_id=sarah_agent.id, ascending=True)
    assert len(asc_passages) == 5
    for i in range(1, len(asc_passages)):
        assert asc_passages[i-1].created_at <= asc_passages[i].created_at

    # Test descending order
    desc_passages = server.agent_manager.list_passages(actor=default_user, agent_id=sarah_agent.id, ascending=False)
    assert len(desc_passages) == 5
    for i in range(1, len(desc_passages)):
        assert desc_passages[i-1].created_at >= desc_passages[i].created_at


def test_agent_list_passages_pagination(server, default_user, sarah_agent, agent_passages_setup):
    """Test pagination of agent passages"""
    
    # Test limit
    limited_passages = server.agent_manager.list_passages(actor=default_user, agent_id=sarah_agent.id, limit=3)
    assert len(limited_passages) == 3

    # Test cursor-based pagination
    first_page = server.agent_manager.list_passages(actor=default_user, agent_id=sarah_agent.id, limit=2, ascending=True)
    assert len(first_page) == 2
    
    second_page = server.agent_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        cursor=first_page[-1].id,
        limit=2,
        ascending=True
    )
    assert len(second_page) == 2
    assert first_page[-1].id != second_page[0].id
    assert first_page[-1].created_at <= second_page[0].created_at


def test_agent_list_passages_text_search(server, default_user, sarah_agent, agent_passages_setup):
    """Test text search functionality of agent passages"""
    
    # Test text search for source passages
    source_text_passages = server.agent_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        query_text="Source passage"
    )
    assert len(source_text_passages) == 3

    # Test text search for agent passages
    agent_text_passages = server.agent_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        query_text="Agent passage"
    )
    assert len(agent_text_passages) == 2


def test_agent_list_passages_agent_only(server, default_user, sarah_agent, agent_passages_setup):
    """Test text search functionality of agent passages"""
    
    # Test text search for agent passages
    agent_text_passages = server.agent_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        agent_only=True
    )
    assert len(agent_text_passages) == 2


def test_agent_list_passages_filtering(server, default_user, sarah_agent, default_source, agent_passages_setup):
    """Test filtering functionality of agent passages"""
    
    # Test source filtering
    source_filtered = server.agent_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        source_id=default_source.id
    )
    assert len(source_filtered) == 3

    # Test date filtering
    now = datetime.utcnow()
    future_date = now + timedelta(days=1)
    past_date = now - timedelta(days=1)
    
    date_filtered = server.agent_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        start_date=past_date,
        end_date=future_date
    )
    assert len(date_filtered) == 5


def test_agent_list_passages_vector_search(server, default_user, sarah_agent, default_source):
    """Test vector search functionality of agent passages"""
    embed_model = embedding_model(DEFAULT_EMBEDDING_CONFIG)

    # Create passages with known embeddings
    passages = []
    
    # Create passages with different embeddings
    test_passages = [
        "I like red",
        "random text",
        "blue shoes",
    ]

    server.agent_manager.attach_source(agent_id=sarah_agent.id, source_id=default_source.id, actor=default_user)
    
    for i, text in enumerate(test_passages):
        embedding = embed_model.get_text_embedding(text)
        if i % 2 == 0:
            passage = PydanticPassage(
                text=text,
                organization_id=default_user.organization_id,
                agent_id=sarah_agent.id,
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
                embedding=embedding
            )
        else:
            passage = PydanticPassage(
                text=text,
                organization_id=default_user.organization_id,
                source_id=default_source.id,
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
                embedding=embedding
            )
        created_passage = server.passage_manager.create_passage(passage, default_user)
        passages.append(created_passage)

    # Query vector similar to "red" embedding
    query_key = "What's my favorite color?"
    
    # Test vector search with all passages
    results = server.agent_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        query_text=query_key,
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
        embed_query=True,
    )
    
    # Verify results are ordered by similarity
    assert len(results) == 3
    assert results[0].text == "I like red"
    assert "random" in results[1].text or "random" in results[2].text
    assert "blue" in results[1].text or "blue" in results[2].text

    # Test vector search with agent_only=True
    agent_only_results = server.agent_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        query_text=query_key,
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
        embed_query=True,
        agent_only=True
    )
    
    # Verify agent-only results
    assert len(agent_only_results) == 2
    assert agent_only_results[0].text == "I like red"
    assert agent_only_results[1].text == "blue shoes"


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
# Passage Manager Tests
# ======================================================================================================================

def test_passage_create_agentic(server: SyncServer, agent_passage_fixture, default_user):
    """Test creating a passage using agent_passage_fixture fixture"""
    assert agent_passage_fixture.id is not None
    assert agent_passage_fixture.text == "Hello, I am an agent passage"

    # Verify we can retrieve it
    retrieved = server.passage_manager.get_passage_by_id(
        agent_passage_fixture.id,
        actor=default_user,
    )
    assert retrieved is not None
    assert retrieved.id == agent_passage_fixture.id
    assert retrieved.text == agent_passage_fixture.text


def test_passage_create_source(server: SyncServer, source_passage_fixture, default_user):
    """Test creating a source passage."""
    assert source_passage_fixture is not None
    assert source_passage_fixture.text == "Hello, I am a source passage"

    # Verify we can retrieve it
    retrieved = server.passage_manager.get_passage_by_id(
        source_passage_fixture.id,
        actor=default_user,
    )
    assert retrieved is not None
    assert retrieved.id == source_passage_fixture.id
    assert retrieved.text == source_passage_fixture.text


def test_passage_create_invalid(server: SyncServer, agent_passage_fixture, default_user):
    """Test creating an agent passage."""
    assert agent_passage_fixture is not None
    assert agent_passage_fixture.text == "Hello, I am an agent passage"
    
    # Try to create an invalid passage (with both agent_id and source_id)
    with pytest.raises(AssertionError):
        server.passage_manager.create_passage(
            PydanticPassage(
                text="Invalid passage",
                agent_id="123",
                source_id="456",
                organization_id=default_user.organization_id,
                embedding=[0.1] * 1024,
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
            ),
            actor=default_user
        )


def test_passage_get_by_id(server: SyncServer, agent_passage_fixture, source_passage_fixture, default_user):
    """Test retrieving a passage by ID"""
    retrieved = server.passage_manager.get_passage_by_id(agent_passage_fixture.id, actor=default_user)
    assert retrieved is not None
    assert retrieved.id == agent_passage_fixture.id
    assert retrieved.text == agent_passage_fixture.text

    retrieved = server.passage_manager.get_passage_by_id(source_passage_fixture.id, actor=default_user)
    assert retrieved is not None
    assert retrieved.id == source_passage_fixture.id
    assert retrieved.text == source_passage_fixture.text


def test_passage_cascade_deletion(server: SyncServer, agent_passage_fixture, source_passage_fixture, default_user, default_source, sarah_agent):
    """Test that passages are deleted when their parent (agent or source) is deleted."""
    # Verify passages exist
    agent_passage = server.passage_manager.get_passage_by_id(agent_passage_fixture.id, default_user)
    source_passage = server.passage_manager.get_passage_by_id(source_passage_fixture.id, default_user)
    assert agent_passage is not None
    assert source_passage is not None
    
    # Delete agent and verify its passages are deleted
    server.agent_manager.delete_agent(sarah_agent.id, default_user)
    agentic_passages = server.agent_manager.list_passages(actor=default_user, agent_id=sarah_agent.id, agent_only=True)
    assert len(agentic_passages) == 0
    
    # Delete source and verify its passages are deleted
    server.source_manager.delete_source(default_source.id, default_user)
    with pytest.raises(NoResultFound):
        server.passage_manager.get_passage_by_id(source_passage_fixture.id, default_user)


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
# ToolManager Tests
# ======================================================================================================================
def test_create_tool(server: SyncServer, print_tool, default_user, default_organization):
    # Assertions to ensure the created tool matches the expected values
    assert print_tool.created_by_id == default_user.id
    assert print_tool.organization_id == default_organization.id


@pytest.mark.skipif(USING_SQLITE, reason="Skipped because using SQLite")
def test_create_tool_duplicate_name(server: SyncServer, print_tool, default_user, default_organization):
    data = print_tool.model_dump(exclude=["id"])
    tool = PydanticTool(**data)

    with pytest.raises(UniqueConstraintViolationError):
        server.tool_manager.create_tool(tool, actor=default_user)


def test_get_tool_by_id(server: SyncServer, print_tool, default_user):
    # Fetch the tool by ID using the manager method
    fetched_tool = server.tool_manager.get_tool_by_id(print_tool.id, actor=default_user)

    # Assertions to check if the fetched tool matches the created tool
    assert fetched_tool.id == print_tool.id
    assert fetched_tool.name == print_tool.name
    assert fetched_tool.description == print_tool.description
    assert fetched_tool.tags == print_tool.tags
    assert fetched_tool.source_code == print_tool.source_code
    assert fetched_tool.source_type == print_tool.source_type


def test_get_tool_with_actor(server: SyncServer, print_tool, default_user):
    # Fetch the print_tool by name and organization ID
    fetched_tool = server.tool_manager.get_tool_by_name(print_tool.name, actor=default_user)

    # Assertions to check if the fetched tool matches the created tool
    assert fetched_tool.id == print_tool.id
    assert fetched_tool.name == print_tool.name
    assert fetched_tool.created_by_id == default_user.id
    assert fetched_tool.description == print_tool.description
    assert fetched_tool.tags == print_tool.tags
    assert fetched_tool.source_code == print_tool.source_code
    assert fetched_tool.source_type == print_tool.source_type


def test_list_tools(server: SyncServer, print_tool, default_user):
    # List tools (should include the one created by the fixture)
    tools = server.tool_manager.list_tools(actor=default_user)

    # Assertions to check that the created tool is listed
    assert len(tools) == 1
    assert any(t.id == print_tool.id for t in tools)


def test_update_tool_by_id(server: SyncServer, print_tool, default_user):
    updated_description = "updated_description"

    # Create a ToolUpdate object to modify the print_tool's description
    tool_update = ToolUpdate(description=updated_description)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(print_tool.id, tool_update, actor=default_user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(print_tool.id, actor=default_user)

    # Assertions to check if the update was successful
    assert updated_tool.description == updated_description


def test_update_tool_source_code_refreshes_schema_and_name(server: SyncServer, print_tool, default_user):
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
    og_json_schema = print_tool.json_schema

    source_code = parse_source_code(counter_tool)

    # Create a ToolUpdate object to modify the tool's source_code
    tool_update = ToolUpdate(source_code=source_code)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(print_tool.id, tool_update, actor=default_user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(print_tool.id, actor=default_user)

    # Assertions to check if the update was successful, and json_schema is updated as well
    assert updated_tool.source_code == source_code
    assert updated_tool.json_schema != og_json_schema

    new_schema = derive_openai_json_schema(source_code=updated_tool.source_code)
    assert updated_tool.json_schema == new_schema


def test_update_tool_source_code_refreshes_schema_only(server: SyncServer, print_tool, default_user):
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
    og_json_schema = print_tool.json_schema

    source_code = parse_source_code(counter_tool)
    name = "counter_tool"

    # Create a ToolUpdate object to modify the tool's source_code
    tool_update = ToolUpdate(name=name, source_code=source_code)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(print_tool.id, tool_update, actor=default_user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(print_tool.id, actor=default_user)

    # Assertions to check if the update was successful, and json_schema is updated as well
    assert updated_tool.source_code == source_code
    assert updated_tool.json_schema != og_json_schema

    new_schema = derive_openai_json_schema(source_code=updated_tool.source_code, name=updated_tool.name)
    assert updated_tool.json_schema == new_schema
    assert updated_tool.name == name


def test_update_tool_multi_user(server: SyncServer, print_tool, default_user, other_user):
    updated_description = "updated_description"

    # Create a ToolUpdate object to modify the print_tool's description
    tool_update = ToolUpdate(description=updated_description)

    # Update the print_tool using the manager method, but WITH THE OTHER USER'S ID!
    server.tool_manager.update_tool_by_id(print_tool.id, tool_update, actor=other_user)

    # Check that the created_by and last_updated_by fields are correct
    # Fetch the updated print_tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(print_tool.id, actor=default_user)

    assert updated_tool.last_updated_by_id == other_user.id
    assert updated_tool.created_by_id == default_user.id


def test_delete_tool_by_id(server: SyncServer, print_tool, default_user):
    # Delete the print_tool using the manager method
    server.tool_manager.delete_tool_by_id(print_tool.id, actor=default_user)

    tools = server.tool_manager.list_tools(actor=default_user)
    assert len(tools) == 0


# ======================================================================================================================
# Message Manager Tests
# ======================================================================================================================


def test_message_create(server: SyncServer, hello_world_message_fixture, default_user):
    """Test creating a message using hello_world_message_fixture fixture"""
    assert hello_world_message_fixture.id is not None
    assert hello_world_message_fixture.text == "Hello, world!"
    assert hello_world_message_fixture.role == "user"

    # Verify we can retrieve it
    retrieved = server.message_manager.get_message_by_id(
        hello_world_message_fixture.id,
        actor=default_user,
    )
    assert retrieved is not None
    assert retrieved.id == hello_world_message_fixture.id
    assert retrieved.text == hello_world_message_fixture.text
    assert retrieved.role == hello_world_message_fixture.role


def test_message_get_by_id(server: SyncServer, hello_world_message_fixture, default_user):
    """Test retrieving a message by ID"""
    retrieved = server.message_manager.get_message_by_id(hello_world_message_fixture.id, actor=default_user)
    assert retrieved is not None
    assert retrieved.id == hello_world_message_fixture.id
    assert retrieved.text == hello_world_message_fixture.text


def test_message_update(server: SyncServer, hello_world_message_fixture, default_user, other_user):
    """Test updating a message"""
    new_text = "Updated text"
    updated = server.message_manager.update_message_by_id(hello_world_message_fixture.id, MessageUpdate(text=new_text), actor=other_user)
    assert updated is not None
    assert updated.text == new_text
    retrieved = server.message_manager.get_message_by_id(hello_world_message_fixture.id, actor=default_user)
    assert retrieved.text == new_text

    # Assert that orm metadata fields are populated
    assert retrieved.created_by_id == default_user.id
    assert retrieved.last_updated_by_id == other_user.id


def test_message_delete(server: SyncServer, hello_world_message_fixture, default_user):
    """Test deleting a message"""
    server.message_manager.delete_message_by_id(hello_world_message_fixture.id, actor=default_user)
    retrieved = server.message_manager.get_message_by_id(hello_world_message_fixture.id, actor=default_user)
    assert retrieved is None


def test_message_size(server: SyncServer, hello_world_message_fixture, default_user):
    """Test counting messages with filters"""
    base_message = hello_world_message_fixture

    # Create additional test messages
    messages = [
        PydanticMessage(
            organization_id=default_user.organization_id, agent_id=base_message.agent_id, role=base_message.role, text=f"Test message {i}"
        )
        for i in range(4)
    ]
    server.message_manager.create_many_messages(messages, actor=default_user)

    # Test total count
    total = server.message_manager.size(actor=default_user, role=MessageRole.user)
    assert total == 6  # login message + base message + 4 test messages
    # TODO: change login message to be a system not user message

    # Test count with agent filter
    agent_count = server.message_manager.size(actor=default_user, agent_id=base_message.agent_id, role=MessageRole.user)
    assert agent_count == 6

    # Test count with role filter
    role_count = server.message_manager.size(actor=default_user, role=base_message.role)
    assert role_count == 6

    # Test count with non-existent filter
    empty_count = server.message_manager.size(actor=default_user, agent_id="non-existent", role=MessageRole.user)
    assert empty_count == 0


def create_test_messages(server: SyncServer, base_message: PydanticMessage, default_user) -> list[PydanticMessage]:
    """Helper function to create test messages for all tests"""
    messages = [
        PydanticMessage(
            organization_id=default_user.organization_id, agent_id=base_message.agent_id, role=base_message.role, text=f"Test message {i}"
        )
        for i in range(4)
    ]
    server.message_manager.create_many_messages(messages, actor=default_user)
    return messages


def test_message_listing_basic(server: SyncServer, hello_world_message_fixture, default_user, sarah_agent):
    """Test basic message listing with limit"""
    create_test_messages(server, hello_world_message_fixture, default_user)

    results = server.message_manager.list_user_messages_for_agent(agent_id=sarah_agent.id, limit=3, actor=default_user)
    assert len(results) == 3


def test_message_listing_cursor(server: SyncServer, hello_world_message_fixture, default_user, sarah_agent):
    """Test cursor-based pagination functionality"""
    create_test_messages(server, hello_world_message_fixture, default_user)

    # Make sure there are 5 messages
    assert server.message_manager.size(actor=default_user, role=MessageRole.user) == 6

    # Get first page
    first_page = server.message_manager.list_user_messages_for_agent(agent_id=sarah_agent.id, actor=default_user, limit=3)
    assert len(first_page) == 3

    last_id_on_first_page = first_page[-1].id

    # Get second page
    second_page = server.message_manager.list_user_messages_for_agent(
        agent_id=sarah_agent.id, actor=default_user, cursor=last_id_on_first_page, limit=3
    )
    assert len(second_page) == 3  # Should have 2 remaining messages
    assert all(r1.id != r2.id for r1 in first_page for r2 in second_page)


def test_message_listing_filtering(server: SyncServer, hello_world_message_fixture, default_user, sarah_agent):
    """Test filtering messages by agent ID"""
    create_test_messages(server, hello_world_message_fixture, default_user)

    agent_results = server.message_manager.list_user_messages_for_agent(agent_id=sarah_agent.id, actor=default_user, limit=10)
    assert len(agent_results) == 6  # login message + base message + 4 test messages
    assert all(msg.agent_id == hello_world_message_fixture.agent_id for msg in agent_results)


def test_message_listing_text_search(server: SyncServer, hello_world_message_fixture, default_user, sarah_agent):
    """Test searching messages by text content"""
    create_test_messages(server, hello_world_message_fixture, default_user)

    search_results = server.message_manager.list_user_messages_for_agent(
        agent_id=sarah_agent.id, actor=default_user, query_text="Test message", limit=10
    )
    assert len(search_results) == 4
    assert all("Test message" in msg.text for msg in search_results)

    # Test no results
    search_results = server.message_manager.list_user_messages_for_agent(
        agent_id=sarah_agent.id, actor=default_user, query_text="Letta", limit=10
    )
    assert len(search_results) == 0


def test_message_listing_date_range_filtering(server: SyncServer, hello_world_message_fixture, default_user, sarah_agent):
    """Test filtering messages by date range"""
    create_test_messages(server, hello_world_message_fixture, default_user)
    now = datetime.utcnow()

    date_results = server.message_manager.list_user_messages_for_agent(
        agent_id=sarah_agent.id, actor=default_user, start_date=now - timedelta(minutes=1), end_date=now + timedelta(minutes=1), limit=10
    )
    assert len(date_results) > 0


# ======================================================================================================================
# Block Manager Tests
# ======================================================================================================================


def test_create_block(server: SyncServer, default_user):
    block_manager = BlockManager()
    block_create = PydanticBlock(
        label="human",
        is_template=True,
        value="Sample content",
        template_name="sample_template",
        description="A test block",
        limit=1000,
        metadata_={"example": "data"},
    )

    block = block_manager.create_or_update_block(block_create, actor=default_user)

    # Assertions to ensure the created block matches the expected values
    assert block.label == block_create.label
    assert block.is_template == block_create.is_template
    assert block.value == block_create.value
    assert block.template_name == block_create.template_name
    assert block.description == block_create.description
    assert block.limit == block_create.limit
    assert block.metadata_ == block_create.metadata_
    assert block.organization_id == default_user.organization_id


def test_get_blocks(server, default_user):
    block_manager = BlockManager()

    # Create blocks to retrieve later
    block_manager.create_or_update_block(PydanticBlock(label="human", value="Block 1"), actor=default_user)
    block_manager.create_or_update_block(PydanticBlock(label="persona", value="Block 2"), actor=default_user)

    # Retrieve blocks by different filters
    all_blocks = block_manager.get_blocks(actor=default_user)
    assert len(all_blocks) == 2

    human_blocks = block_manager.get_blocks(actor=default_user, label="human")
    assert len(human_blocks) == 1
    assert human_blocks[0].label == "human"

    persona_blocks = block_manager.get_blocks(actor=default_user, label="persona")
    assert len(persona_blocks) == 1
    assert persona_blocks[0].label == "persona"


def test_update_block(server: SyncServer, default_user):
    block_manager = BlockManager()
    block = block_manager.create_or_update_block(PydanticBlock(label="persona", value="Original Content"), actor=default_user)

    # Update block's content
    update_data = BlockUpdate(value="Updated Content", description="Updated description")
    block_manager.update_block(block_id=block.id, block_update=update_data, actor=default_user)

    # Retrieve the updated block
    updated_block = block_manager.get_blocks(actor=default_user, id=block.id)[0]

    # Assertions to verify the update
    assert updated_block.value == "Updated Content"
    assert updated_block.description == "Updated description"


def test_update_block_limit(server: SyncServer, default_user):

    block_manager = BlockManager()
    block = block_manager.create_or_update_block(PydanticBlock(label="persona", value="Original Content"), actor=default_user)

    limit = len("Updated Content") * 2000
    update_data = BlockUpdate(value="Updated Content" * 2000, description="Updated description", limit=limit)

    # Check that a large block fails
    try:
        block_manager.update_block(block_id=block.id, block_update=update_data, actor=default_user)
        assert False
    except Exception:
        pass

    block_manager.update_block(block_id=block.id, block_update=update_data, actor=default_user)
    # Retrieve the updated block
    updated_block = block_manager.get_blocks(actor=default_user, id=block.id)[0]
    # Assertions to verify the update
    assert updated_block.value == "Updated Content" * 2000
    assert updated_block.description == "Updated description"


def test_delete_block(server: SyncServer, default_user):
    block_manager = BlockManager()

    # Create and delete a block
    block = block_manager.create_or_update_block(PydanticBlock(label="human", value="Sample content"), actor=default_user)
    block_manager.delete_block(block_id=block.id, actor=default_user)

    # Verify that the block was deleted
    blocks = block_manager.get_blocks(actor=default_user)
    assert len(blocks) == 0


# ======================================================================================================================
# SourceManager Tests - Sources
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


def test_create_sources_with_same_name_does_not_error(server: SyncServer, default_user):
    """Test creating a new source."""
    name = "Test Source"
    source_pydantic = PydanticSource(
        name=name,
        description="This is a test source.",
        metadata_={"type": "medical"},
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
    )
    source = server.source_manager.create_source(source=source_pydantic, actor=default_user)
    source_pydantic = PydanticSource(
        name=name,
        description="This is a different test source.",
        metadata_={"type": "legal"},
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
    )
    same_source = server.source_manager.create_source(source=source_pydantic, actor=default_user)

    assert source.name == same_source.name
    assert source.id != same_source.id


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
    if USING_SQLITE:
        time.sleep(CREATE_DELAY_SQLITE)
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
# Source Manager Tests - Files
# ======================================================================================================================

def test_get_file_by_id(server: SyncServer, default_user, default_source):
    """Test retrieving a file by ID."""
    file_metadata = PydanticFileMetadata(
        file_name="Retrieve File",
        file_path="/path/to/retrieve_file.txt",
        file_type="text/plain",
        file_size=2048,
        source_id=default_source.id,
    )
    created_file = server.source_manager.create_file(file_metadata=file_metadata, actor=default_user)

    # Retrieve the file by ID
    retrieved_file = server.source_manager.get_file_by_id(file_id=created_file.id, actor=default_user)

    # Assertions to verify the retrieved file matches the created one
    assert retrieved_file.id == created_file.id
    assert retrieved_file.file_name == created_file.file_name
    assert retrieved_file.file_path == created_file.file_path
    assert retrieved_file.file_type == created_file.file_type


def test_list_files(server: SyncServer, default_user, default_source):
    """Test listing files with pagination."""
    # Create multiple files
    server.source_manager.create_file(
        PydanticFileMetadata(file_name="File 1", file_path="/path/to/file1.txt", file_type="text/plain", source_id=default_source.id),
        actor=default_user,
    )
    if USING_SQLITE:
        time.sleep(CREATE_DELAY_SQLITE)
    server.source_manager.create_file(
        PydanticFileMetadata(file_name="File 2", file_path="/path/to/file2.txt", file_type="text/plain", source_id=default_source.id),
        actor=default_user,
    )

    # List files without pagination
    files = server.source_manager.list_files(source_id=default_source.id, actor=default_user)
    assert len(files) == 2

    # List files with pagination
    paginated_files = server.source_manager.list_files(source_id=default_source.id, actor=default_user, limit=1)
    assert len(paginated_files) == 1

    # Ensure cursor-based pagination works
    next_page = server.source_manager.list_files(source_id=default_source.id, actor=default_user, cursor=paginated_files[-1].id, limit=1)
    assert len(next_page) == 1
    assert next_page[0].file_name != paginated_files[0].file_name


def test_delete_file(server: SyncServer, default_user, default_source):
    """Test deleting a file."""
    file_metadata = PydanticFileMetadata(
        file_name="Delete File", file_path="/path/to/delete_file.txt", file_type="text/plain", source_id=default_source.id
    )
    created_file = server.source_manager.create_file(file_metadata=file_metadata, actor=default_user)

    # Delete the file
    deleted_file = server.source_manager.delete_file(file_id=created_file.id, actor=default_user)

    # Assertions to verify deletion
    assert deleted_file.id == created_file.id

    # Verify that the file no longer appears in list_files
    files = server.source_manager.list_files(source_id=default_source.id, actor=default_user)
    assert len(files) == 0


# ======================================================================================================================
# SandboxConfigManager Tests - Sandbox Configs
# ======================================================================================================================

def test_create_or_update_sandbox_config(server: SyncServer, default_user):
    sandbox_config_create = SandboxConfigCreate(
        config=E2BSandboxConfig(),
    )
    created_config = server.sandbox_config_manager.create_or_update_sandbox_config(sandbox_config_create, actor=default_user)

    # Assertions
    assert created_config.type == SandboxType.E2B
    assert created_config.get_e2b_config() == sandbox_config_create.config
    assert created_config.organization_id == default_user.organization_id


def test_default_e2b_settings_sandbox_config(server: SyncServer, default_user):
    created_config = server.sandbox_config_manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.E2B, actor=default_user)
    e2b_config = created_config.get_e2b_config()

    # Assertions
    assert e2b_config.timeout == 5 * 60
    assert e2b_config.template == tool_settings.e2b_sandbox_template_id


def test_update_existing_sandbox_config(server: SyncServer, sandbox_config_fixture, default_user):
    update_data = SandboxConfigUpdate(config=E2BSandboxConfig(template="template_2", timeout=120))
    updated_config = server.sandbox_config_manager.update_sandbox_config(sandbox_config_fixture.id, update_data, actor=default_user)

    # Assertions
    assert updated_config.config["template"] == "template_2"
    assert updated_config.config["timeout"] == 120


def test_delete_sandbox_config(server: SyncServer, sandbox_config_fixture, default_user):
    deleted_config = server.sandbox_config_manager.delete_sandbox_config(sandbox_config_fixture.id, actor=default_user)

    # Assertions to verify deletion
    assert deleted_config.id == sandbox_config_fixture.id

    # Verify it no longer exists
    config_list = server.sandbox_config_manager.list_sandbox_configs(actor=default_user)
    assert sandbox_config_fixture.id not in [config.id for config in config_list]


def test_get_sandbox_config_by_type(server: SyncServer, sandbox_config_fixture, default_user):
    retrieved_config = server.sandbox_config_manager.get_sandbox_config_by_type(sandbox_config_fixture.type, actor=default_user)

    # Assertions to verify correct retrieval
    assert retrieved_config.id == sandbox_config_fixture.id
    assert retrieved_config.type == sandbox_config_fixture.type


def test_list_sandbox_configs(server: SyncServer, default_user):
    # Creating multiple sandbox configs
    config_a = SandboxConfigCreate(
        config=E2BSandboxConfig(),
    )
    config_b = SandboxConfigCreate(
        config=LocalSandboxConfig(sandbox_dir=""),
    )
    server.sandbox_config_manager.create_or_update_sandbox_config(config_a, actor=default_user)
    if USING_SQLITE:
        time.sleep(CREATE_DELAY_SQLITE)
    server.sandbox_config_manager.create_or_update_sandbox_config(config_b, actor=default_user)

    # List configs without pagination
    configs = server.sandbox_config_manager.list_sandbox_configs(actor=default_user)
    assert len(configs) >= 2

    # List configs with pagination
    paginated_configs = server.sandbox_config_manager.list_sandbox_configs(actor=default_user, limit=1)
    assert len(paginated_configs) == 1

    next_page = server.sandbox_config_manager.list_sandbox_configs(actor=default_user, cursor=paginated_configs[-1].id, limit=1)
    assert len(next_page) == 1
    assert next_page[0].id != paginated_configs[0].id


# ======================================================================================================================
# SandboxConfigManager Tests - Environment Variables
# ======================================================================================================================

def test_create_sandbox_env_var(server: SyncServer, sandbox_config_fixture, default_user):
    env_var_create = SandboxEnvironmentVariableCreate(key="TEST_VAR", value="test_value", description="A test environment variable.")
    created_env_var = server.sandbox_config_manager.create_sandbox_env_var(
        env_var_create, sandbox_config_id=sandbox_config_fixture.id, actor=default_user
    )

    # Assertions
    assert created_env_var.key == env_var_create.key
    assert created_env_var.value == env_var_create.value
    assert created_env_var.organization_id == default_user.organization_id


def test_update_sandbox_env_var(server: SyncServer, sandbox_env_var_fixture, default_user):
    update_data = SandboxEnvironmentVariableUpdate(value="updated_value")
    updated_env_var = server.sandbox_config_manager.update_sandbox_env_var(sandbox_env_var_fixture.id, update_data, actor=default_user)

    # Assertions
    assert updated_env_var.value == "updated_value"
    assert updated_env_var.id == sandbox_env_var_fixture.id


def test_delete_sandbox_env_var(server: SyncServer, sandbox_config_fixture, sandbox_env_var_fixture, default_user):
    deleted_env_var = server.sandbox_config_manager.delete_sandbox_env_var(sandbox_env_var_fixture.id, actor=default_user)

    # Assertions to verify deletion
    assert deleted_env_var.id == sandbox_env_var_fixture.id

    # Verify it no longer exists
    env_vars = server.sandbox_config_manager.list_sandbox_env_vars(sandbox_config_id=sandbox_config_fixture.id, actor=default_user)
    assert sandbox_env_var_fixture.id not in [env_var.id for env_var in env_vars]


def test_list_sandbox_env_vars(server: SyncServer, sandbox_config_fixture, default_user):
    # Creating multiple environment variables
    env_var_create_a = SandboxEnvironmentVariableCreate(key="VAR1", value="value1")
    env_var_create_b = SandboxEnvironmentVariableCreate(key="VAR2", value="value2")
    server.sandbox_config_manager.create_sandbox_env_var(env_var_create_a, sandbox_config_id=sandbox_config_fixture.id, actor=default_user)
    if USING_SQLITE:
        time.sleep(CREATE_DELAY_SQLITE)
    server.sandbox_config_manager.create_sandbox_env_var(env_var_create_b, sandbox_config_id=sandbox_config_fixture.id, actor=default_user)

    # List env vars without pagination
    env_vars = server.sandbox_config_manager.list_sandbox_env_vars(sandbox_config_id=sandbox_config_fixture.id, actor=default_user)
    assert len(env_vars) >= 2

    # List env vars with pagination
    paginated_env_vars = server.sandbox_config_manager.list_sandbox_env_vars(
        sandbox_config_id=sandbox_config_fixture.id, actor=default_user, limit=1
    )
    assert len(paginated_env_vars) == 1

    next_page = server.sandbox_config_manager.list_sandbox_env_vars(
        sandbox_config_id=sandbox_config_fixture.id, actor=default_user, cursor=paginated_env_vars[-1].id, limit=1
    )
    assert len(next_page) == 1
    assert next_page[0].id != paginated_env_vars[0].id


def test_get_sandbox_env_var_by_key(server: SyncServer, sandbox_env_var_fixture, default_user):
    retrieved_env_var = server.sandbox_config_manager.get_sandbox_env_var_by_key_and_sandbox_config_id(
        sandbox_env_var_fixture.key, sandbox_env_var_fixture.sandbox_config_id, actor=default_user
    )

    # Assertions to verify correct retrieval
    assert retrieved_env_var.id == sandbox_env_var_fixture.id
    assert retrieved_env_var.key == sandbox_env_var_fixture.key


# ======================================================================================================================
# JobManager Tests
# ======================================================================================================================

def test_create_job(server: SyncServer, default_user):
    """Test creating a job."""
    job_data = PydanticJob(
        status=JobStatus.created,
        metadata_={"type": "test"},
    )

    created_job = server.job_manager.create_job(job_data, actor=default_user)

    # Assertions to ensure the created job matches the expected values
    assert created_job.user_id == default_user.id
    assert created_job.status == JobStatus.created
    assert created_job.metadata_ == {"type": "test"}


def test_get_job_by_id(server: SyncServer, default_user):
    """Test fetching a job by ID."""
    # Create a job
    job_data = PydanticJob(
        status=JobStatus.created,
        metadata_={"type": "test"},
    )
    created_job = server.job_manager.create_job(job_data, actor=default_user)

    # Fetch the job by ID
    fetched_job = server.job_manager.get_job_by_id(created_job.id, actor=default_user)

    # Assertions to ensure the fetched job matches the created job
    assert fetched_job.id == created_job.id
    assert fetched_job.status == JobStatus.created
    assert fetched_job.metadata_ == {"type": "test"}


def test_list_jobs(server: SyncServer, default_user):
    """Test listing jobs."""
    # Create multiple jobs
    for i in range(3):
        job_data = PydanticJob(
            status=JobStatus.created,
            metadata_={"type": f"test-{i}"},
        )
        server.job_manager.create_job(job_data, actor=default_user)

    # List jobs
    jobs = server.job_manager.list_jobs(actor=default_user)

    # Assertions to check that the created jobs are listed
    assert len(jobs) == 3
    assert all(job.user_id == default_user.id for job in jobs)
    assert all(job.metadata_["type"].startswith("test") for job in jobs)


def test_update_job_by_id(server: SyncServer, default_user):
    """Test updating a job by its ID."""
    # Create a job
    job_data = PydanticJob(
        status=JobStatus.created,
        metadata_={"type": "test"},
    )
    created_job = server.job_manager.create_job(job_data, actor=default_user)

    # Update the job
    update_data = JobUpdate(status=JobStatus.completed, metadata_={"type": "updated"})
    updated_job = server.job_manager.update_job_by_id(created_job.id, update_data, actor=default_user)

    # Assertions to ensure the job was updated
    assert updated_job.status == JobStatus.completed
    assert updated_job.metadata_ == {"type": "updated"}
    assert updated_job.completed_at is not None


def test_delete_job_by_id(server: SyncServer, default_user):
    """Test deleting a job by its ID."""
    # Create a job
    job_data = PydanticJob(
        status=JobStatus.created,
        metadata_={"type": "test"},
    )
    created_job = server.job_manager.create_job(job_data, actor=default_user)

    # Delete the job
    server.job_manager.delete_job_by_id(created_job.id, actor=default_user)

    # List jobs to ensure the job was deleted
    jobs = server.job_manager.list_jobs(actor=default_user)
    assert len(jobs) == 0


def test_update_job_auto_complete(server: SyncServer, default_user):
    """Test that updating a job's status to 'completed' automatically sets completed_at."""
    # Create a job
    job_data = PydanticJob(
        status=JobStatus.created,
        metadata_={"type": "test"},
    )
    created_job = server.job_manager.create_job(job_data, actor=default_user)

    # Update the job's status to 'completed'
    update_data = JobUpdate(status=JobStatus.completed)
    updated_job = server.job_manager.update_job_by_id(created_job.id, update_data, actor=default_user)

    # Assertions to check that completed_at was set
    assert updated_job.status == JobStatus.completed
    assert updated_job.completed_at is not None


def test_get_job_not_found(server: SyncServer, default_user):
    """Test fetching a non-existent job."""
    non_existent_job_id = "nonexistent-id"
    with pytest.raises(NoResultFound):
        server.job_manager.get_job_by_id(non_existent_job_id, actor=default_user)


def test_delete_job_not_found(server: SyncServer, default_user):
    """Test deleting a non-existent job."""
    non_existent_job_id = "nonexistent-id"
    with pytest.raises(NoResultFound):
        server.job_manager.delete_job_by_id(non_existent_job_id, actor=default_user)


def test_list_jobs_pagination(server: SyncServer, default_user):
    """Test listing jobs with pagination."""
    # Create multiple jobs
    for i in range(10):
        job_data = PydanticJob(
            status=JobStatus.created,
            metadata_={"type": f"test-{i}"},
        )
        server.job_manager.create_job(job_data, actor=default_user)

    # List jobs with a limit
    jobs = server.job_manager.list_jobs(actor=default_user, limit=5)

    # Assertions to check pagination
    assert len(jobs) == 5
    assert all(job.user_id == default_user.id for job in jobs)


def test_list_jobs_by_status(server: SyncServer, default_user):
    """Test listing jobs filtered by status."""
    # Create multiple jobs with different statuses
    job_data_created = PydanticJob(
        status=JobStatus.created,
        metadata_={"type": "test-created"},
    )
    job_data_in_progress = PydanticJob(
        status=JobStatus.running,
        metadata_={"type": "test-running"},
    )
    job_data_completed = PydanticJob(
        status=JobStatus.completed,
        metadata_={"type": "test-completed"},
    )

    server.job_manager.create_job(job_data_created, actor=default_user)
    server.job_manager.create_job(job_data_in_progress, actor=default_user)
    server.job_manager.create_job(job_data_completed, actor=default_user)

    # List jobs filtered by status
    created_jobs = server.job_manager.list_jobs(actor=default_user, statuses=[JobStatus.created])
    in_progress_jobs = server.job_manager.list_jobs(actor=default_user, statuses=[JobStatus.running])
    completed_jobs = server.job_manager.list_jobs(actor=default_user, statuses=[JobStatus.completed])

    # Assertions
    assert len(created_jobs) == 1
    assert created_jobs[0].metadata_["type"] == job_data_created.metadata_["type"]

    assert len(in_progress_jobs) == 1
    assert in_progress_jobs[0].metadata_["type"] == job_data_in_progress.metadata_["type"]

    assert len(completed_jobs) == 1
    assert completed_jobs[0].metadata_["type"] == job_data_completed.metadata_["type"]
