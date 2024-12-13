import os
import time
from datetime import datetime, timedelta

from numpy import source
import pytest
from sqlalchemy import all_, delete

from letta.embeddings import embedding_model
import letta.utils as utils
from letta.functions.functions import derive_openai_json_schema, parse_source_code
from letta.metadata import AgentModel
from letta.orm.sqlite_functions import verify_embedding_dimension, convert_array
from letta.orm import (
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
    Tool,
    ToolsAgents,
    User,
)
from letta.orm.agents_tags import AgentsTags
from letta.orm.errors import (
    ForeignKeyConstraintViolationError,
    NoResultFound,
    UniqueConstraintViolationError,
)
from letta.schemas.agent import CreateAgent
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.block import BlockUpdate, CreateBlock
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import JobStatus, MessageRole
from letta.schemas.file import FileMetadata as PydanticFileMetadata
from letta.schemas.job import Job as PydanticJob
from letta.schemas.job import JobUpdate
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.message import MessageUpdate
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
from letta.services.block_manager import BlockManager
from letta.services.organization_manager import OrganizationManager
from letta.services.passage_manager import PassageManager
from letta.services.tool_manager import ToolManager
from letta.settings import tool_settings

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
        session.execute(delete(AgentsTags))
        session.execute(delete(SandboxEnvironmentVariable))
        session.execute(delete(SandboxConfig))
        session.execute(delete(Block))
        session.execute(delete(FileMetadata))
        session.execute(delete(Source))
        session.execute(delete(Tool))  # Clear all records from the Tool table
        session.execute(delete(AgentModel))
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
def default_file(server: SyncServer, default_source, default_user, default_organization):
    file = server.source_manager.create_file(
        PydanticFileMetadata(
            file_name="test_file", organization_id=default_organization.id, source_id=default_source.id),
        actor=default_user,
    )
    yield file


@pytest.fixture
def sarah_agent(server: SyncServer, default_user, default_organization):
    """Fixture to create and return a sample agent within the default organization."""
    agent_state = server.create_agent(
        request=CreateAgent(
            name="sarah_agent",
            # memory_blocks=[CreateBlock(label="human", value="Charles"), CreateBlock(label="persona", value="I am a helpful assistant")],
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


def test_passage_cascade_deletion(server: SyncServer, agent_passage_fixture, source_passage_fixture, default_user, default_source):
    """Test that passages are deleted when their parent (agent or source) is deleted."""
    # Verify passages exist
    agent_passage = server.passage_manager.get_passage_by_id(agent_passage_fixture.id, default_user)
    source_passage = server.passage_manager.get_passage_by_id(source_passage_fixture.id, default_user)
    assert agent_passage is not None
    assert source_passage is not None
    
    # Delete agent and verify its passages are deleted

    # TODO: add in with Agents ORM
    # server.agent_manager.delete_agent(sarah_agent.id, default_user)
    # with pytest.raises(ValueError):
    #     server.passage_manager.get_passage_by_id(agent_passage_fixture.id, default_user)
    
    # Delete source and verify its passages are deleted
    server.source_manager.delete_source(default_source.id, default_user)
    with pytest.raises(NoResultFound):
        server.passage_manager.get_passage_by_id(source_passage_fixture.id, default_user)


def test_passage_listing(server: SyncServer, agent_passage_fixture, source_passage_fixture, default_user, sarah_agent, default_source):
    """Test listing passages by type (agent or source)."""
    # List only agent passages
    agent_passages = server.passage_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
    )
    assert len(agent_passages) == 1
    assert all(p.agent_id == sarah_agent.id for p in agent_passages)
    
    # List only source passages
    source_passages = server.passage_manager.list_passages(
        actor=default_user,
        source_id=default_source.id,
    )
    assert len(source_passages) == 1
    assert all(p.source_id == default_source.id for p in source_passages)

    # List all passages in the user's organization
    all_passages = server.passage_manager.list_passages(
        actor=default_user,
    )
    assert len(all_passages) == 2


def test_passage_update(server: SyncServer, agent_passage_fixture, source_passage_fixture, default_user):
    """Test updating a passage"""
    new_text = "Updated text"
    agent_passage_fixture.text = new_text
    updated = server.passage_manager.update_passage_by_id(agent_passage_fixture.id, agent_passage_fixture, actor=default_user)
    assert updated is not None
    assert updated.text == new_text
    retrieved = server.passage_manager.get_passage_by_id(agent_passage_fixture.id, actor=default_user)
    assert retrieved.text == new_text

    source_passage_fixture.text = new_text
    updated = server.passage_manager.update_passage_by_id(source_passage_fixture.id, source_passage_fixture, actor=default_user)
    assert updated is not None
    assert updated.text == new_text
    retrieved = server.passage_manager.get_passage_by_id(source_passage_fixture.id, actor=default_user)
    assert retrieved.text == new_text


def test_passage_delete(server: SyncServer, agent_passage_fixture, source_passage_fixture, default_user):
    """Test deleting a passage"""
    server.passage_manager.delete_passage_by_id(agent_passage_fixture.id, actor=default_user)
    with pytest.raises(NoResultFound):
        server.passage_manager.get_passage_by_id(agent_passage_fixture.id, actor=default_user)

    server.passage_manager.delete_passage_by_id(source_passage_fixture.id, actor=default_user)
    with pytest.raises(NoResultFound):
        server.passage_manager.get_passage_by_id(source_passage_fixture.id, actor=default_user)

    with pytest.raises(NoResultFound):
        server.passage_manager.delete_passage_by_id(agent_passage_fixture.id, actor=default_user)
    with pytest.raises(NoResultFound):
        server.passage_manager.delete_passage_by_id(source_passage_fixture.id, actor=default_user)


def test_passage_size(server: SyncServer, agent_passage_fixture, source_passage_fixture, create_test_passages, default_user):
    """Test counting passages with filters"""

    # Test total count
    total = server.passage_manager.size(actor=default_user)
    assert total == 12 # 1 base passage +5 agent passages + 5 source passages

    # Test count with agent filter
    agent_count = server.passage_manager.size(actor=default_user, agent_id=agent_passage_fixture.agent_id)
    assert agent_count == 6

    # Test count with source filter
    source_count = server.passage_manager.size(actor=default_user, source_id=source_passage_fixture.source_id)
    assert source_count == 6

    # Test count with non-existent filter
    empty_count = server.passage_manager.size(actor=default_user, agent_id="non-existent")
    assert empty_count == 0


def test_passage_listing_limit(server: SyncServer, create_test_passages, default_user):
    """Test basic passage listing with limit"""
    results = server.passage_manager.list_passages(actor=default_user, limit=3)
    assert len(results) == 3


def test_passage_listing_filtering(server: SyncServer, create_test_passages, default_user, sarah_agent, default_source):
    """Test filtering passages by agent ID"""
    agent_results = server.passage_manager.list_passages(agent_id=sarah_agent.id, actor=default_user)
    assert len(agent_results) == 5  # 5 agent passages
    assert all(msg.agent_id == sarah_agent.id for msg in agent_results)

    source_results = server.passage_manager.list_passages(source_id=default_source.id, actor=default_user)
    assert len(source_results) == 5  # 5 source passages
    assert all(msg.source_id == default_source.id for msg in source_results)

    all_results = server.passage_manager.list_passages(actor=default_user)
    assert len(all_results) == 10

    all_results_with_filters = server.passage_manager.list_passages(agent_id=sarah_agent.id, source_id=default_source.id, actor=default_user)
    assert len(all_results_with_filters) == 0


def test_passage_listing_text_search(server: SyncServer, agent_passage_fixture, create_test_passages, default_user):
    """Test searching passages by text content"""
    search_results = server.passage_manager.list_passages(
        actor=default_user, query_text="Agent passage"
    )
    assert len(search_results) == 6
    assert all("Agent passage".lower() in msg.text.lower() for msg in search_results)
    
    # Test no results
    search_results = server.passage_manager.list_passages(
        actor=default_user, query_text="Letta"
    )
    assert len(search_results) == 0


def test_passage_listing_date_range_filtering(server: SyncServer, agent_passage_fixture, default_user, default_file, sarah_agent):
    """Test filtering passages by date range with various scenarios"""
    # Set up test data with known dates
    base_time = datetime.utcnow()
    
    # Create passages at different times
    passages = []
    time_offsets = [
        timedelta(days=-2),    # 2 days ago
        timedelta(days=-1),    # Yesterday
        timedelta(hours=-2),   # 2 hours ago
        timedelta(minutes=-30), # 30 minutes ago
        timedelta(minutes=-1),  # 1 minute ago
        timedelta(minutes=0),   # Now
    ]
    
    for i, offset in enumerate(time_offsets):
        timestamp = base_time + offset
        passage = server.passage_manager.create_passage(
            PydanticPassage(
                text=f"Test passage {i}",
                agent_id=sarah_agent.id,
                file_id=default_file.id,
                organization_id=default_user.organization_id,
                embedding=[0.1, 0.2, 0.3],
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
                created_at=timestamp
            ),
            actor=default_user
        )
        passages.append(passage)

    # Test cases
    test_cases = [
        {
            "name": "Recent passages (last hour)",
            "start_date": base_time - timedelta(hours=1),
            "end_date": base_time + timedelta(minutes=1),
            "expected_count": 1 + 3,  # Should include base + -30min, -1min, and now
        },
        {
            "name": "Yesterday's passages",
            "start_date": base_time - timedelta(days=1, hours=12),
            "end_date": base_time - timedelta(hours=12),
            "expected_count": 1,  # Should only include yesterday's passage
        },
        {
            "name": "Future time range",
            "start_date": base_time + timedelta(days=1),
            "end_date": base_time + timedelta(days=2),
            "expected_count": 0,  # Should find no passages
        },
        {
            "name": "All time",
            "start_date": base_time - timedelta(days=3),
            "end_date": base_time + timedelta(days=1),
            "expected_count": 1 + len(passages),  # Should find all passages
        },
        {
            "name": "Exact timestamp match",
            "start_date": passages[0].created_at - timedelta(microseconds=1),
            "end_date": passages[0].created_at + timedelta(microseconds=1),
            "expected_count": 1,  # Should find exactly one passage
        },
        {
            "name": "Small time window",
            "start_date": base_time - timedelta(seconds=30),
            "end_date": base_time + timedelta(seconds=30),
            "expected_count": 1 + 1,  # date + "now"
        }
    ]

    # Run test cases
    for case in test_cases:
        results = server.passage_manager.list_passages(
            agent_id=sarah_agent.id,
            actor=default_user,
            start_date=case["start_date"],
            end_date=case["end_date"],
        )
        
        # Verify count
        assert len(results) == case["expected_count"], \
            f"Test case '{case['name']}' failed: expected {case['expected_count']} passages, got {len(results)}"

    # Test edge cases
    
    # Test with start_date but no end_date
    results_start_only = server.passage_manager.list_passages(
        agent_id=sarah_agent.id,
        actor=default_user,
        start_date=base_time - timedelta(minutes=2),
        end_date=None,
        limit=10
    )
    assert len(results_start_only) >= 2, "Should find passages after start_date"

    # Test with end_date but no start_date
    results_end_only = server.passage_manager.list_passages(
        agent_id=sarah_agent.id,
        actor=default_user,
        start_date=None,
        end_date=base_time - timedelta(days=1),
        limit=10
    )
    assert len(results_end_only) >= 1, "Should find passages before end_date"

    # Test limit enforcement
    limited_results = server.passage_manager.list_passages(
        agent_id=sarah_agent.id,
        actor=default_user,
        start_date=base_time - timedelta(days=3),
        end_date=base_time + timedelta(days=1),
        limit=3
    )
    assert len(limited_results) <= 3, "Should respect the limit parameter"


def test_passage_vector_search(server: SyncServer, default_user, default_file, sarah_agent):
    """Test vector search functionality for passages."""
    passage_manager = server.passage_manager
    embed_model = embedding_model(DEFAULT_EMBEDDING_CONFIG) 
    
    # Create passages with known embeddings
    passages = []
    
    # Create passages with different embeddings
    test_passages = [
        "I like red",
        "random text",
        "blue shoes",
    ]
    
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
                source_id=default_file.id,
                embedding_config=DEFAULT_EMBEDDING_CONFIG,
                embedding=embedding
            )
        created_passage = passage_manager.create_passage(passage, default_user)
        passages.append(created_passage)
    assert passage_manager.size(actor=default_user) == len(passages)
    
    # Query vector similar to "cats" embedding
    query_key = "What's my favorite color?"
    
    # List passages with vector search
    results = passage_manager.list_passages(
        actor=default_user,
        query_text=query_key,
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
        embed_query=True,
    )
    
    # Verify results are ordered by similarity
    assert len(results) == 3
    assert results[0].text == "I like red"
    assert results[1].text == "random text" # For some reason the embedding model doesn't like "blue shoes"
    assert results[2].text == "blue shoes"

    # List passages with vector search
    results = passage_manager.list_passages(
        actor=default_user,
        agent_id=sarah_agent.id,
        query_text=query_key,
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
        embed_query=True,
    )
    
    # Verify results are ordered by similarity
    assert len(results) == 2
    assert results[0].text == "I like red"
    assert results[1].text == "blue shoes"

    # List passages with vector search
    results = passage_manager.list_passages(
        actor=default_user,
        source_id=default_file.id,
        query_text=query_key,
        embedding_config=DEFAULT_EMBEDDING_CONFIG,
        embed_query=True,
    )
    
    # Verify results are ordered by similarity
    assert len(results) == 1
    assert results[0].text == "random text"


def test_passage_listing_cursor_pagination(server: SyncServer, default_user, sarah_agent, default_source):
    """Test cursor-based pagination with both ascending and descending order"""
    # Create passages with controlled timestamps for predictable ordering
    base_time = datetime(2023, 1, 1, 12, 0)
    
    # Create interleaved source and agent passages
    passages = []
    for i in range(6):
        if i % 2 == 0:
            # Create source passage
            passage = server.passage_manager.create_passage(
                PydanticPassage(
                    text=f"Source passage {i//2}",
                    organization_id=default_user.organization_id,
                    source_id=default_source.id,
                    embedding=[0.1],
                    embedding_config=DEFAULT_EMBEDDING_CONFIG,
                    created_at=base_time + timedelta(hours=i)
                ),
                actor=default_user
            )
        else:
            # Create agent passage
            passage = server.passage_manager.create_passage(
                PydanticPassage(
                    text=f"Agent passage {i//2}",
                    organization_id=default_user.organization_id,
                    agent_id=sarah_agent.id,
                    embedding=[0.1],
                    embedding_config=DEFAULT_EMBEDDING_CONFIG,
                    created_at=base_time + timedelta(hours=i)
                ),
                actor=default_user
            )
        passages.append(passage)

    # Test ascending order pagination
    
    # Get first page starting from beginning
    first_page = server.passage_manager.list_passages(
        actor=default_user,
        limit=2,
        ascending=True
    )
    assert len(first_page) == 2
    assert first_page[0].text == "Source passage 0"
    assert first_page[1].text == "Agent passage 0"

    # Get second page using cursor from first page
    second_page = server.passage_manager.list_passages(
        actor=default_user,
        cursor=first_page[-1].id,
        limit=2,
        ascending=True
    )
    assert len(second_page) == 2
    assert second_page[0].text == "Source passage 1"
    assert second_page[1].text == "Agent passage 1"

    # Get third page using cursor from second page
    third_page = server.passage_manager.list_passages(
        actor=default_user,
        cursor=second_page[-1].id,
        limit=2,
        ascending=True
    )
    assert len(third_page) == 2
    assert third_page[0].text == "Source passage 2"
    assert third_page[1].text == "Agent passage 2"

    # Test descending order pagination
    
    # Get first page starting from end
    first_page_desc = server.passage_manager.list_passages(
        actor=default_user,
        limit=2,
        ascending=False
    )
    assert len(first_page_desc) == 2
    assert first_page_desc[0].text == "Agent passage 2"
    assert first_page_desc[1].text == "Source passage 2"

    # Get second page using cursor from first page
    second_page_desc = server.passage_manager.list_passages(
        actor=default_user,
        cursor=first_page_desc[-1].id,
        limit=2,
        ascending=False
    )
    assert len(second_page_desc) == 2
    assert second_page_desc[0].text == "Agent passage 1"
    assert second_page_desc[1].text == "Source passage 1"

    # Get third page using cursor from second page
    third_page_desc = server.passage_manager.list_passages(
        actor=default_user,
        cursor=second_page_desc[-1].id,
        limit=2,
        ascending=False
    )
    assert len(third_page_desc) == 2
    assert third_page_desc[0].text == "Agent passage 0"
    assert third_page_desc[1].text == "Source passage 0"


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
# Source Manager Tests - Sources
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
# BlocksAgentsManager Tests
# ======================================================================================================================
def test_add_block_to_agent(server, sarah_agent, default_user, default_block):
    block_association = server.blocks_agents_manager.add_block_to_agent(
        agent_id=sarah_agent.id, block_id=default_block.id, block_label=default_block.label
    )

    assert block_association.agent_id == sarah_agent.id
    assert block_association.block_id == default_block.id
    assert block_association.block_label == default_block.label


def test_change_label_on_block_reflects_in_block_agents_table(server, sarah_agent, default_user, default_block):
    # Add the block
    block_association = server.blocks_agents_manager.add_block_to_agent(
        agent_id=sarah_agent.id, block_id=default_block.id, block_label=default_block.label
    )
    assert block_association.block_label == default_block.label

    # Change the block label
    new_label = "banana"
    block = server.block_manager.update_block(block_id=default_block.id, block_update=BlockUpdate(label=new_label), actor=default_user)
    assert block.label == new_label

    # Get the association
    labels = server.blocks_agents_manager.list_block_labels_for_agent(agent_id=sarah_agent.id)
    assert new_label in labels
    assert default_block.label not in labels


@pytest.mark.skipif(USING_SQLITE, reason="Skipped because using SQLite")
def test_add_block_to_agent_nonexistent_block(server, sarah_agent, default_user):
    with pytest.raises(ForeignKeyConstraintViolationError):
        server.blocks_agents_manager.add_block_to_agent(
            agent_id=sarah_agent.id, block_id="nonexistent_block", block_label="nonexistent_label"
        )


def test_add_block_to_agent_duplicate_label(server, sarah_agent, default_user, default_block, other_block):
    server.blocks_agents_manager.add_block_to_agent(agent_id=sarah_agent.id, block_id=default_block.id, block_label=default_block.label)

    with pytest.warns(UserWarning, match=f"Block label '{default_block.label}' already exists for agent '{sarah_agent.id}'"):
        server.blocks_agents_manager.add_block_to_agent(agent_id=sarah_agent.id, block_id=other_block.id, block_label=default_block.label)


def test_remove_block_with_label_from_agent(server, sarah_agent, default_user, default_block):
    server.blocks_agents_manager.add_block_to_agent(agent_id=sarah_agent.id, block_id=default_block.id, block_label=default_block.label)

    removed_block = server.blocks_agents_manager.remove_block_with_label_from_agent(
        agent_id=sarah_agent.id, block_label=default_block.label
    )

    assert removed_block.block_label == default_block.label
    assert removed_block.block_id == default_block.id
    assert removed_block.agent_id == sarah_agent.id

    with pytest.raises(ValueError, match=f"Block label '{default_block.label}' not found for agent '{sarah_agent.id}'"):
        server.blocks_agents_manager.remove_block_with_label_from_agent(agent_id=sarah_agent.id, block_label=default_block.label)


def test_update_block_id_for_agent(server, sarah_agent, default_user, default_block, other_block):
    server.blocks_agents_manager.add_block_to_agent(agent_id=sarah_agent.id, block_id=default_block.id, block_label=default_block.label)

    updated_block = server.blocks_agents_manager.update_block_id_for_agent(
        agent_id=sarah_agent.id, block_label=default_block.label, new_block_id=other_block.id
    )

    assert updated_block.block_id == other_block.id
    assert updated_block.block_label == default_block.label
    assert updated_block.agent_id == sarah_agent.id


def test_list_block_ids_for_agent(server, sarah_agent, default_user, default_block, other_block):
    server.blocks_agents_manager.add_block_to_agent(agent_id=sarah_agent.id, block_id=default_block.id, block_label=default_block.label)
    server.blocks_agents_manager.add_block_to_agent(agent_id=sarah_agent.id, block_id=other_block.id, block_label=other_block.label)

    retrieved_block_ids = server.blocks_agents_manager.list_block_ids_for_agent(agent_id=sarah_agent.id)

    assert set(retrieved_block_ids) == {default_block.id, other_block.id}


def test_list_agent_ids_with_block(server, sarah_agent, charles_agent, default_user, default_block):
    server.blocks_agents_manager.add_block_to_agent(agent_id=sarah_agent.id, block_id=default_block.id, block_label=default_block.label)
    server.blocks_agents_manager.add_block_to_agent(agent_id=charles_agent.id, block_id=default_block.id, block_label=default_block.label)

    agent_ids = server.blocks_agents_manager.list_agent_ids_with_block(block_id=default_block.id)

    assert sarah_agent.id in agent_ids
    assert charles_agent.id in agent_ids
    assert len(agent_ids) == 2


@pytest.mark.skipif(USING_SQLITE, reason="Skipped because using SQLite")
def test_add_block_to_agent_with_deleted_block(server, sarah_agent, default_user, default_block):
    block_manager = BlockManager()
    block_manager.delete_block(block_id=default_block.id, actor=default_user)

    with pytest.raises(ForeignKeyConstraintViolationError):
        server.blocks_agents_manager.add_block_to_agent(agent_id=sarah_agent.id, block_id=default_block.id, block_label=default_block.label)


# ======================================================================================================================
# ToolsAgentsManager Tests
# ======================================================================================================================
def test_add_tool_to_agent(server, sarah_agent, default_user, print_tool):
    tool_association = server.tools_agents_manager.add_tool_to_agent(
        agent_id=sarah_agent.id, tool_id=print_tool.id, tool_name=print_tool.name
    )

    assert tool_association.agent_id == sarah_agent.id
    assert tool_association.tool_id == print_tool.id
    assert tool_association.tool_name == print_tool.name


def test_change_name_on_tool_reflects_in_tool_agents_table(server, sarah_agent, default_user, print_tool):
    # Add the tool
    tool_association = server.tools_agents_manager.add_tool_to_agent(
        agent_id=sarah_agent.id, tool_id=print_tool.id, tool_name=print_tool.name
    )
    assert tool_association.tool_name == print_tool.name

    # Change the tool name
    new_name = "banana"
    tool = server.tool_manager.update_tool_by_id(tool_id=print_tool.id, tool_update=ToolUpdate(name=new_name), actor=default_user)
    assert tool.name == new_name

    # Get the association
    names = server.tools_agents_manager.list_tool_names_for_agent(agent_id=sarah_agent.id)
    assert new_name in names
    assert print_tool.name not in names


@pytest.mark.skipif(USING_SQLITE, reason="Skipped because using SQLite")
def test_add_tool_to_agent_nonexistent_tool(server, sarah_agent, default_user):
    with pytest.raises(ForeignKeyConstraintViolationError):
        server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id="nonexistent_tool", tool_name="nonexistent_name")


def test_add_tool_to_agent_duplicate_name(server, sarah_agent, default_user, print_tool, other_tool):
    server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=print_tool.id, tool_name=print_tool.name)

    with pytest.warns(UserWarning, match=f"Tool name '{print_tool.name}' already exists for agent '{sarah_agent.id}'"):
        server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=other_tool.id, tool_name=print_tool.name)


def test_remove_tool_with_name_from_agent(server, sarah_agent, default_user, print_tool):
    server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=print_tool.id, tool_name=print_tool.name)

    removed_tool = server.tools_agents_manager.remove_tool_with_name_from_agent(agent_id=sarah_agent.id, tool_name=print_tool.name)

    assert removed_tool.tool_name == print_tool.name
    assert removed_tool.tool_id == print_tool.id
    assert removed_tool.agent_id == sarah_agent.id

    with pytest.raises(ValueError, match=f"Tool name '{print_tool.name}' not found for agent '{sarah_agent.id}'"):
        server.tools_agents_manager.remove_tool_with_name_from_agent(agent_id=sarah_agent.id, tool_name=print_tool.name)


def test_list_tool_ids_for_agent(server, sarah_agent, default_user, print_tool, other_tool):
    server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=print_tool.id, tool_name=print_tool.name)
    server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=other_tool.id, tool_name=other_tool.name)

    retrieved_tool_ids = server.tools_agents_manager.list_tool_ids_for_agent(agent_id=sarah_agent.id)

    assert set(retrieved_tool_ids) == {print_tool.id, other_tool.id}


def test_list_agent_ids_with_tool(server, sarah_agent, charles_agent, default_user, print_tool):
    server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=print_tool.id, tool_name=print_tool.name)
    server.tools_agents_manager.add_tool_to_agent(agent_id=charles_agent.id, tool_id=print_tool.id, tool_name=print_tool.name)

    agent_ids = server.tools_agents_manager.list_agent_ids_with_tool(tool_id=print_tool.id)

    assert sarah_agent.id in agent_ids
    assert charles_agent.id in agent_ids
    assert len(agent_ids) == 2


@pytest.mark.skipif(USING_SQLITE, reason="Skipped because using SQLite")
def test_add_tool_to_agent_with_deleted_tool(server, sarah_agent, default_user, print_tool):
    tool_manager = ToolManager()
    tool_manager.delete_tool_by_id(tool_id=print_tool.id, actor=default_user)

    with pytest.raises(ForeignKeyConstraintViolationError):
        server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=print_tool.id, tool_name=print_tool.name)


def test_remove_all_agent_tools(server, sarah_agent, default_user, print_tool, other_tool):
    server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=print_tool.id, tool_name=print_tool.name)
    server.tools_agents_manager.add_tool_to_agent(agent_id=sarah_agent.id, tool_id=other_tool.id, tool_name=other_tool.name)

    server.tools_agents_manager.remove_all_agent_tools(agent_id=sarah_agent.id)

    retrieved_tool_ids = server.tools_agents_manager.list_tool_ids_for_agent(agent_id=sarah_agent.id)

    assert not retrieved_tool_ids


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
