import os
import threading
import time
import uuid
from typing import List, Union

import pytest
from dotenv import load_dotenv
from sqlalchemy import delete

from letta import LocalClient, RESTClient, create_client
from letta.orm import SandboxConfig, SandboxEnvironmentVariable
from letta.schemas.agent import AgentState
from letta.schemas.block import BlockCreate
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.sandbox_config import LocalSandboxConfig, SandboxType
from letta.settings import tool_settings
from letta.utils import create_random_username

# Constants
SERVER_PORT = 8283
SANDBOX_DIR = "/tmp/sandbox"
UPDATED_SANDBOX_DIR = "/tmp/updated_sandbox"
ENV_VAR_KEY = "TEST_VAR"
UPDATED_ENV_VAR_KEY = "UPDATED_VAR"
ENV_VAR_VALUE = "test_value"
UPDATED_ENV_VAR_VALUE = "updated_value"
ENV_VAR_DESCRIPTION = "A test environment variable"


def run_server():
    load_dotenv()

    from letta.server.rest_api.app import start_server

    print("Starting server...")
    start_server(debug=True)


@pytest.fixture(
    params=[{"server": True}, {"server": False}],  # whether to use REST API server
    scope="module",
)
def client(request):
    if request.param["server"]:
        # Get URL from environment or start server
        server_url = os.getenv("LETTA_SERVER_URL", f"http://localhost:{SERVER_PORT}")
        if not os.getenv("LETTA_SERVER_URL"):
            print("Starting server thread")
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(5)
        print("Running client tests with server:", server_url)
        client = create_client(base_url=server_url, token=None)
    else:
        client = create_client()

    client.set_default_llm_config(LLMConfig.default_config("gpt-4"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))
    yield client


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client: Union[LocalClient, RESTClient]):
    agent_state = client.create_agent(name=f"test_client_{str(uuid.uuid4())}")
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


@pytest.fixture(autouse=True)
def clear_tables():
    """Clear the sandbox tables before each test."""
    from letta.server.server import db_context

    with db_context() as session:
        session.execute(delete(SandboxEnvironmentVariable))
        session.execute(delete(SandboxConfig))
        session.commit()


@pytest.fixture
def mock_e2b_api_key_none():
    # Store the original value of e2b_api_key
    original_api_key = tool_settings.e2b_api_key

    # Set e2b_api_key to None
    tool_settings.e2b_api_key = None

    # Yield control to the test
    yield

    # Restore the original value of e2b_api_key
    tool_settings.e2b_api_key = original_api_key


def test_sandbox_config_and_env_var_basic(client: Union[LocalClient, RESTClient]):
    """
    Test sandbox config and environment variable functions for both LocalClient and RESTClient.
    """

    # 1. Create a sandbox config
    local_config = LocalSandboxConfig(sandbox_dir=SANDBOX_DIR)
    sandbox_config = client.create_sandbox_config(config=local_config)

    # Assert the created sandbox config
    assert sandbox_config.id is not None
    assert sandbox_config.type == SandboxType.LOCAL

    # 2. Update the sandbox config
    updated_config = LocalSandboxConfig(sandbox_dir=UPDATED_SANDBOX_DIR)
    sandbox_config = client.update_sandbox_config(sandbox_config_id=sandbox_config.id, config=updated_config)
    assert sandbox_config.config["sandbox_dir"] == UPDATED_SANDBOX_DIR

    # 3. List all sandbox configs
    sandbox_configs = client.list_sandbox_configs(limit=10)
    assert isinstance(sandbox_configs, List)
    assert len(sandbox_configs) == 1
    assert sandbox_configs[0].id == sandbox_config.id

    # 4. Create an environment variable
    env_var = client.create_sandbox_env_var(
        sandbox_config_id=sandbox_config.id, key=ENV_VAR_KEY, value=ENV_VAR_VALUE, description=ENV_VAR_DESCRIPTION
    )
    assert env_var.id is not None
    assert env_var.key == ENV_VAR_KEY
    assert env_var.value == ENV_VAR_VALUE
    assert env_var.description == ENV_VAR_DESCRIPTION

    # 5. Update the environment variable
    updated_env_var = client.update_sandbox_env_var(env_var_id=env_var.id, key=UPDATED_ENV_VAR_KEY, value=UPDATED_ENV_VAR_VALUE)
    assert updated_env_var.key == UPDATED_ENV_VAR_KEY
    assert updated_env_var.value == UPDATED_ENV_VAR_VALUE

    # 6. List environment variables
    env_vars = client.list_sandbox_env_vars(sandbox_config_id=sandbox_config.id)
    assert isinstance(env_vars, List)
    assert len(env_vars) == 1
    assert env_vars[0].key == UPDATED_ENV_VAR_KEY

    # 7. Delete the environment variable
    client.delete_sandbox_env_var(env_var_id=env_var.id)

    # 8. Delete the sandbox config
    client.delete_sandbox_config(sandbox_config_id=sandbox_config.id)


def test_add_and_manage_tags_for_agent(client: Union[LocalClient, RESTClient], agent: AgentState):
    """
    Comprehensive happy path test for adding, retrieving, and managing tags on an agent.
    """
    tags_to_add = ["test_tag_1", "test_tag_2", "test_tag_3"]

    # Step 0: create an agent with tags
    tagged_agent = client.create_agent(tags=tags_to_add)
    assert set(tagged_agent.tags) == set(tags_to_add), f"Expected tags {tags_to_add}, but got {tagged_agent.tags}"

    # Step 1: Add multiple tags to the agent
    client.update_agent(agent_id=agent.id, tags=tags_to_add)

    # Step 2: Retrieve tags for the agent and verify they match the added tags
    retrieved_tags = client.get_agent(agent_id=agent.id).tags
    assert set(retrieved_tags) == set(tags_to_add), f"Expected tags {tags_to_add}, but got {retrieved_tags}"

    # Step 3: Retrieve agents by each tag to ensure the agent is associated correctly
    for tag in tags_to_add:
        agents_with_tag = client.list_agents(tags=[tag])
        assert agent.id in [a.id for a in agents_with_tag], f"Expected agent {agent.id} to be associated with tag '{tag}'"

    # Step 4: Delete a specific tag from the agent and verify its removal
    tag_to_delete = tags_to_add.pop()
    client.update_agent(agent_id=agent.id, tags=tags_to_add)

    # Verify the tag is removed from the agent's tags
    remaining_tags = client.get_agent(agent_id=agent.id).tags
    assert tag_to_delete not in remaining_tags, f"Tag '{tag_to_delete}' was not removed as expected"
    assert set(remaining_tags) == set(tags_to_add), f"Expected remaining tags to be {tags_to_add[1:]}, but got {remaining_tags}"

    # Step 5: Delete all remaining tags from the agent
    client.update_agent(agent_id=agent.id, tags=[])

    # Verify all tags are removed
    final_tags = client.get_agent(agent_id=agent.id).tags
    assert len(final_tags) == 0, f"Expected no tags, but found {final_tags}"


def test_update_agent_memory_label(client: Union[LocalClient, RESTClient], agent: AgentState):
    """Test that we can update the label of a block in an agent's memory"""

    agent = client.create_agent(name=create_random_username())

    try:
        current_labels = agent.memory.list_block_labels()
        example_label = current_labels[0]
        example_new_label = "example_new_label"
        assert example_new_label not in current_labels

        client.update_agent_memory_label(agent_id=agent.id, current_label=example_label, new_label=example_new_label)

        updated_agent = client.get_agent(agent_id=agent.id)
        assert example_new_label in updated_agent.memory.list_block_labels()

    finally:
        client.delete_agent(agent.id)


def test_add_remove_agent_memory_block(client: Union[LocalClient, RESTClient], agent: AgentState):
    """Test that we can add and remove a block from an agent's memory"""

    agent = client.create_agent(name=create_random_username())

    try:
        current_labels = agent.memory.list_block_labels()
        example_new_label = "example_new_label"
        example_new_value = "example value"
        assert example_new_label not in current_labels

        # Link a new memory block
        client.add_agent_memory_block(
            agent_id=agent.id,
            create_block=BlockCreate(
                label=example_new_label,
                value=example_new_value,
                limit=1000,
            ),
        )

        updated_agent = client.get_agent(agent_id=agent.id)
        assert example_new_label in updated_agent.memory.list_block_labels()

        # Now unlink the block
        client.remove_agent_memory_block(agent_id=agent.id, block_label=example_new_label)

        updated_agent = client.get_agent(agent_id=agent.id)
        assert example_new_label not in updated_agent.memory.list_block_labels()

    finally:
        client.delete_agent(agent.id)


# def test_core_memory_token_limits(client: Union[LocalClient, RESTClient], agent: AgentState):
#     """Test that the token limit is enforced for the core memory blocks"""

#     # Create an agent
#     new_agent = client.create_agent(
#         name="test-core-memory-token-limits",
#         tools=BASE_TOOLS,
#         memory=ChatMemory(human="The humans name is Joe.", persona="My name is Sam.", limit=2000),
#     )

#     try:
#         # Then intentionally set the limit to be extremely low
#         client.update_agent(
#             agent_id=new_agent.id,
#             memory=ChatMemory(human="The humans name is Joe.", persona="My name is Sam.", limit=100),
#         )

#         # TODO we should probably not allow updating the core memory limit if

#         # TODO in which case we should modify this test to actually to a proper token counter check

#     finally:
#         client.delete_agent(new_agent.id)


def test_update_agent_memory_limit(client: Union[LocalClient, RESTClient], agent: AgentState):
    """Test that we can update the limit of a block in an agent's memory"""

    agent = client.create_agent(name=create_random_username())

    try:
        current_labels = agent.memory.list_block_labels()
        example_label = current_labels[0]
        example_new_limit = 1
        current_block = agent.memory.get_block(label=example_label)
        current_block_length = len(current_block.value)

        assert example_new_limit != agent.memory.get_block(label=example_label).limit
        assert example_new_limit < current_block_length

        # We expect this to throw a value error
        with pytest.raises(ValueError):
            client.update_agent_memory_limit(agent_id=agent.id, block_label=example_label, limit=example_new_limit)

        # Now try the same thing with a higher limit
        example_new_limit = current_block_length + 10000
        assert example_new_limit > current_block_length
        client.update_agent_memory_limit(agent_id=agent.id, block_label=example_label, limit=example_new_limit)

        updated_agent = client.get_agent(agent_id=agent.id)
        assert example_new_limit == updated_agent.memory.get_block(label=example_label).limit

    finally:
        client.delete_agent(agent.id)
