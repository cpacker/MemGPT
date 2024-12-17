import uuid
from pathlib import Path

import pytest
from sqlalchemy import delete

from letta import create_client
from letta.functions.function_sets.base import core_memory_replace
from letta.orm import SandboxConfig, SandboxEnvironmentVariable
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.schemas.organization import Organization
from letta.schemas.sandbox_config import LocalSandboxConfig, SandboxConfigCreate
from letta.schemas.user import User
from letta.services.organization_manager import OrganizationManager
from letta.services.sandbox_config_manager import SandboxConfigManager
from letta.services.tool_execution_sandbox import ToolExecutionSandbox
from letta.services.tool_manager import ToolManager
from letta.services.user_manager import UserManager
from letta.settings import tool_settings
from tests.helpers.utils import create_tool_from_func

# Constants
namespace = uuid.NAMESPACE_DNS
org_name = str(uuid.uuid5(namespace, "test-tool-execution-sandbox-org"))
user_name = str(uuid.uuid5(namespace, "test-tool-execution-sandbox-user"))


# Fixtures
@pytest.fixture(autouse=True)
def clear_tables():
    """Fixture to clear the organization table before each test."""
    from letta.server.server import db_context

    with db_context() as session:
        session.execute(delete(SandboxEnvironmentVariable))
        session.execute(delete(SandboxConfig))
        session.commit()  # Commit the deletion

    # Kill all sandboxes
    from e2b_code_interpreter import Sandbox

    for sandbox in Sandbox.list():
        Sandbox.connect(sandbox.sandbox_id).kill()


@pytest.fixture
def check_e2b_key_is_set():
    original_api_key = tool_settings.e2b_api_key
    assert original_api_key is not None, "Missing e2b key! Cannot execute these tests."
    yield


@pytest.fixture
def check_composio_key_set():
    original_api_key = tool_settings.composio_api_key
    assert original_api_key is not None, "Missing composio key! Cannot execute this test."
    yield


@pytest.fixture
def test_organization():
    """Fixture to create and return the default organization."""
    org = OrganizationManager().create_organization(Organization(name=org_name))
    yield org


@pytest.fixture
def test_user(test_organization):
    """Fixture to create and return the default user within the default organization."""
    user = UserManager().create_user(User(name=user_name, organization_id=test_organization.id))
    yield user


@pytest.fixture
def core_memory_replace_tool(test_user):
    tool = create_tool_from_func(core_memory_replace)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


@pytest.fixture
def agent_state():
    client = create_client()
    agent_state = client.create_agent(
        memory=ChatMemory(persona="This is the persona", human="My name is Chad"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        llm_config=LLMConfig.default_config(model_name="gpt-4"),
    )
    yield agent_state


@pytest.fixture
def custom_test_sandbox_config(test_user):
    """
    Fixture to create a consistent local sandbox configuration for tests.

    Args:
        test_user: The test user to be used for creating the sandbox configuration.

    Returns:
        A tuple containing the SandboxConfigManager and the created sandbox configuration.
    """
    # Create the SandboxConfigManager
    manager = SandboxConfigManager(tool_settings)

    # Set the sandbox to be within the external codebase path and use a venv
    external_codebase_path = str(Path(__file__).parent / "test_tool_sandbox" / "restaurant_management_system")
    local_sandbox_config = LocalSandboxConfig(sandbox_dir=external_codebase_path, use_venv=True)

    # Create the sandbox configuration
    config_create = SandboxConfigCreate(config=local_sandbox_config.model_dump())

    # Create or update the sandbox configuration
    manager.create_or_update_sandbox_config(sandbox_config_create=config_create, actor=test_user)

    return manager, local_sandbox_config


@pytest.mark.local_sandbox
def test_local_sandbox_core_memory_replace(mock_e2b_api_key_none, core_memory_replace_tool, test_user, agent_state):
    new_name = "Matt"
    args = {"label": "human", "old_content": "Chad", "new_content": new_name}
    sandbox = ToolExecutionSandbox(core_memory_replace_tool.name, args, user_id=test_user.id)

    # run the sandbox
    result = sandbox.run(agent_state=agent_state)
    assert new_name in result.agent_state.memory.get_block("human").value
    assert result.func_return is None


@pytest.mark.local_sandbox
def test_local_sandbox_core_memory_replace_errors(mock_e2b_api_key_none, core_memory_replace_tool, test_user, agent_state):
    nonexistent_name = "Alexander Wang"
    args = {"label": "human", "old_content": nonexistent_name, "new_content": "Matt"}
    sandbox = ToolExecutionSandbox(core_memory_replace_tool.name, args, user_id=test_user.id)

    # run the sandbox
    result = sandbox.run(agent_state=agent_state)
    assert len(result.stderr) != 0, "stderr not empty"
    assert (
        f"ValueError: Old content '{nonexistent_name}' not found in memory block 'human'" in result.stderr[0]
    ), "stderr contains expected error"


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_escape_strings_in_args(check_e2b_key_is_set, core_memory_replace_tool, test_user, agent_state):
    new_name = "Matt"
    args = {"label": "human", "old_content": "Chad", "new_content": new_name + "\n"}
    sandbox = ToolExecutionSandbox(core_memory_replace_tool.name, args, user_id=test_user.id)

    # run the sandbox
    result = sandbox.run(agent_state=agent_state)
    assert new_name in result.agent_state.memory.get_block("human").value
    assert result.func_return is None


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_core_memory_replace_errors(check_e2b_key_is_set, core_memory_replace_tool, test_user, agent_state):
    nonexistent_name = "Alexander Wang"
    args = {"label": "human", "old_content": nonexistent_name, "new_content": "Matt"}
    sandbox = ToolExecutionSandbox(core_memory_replace_tool.name, args, user_id=test_user.id)

    # run the sandbox
    result = sandbox.run(agent_state=agent_state)
    assert len(result.stderr) != 0, "stderr not empty"
    assert (
        f"ValueError: Old content '{nonexistent_name}' not found in memory block 'human'" in result.stderr[0]
    ), "stderr contains expected error"
