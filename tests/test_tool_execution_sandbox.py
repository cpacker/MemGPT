import secrets
import string
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from composio import Action
from sqlalchemy import delete

from letta import create_client
from letta.functions.functions import parse_source_code
from letta.functions.schema_generator import generate_schema
from letta.orm import SandboxConfig, SandboxEnvironmentVariable
from letta.schemas.agent import AgentState
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.schemas.organization import Organization
from letta.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfigCreate,
    SandboxConfigUpdate,
    SandboxEnvironmentVariableCreate,
    SandboxType,
)
from letta.schemas.tool import Tool, ToolCreate
from letta.schemas.user import User
from letta.services.organization_manager import OrganizationManager
from letta.services.sandbox_config_manager import SandboxConfigManager
from letta.services.tool_execution_sandbox import ToolExecutionSandbox
from letta.services.tool_manager import ToolManager
from letta.services.user_manager import UserManager
from letta.settings import tool_settings

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
def mock_e2b_api_key_none():
    # Store the original value of e2b_api_key
    original_api_key = tool_settings.e2b_api_key

    # Set e2b_api_key to None
    tool_settings.e2b_api_key = None

    # Yield control to the test
    yield

    # Restore the original value of e2b_api_key
    tool_settings.e2b_api_key = original_api_key


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
def add_integers_tool(test_user):
    def add(x: int, y: int) -> int:
        """
        Simple function that adds two integers.

        Parameters:
            x (int): The first integer to add.
            y (int): The second integer to add.

        Returns:
            int: The result of adding x and y.
        """
        return x + y

    tool = create_tool_from_func(add)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


@pytest.fixture
def cowsay_tool(test_user):
    # This defines a tool for a package we definitely do NOT have in letta
    # If this test passes, that means the tool was correctly executed in a separate Python environment
    def cowsay() -> str:
        """
        Simple function that uses the cowsay package to print out the secret word env variable.

        Returns:
            str: The cowsay ASCII art.
        """
        import os

        import cowsay

        cowsay.cow(os.getenv("secret_word"))

    tool = create_tool_from_func(cowsay)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


@pytest.fixture
def get_env_tool(test_user):
    def get_env() -> str:
        """
        Simple function that returns the secret word env variable.

        Returns:
            str: The secret word
        """
        import os

        secret_word = os.getenv("secret_word")
        print(secret_word)
        return secret_word

    tool = create_tool_from_func(get_env)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


@pytest.fixture
def list_tool(test_user):
    def create_list():
        """Simple function that returns a list"""

        return [1] * 5

    tool = create_tool_from_func(create_list)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


@pytest.fixture
def composio_github_star_tool(test_user):
    tool_manager = ToolManager()
    tool_create = ToolCreate.from_composio(action=Action.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER)
    tool = tool_manager.create_or_update_tool(pydantic_tool=Tool(**tool_create.model_dump()), actor=test_user)
    yield tool


@pytest.fixture
def clear_core_memory(test_user):
    def clear_memory(agent_state: AgentState):
        """Clear the core memory"""
        agent_state.memory.get_block("human").value = ""
        agent_state.memory.get_block("persona").value = ""

    tool = create_tool_from_func(clear_memory)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


# Utility functions
def create_tool_from_func(func: callable):
    return Tool(
        name=func.__name__,
        description="",
        source_type="python",
        tags=[],
        source_code=parse_source_code(func),
        json_schema=generate_schema(func, None),
    )


# Local sandbox tests
@pytest.mark.local_sandbox
def test_local_sandbox_default(mock_e2b_api_key_none, add_integers_tool, test_user):
    args = {"x": 10, "y": 5}

    # Mock and assert correct pathway was invoked
    with patch.object(ToolExecutionSandbox, "run_local_dir_sandbox") as mock_run_local_dir_sandbox:
        sandbox = ToolExecutionSandbox(add_integers_tool.name, args, user_id=test_user.id)
        sandbox.run()
        mock_run_local_dir_sandbox.assert_called_once()

    # Run again to get actual response
    sandbox = ToolExecutionSandbox(add_integers_tool.name, args, user_id=test_user.id)
    result = sandbox.run()
    assert result.func_return == args["x"] + args["y"]


@pytest.mark.local_sandbox
def test_local_sandbox_stateful_tool(mock_e2b_api_key_none, clear_core_memory, test_user):
    args = {}

    client = create_client()
    agent_state = client.create_agent(
        memory=ChatMemory(persona="This is the persona", human="This is the human"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        llm_config=LLMConfig.default_config(model_name="gpt-4"),
    )

    # Run again to get actual response
    sandbox = ToolExecutionSandbox(clear_core_memory.name, args, user_id=test_user.id)
    result = sandbox.run(agent_state=agent_state)
    assert result.agent_state.memory.get_block("human").value == ""
    assert result.agent_state.memory.get_block("persona").value == ""
    assert result.func_return is None


@pytest.mark.local_sandbox
def test_local_sandbox_with_list_rv(mock_e2b_api_key_none, list_tool, test_user):
    sandbox = ToolExecutionSandbox(list_tool.name, {}, user_id=test_user.id)
    result = sandbox.run()
    assert len(result.func_return) == 5


@pytest.mark.local_sandbox
def test_local_sandbox_env(mock_e2b_api_key_none, get_env_tool, test_user):
    manager = SandboxConfigManager(tool_settings)

    # Make a custom local sandbox config
    sandbox_dir = str(Path(__file__).parent / "test_tool_sandbox")
    config_create = SandboxConfigCreate(config=LocalSandboxConfig(sandbox_dir=sandbox_dir).model_dump())
    config = manager.create_or_update_sandbox_config(config_create, test_user)

    # Make a environment variable with a long random string
    key = "secret_word"
    long_random_string = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
    manager.create_sandbox_env_var(
        SandboxEnvironmentVariableCreate(key=key, value=long_random_string), sandbox_config_id=config.id, actor=test_user
    )

    # Create tool and args
    args = {}

    # Run the custom sandbox
    sandbox = ToolExecutionSandbox(get_env_tool.name, args, user_id=test_user.id)
    result = sandbox.run()

    assert long_random_string in result.func_return


@pytest.mark.local_sandbox
def test_local_sandbox_e2e_composio_star_github(mock_e2b_api_key_none, check_composio_key_set, composio_github_star_tool, test_user):
    # Add the composio key
    manager = SandboxConfigManager(tool_settings)
    config = manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.LOCAL, actor=test_user)

    manager.create_sandbox_env_var(
        SandboxEnvironmentVariableCreate(key="COMPOSIO_API_KEY", value=tool_settings.composio_api_key),
        sandbox_config_id=config.id,
        actor=test_user,
    )

    result = ToolExecutionSandbox(composio_github_star_tool.name, {"owner": "letta-ai", "repo": "letta"}, user_id=test_user.id).run()
    assert result.func_return["details"] == "Action executed successfully"


# E2B sandbox tests


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_default(check_e2b_key_is_set, add_integers_tool, test_user):
    args = {"x": 10, "y": 5}

    # Mock and assert correct pathway was invoked
    with patch.object(ToolExecutionSandbox, "run_e2b_sandbox") as mock_run_local_dir_sandbox:
        sandbox = ToolExecutionSandbox(add_integers_tool.name, args, user_id=test_user.id)
        sandbox.run()
        mock_run_local_dir_sandbox.assert_called_once()

    # Run again to get actual response
    sandbox = ToolExecutionSandbox(add_integers_tool.name, args, user_id=test_user.id)
    result = sandbox.run()
    assert int(result.func_return) == args["x"] + args["y"]


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_pip_installs(check_e2b_key_is_set, cowsay_tool, test_user):
    manager = SandboxConfigManager(tool_settings)
    config_create = SandboxConfigCreate(config=E2BSandboxConfig(pip_requirements=["cowsay"]).model_dump())
    config = manager.create_or_update_sandbox_config(config_create, test_user)

    # Add an environment variable
    key = "secret_word"
    long_random_string = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
    manager.create_sandbox_env_var(
        SandboxEnvironmentVariableCreate(key=key, value=long_random_string), sandbox_config_id=config.id, actor=test_user
    )

    sandbox = ToolExecutionSandbox(cowsay_tool.name, {}, user_id=test_user.id)
    result = sandbox.run()
    assert long_random_string in result.stdout[0]


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_reuses_same_sandbox(check_e2b_key_is_set, list_tool, test_user):
    sandbox = ToolExecutionSandbox(list_tool.name, {}, user_id=test_user.id)

    # Run the function once
    result = sandbox.run()
    old_config_fingerprint = result.sandbox_config_fingerprint

    # Run it again to ensure that there is still only one running sandbox
    result = sandbox.run()
    new_config_fingerprint = result.sandbox_config_fingerprint

    assert old_config_fingerprint == new_config_fingerprint


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_stateful_tool(check_e2b_key_is_set, clear_core_memory, test_user):
    sandbox = ToolExecutionSandbox(clear_core_memory.name, {}, user_id=test_user.id)

    # create an agent
    client = create_client()
    agent_state = client.create_agent(
        memory=ChatMemory(persona="This is the persona", human="This is the human"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        llm_config=LLMConfig.default_config(model_name="gpt-4"),
    )

    # run the sandbox
    result = sandbox.run(agent_state=agent_state)
    assert result.agent_state.memory.get_block("human").value == ""
    assert result.agent_state.memory.get_block("persona").value == ""
    assert result.func_return is None


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_inject_env_var_existing_sandbox(check_e2b_key_is_set, get_env_tool, test_user):
    manager = SandboxConfigManager(tool_settings)
    config_create = SandboxConfigCreate(config=E2BSandboxConfig().model_dump())
    config = manager.create_or_update_sandbox_config(config_create, test_user)

    # Run the custom sandbox once, assert nothing returns because missing env variable
    sandbox = ToolExecutionSandbox(get_env_tool.name, {}, user_id=test_user.id, force_recreate=True)
    result = sandbox.run()
    # response should be None
    assert result.func_return is None

    # Add an environment variable
    key = "secret_word"
    long_random_string = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
    manager.create_sandbox_env_var(
        SandboxEnvironmentVariableCreate(key=key, value=long_random_string), sandbox_config_id=config.id, actor=test_user
    )

    # Assert that the environment variable gets injected correctly, even when the sandbox is NOT refreshed
    sandbox = ToolExecutionSandbox(get_env_tool.name, {}, user_id=test_user.id)
    result = sandbox.run()
    assert long_random_string in result.func_return


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_config_change_force_recreates_sandbox(check_e2b_key_is_set, list_tool, test_user):
    manager = SandboxConfigManager(tool_settings)
    old_timeout = 5 * 60
    new_timeout = 10 * 60

    # Make the config
    config_create = SandboxConfigCreate(config=E2BSandboxConfig(timeout=old_timeout))
    config = manager.create_or_update_sandbox_config(config_create, test_user)

    # Run the custom sandbox once, assert a failure gets returned because missing environment variable
    sandbox = ToolExecutionSandbox(list_tool.name, {}, user_id=test_user.id)
    result = sandbox.run()
    assert len(result.func_return) == 5
    old_config_fingerprint = result.sandbox_config_fingerprint

    # Change the config
    config_update = SandboxConfigUpdate(config=E2BSandboxConfig(timeout=new_timeout))
    config = manager.update_sandbox_config(config.id, config_update, test_user)

    # Run again
    result = ToolExecutionSandbox(list_tool.name, {}, user_id=test_user.id).run()
    new_config_fingerprint = result.sandbox_config_fingerprint
    assert config.fingerprint() == new_config_fingerprint

    # Assert the fingerprints are different
    assert old_config_fingerprint != new_config_fingerprint


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_with_list_rv(check_e2b_key_is_set, list_tool, test_user):
    sandbox = ToolExecutionSandbox(list_tool.name, {}, user_id=test_user.id)
    result = sandbox.run()
    assert len(result.func_return) == 5


@pytest.mark.e2b_sandboxfunc
def test_e2b_e2e_composio_star_github(check_e2b_key_is_set, check_composio_key_set, composio_github_star_tool, test_user):
    # Add the composio key
    manager = SandboxConfigManager(tool_settings)
    config = manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.E2B, actor=test_user)

    manager.create_sandbox_env_var(
        SandboxEnvironmentVariableCreate(key="COMPOSIO_API_KEY", value=tool_settings.composio_api_key),
        sandbox_config_id=config.id,
        actor=test_user,
    )

    result = ToolExecutionSandbox(composio_github_star_tool.name, {"owner": "letta-ai", "repo": "letta"}, user_id=test_user.id).run()
    assert result.func_return["details"] == "Action executed successfully"
