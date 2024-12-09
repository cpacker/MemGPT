import secrets
import string
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import delete

from letta import create_client
from letta.functions.function_sets.base import core_memory_replace
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
def get_warning_tool(test_user):
    def warn_hello_world() -> str:
        """
        Simple function that warns hello world.

        Returns:
            str: hello world
        """
        import warnings

        msg = "Hello World"
        warnings.warn(msg)
        return msg

    tool = create_tool_from_func(warn_hello_world)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


@pytest.fixture
def always_err_tool(test_user):
    def error() -> str:
        """
        Simple function that errors

        Returns:
            str: not important
        """
        # Raise a unusual error so we know it's from this function
        raise ZeroDivisionError("This is an intentionally weird division!")

    tool = create_tool_from_func(error)
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
    tool_create = ToolCreate.from_composio(action_name="GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER")
    tool = tool_manager.create_or_update_tool(pydantic_tool=Tool(**tool_create.model_dump()), actor=test_user)
    yield tool


@pytest.fixture
def clear_core_memory_tool(test_user):
    def clear_memory(agent_state: AgentState):
        """Clear the core memory"""
        agent_state.memory.get_block("human").value = ""
        agent_state.memory.get_block("persona").value = ""

    tool = create_tool_from_func(clear_memory)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


@pytest.fixture
def core_memory_replace_tool(test_user):
    tool = create_tool_from_func(core_memory_replace)
    tool = ToolManager().create_or_update_tool(tool, test_user)
    yield tool


@pytest.fixture
def external_codebase_tool(test_user):
    from tests.test_tool_sandbox.restaurant_management_system.adjust_menu_prices import (
        adjust_menu_prices,
    )

    tool = create_tool_from_func(adjust_menu_prices)
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
def test_local_sandbox_stateful_tool(mock_e2b_api_key_none, clear_core_memory_tool, test_user, agent_state):
    args = {}
    # Run again to get actual response
    sandbox = ToolExecutionSandbox(clear_core_memory_tool.name, args, user_id=test_user.id)
    result = sandbox.run(agent_state=agent_state)
    assert result.agent_state.memory.get_block("human").value == ""
    assert result.agent_state.memory.get_block("persona").value == ""
    assert result.func_return is None


@pytest.mark.local_sandbox
def test_local_sandbox_core_memory_replace(mock_e2b_api_key_none, core_memory_replace_tool, test_user, agent_state):
    new_name = "Matt"
    args = {"label": "human", "old_content": "Chad", "new_content": new_name}
    sandbox = ToolExecutionSandbox(core_memory_replace_tool.name, args, user_id=test_user.id)

    # run the sandbox
    result = sandbox.run(agent_state=agent_state)
    assert new_name in result.agent_state.memory.get_block("human").value
    assert result.func_return is None


@pytest.mark.e2b_sandbox
def test_local_sandbox_core_memory_replace_errors(mock_e2b_api_key_none, core_memory_replace_tool, test_user, agent_state):
    nonexistent_name = "Alexander Wang"
    args = {"label": "human", "old_content": nonexistent_name, "new_content": "Matt"}
    sandbox = ToolExecutionSandbox(core_memory_replace_tool.name, args, user_id=test_user.id)

    # run the sandbox
    with pytest.raises(ValueError, match=f"Old content '{nonexistent_name}' not found in memory block 'human'"):
        sandbox.run(agent_state=agent_state)


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


@pytest.mark.local_sandbox
def test_local_sandbox_external_codebase(mock_e2b_api_key_none, custom_test_sandbox_config, external_codebase_tool, test_user):
    # Set the args
    args = {"percentage": 10}

    # Run again to get actual response
    sandbox = ToolExecutionSandbox(external_codebase_tool.name, args, user_id=test_user.id)
    result = sandbox.run()

    # Assert that the function return is correct
    assert result.func_return == "Price Adjustments:\nBurger: $8.99 -> $9.89\nFries: $2.99 -> $3.29\nSoda: $1.99 -> $2.19"
    assert "Hello World" in result.stdout[0]


@pytest.mark.local_sandbox
def test_local_sandbox_with_venv_and_warnings_does_not_error(
    mock_e2b_api_key_none, custom_test_sandbox_config, get_warning_tool, test_user
):
    sandbox = ToolExecutionSandbox(get_warning_tool.name, {}, user_id=test_user.id)
    result = sandbox.run()
    assert result.func_return == "Hello World"


@pytest.mark.e2b_sandbox
def test_local_sandbox_with_venv_errors(mock_e2b_api_key_none, custom_test_sandbox_config, always_err_tool, test_user):
    sandbox = ToolExecutionSandbox(always_err_tool.name, {}, user_id=test_user.id)

    # run the sandbox
    with pytest.raises(ZeroDivisionError, match="This is an intentionally weird division!"):
        sandbox.run()


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
def test_e2b_sandbox_stateful_tool(check_e2b_key_is_set, clear_core_memory_tool, test_user, agent_state):
    sandbox = ToolExecutionSandbox(clear_core_memory_tool.name, {}, user_id=test_user.id)

    # run the sandbox
    result = sandbox.run(agent_state=agent_state)
    assert result.agent_state.memory.get_block("human").value == ""
    assert result.agent_state.memory.get_block("persona").value == ""
    assert result.func_return is None


@pytest.mark.e2b_sandbox
def test_e2b_sandbox_core_memory_replace(check_e2b_key_is_set, core_memory_replace_tool, test_user, agent_state):
    new_name = "Matt"
    args = {"label": "human", "old_content": "Chad", "new_content": new_name}
    sandbox = ToolExecutionSandbox(core_memory_replace_tool.name, args, user_id=test_user.id)

    # run the sandbox
    result = sandbox.run(agent_state=agent_state)
    assert new_name in result.agent_state.memory.get_block("human").value
    assert result.func_return is None


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
    with pytest.raises(ValueError, match=f"Old content '{nonexistent_name}' not found in memory block 'human'"):
        sandbox.run(agent_state=agent_state)


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
