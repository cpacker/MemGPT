import secrets
import string
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import delete

from letta.functions.functions import parse_source_code
from letta.functions.schema_generator import generate_schema
from letta.orm import SandboxConfig, SandboxEnvironmentVariable
from letta.schemas.sandbox_config import LocalSandboxConfig
from letta.schemas.sandbox_config import SandboxConfig as PydanticSandboxConfig
from letta.schemas.sandbox_config import (
    SandboxEnvironmentVariable as PydanticSandboxEnvironmentVariable,
)
from letta.schemas.sandbox_config import SandboxType
from letta.schemas.tool import Tool
from letta.services.organization_manager import OrganizationManager
from letta.services.sandbox_config_manager import SandboxConfigManager
from letta.services.tool_execution_sandbox import ToolExecutionSandbox
from letta.services.tool_manager import ToolManager
from letta.services.user_manager import UserManager
from letta.settings import tool_settings

# Constants
VENV_NAME = "test"


@pytest.fixture(autouse=True)
def clear_tables():
    """Fixture to clear the organization table before each test."""
    from letta.server.server import db_context

    with db_context() as session:
        session.execute(delete(SandboxConfig))
        session.execute(delete(SandboxEnvironmentVariable))
        session.commit()  # Commit the deletion


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
def default_organization():
    """Fixture to create and return the default organization."""
    org = OrganizationManager().create_default_organization()
    yield org


@pytest.fixture
def default_user(default_organization):
    """Fixture to create and return the default user within the default organization."""
    user = UserManager().create_default_user(org_id=default_organization.id)
    yield user


@pytest.fixture
def add_integers_tool(default_user):
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
    tool = ToolManager().create_or_update_tool(tool, default_user)
    yield tool


@pytest.fixture
def cowsay_tool(default_user):
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
    tool = ToolManager().create_or_update_tool(tool, default_user)
    yield tool


@pytest.fixture
def print_env_tool(default_user):
    def print_env() -> str:
        """
        Simple function that returns the secret word env variable.

        Returns:
            str: The cowsay ASCII art.
        """
        import os

        return os.getenv("secret_word")

    tool = create_tool_from_func(print_env)
    tool = ToolManager().create_or_update_tool(tool, default_user)
    yield tool


@pytest.fixture
def list_tool(default_user):
    def create_list():
        """Simple function that returns a list"""

        return [1] * 5

    tool = create_tool_from_func(create_list)
    tool = ToolManager().create_or_update_tool(tool, default_user)
    yield tool


def create_tool_from_func(func: callable):
    return Tool(
        name=func.__name__,
        description="",
        source_type="python",
        tags=[],
        source_code=parse_source_code(func),
        json_schema=generate_schema(func, None),
    )


# Tests
def test_local_sandbox_default(mock_e2b_api_key_none, add_integers_tool, default_user):
    args = {"x": 10, "y": 5}

    # Mock and assert correct pathway was invoked
    with patch.object(ToolExecutionSandbox, "run_local_dir_sandbox") as mock_run_local_dir_sandbox:
        sandbox = ToolExecutionSandbox(add_integers_tool.name, args, user_id=default_user.id)
        sandbox.run()
        mock_run_local_dir_sandbox.assert_called_once()

    # Run again to get actual response
    sandbox = ToolExecutionSandbox(add_integers_tool.name, args, user_id=default_user.id)
    response = sandbox.run()
    assert response == args["x"] + args["y"]


def test_local_sandbox_with_list_rv(mock_e2b_api_key_none, list_tool, default_user):
    sandbox = ToolExecutionSandbox(list_tool.name, {}, user_id=default_user.id)
    response = sandbox.run()
    assert len(response) == 5


def test_local_sandbox_custom(mock_e2b_api_key_none, cowsay_tool, default_user):
    manager = SandboxConfigManager()

    # Make a custom local sandbox config
    sandbox_dir = str(Path(__file__).parent / "test_tool_sandbox")
    config = PydanticSandboxConfig(
        type=SandboxType.LOCAL, config=LocalSandboxConfig(venv_name="test", sandbox_dir=sandbox_dir).model_dump()
    )
    manager.create_or_update_sandbox_config(config, default_user)

    # Make a environment variable with a long random string
    key = "secret_word"
    long_random_string = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
    manager.create_sandbox_env_var(PydanticSandboxEnvironmentVariable(key=key, value=long_random_string), default_user)

    # Create tool and args
    args = {}

    # Run the custom sandbox
    sandbox = ToolExecutionSandbox(cowsay_tool.name, args, user_id=default_user.id)
    response = sandbox.run()

    assert long_random_string in response


def test_e2b_sandbox_default(check_e2b_key_is_set, add_integers_tool, default_user):
    args = {"x": 10, "y": 5}

    # Mock and assert correct pathway was invoked
    with patch.object(ToolExecutionSandbox, "run_e2b_sandbox") as mock_run_local_dir_sandbox:
        sandbox = ToolExecutionSandbox(add_integers_tool.name, args, user_id=default_user.id)
        sandbox.run()
        mock_run_local_dir_sandbox.assert_called_once()

    # Run again to get actual response
    sandbox = ToolExecutionSandbox(add_integers_tool.name, args, user_id=default_user.id)
    response = sandbox.run()
    assert int(response) == args["x"] + args["y"]


def test_e2b_sandbox_with_env(check_e2b_key_is_set, print_env_tool, default_user):
    manager = SandboxConfigManager()

    key = "secret_word"
    long_random_string = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
    manager.create_sandbox_env_var(PydanticSandboxEnvironmentVariable(key=key, value=long_random_string), default_user)

    # Create tool and args
    args = {}

    # Run the custom sandbox
    sandbox = ToolExecutionSandbox(print_env_tool.name, args, user_id=default_user.id)
    response = sandbox.run()

    assert long_random_string in response


def test_e2b_sandbox_with_list_rv(check_e2b_key_is_set, list_tool, default_user):
    sandbox = ToolExecutionSandbox(list_tool.name, {}, user_id=default_user.id)
    response = sandbox.run()
    assert len(response) == 5
