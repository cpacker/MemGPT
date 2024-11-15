import os
import threading
import time
from typing import List

import pytest
from dotenv import load_dotenv
from sqlalchemy import delete

from letta import create_client
from letta.orm import SandboxConfig, SandboxEnvironmentVariable
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.sandbox_config import LocalSandboxConfig, SandboxType

# Constants
SERVER_PORT = 8283
DEFAULT_VENV_NAME = "test-venv"
UPDATED_VENV_NAME = "updated-venv"
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


@pytest.fixture(autouse=True)
def clear_tables():
    """Clear the sandbox tables before each test."""
    from letta.server.server import db_context

    with db_context() as session:
        session.execute(delete(SandboxEnvironmentVariable))
        session.execute(delete(SandboxConfig))
        session.commit()


def test_sandbox_config_and_env_var_basic(client):
    """
    Test sandbox config and environment variable functions for both LocalClient and RESTClient.
    """

    # 1. Create a sandbox config
    local_config = LocalSandboxConfig(venv_name=DEFAULT_VENV_NAME, sandbox_dir=SANDBOX_DIR)
    sandbox_config = client.create_sandbox_config(config=local_config)

    # Assert the created sandbox config
    assert sandbox_config.id is not None
    assert sandbox_config.type == SandboxType.LOCAL
    assert sandbox_config.config["venv_name"] == DEFAULT_VENV_NAME

    # 2. Update the sandbox config
    updated_config = LocalSandboxConfig(venv_name=UPDATED_VENV_NAME, sandbox_dir=UPDATED_SANDBOX_DIR)
    sandbox_config = client.update_sandbox_config(sandbox_config_id=sandbox_config.id, config=updated_config)
    assert sandbox_config.config["venv_name"] == UPDATED_VENV_NAME
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
