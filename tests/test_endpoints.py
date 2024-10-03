import json
import os
import uuid

from letta import create_client
from letta.agent import Agent
from letta.embeddings import embedding_model
from letta.llm_api.llm_api_tools import create
from letta.prompts import gpt_system
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.message import Message
from tests.helpers.endpoints_helper import (
    assert_contains_correct_inner_monologue,
    assert_contains_valid_function_call,
    setup_llm_endpoint,
)
from tests.helpers.utils import cleanup

messages = [Message(role="system", text=gpt_system.get_system_text("memgpt_chat")), Message(role="user", text="How are you?")]

# defaults (letta hosted)
embedding_config_path = "configs/embedding_model_configs/letta-hosted.json"
llm_config_path = "configs/llm_model_configs/letta-hosted.json"

# directories
embedding_config_dir = "configs/embedding_model_configs"
llm_config_dir = "configs/llm_model_configs"

# Generate uuid for agent name for this example
namespace = uuid.NAMESPACE_DNS
agent_uuid = str(uuid.uuid5(namespace, "test-endpoints-agent"))


def check_first_response_is_valid_for_llm_endpoint(filename: str, inner_thoughts_in_kwargs: bool = False):
    llm_config, embedding_config = setup_llm_endpoint(filename, embedding_config_path)

    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    agent_state = client.create_agent(name=agent_uuid, llm_config=llm_config, embedding_config=embedding_config)
    tools = [client.get_tool(client.get_tool_id(name=name)) for name in agent_state.tools]
    agent = Agent(
        interface=None,
        tools=tools,
        agent_state=agent_state,
    )

    response = create(
        llm_config=llm_config,
        user_id=uuid.UUID(int=1),  # dummy user_id
        # messages=agent_state.messages,
        messages=agent._messages,
        functions=agent.functions,
        functions_python=agent.functions_python,
    )

    # Basic check
    assert response is not None

    # Select first choice
    choice = response.choices[0]

    # Ensure that the first message returns a "send_message"
    validator_func = lambda function_call: function_call.name == "send_message" or function_call.name == "archival_memory_search"
    assert_contains_valid_function_call(choice.message, validator_func)

    # Assert that the choice has an inner monologue
    assert_contains_correct_inner_monologue(choice, inner_thoughts_in_kwargs)


def run_embedding_endpoint(filename):
    # load JSON file
    config_data = json.load(open(filename, "r"))
    print(config_data)
    embedding_config = EmbeddingConfig(**config_data)
    model = embedding_model(embedding_config)
    query_text = "hello"
    query_vec = model.get_text_embedding(query_text)
    print("vector dim", len(query_vec))
    assert query_vec is not None


def test_llm_endpoint_openai():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    check_first_response_is_valid_for_llm_endpoint(filename)


def test_embedding_endpoint_openai():
    filename = os.path.join(embedding_config_dir, "text-embedding-ada-002.json")
    run_embedding_endpoint(filename)


def test_llm_endpoint_letta_hosted():
    filename = os.path.join(llm_config_dir, "letta-hosted.json")
    check_first_response_is_valid_for_llm_endpoint(filename)


def test_embedding_endpoint_letta_hosted():
    filename = os.path.join(embedding_config_dir, "letta-hosted.json")
    run_embedding_endpoint(filename)


def test_embedding_endpoint_local():
    filename = os.path.join(embedding_config_dir, "local.json")
    run_embedding_endpoint(filename)


def test_llm_endpoint_ollama():
    filename = os.path.join(llm_config_dir, "ollama.json")
    check_first_response_is_valid_for_llm_endpoint(filename)


def test_embedding_endpoint_ollama():
    filename = os.path.join(embedding_config_dir, "ollama.json")
    run_embedding_endpoint(filename)


def test_llm_endpoint_anthropic():
    filename = os.path.join(llm_config_dir, "anthropic.json")
    check_first_response_is_valid_for_llm_endpoint(filename)
