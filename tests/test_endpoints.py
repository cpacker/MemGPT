import json
import os
import uuid

from memgpt import create_client
from memgpt.agent import Agent
from memgpt.config import MemGPTConfig
from memgpt.embeddings import embedding_model
from memgpt.llm_api.llm_api_tools import create
from memgpt.prompts import gpt_system
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.llm_config import LLMConfig
from memgpt.schemas.message import Message

messages = [Message(role="system", text=gpt_system.get_system_text("memgpt_chat")), Message(role="user", text="How are you?")]

# defaults (memgpt hosted)
embedding_config_path = "configs/embedding_model_configs/memgpt-hosted.json"
llm_config_path = "configs/llm_model_configs/memgpt-hosted.json"

# directories
embedding_config_dir = "configs/embedding_model_configs"
llm_config_dir = "configs/llm_model_configs"


def run_llm_endpoint(filename):
    config_data = json.load(open(filename, "r"))
    print(config_data)
    llm_config = LLMConfig(**config_data)
    embedding_config = EmbeddingConfig(**json.load(open(embedding_config_path)))

    # setup config
    config = MemGPTConfig()
    config.default_llm_config = llm_config
    config.default_embedding_config = embedding_config
    config.save()

    client = create_client()
    agent_state = client.create_agent(name="test_agent", llm_config=llm_config, embedding_config=embedding_config)
    tools = [client.get_tool(client.get_tool_id(name=name)) for name in agent_state.tools]
    agent = Agent(
        interface=None,
        tools=tools,
        agent_state=agent_state,
        # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
        first_message_verify_mono=True,
    )

    response = create(
        llm_config=llm_config,
        user_id=uuid.UUID(int=1),  # dummy user_id
        # messages=agent_state.messages,
        messages=agent._messages,
        functions=agent.functions,
        functions_python=agent.functions_python,
    )
    client.delete_agent(agent_state.id)
    assert response is not None


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
    run_llm_endpoint(filename)


def test_embedding_endpoint_openai():
    filename = os.path.join(embedding_config_dir, "text-embedding-ada-002.json")
    run_embedding_endpoint(filename)


def test_llm_endpoint_memgpt_hosted():
    filename = os.path.join(llm_config_dir, "memgpt-hosted.json")
    run_llm_endpoint(filename)


def test_embedding_endpoint_memgpt_hosted():
    filename = os.path.join(embedding_config_dir, "memgpt-hosted.json")
    run_embedding_endpoint(filename)


def test_embedding_endpoint_local():
    filename = os.path.join(embedding_config_dir, "local.json")
    run_embedding_endpoint(filename)


def test_llm_endpoint_ollama():
    filename = os.path.join(llm_config_dir, "ollama.json")
    run_llm_endpoint(filename)


def test_embedding_endpoint_ollama():
    filename = os.path.join(embedding_config_dir, "ollama.json")
    run_embedding_endpoint(filename)


def test_llm_endpoint_anthropic():
    filename = os.path.join(llm_config_dir, "anthropic.json")
    run_llm_endpoint(filename)
