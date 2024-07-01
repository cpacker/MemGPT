import json
import os
import uuid

from memgpt.agent import Agent
from memgpt.data_types import AgentState, Message
from memgpt.embeddings import embedding_model
from memgpt.llm_api.llm_api_tools import create
from memgpt.models.pydantic_models import EmbeddingConfigModel, LLMConfigModel
from memgpt.presets.presets import load_module_tools
from memgpt.prompts import gpt_system

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
    llm_config = LLMConfigModel(**config_data)
    embedding_config = EmbeddingConfigModel(**json.load(open(embedding_config_path)))
    agent_state = AgentState(
        name="test_agent",
        tools=[tool.name for tool in load_module_tools()],
        embedding_config=embedding_config,
        llm_config=llm_config,
        user_id=uuid.UUID(int=1),
        state={"persona": "", "human": "", "messages": None, "memory": {}},
        system="",
    )
    agent = Agent(
        interface=None,
        tools=load_module_tools(),
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
    assert response is not None


def run_embedding_endpoint(filename):
    # load JSON file
    config_data = json.load(open(filename, "r"))
    print(config_data)
    embedding_config = EmbeddingConfigModel(**config_data)
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
