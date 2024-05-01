import json
import os
import uuid

from memgpt.agent import Agent
from memgpt.data_types import Message
from memgpt.embeddings import embedding_model
from memgpt.llm_api.llm_api_tools import create
from memgpt.models.pydantic_models import EmbeddingConfigModel, LLMConfigModel
from memgpt.presets.presets import load_preset
from memgpt.prompts import gpt_system

messages = [Message(role="system", text=gpt_system.get_system_text("memgpt_chat")), Message(role="user", text="How are you?")]

embedding_config_path = "configs/embedding_model_configs/memgpt-hosted.json"
llm_config_path = "configs/llm_model_configs/memgpt-hosted.json"


def test_embedding_endpoints():

    embedding_config_dir = "configs/embedding_model_configs"

    # list JSON files in directory
    for file in os.listdir(embedding_config_dir):
        if file.endswith(".json"):
            # load JSON file
            print("testing", file)
            config_data = json.load(open(os.path.join(embedding_config_dir, file)))
            embedding_config = EmbeddingConfigModel(**config_data)
            model = embedding_model(embedding_config)
            query_text = "hello"
            query_vec = model.get_text_embedding(query_text)
            print("vector dim", len(query_vec))


def test_llm_endpoints():
    llm_config_dir = "configs/llm_model_configs"

    # use openai default config
    embedding_config = EmbeddingConfigModel(**json.load(open(embedding_config_path)))

    # list JSON files in directory
    for file in os.listdir(llm_config_dir):
        if file.endswith(".json"):
            # load JSON file
            print("testing", file)
            config_data = json.load(open(os.path.join(llm_config_dir, file)))
            print(config_data)
            llm_config = LLMConfigModel(**config_data)
            agent = Agent(
                interface=None,
                preset=load_preset("memgpt_chat", user_id=uuid.UUID(int=1)),
                name="test_agent",
                created_by=uuid.UUID(int=1),
                llm_config=llm_config,
                embedding_config=embedding_config,
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
