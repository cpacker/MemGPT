import json
import os
import uuid

import pytest

from letta import create_client
from letta.agent import Agent
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.streaming_interface import StreamingRefreshCLIInterface
from tests.helpers.endpoints_helper import EMBEDDING_CONFIG_PATH
from tests.helpers.utils import cleanup

# constants
LLM_CONFIG_DIR = "tests/configs/llm_model_configs"
SUMMARY_KEY_PHRASE = "The following is a summary"


@pytest.mark.parametrize(
    "config_filename",
    [
        "openai-gpt-4o.json",
        "azure-gpt-4o-mini.json",
        "claude-3-5-haiku.json",
        # "groq.json", TODO: Support groq, rate limiting currently makes it impossible to test
        # "gemini-pro.json", TODO: Gemini is broken
    ],
)
def test_summarizer(config_filename):
    namespace = uuid.NAMESPACE_DNS
    agent_name = str(uuid.uuid5(namespace, f"integration-test-summarizer-{config_filename}"))

    # Get the LLM config
    filename = os.path.join(LLM_CONFIG_DIR, config_filename)
    config_data = json.load(open(filename, "r"))

    # Create client and clean up agents
    llm_config = LLMConfig(**config_data)
    embedding_config = EmbeddingConfig(**json.load(open(EMBEDDING_CONFIG_PATH)))
    client = create_client()
    client.set_default_llm_config(llm_config)
    client.set_default_embedding_config(embedding_config)
    cleanup(client=client, agent_uuid=agent_name)

    # Create agent
    agent_state = client.create_agent(name=agent_name, llm_config=llm_config, embedding_config=embedding_config)
    tools = [client.get_tool(client.get_tool_id(name=tool_name)) for tool_name in agent_state.tools]
    letta_agent = Agent(
        interface=StreamingRefreshCLIInterface(), agent_state=agent_state, tools=tools, first_message_verify_mono=False, user=client.user
    )

    # Make conversation
    messages = [
        "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
        "Octopuses have three hearts, and two of them stop beating when they swim.",
    ]

    for m in messages:
        letta_agent.step_user_message(
            user_message_str=m,
            first_message=False,
            skip_verify=False,
            stream=False,
            ms=client.server.ms,
        )

    # Invoke a summarize
    letta_agent.summarize_messages_inplace(preserve_last_N_messages=False)
    assert SUMMARY_KEY_PHRASE in letta_agent.messages[1]["content"], f"Test failed for config: {config_filename}"
